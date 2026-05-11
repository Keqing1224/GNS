import argparse
import json
import os
import random
import shutil
from typing import Dict, Tuple

import torch
from plyfile import PlyData

from utils.rdgns_importance import compute_importance_scores, parse_importance_weights
from utils.rdgns_sh import apply_sh_degree_mask
from utils.rdgns_size import estimate_model_attribute_size


def build_parser():
    parser = argparse.ArgumentParser(description="RD-GNS stage-1 post-training SH compression.")
    parser.add_argument("-m", "--model_path", required=True, type=str, help="Path to the original GNS output directory.")
    parser.add_argument("--output_model", required=True, type=str, help="Path to the compressed model output directory.")
    parser.add_argument("--iteration", default=30000, type=int, help="Iteration to load. Use -1 to auto-detect the latest iteration.")
    parser.add_argument("--mode", default="adaptive", choices=["adaptive", "uniform", "random", "none"])
    parser.add_argument("--low_ratio", default=0.30, type=float)
    parser.add_argument("--high_ratio", default=0.20, type=float)
    parser.add_argument("--mid_degree", default=1, type=int)
    parser.add_argument("--low_degree", default=0, type=int)
    parser.add_argument("--uniform_degree", default=0, type=int)
    parser.add_argument("--importance_weights", default="0.5,0.3,0.2", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dry_run", action="store_true", help="Compute report only and skip saving model artifacts.")
    parser.add_argument("--report_name", default="rdgns_compression_report.json", type=str)
    return parser


def resolve_iteration_dir(model_path: str, iteration: int) -> Tuple[int, str]:
    point_cloud_root = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(point_cloud_root):
        raise FileNotFoundError(f"Point cloud directory not found: {point_cloud_root}")

    if iteration == -1:
        candidates = []
        for entry in os.listdir(point_cloud_root):
            if not entry.startswith("iteration_"):
                continue
            suffix = entry.split("_")[-1]
            if suffix.isdigit():
                candidates.append(int(suffix))
        if not candidates:
            raise FileNotFoundError(f"No iteration_* directories found in {point_cloud_root}")
        iteration = max(candidates)

    iteration_dir = os.path.join(point_cloud_root, f"iteration_{iteration}")
    ply_path = os.path.join(iteration_dir, "point_cloud.ply")
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"point_cloud.ply not found: {ply_path}")
    return iteration, ply_path


def infer_max_sh_degree_from_ply(ply_path: str) -> int:
    plydata = PlyData.read(ply_path)
    rest_names = [prop.name for prop in plydata.elements[0].properties if prop.name.startswith("f_rest_")]
    rest_coeff_count = len(rest_names)
    if rest_coeff_count == 0:
        return 0
    if rest_coeff_count % 3 != 0:
        raise ValueError(f"Unexpected number of SH rest attributes in {ply_path}: {rest_coeff_count}")

    coeffs_per_channel = rest_coeff_count // 3
    total_coeffs = coeffs_per_channel + 1
    degree = int(round(total_coeffs ** 0.5 - 1))
    if (degree + 1) ** 2 != total_coeffs:
        raise ValueError(f"Unable to infer SH degree from {rest_coeff_count} rest attributes in {ply_path}")
    return degree


def load_gaussian_model_class():
    try:
        from scene.gaussian_model import GaussianModel
    except Exception as exc:
        raise RuntimeError(
            "Failed to import scene.gaussian_model.GaussianModel. "
            "Please run this script in the same environment used by GNS, with its compiled extensions available."
        ) from exc
    return GaussianModel


def validate_args(args, max_sh_degree: int):
    if args.low_ratio < 0 or args.high_ratio < 0:
        raise ValueError("--low_ratio and --high_ratio must be non-negative.")
    if args.low_ratio + args.high_ratio > 1.0:
        raise ValueError("--low_ratio + --high_ratio must be <= 1.0.")

    degree_fields = {
        "mid_degree": args.mid_degree,
        "low_degree": args.low_degree,
        "uniform_degree": args.uniform_degree,
    }
    for name, value in degree_fields.items():
        if value < 0 or value > max_sh_degree:
            raise ValueError(f"--{name} must be within [0, {max_sh_degree}], but got {value}.")

    if not args.dry_run and os.path.abspath(args.model_path) == os.path.abspath(args.output_model):
        raise ValueError("--output_model must be different from --model_path to avoid overwriting the source model.")


def build_degree_assignment(args, importance_scores: torch.Tensor, max_sh_degree: int):
    num_gaussians = importance_scores.shape[0]
    low_count = int(num_gaussians * args.low_ratio)
    high_count = int(num_gaussians * args.high_ratio)
    mid_count = num_gaussians - low_count - high_count

    degree_per_gaussian = torch.full(
        (num_gaussians,),
        fill_value=max_sh_degree,
        dtype=torch.long,
        device=importance_scores.device,
    )
    group_ids = torch.full(
        (num_gaussians,),
        fill_value=2,
        dtype=torch.long,
        device=importance_scores.device,
    )

    if args.mode == "none":
        high_count, mid_count, low_count = num_gaussians, 0, 0
    elif args.mode == "uniform":
        degree_per_gaussian.fill_(args.uniform_degree)
        group_ids.fill_(1)
        high_count, mid_count, low_count = 0, num_gaussians, 0
    else:
        if args.mode == "adaptive":
            sorted_indices = torch.argsort(importance_scores, descending=True)
        elif args.mode == "random":
            rng = random.Random(args.seed)
            shuffled = list(range(num_gaussians))
            rng.shuffle(shuffled)
            sorted_indices = torch.tensor(shuffled, device=importance_scores.device, dtype=torch.long)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

        if high_count > 0:
            degree_per_gaussian[sorted_indices[:high_count]] = max_sh_degree
            group_ids[sorted_indices[:high_count]] = 2
        if mid_count > 0:
            mid_slice = sorted_indices[high_count:high_count + mid_count]
            degree_per_gaussian[mid_slice] = args.mid_degree
            group_ids[mid_slice] = 1
        if low_count > 0:
            low_slice = sorted_indices[high_count + mid_count:]
            degree_per_gaussian[low_slice] = args.low_degree
            group_ids[low_slice] = 0

    return degree_per_gaussian, group_ids, {
        "high_count": high_count,
        "mid_count": mid_count,
        "low_count": low_count,
    }


def compute_group_report(importance_scores: torch.Tensor, group_ids: torch.Tensor) -> Dict[str, float]:
    importance_scores = importance_scores.detach().cpu()
    group_ids = group_ids.detach().cpu()

    high_mask = group_ids == 2
    mid_mask = group_ids == 1
    low_mask = group_ids == 0

    def _mean_for(mask: torch.Tensor) -> float:
        if int(torch.sum(mask).item()) == 0:
            return 0.0
        return float(torch.mean(importance_scores[mask]).item())

    return {
        "top_gaussian_count": int(torch.sum(high_mask).item()),
        "mid_gaussian_count": int(torch.sum(mid_mask).item()),
        "low_gaussian_count": int(torch.sum(low_mask).item()),
        "top_gaussian_average_importance": _mean_for(high_mask),
        "mid_gaussian_average_importance": _mean_for(mid_mask),
        "low_gaussian_average_importance": _mean_for(low_mask),
    }


def save_report(report: Dict, report_path: str):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)


def copy_cfg_args(model_path: str, output_model: str):
    source_cfg = os.path.join(model_path, "cfg_args")
    if os.path.isfile(source_cfg):
        os.makedirs(output_model, exist_ok=True)
        shutil.copy2(source_cfg, os.path.join(output_model, "cfg_args"))


def main():
    parser = build_parser()
    args = parser.parse_args()

    iteration, ply_path = resolve_iteration_dir(args.model_path, args.iteration)
    max_sh_degree = infer_max_sh_degree_from_ply(ply_path)
    validate_args(args, max_sh_degree)

    GaussianModel = load_gaussian_model_class()
    gaussians = GaussianModel(max_sh_degree)
    gaussians.load_ply(ply_path)

    weights = parse_importance_weights(args.importance_weights)
    importance_scores, _, importance_report = compute_importance_scores(gaussians, weights)
    degree_per_gaussian, group_ids, requested_group_counts = build_degree_assignment(args, importance_scores, max_sh_degree)
    gaussians, sh_report = apply_sh_degree_mask(gaussians, degree_per_gaussian, max_sh_degree)

    size_report = estimate_model_attribute_size(gaussians.get_xyz.shape[0], max_sh_degree, degree_per_gaussian)
    group_report = compute_group_report(importance_scores, group_ids)

    report = {
        "model_path": args.model_path,
        "output_model": args.output_model,
        "input_ply_path": ply_path,
        "iteration": int(iteration),
        "mode": args.mode,
        "low_ratio": float(args.low_ratio),
        "high_ratio": float(args.high_ratio),
        "mid_degree": int(args.mid_degree),
        "low_degree": int(args.low_degree),
        "uniform_degree": int(args.uniform_degree),
        "seed": int(args.seed),
        "dry_run": bool(args.dry_run),
        "requested_group_counts": requested_group_counts,
        "importance_report": importance_report,
        "group_report": group_report,
        "sh_report": sh_report,
        "size_report": size_report,
    }

    if not args.dry_run:
        copy_cfg_args(args.model_path, args.output_model)
        output_ply = os.path.join(
            args.output_model,
            "point_cloud",
            f"iteration_{iteration}",
            "point_cloud.ply",
        )
        gaussians.save_ply(output_ply)
        save_report(report, os.path.join(args.output_model, args.report_name))
        print(f"Compressed model saved to: {output_ply}")
        print(f"Compression report saved to: {os.path.join(args.output_model, args.report_name)}")
    else:
        print("Dry run enabled. Model files were not written.")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
