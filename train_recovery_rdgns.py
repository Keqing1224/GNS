import json
import os
import random
import shutil
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.rdgns_mask import apply_sh_mask_, build_sh_keep_mask, degree_counts, load_degree_per_gaussian

try:
    from fused_ssim import fused_ssim as fast_ssim
    SSIM_NAME = "fused_ssim"
except ImportError:
    fast_ssim = None
    SSIM_NAME = "ssim"


def prepare_recovery_output(source_model_path, output_model):
    os.makedirs(output_model, exist_ok=True)

    cfg_args_path = os.path.join(source_model_path, "cfg_args")
    if os.path.isfile(cfg_args_path):
        shutil.copy2(cfg_args_path, os.path.join(output_model, "cfg_args"))

    report_path = os.path.join(source_model_path, "rdgns_compression_report.json")
    if os.path.isfile(report_path):
        shutil.copy2(report_path, os.path.join(output_model, "rdgns_compression_report.json"))

    degree_path = os.path.join(source_model_path, "rdgns_degree_per_gaussian.pt")
    if os.path.isfile(degree_path):
        shutil.copy2(degree_path, os.path.join(output_model, "rdgns_degree_per_gaussian.pt"))


def zero_xyz_learning_rate(gaussians):
    gaussians._xyz.requires_grad_(False)

    optimizers = [gaussians.optimizer]
    if gaussians.shoptimizer is not None:
        optimizers.append(gaussians.shoptimizer)

    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group_name = str(group.get("name", "")).lower()
            if "xyz" in group_name or "position" in group_name:
                group["lr"] = 0.0
                continue
            for param in group["params"]:
                if param is gaussians._xyz:
                    group["lr"] = 0.0
                    break


def save_recovery_point_cloud(gaussians, output_model, iteration):
    output_ply = os.path.join(output_model, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    gaussians.save_ply(output_ply)
    return output_ply


def ensure_recovery_args(args):
    if not hasattr(args, "iterations") or args.iterations is None:
        args.iterations = 3000
    if not hasattr(args, "save_iterations") or args.save_iterations is None:
        args.save_iterations = [args.iterations]
    if not hasattr(args, "test_iterations") or args.test_iterations is None:
        args.test_iterations = [args.iterations]
    if not hasattr(args, "lambda_dssim") or args.lambda_dssim is None:
        args.lambda_dssim = 0.2
    if not hasattr(args, "freeze_xyz") or args.freeze_xyz is None:
        args.freeze_xyz = True
    if not hasattr(args, "enforce_sh_mask") or args.enforce_sh_mask is None:
        args.enforce_sh_mask = True
    if not hasattr(args, "load_iteration") or args.load_iteration is None:
        args.load_iteration = -1
    if not hasattr(args, "dry_run") or args.dry_run is None:
        args.dry_run = False
    if not hasattr(args, "debug_from") or args.debug_from is None:
        args.debug_from = -1
    return args


def main():
    parser = ArgumentParser(description="RD-GNS stage-2 mask-constrained recovery fine-tuning.")
    model = ModelParams(parser, sentinel=True)
    opt_group = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--output_model", required=True, type=str)
    parser.add_argument("--load_iteration", default=-1, type=int)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=None)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--freeze_xyz", dest="freeze_xyz", action="store_true")
    parser.add_argument("--no_freeze_xyz", dest="freeze_xyz", action="store_false")
    parser.add_argument("--enforce_sh_mask", dest="enforce_sh_mask", action="store_true")
    parser.add_argument("--no_enforce_sh_mask", dest="enforce_sh_mask", action="store_false")
    parser.set_defaults(freeze_xyz=True, enforce_sh_mask=True)
    args = get_combined_args(parser)
    args = ensure_recovery_args(args)

    source_model_path = args.model_path
    output_model = os.path.abspath(args.output_model)
    args.output_model = output_model

    if os.path.abspath(source_model_path) == output_model:
        raise ValueError("--output_model must be different from --model_path for recovery fine-tuning.")

    args.save_iterations = sorted(set(args.save_iterations or [args.iterations]))
    args.test_iterations = sorted(set(args.test_iterations or [args.iterations]))

    if args.dry_run:
        degree_path = os.path.join(source_model_path, "rdgns_degree_per_gaussian.pt")
        if not os.path.isfile(degree_path):
            raise FileNotFoundError(
                "Missing rdgns_degree_per_gaussian.pt. "
                "Please rerun compress_gns_attrs.py with the updated Stage-2 version."
            )
        resolved_args = dict(vars(args))
        resolved_args["output_model"] = output_model
        resolved_args["degree_mask_exists"] = True
        print(json.dumps(resolved_args, indent=2, sort_keys=True, default=str))
        return

    safe_state(args.quiet)
    prepare_recovery_output(source_model_path, output_model)

    dataset = model.extract(args)
    opt = opt_group.extract(args)
    pipe = pipeline.extract(args)
    opt.iterations = args.iterations
    opt.lambda_dssim = args.lambda_dssim
    opt.prune_reg = False
    opt.densify_from_iter = args.iterations + 1
    opt.densify_until_iter = args.iterations + 1
    opt.opacity_reset_interval = args.iterations + 1

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=True)
    gaussians.training_setup(opt)

    if args.freeze_xyz:
        zero_xyz_learning_rate(gaussians)

    degree_per_gaussian = load_degree_per_gaussian(source_model_path, gaussians._features_rest.device)
    sh_mask, sh_layout = build_sh_keep_mask(gaussians, degree_per_gaussian, gaussians.max_sh_degree)
    initial_mask_stats = apply_sh_mask_(gaussians, sh_mask) if args.enforce_sh_mask else {}

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    final_loss = None
    final_l1 = None
    mask_stats = initial_mask_stats
    progress_bar = tqdm(range(1, args.iterations + 1), desc="RD-GNS recovery")

    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        rand_idx = random.randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)
        gt_image = viewpoint_cam.original_image.cuda()

        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]

        Ll1 = l1_loss(image, gt_image)
        if fast_ssim is not None:
            ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim_value)

        loss.backward()

        if args.enforce_sh_mask and gaussians._features_rest.grad is not None:
            gaussians._features_rest.grad.mul_(sh_mask.to(dtype=gaussians._features_rest.grad.dtype))

        if gaussians.optimizer is not None:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
        if gaussians.shoptimizer is not None:
            gaussians.shoptimizer.step()
            gaussians.shoptimizer.zero_grad(set_to_none=True)

        if args.enforce_sh_mask:
            mask_stats = apply_sh_mask_(gaussians, sh_mask)

        final_loss = float(loss.item())
        final_l1 = float(Ll1.item())
        progress_bar.set_postfix({"loss": f"{final_loss:.6f}", "l1": f"{final_l1:.6f}", "N_GS": gaussians.get_xyz.shape[0]})

        if iteration in args.save_iterations:
            if args.enforce_sh_mask:
                apply_sh_mask_(gaussians, sh_mask)
            save_recovery_point_cloud(gaussians, output_model, iteration)

    degree_report = degree_counts(degree_per_gaussian)
    recovery_log = {
        "source_model": source_model_path,
        "output_model": output_model,
        "iterations": int(args.iterations),
        "freeze_xyz": bool(args.freeze_xyz),
        "enforce_sh_mask": bool(args.enforce_sh_mask),
        "final_loss": final_loss,
        "final_l1": final_l1,
        "num_gaussians": int(gaussians.get_xyz.shape[0]),
        "degree_counts": degree_report,
        "xyz_frozen": bool(not gaussians._xyz.requires_grad),
        "sh_mask_applied": bool(args.enforce_sh_mask),
        "sh_mask_layout": sh_layout,
        "sh_mask_stats": mask_stats,
        "loaded_iteration": int(scene.loaded_iter),
        "ssim_backend": SSIM_NAME,
    }

    with open(os.path.join(output_model, "recovery_log.json"), "w", encoding="utf-8") as recovery_file:
        json.dump(recovery_log, recovery_file, indent=2)

    print(json.dumps(recovery_log, indent=2))


if __name__ == "__main__":
    main()
