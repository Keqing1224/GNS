import os

import torch

from utils.rdgns_sh import build_rest_keep_mask


MISSING_MASK_ERROR = (
    "Missing rdgns_degree_per_gaussian.pt. "
    "Please rerun compress_gns_attrs.py with the updated Stage-2 version."
)


def load_degree_per_gaussian(model_path, device):
    degree_path = os.path.join(model_path, "rdgns_degree_per_gaussian.pt")
    if not os.path.isfile(degree_path):
        raise FileNotFoundError(MISSING_MASK_ERROR)

    degree_per_gaussian = torch.load(degree_path, map_location=device)
    if not isinstance(degree_per_gaussian, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {degree_path}, but got {type(degree_per_gaussian)}.")
    if degree_per_gaussian.ndim != 1:
        raise ValueError(
            f"Expected rdgns_degree_per_gaussian.pt to have shape [N], but got {tuple(degree_per_gaussian.shape)}."
        )

    return degree_per_gaussian.to(device=device, dtype=torch.long)


def build_sh_keep_mask(gaussians, degree_per_gaussian, max_sh_degree):
    features_rest = gaussians._features_rest
    keep_mask, layout = build_rest_keep_mask(features_rest, degree_per_gaussian, max_sh_degree, expand=True)
    return keep_mask.to(dtype=torch.bool), layout


def apply_sh_mask_(gaussians, sh_mask):
    features_rest = gaussians._features_rest
    if features_rest.ndim != 3:
        raise ValueError(
            f"Expected gaussians._features_rest to be 3D, but got shape {tuple(features_rest.shape)}."
        )

    try:
        broadcast_mask = torch.broadcast_to(sh_mask, features_rest.shape)
    except RuntimeError as exc:
        raise ValueError(
            "SH mask shape is not broadcast-compatible with gaussians._features_rest: "
            f"mask={tuple(sh_mask.shape)}, features_rest={tuple(features_rest.shape)}."
        ) from exc

    with torch.no_grad():
        features_rest.mul_(broadcast_mask.to(device=features_rest.device, dtype=features_rest.dtype))

    total_coefficients = int(features_rest.numel())
    kept_coefficients = int(broadcast_mask.sum().item())
    return {
        "total_feature_rest_values": total_coefficients,
        "kept_feature_rest_values": kept_coefficients,
        "masked_feature_rest_values": total_coefficients - kept_coefficients,
    }


def degree_counts(degree_per_gaussian):
    degree_per_gaussian = degree_per_gaussian.detach().to(dtype=torch.long)
    return {
        "degree_0_count": int(torch.sum(degree_per_gaussian == 0).item()),
        "degree_1_count": int(torch.sum(degree_per_gaussian == 1).item()),
        "degree_2_count": int(torch.sum(degree_per_gaussian == 2).item()),
        "degree_3_count": int(torch.sum(degree_per_gaussian == 3).item()),
    }
