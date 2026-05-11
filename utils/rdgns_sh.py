from typing import Dict

import torch


def total_sh_coeffs_for_degree(degree: int) -> int:
    if degree < 0:
        raise ValueError("SH degree must be non-negative.")
    return (degree + 1) ** 2


def rest_sh_coeffs_for_degree(degree: int) -> int:
    return total_sh_coeffs_for_degree(degree) - 1


def infer_sh_rest_layout(features_rest: torch.Tensor, max_sh_degree: int) -> Dict[str, int]:
    if features_rest.ndim != 3:
        raise ValueError(
            f"Expected gaussians._features_rest to be a 3D tensor, but got shape {tuple(features_rest.shape)}."
        )

    expected_rest_coeffs = rest_sh_coeffs_for_degree(max_sh_degree)
    shape = tuple(features_rest.shape)

    if shape[1] == expected_rest_coeffs:
        return {
            "coeff_axis": 1,
            "channel_axis": 2,
            "num_coeffs": shape[1],
            "num_channels": shape[2],
        }

    if shape[2] == expected_rest_coeffs:
        return {
            "coeff_axis": 2,
            "channel_axis": 1,
            "num_coeffs": shape[2],
            "num_channels": shape[1],
        }

    raise ValueError(
        "Unable to infer SH layout from gaussians._features_rest with shape "
        f"{shape}. Expected one non-batch axis to equal {expected_rest_coeffs} "
        f"for max_sh_degree={max_sh_degree}."
    )


def build_rest_keep_mask(features_rest: torch.Tensor, degree_per_gaussian: torch.Tensor, max_sh_degree: int, expand: bool = False):
    layout = infer_sh_rest_layout(features_rest, max_sh_degree)
    expected_rest_coeffs = rest_sh_coeffs_for_degree(max_sh_degree)

    if degree_per_gaussian.ndim != 1:
        raise ValueError("degree_per_gaussian must be a 1D tensor.")
    if degree_per_gaussian.shape[0] != features_rest.shape[0]:
        raise ValueError(
            "degree_per_gaussian length does not match number of Gaussians: "
            f"{degree_per_gaussian.shape[0]} vs {features_rest.shape[0]}."
        )

    degree_per_gaussian = degree_per_gaussian.to(device=features_rest.device, dtype=torch.long)
    if torch.any(degree_per_gaussian < 0) or torch.any(degree_per_gaussian > max_sh_degree):
        raise ValueError(f"All requested SH degrees must be within [0, {max_sh_degree}].")

    keep_rest_counts = (degree_per_gaussian + 1) ** 2 - 1
    coeff_ids = torch.arange(expected_rest_coeffs, device=features_rest.device, dtype=torch.long)

    if layout["coeff_axis"] == 1:
        keep_mask = coeff_ids.unsqueeze(0) < keep_rest_counts.unsqueeze(1)
        keep_mask = keep_mask.unsqueeze(-1)
    else:
        keep_mask = coeff_ids.unsqueeze(0) < keep_rest_counts.unsqueeze(1)
        keep_mask = keep_mask.unsqueeze(1)

    if expand:
        keep_mask = keep_mask.expand_as(features_rest)

    return keep_mask, layout


def apply_sh_degree_mask(gaussians, degree_per_gaussian: torch.Tensor, max_sh_degree: int):
    features_rest = gaussians._features_rest
    keep_mask, layout = build_rest_keep_mask(features_rest, degree_per_gaussian, max_sh_degree, expand=False)

    with torch.no_grad():
        features_rest.mul_(keep_mask.to(dtype=features_rest.dtype))

    original_total_coeffs = degree_per_gaussian.shape[0] * total_sh_coeffs_for_degree(max_sh_degree)
    kept_total_coeffs = int(torch.sum((degree_per_gaussian + 1) ** 2).item())

    report = {
        "degree_0_count": int(torch.sum(degree_per_gaussian == 0).item()),
        "degree_1_count": int(torch.sum(degree_per_gaussian == 1).item()) if max_sh_degree >= 1 else 0,
        "degree_2_count": int(torch.sum(degree_per_gaussian == 2).item()) if max_sh_degree >= 2 else 0,
        "degree_3_count": int(torch.sum(degree_per_gaussian == 3).item()) if max_sh_degree >= 3 else 0,
        "original_sh_coeff_count": int(original_total_coeffs),
        "kept_sh_coeff_count": int(kept_total_coeffs),
        "kept_sh_coeff_ratio": float(kept_total_coeffs / original_total_coeffs) if original_total_coeffs else 1.0,
        "rest_layout": layout,
    }

    return gaussians, report
