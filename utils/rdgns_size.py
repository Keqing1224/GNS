from typing import Dict

import torch

from utils.rdgns_sh import total_sh_coeffs_for_degree


FLOAT32_BYTES = 4
RGB_CHANNELS = 3
FIXED_ATTRIBUTE_FLOATS = {
    "xyz": 3,
    "opacity": 1,
    "scale": 3,
    "rotation": 4,
}


def estimate_model_attribute_size(num_gaussians: int, max_sh_degree: int, degree_per_gaussian: torch.Tensor) -> Dict[str, float]:
    degree_per_gaussian = degree_per_gaussian.detach().to(dtype=torch.long).cpu()

    fixed_float_count = num_gaussians * sum(FIXED_ATTRIBUTE_FLOATS.values())
    original_sh_coeff_count = num_gaussians * total_sh_coeffs_for_degree(max_sh_degree)
    kept_sh_coeff_count = int(torch.sum((degree_per_gaussian + 1) ** 2).item())

    fixed_bytes = fixed_float_count * FLOAT32_BYTES
    original_sh_bytes = original_sh_coeff_count * RGB_CHANNELS * FLOAT32_BYTES
    compressed_sh_bytes = kept_sh_coeff_count * RGB_CHANNELS * FLOAT32_BYTES

    original_total = fixed_bytes + original_sh_bytes
    compressed_total = fixed_bytes + compressed_sh_bytes

    return {
        "num_gaussians": int(num_gaussians),
        "max_sh_degree": int(max_sh_degree),
        "original_estimated_bytes": int(original_total),
        "compressed_estimated_bytes": int(compressed_total),
        "compression_ratio": float(compressed_total / original_total) if original_total else 1.0,
        "fixed_attribute_bytes": int(fixed_bytes),
        "original_sh_bytes": int(original_sh_bytes),
        "compressed_sh_bytes": int(compressed_sh_bytes),
        "original_sh_coeff_count": int(original_sh_coeff_count),
        "kept_sh_coeff_count": int(kept_sh_coeff_count),
        "kept_sh_coeff_ratio": float(kept_sh_coeff_count / original_sh_coeff_count) if original_sh_coeff_count else 1.0,
        "degree_0_count": int(torch.sum(degree_per_gaussian == 0).item()),
        "degree_1_count": int(torch.sum(degree_per_gaussian == 1).item()) if max_sh_degree >= 1 else 0,
        "degree_2_count": int(torch.sum(degree_per_gaussian == 2).item()) if max_sh_degree >= 2 else 0,
        "degree_3_count": int(torch.sum(degree_per_gaussian == 3).item()) if max_sh_degree >= 3 else 0,
        "note": "Stage-1 estimated attribute compression. PLY may still store full tensors unless packed export is implemented.",
    }
