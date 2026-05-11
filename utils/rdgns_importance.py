from typing import Dict, Iterable, Tuple

import torch


def parse_importance_weights(raw_weights: str) -> Tuple[float, float, float]:
    parts = [part.strip() for part in raw_weights.split(",")]
    if len(parts) != 3:
        raise ValueError("--importance_weights must contain exactly 3 comma-separated values.")

    weights = [float(part) for part in parts]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("--importance_weights must sum to a positive value.")

    return tuple(weight / weight_sum for weight in weights)


def sanitize_tensor(values: torch.Tensor) -> torch.Tensor:
    sanitized = values.detach().to(dtype=torch.float32)
    return torch.nan_to_num(sanitized, nan=0.0, posinf=0.0, neginf=0.0)


def stable_normalize_01(values: torch.Tensor, eps: float = 1e-8, degenerate_fill: float = 0.0) -> torch.Tensor:
    sanitized = sanitize_tensor(values).flatten()
    if sanitized.numel() == 0:
        return sanitized

    min_value = torch.min(sanitized)
    max_value = torch.max(sanitized)
    value_range = max_value - min_value

    if not torch.isfinite(value_range) or value_range.item() < eps:
        return torch.full_like(sanitized, float(degenerate_fill)).clamp_(0.0, 1.0)

    normalized = (sanitized - min_value) / value_range
    return normalized.clamp_(0.0, 1.0)


def tensor_stats(values: torch.Tensor) -> Dict[str, float]:
    sanitized = sanitize_tensor(values).flatten()
    if sanitized.numel() == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

    return {
        "min": float(torch.min(sanitized).item()),
        "max": float(torch.max(sanitized).item()),
        "mean": float(torch.mean(sanitized).item()),
        "std": float(torch.std(sanitized, unbiased=False).item()) if sanitized.numel() > 1 else 0.0,
    }


def _flatten_feature_energy(features: torch.Tensor) -> torch.Tensor:
    sanitized = sanitize_tensor(features)
    if sanitized.numel() == 0:
        return torch.zeros((sanitized.shape[0],), dtype=torch.float32, device=sanitized.device)
    return torch.linalg.norm(sanitized.reshape(sanitized.shape[0], -1), dim=1)


def _get_scale_proxy(gaussians) -> torch.Tensor:
    scaling = sanitize_tensor(gaussians.get_scaling)
    if scaling.ndim == 1:
        return scaling
    return torch.mean(scaling, dim=1)


def _get_sh_energy_proxy(gaussians) -> torch.Tensor:
    features_rest = getattr(gaussians, "get_features_rest", None)
    if features_rest is None:
        return torch.zeros((gaussians.get_xyz.shape[0],), dtype=torch.float32, device=gaussians.get_xyz.device)

    rest_tensor = features_rest.detach()
    if rest_tensor.numel() == 0:
        return torch.zeros((rest_tensor.shape[0],), dtype=torch.float32, device=rest_tensor.device)

    return _flatten_feature_energy(rest_tensor)


def compute_importance_scores(gaussians, weights: Iterable[float]):
    opacity_weight, scale_weight, sh_weight = weights

    opacity_raw = sanitize_tensor(gaussians.get_opacity).reshape(-1)
    scale_raw = _get_scale_proxy(gaussians).reshape(-1)
    sh_energy_raw = _get_sh_energy_proxy(gaussians).reshape(-1)

    opacity_score = stable_normalize_01(opacity_raw)
    scale_score = stable_normalize_01(scale_raw)
    sh_energy_score = stable_normalize_01(sh_energy_raw)

    final_score = (
        opacity_weight * opacity_score
        + scale_weight * scale_score
        + sh_weight * sh_energy_score
    ).clamp_(0.0, 1.0)

    report = {
        "weights": {
            "opacity": float(opacity_weight),
            "scale": float(scale_weight),
            "sh_energy": float(sh_weight),
        },
        "opacity_score_stats": tensor_stats(opacity_score),
        "scale_score_stats": tensor_stats(scale_score),
        "sh_energy_score_stats": tensor_stats(sh_energy_score),
        "importance_score_stats": tensor_stats(final_score),
        "opacity_raw_stats": tensor_stats(opacity_raw),
        "scale_raw_stats": tensor_stats(scale_raw),
        "sh_energy_raw_stats": tensor_stats(sh_energy_raw),
    }

    components = {
        "opacity": opacity_score,
        "scale": scale_score,
        "sh_energy": sh_energy_score,
    }

    return final_score, components, report
