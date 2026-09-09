"""Reusable calculations for the P4 first-principles audit."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def masked_global_l1_oracle(
    required_residual: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    """Best per-sample/step broadcast correction under masked L1 error."""

    residual = np.asarray(required_residual, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if residual.ndim != 4 or mask.shape != residual.shape[:-1]:
        raise ValueError("residual must be [batch, step, entity, feature]")
    correction = np.zeros((*residual.shape[:2], 1, residual.shape[-1]), dtype=np.float64)
    for batch in range(residual.shape[0]):
        for step in range(residual.shape[1]):
            selected = mask[batch, step]
            if np.any(selected):
                correction[batch, step, 0] = np.median(
                    residual[batch, step, selected], axis=0
                )
    error = np.abs(residual - correction)
    feature_mask = np.broadcast_to(mask[..., None], residual.shape)
    valid_count = int(feature_mask.sum())
    return {
        "correction": correction,
        "valid_feature_count": valid_count,
        "global_oracle_mae": float(error[feature_mask].mean()) if valid_count else 0.0,
        "entity_oracle_mae": 0.0,
    }


def shared_offset_preserves_ranking(
    base_logits: np.ndarray,
    correction: np.ndarray,
    valid: np.ndarray,
) -> bool:
    """Return whether correction preserves all pairwise valid-edge rankings."""

    base = np.asarray(base_logits, dtype=np.float64)
    delta = np.asarray(correction, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if base.shape != mask.shape or delta.shape[:-1] != base.shape[:-1]:
        raise ValueError("logit, correction, and mask shapes are incompatible")
    try:
        corrected = base + delta
    except ValueError as error:
        raise ValueError("correction cannot broadcast to logits") from error
    for index in np.ndindex(base.shape[:-1]):
        selected = mask[index]
        before = base[index][selected]
        after = corrected[index][selected]
        if before.size < 2:
            continue
        before_order = np.sign(before[:, None] - before[None, :])
        after_order = np.sign(after[:, None] - after[None, :])
        if not np.array_equal(before_order, after_order):
            return False
    return True


def gradient_cosine_summary(
    gradients: Mapping[str, Sequence[torch.Tensor | None]],
) -> dict[str, Any]:
    """Summarize flattened gradient norms and pairwise cosine conflicts."""

    flattened: dict[str, torch.Tensor] = {}
    for name, values in gradients.items():
        tensors = [value.detach().reshape(-1).double() for value in values if value is not None]
        flattened[name] = torch.cat(tensors) if tensors else torch.zeros(1, dtype=torch.float64)
    norms = {name: float(value.norm()) for name, value in flattened.items()}
    pairs: dict[str, dict[str, Any]] = {}
    for left, right in combinations(sorted(flattened), 2):
        left_value, right_value = flattened[left], flattened[right]
        if left_value.numel() != right_value.numel():
            raise ValueError("gradient vectors must share the same parameter layout")
        denominator = left_value.norm() * right_value.norm()
        cosine = 0.0 if float(denominator) == 0.0 else float(
            torch.dot(left_value, right_value) / denominator
        )
        pairs[f"{left}__{right}"] = {
            "cosine": cosine,
            "conflict": cosine < 0.0,
        }
    return {"norms": norms, "pairs": pairs}


def gate_aware_checkpoint_key(metrics: Mapping[str, float]) -> tuple[int, float, float]:
    """Feasible-first lexicographic checkpoint key for the frozen P4 gates."""

    violations = (
        max(0.0, (-0.05 - float(metrics["validation_link_f1_delta"])) / 0.05),
        max(0.0, (float(metrics["node_x_mae_ratio"]) - 1.25) / 0.25),
        max(0.0, float(metrics["throughput_mae_delta"])),
        max(0.0, float(metrics["rb_occupancy_mae_delta"])),
        max(0.0, float(metrics["task_delay_mae_delta"])),
    )
    return (
        sum(value > 0.0 for value in violations),
        max(violations),
        float(metrics["validation_state_nll"]),
    )


__all__ = [
    "gate_aware_checkpoint_key",
    "gradient_cosine_summary",
    "masked_global_l1_oracle",
    "shared_offset_preserves_ranking",
]
