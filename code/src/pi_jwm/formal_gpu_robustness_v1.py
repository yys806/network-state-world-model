"""Controlled input-perturbation utilities for formal GPU robustness checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


CONTINUOUS_HISTORY_KEYS = (
    ("node_state", "node_present"),
    ("physical_edge_state", "physical_edge_present"),
    ("flow_state", "flow_present"),
    ("task_state", "task_present"),
)


def execution_policy(device: torch.device | str) -> dict[str, Any]:
    """Describe where this robustness evaluation actually executes."""

    device_obj = torch.device(device)
    return {
        "gpu_execution": device_obj.type == "cuda",
        "evaluation_device": str(device_obj),
        "locked_test_accessed": False,
        "result_boundary": "Nonlocked aggregate-baseline input-perturbation diagnosis only.",
    }


def perturb_history(
    history: Mapping[str, Any],
    *,
    noise_scale: float,
    seed: int,
    perturb_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Add normalized Gaussian noise only to valid continuous observations.

    Masks, topology/action identities, action history, and all non-tensor values
    are copied unchanged. ``noise_scale`` is measured in normalized feature
    standard deviations and is deterministic for a given seed/device. When
    ``perturb_keys`` is provided, only those continuous state keys are changed.
    """

    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")
    result: dict[str, Any] = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in history.items()
    }
    available_keys = {state_key for state_key, _ in CONTINUOUS_HISTORY_KEYS}
    selected_keys = tuple(
        state_key for state_key, _ in CONTINUOUS_HISTORY_KEYS
    ) if perturb_keys is None else tuple(perturb_keys)
    unknown_keys = set(selected_keys) - available_keys
    if unknown_keys:
        raise ValueError(f"unknown perturbation keys: {sorted(unknown_keys)}")
    for state_key, mask_key in CONTINUOUS_HISTORY_KEYS:
        if state_key not in selected_keys:
            continue
        value = history.get(state_key)
        mask = history.get(mask_key)
        if not isinstance(value, torch.Tensor) or not isinstance(mask, torch.Tensor):
            continue
        if value.ndim != mask.ndim + 1:
            raise ValueError(f"state/mask rank mismatch for {state_key}")
        if value.shape[:-1] != mask.shape:
            raise ValueError(f"state/mask shape mismatch for {state_key}")
        if noise_scale == 0:
            continue
        generator = torch.Generator(device=value.device)
        generator.manual_seed(int(seed))
        noise = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
        result[state_key] = value + float(noise_scale) * noise * mask.unsqueeze(-1).to(value.dtype)
    return result


__all__ = ["CONTINUOUS_HISTORY_KEYS", "execution_policy", "perturb_history"]
