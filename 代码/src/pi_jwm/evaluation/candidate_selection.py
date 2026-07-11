"""Pure candidate-selection metrics shared by PI-JWM experiment entry points."""

from __future__ import annotations

import numpy as np


def sample_active_sse(predictions: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(predictions["link_rate_true"], dtype=np.float64).squeeze(-1)
    pred = np.asarray(predictions["link_rate_pred"], dtype=np.float64).squeeze(-1)
    active = np.asarray(predictions["link_activity_true"], dtype=np.float64).squeeze(-1) > 0.5
    if truth.shape != pred.shape or truth.shape != active.shape:
        raise ValueError("prediction tensors must share [sample, step, edge] shape")
    error = ((pred - truth) ** 2) * active
    sse = np.sum(error, axis=(1, 2)).astype(np.float64)
    count = np.sum(active, axis=(1, 2)).astype(np.int64)
    return sse, count


def sample_rmse_from_sse(sse: np.ndarray, count: np.ndarray) -> np.ndarray:
    sse = np.asarray(sse, dtype=np.float64).reshape(-1)
    count = np.asarray(count, dtype=np.float64).reshape(-1)
    if sse.shape != count.shape:
        raise ValueError("sse and count must have the same shape")
    rmse = np.zeros_like(sse, dtype=np.float64)
    active = count > 0
    rmse[active] = np.sqrt(sse[active] / count[active])
    return rmse.astype(np.float32)


def mix_actions_by_sample(candidate_actions: list[np.ndarray], choice: np.ndarray) -> np.ndarray:
    if not candidate_actions:
        raise ValueError("candidate_actions must not be empty")
    choice = np.asarray(choice, dtype=np.int64).reshape(-1)
    mixed = np.asarray(candidate_actions[0], dtype=np.float32).copy()
    sample_count = mixed.shape[0]
    if choice.shape[0] != sample_count:
        raise ValueError("choice length must match sample count")
    for actions in candidate_actions:
        if np.asarray(actions).shape != mixed.shape:
            raise ValueError("all candidate actions must share shape")
    for sample_idx, candidate_idx in enumerate(choice):
        if int(candidate_idx) < 0 or int(candidate_idx) >= len(candidate_actions):
            raise ValueError("choice contains an out-of-range candidate index")
        mixed[sample_idx] = candidate_actions[int(candidate_idx)][sample_idx]
    return mixed.astype(np.float32)


def choose_best_single_by_sample_sse(
    sample_sse: np.ndarray,
    active_count: np.ndarray,
) -> tuple[int, np.ndarray]:
    sample_sse = np.asarray(sample_sse, dtype=np.float64)
    active_count = np.asarray(active_count, dtype=np.float64).reshape(-1)
    if sample_sse.ndim != 2 or sample_sse.shape[0] != active_count.shape[0]:
        raise ValueError("sample_sse must have shape [sample, candidate] and match active_count")
    active_samples = active_count > 0
    totals = np.sum(sample_sse[active_samples], axis=0)
    count_total = float(np.sum(active_count[active_samples]))
    rmse = np.sqrt(totals / max(count_total, 1.0)).astype(np.float32)
    return int(np.argmin(rmse)), rmse


def choice_rmse_from_sample_sse(
    sample_sse: np.ndarray,
    active_count: np.ndarray,
    choice: np.ndarray,
) -> float:
    sample_sse = np.asarray(sample_sse, dtype=np.float64)
    active_count = np.asarray(active_count, dtype=np.float64).reshape(-1)
    choice = np.asarray(choice, dtype=np.int64).reshape(-1)
    if (
        sample_sse.ndim != 2
        or sample_sse.shape[0] != active_count.shape[0]
        or choice.shape[0] != sample_sse.shape[0]
    ):
        raise ValueError("sample_sse, active_count, and choice must share sample count")
    if np.any(choice < 0) or np.any(choice >= sample_sse.shape[1]):
        raise ValueError("choice contains an out-of-range candidate index")
    active_samples = active_count > 0
    selected_sse = sample_sse[np.arange(sample_sse.shape[0]), choice]
    return float(
        np.sqrt(
            np.sum(selected_sse[active_samples])
            / max(float(np.sum(active_count[active_samples])), 1.0)
        )
    )
