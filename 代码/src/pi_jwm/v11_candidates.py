"""Support-constrained candidate construction for the PI-JWM v11 selector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .v11_selector import project_candidate_actions


OFFLOAD_COUNT_DIM = 0
RB_COUNT_DIM = 1
RB_TOTAL_DIM = 2
CPU_COUNT_DIM = 3
CPU_TOTAL_DIM = 4
RETURN_COUNT_DIM = 5


@dataclass(frozen=True)
class CandidateLibrary:
    actions: np.ndarray
    candidate_names: tuple[str, ...]
    action_families: tuple[str, ...]
    candidate_mask: np.ndarray
    applicability_mask: np.ndarray
    action_applied: np.ndarray

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        mask = np.asarray(self.candidate_mask, dtype=bool)
        applicability = np.asarray(self.applicability_mask, dtype=bool)
        applied = np.asarray(self.action_applied, dtype=bool)
        if actions.ndim != 5:
            raise ValueError("candidate library actions must be [sample,candidate,step,edge,dim]")
        if len(self.candidate_names) != actions.shape[1] or len(self.action_families) != actions.shape[1]:
            raise ValueError("candidate metadata must match candidate dimension")
        if (
            mask.shape != actions.shape[:2]
            or applicability.shape != actions.shape[:2]
            or applied.shape != actions.shape[:2]
        ):
            raise ValueError("candidate mask/applicability/applied arrays must be [sample,candidate]")
        if len(set(self.candidate_names)) != len(self.candidate_names):
            raise ValueError("candidate names must be unique")
        if len(self.candidate_names) > 32:
            raise ValueError("selector candidate library may contain at most 32 candidates")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "candidate_mask", mask)
        object.__setattr__(self, "applicability_mask", applicability)
        object.__setattr__(self, "action_applied", applied)


def positive_value_quantiles(
    values: np.ndarray,
    quantiles: Sequence[float] = (0.5, 0.75),
    min_value: float = 1.0,
) -> dict[float, float]:
    """Compute deterministic train-only positive-value magnitude anchors."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    selected = array[np.isfinite(array) & (array >= float(min_value))]
    if selected.size == 0:
        raise ValueError("positive value quantiles require at least one supported value")
    result = {}
    for quantile in quantiles:
        value = float(quantile)
        if not 0.0 < value < 1.0:
            raise ValueError("quantiles must lie in (0, 1)")
        result[value] = float(np.quantile(selected, value, method="linear"))
    return result


def infer_stage_from_observable_actions(actions: np.ndarray) -> np.ndarray:
    """Infer a coarse task stage from deployable baseline action counts."""
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 4 or values.shape[-1] < 6:
        raise ValueError("actions must be [sample,step,edge,dim>=6]")
    totals = np.sum(np.clip(values[..., [OFFLOAD_COUNT_DIM, CPU_COUNT_DIM, RETURN_COUNT_DIM]], 0.0, None), axis=(1, 2))
    stage_names = np.asarray(["offload", "compute", "return"], dtype=object)
    maximum = np.max(totals, axis=1)
    selected = stage_names[np.argmax(totals, axis=1)]
    selected[maximum <= 0.0] = "unknown"
    return selected.astype(str)


def _topk_mask(score: np.ndarray, eligible: np.ndarray, k: int) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    eligible = np.asarray(eligible, dtype=bool)
    if score.shape != eligible.shape or score.ndim != 3:
        raise ValueError("selection score and support mask must be [sample,step,edge]")
    result = np.zeros_like(eligible)
    for sample in range(score.shape[0]):
        for step in range(1, score.shape[1]):
            indices = np.flatnonzero(eligible[sample, step])
            if indices.size == 0:
                continue
            order = np.lexsort((indices, -score[sample, step, indices]))
            selected = indices[order[: min(int(k), indices.size)]]
            result[sample, step, selected] = True
    return result


def _repair_rb(
    baseline: np.ndarray,
    selection_score: np.ndarray,
    target: np.ndarray | float,
    eligible: np.ndarray,
    k: int,
    pattern: str,
    cap_scale: float | None = None,
) -> np.ndarray:
    result = np.asarray(baseline, dtype=np.float32).copy()
    selected = _topk_mask(selection_score, eligible, int(k))
    target_values = np.broadcast_to(np.asarray(target, dtype=np.float32), selection_score.shape)
    if str(pattern) not in {"persistent", "decayed"}:
        raise ValueError(f"unknown repair pattern: {pattern}")
    for step in range(1, result.shape[1]):
        alpha = 1.0 if pattern == "persistent" or step == 1 else 0.5 ** (step - 1)
        current = result[:, step, :, RB_TOTAL_DIM]
        blended = (1.0 - alpha) * current + alpha * target_values[:, step]
        mask = selected[:, step]
        result[:, step, :, RB_COUNT_DIM] = np.where(
            mask,
            np.maximum(result[:, step, :, RB_COUNT_DIM], 1.0),
            result[:, step, :, RB_COUNT_DIM],
        )
        result[:, step, :, RB_TOTAL_DIM] = np.where(mask, blended, current)
        if cap_scale is not None:
            base_total = np.sum(np.clip(baseline[:, step, :, RB_TOTAL_DIM], 0.0, None), axis=1)
            repaired_total = np.sum(np.clip(result[:, step, :, RB_TOTAL_DIM], 0.0, None), axis=1)
            cap = np.maximum(base_total * float(cap_scale), base_total)
            scale = np.minimum(1.0, cap / np.maximum(repaired_total, 1e-8))
            result[:, step, :, RB_TOTAL_DIM] *= scale[:, None]
    result[:, 0] = baseline[:, 0]
    return result


def _stage_coupled_candidate(
    baseline: np.ndarray,
    selection_score: np.ndarray,
    eligible: np.ndarray,
    family: str,
    magnitude: float,
    k: int,
) -> np.ndarray:
    result = np.asarray(baseline, dtype=np.float32).copy()
    selected = _topk_mask(selection_score, eligible, int(k))
    for step in range(1, result.shape[1]):
        mask = selected[:, step]
        if family == "offload_rb":
            result[:, step, :, OFFLOAD_COUNT_DIM] = np.where(mask, 1.0, result[:, step, :, OFFLOAD_COUNT_DIM])
            result[:, step, :, RB_COUNT_DIM] = np.where(mask, 1.0, result[:, step, :, RB_COUNT_DIM])
            result[:, step, :, RB_TOTAL_DIM] = np.where(mask, float(magnitude), result[:, step, :, RB_TOTAL_DIM])
        elif family == "compute_cpu":
            result[:, step, :, CPU_COUNT_DIM] = np.where(mask, 1.0, result[:, step, :, CPU_COUNT_DIM])
            result[:, step, :, CPU_TOTAL_DIM] = np.where(mask, float(magnitude), result[:, step, :, CPU_TOTAL_DIM])
        elif family == "return_route":
            result[:, step, :, RETURN_COUNT_DIM] = np.where(mask, 1.0, result[:, step, :, RETURN_COUNT_DIM])
            result[:, step, :, RB_COUNT_DIM] = np.where(mask, 1.0, result[:, step, :, RB_COUNT_DIM])
            result[:, step, :, RB_TOTAL_DIM] = np.where(mask, float(magnitude), result[:, step, :, RB_TOTAL_DIM])
        else:
            raise ValueError(f"unknown coupled family: {family}")
    result[:, 0] = baseline[:, 0]
    return result


def _repair_relative(
    baseline: np.ndarray,
    selection_score: np.ndarray,
    eligible: np.ndarray,
    k: int,
    scale: float,
) -> np.ndarray:
    """Apply a fixed local residual around supported baseline magnitudes."""
    result = np.asarray(baseline, dtype=np.float32).copy()
    selected = _topk_mask(selection_score, eligible, int(k))
    for step in range(1, result.shape[1]):
        mask = selected[:, step]
        baseline_value = np.clip(baseline[:, step, :, RB_TOTAL_DIM], 0.0, None)
        result[:, step, :, RB_COUNT_DIM] = np.where(
            mask,
            np.maximum(result[:, step, :, RB_COUNT_DIM], 1.0),
            result[:, step, :, RB_COUNT_DIM],
        )
        result[:, step, :, RB_TOTAL_DIM] = np.where(
            mask,
            baseline_value * float(scale),
            result[:, step, :, RB_TOTAL_DIM],
        )
    result[:, 0] = baseline[:, 0]
    return result


def build_support_constrained_candidates(
    baseline_actions: np.ndarray,
    selection_score: np.ndarray,
    value_head: np.ndarray,
    train_positive_quantiles: Mapping[float, float],
    support_mask: np.ndarray,
    valid_element_mask: np.ndarray,
    stages: Sequence[str],
) -> CandidateLibrary:
    """Build the fixed 28-candidate v11 library without an old-grid sweep."""
    baseline = np.asarray(baseline_actions, dtype=np.float32)
    score = np.asarray(selection_score, dtype=np.float32)
    predicted_value = np.asarray(value_head, dtype=np.float32)
    support = np.asarray(support_mask, dtype=bool)
    valid = np.asarray(valid_element_mask, dtype=bool)
    if baseline.ndim != 4 or baseline.shape[-1] < 6:
        raise ValueError("baseline_actions must be [sample,step,edge,dim>=6]")
    expected = baseline.shape[:3]
    if score.shape != expected or predicted_value.shape != expected or support.shape != expected or valid.shape != expected:
        raise ValueError("candidate score/value/support tensors must match baseline sample-step-edge shape")
    q50 = float(train_positive_quantiles[0.5])
    q75 = float(train_positive_quantiles[0.75])
    eligible = support & valid
    names = ["identity"]
    families = ["identity"]
    actions = [baseline.copy()]
    initial_mask = [np.ones((baseline.shape[0],), dtype=bool)]

    names.append("ranked_allocation_baseline")
    families.append("rb_repair")
    actions.append(_repair_rb(baseline, score, predicted_value, eligible, 16, "persistent", cap_scale=1.15))
    initial_mask.append(np.ones((baseline.shape[0],), dtype=bool))

    magnitude_sources: tuple[tuple[str, np.ndarray | float], ...] = (
        ("value_head", predicted_value),
        ("q50", q50),
        ("q75", q75),
    )
    for k in (8, 16, 32):
        for magnitude_name, magnitude in magnitude_sources:
            for pattern in ("persistent", "decayed"):
                names.append(f"rb_repair__k{k}__{magnitude_name}__{pattern}")
                families.append("rb_repair")
                actions.append(_repair_rb(baseline, score, magnitude, eligible, k, pattern))
                initial_mask.append(np.ones((baseline.shape[0],), dtype=bool))

    stage_values = np.asarray(stages).astype(str).reshape(-1)
    if stage_values.shape[0] != baseline.shape[0]:
        raise ValueError("stages must match baseline sample dimension")
    for family, required_stage in (
        ("offload_rb", "offload"),
        ("compute_cpu", "compute"),
        ("return_route", "return"),
    ):
        for level, magnitude, k in (("low", q50, 8), ("high", q75, 16)):
            names.append(f"{family}_{level}")
            families.append(family)
            actions.append(_stage_coupled_candidate(baseline, score, eligible, family, magnitude, k))
            initial_mask.append(stage_values == required_stage)

    for cap_scale in (1.05, 1.10):
        names.append(f"historical_ranked_cap{int(round(cap_scale * 100)):03d}")
        families.append("historical_control")
        actions.append(_repair_rb(baseline, score, predicted_value, eligible, 16, "persistent", cap_scale=cap_scale))
        initial_mask.append(np.ones((baseline.shape[0],), dtype=bool))

    for direction, percent, k, scale in (
        ("expand", 25, 8, 1.25),
        ("expand", 50, 16, 1.50),
        ("shrink", 25, 8, 0.75),
        ("shrink", 50, 16, 0.50),
    ):
        names.append(f"benefit_residual__{direction}{percent}__k{k}")
        families.append("rb_benefit_residual")
        actions.append(_repair_relative(baseline, score, eligible, k, scale))
        initial_mask.append(np.ones((baseline.shape[0],), dtype=bool))

    stacked = np.stack(actions, axis=1)
    projected, applied = project_candidate_actions(
        stacked,
        baseline_actions=baseline,
        valid_element_mask=valid,
        candidate_families=families,
        stages=stage_values,
    )
    applicability_mask = np.stack(initial_mask, axis=1)
    candidate_mask = applicability_mask & applied
    candidate_mask[:, 0] = True
    return CandidateLibrary(
        actions=projected,
        candidate_names=tuple(names),
        action_families=tuple(families),
        candidate_mask=candidate_mask,
        applicability_mask=applicability_mask,
        action_applied=applied,
    )
