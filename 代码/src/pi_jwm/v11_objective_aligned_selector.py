"""Objective-aligned opportunity and candidate-benefit selector primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pi_jwm.v11_selector import CandidateOutcome


@dataclass(frozen=True)
class DecisionAlignedTargets:
    """Train-only SSE-benefit targets and bounded sample impact weights."""

    candidate_benefit: np.ndarray
    opportunity: np.ndarray
    positive_opportunity: np.ndarray
    valid_sample: np.ndarray
    sample_weight: np.ndarray
    benefit_scale: float
    weight_cap: float


def build_decision_aligned_targets(
    outcome: CandidateOutcome,
    candidate_mask: np.ndarray,
    weight_cap: float = 5.0,
    base_weight: float = 0.25,
    benefit_scale: float | None = None,
) -> DecisionAlignedTargets:
    """Align selector labels with aggregate SSE rather than equal-sample RMSE."""

    mask = np.asarray(candidate_mask, dtype=bool)
    if mask.shape != outcome.active_sse.shape:
        raise ValueError("candidate mask must match outcome dimensions")
    if not np.all(mask[:, outcome.default_index]):
        raise ValueError("candidate mask must include the ranked default")
    if not np.isfinite(weight_cap) or float(weight_cap) <= 0.0:
        raise ValueError("weight cap must be finite and positive")
    if not np.isfinite(base_weight) or float(base_weight) < 0.0:
        raise ValueError("base weight must be finite and non-negative")

    default_sse = outcome.active_sse[:, outcome.default_index, None]
    candidate_benefit = np.where(
        mask,
        default_sse - outcome.active_sse,
        np.nan,
    ).astype(np.float32)
    valid_sample = outcome.active_count > 0
    opportunity = np.zeros(outcome.active_count.shape, dtype=np.float32)
    if np.any(valid_sample):
        opportunity[valid_sample] = np.maximum(
            0.0,
            np.nanmax(candidate_benefit[valid_sample], axis=1),
        )
    positive_opportunity = valid_sample & (opportunity > 1e-8)

    if benefit_scale is None:
        scale = (
            float(np.median(opportunity[positive_opportunity]))
            if np.any(positive_opportunity)
            else 1.0
        )
    else:
        scale = float(benefit_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("benefit scale must be finite and positive")

    sample_weight = np.zeros_like(opportunity)
    sample_weight[valid_sample] = float(base_weight) + np.minimum(
        opportunity[valid_sample] / scale,
        float(weight_cap),
    )
    return DecisionAlignedTargets(
        candidate_benefit=candidate_benefit,
        opportunity=opportunity,
        positive_opportunity=positive_opportunity,
        valid_sample=valid_sample,
        sample_weight=sample_weight,
        benefit_scale=scale,
        weight_cap=float(weight_cap),
    )
