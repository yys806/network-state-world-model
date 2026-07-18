"""Objective-aligned opportunity and candidate-benefit selector primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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


class OpportunityBenefitRanker(nn.Module):
    """Permutation-equivariant candidate benefit and sample opportunity model."""

    def __init__(
        self,
        candidate_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        num_stages: int = 4,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(int(candidate_dim), hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(int(context_dim), hidden),
            nn.ReLU(),
        )
        self.stage_embedding = nn.Embedding(int(num_stages), hidden)
        candidate_joint_dim = hidden * 4
        opportunity_joint_dim = hidden * 3
        self.candidate_benefit_head = nn.Sequential(
            nn.Linear(candidate_joint_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.candidate_uncertainty_head = nn.Sequential(
            nn.Linear(candidate_joint_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.opportunity_head = nn.Sequential(
            nn.Linear(opportunity_joint_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.opportunity_uncertainty_head = nn.Sequential(
            nn.Linear(opportunity_joint_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        context: torch.Tensor,
        candidate_mask: torch.Tensor,
        stage_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if candidate_features.ndim != 3 or context.ndim != 2:
            raise ValueError(
                "ranker inputs must be [sample,candidate,feature] and [sample,context]"
            )
        if context.shape[0] != candidate_features.shape[0]:
            raise ValueError("candidate features and context must share sample dimension")
        mask = candidate_mask.to(dtype=torch.bool)
        if mask.shape != candidate_features.shape[:2] or not bool(mask.any(dim=1).all()):
            raise ValueError("candidate mask must contain a valid candidate per sample")
        encoded = self.candidate_encoder(candidate_features)
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        pooled = torch.sum(encoded * weights, dim=1) / torch.clamp(
            torch.sum(weights, dim=1), min=1.0
        )
        context_encoded = self.context_encoder(context)
        if stage_ids is None:
            stage_ids = torch.zeros(
                (candidate_features.shape[0],),
                dtype=torch.long,
                device=candidate_features.device,
            )
        stage_encoded = self.stage_embedding(stage_ids.to(dtype=torch.long))
        candidate_count = candidate_features.shape[1]
        candidate_joint = torch.cat(
            [
                encoded,
                pooled[:, None, :].expand(-1, candidate_count, -1),
                context_encoded[:, None, :].expand(-1, candidate_count, -1),
                stage_encoded[:, None, :].expand(-1, candidate_count, -1),
            ],
            dim=-1,
        )
        opportunity_joint = torch.cat(
            [pooled, context_encoded, stage_encoded], dim=-1
        )
        candidate_benefit = self.candidate_benefit_head(candidate_joint).squeeze(-1)
        candidate_uncertainty = (
            F.softplus(self.candidate_uncertainty_head(candidate_joint).squeeze(-1))
            + 1e-6
        )
        opportunity = self.opportunity_head(opportunity_joint).squeeze(-1)
        opportunity_uncertainty = (
            F.softplus(self.opportunity_uncertainty_head(opportunity_joint).squeeze(-1))
            + 1e-6
        )
        return {
            "predicted_candidate_benefit": candidate_benefit.masked_fill(
                ~mask, -1e9
            ),
            "candidate_uncertainty": candidate_uncertainty.masked_fill(~mask, 0.0),
            "predicted_opportunity": opportunity,
            "opportunity_uncertainty": opportunity_uncertainty,
        }


def weighted_listwise_benefit_loss(
    predicted_benefit: torch.Tensor,
    target_benefit: torch.Tensor,
    candidate_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    temperature: float = 0.25,
) -> torch.Tensor:
    """Cross entropy over candidate benefit distributions with impact weights."""

    if predicted_benefit.shape != target_benefit.shape:
        raise ValueError("predicted and target benefit must share shape")
    mask = candidate_mask.to(dtype=torch.bool)
    if mask.shape != predicted_benefit.shape:
        raise ValueError("candidate mask must match benefit tensors")
    weights = sample_weight.reshape(-1).to(predicted_benefit.dtype)
    if weights.shape[0] != predicted_benefit.shape[0]:
        raise ValueError("sample weight must contain one value per sample")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise ValueError("sample weight must be finite and non-negative")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    finite_target = torch.isfinite(target_benefit.masked_fill(~mask, 0.0)).all(dim=1)
    valid = mask.any(dim=1) & finite_target & (weights > 0.0)
    if not bool(valid.any()):
        raise ValueError("weighted listwise loss requires a positive-weight row")
    target_logits = (target_benefit / float(temperature)).masked_fill(~mask, -1e9)
    target = torch.softmax(target_logits[valid], dim=1)
    predicted_log_probability = torch.log_softmax(
        predicted_benefit.masked_fill(~mask, -1e9)[valid], dim=1
    )
    per_sample = -(target * predicted_log_probability).sum(dim=1)
    valid_weight = weights[valid]
    return torch.sum(per_sample * valid_weight) / torch.clamp(
        torch.sum(valid_weight), min=1e-8
    )
