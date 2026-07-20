"""Token-level opportunity and candidate-benefit selector for PI-JWM v11."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .v11_interactions import CandidateInteractionBatch
from .v11_selector import CandidateBatch, CandidateOutcome


class InteractionCandidateBenefitRanker(nn.Module):
    """Encode sparse edge-step interactions before candidate-set ranking."""

    def __init__(
        self,
        candidate_dim: int,
        context_dim: int,
        token_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        num_stages: int = 4,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim)
        self.token_encoder = nn.Sequential(
            nn.Linear(int(token_dim), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(int(candidate_dim), hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(int(context_dim), hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.stage_embedding = nn.Embedding(int(num_stages), hidden)
        candidate_joint = hidden * 6
        sample_joint = hidden * 3
        self.candidate_trunk = nn.Sequential(
            nn.Linear(candidate_joint, hidden), nn.ReLU(), nn.Dropout(float(dropout))
        )
        self.score_head = nn.Linear(hidden, 1)
        self.benefit_head = nn.Linear(hidden, 1)
        self.sign_head = nn.Linear(hidden, 1)
        self.uncertainty_head = nn.Linear(hidden, 1)
        self.opportunity_head = nn.Sequential(
            nn.Linear(sample_joint, hidden), nn.ReLU(), nn.Dropout(float(dropout)), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        context: torch.Tensor,
        interaction_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        stage_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if candidate_features.ndim != 3 or context.ndim != 2 or interaction_tokens.ndim != 4:
            raise ValueError("interaction ranker inputs have invalid dimensions")
        if interaction_tokens.shape[:2] != candidate_features.shape[:2]:
            raise ValueError("interaction tokens must match sample and candidate dimensions")
        token_valid = token_mask.to(dtype=torch.bool)
        if token_valid.shape != interaction_tokens.shape[:3]:
            raise ValueError("token mask must match interaction token dimensions")
        candidate_valid = candidate_mask.to(dtype=torch.bool)
        if candidate_valid.shape != candidate_features.shape[:2]:
            raise ValueError("candidate mask must match candidate dimensions")

        clean_tokens = interaction_tokens.masked_fill(~token_valid[..., None], 0.0)
        encoded_tokens = self.token_encoder(clean_tokens)
        token_weights = token_valid.to(encoded_tokens.dtype)[..., None]
        token_count = torch.clamp(token_weights.sum(dim=2), min=1.0)
        token_mean = (encoded_tokens * token_weights).sum(dim=2) / token_count
        token_max = encoded_tokens.masked_fill(~token_valid[..., None], -torch.inf).max(dim=2).values
        token_max = torch.where(torch.isfinite(token_max), token_max, torch.zeros_like(token_max))

        candidate_encoded = self.candidate_encoder(candidate_features)
        candidate_weights = candidate_valid.to(candidate_encoded.dtype)[..., None]
        candidate_count = torch.clamp(candidate_weights.sum(dim=1), min=1.0)
        candidate_pool = (candidate_encoded * candidate_weights).sum(dim=1) / candidate_count
        context_encoded = self.context_encoder(context)
        if stage_ids is None:
            stage_ids = torch.zeros(
                candidate_features.shape[0], dtype=torch.long, device=candidate_features.device
            )
        stage_encoded = self.stage_embedding(stage_ids.to(dtype=torch.long))

        count = candidate_features.shape[1]
        joint = torch.cat(
            [
                candidate_encoded,
                token_mean,
                token_max,
                candidate_pool[:, None].expand(-1, count, -1),
                context_encoded[:, None].expand(-1, count, -1),
                stage_encoded[:, None].expand(-1, count, -1),
            ],
            dim=-1,
        )
        candidate_hidden = self.candidate_trunk(joint)
        score = self.score_head(candidate_hidden).squeeze(-1)
        benefit = self.benefit_head(candidate_hidden).squeeze(-1)
        sign = self.sign_head(candidate_hidden).squeeze(-1)
        uncertainty = F.softplus(self.uncertainty_head(candidate_hidden).squeeze(-1)) + 1e-6
        score = score.masked_fill(~candidate_valid, -1e9)
        benefit = benefit.masked_fill(~candidate_valid, 0.0)
        sign = sign.masked_fill(~candidate_valid, -1e9)
        uncertainty = uncertainty.masked_fill(~candidate_valid, 0.0)
        opportunity_input = torch.cat([candidate_pool, context_encoded, stage_encoded], dim=-1)
        opportunity = self.opportunity_head(opportunity_input).squeeze(-1)
        return {
            "opportunity_logit": opportunity,
            "candidate_score": score,
            "predicted_benefit": benefit,
            "candidate_sign_logit": sign,
            "uncertainty": uncertainty,
        }


def interaction_selector_loss(
    outputs: dict[str, torch.Tensor],
    candidate_benefit: torch.Tensor,
    candidate_mask: torch.Tensor,
    default_index: int,
    benefit_scale: float = 1.0,
    temperature: float = 0.25,
) -> dict[str, Any]:
    """Train opportunity on all rows and candidate order only where headroom exists."""
    benefit = candidate_benefit
    mask = candidate_mask.to(dtype=torch.bool)
    score = outputs["candidate_score"]
    prediction = outputs["predicted_benefit"]
    sign_logit = outputs["candidate_sign_logit"]
    if benefit.shape != mask.shape or score.shape != mask.shape:
        raise ValueError("benefit, scores, and candidate mask must share shape")
    if not 0 <= int(default_index) < benefit.shape[1]:
        raise ValueError("default index outside candidate dimension")
    if float(benefit_scale) <= 0.0 or float(temperature) <= 0.0:
        raise ValueError("benefit scale and temperature must be positive")
    finite = torch.isfinite(benefit)
    legal = mask & finite
    if not bool(legal.any()):
        raise ValueError("interaction selector loss requires a legal candidate")
    masked_benefit = benefit.masked_fill(~legal, -torch.inf)
    best_benefit = masked_benefit.max(dim=1).values
    opportunity_target = best_benefit > 1e-8
    opportunity = F.binary_cross_entropy_with_logits(
        outputs["opportunity_logit"], opportunity_target.to(score.dtype)
    )

    non_default = legal.clone()
    non_default[:, int(default_index)] = False
    if bool(non_default.any()):
        sign = F.binary_cross_entropy_with_logits(
            sign_logit[non_default], (benefit[non_default] > 1e-8).to(score.dtype)
        )
    else:
        sign = score.sum() * 0.0

    transformed = torch.asinh(benefit / float(benefit_scale))
    opportunity_rows = opportunity_target & legal.any(dim=1)
    if bool(opportunity_rows.any()):
        row_mask = legal[opportunity_rows]
        target_logits = (transformed[opportunity_rows] / float(temperature)).masked_fill(
            ~row_mask, -1e9
        )
        target = torch.softmax(target_logits, dim=1)
        log_probability = torch.log_softmax(
            score[opportunity_rows].masked_fill(~row_mask, -1e9), dim=1
        )
        ranking = -(target * log_probability).sum(dim=1).mean()
        regression = F.smooth_l1_loss(
            prediction[opportunity_rows][row_mask], transformed[opportunity_rows][row_mask]
        )
    else:
        ranking = score.sum() * 0.0
        regression = score.sum() * 0.0
    loss = ranking + 0.25 * opportunity + 0.25 * sign + 0.10 * regression
    return {
        "loss": loss,
        "ranking": ranking,
        "opportunity": opportunity,
        "sign": sign,
        "regression": regression,
        "opportunity_row_count": int(opportunity_rows.sum().item()),
    }


@dataclass(frozen=True)
class InteractionNormalizer:
    candidate_mean: np.ndarray
    candidate_scale: np.ndarray
    context_mean: np.ndarray
    context_scale: np.ndarray
    token_mean: np.ndarray
    token_scale: np.ndarray


@dataclass
class FittedInteractionSelector:
    model: InteractionCandidateBenefitRanker
    normalizer: InteractionNormalizer
    benefit_scale: float
    history: list[dict[str, float]]
    hidden_dim: int
    dropout: float


def _mean_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or not np.all(np.isfinite(array)):
        raise ValueError("normalization requires a finite non-empty matrix")
    mean = array.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = array.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def fit_interaction_normalizer(
    batch: CandidateBatch,
    interactions: CandidateInteractionBatch,
) -> InteractionNormalizer:
    if interactions.tokens.shape[:2] != batch.candidate_features.shape[:2]:
        raise ValueError("candidate and interaction dimensions must match")
    candidate_mean, candidate_scale = _mean_scale(
        batch.candidate_features[batch.candidate_mask]
    )
    context_mean, context_scale = _mean_scale(batch.context)
    token_mean, token_scale = _mean_scale(interactions.tokens[interactions.token_mask])
    return InteractionNormalizer(
        candidate_mean=candidate_mean,
        candidate_scale=candidate_scale,
        context_mean=context_mean,
        context_scale=context_scale,
        token_mean=token_mean,
        token_scale=token_scale,
    )


def _normalized_slice(
    batch: CandidateBatch,
    interactions: CandidateInteractionBatch,
    normalizer: InteractionNormalizer,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate = np.clip(
        (batch.candidate_features[indices] - normalizer.candidate_mean)
        / normalizer.candidate_scale,
        -10.0,
        10.0,
    ).astype(np.float32)
    context = np.clip(
        (batch.context[indices] - normalizer.context_mean) / normalizer.context_scale,
        -10.0,
        10.0,
    ).astype(np.float32)
    token_mask = interactions.token_mask[indices]
    tokens = interactions.tokens[indices].copy()
    tokens[~token_mask] = normalizer.token_mean
    tokens = np.clip(
        (tokens - normalizer.token_mean) / normalizer.token_scale, -10.0, 10.0
    ).astype(np.float32)
    tokens[~token_mask] = 0.0
    return candidate, context, tokens, token_mask


def _stage_array(stages: np.ndarray) -> np.ndarray:
    vocabulary = {"unknown": 0, "offload": 1, "compute": 2, "return": 3}
    return np.asarray(
        [vocabulary.get(str(value).lower(), 0) for value in stages], dtype=np.int64
    )


def _legal_candidate_mask(batch: CandidateBatch, outcome: CandidateOutcome) -> np.ndarray:
    legal = batch.candidate_mask.copy()
    if outcome.action_applicable is not None:
        legal &= outcome.action_applicable
    if outcome.action_applied is not None:
        legal &= outcome.action_applied
    legal[:, outcome.default_index] = batch.candidate_mask[:, outcome.default_index]
    return legal


def fit_interaction_selector(
    batch: CandidateBatch,
    outcome: CandidateOutcome,
    interactions: CandidateInteractionBatch,
    hidden_dim: int = 64,
    dropout: float = 0.0,
    temperature: float = 0.25,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 17,
    device: str | torch.device = "cpu",
) -> FittedInteractionSelector:
    if batch.candidate_features.shape[:2] != outcome.active_sse.shape:
        raise ValueError("candidate batch and outcome dimensions must match")
    if interactions.tokens.shape[:2] != outcome.active_sse.shape:
        raise ValueError("interaction and outcome dimensions must match")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    selected_device = torch.device(device)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    normalizer = fit_interaction_normalizer(batch, interactions)
    model = InteractionCandidateBenefitRanker(
        candidate_dim=batch.candidate_features.shape[2],
        context_dim=batch.context.shape[1],
        token_dim=interactions.tokens.shape[3],
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    ).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-4
    )
    legal = _legal_candidate_mask(batch, outcome)
    valid_indices = np.flatnonzero(outcome.active_count > 0).astype(np.int64)
    if valid_indices.size == 0:
        raise ValueError("interaction selector training requires active-rate targets")
    benefit = outcome.active_sse[:, outcome.default_index, None] - outcome.active_sse
    benefit = benefit.astype(np.float32)
    benefit[~legal] = np.nan
    positive = benefit[legal & np.isfinite(benefit) & (benefit > 1e-8)]
    benefit_scale = float(np.median(positive)) if positive.size else 1.0
    benefit_scale = max(benefit_scale, 1e-6)
    stage_ids = _stage_array(batch.stage)
    rng = np.random.default_rng(int(seed))
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(valid_indices)
        totals: dict[str, float] = {
            "loss": 0.0,
            "ranking": 0.0,
            "opportunity": 0.0,
            "sign": 0.0,
            "regression": 0.0,
        }
        seen = 0
        for start in range(0, order.size, int(batch_size)):
            indices = order[start : start + int(batch_size)]
            candidate, context, tokens, token_mask = _normalized_slice(
                batch, interactions, normalizer, indices
            )
            optimizer.zero_grad()
            outputs = model(
                torch.from_numpy(candidate).to(selected_device),
                torch.from_numpy(context).to(selected_device),
                torch.from_numpy(tokens).to(selected_device),
                torch.from_numpy(token_mask).to(selected_device),
                torch.from_numpy(legal[indices]).to(selected_device),
                torch.from_numpy(stage_ids[indices]).to(selected_device),
            )
            losses = interaction_selector_loss(
                outputs,
                torch.from_numpy(benefit[indices]).to(selected_device),
                torch.from_numpy(legal[indices]).to(selected_device),
                default_index=outcome.default_index,
                benefit_scale=benefit_scale,
                temperature=float(temperature),
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            count = int(indices.size)
            seen += count
            for name in totals:
                totals[name] += float(losses[name].detach().cpu()) * count
        history.append(
            {"epoch": float(epoch + 1), **{name: value / max(seen, 1) for name, value in totals.items()}}
        )
    model.eval()
    return FittedInteractionSelector(
        model=model,
        normalizer=normalizer,
        benefit_scale=benefit_scale,
        history=history,
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    )


def predict_interaction_selector(
    fitted: FittedInteractionSelector,
    batch: CandidateBatch,
    interactions: CandidateInteractionBatch,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    fitted.model.eval()
    device = next(fitted.model.parameters()).device
    stage_ids = _stage_array(batch.stage)
    collected: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, batch.context.shape[0], int(batch_size)):
            indices = np.arange(
                start, min(start + int(batch_size), batch.context.shape[0]), dtype=np.int64
            )
            candidate, context, tokens, token_mask = _normalized_slice(
                batch, interactions, fitted.normalizer, indices
            )
            outputs = fitted.model(
                torch.from_numpy(candidate).to(device),
                torch.from_numpy(context).to(device),
                torch.from_numpy(tokens).to(device),
                torch.from_numpy(token_mask).to(device),
                torch.from_numpy(batch.candidate_mask[indices]).to(device),
                torch.from_numpy(stage_ids[indices]).to(device),
            )
            for name, values in outputs.items():
                collected.setdefault(name, []).append(values.detach().cpu().numpy())
    result = {name: np.concatenate(values, axis=0).astype(np.float32) for name, values in collected.items()}
    result["opportunity_probability"] = 1.0 / (
        1.0 + np.exp(-result["opportunity_logit"])
    )
    result["candidate_sign_probability"] = 1.0 / (
        1.0 + np.exp(-np.clip(result["candidate_sign_logit"], -30.0, 30.0))
    )
    result["transformed_benefit"] = result["predicted_benefit"].copy()
    result["predicted_benefit"] = (
        np.sinh(np.clip(result["predicted_benefit"], -10.0, 10.0))
        * float(fitted.benefit_scale)
    ).astype(np.float32)
    return result


def select_interaction_candidates(
    opportunity_probability: np.ndarray,
    candidate_score: np.ndarray,
    candidate_sign_probability: np.ndarray,
    candidate_mask: np.ndarray,
    default_index: int,
    opportunity_threshold: float,
    sign_threshold: float,
    pareto_allowed: np.ndarray | None = None,
) -> np.ndarray:
    opportunity = np.asarray(opportunity_probability, dtype=np.float32).reshape(-1)
    score = np.asarray(candidate_score, dtype=np.float32)
    sign = np.asarray(candidate_sign_probability, dtype=np.float32)
    mask = np.asarray(candidate_mask, dtype=bool)
    if score.shape != sign.shape or score.shape != mask.shape or score.shape[0] != opportunity.size:
        raise ValueError("interaction selector prediction dimensions must match")
    allowed = mask.copy()
    if pareto_allowed is not None:
        pareto = np.asarray(pareto_allowed, dtype=bool)
        if pareto.shape != allowed.shape:
            raise ValueError("Pareto allowed mask must match candidate dimensions")
        allowed &= pareto
    default = int(default_index)
    allowed[:, default] = mask[:, default]
    choice = np.full(opportunity.shape[0], default, dtype=np.int64)
    for sample in range(opportunity.shape[0]):
        if opportunity[sample] < float(opportunity_threshold):
            continue
        eligible = allowed[sample] & (sign[sample] >= float(sign_threshold))
        eligible[default] = False
        candidates = np.flatnonzero(eligible)
        if candidates.size:
            choice[sample] = int(candidates[np.argmax(score[sample, candidates])])
    return choice
