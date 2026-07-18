"""Objective-aligned opportunity and candidate-benefit selector primitives."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pi_jwm.evaluation.candidate_selection import choice_rmse_from_sample_sse
from pi_jwm.v11_selector import CandidateBatch, CandidateOutcome


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


@dataclass
class FittedObjectiveAlignedSelector:
    model: OpportunityBenefitRanker
    history: list[dict[str, float]]
    benefit_scale: float
    weight_cap: float
    candidate_mean: np.ndarray
    candidate_scale: np.ndarray
    context_mean: np.ndarray
    context_scale: np.ndarray
    hidden_dim: int
    dropout: float
    temperature: float
    impact_weighting: bool


def _stage_ids(stages: np.ndarray) -> torch.Tensor:
    vocabulary = {"unknown": 0, "offload": 1, "compute": 2, "return": 3}
    return torch.tensor(
        [vocabulary.get(str(value).lower(), 0) for value in stages],
        dtype=torch.long,
    )


def _normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    return mean, scale


def _weighted_candidate_huber(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    valid = mask & torch.isfinite(target)
    if not bool(valid.any()):
        return predicted.sum() * 0.0
    loss = F.huber_loss(predicted[valid], target[valid], reduction="none")
    expanded_weight = sample_weight[:, None].expand_as(predicted)[valid]
    return torch.sum(loss * expanded_weight) / torch.clamp(
        torch.sum(expanded_weight), min=1e-8
    )


def _heteroscedastic_nll(
    residual: torch.Tensor,
    uncertainty: torch.Tensor,
) -> torch.Tensor:
    sigma = torch.clamp(uncertainty, min=1e-4)
    return torch.mean(0.5 * torch.square(residual / sigma) + torch.log(sigma))


def fit_objective_aligned_selector(
    batch: CandidateBatch,
    outcome: CandidateOutcome,
    hidden_dim: int = 64,
    weight_cap: float = 5.0,
    epochs: int = 200,
    learning_rate: float = 3e-3,
    temperature: float = 0.25,
    dropout: float = 0.0,
    seed: int = 17,
    device: str | torch.device = "cpu",
    group_ids: np.ndarray | None = None,
    impact_weighting: bool = True,
) -> FittedObjectiveAlignedSelector:
    """Fit the objective-aligned selector using train-only statistics."""

    if batch.candidate_features.shape[:2] != outcome.active_sse.shape:
        raise ValueError("candidate batch and outcome shapes must match")
    if int(epochs) < 1:
        raise ValueError("epochs must be positive")
    groups = None if group_ids is None else np.asarray(group_ids).reshape(-1)
    if groups is not None and groups.shape[0] != batch.candidate_features.shape[0]:
        raise ValueError("group_ids must contain one value per sample")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    selected_device = torch.device(device)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    targets = build_decision_aligned_targets(
        outcome,
        batch.candidate_mask,
        weight_cap=float(weight_cap),
    )
    if not np.any(targets.valid_sample):
        raise ValueError("selector fitting requires at least one active target")
    valid_candidate = batch.candidate_features[batch.candidate_mask]
    candidate_mean, candidate_scale = _normalization(valid_candidate)
    context_mean, context_scale = _normalization(batch.context)
    normalized_candidate = np.clip(
        (batch.candidate_features - candidate_mean) / candidate_scale,
        -10.0,
        10.0,
    ).astype(np.float32)
    normalized_context = np.clip(
        (batch.context - context_mean) / context_scale,
        -10.0,
        10.0,
    ).astype(np.float32)

    model = OpportunityBenefitRanker(
        candidate_dim=batch.candidate_features.shape[2],
        context_dim=batch.context.shape[1],
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    ).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    candidate = torch.from_numpy(normalized_candidate).to(selected_device)
    context = torch.from_numpy(normalized_context).to(selected_device)
    mask = torch.from_numpy(batch.candidate_mask).to(selected_device)
    stage = _stage_ids(batch.stage).to(selected_device)
    valid_sample = torch.from_numpy(targets.valid_sample).to(selected_device)
    positive_opportunity = torch.from_numpy(targets.positive_opportunity).to(
        selected_device
    )
    sample_weight = torch.from_numpy(targets.sample_weight).to(selected_device)
    if not bool(impact_weighting):
        sample_weight = valid_sample.to(dtype=torch.float32)
    candidate_benefit = torch.from_numpy(
        targets.candidate_benefit / targets.benefit_scale
    ).to(selected_device)
    opportunity = torch.from_numpy(
        targets.opportunity / targets.benefit_scale
    ).to(selected_device)
    history: list[dict[str, float]] = []

    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        output = model(candidate, context, mask, stage)
        listwise_weight = sample_weight * positive_opportunity.to(sample_weight.dtype)
        if bool(positive_opportunity.any()):
            listwise = weighted_listwise_benefit_loss(
                output["predicted_candidate_benefit"],
                candidate_benefit,
                mask,
                listwise_weight,
                temperature=float(temperature),
            )
        else:
            listwise = output["predicted_candidate_benefit"].sum() * 0.0
        benefit = _weighted_candidate_huber(
            output["predicted_candidate_benefit"],
            candidate_benefit,
            mask & valid_sample[:, None],
            sample_weight,
        )
        opportunity_loss = F.huber_loss(
            output["predicted_opportunity"][valid_sample],
            opportunity[valid_sample],
        )
        candidate_valid = mask & valid_sample[:, None]
        uncertainty = _heteroscedastic_nll(
            output["predicted_candidate_benefit"][candidate_valid]
            - candidate_benefit[candidate_valid],
            output["candidate_uncertainty"][candidate_valid],
        ) + _heteroscedastic_nll(
            output["predicted_opportunity"][valid_sample]
            - opportunity[valid_sample],
            output["opportunity_uncertainty"][valid_sample],
        )
        group_losses = []
        if groups is not None:
            for group_value in np.unique(groups):
                group = torch.from_numpy(
                    (groups == group_value) & targets.positive_opportunity
                ).to(selected_device)
                if bool(group.any()):
                    group_losses.append(
                        weighted_listwise_benefit_loss(
                            output["predicted_candidate_benefit"],
                            candidate_benefit,
                            mask,
                            sample_weight * group.to(sample_weight.dtype),
                            temperature=float(temperature),
                        )
                    )
        worst_group = (
            torch.stack(group_losses).max()
            if group_losses
            else listwise.detach() * 0.0
        )
        loss = (
            listwise
            + 0.5 * benefit
            + 0.5 * opportunity_loss
            + 0.10 * worst_group
            + 0.05 * uncertainty
        )
        loss.backward()
        optimizer.step()
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(loss.detach()),
                "listwise": float(listwise.detach()),
                "candidate_benefit": float(benefit.detach()),
                "opportunity": float(opportunity_loss.detach()),
                "worst_group": float(worst_group.detach()),
                "uncertainty": float(uncertainty.detach()),
            }
        )

    model.eval()
    return FittedObjectiveAlignedSelector(
        model=model,
        history=history,
        benefit_scale=float(targets.benefit_scale),
        weight_cap=float(weight_cap),
        candidate_mean=candidate_mean,
        candidate_scale=candidate_scale,
        context_mean=context_mean,
        context_scale=context_scale,
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        temperature=float(temperature),
        impact_weighting=bool(impact_weighting),
    )


def predict_objective_aligned_selector(
    fitted: FittedObjectiveAlignedSelector,
    batch: CandidateBatch,
) -> dict[str, np.ndarray]:
    """Predict candidate and opportunity benefit in original SSE units."""

    if batch.candidate_features.shape[2] != fitted.candidate_mean.shape[0]:
        raise ValueError("candidate feature dimension does not match checkpoint")
    if batch.context.shape[1] != fitted.context_mean.shape[0]:
        raise ValueError("context feature dimension does not match checkpoint")
    device = next(fitted.model.parameters()).device
    candidate = np.clip(
        (batch.candidate_features - fitted.candidate_mean) / fitted.candidate_scale,
        -10.0,
        10.0,
    ).astype(np.float32)
    context = np.clip(
        (batch.context - fitted.context_mean) / fitted.context_scale,
        -10.0,
        10.0,
    ).astype(np.float32)
    fitted.model.eval()
    with torch.no_grad():
        output = fitted.model(
            torch.from_numpy(candidate).to(device),
            torch.from_numpy(context).to(device),
            torch.from_numpy(batch.candidate_mask).to(device),
            _stage_ids(batch.stage).to(device),
        )
    scale = float(fitted.benefit_scale)
    return {
        name: (value.detach().cpu().numpy().astype(np.float32) * scale)
        for name, value in output.items()
    }


def save_objective_aligned_checkpoint(
    path: str | Path,
    fitted: FittedObjectiveAlignedSelector,
    configuration_digest: str,
    training_seed: int,
) -> None:
    """Save a weights-only-safe objective-aligned selector checkpoint."""

    digest = str(configuration_digest)
    if len(digest) != 64:
        raise ValueError("configuration digest must contain 64 characters")
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu()
                for name, value in fitted.model.state_dict().items()
            },
            "candidate_dim": int(fitted.candidate_mean.shape[0]),
            "context_dim": int(fitted.context_mean.shape[0]),
            "hidden_dim": int(fitted.hidden_dim),
            "dropout": float(fitted.dropout),
            "temperature": float(fitted.temperature),
            "benefit_scale": float(fitted.benefit_scale),
            "weight_cap": float(fitted.weight_cap),
            "impact_weighting": bool(fitted.impact_weighting),
            "candidate_mean": torch.from_numpy(fitted.candidate_mean.copy()),
            "candidate_scale": torch.from_numpy(fitted.candidate_scale.copy()),
            "context_mean": torch.from_numpy(fitted.context_mean.copy()),
            "context_scale": torch.from_numpy(fitted.context_scale.copy()),
            "history": fitted.history,
            "configuration_digest": digest,
            "training_seed": int(training_seed),
        },
        checkpoint_path,
    )


def load_objective_aligned_checkpoint(
    path: str | Path,
    expected_configuration_digest: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[FittedObjectiveAlignedSelector, dict[str, Any]]:
    """Load and validate a weights-only-safe selector checkpoint."""

    payload = torch.load(
        Path(path), map_location=torch.device(device), weights_only=True
    )
    required = {
        "state_dict",
        "candidate_dim",
        "context_dim",
        "hidden_dim",
        "dropout",
        "temperature",
        "benefit_scale",
        "weight_cap",
        "candidate_mean",
        "candidate_scale",
        "context_mean",
        "context_scale",
        "configuration_digest",
        "training_seed",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"objective-aligned checkpoint missing fields: {missing}")
    digest = str(payload["configuration_digest"])
    if expected_configuration_digest is not None and digest != str(
        expected_configuration_digest
    ):
        raise ValueError("objective-aligned checkpoint configuration digest mismatch")
    model = OpportunityBenefitRanker(
        candidate_dim=int(payload["candidate_dim"]),
        context_dim=int(payload["context_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        dropout=float(payload["dropout"]),
    ).to(torch.device(device))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    def array(name: str) -> np.ndarray:
        return payload[name].detach().cpu().numpy().astype(np.float32)

    fitted = FittedObjectiveAlignedSelector(
        model=model,
        history=list(payload.get("history", [])),
        benefit_scale=float(payload["benefit_scale"]),
        weight_cap=float(payload["weight_cap"]),
        candidate_mean=array("candidate_mean"),
        candidate_scale=array("candidate_scale"),
        context_mean=array("context_mean"),
        context_scale=array("context_scale"),
        hidden_dim=int(payload["hidden_dim"]),
        dropout=float(payload["dropout"]),
        temperature=float(payload["temperature"]),
        impact_weighting=bool(payload.get("impact_weighting", True)),
    )
    metadata = {
        "configuration_digest": digest,
        "training_seed": int(payload["training_seed"]),
        "hidden_dim": int(payload["hidden_dim"]),
        "weight_cap": float(payload["weight_cap"]),
        "impact_weighting": bool(payload.get("impact_weighting", True)),
    }
    return fitted, metadata


@dataclass(frozen=True)
class OpportunityCalibration:
    quantile: float
    threshold: float
    rmse: float
    defer_ratio: float
    curve: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class ObjectiveAlignedDecision:
    candidate_index: np.ndarray
    proposed_candidate_index: np.ndarray
    predicted_benefit: np.ndarray
    candidate_uncertainty: np.ndarray
    candidate_lcb: np.ndarray
    predicted_opportunity: np.ndarray
    opportunity_uncertainty: np.ndarray
    opportunity_lcb: np.ndarray
    deferred: np.ndarray
    defer_reason: tuple[str, ...]

    def to_records(self, sample_ids: np.ndarray | None = None) -> list[dict[str, Any]]:
        count = self.candidate_index.shape[0]
        ids = np.arange(count) if sample_ids is None else np.asarray(sample_ids).reshape(-1)
        if ids.shape[0] != count:
            raise ValueError("sample_ids must contain one value per decision")
        return [
            {
                "sample_id": int(ids[index]),
                "candidate_index": int(self.candidate_index[index]),
                "proposed_candidate_index": int(
                    self.proposed_candidate_index[index]
                ),
                "predicted_benefit": float(self.predicted_benefit[index]),
                "candidate_uncertainty": float(
                    self.candidate_uncertainty[index]
                ),
                "candidate_lcb": float(self.candidate_lcb[index]),
                "predicted_opportunity": float(
                    self.predicted_opportunity[index]
                ),
                "opportunity_uncertainty": float(
                    self.opportunity_uncertainty[index]
                ),
                "opportunity_lcb": float(self.opportunity_lcb[index]),
                "deferred": bool(self.deferred[index]),
                "defer_reason": self.defer_reason[index],
            }
            for index in range(count)
        ]


def calibrate_opportunity_threshold(
    opportunity_lcb: np.ndarray,
    candidate_choice: np.ndarray,
    outcome: CandidateOutcome,
    quantiles: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9),
) -> OpportunityCalibration:
    """Choose the opportunity threshold on calibration outcomes only."""

    lower_bound = np.asarray(opportunity_lcb, dtype=np.float64).reshape(-1)
    choice = np.asarray(candidate_choice, dtype=np.int64).reshape(-1)
    if lower_bound.shape[0] != outcome.active_sse.shape[0] or choice.shape != lower_bound.shape:
        raise ValueError("calibration arrays must contain one value per sample")
    if not np.all(np.isfinite(lower_bound)):
        raise ValueError("opportunity lower confidence bounds must be finite")
    if np.any(choice < 0) or np.any(choice >= outcome.active_sse.shape[1]):
        raise ValueError("candidate choice outside outcome dimensions")
    fixed_quantiles = tuple(float(value) for value in quantiles)
    if fixed_quantiles != (0.0, 0.25, 0.5, 0.75, 0.9):
        raise ValueError("opportunity calibration quantiles are protocol-fixed")
    rows = []
    for quantile in fixed_quantiles:
        threshold = float(np.quantile(lower_bound, quantile))
        execute = lower_bound > threshold
        calibrated_choice = np.where(execute, choice, outcome.default_index).astype(
            np.int64
        )
        rmse = choice_rmse_from_sample_sse(
            outcome.active_sse,
            outcome.active_count,
            calibrated_choice,
        )
        if rmse is None:
            raise ValueError("opportunity calibration requires an active target")
        rows.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                "rmse": float(rmse),
                "defer_ratio": float(np.mean(~execute)),
            }
        )
    best = min(
        rows,
        key=lambda row: (
            row["rmse"],
            -row["defer_ratio"],
            -row["threshold"],
            row["quantile"],
        ),
    )
    return OpportunityCalibration(
        quantile=float(best["quantile"]),
        threshold=float(best["threshold"]),
        rmse=float(best["rmse"]),
        defer_ratio=float(best["defer_ratio"]),
        curve=tuple(dict(row) for row in rows),
    )


def _total_uncertainty(
    ensemble_prediction: np.ndarray,
    ensemble_uncertainty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(ensemble_prediction, dtype=np.float64)
    uncertainty = np.asarray(ensemble_uncertainty, dtype=np.float64)
    if prediction.shape != uncertainty.shape or prediction.ndim < 2:
        raise ValueError("ensemble prediction and uncertainty must share shape")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(uncertainty)):
        raise ValueError("ensemble values must be finite")
    if np.any(uncertainty < 0.0):
        raise ValueError("predicted uncertainty must be non-negative")
    mean = prediction.mean(axis=0)
    epistemic = prediction.std(axis=0, ddof=0)
    aleatoric_variance = np.mean(np.square(uncertainty), axis=0)
    total = np.sqrt(np.square(epistemic) + aleatoric_variance)
    return mean, total


def _pareto_dominated(task: np.ndarray, energy: np.ndarray) -> np.ndarray:
    dominated = np.zeros(task.shape[0], dtype=bool)
    for candidate in range(task.shape[0]):
        no_worse = (task >= task[candidate]) & (energy <= energy[candidate])
        strictly_better = (task > task[candidate]) | (energy < energy[candidate])
        dominated[candidate] = bool(np.any(no_worse & strictly_better))
    return dominated


def select_objective_aligned(
    ensemble_candidate_benefit: np.ndarray,
    ensemble_candidate_uncertainty: np.ndarray,
    ensemble_opportunity: np.ndarray,
    ensemble_opportunity_uncertainty: np.ndarray,
    candidate_mask: np.ndarray,
    default_index: int,
    opportunity_threshold: float,
    task_delta: np.ndarray | None = None,
    energy_delta: np.ndarray | None = None,
    z_value: float = 1.64,
) -> ObjectiveAlignedDecision:
    """Select a positive-benefit candidate or defer to the ranked baseline."""

    candidate_prediction = np.asarray(ensemble_candidate_benefit, dtype=np.float64)
    if candidate_prediction.ndim != 3 or candidate_prediction.shape[0] < 1:
        raise ValueError("candidate benefit ensemble must be [model,sample,candidate]")
    candidate_mean, candidate_std = _total_uncertainty(
        candidate_prediction,
        ensemble_candidate_uncertainty,
    )
    opportunity_prediction = np.asarray(ensemble_opportunity, dtype=np.float64)
    if opportunity_prediction.ndim != 2 or opportunity_prediction.shape[0] < 1:
        raise ValueError("opportunity ensemble must be [model,sample]")
    opportunity_mean, opportunity_std = _total_uncertainty(
        opportunity_prediction,
        ensemble_opportunity_uncertainty,
    )
    _, sample_count, candidate_count = candidate_prediction.shape
    if opportunity_mean.shape != (sample_count,):
        raise ValueError("opportunity and candidate ensembles must share samples")
    mask = np.asarray(candidate_mask, dtype=bool)
    if mask.shape != (sample_count, candidate_count):
        raise ValueError("candidate mask must match ensemble dimensions")
    default = int(default_index)
    if not 0 <= default < candidate_count or not np.all(mask[:, default]):
        raise ValueError("ranked default must be valid for every sample")
    if not np.isfinite(opportunity_threshold):
        raise ValueError("opportunity threshold must be finite")
    if (task_delta is None) != (energy_delta is None):
        raise ValueError("task and energy deltas must be supplied together")
    if task_delta is not None:
        task = np.asarray(task_delta, dtype=np.float64)
        energy = np.asarray(energy_delta, dtype=np.float64)
        if task.shape != mask.shape or energy.shape != mask.shape:
            raise ValueError("task and energy deltas must match candidates")
    else:
        task = energy = None

    candidate_lcb_all = candidate_mean - float(z_value) * candidate_std
    opportunity_lcb_all = opportunity_mean - float(z_value) * opportunity_std
    selected = np.full(sample_count, default, dtype=np.int64)
    proposed = np.full(sample_count, default, dtype=np.int64)
    predicted_benefit = np.zeros(sample_count, dtype=np.float64)
    selected_uncertainty = np.zeros(sample_count, dtype=np.float64)
    selected_lcb = np.zeros(sample_count, dtype=np.float64)
    deferred = np.ones(sample_count, dtype=bool)
    reasons = []
    for sample in range(sample_count):
        allowed = mask[sample].copy()
        allowed[default] = False
        pareto_removed = False
        if task is not None and energy is not None:
            dominated = _pareto_dominated(task[sample], energy[sample])
            pareto_removed = bool(np.any(allowed & dominated))
            allowed &= ~dominated
        if not np.any(allowed):
            reasons.append(
                "pareto_dominated" if pareto_removed else "no_nondefault_candidate"
            )
            continue
        candidate = int(
            np.argmax(np.where(allowed, candidate_mean[sample], -np.inf))
        )
        proposed[sample] = candidate
        predicted_benefit[sample] = candidate_mean[sample, candidate]
        selected_uncertainty[sample] = candidate_std[sample, candidate]
        selected_lcb[sample] = candidate_lcb_all[sample, candidate]
        if opportunity_lcb_all[sample] <= float(opportunity_threshold):
            reasons.append("opportunity_below_threshold")
        elif selected_lcb[sample] <= 0.0:
            reasons.append("candidate_nonpositive_lcb")
        else:
            selected[sample] = candidate
            deferred[sample] = False
            reasons.append("")
    return ObjectiveAlignedDecision(
        candidate_index=selected,
        proposed_candidate_index=proposed,
        predicted_benefit=predicted_benefit.astype(np.float32),
        candidate_uncertainty=selected_uncertainty.astype(np.float32),
        candidate_lcb=selected_lcb.astype(np.float32),
        predicted_opportunity=opportunity_mean.astype(np.float32),
        opportunity_uncertainty=opportunity_std.astype(np.float32),
        opportunity_lcb=opportunity_lcb_all.astype(np.float32),
        deferred=deferred,
        defer_reason=tuple(reasons),
    )
