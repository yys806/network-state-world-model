"""CPU-auditable candidate-action rollout mechanism for PI-JWM.

This module is deliberately a mechanism prototype.  It provides the exact
candidate-wise world-model loop needed by the formal planner definition, but
does not claim that a trained policy, legal-action generator, or task-specific
objective has been frozen for the PI-JWM experiments.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


REQUIRED_FUTURE_ACTION_FIELDS = (
    "task_action",
    "task_action_present",
    "task_action_node_index",
    "task_action_source_node_index",
)
MECHANISM_SCHEMA_VERSION = "PI-JWM-formal-candidate-rollout-mechanism-v1"


def _clone_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    return value


def _tensor_fingerprint(namespace: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(path.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, Mapping):
            for key in sorted(value, key=str):
                visit(value[key], f"{path}.{key}")

    visit(namespace, "root")
    return digest.hexdigest()


def _finite_float(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


@dataclass(frozen=True)
class CandidateAction:
    """One legal action sequence sharing the planner's starting history."""

    candidate_id: str
    future_action: Mapping[str, Tensor]
    legal: bool = True
    illegal_reason: str | None = None

    def __post_init__(self) -> None:
        if not str(self.candidate_id).strip():
            raise ValueError("candidate_id cannot be empty")
        if bool(self.legal) and self.illegal_reason:
            raise ValueError("legal candidate cannot carry illegal_reason")
        if not bool(self.legal) and not str(self.illegal_reason or "").strip():
            raise ValueError("illegal candidate requires illegal_reason")
        missing = sorted(set(REQUIRED_FUTURE_ACTION_FIELDS) - set(self.future_action))
        if missing:
            raise ValueError(f"candidate future_action is missing fields: {missing}")
        for field in REQUIRED_FUTURE_ACTION_FIELDS:
            value = self.future_action[field]
            if not isinstance(value, Tensor) or value.ndim < 2:
                raise ValueError(f"candidate future_action.{field} must be a tensor")
            if not torch.isfinite(value.float()).all():
                raise ValueError(f"candidate future_action.{field} must be finite")


@dataclass(frozen=True)
class CandidatePrediction:
    """Task/state/cost/risk outcomes extracted from one model rollout."""

    future_state: Mapping[str, Tensor]
    task_outcome: float
    cost: float
    risk: float
    objective: float

    @classmethod
    def create(
        cls,
        *,
        future_state: Mapping[str, Tensor],
        task_outcome: Any,
        cost: Any,
        risk: Any,
        objective: Any,
    ) -> "CandidatePrediction":
        if not future_state:
            raise ValueError("future_state extraction cannot be empty")
        copied: dict[str, Tensor] = {}
        for name, value in future_state.items():
            if not isinstance(value, Tensor) or not torch.isfinite(value).all():
                raise ValueError(f"future_state[{name!r}] must be a finite tensor")
            copied[str(name)] = value.detach().clone()
        return cls(
            future_state=copied,
            task_outcome=_finite_float(task_outcome, field="task_outcome"),
            cost=_finite_float(cost, field="cost"),
            risk=_finite_float(risk, field="risk"),
            objective=_finite_float(objective, field="objective"),
        )


@dataclass(frozen=True)
class CandidateRolloutRecord:
    candidate: CandidateAction
    prediction: CandidatePrediction
    common_belief_fingerprint: str


@dataclass(frozen=True)
class PlannerDecision:
    """Auditable output of one planning/replanning call."""

    schema_version: str
    mechanism_status: str
    records: tuple[CandidateRolloutRecord, ...]
    selected_candidate_id: str
    selected_candidate_index: int
    selected_first_action: Mapping[str, Tensor]
    common_belief_fingerprint: str
    replan_count: int


PredictionExtractor = Callable[[Mapping[str, Tensor], CandidateAction], Mapping[str, Any]]
Objective = Callable[[CandidatePrediction], float]
CandidateGenerator = Callable[[Mapping[str, Any]], Sequence[CandidateAction]]


class FormalCandidateRolloutPlanner:
    """Run every legal candidate through one action-conditioned world model."""

    def __init__(
        self,
        *,
        world_model: Callable[[Mapping[str, Mapping[str, Tensor]]], Mapping[str, Tensor]],
        candidate_generator: CandidateGenerator,
        prediction_extractor: PredictionExtractor,
        objective: Objective,
    ) -> None:
        self.world_model = world_model
        self.candidate_generator = candidate_generator
        self.prediction_extractor = prediction_extractor
        self.objective = objective
        self._replan_count = 0

    @staticmethod
    def _validate_batch(batch: Mapping[str, Any]) -> None:
        for namespace in ("history", "future_action", "static"):
            if namespace not in batch or not isinstance(batch[namespace], Mapping):
                raise ValueError(f"batch must contain mapping namespace {namespace!r}")
        base = batch["future_action"]
        missing = sorted(set(REQUIRED_FUTURE_ACTION_FIELDS) - set(base))
        if missing:
            raise ValueError(f"batch future_action is missing fields: {missing}")
        for field in REQUIRED_FUTURE_ACTION_FIELDS:
            if not isinstance(base[field], Tensor) or not torch.isfinite(
                base[field].float()
            ).all():
                raise ValueError(f"batch future_action.{field} must be finite")

    @staticmethod
    def _inject_candidate(
        batch: Mapping[str, Any], candidate: CandidateAction
    ) -> dict[str, Any]:
        candidate_batch = _clone_tree(batch)
        base = batch["future_action"]
        injected: dict[str, Tensor] = {}
        for field in REQUIRED_FUTURE_ACTION_FIELDS:
            value = candidate.future_action[field]
            if value.shape != base[field].shape:
                raise ValueError(
                    f"candidate {candidate.candidate_id!r} field {field!r} shape "
                    f"{tuple(value.shape)} differs from {tuple(base[field].shape)}"
                )
            injected[field] = value.detach().clone()
        candidate_batch["future_action"] = injected
        return candidate_batch

    def plan(self, batch: Mapping[str, Any]) -> PlannerDecision:
        self._validate_batch(batch)
        common_belief = _tensor_fingerprint(
            {"history": batch["history"], "static": batch["static"]}
        )
        candidates = tuple(self.candidate_generator(_clone_tree(batch)))
        if not candidates:
            raise ValueError("candidate generator returned no candidates")
        legal = tuple(candidate for candidate in candidates if candidate.legal)
        if not legal:
            raise ValueError("candidate generator returned no legal candidates")
        ids = [candidate.candidate_id for candidate in legal]
        if len(set(ids)) != len(ids):
            raise ValueError("legal candidate ids must be unique")

        records: list[CandidateRolloutRecord] = []
        was_training = getattr(self.world_model, "training", None)
        try:
            if hasattr(self.world_model, "eval"):
                self.world_model.eval()
            with torch.no_grad():
                for candidate in legal:
                    candidate_batch = self._inject_candidate(batch, candidate)
                    outputs = self.world_model(candidate_batch)
                    raw = self.prediction_extractor(outputs, candidate)
                    if not isinstance(raw, Mapping):
                        raise ValueError("prediction_extractor must return a mapping")
                    prediction_without_objective = CandidatePrediction.create(
                        future_state=raw.get("future_state", {}),
                        task_outcome=raw.get("task_outcome"),
                        cost=raw.get("cost"),
                        risk=raw.get("risk"),
                        objective=0.0,
                    )
                    prediction = CandidatePrediction.create(
                        future_state=prediction_without_objective.future_state,
                        task_outcome=prediction_without_objective.task_outcome,
                        cost=prediction_without_objective.cost,
                        risk=prediction_without_objective.risk,
                        objective=self.objective(prediction_without_objective),
                    )
                    records.append(
                        CandidateRolloutRecord(candidate, prediction, common_belief)
                    )
        finally:
            if was_training is True and hasattr(self.world_model, "train"):
                self.world_model.train()

        selected_index, selected_record = min(
            enumerate(records), key=lambda item: item[1].prediction.objective
        )
        selected_first_action = {
            field: selected_record.candidate.future_action[field][:, :1].detach().clone()
            for field in REQUIRED_FUTURE_ACTION_FIELDS
        }
        decision = PlannerDecision(
            schema_version=MECHANISM_SCHEMA_VERSION,
            mechanism_status="prototype_only",
            records=tuple(records),
            selected_candidate_id=selected_record.candidate.candidate_id,
            selected_candidate_index=selected_index,
            selected_first_action=selected_first_action,
            common_belief_fingerprint=common_belief,
            replan_count=self._replan_count,
        )
        return decision

    def replan(self, updated_batch: Mapping[str, Any]) -> PlannerDecision:
        """Consume a new observation/history and repeat candidate evaluation."""

        self._replan_count += 1
        return self.plan(updated_batch)


__all__ = [
    "CandidateAction",
    "CandidatePrediction",
    "CandidateRolloutRecord",
    "FormalCandidateRolloutPlanner",
    "MECHANISM_SCHEMA_VERSION",
    "PlannerDecision",
    "REQUIRED_FUTURE_ACTION_FIELDS",
]
