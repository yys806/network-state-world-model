"""Frozen PI-JWM R5 combinations and formal multi-seed protocol."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .r4_module_registry import R4ModuleConfig, reference_r4_config


R5_PROTOCOL_SCHEMA = "PIJWM-R5-Formal-Combination-Protocol-v1"
REQUIRED_PUBLIC_METRIC_GATES = (
    "event.information_link_activity.auprc",
    "link.active_only_rate.mae",
    "task.lifecycle.macro_f1",
    "selection.required_continuous.normalized_error",
)


@dataclass(frozen=True)
class R5FormalProtocol:
    training_seeds: tuple[int, ...]
    max_epochs: int
    patience: int
    effective_batch_size: int
    minimum_improvement: float
    checkpoint_split: str = "validation"
    calibration_split: str = "calibration"

    def __post_init__(self) -> None:
        if self.training_seeds != (20260803, 20260804, 20260805):
            raise ValueError("R5 requires the frozen three training seeds")
        if self.max_epochs != 100 or self.patience != 10:
            raise ValueError("R5 formal epoch or patience budget drifted")
        if self.effective_batch_size != 32:
            raise ValueError("R5 effective batch size must remain 32")
        if self.minimum_improvement != 1.0e-4:
            raise ValueError("R5 minimum improvement must remain 1e-4")
        if self.checkpoint_split != "validation":
            raise ValueError("R5 checkpoints must be selected on validation")
        if self.calibration_split != "calibration":
            raise ValueError("R5 calibration must use calibration only")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["training_seeds"] = list(self.training_seeds)
        value["schema_version"] = R5_PROTOCOL_SCHEMA
        return value


@dataclass(frozen=True)
class R5Combination:
    combination_id: str
    label: str
    question: str
    config: R4ModuleConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "combination_id": self.combination_id,
            "label": self.label,
            "question": self.question,
            "config": asdict(self.config),
            "components": self.config.component_names(),
        }


def load_r5_protocol(evaluation_root: str | Path) -> R5FormalProtocol:
    path = Path(evaluation_root) / "fair_experiment_protocol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    budget = payload.get("budgets", {}).get("formal_comparison", {})
    common = payload.get("common_training", {})
    protocol = R5FormalProtocol(
        training_seeds=tuple(int(seed) for seed in budget.get("training_seeds", [])),
        max_epochs=int(budget.get("max_epochs", 0)),
        patience=int(budget.get("early_stopping_patience", 0)),
        effective_batch_size=int(common.get("batch_size", 0)),
        minimum_improvement=float(common.get("minimum_improvement", 0.0)),
    )
    selection = payload.get("checkpoint_selection", {})
    metric_ids = tuple(
        str(term.get("metric_id")) for term in selection.get("terms", [])
    )
    if metric_ids != REQUIRED_PUBLIC_METRIC_GATES:
        raise ValueError("R5 public metric gates drifted from the frozen R2 protocol")
    return protocol


def r5_combination_matrix(
    *,
    hidden_dim: int = 16,
    history_steps: int = 8,
    information_rate_mean: float | None = None,
    information_rate_scale: float | None = None,
) -> dict[str, R5Combination]:
    reference = reference_r4_config(
        hidden_dim=hidden_dim,
        history_steps=history_steps,
        information_rate_mean=information_rate_mean,
        information_rate_scale=information_rate_scale,
    )
    rssm = replace(reference, dynamics="graph_rssm_v1")
    return {
        "A": R5Combination(
            "A",
            "reference_graph_gru",
            "Does the deterministic R3/R4 reference remain a stable formal control?",
            reference,
        ),
        "B": R5Combination(
            "B",
            "graph_rssm",
            "Does the R4 winning dynamics remain stable across training seeds?",
            rssm,
        ),
        "C": R5Combination(
            "C",
            "graph_rssm_heteroscedastic",
            "Does an explicit predictive distribution improve calibrated forecasts?",
            replace(rssm, head="heteroscedastic_typed_v1"),
        ),
        "D": R5Combination(
            "D",
            "graph_rssm_explicit_dag",
            "Does directed DAG propagation improve task evolution under RSSM dynamics?",
            replace(rssm, dag="explicit_dag_message_passing_v1"),
        ),
        "E": R5Combination(
            "E",
            "graph_rssm_soft_presence",
            "Does predicted topology recursion improve long rollout under RSSM dynamics?",
            replace(rssm, presence="soft_predicted_presence_v1"),
        ),
    }


def get_r5_combination(
    combination_id: str,
    **config_options: Any,
) -> R5Combination:
    matrix = r5_combination_matrix(**config_options)
    try:
        return matrix[str(combination_id)]
    except KeyError as error:
        raise ValueError(f"unknown R5 combination: {combination_id}") from error


def validate_r5_splits(splits: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(split) for split in splits)
    if "locked_test" in normalized:
        raise ValueError("locked_test cannot be used before R9")
    unknown = set(normalized) - {"train", "validation", "calibration"}
    if unknown:
        raise ValueError(f"unknown R5 split: {sorted(unknown)}")
    if not normalized:
        raise ValueError("R5 splits must be non-empty")
    return normalized


__all__ = [
    "R5Combination",
    "R5FormalProtocol",
    "R5_PROTOCOL_SCHEMA",
    "REQUIRED_PUBLIC_METRIC_GATES",
    "get_r5_combination",
    "load_r5_protocol",
    "r5_combination_matrix",
    "validate_r5_splits",
]
