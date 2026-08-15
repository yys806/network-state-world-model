"""Strict R4 checkpoints bound to components, budget, and upstream provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .r3_checkpoint import REQUIRED_BINDINGS
from .r4_module_registry import R4ModuleConfig, R4_REGISTRY_SCHEMA
from .r4_world_model import R4_MODEL_SCHEMA, R4WorldModel, build_r4_world_model


R4_CHECKPOINT_SCHEMA = "PIJWM-R4-Strict-Checkpoint-v1"


@dataclass(frozen=True)
class R4TrainingBudget:
    epochs: int
    patience: int
    learning_rate: float
    training_seed: int

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.patience <= 0:
            raise ValueError("epochs and patience must be positive")
        if self.patience > self.epochs:
            raise ValueError("patience cannot exceed epochs")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


@dataclass
class LoadedR4Checkpoint:
    model: R4WorldModel
    optimizer_state: dict[str, Any] | None
    bindings: dict[str, str]
    budget: R4TrainingBudget
    seed: int


def _validate_bindings(bindings: Mapping[str, str]) -> dict[str, str]:
    if set(bindings) != set(REQUIRED_BINDINGS):
        raise ValueError(
            "checkpoint bindings must be exactly: " + ", ".join(REQUIRED_BINDINGS)
        )
    normalized = {key: str(bindings[key]).lower() for key in REQUIRED_BINDINGS}
    for key, value in normalized.items():
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{key} must be a 64-character SHA-256 digest")
    return normalized


def save_r4_checkpoint(
    path: str | Path,
    model: R4WorldModel,
    optimizer: torch.optim.Optimizer | None,
    bindings: Mapping[str, str],
    budget: R4TrainingBudget,
    *,
    seed: int,
) -> None:
    if not isinstance(model, R4WorldModel):
        raise TypeError("R4 checkpoint accepts only R4WorldModel")
    if not isinstance(budget, R4TrainingBudget):
        raise TypeError("budget must be R4TrainingBudget")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": R4_CHECKPOINT_SCHEMA,
        "registry_schema_version": R4_REGISTRY_SCHEMA,
        "model_schema_version": R4_MODEL_SCHEMA,
        "components": model.component_registry(),
        "config": asdict(model.config),
        "budget": asdict(budget),
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "bindings": _validate_bindings(bindings),
        "seed": int(seed),
    }
    torch.save(envelope, target)


def load_r4_checkpoint(
    path: str | Path,
    *,
    expected_bindings: Mapping[str, str],
) -> LoadedR4Checkpoint:
    envelope = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(envelope, Mapping):
        raise ValueError("R4 checkpoint envelope is not a mapping")
    if envelope.get("schema_version") != R4_CHECKPOINT_SCHEMA:
        raise ValueError("R4 checkpoint schema_version is incompatible")
    if envelope.get("registry_schema_version") != R4_REGISTRY_SCHEMA:
        raise ValueError("R4 checkpoint registry schema_version is incompatible")
    if envelope.get("model_schema_version") != R4_MODEL_SCHEMA:
        raise ValueError("R4 checkpoint model schema_version is incompatible")

    stored_bindings = _validate_bindings(dict(envelope.get("bindings", {})))
    expected = _validate_bindings(expected_bindings)
    for key in REQUIRED_BINDINGS:
        if stored_bindings[key] != expected[key]:
            raise ValueError(f"R4 checkpoint binding mismatch: {key}")

    config_values = envelope.get("config")
    budget_values = envelope.get("budget")
    if not isinstance(config_values, Mapping):
        raise ValueError("R4 checkpoint config is missing")
    if not isinstance(budget_values, Mapping):
        raise ValueError("R4 checkpoint budget is missing")
    try:
        config = R4ModuleConfig(**dict(config_values))
        model = build_r4_world_model(config)
        if dict(envelope.get("components", {})) != model.component_registry():
            raise ValueError("R4 checkpoint components are unknown or incompatible")
        model.load_state_dict(envelope["model_state"], strict=True)
        budget = R4TrainingBudget(**dict(budget_values))
    except (KeyError, TypeError, RuntimeError, ValueError) as error:
        if "components" in str(error):
            raise
        raise ValueError(f"R4 checkpoint model state is incompatible: {error}") from error

    optimizer_state = envelope.get("optimizer_state")
    if optimizer_state is not None and not isinstance(optimizer_state, dict):
        raise ValueError("R4 checkpoint optimizer_state is invalid")
    return LoadedR4Checkpoint(
        model=model,
        optimizer_state=optimizer_state,
        bindings=stored_bindings,
        budget=budget,
        seed=int(envelope.get("seed", -1)),
    )


__all__ = [
    "LoadedR4Checkpoint",
    "R4_CHECKPOINT_SCHEMA",
    "R4TrainingBudget",
    "load_r4_checkpoint",
    "save_r4_checkpoint",
]
