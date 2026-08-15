"""Strict, provenance-bound checkpoints for the PI-JWM R3 reference model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .r3_world_model import (
    REFERENCE_COMPONENTS,
    R3_MODEL_SCHEMA,
    R3ReferenceConfig,
    R3ReferenceWorldModel,
)


R3_CHECKPOINT_SCHEMA = "PIJWM-R3-Strict-Checkpoint-v1"
REQUIRED_BINDINGS = (
    "tensor_contract_sha256",
    "dataset_protocol_sha256",
    "normalization_sha256",
    "metric_registry_sha256",
    "source_code_sha256",
)


@dataclass
class LoadedR3Checkpoint:
    model: R3ReferenceWorldModel
    optimizer_state: dict[str, Any] | None
    bindings: dict[str, str]
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


def save_r3_checkpoint(
    path: str | Path,
    model: R3ReferenceWorldModel,
    optimizer: torch.optim.Optimizer | None,
    bindings: Mapping[str, str],
    *,
    seed: int,
) -> None:
    if not isinstance(model, R3ReferenceWorldModel):
        raise TypeError("R3 checkpoint accepts only R3ReferenceWorldModel")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": R3_CHECKPOINT_SCHEMA,
        "model_schema_version": R3_MODEL_SCHEMA,
        "components": model.component_registry(),
        "config": asdict(model.config),
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "bindings": _validate_bindings(bindings),
        "seed": int(seed),
    }
    torch.save(envelope, path)


def load_r3_checkpoint(
    path: str | Path,
    *,
    expected_bindings: Mapping[str, str],
) -> LoadedR3Checkpoint:
    path = Path(path)
    envelope = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(envelope, Mapping):
        raise ValueError("R3 checkpoint envelope is not a mapping")
    if envelope.get("schema_version") != R3_CHECKPOINT_SCHEMA:
        raise ValueError("R3 checkpoint schema_version is incompatible")
    if envelope.get("model_schema_version") != R3_MODEL_SCHEMA:
        raise ValueError("R3 checkpoint model_schema_version is incompatible")
    if dict(envelope.get("components", {})) != REFERENCE_COMPONENTS:
        raise ValueError("R3 checkpoint components are unknown or incompatible")

    stored_bindings = _validate_bindings(dict(envelope.get("bindings", {})))
    expected = _validate_bindings(expected_bindings)
    for key in REQUIRED_BINDINGS:
        if stored_bindings[key] != expected[key]:
            raise ValueError(f"R3 checkpoint binding mismatch: {key}")

    config_values = envelope.get("config")
    if not isinstance(config_values, Mapping):
        raise ValueError("R3 checkpoint config is missing")
    try:
        config = R3ReferenceConfig(**dict(config_values))
        model = R3ReferenceWorldModel(config)
        model.load_state_dict(envelope["model_state"], strict=True)
    except (KeyError, TypeError, RuntimeError, ValueError) as error:
        raise ValueError(f"R3 checkpoint model state is incompatible: {error}") from error
    optimizer_state = envelope.get("optimizer_state")
    if optimizer_state is not None and not isinstance(optimizer_state, dict):
        raise ValueError("R3 checkpoint optimizer_state is invalid")
    return LoadedR3Checkpoint(
        model=model,
        optimizer_state=optimizer_state,
        bindings=stored_bindings,
        seed=int(envelope.get("seed", -1)),
    )


__all__ = [
    "LoadedR3Checkpoint",
    "R3_CHECKPOINT_SCHEMA",
    "REQUIRED_BINDINGS",
    "load_r3_checkpoint",
    "save_r3_checkpoint",
]
