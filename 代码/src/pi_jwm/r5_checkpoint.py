"""Strict checkpoints for the PI-JWM R5 approved model combinations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .r3_checkpoint import REQUIRED_BINDINGS
from .r4_module_registry import R4_REGISTRY_SCHEMA
from .r5_protocol import R5FormalProtocol, R5_PROTOCOL_SCHEMA
from .r5_world_model import R5_MODEL_SCHEMA, R5WorldModel, build_r5_world_model


R5_CHECKPOINT_SCHEMA = "PIJWM-R5-Strict-Combination-Checkpoint-v1"
R5_REQUIRED_BINDINGS = REQUIRED_BINDINGS + (
    "r4_screening_manifest_sha256",
    "r5_protocol_sha256",
)


@dataclass
class LoadedR5Checkpoint:
    model: R5WorldModel
    optimizer_state: dict[str, Any] | None
    bindings: dict[str, str]
    protocol: R5FormalProtocol
    learning_rate: float
    seed: int


def _validate_bindings(bindings: Mapping[str, str]) -> dict[str, str]:
    if set(bindings) != set(R5_REQUIRED_BINDINGS):
        raise ValueError(
            "R5 checkpoint bindings must be exactly: "
            + ", ".join(R5_REQUIRED_BINDINGS)
        )
    normalized = {key: str(bindings[key]).lower() for key in R5_REQUIRED_BINDINGS}
    for key, value in normalized.items():
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{key} must be a 64-character SHA-256 digest")
    return normalized


def _protocol_from_dict(values: Mapping[str, Any]) -> R5FormalProtocol:
    payload = dict(values)
    if payload.pop("schema_version", None) != R5_PROTOCOL_SCHEMA:
        raise ValueError("R5 checkpoint protocol schema_version is incompatible")
    payload["training_seeds"] = tuple(int(seed) for seed in payload["training_seeds"])
    return R5FormalProtocol(**payload)


def save_r5_checkpoint(
    path: str | Path,
    model: R5WorldModel,
    optimizer: torch.optim.Optimizer | None,
    bindings: Mapping[str, str],
    protocol: R5FormalProtocol,
    *,
    learning_rate: float,
    seed: int,
) -> None:
    if not isinstance(model, R5WorldModel):
        raise TypeError("R5 checkpoint accepts only R5WorldModel")
    if not isinstance(protocol, R5FormalProtocol):
        raise TypeError("protocol must be R5FormalProtocol")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if int(seed) not in protocol.training_seeds:
        raise ValueError("seed is outside the frozen R5 protocol")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": R5_CHECKPOINT_SCHEMA,
            "registry_schema_version": R4_REGISTRY_SCHEMA,
            "model_schema_version": R5_MODEL_SCHEMA,
            "combination_id": model.combination_id,
            "components": model.component_registry(),
            "config": asdict(model.config),
            "protocol": protocol.to_dict(),
            "learning_rate": float(learning_rate),
            "model_state": model.state_dict(),
            "optimizer_state": None if optimizer is None else optimizer.state_dict(),
            "bindings": _validate_bindings(bindings),
            "seed": int(seed),
        },
        target,
    )


def load_r5_checkpoint(
    path: str | Path,
    *,
    expected_bindings: Mapping[str, str],
    expected_protocol: R5FormalProtocol,
) -> LoadedR5Checkpoint:
    envelope = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(envelope, Mapping):
        raise ValueError("R5 checkpoint envelope is not a mapping")
    if envelope.get("schema_version") != R5_CHECKPOINT_SCHEMA:
        raise ValueError("R5 checkpoint schema_version is incompatible")
    if envelope.get("registry_schema_version") != R4_REGISTRY_SCHEMA:
        raise ValueError("R5 checkpoint registry schema_version is incompatible")
    if envelope.get("model_schema_version") != R5_MODEL_SCHEMA:
        raise ValueError("R5 checkpoint model schema_version is incompatible")

    stored_bindings = _validate_bindings(dict(envelope.get("bindings", {})))
    expected = _validate_bindings(expected_bindings)
    for key in R5_REQUIRED_BINDINGS:
        if stored_bindings[key] != expected[key]:
            raise ValueError(f"R5 checkpoint binding mismatch: {key}")

    protocol_values = envelope.get("protocol")
    if not isinstance(protocol_values, Mapping):
        raise ValueError("R5 checkpoint protocol is missing")
    protocol = _protocol_from_dict(protocol_values)
    if protocol != expected_protocol:
        raise ValueError("R5 checkpoint formal protocol mismatch")

    config_values = envelope.get("config")
    if not isinstance(config_values, Mapping):
        raise ValueError("R5 checkpoint config is missing")
    combination_id = str(envelope.get("combination_id", ""))
    try:
        model = build_r5_world_model(
            combination_id,
            hidden_dim=int(config_values["hidden_dim"]),
            history_steps=int(config_values["history_steps"]),
            information_rate_mean=config_values.get("information_rate_mean"),
            information_rate_scale=config_values.get("information_rate_scale"),
        )
        if asdict(model.config) != dict(config_values):
            raise ValueError("R5 checkpoint combination configuration is incompatible")
        if dict(envelope.get("components", {})) != model.component_registry():
            raise ValueError("R5 checkpoint combination components are incompatible")
        model.load_state_dict(envelope["model_state"], strict=True)
    except (KeyError, TypeError, RuntimeError, ValueError) as error:
        if "combination" in str(error):
            raise
        raise ValueError(f"R5 checkpoint model state is incompatible: {error}") from error

    seed = int(envelope.get("seed", -1))
    if seed not in protocol.training_seeds:
        raise ValueError("R5 checkpoint seed is outside the frozen protocol")
    learning_rate = float(envelope.get("learning_rate", 0.0))
    if learning_rate <= 0.0:
        raise ValueError("R5 checkpoint learning_rate is invalid")
    optimizer_state = envelope.get("optimizer_state")
    if optimizer_state is not None and not isinstance(optimizer_state, dict):
        raise ValueError("R5 checkpoint optimizer_state is invalid")
    return LoadedR5Checkpoint(
        model=model,
        optimizer_state=optimizer_state,
        bindings=stored_bindings,
        protocol=protocol,
        learning_rate=learning_rate,
        seed=seed,
    )


__all__ = [
    "LoadedR5Checkpoint",
    "R5_CHECKPOINT_SCHEMA",
    "R5_REQUIRED_BINDINGS",
    "load_r5_checkpoint",
    "save_r5_checkpoint",
]
