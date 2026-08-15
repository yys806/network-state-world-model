"""Strict checkpoints for the PI-JWM R5 module-confirmation matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .r4_module_registry import R4_REGISTRY_SCHEMA
from .r5_checkpoint import R5_REQUIRED_BINDINGS
from .r5_module_confirmation import build_confirmation_model
from .r5_protocol import R5FormalProtocol, R5_PROTOCOL_SCHEMA
from .r5_world_model import R5_MODEL_SCHEMA, R5WorldModel


CONFIRMATION_CHECKPOINT_SCHEMA = "PIJWM-R5-Module-Confirmation-Checkpoint-v1"
CONFIRMATION_REQUIRED_BINDINGS = R5_REQUIRED_BINDINGS + (
    "existing_r5_manifest_sha256",
    "confirmation_matrix_sha256",
)


@dataclass
class LoadedConfirmationCheckpoint:
    model: R5WorldModel
    optimizer_state: dict[str, Any] | None
    bindings: dict[str, str]
    protocol: R5FormalProtocol
    learning_rate: float
    seed: int


def _validate_bindings(bindings: Mapping[str, str]) -> dict[str, str]:
    if set(bindings) != set(CONFIRMATION_REQUIRED_BINDINGS):
        raise ValueError(
            "confirmation checkpoint bindings must be exactly: "
            + ", ".join(CONFIRMATION_REQUIRED_BINDINGS)
        )
    normalized = {
        key: str(bindings[key]).lower() for key in CONFIRMATION_REQUIRED_BINDINGS
    }
    for key, value in normalized.items():
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{key} must be a 64-character SHA-256 digest")
    return normalized


def _protocol_from_dict(values: Mapping[str, Any]) -> R5FormalProtocol:
    payload = dict(values)
    if payload.pop("schema_version", None) != R5_PROTOCOL_SCHEMA:
        raise ValueError("confirmation checkpoint protocol schema_version is incompatible")
    payload["training_seeds"] = tuple(int(seed) for seed in payload["training_seeds"])
    return R5FormalProtocol(**payload)


def save_confirmation_checkpoint(
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
        raise TypeError("confirmation checkpoint accepts only R5WorldModel")
    if model.combination_id not in {"F", "G", "H", "J"}:
        raise ValueError("confirmation checkpoint accepts only newly trained F/G/H/J")
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
            "schema_version": CONFIRMATION_CHECKPOINT_SCHEMA,
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


def load_confirmation_checkpoint(
    path: str | Path,
    *,
    expected_bindings: Mapping[str, str],
    expected_protocol: R5FormalProtocol,
) -> LoadedConfirmationCheckpoint:
    envelope = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(envelope, Mapping):
        raise ValueError("confirmation checkpoint envelope is not a mapping")
    if envelope.get("schema_version") != CONFIRMATION_CHECKPOINT_SCHEMA:
        raise ValueError("confirmation checkpoint schema_version is incompatible")
    if envelope.get("registry_schema_version") != R4_REGISTRY_SCHEMA:
        raise ValueError("confirmation checkpoint registry schema_version is incompatible")
    if envelope.get("model_schema_version") != R5_MODEL_SCHEMA:
        raise ValueError("confirmation checkpoint model schema_version is incompatible")

    stored_bindings = _validate_bindings(dict(envelope.get("bindings", {})))
    expected = _validate_bindings(expected_bindings)
    for key in CONFIRMATION_REQUIRED_BINDINGS:
        if stored_bindings[key] != expected[key]:
            raise ValueError(f"confirmation checkpoint binding mismatch: {key}")

    protocol_values = envelope.get("protocol")
    if not isinstance(protocol_values, Mapping):
        raise ValueError("confirmation checkpoint protocol is missing")
    protocol = _protocol_from_dict(protocol_values)
    if protocol != expected_protocol:
        raise ValueError("confirmation checkpoint formal protocol mismatch")

    config_values = envelope.get("config")
    if not isinstance(config_values, Mapping):
        raise ValueError("confirmation checkpoint config is missing")
    combination_id = str(envelope.get("combination_id", ""))
    try:
        model = build_confirmation_model(
            combination_id,
            hidden_dim=int(config_values["hidden_dim"]),
            history_steps=int(config_values["history_steps"]),
            information_rate_mean=config_values.get("information_rate_mean"),
            information_rate_scale=config_values.get("information_rate_scale"),
        )
        if asdict(model.config) != dict(config_values):
            raise ValueError("confirmation checkpoint configuration is incompatible")
        if dict(envelope.get("components", {})) != model.component_registry():
            raise ValueError("confirmation checkpoint components are incompatible")
        model.load_state_dict(envelope["model_state"], strict=True)
    except (KeyError, TypeError, RuntimeError, ValueError) as error:
        if "confirmation combination" in str(error):
            raise
        raise ValueError(f"confirmation checkpoint model state is incompatible: {error}") from error

    seed = int(envelope.get("seed", -1))
    if seed not in protocol.training_seeds:
        raise ValueError("confirmation checkpoint seed is outside the frozen protocol")
    learning_rate = float(envelope.get("learning_rate", 0.0))
    if learning_rate <= 0.0:
        raise ValueError("confirmation checkpoint learning_rate is invalid")
    optimizer_state = envelope.get("optimizer_state")
    if optimizer_state is not None and not isinstance(optimizer_state, dict):
        raise ValueError("confirmation checkpoint optimizer_state is invalid")
    return LoadedConfirmationCheckpoint(
        model=model,
        optimizer_state=optimizer_state,
        bindings=stored_bindings,
        protocol=protocol,
        learning_rate=learning_rate,
        seed=seed,
    )


__all__ = [
    "CONFIRMATION_CHECKPOINT_SCHEMA",
    "CONFIRMATION_REQUIRED_BINDINGS",
    "LoadedConfirmationCheckpoint",
    "load_confirmation_checkpoint",
    "save_confirmation_checkpoint",
]
