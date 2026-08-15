"""Approved compositional world models for PI-JWM R5 formal comparison."""

from __future__ import annotations

from torch import nn

from .r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel
from .r4_world_model import (
    R4WorldModel,
    _ExplicitDAGBackend,
    _GraphRSSMBackend,
    _HeteroscedasticBackend,
    _SoftPredictedPresenceBackend,
)
from .r5_protocol import get_r5_combination


R5_MODEL_SCHEMA = "PIJWM-R5-Approved-Combination-World-Model-v1"


class R5WorldModel(R4WorldModel):
    """R4-compatible public model bound to one approved R5 combination."""

    schema_version = R5_MODEL_SCHEMA

    def __init__(self, combination_id: str, config, backend: nn.Module) -> None:
        super().__init__(config, backend)
        self.combination_id = str(combination_id)


def build_r5_world_model(
    combination_id: str,
    *,
    hidden_dim: int = 16,
    history_steps: int = 8,
    information_rate_mean: float | None = None,
    information_rate_scale: float | None = None,
) -> R5WorldModel:
    combination = get_r5_combination(
        combination_id,
        hidden_dim=hidden_dim,
        history_steps=history_steps,
        information_rate_mean=information_rate_mean,
        information_rate_scale=information_rate_scale,
    )
    config = combination.config
    backend_config = R3ReferenceConfig(
        hidden_dim=config.hidden_dim,
        history_steps=config.history_steps,
        use_cross_graph_coupling=True,
    )

    base: nn.Module = R3ReferenceWorldModel(backend_config)
    if config.dag == "explicit_dag_message_passing_v1":
        base = _ExplicitDAGBackend(backend_config)
    elif config.presence == "soft_predicted_presence_v1":
        base = _SoftPredictedPresenceBackend(backend_config)

    backend: nn.Module = base
    if config.dynamics == "graph_rssm_v1":
        backend = _GraphRSSMBackend(backend_config, base=backend)
    if config.head == "heteroscedastic_typed_v1":
        backend = _HeteroscedasticBackend(backend_config, base=backend)
    return R5WorldModel(combination.combination_id, config, backend)


__all__ = [
    "R5_MODEL_SCHEMA",
    "R5WorldModel",
    "build_r5_world_model",
]
