"""Legacy residual architecture adapted to the frozen PI-JWM v3 graph contract."""

from __future__ import annotations

from typing import Mapping

import torch

from .r3_preflight_data import ExplicitStateBatch
from .r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel


LEGACY_ADAPTED_DYNAMICS = "legacy_directed_dynamic_residual_v2_adapted_v1"
LEGACY_SOURCE_MODEL_ID = "coupled_directed_dynamic_residual_v2"


class LegacyDirectedResidualBackend(R3ReferenceWorldModel):
    """Current dual-graph model with the old residual/dynamic-topology mechanism.

    The node, edge and coupling meanings come exclusively from ``ExplicitStateBatch``.
    Only the old architecture's fixed-last-observation residual prediction and
    soft relation activity update are retained for a same-protocol control.
    """

    adapted_from = LEGACY_SOURCE_MODEL_ID
    adaptation_boundary = "current_r1_graph_semantics_and_r2_prediction_interface"

    def __init__(self, config: R3ReferenceConfig) -> None:
        super().__init__(config)
        self._residual_bases: dict[str, torch.Tensor] | None = None
        self._predicted_presence: dict[str, torch.Tensor] | None = None

    def infer_belief(self, batch: ExplicitStateBatch):
        self._predicted_presence = None
        self._residual_bases = {
            name: batch.history[name][:, -1]
            for name in (
                "physical_node_state",
                "physical_edge_state",
                "information_node_state",
                "information_edge_state",
                "data_flow_state",
                "task_state",
                "task_dag_state",
            )
        }
        return super().infer_belief(batch)

    def _batch_with_predicted_presence(
        self,
        batch: ExplicitStateBatch,
    ) -> ExplicitStateBatch:
        if self._predicted_presence is None:
            return batch
        history = dict(batch.history)
        for key, probability in self._predicted_presence.items():
            observed = history[key]
            history[key] = torch.cat(
                (observed[:, :-1].to(probability.dtype), probability.unsqueeze(1)),
                dim=1,
            )
        return ExplicitStateBatch(
            history=history,
            history_action=batch.history_action,
            future_action=batch.future_action,
            target=batch.target,
            static=batch.static,
            metadata=batch.metadata,
        )

    def _graph_update(self, state, batch: ExplicitStateBatch):
        return super()._graph_update(
            state,
            self._batch_with_predicted_presence(batch),
        )

    def _refresh_global(self, state, batch: ExplicitStateBatch):
        return super()._refresh_global(
            state,
            self._batch_with_predicted_presence(batch),
        )

    def _predict(self, state):
        if self._residual_bases is None:
            raise RuntimeError("legacy residual rollout requires infer_belief first")
        explicit, logits = super()._predict(state)
        explicit = {
            name: self._residual_bases[name] + prediction
            for name, prediction in explicit.items()
        }
        # The legacy v2 implementation anchors every rollout step to the last
        # observed explicit state; latent dynamics remain recursive separately.
        self._predicted_presence = {
            name: torch.sigmoid(logits[name])
            for name in (
                "physical_node_present",
                "physical_edge_present",
                "information_node_present",
                "information_edge_present",
                "data_flow_present",
                "task_present",
                "task_dag_state_present",
            )
        }
        return explicit, logits


__all__ = [
    "LEGACY_ADAPTED_DYNAMICS",
    "LEGACY_SOURCE_MODEL_ID",
    "LegacyDirectedResidualBackend",
]
