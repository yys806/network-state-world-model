"""Reference explicit-plus-latent world-model pipeline for PI-JWM R3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .formal_directed_graph_ops_v2 import (
    directed_relation_messages,
    weighted_index_mean,
)
from .r3_preflight_data import ExplicitStateBatch


R3_MODEL_SCHEMA = "PIJWM-R3-Reference-World-Model-v1"
REFERENCE_COMPONENTS = {
    "field_encoder": "masked_mlp_v1",
    "graph_encoder": "directed_relational_mean_v1",
    "coupling": "gated_cip_cep_cfl_v1",
    "dynamics": "deterministic_graph_gru_v1",
    "head": "deterministic_typed_v1",
}


@dataclass(frozen=True)
class R3ReferenceConfig:
    hidden_dim: int = 16
    history_steps: int = 8
    use_cross_graph_coupling: bool = True
    field_encoder: str = REFERENCE_COMPONENTS["field_encoder"]
    graph_encoder: str = REFERENCE_COMPONENTS["graph_encoder"]
    coupling: str = REFERENCE_COMPONENTS["coupling"]
    dynamics: str = REFERENCE_COMPONENTS["dynamics"]
    head: str = REFERENCE_COMPONENTS["head"]


@dataclass
class BeliefSequence:
    physical_latent: torch.Tensor
    information_latent: torch.Tensor
    business_latent: torch.Tensor
    joint_latent: torch.Tensor


@dataclass
class R3RolloutOutput:
    predicted_explicit: dict[str, torch.Tensor]
    predicted_logits: dict[str, torch.Tensor]
    predicted_belief: BeliefSequence


@dataclass
class _LatentState:
    physical_node: torch.Tensor
    physical_edge: torch.Tensor
    information_node: torch.Tensor
    information_edge: torch.Tensor
    flow: torch.Tensor
    task: torch.Tensor
    business: torch.Tensor
    joint: torch.Tensor


class _MaskedTemporalEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(feature_dim * 2, hidden_dim)
        self.history = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(
        self,
        value: torch.Tensor,
        feature_mask: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        if value.ndim != 4 or feature_mask.shape != value.shape:
            raise ValueError("temporal value and feature mask must be [batch,time,entity,feature]")
        if present.shape != value.shape[:-1]:
            raise ValueError("temporal presence does not match values")
        batch, steps, entities, _ = value.shape
        encoded = self.input(
            torch.cat((value, feature_mask.to(value.dtype)), dim=-1)
        )
        encoded = encoded * present.unsqueeze(-1).to(value.dtype)
        sequence = encoded.permute(0, 2, 1, 3).reshape(
            batch * entities, steps, encoded.shape[-1]
        )
        output, _ = self.history(sequence)
        return output[:, -1].reshape(batch, entities, encoded.shape[-1])


def _entity_mask(present: torch.Tensor, feature_dim: int) -> torch.Tensor:
    return present.unsqueeze(-1).expand(*present.shape, feature_dim)


def _gru(cell: nn.GRUCell, update: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    shape = current.shape
    return cell(
        update.reshape(-1, shape[-1]), current.reshape(-1, shape[-1])
    ).reshape(shape)


def _gather_entities(latent: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if index.ndim == 1:
        index = index.unsqueeze(0).expand(latent.shape[0], -1)
    if index.ndim != 2 or index.shape[0] != latent.shape[0]:
        raise ValueError("cross-graph index must be batch-aligned rank-2")
    count = latent.shape[1]
    safe = index.clamp(0, max(count - 1, 0)).long()
    selected = torch.gather(
        latent, 1, safe.unsqueeze(-1).expand(-1, -1, latent.shape[-1])
    )
    valid = (index >= 0) & (index < count)
    return selected * valid.unsqueeze(-1).to(latent.dtype)


def _masked_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    numeric = weight.to(value.dtype)
    numerator = (value * numeric.unsqueeze(-1)).sum(dim=1)
    denominator = numeric.sum(dim=1, keepdim=True).clamp_min(1.0)
    return numerator / denominator


class R3ReferenceWorldModel(nn.Module):
    """Minimal v3-native model used only to prove the R3 execution contract."""

    schema_version = R3_MODEL_SCHEMA
    latent_dynamics = "deterministic_reference_only"

    def __init__(self, config: R3ReferenceConfig) -> None:
        super().__init__()
        if config.hidden_dim <= 0 or config.history_steps <= 0:
            raise ValueError("hidden_dim and history_steps must be positive")
        for field in REFERENCE_COMPONENTS:
            actual = getattr(config, field)
            expected = REFERENCE_COMPONENTS[field]
            if actual != expected:
                raise ValueError(
                    f"unsupported {field}: {actual}; only executable component is {expected}"
                )
        self.config = config
        hidden = config.hidden_dim
        self.physical_node_encoder = _MaskedTemporalEncoder(9, hidden)
        self.physical_edge_encoder = _MaskedTemporalEncoder(7, hidden)
        self.information_node_encoder = _MaskedTemporalEncoder(7, hidden)
        self.information_edge_encoder = _MaskedTemporalEncoder(18, hidden)
        self.flow_encoder = _MaskedTemporalEncoder(5, hidden)
        self.task_encoder = _MaskedTemporalEncoder(8, hidden)
        self.dag_encoder = _MaskedTemporalEncoder(3, hidden)
        self.history_action_encoder = _MaskedTemporalEncoder(8, hidden)

        self.physical_node_graph = nn.Linear(hidden * 2, hidden)
        self.physical_edge_graph = nn.Linear(hidden * 2, hidden)
        self.information_node_graph = nn.Linear(hidden * 2, hidden)
        self.information_edge_graph = nn.Linear(hidden * 2, hidden)
        self.physical_node_transition = nn.GRUCell(hidden, hidden)
        self.physical_edge_transition = nn.GRUCell(hidden, hidden)
        self.information_node_transition = nn.GRUCell(hidden, hidden)
        self.information_edge_transition = nn.GRUCell(hidden, hidden)
        self.flow_transition = nn.GRUCell(hidden, hidden)
        self.task_transition = nn.GRUCell(hidden, hidden)
        self.business_transition = nn.GRUCell(hidden, hidden)
        self.joint_transition = nn.GRUCell(hidden, hidden)

        self.cip_to_physical = nn.Linear(hidden, hidden)
        self.cip_to_information = nn.Linear(hidden, hidden)
        self.cep_to_physical = nn.Linear(hidden, hidden)
        self.cep_to_information = nn.Linear(hidden, hidden)
        self.cfl_to_flow = nn.Linear(hidden, hidden)
        self.cfl_to_information = nn.Linear(hidden, hidden)
        self.coupler = nn.ModuleDict(
            {
                "physical_node": nn.GRUCell(hidden, hidden),
                "information_node": nn.GRUCell(hidden, hidden),
                "physical_edge": nn.GRUCell(hidden, hidden),
                "information_edge": nn.GRUCell(hidden, hidden),
                "flow": nn.GRUCell(hidden, hidden),
            }
        )

        self.action_encoder = nn.Linear(8, hidden)
        self.action_to_information = nn.Linear(hidden, hidden)
        self.action_to_joint = nn.Linear(hidden, hidden)
        self.business_fusion = nn.Linear(hidden * 2, hidden)
        self.joint_fusion = nn.Linear(hidden * 3, hidden)

        self.state_heads = nn.ModuleDict(
            {
                "physical_node_state": nn.Linear(hidden, 9),
                "physical_edge_state": nn.Linear(hidden, 7),
                "information_node_state": nn.Linear(hidden, 7),
                "information_edge_state": nn.Linear(hidden, 18),
                "data_flow_state": nn.Linear(hidden, 5),
                "task_state": nn.Linear(hidden, 8),
                "task_dag_state": nn.Linear(hidden, 3),
            }
        )
        self.presence_heads = nn.ModuleDict(
            {
                "physical_node_present": nn.Linear(hidden, 1),
                "physical_edge_present": nn.Linear(hidden, 1),
                "information_node_present": nn.Linear(hidden, 1),
                "information_edge_present": nn.Linear(hidden, 1),
                "data_flow_present": nn.Linear(hidden, 1),
                "task_present": nn.Linear(hidden, 1),
                "task_dag_state_present": nn.Linear(hidden, 1),
            }
        )
        self.link_activity_head = nn.Linear(hidden, 1)
        self.task_lifecycle_head = nn.Linear(hidden, 5)

    def component_registry(self) -> dict[str, str]:
        return dict(REFERENCE_COMPONENTS)

    def _temporal_latents(self, batch: ExplicitStateBatch) -> _LatentState:
        h = batch.history
        if h["physical_node_state"].shape[1] != self.config.history_steps:
            raise ValueError("history length does not match R3 configuration")

        def encode(prefix: str, encoder: _MaskedTemporalEncoder) -> torch.Tensor:
            value = h[f"{prefix}_state"]
            present = h[f"{prefix}_present"].bool()
            mask_key = f"{prefix}_feature_mask"
            feature_mask = (
                h[mask_key].bool()
                if mask_key in h
                else _entity_mask(present, value.shape[-1])
            )
            return encoder(value, feature_mask, present)

        physical_node = encode("physical_node", self.physical_node_encoder)
        physical_edge = encode("physical_edge", self.physical_edge_encoder)
        information_node = encode("information_node", self.information_node_encoder)
        information_edge = encode("information_edge", self.information_edge_encoder)
        flow = encode("data_flow", self.flow_encoder)
        task = encode("task", self.task_encoder)
        dag = self.dag_encoder(
            h["task_dag_state"],
            _entity_mask(h["task_dag_state_present"].bool(), 3),
            h["task_dag_state_present"].bool(),
        )
        history_action = batch.history_action
        action_present = history_action["task_action_present"].bool()
        action = self.history_action_encoder(
            history_action["task_action"],
            _entity_mask(action_present, 8),
            action_present,
        )
        task = task + dag + action
        batch_size = physical_node.shape[0]
        zeros = physical_node.new_zeros((batch_size, self.config.hidden_dim))
        return _LatentState(
            physical_node=physical_node,
            physical_edge=physical_edge,
            information_node=information_node,
            information_edge=information_edge,
            flow=flow,
            task=task,
            business=zeros,
            joint=zeros,
        )

    def _graph_update(
        self, state: _LatentState, batch: ExplicitStateBatch
    ) -> _LatentState:
        h, static = batch.history, batch.static
        p_node_weight = h["physical_node_present"][:, -1].to(state.physical_node.dtype)
        p_edge_weight = h["physical_edge_present"][:, -1].to(state.physical_edge.dtype)
        p_in, p_out, p_context = directed_relation_messages(
            state.physical_node,
            state.physical_edge,
            static["physical_edge_endpoint_index"],
            p_node_weight,
            p_edge_weight,
        )
        state.physical_node = _gru(
            self.physical_node_transition,
            self.physical_node_graph(torch.cat((p_in, p_out), dim=-1)),
            state.physical_node,
        )
        state.physical_edge = _gru(
            self.physical_edge_transition,
            self.physical_edge_graph(p_context),
            state.physical_edge,
        )

        i_node_weight = h["information_node_present"][:, -1].to(
            state.information_node.dtype
        )
        i_edge_weight = h["information_edge_present"][:, -1].to(
            state.information_edge.dtype
        )
        i_in, i_out, i_context = directed_relation_messages(
            state.information_node,
            state.information_edge,
            static["information_edge_endpoint_index"],
            i_node_weight,
            i_edge_weight,
        )
        state.information_node = _gru(
            self.information_node_transition,
            self.information_node_graph(torch.cat((i_in, i_out), dim=-1)),
            state.information_node,
        )
        state.information_edge = _gru(
            self.information_edge_transition,
            self.information_edge_graph(i_context),
            state.information_edge,
        )
        return state

    def _cross_couple(
        self, state: _LatentState, batch: ExplicitStateBatch
    ) -> _LatentState:
        if not self.config.use_cross_graph_coupling:
            return state
        static, h = batch.static, batch.history
        batch_size = state.physical_node.shape[0]

        cip = static["cip_agent_node_index"].long()
        physical_for_agent = _gather_entities(state.physical_node, cip)
        agent_valid = (cip >= 0) & (cip < state.physical_node.shape[1])
        information_to_physical = weighted_index_mean(
            state.information_node,
            cip,
            state.physical_node.shape[1],
            agent_valid.to(state.information_node.dtype),
        )
        state.information_node = _gru(
            self.coupler["information_node"],
            self.cip_to_information(physical_for_agent),
            state.information_node,
        )
        state.physical_node = _gru(
            self.coupler["physical_node"],
            self.cip_to_physical(information_to_physical),
            state.physical_node,
        )

        cep = static["cep_information_to_physical_edge_index"].long()
        physical_for_information_edge = _gather_entities(state.physical_edge, cep)
        cep_valid = (cep >= 0) & (cep < state.physical_edge.shape[1])
        information_to_physical_edge = weighted_index_mean(
            state.information_edge,
            cep,
            state.physical_edge.shape[1],
            cep_valid.to(state.information_edge.dtype),
        )
        state.information_edge = _gru(
            self.coupler["information_edge"],
            self.cep_to_information(physical_for_information_edge),
            state.information_edge,
        )
        state.physical_edge = _gru(
            self.coupler["physical_edge"],
            self.cep_to_physical(information_to_physical_edge),
            state.physical_edge,
        )

        cfl = static["cfl_information_edge_index"].long()
        information_for_flow = _gather_entities(state.information_edge, cfl)
        flow_weight = static.get(
            "data_flow_valid",
            torch.ones(
                (batch_size, state.flow.shape[1]),
                dtype=torch.bool,
                device=state.flow.device,
            ),
        ).to(state.flow.dtype)
        cfl_valid = (cfl >= 0) & (cfl < state.information_edge.shape[1])
        flow_to_information = weighted_index_mean(
            state.flow,
            cfl,
            state.information_edge.shape[1],
            flow_weight * cfl_valid.to(flow_weight.dtype),
        )
        state.flow = _gru(
            self.coupler["flow"],
            self.cfl_to_flow(information_for_flow),
            state.flow,
        )
        state.information_edge = _gru(
            self.coupler["information_edge"],
            self.cfl_to_information(flow_to_information),
            state.information_edge,
        )
        return state

    def _refresh_global(
        self, state: _LatentState, batch: ExplicitStateBatch
    ) -> _LatentState:
        h, static = batch.history, batch.static
        p_pool = _masked_mean(
            state.physical_node, h["physical_node_present"][:, -1]
        )
        i_pool = _masked_mean(
            state.information_node, h["information_node_present"][:, -1]
        )
        flow_weight = static.get(
            "data_flow_valid", h["data_flow_present"][:, -1]
        ).bool() & h["data_flow_present"][:, -1].bool()
        task_weight = static.get("task_valid", h["task_present"][:, -1]).bool()
        task_weight = task_weight & h["task_present"][:, -1].bool()
        flow_pool = _masked_mean(state.flow, flow_weight)
        task_pool = _masked_mean(state.task, task_weight)
        state.business = torch.tanh(
            self.business_fusion(torch.cat((flow_pool, task_pool), dim=-1))
        )
        state.joint = torch.tanh(
            self.joint_fusion(torch.cat((p_pool, i_pool, state.business), dim=-1))
        )
        return state

    def infer_belief(self, batch: ExplicitStateBatch) -> _LatentState:
        state = self._temporal_latents(batch)
        state = self._graph_update(state, batch)
        state = self._cross_couple(state, batch)
        return self._refresh_global(state, batch)

    def _scatter_action_to_information(
        self,
        encoded_action: torch.Tensor,
        role_index: torch.Tensor,
        action_weight: torch.Tensor,
        information_count: int,
    ) -> torch.Tensor:
        message = encoded_action.new_zeros(
            (encoded_action.shape[0], information_count, encoded_action.shape[-1])
        )
        for role in range(role_index.shape[-1]):
            message = message + weighted_index_mean(
                encoded_action,
                role_index[..., role],
                information_count,
                action_weight,
            )
        return message

    def _predict(self, state: _LatentState) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        entity_by_state = {
            "physical_node_state": state.physical_node,
            "physical_edge_state": state.physical_edge,
            "information_node_state": state.information_node,
            "information_edge_state": state.information_edge,
            "data_flow_state": state.flow,
            "task_state": state.task,
            "task_dag_state": state.task,
        }
        explicit = {
            name: self.state_heads[name](entity)
            for name, entity in entity_by_state.items()
        }
        entity_by_presence = {
            "physical_node_present": state.physical_node,
            "physical_edge_present": state.physical_edge,
            "information_node_present": state.information_node,
            "information_edge_present": state.information_edge,
            "data_flow_present": state.flow,
            "task_present": state.task,
            "task_dag_state_present": state.task,
        }
        logits = {
            name: self.presence_heads[name](entity).squeeze(-1)
            for name, entity in entity_by_presence.items()
        }
        logits["information_link_activity"] = self.link_activity_head(
            state.information_edge
        ).squeeze(-1)
        logits["task_lifecycle"] = self.task_lifecycle_head(state.task)
        return explicit, logits

    def rollout(
        self,
        belief: _LatentState,
        future_action: Mapping[str, torch.Tensor],
        batch: ExplicitStateBatch,
        *,
        rollout_steps: int,
    ) -> R3RolloutOutput:
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        required = (
            "task_action",
            "task_action_present",
            "task_action_information_node_index",
        )
        if any(name not in future_action for name in required):
            raise ValueError("future action tensors are incomplete")
        if any(future_action[name].shape[1] < rollout_steps for name in required):
            raise ValueError("future action horizon is shorter than rollout_steps")

        state = belief
        explicit_steps: dict[str, list[torch.Tensor]] = {}
        logit_steps: dict[str, list[torch.Tensor]] = {}
        physical_steps: list[torch.Tensor] = []
        information_steps: list[torch.Tensor] = []
        business_steps: list[torch.Tensor] = []
        joint_steps: list[torch.Tensor] = []
        for step in range(rollout_steps):
            action_present = future_action["task_action_present"][:, step].to(
                state.task.dtype
            )
            encoded_action = self.action_encoder(future_action["task_action"][:, step])
            encoded_action = encoded_action * action_present.unsqueeze(-1)
            state.task = _gru(self.task_transition, encoded_action, state.task)
            information_action = self._scatter_action_to_information(
                encoded_action,
                future_action["task_action_information_node_index"][:, step].long(),
                action_present,
                state.information_node.shape[1],
            )
            state.information_node = _gru(
                self.information_node_transition,
                self.action_to_information(information_action),
                state.information_node,
            )
            pooled_action = _masked_mean(encoded_action, action_present)
            action_conditioned_joint = self.joint_transition(
                self.action_to_joint(pooled_action), state.joint
            )
            state = self._graph_update(state, batch)
            state = self._cross_couple(state, batch)
            refreshed = self._refresh_global(state, batch)
            refreshed.joint = self.joint_transition(
                refreshed.joint, action_conditioned_joint
            )
            state = refreshed
            explicit, logits = self._predict(state)
            for name, value in explicit.items():
                explicit_steps.setdefault(name, []).append(value)
            for name, value in logits.items():
                logit_steps.setdefault(name, []).append(value)
            physical_steps.append(state.physical_node)
            information_steps.append(state.information_node)
            business_steps.append(state.business)
            joint_steps.append(state.joint)
        return R3RolloutOutput(
            predicted_explicit={
                name: torch.stack(values, dim=1)
                for name, values in explicit_steps.items()
            },
            predicted_logits={
                name: torch.stack(values, dim=1)
                for name, values in logit_steps.items()
            },
            predicted_belief=BeliefSequence(
                physical_latent=torch.stack(physical_steps, dim=1),
                information_latent=torch.stack(information_steps, dim=1),
                business_latent=torch.stack(business_steps, dim=1),
                joint_latent=torch.stack(joint_steps, dim=1),
            ),
        )

    def forward(
        self, batch: ExplicitStateBatch, *, rollout_steps: int
    ) -> R3RolloutOutput:
        belief = self.infer_belief(batch)
        return self.rollout(
            belief,
            batch.future_action,
            batch,
            rollout_steps=rollout_steps,
        )


__all__ = [
    "BeliefSequence",
    "R3_MODEL_SCHEMA",
    "R3ReferenceConfig",
    "R3ReferenceWorldModel",
    "R3RolloutOutput",
    "REFERENCE_COMPONENTS",
]
