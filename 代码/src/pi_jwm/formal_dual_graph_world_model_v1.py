"""Action-conditioned strict dual-graph rollout models for formal PI-JWM data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch
from torch import nn

from .formal_graph_ops_v1 import (
    couple_agent_physical,
    couple_flow_bearer,
    dag_message_pass,
    information_message_pass,
    masked_index_mean,
    physical_message_pass,
)


COMPONENT_FEATURES = {
    "node": 7,
    "physical_edge": 5,
    "flow": 5,
    "task": 8,
}


@dataclass(frozen=True)
class FormalWorldModelConfig:
    mode: Literal["pooled_gru", "independent_dual_gnn", "coupled_dual_gnn"]
    hidden_dim: int = 16
    history_steps: int = 8
    horizon_steps: int = 3
    use_dag: bool = True
    use_cpu_action: bool = True
    use_cross_coupling: bool = True
    physical_only: bool = False
    information_only: bool = False
    log_variance_min: float = -8.0
    log_variance_max: float = 5.0


class _GatedMessage(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, current: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat((current, message), dim=-1)))
        return gate * message


class _DistributionHead(nn.Module):
    def __init__(self, hidden_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.mean = nn.Linear(hidden_dim, feature_dim)
        self.log_variance = nn.Linear(hidden_dim, feature_dim)

    def forward(self, latent: torch.Tensor, minimum: float, maximum: float) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean(latent), self.log_variance(latent).clamp(minimum, maximum)


def _temporal_encode(
    encoder: nn.Module,
    recurrent: nn.GRU,
    value: torch.Tensor,
    present: torch.Tensor,
) -> torch.Tensor:
    batch_size, steps, entity_count, _ = value.shape
    encoded = encoder(value) * present.unsqueeze(-1).to(value.dtype)
    sequence = encoded.permute(0, 2, 1, 3).reshape(batch_size * entity_count, steps, -1)
    output, _ = recurrent(sequence)
    return output[:, -1].reshape(batch_size, entity_count, -1)


def _masked_entity_mean(latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numeric = mask.unsqueeze(-1).to(latent.dtype)
    return (latent * numeric).sum(dim=1) / numeric.sum(dim=1).clamp_min(1.0)


def _scatter_task_messages(
    task_message: torch.Tensor,
    task_node_index: torch.Tensor,
    task_mask: torch.Tensor,
    node_count: int,
) -> torch.Tensor:
    endpoint_count = task_node_index.shape[-1]
    messages = task_message.unsqueeze(2).expand(-1, -1, endpoint_count, -1).flatten(1, 2)
    indices = task_node_index.flatten(1, 2)
    valid = task_mask.unsqueeze(-1).expand(-1, -1, endpoint_count).flatten(1, 2)
    return masked_index_mean(messages, indices, node_count, valid)


def _gather_attached_nodes(
    node_message: torch.Tensor,
    agent_node_index: torch.Tensor,
) -> torch.Tensor:
    node_count = node_message.shape[1]
    safe = agent_node_index.clamp(0, max(node_count - 1, 0)).long()
    gathered = torch.gather(
        node_message,
        1,
        safe.unsqueeze(-1).expand(-1, -1, node_message.shape[-1]),
    )
    valid = (agent_node_index >= 0) & (agent_node_index < node_count)
    return gathered * valid.unsqueeze(-1).to(gathered.dtype)


def _gather_task_nodes(
    node_latent: torch.Tensor,
    task_node_index: torch.Tensor,
    task_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, task_count, endpoint_count = task_node_index.shape
    node_count = node_latent.shape[1]
    safe = task_node_index.clamp(0, max(node_count - 1, 0)).long()
    expanded_nodes = node_latent.unsqueeze(1).expand(-1, task_count, -1, -1)
    gathered = torch.gather(
        expanded_nodes,
        2,
        safe.unsqueeze(-1).expand(-1, -1, -1, node_latent.shape[-1]),
    )
    valid = (
        (task_node_index >= 0)
        & (task_node_index < node_count)
        & task_mask.unsqueeze(-1)
    )
    numeric = valid.unsqueeze(-1).to(gathered.dtype)
    return (gathered * numeric).sum(dim=2) / numeric.sum(dim=2).clamp_min(1.0)


def _dag_relation_latent(task_latent: torch.Tensor, dag_edge_index: torch.Tensor) -> torch.Tensor:
    batch_size, task_count, feature_count = task_latent.shape
    parents = dag_edge_index[:, 0]
    children = dag_edge_index[:, 1]
    safe_parent = parents.clamp(0, max(task_count - 1, 0)).long()
    safe_child = children.clamp(0, max(task_count - 1, 0)).long()
    parent_latent = torch.gather(
        task_latent, 1, safe_parent.unsqueeze(-1).expand(-1, -1, feature_count)
    )
    child_latent = torch.gather(
        task_latent, 1, safe_child.unsqueeze(-1).expand(-1, -1, feature_count)
    )
    valid = (
        (parents >= 0)
        & (parents < task_count)
        & (children >= 0)
        & (children < task_count)
    )
    return 0.5 * (parent_latent + child_latent) * valid.unsqueeze(-1).to(task_latent.dtype)


class FormalDualGraphWorldModel(nn.Module):
    """Shared-interface pooled, independent-graph, and coupled PI-JWM models."""

    def __init__(self, config: FormalWorldModelConfig) -> None:
        super().__init__()
        if config.mode not in {"pooled_gru", "independent_dual_gnn", "coupled_dual_gnn"}:
            raise ValueError(f"unsupported mode: {config.mode}")
        if config.hidden_dim <= 0 or config.history_steps <= 0 or config.horizon_steps <= 0:
            raise ValueError("hidden_dim and rollout lengths must be positive")
        if config.physical_only and config.information_only:
            raise ValueError("physical_only and information_only cannot both be true")
        self.config = config
        hidden = config.hidden_dim
        self.cross_coupling_enabled = (
            config.mode == "coupled_dual_gnn"
            and config.use_cross_coupling
            and not config.physical_only
            and not config.information_only
        )

        self.node_encoder = nn.Linear(7, hidden)
        self.edge_encoder = nn.Linear(5, hidden)
        self.flow_encoder = nn.Linear(5, hidden)
        self.task_encoder = nn.Linear(8, hidden)
        self.action_encoder = nn.Linear(8, hidden)
        self.dag_state_encoder = nn.Linear(3, hidden)
        self.agent_encoder = nn.Linear(7, hidden)
        self.node_history = nn.GRU(hidden, hidden, batch_first=True)
        self.edge_history = nn.GRU(hidden, hidden, batch_first=True)
        self.flow_history = nn.GRU(hidden, hidden, batch_first=True)
        self.task_history = nn.GRU(hidden, hidden, batch_first=True)
        self.agent_history = nn.GRU(hidden, hidden, batch_first=True)

        self.global_context = nn.Linear(hidden * 5, hidden)
        self.node_transition = nn.GRUCell(hidden, hidden)
        self.edge_transition = nn.GRUCell(hidden, hidden)
        self.agent_transition = nn.GRUCell(hidden, hidden)
        self.flow_transition = nn.GRUCell(hidden, hidden)
        self.task_transition = nn.GRUCell(hidden, hidden)
        self.node_cip_gate = _GatedMessage(hidden)
        self.agent_cip_gate = _GatedMessage(hidden)
        self.edge_cfe_gate = _GatedMessage(hidden)
        self.flow_cfe_gate = _GatedMessage(hidden)
        self.task_node_gate = _GatedMessage(hidden)

        self.state_heads = nn.ModuleDict(
            {
                name: _DistributionHead(hidden, feature_dim)
                for name, feature_dim in COMPONENT_FEATURES.items()
            }
        )
        self.dag_state_head = _DistributionHead(hidden, 3)
        self.presence_heads = nn.ModuleDict(
            {name: nn.Linear(hidden, 1) for name in COMPONENT_FEATURES}
        )
        self.link_activity_head = nn.Linear(hidden, 1)
        self.task_lifecycle_head = nn.Linear(hidden, 5)
        self.dag_release_head = nn.Linear(hidden, 1)
        self.dag_edge_presence_head = nn.Linear(hidden, 1)

    @staticmethod
    def _apply_transition(cell: nn.GRUCell, message: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        shape = latent.shape
        return cell(message.reshape(-1, shape[-1]), latent.reshape(-1, shape[-1])).reshape(shape)

    def _initial_latents(self, history: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        node = _temporal_encode(
            self.node_encoder, self.node_history, history["node_state"], history["node_present"]
        )
        edge = _temporal_encode(
            self.edge_encoder,
            self.edge_history,
            history["physical_edge_state"],
            history["physical_edge_present"],
        )
        flow = _temporal_encode(
            self.flow_encoder, self.flow_history, history["flow_state"], history["flow_present"]
        )
        task_state = self.task_encoder(history["task_state"])
        action_history = history["task_action"].clone()
        if not self.config.use_cpu_action:
            action_history[..., 5:8] = 0.0
        task_input = (
            task_state
            + self.action_encoder(action_history)
            * history["task_action_present"].unsqueeze(-1).to(task_state.dtype)
            + self.dag_state_encoder(history["task_dag_state"])
            * history["task_dag_state_present"].unsqueeze(-1).to(task_state.dtype)
        )
        task_input = task_input * history["task_present"].unsqueeze(-1).to(task_input.dtype)
        batch_size, steps, task_count, hidden = task_input.shape
        task_sequence = task_input.permute(0, 2, 1, 3).reshape(batch_size * task_count, steps, hidden)
        task_output, _ = self.task_history(task_sequence)
        task = task_output[:, -1].reshape(batch_size, task_count, hidden)
        agent = _temporal_encode(
            self.agent_encoder, self.agent_history, history["node_state"], history["node_present"]
        )
        return node, edge, agent, flow, task

    def forward(self, batch: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        history = batch["history"]
        future_action = batch["future_action"]
        static = batch["static"]
        if history["node_state"].shape[1] != self.config.history_steps:
            raise ValueError("history length does not match model configuration")
        if future_action["task_action"].shape[1] != self.config.horizon_steps:
            raise ValueError("future action length does not match model configuration")

        node, edge, agent, flow, task = self._initial_latents(history)
        node_mask = static["node_kind_index"] >= 0
        edge_mask = history["physical_edge_present"][:, -1].bool()
        flow_mask = history["flow_present"][:, -1].bool()
        task_mask = static["task_valid"].bool()
        agent_mask = (static["agent_node_index"] >= 0) & node_mask
        dag_edge_mask = history["dag_edge_present"][:, -1].bool() & static["dag_edge_valid"].bool()
        task_node_index = history["task_node_index"][:, -1]
        bearer_mask = history["flow_bearer_mask"][:, -1]

        outputs: dict[str, list[torch.Tensor]] = {}
        for step in range(self.config.horizon_steps):
            raw_action = future_action["task_action"][:, step].clone()
            if not self.config.use_cpu_action:
                raw_action[..., 5:8] = 0.0
            action_present = future_action["task_action_present"][:, step].bool()
            action = self.action_encoder(raw_action) * action_present.unsqueeze(-1).to(raw_action.dtype)
            action_to_node = _scatter_task_messages(
                action,
                future_action["task_action_node_index"][:, step],
                action_present,
                node.shape[1],
            )
            action_to_agent = _gather_attached_nodes(action_to_node, static["agent_node_index"])

            zero_node = torch.zeros_like(node)
            zero_edge = torch.zeros_like(edge)
            zero_agent = torch.zeros_like(agent)
            zero_flow = torch.zeros_like(flow)
            zero_task = torch.zeros_like(task)
            node_message, edge_message = zero_node, zero_edge
            agent_message, flow_message = zero_agent, zero_flow
            task_message = zero_task

            if self.config.mode == "pooled_gru":
                global_message = torch.tanh(
                    self.global_context(
                        torch.cat(
                            (
                                _masked_entity_mean(node, node_mask),
                                _masked_entity_mean(edge, edge_mask),
                                _masked_entity_mean(flow, flow_mask),
                                _masked_entity_mean(task, task_mask),
                                _masked_entity_mean(action, action_present),
                            ),
                            dim=-1,
                        )
                    )
                )
                node_message = global_message.unsqueeze(1).expand_as(node)
                edge_message = global_message.unsqueeze(1).expand_as(edge)
                agent_message = global_message.unsqueeze(1).expand_as(agent)
                flow_message = global_message.unsqueeze(1).expand_as(flow)
                task_message = global_message.unsqueeze(1).expand_as(task)
            else:
                if not self.config.information_only:
                    node_message, edge_message = physical_message_pass(
                        node,
                        edge,
                        static["physical_edge_endpoint_index"],
                        node_mask,
                        edge_mask,
                    )
                if not self.config.physical_only:
                    agent_message, flow_message = information_message_pass(
                        agent,
                        flow,
                        static["flow_endpoint_index"],
                        agent_mask,
                        flow_mask,
                    )
                if self.config.use_dag:
                    task_message = dag_message_pass(
                        task,
                        static["dag_edge_index"],
                        dag_edge_mask,
                        task_mask,
                    )

            if self.cross_coupling_enabled:
                agent_from_node, node_from_agent = couple_agent_physical(
                    agent,
                    node,
                    static["agent_node_index"],
                    agent_mask,
                    node_mask,
                )
                flow_from_edge, edge_from_flow = couple_flow_bearer(
                    flow,
                    edge,
                    bearer_mask,
                    flow_mask,
                    edge_mask,
                )
                task_to_node = _scatter_task_messages(task, task_node_index, task_mask, node.shape[1])
                task_to_agent = _gather_attached_nodes(task_to_node, static["agent_node_index"])
                node_message = (
                    node_message
                    + self.node_cip_gate(node, node_from_agent)
                    + self.task_node_gate(node, task_to_node)
                )
                agent_message = (
                    agent_message
                    + self.agent_cip_gate(agent, agent_from_node)
                    + self.task_node_gate(agent, task_to_agent)
                )
                edge_message = edge_message + self.edge_cfe_gate(edge, edge_from_flow)
                flow_message = flow_message + self.flow_cfe_gate(flow, flow_from_edge)
                task_message = task_message + self.task_node_gate(
                    task, _gather_task_nodes(node, task_node_index, task_mask)
                )

            node = self._apply_transition(self.node_transition, node_message + action_to_node, node)
            edge = self._apply_transition(self.edge_transition, edge_message, edge)
            agent = self._apply_transition(self.agent_transition, agent_message + action_to_agent, agent)
            flow = self._apply_transition(self.flow_transition, flow_message, flow)
            task = self._apply_transition(self.task_transition, task_message + action, task)

            latent_by_name = {
                "node": node,
                "physical_edge": edge,
                "flow": flow,
                "task": task,
            }
            for name, latent in latent_by_name.items():
                mean, log_variance = self.state_heads[name](
                    latent,
                    self.config.log_variance_min,
                    self.config.log_variance_max,
                )
                outputs.setdefault(f"{name}_state_mean", []).append(mean)
                outputs.setdefault(f"{name}_state_log_variance", []).append(log_variance)
                outputs.setdefault(f"{name}_presence_logits", []).append(
                    self.presence_heads[name](latent).squeeze(-1)
                )
            dag_mean, dag_log_variance = self.dag_state_head(
                task,
                self.config.log_variance_min,
                self.config.log_variance_max,
            )
            outputs.setdefault("task_dag_state_mean", []).append(dag_mean)
            outputs.setdefault("task_dag_state_log_variance", []).append(dag_log_variance)
            outputs.setdefault("link_activity_logits", []).append(
                self.link_activity_head(edge).squeeze(-1)
            )
            outputs.setdefault("task_lifecycle_logits", []).append(self.task_lifecycle_head(task))
            outputs.setdefault("dag_release_logits", []).append(self.dag_release_head(task).squeeze(-1))
            dag_relation = _dag_relation_latent(task, static["dag_edge_index"])
            outputs.setdefault("dag_edge_presence_logits", []).append(
                self.dag_edge_presence_head(dag_relation).squeeze(-1)
            )

        stacked = {key: torch.stack(values, dim=1) for key, values in outputs.items()}
        for name in COMPONENT_FEATURES:
            stacked[f"{name}_state"] = stacked[f"{name}_state_mean"]
        return stacked


__all__ = [
    "COMPONENT_FEATURES",
    "FormalDualGraphWorldModel",
    "FormalWorldModelConfig",
]
