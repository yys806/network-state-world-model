"""Directed dynamic dual-graph world model for formal PI-JWM data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .formal_directed_graph_ops_v2 import (
    direct_bearer_candidates,
    directed_relation_messages,
    weighted_index_mean,
)


COMPONENT_FEATURES = {
    "node": 7,
    "physical_edge": 5,
    "flow": 5,
    "task": 8,
}


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

    def forward(
        self, latent: torch.Tensor, minimum: float, maximum: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


@dataclass(frozen=True)
class FormalDirectedDynamicWorldModelConfig:
    hidden_dim: int = 16
    history_steps: int = 8
    horizon_steps: int = 3
    use_dag: bool = True
    use_cpu_action: bool = True
    use_cross_coupling: bool = True
    residual_state_prediction: bool = False
    use_system_energy_head: bool = False
    log_variance_min: float = -8.0
    log_variance_max: float = 5.0


def _node_to_agent_index(
    agent_node_index: torch.Tensor,
    node_count: int,
) -> torch.Tensor:
    batch_size, agent_count = agent_node_index.shape
    inverse = torch.full(
        (batch_size, node_count),
        -1,
        dtype=torch.long,
        device=agent_node_index.device,
    )
    for batch_index in range(batch_size):
        nodes = agent_node_index[batch_index]
        valid = (nodes >= 0) & (nodes < node_count)
        if torch.any(valid):
            inverse[batch_index, nodes[valid].long()] = torch.arange(
                agent_count, device=agent_node_index.device
            )[valid]
    return inverse


def _map_node_endpoints_to_agents(
    node_endpoint_index: torch.Tensor,
    node_to_agent: torch.Tensor,
) -> torch.Tensor:
    batch_size = node_to_agent.shape[0]
    if node_endpoint_index.ndim == 2:
        endpoints = node_endpoint_index.unsqueeze(0).expand(batch_size, -1, -1)
    elif node_endpoint_index.ndim == 3 and node_endpoint_index.shape[0] == batch_size:
        endpoints = node_endpoint_index
    else:
        raise ValueError("node_endpoint_index must be batch aligned")
    node_count = node_to_agent.shape[1]
    safe = endpoints.clamp(0, max(node_count - 1, 0)).long()
    mapped = torch.gather(
        node_to_agent.unsqueeze(1).expand(-1, endpoints.shape[1], -1),
        2,
        safe,
    )
    valid = (endpoints >= 0) & (endpoints < node_count)
    return torch.where(valid, mapped, torch.full_like(mapped, -1))


class FormalDirectedDynamicWorldModelV2(nn.Module):
    """PI-JWM v2 with information-side agent initialization."""

    model_version = "directed_dynamic_v2"
    latent_dynamics = "deterministic"

    def __init__(self, config: FormalDirectedDynamicWorldModelConfig) -> None:
        super().__init__()
        if min(config.hidden_dim, config.history_steps, config.horizon_steps) <= 0:
            raise ValueError("hidden_dim and rollout lengths must be positive")
        self.config = config
        hidden = config.hidden_dim
        self.node_encoder = nn.Linear(7, hidden)
        self.edge_encoder = nn.Linear(5, hidden)
        self.flow_encoder = nn.Linear(5, hidden)
        self.task_encoder = nn.Linear(8, hidden)
        self.action_encoder = nn.Linear(8, hidden)
        self.dag_state_encoder = nn.Linear(3, hidden)
        self.node_history = nn.GRU(hidden, hidden, batch_first=True)
        self.edge_history = nn.GRU(hidden, hidden, batch_first=True)
        self.flow_history = nn.GRU(hidden, hidden, batch_first=True)
        self.task_history = nn.GRU(hidden, hidden, batch_first=True)
        self.agent_flow_in = nn.Linear(hidden, hidden)
        self.agent_flow_out = nn.Linear(hidden, hidden)
        self.agent_task_roles = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(4))
        self.agent_history = nn.GRU(hidden, hidden, batch_first=True)

        self.physical_incoming = nn.Linear(hidden, hidden)
        self.physical_outgoing = nn.Linear(hidden, hidden)
        self.physical_relation = nn.Linear(hidden * 3, hidden)
        self.information_incoming = nn.Linear(hidden, hidden)
        self.information_outgoing = nn.Linear(hidden, hidden)
        self.information_relation = nn.Linear(hidden * 3, hidden)
        self.node_transition = nn.GRUCell(hidden, hidden)
        self.edge_transition = nn.GRUCell(hidden, hidden)
        self.agent_transition = nn.GRUCell(hidden, hidden)
        self.flow_transition = nn.GRUCell(hidden, hidden)
        self.task_transition = nn.GRUCell(hidden, hidden)
        self.node_cip_gate = _GatedMessage(hidden)
        self.agent_cip_gate = _GatedMessage(hidden)
        self.edge_cfe_gate = _GatedMessage(hidden)
        self.flow_cfe_gate = _GatedMessage(hidden)
        self.task_agent_gate = _GatedMessage(hidden)

        self.state_heads = nn.ModuleDict(
            {
                name: _DistributionHead(hidden, feature_dim)
                for name, feature_dim in COMPONENT_FEATURES.items()
            }
        )
        self.dag_state_head = _DistributionHead(hidden, 3)
        self.uav_energy_head = (
            _DistributionHead(hidden, 1) if config.use_system_energy_head else None
        )
        self.presence_heads = nn.ModuleDict(
            {name: nn.Linear(hidden, 1) for name in COMPONENT_FEATURES}
        )
        self.link_activity_head = nn.Linear(hidden, 1)
        self.task_lifecycle_head = nn.Linear(hidden, 5)
        self.dag_release_head = nn.Linear(hidden, 1)
        self.dag_edge_presence_head = nn.Linear(hidden, 1)

    def encode_information_agent_history(
        self,
        history: Mapping[str, torch.Tensor],
        static: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode agents only from flow/task evidence and their real endpoint roles."""

        flow_state = history["flow_state"]
        task_state = history["task_state"]
        if flow_state.shape[1] != self.config.history_steps:
            raise ValueError("history length does not match model configuration")
        batch_size, steps, _, _ = flow_state.shape
        agent_node_index = static["agent_node_index"].long()
        agent_count = agent_node_index.shape[1]
        node_count = static["node_kind_index"].shape[1]
        node_to_agent = _node_to_agent_index(agent_node_index, node_count)
        flow_endpoints = _map_node_endpoints_to_agents(
            static["flow_endpoint_index"].long(), node_to_agent
        )
        agent_weight = (
            (agent_node_index >= 0)
            & (agent_node_index < node_count)
        ).to(flow_state.dtype)
        flow_valid = static["flow_valid"].to(flow_state.dtype)
        task_valid = static["task_valid"].to(task_state.dtype)

        sequence = []
        for step in range(steps):
            encoded_flow = self.flow_encoder(flow_state[:, step])
            flow_weight = history["flow_present"][:, step].to(flow_state.dtype) * flow_valid
            zeros = encoded_flow.new_zeros((batch_size, agent_count, encoded_flow.shape[-1]))
            incoming, outgoing, _ = directed_relation_messages(
                zeros,
                encoded_flow,
                flow_endpoints,
                agent_weight,
                flow_weight,
            )
            agent_input = self.agent_flow_in(incoming) + self.agent_flow_out(outgoing)

            encoded_task = self.task_encoder(task_state[:, step])
            task_weight = history["task_present"][:, step].to(task_state.dtype) * task_valid
            task_nodes = history["task_node_index"][:, step].long()
            for role_index, role_projection in enumerate(self.agent_task_roles):
                mapped_role = _map_node_endpoints_to_agents(
                    task_nodes[:, :, role_index : role_index + 1], node_to_agent
                ).squeeze(-1)
                role_message = weighted_index_mean(
                    encoded_task,
                    mapped_role,
                    agent_count,
                    task_weight,
                )
                agent_input = agent_input + role_projection(role_message)
            sequence.append(agent_input * agent_weight.unsqueeze(-1))

        stacked = torch.stack(sequence, dim=1)
        flattened = stacked.permute(0, 2, 1, 3).reshape(
            batch_size * agent_count, steps, -1
        )
        output, _ = self.agent_history(flattened)
        return output[:, -1].reshape(batch_size, agent_count, -1) * agent_weight.unsqueeze(-1)

    @staticmethod
    def _apply_transition(
        cell: nn.GRUCell, message: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        shape = latent.shape
        return cell(message.reshape(-1, shape[-1]), latent.reshape(-1, shape[-1])).reshape(shape)

    @staticmethod
    def _scatter_task_messages(
        task_message: torch.Tensor,
        task_node_index: torch.Tensor,
        task_weight: torch.Tensor,
        node_count: int,
    ) -> torch.Tensor:
        endpoint_count = task_node_index.shape[-1]
        messages = task_message.unsqueeze(2).expand(-1, -1, endpoint_count, -1).flatten(1, 2)
        indices = task_node_index.flatten(1, 2)
        weights = task_weight.unsqueeze(-1).expand(-1, -1, endpoint_count).flatten(1, 2)
        return weighted_index_mean(messages, indices, node_count, weights)

    @staticmethod
    def _gather_attached_nodes(
        node_message: torch.Tensor,
        agent_node_index: torch.Tensor,
        agent_weight: torch.Tensor,
    ) -> torch.Tensor:
        node_count = node_message.shape[1]
        safe = agent_node_index.clamp(0, max(node_count - 1, 0)).long()
        gathered = torch.gather(
            node_message,
            1,
            safe.unsqueeze(-1).expand(-1, -1, node_message.shape[-1]),
        )
        valid = (agent_node_index >= 0) & (agent_node_index < node_count)
        return gathered * valid.unsqueeze(-1).to(gathered.dtype) * agent_weight.unsqueeze(-1)

    @staticmethod
    def _gather_task_nodes(
        node_latent: torch.Tensor,
        task_node_index: torch.Tensor,
        task_weight: torch.Tensor,
    ) -> torch.Tensor:
        _, task_count, _ = task_node_index.shape
        node_count = node_latent.shape[1]
        safe = task_node_index.clamp(0, max(node_count - 1, 0)).long()
        expanded = node_latent.unsqueeze(1).expand(-1, task_count, -1, -1)
        gathered = torch.gather(
            expanded,
            2,
            safe.unsqueeze(-1).expand(-1, -1, -1, node_latent.shape[-1]),
        )
        valid = (task_node_index >= 0) & (task_node_index < node_count)
        numeric = valid.unsqueeze(-1).to(gathered.dtype)
        mean = (gathered * numeric).sum(dim=2) / numeric.sum(dim=2).clamp_min(1.0)
        return mean * task_weight.unsqueeze(-1)

    @staticmethod
    def _cip_messages(
        agent: torch.Tensor,
        node: torch.Tensor,
        attachment: torch.Tensor,
        agent_weight: torch.Tensor,
        node_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_count = node.shape[1]
        safe = attachment.clamp(0, max(node_count - 1, 0)).long()
        gathered_node = torch.gather(
            node,
            1,
            safe.unsqueeze(-1).expand(-1, -1, node.shape[-1]),
        )
        attached_node_weight = torch.gather(node_weight, 1, safe)
        valid_weight = (
            agent_weight
            * attached_node_weight
            * ((attachment >= 0) & (attachment < node_count)).to(agent.dtype)
        )
        agent_from_node = gathered_node * valid_weight.unsqueeze(-1)
        node_from_agent = weighted_index_mean(
            agent, attachment, node_count, valid_weight
        )
        return agent_from_node, node_from_agent

    @staticmethod
    def _cfe_messages(
        flow: torch.Tensor,
        edge: torch.Tensor,
        cfe_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relation = cfe_weight.to(flow.dtype).clamp_min(0.0)
        support = (relation > 0).to(flow.dtype)
        flow_from_edge = torch.bmm(relation, edge) / support.sum(
            dim=2, keepdim=True
        ).clamp_min(1.0)
        transposed = relation.transpose(1, 2)
        transposed_support = support.transpose(1, 2)
        edge_from_flow = torch.bmm(transposed, flow) / transposed_support.sum(
            dim=2, keepdim=True
        ).clamp_min(1.0)
        return flow_from_edge, edge_from_flow

    @staticmethod
    def _dag_messages(
        task: torch.Tensor,
        dag_edge_index: torch.Tensor,
        dag_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, task_count, _ = task.shape
        if dag_edge_index.ndim == 2:
            edge_index = dag_edge_index.unsqueeze(0).expand(batch_size, -1, -1)
        elif dag_edge_index.ndim == 3 and dag_edge_index.shape[0] == batch_size:
            edge_index = dag_edge_index
        else:
            raise ValueError("dag_edge_index must be batch aligned")
        parents = edge_index[:, 0]
        children = edge_index[:, 1]
        safe_parent = parents.clamp(0, max(task_count - 1, 0)).long()
        safe_child = children.clamp(0, max(task_count - 1, 0)).long()
        parent = torch.gather(task, 1, safe_parent.unsqueeze(-1).expand(-1, -1, task.shape[-1]))
        child = torch.gather(task, 1, safe_child.unsqueeze(-1).expand(-1, -1, task.shape[-1]))
        valid = (
            (parents >= 0)
            & (parents < task_count)
            & (children >= 0)
            & (children < task_count)
        ).to(task.dtype)
        weight = dag_weight * valid
        message = weighted_index_mean(parent, children, task_count, weight)
        relation_latent = 0.5 * (parent + child) * weight.unsqueeze(-1)
        return message, relation_latent

    def _directed_pass(
        self,
        entity: torch.Tensor,
        relation: torch.Tensor,
        endpoints: torch.Tensor,
        entity_weight: torch.Tensor,
        relation_weight: torch.Tensor,
        incoming_layer: nn.Linear,
        outgoing_layer: nn.Linear,
        relation_layer: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        incoming, outgoing, ordered_context = directed_relation_messages(
            entity, relation, endpoints, entity_weight, relation_weight
        )
        entity_message = torch.tanh(
            incoming_layer(incoming) + outgoing_layer(outgoing)
        )
        relation_message = torch.tanh(
            relation_layer(torch.cat((relation, ordered_context), dim=-1))
        ) * relation_weight.unsqueeze(-1)
        return entity_message, relation_message

    def _initial_latents(
        self,
        history: Mapping[str, torch.Tensor],
        static: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
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
        task_sequence = task_input.permute(0, 2, 1, 3).reshape(
            batch_size * task_count, steps, hidden
        )
        task_output, _ = self.task_history(task_sequence)
        task = task_output[:, -1].reshape(batch_size, task_count, hidden)
        agent = self.encode_information_agent_history(history, static)
        return node, edge, agent, flow, task

    def forward(
        self, batch: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        history = batch["history"]
        future_action = batch["future_action"]
        static = batch["static"]
        if future_action["task_action"].shape[1] != self.config.horizon_steps:
            raise ValueError("future action length does not match model configuration")

        node, edge, agent, flow, task = self._initial_latents(history, static)
        residual_bases = {
            name: history[f"{name}_state"][:, -1] for name in COMPONENT_FEATURES
        }
        dag_residual_base = history["task_dag_state"][:, -1]
        node_weight = (static["node_kind_index"] >= 0).to(node.dtype)
        edge_valid = (
            (static["physical_edge_endpoint_index"][..., 0] >= 0)
            & (static["physical_edge_endpoint_index"][..., 1] >= 0)
        ).to(node.dtype)
        flow_valid = static["flow_valid"].to(node.dtype)
        task_valid = static["task_valid"].to(node.dtype)
        dag_valid = static["dag_edge_valid"].to(node.dtype)
        agent_node_index = static["agent_node_index"].long()
        agent_weight = (
            (agent_node_index >= 0)
            & (agent_node_index < node.shape[1])
        ).to(node.dtype)
        node_to_agent = _node_to_agent_index(agent_node_index, node.shape[1])
        agent_flow_endpoints = _map_node_endpoints_to_agents(
            static["flow_endpoint_index"].long(), node_to_agent
        )
        task_node_index = history["task_node_index"][:, -1].long()
        task_agent_index = _map_node_endpoints_to_agents(
            task_node_index, node_to_agent
        )

        edge_weight = history["physical_edge_present"][:, -1].to(node.dtype) * edge_valid
        flow_weight = history["flow_present"][:, -1].to(node.dtype) * flow_valid
        task_weight = history["task_present"][:, -1].to(node.dtype) * task_valid
        dag_weight = history["dag_edge_present"][:, -1].to(node.dtype) * dag_valid
        cfe_weight = (
            history["flow_bearer_mask"][:, -1].to(node.dtype)
            * flow_weight.unsqueeze(-1)
            * edge_weight.unsqueeze(1)
        )
        bearer_candidate = direct_bearer_candidates(
            static["flow_endpoint_index"],
            static["physical_edge_endpoint_index"],
            flow_valid,
            edge_valid,
        ).to(node.dtype)

        outputs: dict[str, list[torch.Tensor]] = {}
        audit_edge_weight: list[torch.Tensor] = []
        audit_flow_weight: list[torch.Tensor] = []
        audit_task_weight: list[torch.Tensor] = []
        audit_dag_weight: list[torch.Tensor] = []
        audit_cfe_weight: list[torch.Tensor] = []
        for step in range(self.config.horizon_steps):
            audit_edge_weight.append(edge_weight)
            audit_flow_weight.append(flow_weight)
            audit_task_weight.append(task_weight)
            audit_dag_weight.append(dag_weight)
            audit_cfe_weight.append(cfe_weight)

            raw_action = future_action["task_action"][:, step].clone()
            if not self.config.use_cpu_action:
                raw_action[..., 5:8] = 0.0
            action_present = future_action["task_action_present"][:, step].to(node.dtype)
            action = self.action_encoder(raw_action) * action_present.unsqueeze(-1)
            action_to_node = self._scatter_task_messages(
                action,
                future_action["task_action_node_index"][:, step],
                action_present,
                node.shape[1],
            )
            action_to_agent = self._gather_attached_nodes(
                action_to_node, agent_node_index, agent_weight
            )

            node_message, edge_message = self._directed_pass(
                node,
                edge,
                static["physical_edge_endpoint_index"],
                node_weight,
                edge_weight,
                self.physical_incoming,
                self.physical_outgoing,
                self.physical_relation,
            )
            agent_message, flow_message = self._directed_pass(
                agent,
                flow,
                agent_flow_endpoints,
                agent_weight,
                flow_weight,
                self.information_incoming,
                self.information_outgoing,
                self.information_relation,
            )
            if self.config.use_dag:
                task_message, dag_relation_latent = self._dag_messages(
                    task, static["dag_edge_index"], dag_weight
                )
            else:
                task_message = torch.zeros_like(task)
                dag_relation_latent = task.new_zeros(
                    (task.shape[0], dag_weight.shape[1], task.shape[-1])
                )

            if self.config.use_cross_coupling:
                agent_from_node, node_from_agent = self._cip_messages(
                    agent,
                    node,
                    agent_node_index,
                    agent_weight,
                    node_weight,
                )
                flow_from_edge, edge_from_flow = self._cfe_messages(flow, edge, cfe_weight)
                task_to_agent = self._scatter_task_messages(
                    task, task_agent_index, task_weight, agent.shape[1]
                )
                agent_cip_message = self.agent_cip_gate(agent, agent_from_node)
                task_from_agent = self._gather_task_nodes(
                    agent + agent_cip_message,
                    task_agent_index,
                    task_weight,
                )
                node_message = node_message + self.node_cip_gate(node, node_from_agent)
                agent_message = (
                    agent_message
                    + agent_cip_message
                    + self.task_agent_gate(agent, task_to_agent)
                )
                edge_message = edge_message + self.edge_cfe_gate(edge, edge_from_flow)
                flow_message = flow_message + self.flow_cfe_gate(flow, flow_from_edge)
                task_message = task_message + self.task_agent_gate(task, task_from_agent)

            node = self._apply_transition(self.node_transition, node_message + action_to_node, node)
            edge = self._apply_transition(self.edge_transition, edge_message, edge)
            agent = self._apply_transition(
                self.agent_transition, agent_message + action_to_agent, agent
            )
            flow = self._apply_transition(self.flow_transition, flow_message, flow)
            task = self._apply_transition(self.task_transition, task_message + action, task)

            latent_by_name = {
                "node": node,
                "physical_edge": edge,
                "flow": flow,
                "task": task,
            }
            step_presence: dict[str, torch.Tensor] = {}
            for name, latent in latent_by_name.items():
                mean, log_variance = self.state_heads[name](
                    latent,
                    self.config.log_variance_min,
                    self.config.log_variance_max,
                )
                if self.config.residual_state_prediction:
                    mean = residual_bases[name] + mean
                outputs.setdefault(f"{name}_state_mean", []).append(mean)
                outputs.setdefault(f"{name}_state_log_variance", []).append(log_variance)
                presence = self.presence_heads[name](latent).squeeze(-1)
                outputs.setdefault(f"{name}_presence_logits", []).append(presence)
                step_presence[name] = presence

            dag_mean, dag_log_variance = self.dag_state_head(
                task,
                self.config.log_variance_min,
                self.config.log_variance_max,
            )
            if self.config.residual_state_prediction:
                dag_mean = dag_residual_base + dag_mean
            outputs.setdefault("task_dag_state_mean", []).append(dag_mean)
            outputs.setdefault("task_dag_state_log_variance", []).append(dag_log_variance)
            outputs.setdefault("link_activity_logits", []).append(
                self.link_activity_head(edge).squeeze(-1)
            )
            outputs.setdefault("task_lifecycle_logits", []).append(
                self.task_lifecycle_head(task)
            )
            outputs.setdefault("dag_release_logits", []).append(
                self.dag_release_head(task).squeeze(-1)
            )
            dag_edge_logits = self.dag_edge_presence_head(dag_relation_latent).squeeze(-1)
            outputs.setdefault("dag_edge_presence_logits", []).append(dag_edge_logits)
            if self.uav_energy_head is not None:
                energy_mean, energy_log_variance = self.uav_energy_head(
                    node,
                    self.config.log_variance_min,
                    self.config.log_variance_max,
                )
                outputs.setdefault("uav_energy_delta_mean", []).append(
                    torch.nn.functional.softplus(energy_mean.squeeze(-1))
                )
                outputs.setdefault("uav_energy_delta_log_variance", []).append(
                    energy_log_variance.squeeze(-1)
                )

            edge_weight = torch.sigmoid(step_presence["physical_edge"]) * edge_valid
            flow_weight = torch.sigmoid(step_presence["flow"]) * flow_valid
            task_weight = torch.sigmoid(step_presence["task"]) * task_valid
            dag_weight = torch.sigmoid(dag_edge_logits) * dag_valid
            cfe_weight = (
                bearer_candidate
                * flow_weight.unsqueeze(-1)
                * edge_weight.unsqueeze(1)
            )

        stacked = {key: torch.stack(value, dim=1) for key, value in outputs.items()}
        stacked["rollout_edge_weight"] = torch.stack(audit_edge_weight, dim=1)
        stacked["rollout_flow_weight"] = torch.stack(audit_flow_weight, dim=1)
        stacked["rollout_task_weight"] = torch.stack(audit_task_weight, dim=1)
        stacked["rollout_dag_weight"] = torch.stack(audit_dag_weight, dim=1)
        stacked["rollout_cfe_weight"] = torch.stack(audit_cfe_weight, dim=1)
        return stacked


__all__ = [
    "FormalDirectedDynamicWorldModelConfig",
    "FormalDirectedDynamicWorldModelV2",
]
