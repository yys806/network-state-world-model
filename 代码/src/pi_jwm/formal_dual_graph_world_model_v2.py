"""Directed dynamic dual-graph world model for formal PI-JWM data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .formal_directed_graph_ops_v2 import (
    directed_relation_messages,
    weighted_index_mean,
)


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

    def __init__(self, config: FormalDirectedDynamicWorldModelConfig) -> None:
        super().__init__()
        if min(config.hidden_dim, config.history_steps, config.horizon_steps) <= 0:
            raise ValueError("hidden_dim and rollout lengths must be positive")
        self.config = config
        hidden = config.hidden_dim
        self.flow_encoder = nn.Linear(5, hidden)
        self.task_encoder = nn.Linear(8, hidden)
        self.agent_flow_in = nn.Linear(hidden, hidden)
        self.agent_flow_out = nn.Linear(hidden, hidden)
        self.agent_task_roles = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(4))
        self.agent_history = nn.GRU(hidden, hidden, batch_first=True)

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


__all__ = [
    "FormalDirectedDynamicWorldModelConfig",
    "FormalDirectedDynamicWorldModelV2",
]
