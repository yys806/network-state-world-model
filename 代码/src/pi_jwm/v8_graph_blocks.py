"""Reusable PI-JWM v8 physical-information graph blocks.

v8 starts by making message passing explicit: edge states are updated with
physical, information, action, and endpoint-node context before they are
aggregated back to node states.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class V8GraphBlockConfig:
    hidden_dim: int
    graph_mode: str = "dual"
    fusion_mode: str = "gated"
    fusion_num_heads: int = 4


class EdgeUpdateBlock(nn.Module):
    """Update per-edge latent states from dual-graph edge features and actions."""

    def __init__(self, config: V8GraphBlockConfig):
        super().__init__()
        _validate_config(config)
        hidden = config.hidden_dim
        self.config = config
        if config.fusion_mode == "gated":
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden * 3, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 3),
            )
            self.fusion_attention = None
            self.attention_norm = None
        else:
            if hidden % config.fusion_num_heads != 0:
                raise ValueError("hidden_dim must be divisible by fusion_num_heads")
            self.fusion_gate = None
            self.fusion_attention = nn.MultiheadAttention(hidden, config.fusion_num_heads, batch_first=True)
            self.attention_norm = nn.LayerNorm(hidden)
        self.edge_project = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.output_norm = nn.LayerNorm(hidden)

    def forward(
        self,
        node_state: Tensor,
        physical_edge_state: Tensor,
        info_edge_state: Tensor,
        action_state: Tensor,
        edge_src_idx: Tensor,
        edge_dst_idx: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        _validate_graph_inputs(
            node_state,
            physical_edge_state,
            info_edge_state,
            action_state,
            edge_src_idx,
            edge_dst_idx,
        )

        physical, information = self._apply_graph_mode(physical_edge_state, info_edge_state)
        tokens = torch.stack([physical, information, action_state], dim=-2)
        if self.config.fusion_mode == "gated":
            fusion_logits = self.fusion_gate(torch.cat([physical, information, action_state], dim=-1))
            fusion_weights = torch.softmax(fusion_logits, dim=-1)
            fused_edge = self.edge_project((tokens * fusion_weights.unsqueeze(-1)).sum(dim=-2))
            diagnostics = {"edge_fusion_weights": fusion_weights}
        else:
            batch_size, num_edges, _, hidden_dim = tokens.shape
            flat_tokens = tokens.reshape(batch_size * num_edges, 3, hidden_dim)
            attended, attention = self.fusion_attention(
                flat_tokens,
                flat_tokens,
                flat_tokens,
                need_weights=True,
                average_attn_weights=True,
            )
            attended = self.attention_norm(attended + flat_tokens)
            fused_edge = self.edge_project(attended.mean(dim=1).reshape(batch_size, num_edges, hidden_dim))
            diagnostics = {"edge_fusion_attention": attention.reshape(batch_size, num_edges, 3, 3)}

        edge_src_idx = edge_src_idx.to(device=node_state.device, dtype=torch.long)
        edge_dst_idx = edge_dst_idx.to(device=node_state.device, dtype=torch.long)
        src_node = node_state.index_select(1, edge_src_idx)
        dst_node = node_state.index_select(1, edge_dst_idx)
        edge_delta = self.edge_update(torch.cat([fused_edge, src_node, dst_node], dim=-1))
        updated_edge = self.output_norm(fused_edge + edge_delta)

        return updated_edge, diagnostics

    def _apply_graph_mode(self, physical: Tensor, information: Tensor) -> tuple[Tensor, Tensor]:
        if self.config.graph_mode == "physical_only":
            return physical, torch.zeros_like(information)
        if self.config.graph_mode == "information_only":
            return torch.zeros_like(physical), information
        return physical, information


class DualGraphMessagePassing(nn.Module):
    """One PI-JWM v8 edge-update and node-aggregation message passing layer."""

    def __init__(self, config: V8GraphBlockConfig):
        super().__init__()
        _validate_config(config)
        hidden = config.hidden_dim
        self.edge_update = EdgeUpdateBlock(config)
        self.message_project = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.node_update = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.output_norm = nn.LayerNorm(hidden)

    def forward(
        self,
        node_state: Tensor,
        physical_edge_state: Tensor,
        info_edge_state: Tensor,
        action_state: Tensor,
        edge_src_idx: Tensor,
        edge_dst_idx: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        updated_edge, diagnostics = self.edge_update(
            node_state=node_state,
            physical_edge_state=physical_edge_state,
            info_edge_state=info_edge_state,
            action_state=action_state,
            edge_src_idx=edge_src_idx,
            edge_dst_idx=edge_dst_idx,
        )

        edge_dst_idx = edge_dst_idx.to(device=node_state.device, dtype=torch.long)
        message = self.message_project(updated_edge)
        aggregated = torch.zeros_like(node_state)
        aggregated.index_add_(1, edge_dst_idx, message)

        node_count = node_state.shape[1]
        in_degree = torch.bincount(edge_dst_idx, minlength=node_count).to(device=node_state.device)
        aggregated = aggregated / in_degree.clamp_min(1).to(dtype=node_state.dtype).view(1, -1, 1)

        node_delta = self.node_update(torch.cat([node_state, aggregated], dim=-1))
        updated_node = self.output_norm(node_state + node_delta)
        diagnostics = {**diagnostics, "node_in_degree": in_degree}

        return updated_node, updated_edge, diagnostics


def _validate_config(config: V8GraphBlockConfig) -> None:
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if config.graph_mode not in {"dual", "physical_only", "information_only"}:
        raise ValueError("graph_mode must be one of: dual, physical_only, information_only")
    if config.fusion_mode not in {"gated", "cross_attention"}:
        raise ValueError("fusion_mode must be one of: gated, cross_attention")
    if config.fusion_num_heads <= 0:
        raise ValueError("fusion_num_heads must be positive")


def _validate_graph_inputs(
    node_state: Tensor,
    physical_edge_state: Tensor,
    info_edge_state: Tensor,
    action_state: Tensor,
    edge_src_idx: Tensor,
    edge_dst_idx: Tensor,
) -> None:
    if node_state.ndim != 3:
        raise ValueError("node_state must have shape (batch, nodes, hidden)")
    if physical_edge_state.shape != info_edge_state.shape or physical_edge_state.shape != action_state.shape:
        raise ValueError("edge states must share shape (batch, edges, hidden)")
    if physical_edge_state.ndim != 3:
        raise ValueError("edge states must have shape (batch, edges, hidden)")
    if node_state.shape[0] != physical_edge_state.shape[0]:
        raise ValueError("node and edge batch sizes must match")
    if node_state.shape[-1] != physical_edge_state.shape[-1]:
        raise ValueError("node and edge hidden dimensions must match")
    num_edges = physical_edge_state.shape[1]
    if edge_src_idx.shape != (num_edges,) or edge_dst_idx.shape != (num_edges,):
        raise ValueError("edge index tensors must have shape (edges,)")
    if edge_src_idx.numel() == 0:
        return
    num_nodes = node_state.shape[1]
    if int(edge_src_idx.min()) < 0 or int(edge_dst_idx.min()) < 0:
        raise ValueError("edge indices must be non-negative")
    if int(edge_src_idx.max()) >= num_nodes or int(edge_dst_idx.max()) >= num_nodes:
        raise ValueError("edge indices must be smaller than the number of nodes")
