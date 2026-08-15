"""Minimal PI-JWM v6 dual-graph rollout module.

This module defines the first reusable PI-JWM v6 skeleton: it separates
physical-edge dynamics from information-edge dynamics, fuses them with future
actions, and rolls out future node, link, and task states.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class V6DualGraphConfig:
    node_dim: int
    physical_edge_dim: int
    info_edge_dim: int
    action_dim: int
    task_dim: int
    hidden_dim: int = 64
    horizon: int = 1
    graph_mode: str = "dual"
    fusion_mode: str = "concat"
    fusion_num_heads: int = 4
    return_fusion_diagnostics: bool = False
    rate_head_mode: str = "direct"
    rate_gate_temperature: float = 1.0
    rate_gate_floor: float = 0.0
    active_rate_auxiliary: bool = False


@dataclass(frozen=True)
class V6DualGraphBatch:
    node_history: Tensor
    physical_edge_history: Tensor
    info_edge_history: Tensor
    action_history: Tensor
    future_actions: Tensor
    task_history: Tensor
    link_rate_baseline: Tensor | None = None


class V6DualGraphRollout(nn.Module):
    """Action-conditioned physical-information dual-graph rollout skeleton."""

    def __init__(self, config: V6DualGraphConfig):
        super().__init__()
        if config.graph_mode not in {"dual", "physical_only", "information_only"}:
            raise ValueError("graph_mode must be one of: dual, physical_only, information_only")
        if config.fusion_mode not in {"concat", "gated", "cross_attention", "hybrid_attention"}:
            raise ValueError("fusion_mode must be one of: concat, gated, cross_attention, hybrid_attention")
        if config.rate_head_mode not in {"direct", "activity_gated", "residual_activity_gated"}:
            raise ValueError("rate_head_mode must be one of: direct, activity_gated, residual_activity_gated")
        if config.rate_gate_temperature <= 0.0:
            raise ValueError("rate_gate_temperature must be positive")
        if not 0.0 <= config.rate_gate_floor < 1.0:
            raise ValueError("rate_gate_floor must satisfy 0.0 <= floor < 1.0")
        self.config = config
        hidden = config.hidden_dim

        self.node_encoder = _mlp(config.node_dim, hidden)
        self.physical_edge_encoder = _mlp(config.physical_edge_dim, hidden)
        self.info_edge_encoder = _mlp(config.info_edge_dim, hidden)
        self.action_encoder = _mlp(config.action_dim, hidden)
        self.task_encoder = _mlp(config.task_dim, hidden)

        if config.fusion_mode == "concat":
            self.edge_fusion = ConcatDualGraphFusion(hidden)
        elif config.fusion_mode == "gated":
            self.edge_fusion = GatedDualGraphFusion(hidden)
        elif config.fusion_mode == "cross_attention":
            self.edge_fusion = CrossAttentionDualGraphFusion(hidden, config.fusion_num_heads)
        else:
            self.edge_fusion = HybridAttentionDualGraphFusion(hidden, config.fusion_num_heads)
        self.edge_rollout = nn.GRUCell(hidden * 2, hidden)
        self.node_rollout = nn.GRUCell(hidden * 2, hidden)
        self.task_rollout = nn.GRUCell(hidden * 3, hidden)

        self.node_head = nn.Linear(hidden, config.node_dim)
        self.link_activity_head = nn.Linear(hidden, 1)
        self.link_rate_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.link_rate_residual_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.link_active_rate_aux_head = (
            nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            if config.active_rate_auxiliary
            else None
        )
        self.task_head = nn.Linear(hidden, config.task_dim)

    def forward(self, batch: V6DualGraphBatch) -> dict[str, Tensor]:
        self._validate_batch(batch)

        node_state = self._encode_sequence(batch.node_history, self.node_encoder).mean(dim=1)
        physical_edge_state = self._encode_sequence(
            batch.physical_edge_history, self.physical_edge_encoder
        ).mean(dim=1)
        info_edge_state = self._encode_sequence(batch.info_edge_history, self.info_edge_encoder).mean(dim=1)
        action_state = self._encode_sequence(batch.action_history, self.action_encoder).mean(dim=1)
        task_state = self._encode_sequence(batch.task_history, self.task_encoder).mean(dim=1)

        if self.config.graph_mode == "physical_only":
            info_edge_state = torch.zeros_like(info_edge_state)
        elif self.config.graph_mode == "information_only":
            physical_edge_state = torch.zeros_like(physical_edge_state)
        edge_state, fusion_diagnostics = self.edge_fusion(
            physical_edge_state,
            info_edge_state,
            action_state,
        )
        global_edge_state = edge_state.mean(dim=1)

        node_predictions = []
        activity_logits = []
        rate_predictions = []
        rate_value_predictions = []
        activity_prob_predictions = []
        rate_base_predictions = []
        rate_residual_predictions = []
        activity_gate_predictions = []
        active_rate_aux_predictions = []
        task_predictions = []

        for step in range(self.config.horizon):
            future_action = self.action_encoder(batch.future_actions[:, step])
            edge_state = self.edge_rollout(
                torch.cat([edge_state, future_action], dim=-1).reshape(-1, self.config.hidden_dim * 2),
                edge_state.reshape(-1, self.config.hidden_dim),
            ).reshape_as(edge_state)

            global_edge_state = edge_state.mean(dim=1)
            node_condition = global_edge_state.unsqueeze(1).expand(-1, node_state.shape[1], -1)
            node_state = self.node_rollout(
                torch.cat([node_state, node_condition], dim=-1).reshape(-1, self.config.hidden_dim * 2),
                node_state.reshape(-1, self.config.hidden_dim),
            ).reshape_as(node_state)

            global_node_state = node_state.mean(dim=1)
            task_state = self.task_rollout(
                torch.cat([task_state, global_edge_state, global_node_state], dim=-1),
                task_state,
            )

            activity_logit = self.link_activity_head(edge_state)
            rate_value = self.link_rate_head(edge_state)
            if self.config.rate_head_mode == "activity_gated":
                activity_prob = torch.sigmoid(activity_logit)
                rate_prediction = activity_prob * rate_value
                rate_value_predictions.append(rate_value)
                activity_prob_predictions.append(activity_prob)
            elif self.config.rate_head_mode == "residual_activity_gated":
                activity_gate = self._rate_activity_gate(activity_logit)
                rate_residual = self.link_rate_residual_head(edge_state)
                rate_prediction = rate_value + activity_gate * rate_residual
                rate_base_predictions.append(rate_value)
                rate_residual_predictions.append(rate_residual)
                activity_gate_predictions.append(activity_gate)
            else:
                rate_prediction = rate_value

            if self.link_active_rate_aux_head is not None:
                active_rate_aux_predictions.append(self.link_active_rate_aux_head(edge_state))

            node_predictions.append(self.node_head(node_state))
            activity_logits.append(activity_logit)
            rate_predictions.append(rate_prediction)
            task_predictions.append(self.task_head(task_state))

        outputs = {
            "node": torch.stack(node_predictions, dim=1),
            "link_activity_logit": torch.stack(activity_logits, dim=1),
            "link_rate": torch.stack(rate_predictions, dim=1),
            "task": torch.stack(task_predictions, dim=1),
        }
        if self.config.rate_head_mode == "activity_gated":
            outputs["link_rate_value"] = torch.stack(rate_value_predictions, dim=1)
            outputs["link_activity_prob"] = torch.stack(activity_prob_predictions, dim=1)
        elif self.config.rate_head_mode == "residual_activity_gated":
            outputs["link_rate_base"] = torch.stack(rate_base_predictions, dim=1)
            outputs["link_rate_residual"] = torch.stack(rate_residual_predictions, dim=1)
            outputs["link_activity_gate"] = torch.stack(activity_gate_predictions, dim=1)
        if self.link_active_rate_aux_head is not None:
            outputs["link_active_rate_aux"] = torch.stack(active_rate_aux_predictions, dim=1)
        if self.config.return_fusion_diagnostics:
            outputs.update(fusion_diagnostics)
        return outputs

    def _validate_batch(self, batch: V6DualGraphBatch) -> None:
        if batch.future_actions.shape[1] != self.config.horizon:
            raise ValueError("future_actions horizon does not match config.horizon")

    def _rate_activity_gate(self, activity_logit: Tensor) -> Tensor:
        gate = torch.sigmoid(activity_logit / self.config.rate_gate_temperature)
        if self.config.rate_gate_floor > 0.0:
            gate = self.config.rate_gate_floor + (1.0 - self.config.rate_gate_floor) * gate
        return gate

    @staticmethod
    def _encode_sequence(values: Tensor, encoder: nn.Module) -> Tensor:
        leading = values.shape[:-1]
        encoded = encoder(values.reshape(-1, values.shape[-1]))
        return encoded.reshape(*leading, encoded.shape[-1])


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.LayerNorm(hidden_dim),
    )


class ConcatDualGraphFusion(nn.Module):
    """v6 default fusion: concatenate physical, information, and action states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, physical: Tensor, information: Tensor, action: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        return self.project(torch.cat([physical, information, action], dim=-1)), {}


class GatedDualGraphFusion(nn.Module):
    """Per-edge modality gate for physical, information, and action states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, physical: Tensor, information: Tensor, action: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        tokens = torch.stack([physical, information, action], dim=-2)
        gate_logits = self.gate(torch.cat([physical, information, action], dim=-1))
        weights = torch.softmax(gate_logits, dim=-1)
        fused = (tokens * weights.unsqueeze(-1)).sum(dim=-2)
        return self.project(fused), {"fusion_weights": weights}


class CrossAttentionDualGraphFusion(nn.Module):
    """Per-edge modality-token attention over physical, information, and action states."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by fusion_num_heads")
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, physical: Tensor, information: Tensor, action: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        batch_size, num_edges, _ = physical.shape
        tokens = torch.stack([physical, information, action], dim=2).reshape(batch_size * num_edges, 3, -1)
        attended, attention = self.attention(tokens, tokens, tokens, need_weights=True, average_attn_weights=True)
        attended = self.norm(attended + tokens)
        fused = self.project(attended.mean(dim=1).reshape(batch_size, num_edges, -1))
        return fused, {"fusion_attention": attention.reshape(batch_size, num_edges, 3, 3)}


class HybridAttentionDualGraphFusion(nn.Module):
    """Attention fusion with a residual concat path for sparse rollout targets."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by fusion_num_heads")
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn_project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.concat_project = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, physical: Tensor, information: Tensor, action: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        batch_size, num_edges, _ = physical.shape
        tokens = torch.stack([physical, information, action], dim=2).reshape(batch_size * num_edges, 3, -1)
        attended, attention = self.attention(tokens, tokens, tokens, need_weights=True, average_attn_weights=True)
        attended = self.attn_norm(attended + tokens)
        attention_path = self.attn_project(attended.mean(dim=1).reshape(batch_size, num_edges, -1))
        concat_input = torch.cat([physical, information, action], dim=-1)
        concat_path = self.concat_project(concat_input)
        gate = self.residual_gate(torch.cat([concat_input, attention_path], dim=-1))
        fused = self.output_norm(gate * attention_path + (1.0 - gate) * concat_path)
        return fused, {
            "fusion_attention": attention.reshape(batch_size, num_edges, 3, 3),
            "fusion_residual_gate": gate,
        }
