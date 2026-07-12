"""PI-JWM v8 full-world-model entry point.

This module keeps the v6 batch/output contract while replacing the implicit
edge-to-node coupling with explicit v8 dual-graph message passing.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pi_jwm.v6_dual_graph import V6DualGraphBatch
from pi_jwm.v8_graph_blocks import DualGraphMessagePassing, V8GraphBlockConfig


@dataclass(frozen=True)
class V8FullWorldModelConfig:
    node_dim: int
    physical_edge_dim: int
    info_edge_dim: int
    action_dim: int
    task_dim: int
    edge_src_idx: Tensor
    edge_dst_idx: Tensor
    hidden_dim: int = 64
    horizon: int = 1
    graph_mode: str = "dual"
    fusion_mode: str = "gated"
    fusion_num_heads: int = 4
    active_rate_auxiliary: bool = False
    active_rate_head_mode: str = "mlp"
    num_rate_experts: int = 4
    rate_output_mode: str = "direct"
    history_encoder: str = "mean"
    latent_transition_mode: str = "message_passing"
    return_message_diagnostics: bool = False
    activity_memory_dim: int = 0
    activity_memory_routing: str = "none"
    adaptive_edge_context: str = "none"
    adaptive_edge_topk: int = 8


class V8FullWorldModelRollout(nn.Module):
    """Action-conditioned PI-JWM rollout with explicit dual-graph message passing."""

    def __init__(self, config: V8FullWorldModelConfig):
        super().__init__()
        if config.horizon <= 0:
            raise ValueError("horizon must be positive")
        if config.history_encoder not in {"mean", "temporal_conv", "stgcn_light", "stgcn_full"}:
            raise ValueError("history_encoder must be one of: mean, temporal_conv, stgcn_light, stgcn_full")
        if config.latent_transition_mode not in {"message_passing", "recurrent"}:
            raise ValueError("latent_transition_mode must be one of: message_passing, recurrent")
        if config.active_rate_head_mode not in {"mlp", "moe"}:
            raise ValueError("active_rate_head_mode must be one of: mlp, moe")
        if config.rate_output_mode not in {"direct", "hurdle_soft", "hurdle_dual", "hurdle_mass"}:
            raise ValueError("rate_output_mode must be one of: direct, hurdle_soft, hurdle_dual, hurdle_mass")
        if config.activity_memory_dim < 0:
            raise ValueError("activity_memory_dim must be non-negative")
        if config.activity_memory_routing not in {"none", "activity_only"}:
            raise ValueError("activity_memory_routing must be one of: none, activity_only")
        if config.activity_memory_routing == "activity_only" and config.activity_memory_dim <= 0:
            raise ValueError("activity_memory_dim must be positive when activity_memory_routing='activity_only'")
        if config.activity_memory_dim >= config.info_edge_dim:
            raise ValueError("activity_memory_dim must be smaller than info_edge_dim")
        if config.adaptive_edge_context not in {"none", "sparse_attention"}:
            raise ValueError("adaptive_edge_context must be one of: none, sparse_attention")
        if config.adaptive_edge_topk <= 0:
            raise ValueError("adaptive_edge_topk must be positive")
        if config.active_rate_head_mode == "moe" and config.num_rate_experts <= 1:
            raise ValueError("num_rate_experts must be greater than 1 when active_rate_head_mode='moe'")
        self.config = config
        hidden = config.hidden_dim
        base_info_edge_dim = config.info_edge_dim - config.activity_memory_dim

        self.node_encoder = _mlp(config.node_dim, hidden)
        self.physical_edge_encoder = _mlp(config.physical_edge_dim, hidden)
        self.info_edge_encoder = _mlp(base_info_edge_dim, hidden)
        self.activity_memory_encoder = _mlp(config.activity_memory_dim, hidden) if config.activity_memory_dim > 0 else None
        self.action_encoder = _mlp(config.action_dim, hidden)
        self.task_encoder = _mlp(config.task_dim, hidden)
        uses_temporal_history = config.history_encoder in {"temporal_conv", "stgcn_light", "stgcn_full"}
        self.node_temporal_encoder = _temporal_conv(hidden) if uses_temporal_history else None
        self.physical_edge_temporal_encoder = (
            _temporal_conv(hidden) if uses_temporal_history else None
        )
        self.info_edge_temporal_encoder = _temporal_conv(hidden) if uses_temporal_history else None
        self.action_temporal_encoder = _temporal_conv(hidden) if uses_temporal_history else None
        self.task_temporal_encoder = _temporal_conv(hidden) if uses_temporal_history else None

        graph_config = V8GraphBlockConfig(
            hidden_dim=hidden,
            graph_mode=config.graph_mode,
            fusion_mode=config.fusion_mode,
            fusion_num_heads=config.fusion_num_heads,
        )
        self.history_message_passing = (
            DualGraphMessagePassing(graph_config) if config.history_encoder == "stgcn_light" else None
        )
        self.node_stgcn_full_encoder = (
            STGCNFullHistoryEncoder(hidden, entity_kind="node") if config.history_encoder == "stgcn_full" else None
        )
        self.edge_stgcn_full_encoder = (
            STGCNFullHistoryEncoder(hidden, entity_kind="edge") if config.history_encoder == "stgcn_full" else None
        )
        self.initial_message_passing = DualGraphMessagePassing(graph_config)
        self.rollout_message_passing = DualGraphMessagePassing(graph_config)
        self.node_latent_rollout = (
            nn.GRUCell(hidden, hidden) if config.latent_transition_mode == "recurrent" else None
        )
        self.edge_latent_rollout = (
            nn.GRUCell(hidden, hidden) if config.latent_transition_mode == "recurrent" else None
        )
        self.adaptive_edge_context = (
            SparseAdaptiveEdgeContext(hidden, config.adaptive_edge_topk)
            if config.adaptive_edge_context == "sparse_attention"
            else None
        )
        self.task_rollout = nn.GRUCell(hidden * 3, hidden)

        self.node_head = nn.Linear(hidden, config.node_dim)
        activity_head_dim = hidden * 2 if config.activity_memory_routing == "activity_only" else hidden
        self.link_activity_head = nn.Linear(activity_head_dim, 1)
        self.link_rate_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.link_positive_rate_head = (
            nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            if config.rate_output_mode in {"hurdle_soft", "hurdle_dual", "hurdle_mass"}
            else None
        )
        self.link_active_mass_total_head = (
            nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            if config.rate_output_mode == "hurdle_mass"
            else None
        )
        self.link_active_mass_alloc_head = (
            nn.Sequential(nn.Linear(hidden * 2 + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            if config.rate_output_mode == "hurdle_mass"
            else None
        )
        self.link_active_rate_aux_head = self._build_active_rate_aux_head(config)
        self.task_head = nn.Linear(hidden, config.task_dim)

        self.register_buffer("edge_src_idx", config.edge_src_idx.detach().clone().to(dtype=torch.long))
        self.register_buffer("edge_dst_idx", config.edge_dst_idx.detach().clone().to(dtype=torch.long))

    def forward(self, batch: V6DualGraphBatch) -> dict[str, Tensor]:
        self._validate_batch(batch)

        if self.config.history_encoder == "stgcn_full":
            node_state, physical_edge_state, info_edge_state, action_state, task_state = (
                self._encode_stgcn_full_history(batch)
            )
            activity_memory_state = self._encode_activity_memory_history(batch)
        elif self.config.history_encoder == "stgcn_light":
            node_state, physical_edge_state, info_edge_state, action_state, task_state = (
                self._encode_stgcn_light_history(batch)
            )
            activity_memory_state = self._encode_activity_memory_history(batch)
        else:
            node_state = self._encode_history(batch.node_history, self.node_encoder, self.node_temporal_encoder)
            physical_edge_state = self._encode_history(
                batch.physical_edge_history,
                self.physical_edge_encoder,
                self.physical_edge_temporal_encoder,
            )
            info_edge_history = self._base_info_edge_history(batch.info_edge_history)
            info_edge_state = self._encode_history(
                info_edge_history,
                self.info_edge_encoder,
                self.info_edge_temporal_encoder,
            )
            activity_memory_state = self._encode_activity_memory_history(batch)
            action_state = self._encode_history(batch.action_history, self.action_encoder, self.action_temporal_encoder)
            task_state = self._encode_history(batch.task_history, self.task_encoder, self.task_temporal_encoder)

        node_state, edge_state, diagnostics = self.initial_message_passing(
            node_state=node_state,
            physical_edge_state=physical_edge_state,
            info_edge_state=info_edge_state,
            action_state=action_state,
            edge_src_idx=self.edge_src_idx,
            edge_dst_idx=self.edge_dst_idx,
        )

        node_predictions = []
        activity_logits = []
        rate_predictions = []
        positive_rate_predictions = []
        active_mass_rate_predictions = []
        active_mass_total_predictions = []
        active_mass_alloc_logits = []
        active_rate_aux_predictions = []
        rate_expert_weight_predictions = []
        task_predictions = []
        step_diagnostics = diagnostics
        adaptive_context_attention = None

        for step in range(self.config.horizon):
            future_action = self.action_encoder(batch.future_actions[:, step])
            candidate_node_state, candidate_edge_state, step_diagnostics = self.rollout_message_passing(
                node_state=node_state,
                physical_edge_state=edge_state,
                info_edge_state=edge_state,
                action_state=future_action,
                edge_src_idx=self.edge_src_idx,
                edge_dst_idx=self.edge_dst_idx,
            )
            node_state, edge_state = self._rollout_latent_transition(
                node_state=node_state,
                edge_state=edge_state,
                candidate_node_state=candidate_node_state,
                candidate_edge_state=candidate_edge_state,
            )
            if self.adaptive_edge_context is not None:
                edge_state, adaptive_context_attention = self.adaptive_edge_context(edge_state)

            global_edge_state = edge_state.mean(dim=1)
            global_node_state = node_state.mean(dim=1)
            task_state = self.task_rollout(
                torch.cat([task_state, global_edge_state, global_node_state], dim=-1),
                task_state,
            )

            node_predictions.append(self.node_head(node_state))
            activity_input = self._activity_head_input(edge_state, activity_memory_state)
            activity_logit = self.link_activity_head(activity_input)
            activity_logits.append(activity_logit)
            if self.config.rate_output_mode == "hurdle_soft":
                positive_rate = self.link_positive_rate_head(edge_state)
                positive_rate_predictions.append(positive_rate)
                rate_predictions.append(torch.sigmoid(activity_logit) * positive_rate)
            elif self.config.rate_output_mode == "hurdle_dual":
                positive_rate = self.link_positive_rate_head(edge_state)
                positive_rate_predictions.append(positive_rate)
                rate_predictions.append(self.link_rate_head(edge_state))
            elif self.config.rate_output_mode == "hurdle_mass":
                positive_rate = self.link_positive_rate_head(edge_state)
                positive_rate_predictions.append(positive_rate)
                mass_rate, mass_total, alloc_logit = self._predict_active_mass_rate(
                    edge_state=edge_state,
                    node_state=node_state,
                    task_state=task_state,
                    activity_logit=activity_logit,
                    positive_rate=positive_rate,
                )
                active_mass_rate_predictions.append(mass_rate)
                active_mass_total_predictions.append(mass_total)
                active_mass_alloc_logits.append(alloc_logit)
                rate_predictions.append(mass_rate)
            else:
                rate_predictions.append(self.link_rate_head(edge_state))
            if self.link_active_rate_aux_head is not None:
                active_rate_aux, rate_expert_weights = self._predict_active_rate_aux(edge_state)
                active_rate_aux_predictions.append(active_rate_aux)
                if rate_expert_weights is not None:
                    rate_expert_weight_predictions.append(rate_expert_weights)
            task_predictions.append(self.task_head(task_state))

        outputs = {
            "node": torch.stack(node_predictions, dim=1),
            "link_activity_logit": torch.stack(activity_logits, dim=1),
            "link_rate": torch.stack(rate_predictions, dim=1),
            "task": torch.stack(task_predictions, dim=1),
        }
        if self.config.rate_output_mode in {"hurdle_soft", "hurdle_dual", "hurdle_mass"}:
            outputs["link_positive_rate"] = torch.stack(positive_rate_predictions, dim=1)
        if self.config.rate_output_mode == "hurdle_dual":
            activity_prob = torch.sigmoid(outputs["link_activity_logit"])
            outputs["link_hurdle_rate"] = activity_prob * outputs["link_positive_rate"]
        if self.config.rate_output_mode == "hurdle_mass":
            outputs["link_active_mass_rate"] = torch.stack(active_mass_rate_predictions, dim=1)
            outputs["link_active_mass_total"] = torch.stack(active_mass_total_predictions, dim=1)
            outputs["link_active_mass_alloc_logit"] = torch.stack(active_mass_alloc_logits, dim=1)
        if self.link_active_rate_aux_head is not None:
            outputs["link_active_rate_aux"] = torch.stack(active_rate_aux_predictions, dim=1)
            if rate_expert_weight_predictions:
                outputs["rate_expert_weights"] = torch.stack(rate_expert_weight_predictions, dim=1)
        if self.config.return_message_diagnostics:
            outputs["message_node_in_degree"] = step_diagnostics["node_in_degree"]
            if "edge_fusion_weights" in step_diagnostics:
                outputs["message_edge_fusion_weights"] = step_diagnostics["edge_fusion_weights"]
            if "edge_fusion_attention" in step_diagnostics:
                outputs["message_edge_fusion_attention"] = step_diagnostics["edge_fusion_attention"]
            if adaptive_context_attention is not None:
                outputs["adaptive_edge_context_attention"] = adaptive_context_attention
        return outputs

    def _build_active_rate_aux_head(self, config: V8FullWorldModelConfig) -> nn.Module | None:
        if not config.active_rate_auxiliary:
            return None
        if config.active_rate_head_mode == "mlp":
            return nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU(), nn.Linear(config.hidden_dim, 1))
        return MixtureOfExpertsRateHead(config.hidden_dim, config.num_rate_experts)

    def _predict_active_rate_aux(self, edge_state: Tensor) -> tuple[Tensor, Tensor | None]:
        if isinstance(self.link_active_rate_aux_head, MixtureOfExpertsRateHead):
            return self.link_active_rate_aux_head(edge_state)
        return self.link_active_rate_aux_head(edge_state), None

    def _predict_active_mass_rate(
        self,
        edge_state: Tensor,
        node_state: Tensor,
        task_state: Tensor,
        activity_logit: Tensor,
        positive_rate: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        global_edge_state = edge_state.mean(dim=1)
        global_node_state = node_state.mean(dim=1)
        mass_context = torch.cat([global_edge_state, global_node_state, task_state], dim=-1)
        mass_total = self.link_active_mass_total_head(mass_context).unsqueeze(1)
        alloc_input = torch.cat([edge_state, positive_rate, activity_logit, task_state.unsqueeze(1).expand_as(edge_state)], dim=-1)
        alloc_logit = self.link_active_mass_alloc_head(alloc_input)
        activity_prob = torch.sigmoid(activity_logit)
        positive_score = nn.functional.softplus(positive_rate)
        alloc_weight = torch.softmax(alloc_logit + torch.log(activity_prob.clamp_min(1e-6) * positive_score + 1e-6), dim=1)
        return mass_total * alloc_weight, mass_total, alloc_logit

    def _encode_stgcn_full_history(
        self,
        batch: V6DualGraphBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        node_sequence = self._encode_sequence(batch.node_history, self.node_encoder)
        physical_edge_sequence = self._encode_sequence(batch.physical_edge_history, self.physical_edge_encoder)
        info_edge_sequence = self._encode_sequence(self._base_info_edge_history(batch.info_edge_history), self.info_edge_encoder)
        action_sequence = self._encode_sequence(batch.action_history, self.action_encoder)
        task_sequence = self._encode_sequence(batch.task_history, self.task_encoder)

        node_adjacency = self._normalized_node_adjacency(
            node_count=node_sequence.shape[2],
            device=node_sequence.device,
            dtype=node_sequence.dtype,
        )
        edge_adjacency = self._normalized_edge_adjacency(
            edge_count=physical_edge_sequence.shape[2],
            device=physical_edge_sequence.device,
            dtype=physical_edge_sequence.dtype,
        )
        node_state = self.node_stgcn_full_encoder(node_sequence, node_adjacency)
        physical_edge_state = self.edge_stgcn_full_encoder(physical_edge_sequence, edge_adjacency)
        info_edge_state = self.edge_stgcn_full_encoder(info_edge_sequence, edge_adjacency)
        action_state = self.edge_stgcn_full_encoder(action_sequence, edge_adjacency)
        task_state = self._pool_encoded_history(task_sequence, self.task_temporal_encoder)
        return node_state, physical_edge_state, info_edge_state, action_state, task_state

    def _encode_stgcn_light_history(
        self,
        batch: V6DualGraphBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        node_sequence = self._encode_sequence(batch.node_history, self.node_encoder)
        physical_edge_sequence = self._encode_sequence(batch.physical_edge_history, self.physical_edge_encoder)
        info_edge_sequence = self._encode_sequence(self._base_info_edge_history(batch.info_edge_history), self.info_edge_encoder)
        action_sequence = self._encode_sequence(batch.action_history, self.action_encoder)
        task_sequence = self._encode_sequence(batch.task_history, self.task_encoder)

        graph_node_steps = []
        graph_edge_steps = []
        for step in range(node_sequence.shape[1]):
            node_step, edge_step, _ = self.history_message_passing(
                node_state=node_sequence[:, step],
                physical_edge_state=physical_edge_sequence[:, step],
                info_edge_state=info_edge_sequence[:, step],
                action_state=action_sequence[:, step],
                edge_src_idx=self.edge_src_idx,
                edge_dst_idx=self.edge_dst_idx,
            )
            graph_node_steps.append(node_step)
            graph_edge_steps.append(edge_step)

        graph_node_sequence = torch.stack(graph_node_steps, dim=1)
        graph_edge_sequence = torch.stack(graph_edge_steps, dim=1)
        node_state = self._pool_encoded_history(graph_node_sequence, self.node_temporal_encoder)
        physical_edge_state = self._pool_encoded_history(
            graph_edge_sequence,
            self.physical_edge_temporal_encoder,
        )
        info_edge_state = self._pool_encoded_history(graph_edge_sequence, self.info_edge_temporal_encoder)
        action_state = self._pool_encoded_history(action_sequence, self.action_temporal_encoder)
        task_state = self._pool_encoded_history(task_sequence, self.task_temporal_encoder)
        return node_state, physical_edge_state, info_edge_state, action_state, task_state

    def _rollout_latent_transition(
        self,
        node_state: Tensor,
        edge_state: Tensor,
        candidate_node_state: Tensor,
        candidate_edge_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.config.latent_transition_mode == "message_passing":
            return candidate_node_state, candidate_edge_state
        next_node_state = self._apply_gru_entity_rollout(
            self.node_latent_rollout,
            candidate_node_state,
            node_state,
        )
        next_edge_state = self._apply_gru_entity_rollout(
            self.edge_latent_rollout,
            candidate_edge_state,
            edge_state,
        )
        return next_node_state, next_edge_state

    @staticmethod
    def _apply_gru_entity_rollout(gru_cell: nn.GRUCell, candidate_state: Tensor, previous_state: Tensor) -> Tensor:
        entity_shape = candidate_state.shape[:-1]
        hidden = candidate_state.shape[-1]
        updated = gru_cell(
            candidate_state.reshape(-1, hidden),
            previous_state.reshape(-1, hidden),
        )
        return updated.reshape(*entity_shape, hidden)

    def _validate_batch(self, batch: V6DualGraphBatch) -> None:
        if batch.future_actions.shape[1] != self.config.horizon:
            raise ValueError("future_actions horizon does not match config.horizon")
        if batch.node_history.shape[2] <= int(torch.max(self.edge_src_idx).item()):
            raise ValueError("edge_src_idx refers to a node outside node_history")
        if batch.node_history.shape[2] <= int(torch.max(self.edge_dst_idx).item()):
            raise ValueError("edge_dst_idx refers to a node outside node_history")
        if batch.physical_edge_history.shape[2] != self.edge_src_idx.numel():
            raise ValueError("edge index count must match edge history count")
        if batch.info_edge_history.shape[-1] != self.config.info_edge_dim:
            raise ValueError("info_edge_history feature dimension does not match config.info_edge_dim")

    @staticmethod
    def _encode_sequence(values: Tensor, encoder: nn.Module) -> Tensor:
        leading = values.shape[:-1]
        encoded = encoder(values.reshape(-1, values.shape[-1]))
        return encoded.reshape(*leading, encoded.shape[-1])

    def _encode_history(
        self,
        values: Tensor,
        encoder: nn.Module,
        temporal_encoder: nn.Module | None,
    ) -> Tensor:
        encoded = self._encode_sequence(values, encoder)
        return self._pool_encoded_history(encoded, temporal_encoder)

    def _pool_encoded_history(
        self,
        encoded: Tensor,
        temporal_encoder: nn.Module | None,
    ) -> Tensor:
        if temporal_encoder is None:
            return encoded.mean(dim=1)
        batch_size, history = encoded.shape[:2]
        entity_shape = encoded.shape[2:-1]
        hidden = encoded.shape[-1]
        flattened = encoded.reshape(batch_size, history, -1, hidden).permute(0, 2, 3, 1)
        temporal_input = flattened.reshape(batch_size * flattened.shape[1], hidden, history)
        temporal_output = temporal_encoder(temporal_input)
        last_state = temporal_output[..., -1].reshape(batch_size, *entity_shape, hidden)
        return last_state

    def _base_info_edge_history(self, info_edge_history: Tensor) -> Tensor:
        if self.config.activity_memory_dim <= 0:
            return info_edge_history
        return info_edge_history[..., : -self.config.activity_memory_dim]

    def _activity_memory_history(self, info_edge_history: Tensor) -> Tensor:
        if self.config.activity_memory_dim <= 0:
            raise ValueError("activity_memory_dim must be positive")
        return info_edge_history[..., -self.config.activity_memory_dim :]

    def _encode_activity_memory_history(self, batch: V6DualGraphBatch) -> Tensor | None:
        if self.config.activity_memory_routing != "activity_only":
            return None
        if self.activity_memory_encoder is None:
            return None
        return self._encode_history(
            self._activity_memory_history(batch.info_edge_history),
            self.activity_memory_encoder,
            self.info_edge_temporal_encoder,
        )

    def _activity_head_input(self, edge_state: Tensor, activity_memory_state: Tensor | None) -> Tensor:
        if self.config.activity_memory_routing != "activity_only":
            return edge_state
        if activity_memory_state is None:
            raise ValueError("activity_memory_state is required for activity_only routing")
        return torch.cat([edge_state, activity_memory_state], dim=-1)

    def _normalized_node_adjacency(self, node_count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        adjacency = torch.eye(node_count, device=device, dtype=dtype)
        adjacency[self.edge_src_idx, self.edge_dst_idx] = 1.0
        adjacency[self.edge_dst_idx, self.edge_src_idx] = 1.0
        return _row_normalize_adjacency(adjacency)

    def _normalized_edge_adjacency(self, edge_count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        src = self.edge_src_idx
        dst = self.edge_dst_idx
        shares_endpoint = (
            (src[:, None] == src[None, :])
            | (src[:, None] == dst[None, :])
            | (dst[:, None] == src[None, :])
            | (dst[:, None] == dst[None, :])
        )
        adjacency = shares_endpoint.to(device=device, dtype=dtype)
        adjacency = adjacency[:edge_count, :edge_count]
        adjacency.fill_diagonal_(1.0)
        return _row_normalize_adjacency(adjacency)


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.LayerNorm(hidden_dim),
    )


def _temporal_conv(hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
    )


def _row_normalize_adjacency(adjacency: Tensor) -> Tensor:
    degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


class TemporalGatedConv(nn.Module):
    """Temporal convolution with a GLU-style gate over each entity history."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.gate = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, sequence: Tensor) -> Tensor:
        batch_size, history, entity_count, hidden = sequence.shape
        values = sequence.permute(0, 2, 3, 1).reshape(batch_size * entity_count, hidden, history)
        gated = torch.tanh(self.conv(values)) * torch.sigmoid(self.gate(values))
        return gated.reshape(batch_size, entity_count, hidden, history).permute(0, 3, 1, 2)


class GraphConvolution(nn.Module):
    """Simple first-order graph convolution over entity states at each time step."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, sequence: Tensor, adjacency: Tensor) -> Tensor:
        neighbor_state = torch.einsum("ij,btjh->btih", adjacency, sequence)
        return self.self_projection(sequence) + self.neighbor_projection(neighbor_state)


class STGCNFullHistoryEncoder(nn.Module):
    """Temporal-gated -> graph-conv -> temporal-gated history encoder."""

    def __init__(self, hidden_dim: int, entity_kind: str):
        super().__init__()
        self.entity_kind = entity_kind
        self.temporal_in = TemporalGatedConv(hidden_dim)
        self.graph_conv = GraphConvolution(hidden_dim)
        self.temporal_out = TemporalGatedConv(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequence: Tensor, adjacency: Tensor) -> Tensor:
        residual = sequence
        encoded = self.temporal_in(sequence)
        encoded = torch.relu(self.graph_conv(encoded, adjacency))
        encoded = self.temporal_out(encoded)
        encoded = self.norm(encoded + residual)
        return encoded[:, -1]


class SparseAdaptiveEdgeContext(nn.Module):
    """Top-k learned edge-to-edge context branch for sparse candidate graphs."""

    def __init__(self, hidden_dim: int, topk: int):
        super().__init__()
        if topk <= 0:
            raise ValueError("topk must be positive")
        self.topk = int(topk)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, edge_state: Tensor) -> tuple[Tensor, Tensor]:
        if edge_state.ndim != 3:
            raise ValueError("edge_state must have shape (batch, edges, hidden)")
        hidden = edge_state.shape[-1]
        edge_count = edge_state.shape[1]
        topk = min(self.topk, edge_count)
        query = self.query(edge_state)
        key = self.key(edge_state)
        score = torch.matmul(query, key.transpose(-1, -2)) / float(hidden) ** 0.5
        top_values, top_indices = torch.topk(score, k=topk, dim=-1)
        sparse_attention = torch.softmax(top_values, dim=-1)
        attention = torch.zeros_like(score)
        attention.scatter_(-1, top_indices, sparse_attention)
        context = torch.matmul(attention, self.value(edge_state))
        updated = self.norm(edge_state + self.output(context))
        return updated, attention


class MixtureOfExpertsRateHead(nn.Module):
    """Small MoE regressor for active-link rate amplitude."""

    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        if num_experts <= 1:
            raise ValueError("num_experts must be greater than 1")
        self.experts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_experts))

    def forward(self, edge_state: Tensor) -> tuple[Tensor, Tensor]:
        expert_outputs = torch.cat([expert(edge_state) for expert in self.experts], dim=-1)
        expert_weights = torch.softmax(self.gate(edge_state), dim=-1)
        prediction = (expert_outputs * expert_weights).sum(dim=-1, keepdim=True)
        return prediction, expert_weights
