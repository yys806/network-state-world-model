"""PI-JWM v7 behavior-cloning action policy.

The policy is the first supervised `state -> action` module for PI-JWM. It
predicts future edge-level scheduler actions from historical state and action
context, using the same physical-information-action fusion options as v7
rollout models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pi_jwm.v6_dual_graph import (
    ConcatDualGraphFusion,
    CrossAttentionDualGraphFusion,
    GatedDualGraphFusion,
    HybridAttentionDualGraphFusion,
    V6DualGraphBatch,
)


@dataclass(frozen=True)
class V7ActionPolicyConfig:
    node_dim: int
    physical_edge_dim: int
    info_edge_dim: int
    action_dim: int
    task_dim: int
    hidden_dim: int = 64
    horizon: int = 3
    fusion_mode: str = "cross_attention"
    fusion_num_heads: int = 4
    return_fusion_diagnostics: bool = False
    value_mode: str = "continuous"
    max_value_bins: int = 1
    value_token_group_count: int = 0
    max_value_tokens: int = 1
    max_value_count_tokens: int = 1
    max_value_total_tokens: int = 1
    use_edge_activity_head: bool = False
    use_step_total_head: bool = False


class V7ActionPolicy(nn.Module):
    """Behavior-cloning policy that predicts future edge actions from state."""

    def __init__(self, config: V7ActionPolicyConfig):
        super().__init__()
        if config.fusion_mode not in {"concat", "gated", "cross_attention", "hybrid_attention"}:
            raise ValueError("fusion_mode must be one of: concat, gated, cross_attention, hybrid_attention")
        if getattr(config, "value_mode", "continuous") not in {
            "continuous",
            "discrete_bins",
            "coupled_tokens",
            "hierarchical_tokens",
        }:
            raise ValueError("value_mode must be one of: continuous, discrete_bins, coupled_tokens, hierarchical_tokens")
        if getattr(config, "value_mode", "continuous") == "discrete_bins" and int(getattr(config, "max_value_bins", 0)) <= 0:
            raise ValueError("max_value_bins must be positive when value_mode is discrete_bins")
        if getattr(config, "value_mode", "continuous") == "coupled_tokens":
            if int(getattr(config, "value_token_group_count", 0)) <= 0:
                raise ValueError("value_token_group_count must be positive when value_mode is coupled_tokens")
            if int(getattr(config, "max_value_tokens", 0)) <= 0:
                raise ValueError("max_value_tokens must be positive when value_mode is coupled_tokens")
        if getattr(config, "value_mode", "continuous") == "hierarchical_tokens":
            if int(getattr(config, "value_token_group_count", 0)) <= 0:
                raise ValueError("value_token_group_count must be positive when value_mode is hierarchical_tokens")
            if int(getattr(config, "max_value_count_tokens", 0)) <= 0:
                raise ValueError("max_value_count_tokens must be positive when value_mode is hierarchical_tokens")
            if int(getattr(config, "max_value_total_tokens", 0)) <= 0:
                raise ValueError("max_value_total_tokens must be positive when value_mode is hierarchical_tokens")
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

        self.rollout = nn.GRUCell(hidden * 3, hidden)
        self.action_logit_head = nn.Linear(hidden, config.action_dim)
        self.edge_logit_head = nn.Linear(hidden, 1) if bool(getattr(config, "use_edge_activity_head", False)) else None
        self.step_total_head = (
            nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))
            if bool(getattr(config, "use_step_total_head", False))
            else None
        )
        if getattr(config, "value_mode", "continuous") == "discrete_bins":
            self.action_value_bin_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, config.action_dim * int(config.max_value_bins)),
            )
        elif getattr(config, "value_mode", "continuous") == "coupled_tokens":
            self.action_value_token_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, int(config.value_token_group_count) * int(config.max_value_tokens)),
            )
        elif getattr(config, "value_mode", "continuous") == "hierarchical_tokens":
            self.action_value_count_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, int(config.value_token_group_count) * int(config.max_value_count_tokens)),
            )
            self.action_value_total_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, int(config.value_token_group_count) * int(config.max_value_total_tokens)),
            )
        else:
            self.action_value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, config.action_dim))

    def forward(self, batch: V6DualGraphBatch) -> dict[str, Tensor]:
        node_state = self._encode_sequence(batch.node_history, self.node_encoder).mean(dim=1)
        physical_edge_state = self._encode_sequence(
            batch.physical_edge_history,
            self.physical_edge_encoder,
        ).mean(dim=1)
        info_edge_state = self._encode_sequence(batch.info_edge_history, self.info_edge_encoder).mean(dim=1)
        action_state = self._encode_sequence(batch.action_history, self.action_encoder).mean(dim=1)
        task_state = self._encode_sequence(batch.task_history, self.task_encoder).mean(dim=1)

        edge_state, diagnostics = self.edge_fusion(physical_edge_state, info_edge_state, action_state)
        global_node = node_state.mean(dim=1)
        policy_context = torch.cat([global_node, task_state], dim=-1).unsqueeze(1).expand(-1, edge_state.shape[1], -1)

        logits = []
        edge_logits = []
        values = []
        value_bin_logits = []
        value_token_logits = []
        value_count_logits = []
        value_total_logits = []
        step_total_logs = []
        for _ in range(self.config.horizon):
            edge_state = self.rollout(
                torch.cat([edge_state, policy_context], dim=-1).reshape(-1, self.config.hidden_dim * 3),
                edge_state.reshape(-1, self.config.hidden_dim),
            ).reshape_as(edge_state)
            logits.append(self.action_logit_head(edge_state))
            if self.edge_logit_head is not None:
                edge_logits.append(self.edge_logit_head(edge_state).squeeze(-1))
            if self.step_total_head is not None:
                step_total_logs.append(self.step_total_head(edge_state.mean(dim=1)))
            if getattr(self.config, "value_mode", "continuous") == "discrete_bins":
                bin_logit = self.action_value_bin_head(edge_state).reshape(
                    *edge_state.shape[:-1],
                    self.config.action_dim,
                    int(self.config.max_value_bins),
                )
                value_bin_logits.append(bin_logit)
            elif getattr(self.config, "value_mode", "continuous") == "coupled_tokens":
                token_logit = self.action_value_token_head(edge_state).reshape(
                    *edge_state.shape[:-1],
                    int(self.config.value_token_group_count),
                    int(self.config.max_value_tokens),
                )
                value_token_logits.append(token_logit)
            elif getattr(self.config, "value_mode", "continuous") == "hierarchical_tokens":
                count_logit = self.action_value_count_head(edge_state).reshape(
                    *edge_state.shape[:-1],
                    int(self.config.value_token_group_count),
                    int(self.config.max_value_count_tokens),
                )
                total_logit = self.action_value_total_head(edge_state).reshape(
                    *edge_state.shape[:-1],
                    int(self.config.value_token_group_count),
                    int(self.config.max_value_total_tokens),
                )
                value_count_logits.append(count_logit)
                value_total_logits.append(total_logit)
            else:
                values.append(torch.relu(self.action_value_head(edge_state)))

        outputs = {
            "action_logit": torch.stack(logits, dim=1),
        }
        if self.edge_logit_head is not None:
            outputs["edge_logit"] = torch.stack(edge_logits, dim=1)
        if self.step_total_head is not None:
            outputs["step_total_log"] = torch.stack(step_total_logs, dim=1)
        if getattr(self.config, "value_mode", "continuous") == "discrete_bins":
            outputs["action_value_bin_logit"] = torch.stack(value_bin_logits, dim=1)
        elif getattr(self.config, "value_mode", "continuous") == "coupled_tokens":
            outputs["action_value_token_logit"] = torch.stack(value_token_logits, dim=1)
        elif getattr(self.config, "value_mode", "continuous") == "hierarchical_tokens":
            outputs["action_value_count_logit"] = torch.stack(value_count_logits, dim=1)
            outputs["action_value_total_logit"] = torch.stack(value_total_logits, dim=1)
        else:
            outputs["action_value"] = torch.stack(values, dim=1)
        if self.config.return_fusion_diagnostics:
            outputs.update(diagnostics)
        return outputs

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
