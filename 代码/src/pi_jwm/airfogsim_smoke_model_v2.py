"""Minimal action-conditioned PI-JWM rollout model for tensor-v2 smoke tests."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


COMPONENT_FEATURES = {
    "node": 7,
    "physical_edge": 5,
    "flow": 5,
    "task": 8,
}


def _masked_entity_mean(state: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    mask = present.to(dtype=state.dtype).unsqueeze(-1)
    numerator = (state * mask).sum(dim=2)
    denominator = mask.sum(dim=2).clamp_min(1.0)
    return numerator / denominator


class _EntityRolloutHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.entity_encoder = nn.Linear(feature_dim, hidden_dim)
        self.state_head = nn.Linear(hidden_dim, feature_dim)
        self.presence_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        last_state: torch.Tensor,
        context: torch.Tensor,
        horizon_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        entity = self.entity_encoder(last_state)[:, None, :, :]
        joint = torch.tanh(entity + context[:, None, None, :] + horizon_embedding[None, :, None, :])
        return self.state_head(joint), self.presence_head(joint).squeeze(-1)


class MinimalDualGraphWorldModel(nn.Module):
    """Small baseline that proves action-conditioned multi-component rollout is trainable."""

    def __init__(self, *, hidden_dim: int = 32, horizon_steps: int = 3) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.horizon_steps = int(horizon_steps)
        if self.hidden_dim <= 0 or self.horizon_steps <= 0:
            raise ValueError("hidden_dim and horizon_steps must be positive")
        summary_dim = sum(COMPONENT_FEATURES.values()) + 5
        self.history_encoder = nn.GRU(summary_dim, self.hidden_dim, batch_first=True)
        self.horizon_embedding = nn.Parameter(torch.randn(self.horizon_steps, self.hidden_dim) * 0.02)
        self.heads = nn.ModuleDict(
            {
                name: _EntityRolloutHead(feature_dim, self.hidden_dim)
                for name, feature_dim in COMPONENT_FEATURES.items()
            }
        )

    def forward(self, history: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        summaries = [
            _masked_entity_mean(history[f"{name}_state"], history[f"{name}_present"])
            for name in COMPONENT_FEATURES
        ]
        action_summary = _masked_entity_mean(history["task_action"], history["task_action_present"])
        encoded, _ = self.history_encoder(torch.cat([*summaries, action_summary], dim=-1))
        context = encoded[:, -1]
        output: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            state, presence_logits = head(
                history[f"{name}_state"][:, -1],
                context,
                self.horizon_embedding,
            )
            output[f"{name}_state"] = state
            output[f"{name}_presence_logits"] = presence_logits
        return output


def _valid_component_mask(name: str, static: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if name == "node":
        mask = static["node_kind_index"] >= 0
    elif name == "physical_edge":
        mask = torch.all(static["physical_edge_endpoint_index"] >= 0, dim=-1)
    elif name == "flow":
        mask = static["flow_valid"].bool()
    elif name == "task":
        mask = static["task_valid"].bool()
    else:
        raise KeyError(name)
    return mask.unsqueeze(0) if mask.ndim == 1 else mask


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numeric_mask = mask.to(dtype=value.dtype)
    return (value * numeric_mask).sum() / numeric_mask.sum().clamp_min(1.0)


def dual_graph_world_model_loss(
    output: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    static: Mapping[str, torch.Tensor],
    *,
    presence_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute mask-safe state MAE plus valid-slot presence BCE."""

    state_losses: list[torch.Tensor] = []
    presence_losses: list[torch.Tensor] = []
    metrics: dict[str, torch.Tensor] = {}
    for name in COMPONENT_FEATURES:
        target_present = target[f"{name}_present"].bool()
        absolute_error = torch.abs(output[f"{name}_state"] - target[f"{name}_state"]).mean(dim=-1)
        state_mae = _masked_mean(absolute_error, target_present)
        state_losses.append(state_mae)
        metrics[f"{name}_state_mae"] = state_mae.detach()

        valid = _valid_component_mask(name, static)[:, None, :].expand_as(target_present)
        binary_loss = F.binary_cross_entropy_with_logits(
            output[f"{name}_presence_logits"],
            target_present.to(dtype=output[f"{name}_presence_logits"].dtype),
            reduction="none",
        )
        presence_bce = _masked_mean(binary_loss, valid)
        presence_losses.append(presence_bce)
        metrics[f"{name}_presence_bce"] = presence_bce.detach()

    state_loss = torch.stack(state_losses).mean()
    presence_loss = torch.stack(presence_losses).mean()
    total = state_loss + float(presence_weight) * presence_loss
    metrics["state_loss"] = state_loss.detach()
    metrics["presence_loss"] = presence_loss.detach()
    metrics["total_loss"] = total.detach()
    return total, metrics

