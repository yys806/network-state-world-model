"""Conservative rollout-aligned action-value calibration for PI-JWM v11 candidates."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


class RolloutAlignedValueCalibrator(nn.Module):
    """Apply a bounded residual and project values onto an observed codebook."""

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        codebook_size: int,
        hidden_dim: int = 16,
        max_relative_delta: float = 0.25,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if horizon <= 0 or action_dim <= 0 or codebook_size <= 0 or hidden_dim <= 0:
            raise ValueError("calibrator dimensions must be positive")
        if max_relative_delta < 0.0:
            raise ValueError("max_relative_delta must be non-negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.codebook_size = int(codebook_size)
        self.max_relative_delta = float(max_relative_delta)
        self.temperature = float(temperature)
        self.residual = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        base_value: Tensor,
        activity_prob: Tensor,
        codebook: Tensor,
        active_mask: Tensor,
        hard: bool = True,
    ) -> Tensor:
        self._validate_inputs(base_value, activity_prob, codebook, active_mask)
        batch, horizon, edge_count, action_dim = base_value.shape
        step_feature = torch.linspace(-1.0, 1.0, horizon, dtype=base_value.dtype, device=base_value.device)
        step_feature = step_feature.reshape(1, horizon, 1, 1).expand(batch, horizon, edge_count, action_dim)
        dim_feature = torch.linspace(-1.0, 1.0, action_dim, dtype=base_value.dtype, device=base_value.device)
        dim_feature = dim_feature.reshape(1, 1, 1, action_dim).expand(batch, horizon, edge_count, action_dim)
        features = torch.stack(
            [torch.log1p(base_value.clamp_min(0.0)), activity_prob, step_feature, dim_feature],
            dim=-1,
        )
        relative_delta = self.max_relative_delta * torch.tanh(self.residual(features).squeeze(-1))
        proposed_value = (base_value.clamp_min(0.0) * (1.0 + relative_delta)).clamp_min(0.0)

        expanded_codebook = codebook.reshape(1, horizon, 1, action_dim, self.codebook_size)
        expanded_codebook = expanded_codebook.expand(batch, horizon, edge_count, action_dim, self.codebook_size)
        distances = torch.abs(proposed_value.unsqueeze(-1) - expanded_codebook)
        soft_weight = torch.softmax(-distances / self.temperature, dim=-1)
        soft_value = torch.sum(soft_weight * expanded_codebook, dim=-1)
        if hard:
            hard_index = torch.argmin(distances, dim=-1, keepdim=True)
            hard_value = torch.gather(expanded_codebook, dim=-1, index=hard_index).squeeze(-1)
            calibrated_value = soft_value + (hard_value - soft_value).detach()
        else:
            calibrated_value = soft_value
        return torch.where(active_mask.to(dtype=torch.bool), calibrated_value, torch.zeros_like(calibrated_value))

    def _validate_inputs(
        self,
        base_value: Tensor,
        activity_prob: Tensor,
        codebook: Tensor,
        active_mask: Tensor,
    ) -> None:
        if base_value.ndim != 4:
            raise ValueError("base_value must have shape [batch, horizon, edge, action_dim]")
        if activity_prob.shape != base_value.shape or active_mask.shape != base_value.shape:
            raise ValueError("activity_prob and active_mask must match base_value shape")
        if base_value.shape[1] != self.horizon or base_value.shape[-1] != self.action_dim:
            raise ValueError("base_value dimensions do not match calibrator configuration")
        if codebook.shape != (self.horizon, self.action_dim, self.codebook_size):
            raise ValueError("codebook must have shape [horizon, action_dim, codebook_size]")


def freeze_module(module: nn.Module) -> nn.Module:
    """Freeze a module while preserving gradients through its tensor operations."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def compute_action_aggregate_loss(
    predicted_action: Tensor,
    target_action: Tensor,
    dims: Iterable[int] = (1, 2, 3, 4),
) -> Tensor:
    """Compare horizon-level action totals across candidate edges."""

    if predicted_action.shape != target_action.shape or predicted_action.ndim != 4:
        raise ValueError("actions must share shape [batch, horizon, edge, action_dim]")
    selected_dims = tuple(int(dim) for dim in dims)
    if not selected_dims:
        return predicted_action.new_tensor(0.0)
    if min(selected_dims) < 0 or max(selected_dims) >= predicted_action.shape[-1]:
        raise ValueError("aggregate action dimension out of range")
    predicted_total = predicted_action[..., selected_dims].sum(dim=-2)
    target_total = target_action[..., selected_dims].sum(dim=-2)
    return nn.functional.mse_loss(predicted_total, target_total)
