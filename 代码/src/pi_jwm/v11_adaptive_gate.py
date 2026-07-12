"""Adaptive step gate for PI-JWM v11 candidate strategy calibration."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def extract_step_gate_features(old_actions: Tensor, new_actions: Tensor) -> Tensor:
    """Build deployable per-step features from decoded old/new action proposals."""

    if old_actions.shape != new_actions.shape or old_actions.ndim != 4:
        raise ValueError("actions must share shape [batch, horizon, edge, action_dim]")
    old_rb_cpu = old_actions[..., 2].sum(dim=-1) + old_actions[..., 4].sum(dim=-1)
    new_rb_cpu = new_actions[..., 2].sum(dim=-1) + new_actions[..., 4].sum(dim=-1)
    delta_rb_cpu = new_rb_cpu - old_rb_cpu
    new_active_count = (new_actions > 1e-9).any(dim=-1).sum(dim=-1).to(new_actions.dtype)
    horizon = old_actions.shape[1]
    step_feature = torch.linspace(-1.0, 1.0, horizon, dtype=old_actions.dtype, device=old_actions.device)
    step_feature = step_feature.reshape(1, horizon).expand(old_actions.shape[0], horizon)
    scale = torch.full_like(new_rb_cpu, 1000.0)
    return torch.stack(
        [
            old_rb_cpu / scale,
            new_rb_cpu / scale,
            delta_rb_cpu / scale,
            new_active_count / float(max(old_actions.shape[2], 1)),
            step_feature,
        ],
        dim=-1,
    )


def mix_actions_with_gate_probability(old_actions: Tensor, new_actions: Tensor, gate_probability: Tensor) -> Tensor:
    if old_actions.shape != new_actions.shape or old_actions.ndim != 4:
        raise ValueError("actions must share shape [batch, horizon, edge, action_dim]")
    if gate_probability.shape != old_actions.shape[:2]:
        raise ValueError("gate_probability must have shape [batch, horizon]")
    gate = gate_probability.to(dtype=old_actions.dtype).reshape(old_actions.shape[0], old_actions.shape[1], 1, 1)
    return old_actions * (1.0 - gate) + new_actions * gate


def hard_gate_from_probability(gate_probability: Tensor, threshold: float = 0.5) -> Tensor:
    return gate_probability >= float(threshold)


class StepAdaptiveGate(nn.Module):
    """Small MLP that selects old vs upward-corrected action proposals per step."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 16) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, horizon, input_dim]")
        return torch.sigmoid(self.net(features).squeeze(-1))


class StepThresholdGate(nn.Module):
    """Learnable threshold gate over predicted RB+CPU step totals."""

    def __init__(self, initial_threshold: float = 450.0, temperature: float = 25.0) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.threshold = nn.Parameter(torch.tensor(float(initial_threshold), dtype=torch.float32))
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature), dtype=torch.float32)))

    def score(self, actions: Tensor) -> Tensor:
        if actions.ndim != 4:
            raise ValueError("actions must have shape [batch, horizon, edge, action_dim]")
        return actions[..., 2].sum(dim=2) + actions[..., 4].sum(dim=2)

    def forward(self, actions: Tensor) -> Tensor:
        temperature = torch.exp(self.log_temperature).clamp_min(1e-3)
        return torch.sigmoid((self.score(actions) - self.threshold) / temperature)

    def hard(self, actions: Tensor) -> Tensor:
        return self.score(actions) >= self.threshold.detach()
