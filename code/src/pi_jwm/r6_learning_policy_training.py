"""CPU-only numerical gates for R6 Actor-Critic and PPO objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .r6_learning_policy import MaskedActorCritic
from .r6_learning_policy_contract import PolicyState, ProposedAction


@dataclass(frozen=True)
class PolicyTrainingBatch:
    state: PolicyState
    action: ProposedAction
    advantage: Tensor
    returns: Tensor
    old_log_prob: Tensor


@dataclass(frozen=True)
class PolicyStepReport:
    objective_id: str
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    ratio_min: float
    ratio_max: float
    gradient_norm: float
    policy_parameter_changed: bool

    def numeric_values(self) -> tuple[float, ...]:
        return (
            self.total_loss,
            self.policy_loss,
            self.value_loss,
            self.entropy,
            self.ratio_min,
            self.ratio_max,
            self.gradient_norm,
        )


def _validate_batch(batch: PolicyTrainingBatch) -> None:
    expected = (batch.state.batch_size,)
    for name, value in (
        ("advantage", batch.advantage),
        ("returns", batch.returns),
        ("old_log_prob", batch.old_log_prob),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} shape must be [batch]")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")


def _check_finite(name: str, *values: Tensor) -> None:
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError(f"{name} values must be finite")


def _take_step(
    *,
    policy: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    total: Tensor,
) -> tuple[float, bool]:
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    squared_norm = 0.0
    for parameter in policy.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise ValueError("policy gradients must be finite")
        squared_norm += float(parameter.grad.detach().pow(2).sum().item())
    gradient_norm = math.sqrt(squared_norm)
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise ValueError("policy gradient norm must be finite and positive")
    optimizer.step()
    for parameter in policy.parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError("updated policy parameters must be finite")
    changed = any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, policy.parameters())
    )
    if not changed:
        raise ValueError("policy optimizer step did not update any parameter")
    return gradient_norm, changed


def actor_critic_cpu_step(
    *,
    policy: MaskedActorCritic,
    batch: PolicyTrainingBatch,
    optimizer: torch.optim.Optimizer,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> PolicyStepReport:
    _validate_batch(batch)
    evaluation = policy.evaluate(batch.state, batch.action)
    policy_loss = -(evaluation.log_prob * batch.advantage.detach()).mean()
    value_loss = F.mse_loss(evaluation.value, batch.returns.detach())
    entropy = evaluation.entropy.mean()
    total = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy
    _check_finite("Actor-Critic", policy_loss, value_loss, entropy, total)
    gradient_norm, changed = _take_step(policy=policy, optimizer=optimizer, total=total)
    return PolicyStepReport(
        objective_id="actor_critic",
        total_loss=float(total.detach().item()),
        policy_loss=float(policy_loss.detach().item()),
        value_loss=float(value_loss.detach().item()),
        entropy=float(entropy.detach().item()),
        ratio_min=1.0,
        ratio_max=1.0,
        gradient_norm=gradient_norm,
        policy_parameter_changed=changed,
    )


def ppo_cpu_step(
    *,
    policy: MaskedActorCritic,
    batch: PolicyTrainingBatch,
    optimizer: torch.optim.Optimizer,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> PolicyStepReport:
    _validate_batch(batch)
    clip = float(clip_epsilon)
    if not 0.0 < clip < 1.0:
        raise ValueError("clip_epsilon must lie in (0, 1)")
    evaluation = policy.evaluate(batch.state, batch.action)
    ratio = torch.exp(evaluation.log_prob - batch.old_log_prob.detach())
    unclipped = ratio * batch.advantage.detach()
    clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * batch.advantage.detach()
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = F.mse_loss(evaluation.value, batch.returns.detach())
    entropy = evaluation.entropy.mean()
    total = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy
    _check_finite("PPO", ratio, policy_loss, value_loss, entropy, total)
    gradient_norm, changed = _take_step(policy=policy, optimizer=optimizer, total=total)
    return PolicyStepReport(
        objective_id="ppo_clipped",
        total_loss=float(total.detach().item()),
        policy_loss=float(policy_loss.detach().item()),
        value_loss=float(value_loss.detach().item()),
        entropy=float(entropy.detach().item()),
        ratio_min=float(ratio.detach().min().item()),
        ratio_max=float(ratio.detach().max().item()),
        gradient_norm=gradient_norm,
        policy_parameter_changed=changed,
    )
