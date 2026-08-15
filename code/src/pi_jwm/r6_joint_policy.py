"""Candidate-set Masked Actor-Critic/PPO policy for PI-JWM R6."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .r6_learning_policy_contract import FORBIDDEN_FUTURE_FIELDS, PolicyIdentity


STATE_MODES = frozenset({"explicit_only", "latent_only", "explicit_latent"})


def _check_tensor(value: Tensor, *, field: str, ndim: int) -> None:
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise ValueError(f"{field} must be a {ndim}D tensor")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"{field} must be finite")


@dataclass(frozen=True)
class JointPolicyState:
    explicit: Tensor
    latent: Tensor
    candidate_descriptors: Tensor
    candidate_mask: Tensor
    identities: tuple[PolicyIdentity, ...]

    @property
    def batch_size(self) -> int:
        return int(self.explicit.shape[0])

    @classmethod
    def create(
        cls,
        *,
        explicit: Tensor,
        latent: Tensor,
        candidate_descriptors: Tensor,
        candidate_mask: Tensor,
        identities: Sequence[PolicyIdentity],
        extra_fields: Mapping[str, Any] | None = None,
    ) -> "JointPolicyState":
        extra = {} if extra_fields is None else dict(extra_fields)
        forbidden = sorted(FORBIDDEN_FUTURE_FIELDS.intersection(extra))
        if forbidden:
            raise ValueError(f"future target fields are forbidden: {forbidden}")
        _check_tensor(explicit, field="explicit", ndim=2)
        _check_tensor(latent, field="latent", ndim=2)
        _check_tensor(candidate_descriptors, field="candidate_descriptors", ndim=3)
        _check_tensor(candidate_mask, field="candidate_mask", ndim=2)
        batch = int(explicit.shape[0])
        if int(latent.shape[0]) != batch or int(candidate_descriptors.shape[0]) != batch:
            raise ValueError("joint policy state batch dimensions differ")
        if candidate_mask.shape != candidate_descriptors.shape[:2]:
            raise ValueError("candidate mask shape differs from descriptors")
        if int(candidate_descriptors.shape[1]) <= 0 or int(candidate_descriptors.shape[2]) <= 0:
            raise ValueError("candidate descriptor dimensions must be positive")
        if len(identities) != batch:
            raise ValueError("joint policy identity batch length differs")
        return cls(
            explicit=explicit.detach().clone(),
            latent=latent.detach().clone(),
            candidate_descriptors=candidate_descriptors.detach().clone(),
            candidate_mask=candidate_mask.detach().clone().bool(),
            identities=tuple(identities),
        )


@dataclass(frozen=True)
class JointPolicyOutput:
    logits: Tensor
    probability: Tensor
    value: Tensor
    effective_mask: Tensor


@dataclass(frozen=True)
class JointPolicyDecision:
    candidate_index: Tensor
    log_prob: Tensor
    entropy: Tensor
    value: Tensor


@dataclass(frozen=True)
class JointPolicyEvaluation:
    log_prob: Tensor
    entropy: Tensor
    value: Tensor


def _effective_mask(mask: Tensor) -> Tensor:
    result = mask.bool().clone()
    empty = ~result.any(dim=1)
    if empty.any():
        result[empty, 0] = True
    return result


class CandidateMaskedActorCritic(nn.Module):
    """Score complete legal candidates against explicit and/or latent state."""

    def __init__(
        self,
        explicit_dim: int,
        latent_dim: int,
        descriptor_dim: int,
        *,
        hidden_dim: int,
        state_mode: str,
    ) -> None:
        super().__init__()
        if min(int(explicit_dim), int(latent_dim), int(descriptor_dim), int(hidden_dim)) <= 0:
            raise ValueError("joint policy dimensions must be positive")
        mode = str(state_mode)
        if mode not in STATE_MODES:
            raise ValueError(f"unsupported joint policy state mode: {mode}")
        self.explicit_dim = int(explicit_dim)
        self.latent_dim = int(latent_dim)
        self.descriptor_dim = int(descriptor_dim)
        self.state_mode = mode
        state_dim = {
            "explicit_only": self.explicit_dim,
            "latent_only": self.latent_dim,
            "explicit_latent": self.explicit_dim + self.latent_dim,
        }[mode]
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh())
        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.descriptor_dim, hidden_dim),
            nn.Tanh(),
        )
        self.scorer = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def _state_input(self, state: JointPolicyState) -> Tensor:
        if state.explicit.shape[1] != self.explicit_dim:
            raise ValueError("explicit state dimension differs from policy")
        if state.latent.shape[1] != self.latent_dim:
            raise ValueError("latent state dimension differs from policy")
        if state.candidate_descriptors.shape[2] != self.descriptor_dim:
            raise ValueError("candidate descriptor dimension differs from policy")
        if self.state_mode == "explicit_only":
            return state.explicit
        if self.state_mode == "latent_only":
            return state.latent
        return torch.cat((state.explicit, state.latent), dim=-1)

    def forward(self, state: JointPolicyState) -> JointPolicyOutput:
        context = self.state_encoder(self._state_input(state))
        candidate = self.candidate_encoder(state.candidate_descriptors)
        raw_logits = self.scorer(torch.tanh(context.unsqueeze(1) + candidate)).squeeze(-1)
        mask = _effective_mask(state.candidate_mask)
        logits = raw_logits.masked_fill(~mask, torch.finfo(raw_logits.dtype).min)
        probability = F.softmax(logits, dim=-1)
        output = JointPolicyOutput(
            logits=logits,
            probability=probability,
            value=self.value_head(context).squeeze(-1),
            effective_mask=mask,
        )
        for name in ("logits", "probability", "value"):
            if not torch.isfinite(getattr(output, name)).all():
                raise ValueError(f"joint policy output {name} must be finite")
        return output

    def evaluate(self, state: JointPolicyState, candidate_index: Tensor) -> JointPolicyEvaluation:
        output = self(state)
        if candidate_index.shape != (state.batch_size,):
            raise ValueError("candidate_index shape must be [batch]")
        if ((candidate_index < 0) | (candidate_index >= output.logits.shape[1])).any():
            raise ValueError("candidate_index is outside the candidate set")
        legal = output.effective_mask.gather(1, candidate_index.long().unsqueeze(1)).squeeze(1)
        if not bool(legal.all().item()):
            raise ValueError("candidate_index is masked by the current state")
        log_distribution = F.log_softmax(output.logits, dim=-1)
        log_prob = log_distribution.gather(1, candidate_index.long().unsqueeze(1)).squeeze(1)
        entropy = -(output.probability * log_distribution).sum(dim=-1)
        evaluation = JointPolicyEvaluation(log_prob, entropy, output.value)
        if any(not torch.isfinite(value).all() for value in evaluation.__dict__.values()):
            raise ValueError("joint policy evaluation must be finite")
        return evaluation

    def act(
        self,
        state: JointPolicyState,
        *,
        deterministic: bool,
        seed: int,
    ) -> JointPolicyDecision:
        output = self(state)
        if deterministic:
            index = output.probability.argmax(dim=-1)
        else:
            generator = torch.Generator(device=output.probability.device)
            generator.manual_seed(int(seed))
            index = torch.multinomial(
                output.probability,
                1,
                generator=generator,
            ).squeeze(-1)
        evaluation = self.evaluate(state, index)
        return JointPolicyDecision(
            candidate_index=index,
            log_prob=evaluation.log_prob,
            entropy=evaluation.entropy,
            value=evaluation.value,
        )


@dataclass(frozen=True)
class JointPolicyTrainingBatch:
    state: JointPolicyState
    candidate_index: Tensor
    advantage: Tensor
    returns: Tensor
    old_log_prob: Tensor


@dataclass(frozen=True)
class JointPolicyStepReport:
    objective_id: str
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    ratio_min: float
    ratio_max: float
    gradient_norm: float
    parameter_changed: bool


def _validate_batch(batch: JointPolicyTrainingBatch) -> None:
    expected = (batch.state.batch_size,)
    for name, value in (
        ("candidate_index", batch.candidate_index),
        ("advantage", batch.advantage),
        ("returns", batch.returns),
        ("old_log_prob", batch.old_log_prob),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} shape must be [batch]")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")


def _take_step(
    policy: CandidateMaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    total: Tensor,
    *,
    max_grad_norm: float,
) -> tuple[float, bool]:
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float(max_grad_norm))
    gradient_norm = float(norm.detach().item() if isinstance(norm, Tensor) else norm)
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise ValueError("joint policy gradient norm must be finite and positive")
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in policy.parameters()):
        raise ValueError("joint policy gradients must be finite")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in policy.parameters()):
        raise ValueError("joint policy parameters must remain finite")
    changed = any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, policy.parameters())
    )
    if not changed:
        raise ValueError("joint policy optimizer step changed no parameter")
    return gradient_norm, changed


def _report(
    objective_id: str,
    total: Tensor,
    policy_loss: Tensor,
    value_loss: Tensor,
    entropy: Tensor,
    ratio: Tensor,
    gradient_norm: float,
    changed: bool,
) -> JointPolicyStepReport:
    return JointPolicyStepReport(
        objective_id=objective_id,
        total_loss=float(total.detach().item()),
        policy_loss=float(policy_loss.detach().item()),
        value_loss=float(value_loss.detach().item()),
        entropy=float(entropy.detach().item()),
        ratio_min=float(ratio.detach().min().item()),
        ratio_max=float(ratio.detach().max().item()),
        gradient_norm=gradient_norm,
        parameter_changed=changed,
    )


def joint_actor_critic_step(
    *,
    policy: CandidateMaskedActorCritic,
    batch: JointPolicyTrainingBatch,
    optimizer: torch.optim.Optimizer,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> JointPolicyStepReport:
    _validate_batch(batch)
    evaluation = policy.evaluate(batch.state, batch.candidate_index)
    policy_loss = -(evaluation.log_prob * batch.advantage.detach()).mean()
    value_loss = F.mse_loss(evaluation.value, batch.returns.detach())
    entropy = evaluation.entropy.mean()
    total = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy
    if not torch.isfinite(total):
        raise ValueError("Actor-Critic loss must be finite")
    norm, changed = _take_step(policy, optimizer, total, max_grad_norm=max_grad_norm)
    ratio = torch.ones_like(evaluation.log_prob)
    return _report("actor_critic", total, policy_loss, value_loss, entropy, ratio, norm, changed)


def joint_ppo_step(
    *,
    policy: CandidateMaskedActorCritic,
    batch: JointPolicyTrainingBatch,
    optimizer: torch.optim.Optimizer,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> JointPolicyStepReport:
    _validate_batch(batch)
    clip = float(clip_epsilon)
    if not 0.0 < clip < 1.0:
        raise ValueError("clip_epsilon must lie in (0, 1)")
    evaluation = policy.evaluate(batch.state, batch.candidate_index)
    ratio = torch.exp(evaluation.log_prob - batch.old_log_prob.detach())
    unclipped = ratio * batch.advantage.detach()
    clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * batch.advantage.detach()
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = F.mse_loss(evaluation.value, batch.returns.detach())
    entropy = evaluation.entropy.mean()
    total = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy
    if any(not torch.isfinite(value).all() for value in (ratio, total)):
        raise ValueError("PPO values must be finite")
    norm, changed = _take_step(policy, optimizer, total, max_grad_norm=max_grad_norm)
    return _report("ppo_clipped", total, policy_loss, value_loss, entropy, ratio, norm, changed)
