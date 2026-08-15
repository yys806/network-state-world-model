"""Masked Actor-Critic policy sharing the frozen R6 safety contract."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .r6_learning_policy_contract import (
    ActionSpec,
    PolicyDecision,
    PolicyEvaluation,
    PolicyOutput,
    PolicyState,
    ProposedAction,
)
from .r6_learning_policy_safety import SafetyProjector


def _effective_mask(mask: Tensor, noop_index: int) -> Tensor:
    if mask.ndim != 2:
        raise ValueError("action mask must be [batch, action]")
    result = mask.bool().clone()
    empty = ~result.any(dim=1)
    if empty.any():
        result[empty, int(noop_index)] = True
    return result


def _masked_logits(logits: Tensor, mask: Tensor, noop_index: int) -> tuple[Tensor, Tensor]:
    effective = _effective_mask(mask, noop_index)
    if logits.shape != effective.shape:
        raise ValueError("policy logits and action mask shapes differ")
    return logits.masked_fill(~effective, torch.finfo(logits.dtype).min), effective


class MaskedActorCritic(nn.Module):
    """Shared encoder with masked discrete heads and a stochastic CPU head."""

    def __init__(
        self,
        *,
        explicit_dim: int,
        latent_dim: int,
        hidden_dim: int,
        spec: ActionSpec,
        projector: SafetyProjector | None = None,
    ) -> None:
        super().__init__()
        if min(int(explicit_dim), int(latent_dim), int(hidden_dim)) <= 0:
            raise ValueError("policy dimensions must be positive")
        self.spec = spec
        self.projector = projector or SafetyProjector()
        self.encoder = nn.Sequential(
            nn.Linear(int(explicit_dim) + int(latent_dim), int(hidden_dim)),
            nn.Tanh(),
        )
        self.offload_head = nn.Linear(int(hidden_dim), spec.offload_count)
        self.rb_head = nn.Linear(int(hidden_dim), spec.rb_count)
        self.cpu_loc_head = nn.Linear(int(hidden_dim), spec.cpu_task_count)
        self.cpu_log_scale_head = nn.Linear(int(hidden_dim), spec.cpu_task_count)
        self.value_head = nn.Linear(int(hidden_dim), 1)

    def forward(self, state: PolicyState) -> PolicyOutput:
        hidden = self.encoder(torch.cat((state.explicit, state.latent), dim=-1))
        offload_logits, _ = _masked_logits(
            self.offload_head(hidden), state.offload_mask, self.spec.offload_noop_index
        )
        rb_logits, _ = _masked_logits(
            self.rb_head(hidden), state.rb_mask, self.spec.rb_noop_index
        )
        cpu_loc = self.cpu_loc_head(hidden)
        cpu_log_scale = self.cpu_log_scale_head(hidden).clamp(-5.0, 2.0)
        cpu_raw = F.softplus(cpu_loc) * state.cpu_task_mask.to(hidden.dtype)
        output = PolicyOutput(
            offload_logits=offload_logits,
            rb_logits=rb_logits,
            offload_prob=F.softmax(offload_logits, dim=-1),
            rb_prob=F.softmax(rb_logits, dim=-1),
            cpu_raw=cpu_raw,
            cpu_loc=cpu_loc,
            cpu_log_scale=cpu_log_scale,
            value=self.value_head(hidden).squeeze(-1),
        )
        for name, value in output.__dict__.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"policy output {name} must be finite")
        return output

    @staticmethod
    def _sample_categorical(probability: Tensor, *, generator: torch.Generator) -> Tensor:
        return torch.multinomial(probability, 1, generator=generator).squeeze(-1)

    def act(
        self,
        state: PolicyState,
        *,
        deterministic: bool,
        seed: int,
    ) -> PolicyDecision:
        output = self(state)
        generator = torch.Generator(device=output.value.device)
        generator.manual_seed(int(seed))
        if deterministic:
            offload_index = output.offload_prob.argmax(dim=-1)
            rb_index = output.rb_prob.argmax(dim=-1)
            cpu_latent = output.cpu_loc
        else:
            offload_index = self._sample_categorical(output.offload_prob, generator=generator)
            rb_index = self._sample_categorical(output.rb_prob, generator=generator)
            noise = torch.randn(
                output.cpu_loc.shape,
                generator=generator,
                device=output.cpu_loc.device,
                dtype=output.cpu_loc.dtype,
            )
            cpu_latent = output.cpu_loc + output.cpu_log_scale.exp() * noise
        proposed = ProposedAction(
            offload_index=offload_index,
            rb_index=rb_index,
            cpu_allocation=F.softplus(cpu_latent) * state.cpu_task_mask.to(cpu_latent.dtype),
            cpu_latent=cpu_latent,
        )
        projected = self.projector.project(state, proposed, self.spec)
        return PolicyDecision(proposed=proposed, projected=projected, output=output)

    def evaluate(self, state: PolicyState, action: ProposedAction) -> PolicyEvaluation:
        output = self(state)
        batch = state.batch_size
        if action.offload_index.shape != (batch,) or action.rb_index.shape != (batch,):
            raise ValueError("discrete action shape must be [batch]")
        if action.cpu_latent is None or action.cpu_latent.shape != output.cpu_loc.shape:
            raise ValueError("cpu_latent is required to evaluate a stochastic CPU action")
        offload_log = F.log_softmax(output.offload_logits, dim=-1)
        rb_log = F.log_softmax(output.rb_logits, dim=-1)
        if ((action.offload_index < 0) | (action.offload_index >= self.spec.offload_count)).any():
            raise ValueError("offload action is outside the ActionSpec")
        if ((action.rb_index < 0) | (action.rb_index >= self.spec.rb_count)).any():
            raise ValueError("RB action is outside the ActionSpec")
        effective_offload_mask = _effective_mask(
            state.offload_mask, self.spec.offload_noop_index
        )
        effective_rb_mask = _effective_mask(state.rb_mask, self.spec.rb_noop_index)
        offload_is_legal = effective_offload_mask.gather(
            1, action.offload_index.long().unsqueeze(1)
        ).squeeze(1)
        rb_is_legal = effective_rb_mask.gather(
            1, action.rb_index.long().unsqueeze(1)
        ).squeeze(1)
        if not bool(offload_is_legal.all().item()):
            raise ValueError("offload action is masked by the current policy state")
        if not bool(rb_is_legal.all().item()):
            raise ValueError("RB action is masked by the current policy state")
        offload_log_prob = offload_log.gather(1, action.offload_index.long().unsqueeze(1)).squeeze(1)
        rb_log_prob = rb_log.gather(1, action.rb_index.long().unsqueeze(1)).squeeze(1)
        cpu_scale = output.cpu_log_scale.exp()
        cpu_distribution = torch.distributions.Normal(output.cpu_loc, cpu_scale)
        cpu_mask = state.cpu_task_mask.to(output.cpu_loc.dtype)
        cpu_log_prob = (cpu_distribution.log_prob(action.cpu_latent) * cpu_mask).sum(dim=-1)
        offload_entropy = -(output.offload_prob * offload_log).sum(dim=-1)
        rb_entropy = -(output.rb_prob * rb_log).sum(dim=-1)
        cpu_entropy = (cpu_distribution.entropy() * cpu_mask).sum(dim=-1)
        evaluation = PolicyEvaluation(
            log_prob=offload_log_prob + rb_log_prob + cpu_log_prob,
            entropy=offload_entropy + rb_entropy + cpu_entropy,
            value=output.value,
        )
        for name, value in evaluation.__dict__.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"policy evaluation {name} must be finite")
        return evaluation
