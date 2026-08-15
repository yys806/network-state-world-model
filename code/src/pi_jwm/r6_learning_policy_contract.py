"""Frozen contracts shared by R6 learning-policy candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


NONLOCKED_SPLITS = frozenset({"train", "validation", "calibration"})
FORBIDDEN_FUTURE_FIELDS = frozenset(
    {"future_target", "target", "future_state", "future_reward", "next_observation"}
)


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be positive")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


@dataclass(frozen=True)
class PolicyIdentity:
    scenario_id: str
    seed: int
    slot: int
    split: str
    protocol_fingerprint: str

    def __post_init__(self) -> None:
        if not str(self.scenario_id).strip():
            raise ValueError("scenario_id cannot be empty")
        if not str(self.protocol_fingerprint).strip():
            raise ValueError("protocol_fingerprint cannot be empty")
        if str(self.split) == "locked_test":
            raise ValueError("locked_test is sealed until R9")
        if str(self.split) not in NONLOCKED_SPLITS:
            raise ValueError(f"unsupported policy split: {self.split}")
        if int(self.slot) < 0:
            raise ValueError("slot must be nonnegative")


@dataclass(frozen=True)
class ActionSpec:
    offload_count: int
    rb_count: int
    cpu_task_count: int
    offload_noop_index: int
    rb_noop_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "offload_count", _positive_int(self.offload_count, field="offload_count"))
        object.__setattr__(self, "rb_count", _positive_int(self.rb_count, field="rb_count"))
        object.__setattr__(self, "cpu_task_count", _positive_int(self.cpu_task_count, field="cpu_task_count"))
        if not 0 <= int(self.offload_noop_index) < self.offload_count:
            raise ValueError("offload_noop_index is outside the action space")
        if not 0 <= int(self.rb_noop_index) < self.rb_count:
            raise ValueError("rb_noop_index is outside the action space")


def _check_tensor(value: Tensor, *, field: str, ndim: int) -> None:
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise ValueError(f"{field} must be a {ndim}D tensor")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"{field} must be finite")


@dataclass(frozen=True)
class PolicyState:
    explicit: Tensor
    latent: Tensor
    offload_mask: Tensor
    rb_mask: Tensor
    cpu_task_mask: Tensor
    cpu_capacity: Tensor
    cpu_task_node_index: Tensor
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
        offload_mask: Tensor,
        rb_mask: Tensor,
        cpu_task_mask: Tensor,
        cpu_capacity: Tensor,
        cpu_task_node_index: Tensor,
        identities: Sequence[PolicyIdentity],
        extra_fields: Mapping[str, Any] | None = None,
    ) -> "PolicyState":
        extra = {} if extra_fields is None else dict(extra_fields)
        forbidden = sorted(FORBIDDEN_FUTURE_FIELDS.intersection(extra))
        if forbidden:
            raise ValueError(f"future target fields are forbidden: {forbidden}")
        _check_tensor(explicit, field="explicit", ndim=2)
        _check_tensor(latent, field="latent", ndim=2)
        _check_tensor(offload_mask, field="offload_mask", ndim=2)
        _check_tensor(rb_mask, field="rb_mask", ndim=2)
        _check_tensor(cpu_task_mask, field="cpu_task_mask", ndim=2)
        _check_tensor(cpu_capacity, field="cpu_capacity", ndim=2)
        _check_tensor(cpu_task_node_index, field="cpu_task_node_index", ndim=2)
        batch = int(explicit.shape[0])
        tensors = {
            "latent": latent,
            "offload_mask": offload_mask,
            "rb_mask": rb_mask,
            "cpu_task_mask": cpu_task_mask,
            "cpu_capacity": cpu_capacity,
            "cpu_task_node_index": cpu_task_node_index,
        }
        for name, value in tensors.items():
            if int(value.shape[0]) != batch:
                raise ValueError(f"{name} batch dimension differs from explicit state")
        if len(identities) != batch:
            raise ValueError("identity batch length differs from explicit state")
        if (cpu_capacity < 0).any():
            raise ValueError("cpu_capacity must be nonnegative")
        if cpu_task_node_index.shape != cpu_task_mask.shape:
            raise ValueError("cpu_task_node_index shape differs from cpu_task_mask")
        valid_node = (cpu_task_node_index >= 0) & (
            cpu_task_node_index < cpu_capacity.shape[1]
        )
        if (cpu_task_mask.bool() & ~valid_node).any():
            raise ValueError("active CPU task has an invalid node index")
        return cls(
            explicit=explicit.detach().clone(),
            latent=latent.detach().clone(),
            offload_mask=offload_mask.detach().clone().bool(),
            rb_mask=rb_mask.detach().clone().bool(),
            cpu_task_mask=cpu_task_mask.detach().clone().bool(),
            cpu_capacity=cpu_capacity.detach().clone(),
            cpu_task_node_index=cpu_task_node_index.detach().clone().long(),
            identities=tuple(identities),
        )


@dataclass(frozen=True)
class ProposedAction:
    offload_index: Tensor
    rb_index: Tensor
    cpu_allocation: Tensor
    cpu_latent: Tensor | None = None


@dataclass(frozen=True)
class ExecutableAction:
    offload_index: Tensor
    rb_index: Tensor
    cpu_allocation: Tensor


@dataclass(frozen=True)
class ProjectionRecord:
    batch_index: int
    action_family: str
    reason: str
    before: float
    after: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.before)) or not math.isfinite(float(self.after)):
            raise ValueError("projection record values must be finite")


@dataclass(frozen=True)
class ProjectionResult:
    action: ExecutableAction
    records: tuple[ProjectionRecord, ...]
    fallback_count: int


@dataclass(frozen=True)
class PolicyOutput:
    offload_logits: Tensor
    rb_logits: Tensor
    offload_prob: Tensor
    rb_prob: Tensor
    cpu_raw: Tensor
    cpu_loc: Tensor
    cpu_log_scale: Tensor
    value: Tensor


@dataclass(frozen=True)
class PolicyEvaluation:
    log_prob: Tensor
    entropy: Tensor
    value: Tensor


@dataclass(frozen=True)
class PolicyDecision:
    proposed: ProposedAction
    projected: ProjectionResult
    output: PolicyOutput
