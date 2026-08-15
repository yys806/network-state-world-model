"""Strict transition identity, rollout segmentation and GAE for PI-JWM R6."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .r6_learning_policy_contract import PolicyIdentity
from .r6_reward_protocol import RewardBreakdown


@dataclass(frozen=True)
class JointTransition:
    identity: PolicyIdentity
    candidate_id: str
    candidate_index: int
    old_log_prob: float
    value: float
    next_value: float
    reward: RewardBreakdown
    terminated: bool
    truncated: bool

    @classmethod
    def create(
        cls,
        *,
        identity: PolicyIdentity,
        candidate_id: str,
        candidate_index: int,
        old_log_prob: float,
        value: float,
        next_value: float,
        reward: RewardBreakdown,
        terminated: bool,
        truncated: bool,
    ) -> "JointTransition":
        if not str(candidate_id).strip():
            raise ValueError("candidate_id cannot be empty")
        if int(candidate_index) < 0:
            raise ValueError("candidate_index must be nonnegative")
        numeric = (float(old_log_prob), float(value), float(next_value))
        if any(not math.isfinite(item) for item in numeric):
            raise ValueError("transition policy values must be finite")
        if not reward.valid or reward.total_reward is None:
            raise ValueError("invalid reward cannot enter an R6 rollout")
        if bool(terminated) and bool(truncated):
            raise ValueError("transition cannot be both terminated and truncated")
        return cls(
            identity=identity,
            candidate_id=str(candidate_id),
            candidate_index=int(candidate_index),
            old_log_prob=numeric[0],
            value=numeric[1],
            next_value=numeric[2],
            reward=reward,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )


@dataclass(frozen=True)
class JointRollout:
    transitions: tuple[JointTransition, ...]

    @classmethod
    def create(cls, transitions: Sequence[JointTransition]) -> "JointRollout":
        values = tuple(transitions)
        if not values:
            raise ValueError("joint rollout cannot be empty")
        first = values[0].identity
        for index, transition in enumerate(values):
            identity = transition.identity
            if identity.scenario_id != first.scenario_id:
                raise ValueError("rollout scenario_id changed within one trajectory")
            if identity.seed != first.seed:
                raise ValueError("rollout seed changed within one trajectory")
            if identity.split != first.split:
                raise ValueError("rollout split changed within one trajectory")
            if identity.protocol_fingerprint != first.protocol_fingerprint:
                raise ValueError("rollout protocol fingerprint changed")
            if identity.slot != first.slot + index:
                raise ValueError("rollout slots must be contiguous")
            if index < len(values) - 1 and (transition.terminated or transition.truncated):
                raise ValueError("rollout contains transitions after an episode boundary")
        return cls(values)


@dataclass(frozen=True)
class GAEOutput:
    advantage: np.ndarray
    returns: np.ndarray
    gamma: float
    gae_lambda: float


def compute_gae(
    rollout: JointRollout,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> GAEOutput:
    discount = float(gamma)
    trace = float(gae_lambda)
    if not 0.0 <= discount <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    if not 0.0 <= trace <= 1.0:
        raise ValueError("gae_lambda must lie in [0, 1]")
    steps = rollout.transitions
    advantage = np.zeros(len(steps), dtype=np.float64)
    running = 0.0
    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        bootstrap_mask = 0.0 if step.terminated else 1.0
        continuation_mask = 0.0 if (step.terminated or step.truncated) else 1.0
        delta = (
            float(step.reward.total_reward)
            + discount * bootstrap_mask * step.next_value
            - step.value
        )
        running = delta + discount * trace * continuation_mask * running
        advantage[index] = running
    values = np.asarray([step.value for step in steps], dtype=np.float64)
    returns = advantage + values
    if not np.isfinite(advantage).all() or not np.isfinite(returns).all():
        raise ValueError("GAE output must be finite")
    return GAEOutput(advantage, returns, discount, trace)
