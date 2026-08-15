from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_learning_policy_contract import PolicyIdentity  # noqa: E402
from pi_jwm.r6_reward_protocol import (  # noqa: E402
    RewardScale,
    ServiceFirstRewardProtocol,
    TransitionFacts,
)
from pi_jwm.r6_rollout import JointRollout, JointTransition, compute_gae  # noqa: E402


SCALE = RewardScale(10.0, 20.0, 30.0, 36, 10, 10, 10, "c" * 64)
REWARD = ServiceFirstRewardProtocol(SCALE)


def _identity(slot: int, *, seed: int = 507, split: str = "validation") -> PolicyIdentity:
    return PolicyIdentity("load_high__density_dense__r07", seed, slot, split, "d" * 64)


def _transition(
    slot: int,
    *,
    reward: float,
    value: float,
    next_value: float,
    terminated: bool = False,
    truncated: bool = False,
) -> JointTransition:
    breakdown = REWARD.score(
        TransitionFacts(
            on_time_completion_count=1 if reward >= 1.0 else 0,
            failure_count=0,
            completed_delay_sum=0.0,
            delivered_data_delta=0.0,
            energy_delta=0.0,
        )
    )
    if float(breakdown.total_reward) != reward:
        raise AssertionError("fixture only supports reward 0 or 1")
    return JointTransition.create(
        identity=_identity(slot),
        candidate_id="deadline_first",
        candidate_index=1,
        old_log_prob=-0.5,
        value=value,
        next_value=next_value,
        reward=breakdown,
        terminated=terminated,
        truncated=truncated,
    )


class R6RolloutTest(unittest.TestCase):
    def test_gae_matches_hand_calculation_and_truncation_bootstraps(self) -> None:
        rollout = JointRollout.create(
            (
                _transition(2, reward=1.0, value=1.0, next_value=2.0),
                _transition(3, reward=1.0, value=2.0, next_value=3.0, truncated=True),
            )
        )
        output = compute_gae(rollout, gamma=0.9, gae_lambda=0.8)
        self.assertAlmostEqual(3.024, float(output.advantage[0]), places=6)
        self.assertAlmostEqual(1.7, float(output.advantage[1]), places=6)
        self.assertAlmostEqual(4.024, float(output.returns[0]), places=6)
        self.assertAlmostEqual(3.7, float(output.returns[1]), places=6)

    def test_true_terminal_does_not_bootstrap(self) -> None:
        rollout = JointRollout.create(
            (
                _transition(2, reward=1.0, value=1.0, next_value=2.0),
                _transition(3, reward=1.0, value=2.0, next_value=99.0, terminated=True),
            )
        )
        output = compute_gae(rollout, gamma=0.9, gae_lambda=0.8)
        self.assertAlmostEqual(-1.0, float(output.advantage[1]), places=6)
        self.assertAlmostEqual(1.08, float(output.advantage[0]), places=6)

    def test_rollout_rejects_cross_identity_gaps_and_invalid_reward(self) -> None:
        with self.assertRaisesRegex(ValueError, "contiguous"):
            JointRollout.create((_transition(2, reward=1.0, value=0.0, next_value=0.0), _transition(4, reward=1.0, value=0.0, next_value=0.0)))
        second = _transition(3, reward=1.0, value=0.0, next_value=0.0)
        second = JointTransition.create(
            identity=_identity(3, seed=508),
            candidate_id=second.candidate_id,
            candidate_index=second.candidate_index,
            old_log_prob=second.old_log_prob,
            value=second.value,
            next_value=second.next_value,
            reward=second.reward,
            terminated=False,
            truncated=False,
        )
        with self.assertRaisesRegex(ValueError, "seed"):
            JointRollout.create((_transition(2, reward=1.0, value=0.0, next_value=0.0), second))
        invalid = REWARD.score(TransitionFacts(1, 0, 0.0, 0.0, 0.0, 1))
        with self.assertRaisesRegex(ValueError, "invalid reward"):
            JointTransition.create(
                identity=_identity(2),
                candidate_id="default",
                candidate_index=0,
                old_log_prob=0.0,
                value=0.0,
                next_value=0.0,
                reward=invalid,
                terminated=False,
                truncated=False,
            )


if __name__ == "__main__":
    unittest.main()
