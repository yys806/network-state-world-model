from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_joint_policy import (  # noqa: E402
    CandidateMaskedActorCritic,
    JointPolicyState,
    JointPolicyTrainingBatch,
    joint_actor_critic_step,
    joint_ppo_step,
)
from pi_jwm.r6_learning_policy_contract import PolicyIdentity  # noqa: E402


def _identity(slot: int = 2) -> PolicyIdentity:
    return PolicyIdentity("load_high__density_dense__r07", 507, slot, "validation", "f" * 64)


def _state(*, mask=None, explicit=None, latent=None) -> JointPolicyState:
    return JointPolicyState.create(
        explicit=torch.tensor([[1.0, 2.0, 3.0]]) if explicit is None else explicit,
        latent=torch.tensor([[0.1, 0.2]]) if latent is None else latent,
        candidate_descriptors=torch.arange(72, dtype=torch.float32).reshape(1, 4, 18) / 72.0,
        candidate_mask=torch.tensor([[True, False, True, True]]) if mask is None else mask,
        identities=(_identity(),),
    )


class R6JointPolicyTest(unittest.TestCase):
    def test_masked_probability_is_normalized_and_empty_mask_falls_back_to_default(self) -> None:
        torch.manual_seed(3)
        policy = CandidateMaskedActorCritic(3, 2, 18, hidden_dim=16, state_mode="explicit_latent")
        output = policy(_state())
        self.assertAlmostEqual(1.0, float(output.probability.sum().detach()), places=6)
        self.assertEqual(0.0, float(output.probability[0, 1].detach()))
        empty = _state(mask=torch.zeros((1, 4), dtype=torch.bool))
        fallback = policy(empty)
        self.assertEqual([1.0, 0.0, 0.0, 0.0], fallback.probability[0].tolist())

    def test_evaluate_rejects_masked_candidate(self) -> None:
        policy = CandidateMaskedActorCritic(3, 2, 18, hidden_dim=16, state_mode="explicit_latent")
        with self.assertRaisesRegex(ValueError, "masked"):
            policy.evaluate(_state(), torch.tensor([1]))

    def test_state_modes_ignore_the_unselected_branch(self) -> None:
        explicit_policy = CandidateMaskedActorCritic(3, 2, 18, hidden_dim=16, state_mode="explicit_only")
        latent_policy = CandidateMaskedActorCritic(3, 2, 18, hidden_dim=16, state_mode="latent_only")
        explicit_first = explicit_policy(_state()).logits
        explicit_second = explicit_policy(_state(latent=torch.tensor([[99.0, -99.0]]))).logits
        torch.testing.assert_close(explicit_first, explicit_second)
        latent_first = latent_policy(_state()).logits
        latent_second = latent_policy(_state(explicit=torch.tensor([[99.0, -99.0, 44.0]]))).logits
        torch.testing.assert_close(latent_first, latent_second)

    def test_actor_critic_and_ppo_take_finite_steps(self) -> None:
        torch.manual_seed(8)
        base = CandidateMaskedActorCritic(3, 2, 18, hidden_dim=16, state_mode="explicit_latent")
        actor = copy.deepcopy(base)
        ppo = copy.deepcopy(base)
        state = _state()
        decision = base.act(state, deterministic=False, seed=9)
        batch = JointPolicyTrainingBatch(
            state=state,
            candidate_index=decision.candidate_index,
            advantage=torch.tensor([1.25]),
            returns=torch.tensor([0.75]),
            old_log_prob=decision.log_prob.detach(),
        )
        actor_report = joint_actor_critic_step(
            policy=actor,
            batch=batch,
            optimizer=torch.optim.Adam(actor.parameters(), lr=1e-3),
        )
        ppo_report = joint_ppo_step(
            policy=ppo,
            batch=batch,
            optimizer=torch.optim.Adam(ppo.parameters(), lr=1e-3),
            clip_epsilon=0.2,
        )
        self.assertTrue(actor_report.parameter_changed)
        self.assertTrue(ppo_report.parameter_changed)
        self.assertGreater(actor_report.gradient_norm, 0.0)
        self.assertGreater(ppo_report.gradient_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
