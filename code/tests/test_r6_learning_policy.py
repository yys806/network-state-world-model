from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_learning_policy import MaskedActorCritic  # noqa: E402
from pi_jwm.r6_learning_policy_contract import (  # noqa: E402
    ActionSpec,
    PolicyIdentity,
    PolicyState,
    ProposedAction,
)


SPEC = ActionSpec(3, 4, 2, 0, 0)


def _state(*, all_offload_illegal: bool = False) -> PolicyState:
    offload = torch.zeros(1, 3, dtype=torch.bool) if all_offload_illegal else torch.tensor([[True, False, True]])
    return PolicyState.create(
        explicit=torch.tensor([[1.0, 2.0, 3.0]]),
        latent=torch.tensor([[0.5, 0.25, -0.5, 1.0]], requires_grad=True),
        offload_mask=offload,
        rb_mask=torch.tensor([[True, False, True, False]]),
        cpu_task_mask=torch.tensor([[True, True]]),
        cpu_capacity=torch.tensor([[10.0]]),
        cpu_task_node_index=torch.tensor([[0, 0]]),
        identities=(PolicyIdentity("s0", 6, 1, "validation", "frozen-r6"),),
    )


class R6LearningPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4)
        self.policy = MaskedActorCritic(
            explicit_dim=3,
            latent_dim=4,
            hidden_dim=16,
            spec=SPEC,
        )

    def test_illegal_actions_have_exactly_zero_probability(self) -> None:
        output = self.policy(_state())
        self.assertEqual(output.offload_prob[0, 1].item(), 0.0)
        self.assertEqual(output.rb_prob[0, 1].item(), 0.0)
        self.assertEqual(output.rb_prob[0, 3].item(), 0.0)
        self.assertAlmostEqual(output.offload_prob.sum().item(), 1.0, places=6)

    def test_all_illegal_mask_enables_only_safe_noop(self) -> None:
        output = self.policy(_state(all_offload_illegal=True))
        self.assertEqual(output.offload_prob[0, 0].item(), 1.0)
        self.assertEqual(output.offload_prob[0, 1:].sum().item(), 0.0)

    def test_sampling_is_reproducible_and_safety_projected(self) -> None:
        left = self.policy.act(_state(), deterministic=False, seed=20260808)
        right = self.policy.act(_state(), deterministic=False, seed=20260808)
        self.assertTrue(torch.equal(left.proposed.offload_index, right.proposed.offload_index))
        self.assertTrue(torch.equal(left.proposed.rb_index, right.proposed.rb_index))
        self.assertTrue(torch.equal(left.proposed.cpu_latent, right.proposed.cpu_latent))
        self.assertTrue(torch.equal(left.projected.action.cpu_allocation, right.projected.action.cpu_allocation))
        self.assertLessEqual(left.projected.action.cpu_allocation.sum().item(), 10.0 + 1e-7)

    def test_evaluate_returns_finite_log_prob_entropy_and_value(self) -> None:
        state = _state()
        decision = self.policy.act(state, deterministic=False, seed=7)
        evaluation = self.policy.evaluate(state, decision.proposed)
        self.assertTrue(torch.isfinite(evaluation.log_prob).all())
        self.assertTrue(torch.isfinite(evaluation.entropy).all())
        self.assertTrue(torch.isfinite(evaluation.value).all())
        self.assertEqual(evaluation.log_prob.shape, (1,))

    def test_evaluate_rejects_discrete_action_masked_by_current_state(self) -> None:
        state = _state()
        decision = self.policy.act(state, deterministic=False, seed=7)
        illegal = ProposedAction(
            offload_index=torch.tensor([1]),
            rb_index=decision.proposed.rb_index,
            cpu_allocation=decision.proposed.cpu_allocation,
            cpu_latent=decision.proposed.cpu_latent,
        )
        with self.assertRaisesRegex(ValueError, "masked"):
            self.policy.evaluate(state, illegal)


if __name__ == "__main__":
    unittest.main()
