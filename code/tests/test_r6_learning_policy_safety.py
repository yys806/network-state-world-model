from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_learning_policy_contract import (  # noqa: E402
    ActionSpec,
    PolicyIdentity,
    PolicyState,
    ProposedAction,
)
from pi_jwm.r6_learning_policy_safety import SafetyProjector  # noqa: E402


def _state() -> PolicyState:
    return PolicyState.create(
        explicit=torch.tensor([[1.0, 2.0, 3.0]]),
        latent=torch.ones(1, 4),
        offload_mask=torch.tensor([[True, False, False]]),
        rb_mask=torch.tensor([[True, True, False]]),
        cpu_task_mask=torch.tensor([[True, True, False]]),
        cpu_capacity=torch.tensor([[10.0]]),
        cpu_task_node_index=torch.tensor([[0, 0, -1]]),
        identities=(PolicyIdentity("s0", 6, 1, "validation", "frozen-r6"),),
    )


SPEC = ActionSpec(3, 3, 3, 0, 0)


class R6LearningPolicySafetyTest(unittest.TestCase):
    def test_masks_illegal_discrete_actions_and_projects_cpu_capacity(self) -> None:
        result = SafetyProjector().project(
            _state(),
            ProposedAction(
                offload_index=torch.tensor([2]),
                rb_index=torch.tensor([1]),
                cpu_allocation=torch.tensor([[8.0, 8.0, 5.0]]),
            ),
            SPEC,
        )
        self.assertEqual(result.action.offload_index.item(), 0)
        self.assertEqual(result.action.rb_index.item(), 1)
        self.assertEqual(result.action.cpu_allocation[0, 2].item(), 0.0)
        self.assertLessEqual(result.action.cpu_allocation.sum().item(), 10.0 + 1e-7)
        reasons = {row.reason for row in result.records}
        self.assertIn("masked_offload_to_noop", reasons)
        self.assertIn("masked_cpu_task", reasons)
        self.assertIn("cpu_capacity_projection", reasons)
        self.assertEqual(result.fallback_count, 1)

    def test_no_legal_discrete_action_falls_back_to_configured_noop(self) -> None:
        state = PolicyState.create(
            explicit=torch.ones(1, 1),
            latent=torch.ones(1, 1),
            offload_mask=torch.zeros(1, 3, dtype=torch.bool),
            rb_mask=torch.zeros(1, 3, dtype=torch.bool),
            cpu_task_mask=torch.ones(1, 3, dtype=torch.bool),
            cpu_capacity=torch.tensor([[1.0]]),
            cpu_task_node_index=torch.tensor([[0, 0, 0]]),
            identities=(PolicyIdentity("s0", 1, 0, "train", "frozen-r6"),),
        )
        result = SafetyProjector().project(
            state,
            ProposedAction(torch.tensor([2]), torch.tensor([2]), torch.zeros(1, 3)),
            SPEC,
        )
        self.assertEqual(result.action.offload_index.item(), 0)
        self.assertEqual(result.action.rb_index.item(), 0)
        self.assertEqual(result.fallback_count, 2)

    def test_cpu_projection_enforces_each_node_capacity(self) -> None:
        state = PolicyState.create(
            explicit=torch.ones(1, 1),
            latent=torch.ones(1, 1),
            offload_mask=torch.ones(1, 3, dtype=torch.bool),
            rb_mask=torch.ones(1, 3, dtype=torch.bool),
            cpu_task_mask=torch.ones(1, 3, dtype=torch.bool),
            cpu_capacity=torch.tensor([[5.0, 10.0]]),
            cpu_task_node_index=torch.tensor([[0, 0, 1]]),
            identities=(PolicyIdentity("s0", 1, 0, "train", "frozen-r6"),),
        )
        result = SafetyProjector().project(
            state,
            ProposedAction(torch.tensor([0]), torch.tensor([0]), torch.tensor([[4.0, 4.0, 9.0]])),
            SPEC,
        )
        self.assertLessEqual(result.action.cpu_allocation[0, :2].sum().item(), 5.0 + 1e-7)
        self.assertLessEqual(result.action.cpu_allocation[0, 2].item(), 10.0 + 1e-7)

    def test_cpu_projection_recovers_float32_rounding_residual(self) -> None:
        proposed_cpu = torch.tensor(
            [[2.2385544776916504, 2.2363696098327637, 0.3658273220062256,
              0.840012788772583, 0.0287783145904541]],
            dtype=torch.float32,
        )
        state = PolicyState.create(
            explicit=torch.ones(1, 1),
            latent=torch.ones(1, 1),
            offload_mask=torch.ones(1, 3, dtype=torch.bool),
            rb_mask=torch.ones(1, 3, dtype=torch.bool),
            cpu_task_mask=torch.ones(1, 5, dtype=torch.bool),
            cpu_capacity=torch.tensor([[3.0]]),
            cpu_task_node_index=torch.zeros(1, 5, dtype=torch.long),
            identities=(PolicyIdentity("s0", 1, 0, "train", "frozen-r6"),),
        )
        result = SafetyProjector().project(
            state,
            ProposedAction(torch.tensor([0]), torch.tensor([0]), proposed_cpu),
            ActionSpec(3, 3, 5, 0, 0),
        )
        self.assertLessEqual(
            float(result.action.cpu_allocation.sum().item()),
            float(state.cpu_capacity[0, 0].item()),
        )

    def test_rejects_nonfinite_shapes_and_wrong_action_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            SafetyProjector().project(
                _state(),
                ProposedAction(torch.tensor([0]), torch.tensor([0]), torch.tensor([[float("nan"), 0.0, 0.0]])),
                SPEC,
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            SafetyProjector().project(
                _state(),
                ProposedAction(torch.tensor([0, 0]), torch.tensor([0]), torch.zeros(1, 3)),
                SPEC,
            )


if __name__ == "__main__":
    unittest.main()
