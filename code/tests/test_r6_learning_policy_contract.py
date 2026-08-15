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
)


def _identity(*, split: str = "validation") -> PolicyIdentity:
    return PolicyIdentity(
        scenario_id="load_low__density_sparse",
        seed=6,
        slot=12,
        split=split,
        protocol_fingerprint="frozen-r6",
    )


def _state(**overrides) -> PolicyState:
    values = {
        "explicit": torch.tensor([[1.0, 2.0, 3.0]]),
        "latent": torch.ones(1, 4, requires_grad=True),
        "offload_mask": torch.tensor([[True, False, True]]),
        "rb_mask": torch.tensor([[True, True, False, False]]),
        "cpu_task_mask": torch.tensor([[True, True]]),
        "cpu_capacity": torch.tensor([[10.0]]),
        "cpu_task_node_index": torch.tensor([[0, 0]]),
        "identities": (_identity(),),
    }
    values.update(overrides)
    return PolicyState.create(**values)


class R6LearningPolicyContractTest(unittest.TestCase):
    def test_state_detaches_world_model_latent_and_preserves_identity(self) -> None:
        state = _state()
        self.assertFalse(state.latent.requires_grad)
        self.assertIsNone(state.latent.grad_fn)
        self.assertEqual(state.identities[0].scenario_id, "load_low__density_sparse")
        self.assertEqual(state.batch_size, 1)

    def test_state_rejects_future_target_and_locked_test_before_use(self) -> None:
        with self.assertRaisesRegex(ValueError, "future target"):
            _state(extra_fields={"future_target": torch.ones(1)})
        with self.assertRaisesRegex(ValueError, "locked_test"):
            _state(identities=(_identity(split="locked_test"),))

    def test_state_rejects_nonfinite_and_shape_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            _state(explicit=torch.tensor([[float("nan"), 0.0, 0.0]]))
        with self.assertRaisesRegex(ValueError, "batch"):
            _state(cpu_capacity=torch.tensor([[1.0], [2.0]]))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            _state(cpu_capacity=torch.tensor([[-1.0]]))
        with self.assertRaisesRegex(ValueError, "node index"):
            _state(cpu_task_node_index=torch.tensor([[0, 1]]))

    def test_action_spec_validates_noop_indices_and_dimensions(self) -> None:
        spec = ActionSpec(
            offload_count=3,
            rb_count=4,
            cpu_task_count=2,
            offload_noop_index=0,
            rb_noop_index=0,
        )
        self.assertEqual(spec.offload_count, 3)
        with self.assertRaisesRegex(ValueError, "offload_noop_index"):
            ActionSpec(3, 4, 2, 3, 0)
        with self.assertRaisesRegex(ValueError, "positive"):
            ActionSpec(3, 0, 2, 0, 0)


if __name__ == "__main__":
    unittest.main()
