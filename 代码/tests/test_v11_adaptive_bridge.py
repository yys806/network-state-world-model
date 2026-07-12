import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11AdaptiveBridgeTest(unittest.TestCase):
    def test_step_rb_total_gate_uses_predicted_step_mass(self):
        from evaluate_v11_adaptive_bridge import compute_step_gate

        actions = torch.zeros((3, 4, 6), dtype=torch.float32)
        actions[0, :, 2] = 10.0
        actions[1, :, 2] = 100.0
        actions[2, :, 2] = 40.0

        gate = compute_step_gate(actions, gate_feature="step_rb_total", gate_threshold=200.0)

        self.assertEqual(gate.tolist(), [False, True, False])

    def test_mix_actions_applies_gate_per_step_and_preserves_shape(self):
        from evaluate_v11_adaptive_bridge import mix_actions_by_step_gate

        old_actions = torch.ones((3, 2, 6), dtype=torch.float32)
        new_actions = torch.full((3, 2, 6), 2.0, dtype=torch.float32)
        gate = torch.tensor([False, True, False])

        mixed = mix_actions_by_step_gate(old_actions, new_actions, gate)

        self.assertEqual(tuple(mixed.shape), tuple(old_actions.shape))
        self.assertTrue(torch.equal(mixed[0], old_actions[0]))
        self.assertTrue(torch.equal(mixed[1], new_actions[1]))
        self.assertTrue(torch.equal(mixed[2], old_actions[2]))

    def test_mix_actions_rejects_shape_mismatch(self):
        from evaluate_v11_adaptive_bridge import mix_actions_by_step_gate

        with self.assertRaises(ValueError):
            mix_actions_by_step_gate(
                torch.zeros((3, 2, 6)),
                torch.zeros((2, 2, 6)),
                torch.zeros((3,), dtype=torch.bool),
            )

    def test_adaptive_dataset_passes_item_to_inner_action_generators(self):
        from evaluate_v11_adaptive_bridge import AdaptivePolicyBridgeDataset

        class BaseDataset:
            def __len__(self):
                return 2

            def __getitem__(self, item):
                world_batch = SimpleNamespace(
                    node_history=torch.zeros((1, 1)),
                    physical_edge_history=torch.zeros((1, 1)),
                    info_edge_history=torch.zeros((1, 1)),
                    action_history=torch.zeros((1, 1)),
                    future_actions=torch.zeros((2, 1, 6)),
                    task_history=torch.zeros((1, 1)),
                    link_rate_baseline=None,
                )
                return world_batch, torch.tensor(float(item))

        class InnerDataset:
            def __init__(self, value):
                self.value = float(value)
                self.policy_dataset = [(object(), None), (object(), None)]
                self.seen_items = []

            def __len__(self):
                return 2

            def raw_future_from_normalized(self, future_actions):
                return torch.ones_like(future_actions)

            def generate_raw_actions(self, policy_batch, world_batch, true_future, item):
                self.seen_items.append(int(item))
                return torch.full_like(true_future, self.value)

            def normalize_future_actions(self, actions):
                return actions

        old_dataset = InnerDataset(1.0)
        new_dataset = InnerDataset(3.0)
        dataset = AdaptivePolicyBridgeDataset(
            BaseDataset(),
            old_dataset,
            new_dataset,
            stats={},
            gate_feature="step_rb_total",
            gate_threshold=2.0,
        )

        bridged, target = dataset[1]

        self.assertEqual(old_dataset.seen_items, [1])
        self.assertEqual(new_dataset.seen_items, [1])
        self.assertEqual(float(target), 1.0)
        self.assertEqual(tuple(bridged.future_actions.shape), (2, 1, 6))


if __name__ == "__main__":
    unittest.main()
