import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11AdaptiveGateTest(unittest.TestCase):
    def test_extract_step_gate_features_uses_deployable_action_totals(self):
        from pi_jwm.v11_adaptive_gate import extract_step_gate_features

        old_actions = torch.zeros((2, 3, 4, 6), dtype=torch.float32)
        new_actions = torch.zeros_like(old_actions)
        old_actions[:, :, :, 2] = 10.0
        new_actions[:, :, :, 2] = 20.0
        new_actions[:, :, :, 4] = 2.0
        features = extract_step_gate_features(old_actions, new_actions)

        self.assertEqual(tuple(features.shape), (2, 3, 5))
        self.assertTrue(torch.all(features[..., 0] > 0.0))
        self.assertTrue(torch.all(features[..., 1] > features[..., 0]))
        self.assertTrue(torch.all(features[..., 2] > 0.0))
        self.assertAlmostEqual(float(features[0, 0, 1]), 88.0 / 1000.0, places=6)

    def test_mix_actions_with_gate_probability_broadcasts_per_step(self):
        from pi_jwm.v11_adaptive_gate import mix_actions_with_gate_probability

        old_actions = torch.ones((1, 2, 3, 6), dtype=torch.float32)
        new_actions = torch.full_like(old_actions, 3.0)
        gate = torch.tensor([[0.0, 0.5]], dtype=torch.float32)

        mixed = mix_actions_with_gate_probability(old_actions, new_actions, gate)

        self.assertTrue(torch.equal(mixed[:, 0], old_actions[:, 0]))
        self.assertTrue(torch.allclose(mixed[:, 1], torch.full_like(mixed[:, 1], 2.0)))

    def test_hard_gate_uses_threshold(self):
        from pi_jwm.v11_adaptive_gate import hard_gate_from_probability

        prob = torch.tensor([[0.49, 0.5, 0.9]], dtype=torch.float32)

        gate = hard_gate_from_probability(prob, threshold=0.5)

        self.assertEqual(gate.tolist(), [[False, True, True]])

    def test_limited_training_indices_keep_full_stats_indices(self):
        from run_v11_adaptive_gate_training import split_stats_and_training_indices

        full = torch.arange(10).numpy()

        stats_idx, train_idx = split_stats_and_training_indices(full, max_train_samples=3)

        self.assertEqual(stats_idx.tolist(), list(range(10)))
        self.assertEqual(train_idx.tolist(), [0, 1, 2])

    def test_reference_indices_remain_full_when_training_is_limited(self):
        from run_v11_adaptive_gate_training import split_reference_and_training_indices

        full = torch.arange(10).numpy()

        reference_idx, train_idx = split_reference_and_training_indices(full, max_train_samples=4)

        self.assertEqual(reference_idx.tolist(), list(range(10)))
        self.assertEqual(train_idx.tolist(), [0, 1, 2, 3])

    def test_weighted_gate_loss_can_disable_bridge_term(self):
        from run_v11_adaptive_gate_training import compute_weighted_gate_loss

        bridge = torch.tensor(10.0)
        bc = torch.tensor(2.0)
        entropy = torch.tensor(1.0)

        loss = compute_weighted_gate_loss(
            bridge,
            bc,
            entropy,
            bridge_weight=0.0,
            bc_weight=3.0,
            entropy_weight=0.5,
        )

        self.assertAlmostEqual(float(loss), 5.5, places=6)

    def test_threshold_gate_initializes_to_manual_rule(self):
        from pi_jwm.v11_adaptive_gate import StepThresholdGate

        old_actions = torch.zeros((1, 3, 2, 6), dtype=torch.float32)
        new_actions = torch.zeros_like(old_actions)
        new_actions[:, 0, :, 2] = 100.0
        new_actions[:, 1, :, 2] = 250.0
        new_actions[:, 2, :, 2] = 400.0

        gate = StepThresholdGate(initial_threshold=450.0, temperature=10.0)
        hard = gate.hard(new_actions)

        self.assertEqual(hard.tolist(), [[False, True, True]])


if __name__ == "__main__":
    unittest.main()
