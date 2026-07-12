import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11RolloutValueCalibratorTrainingTest(unittest.TestCase):
    def test_choose_device_supports_cpu_auto_and_cuda_when_available(self):
        from run_v11_rollout_value_calibrator import choose_device

        self.assertEqual(choose_device("cpu").type, "cpu")
        self.assertIn(choose_device("auto").type, {"cpu", "cuda"})
        if torch.cuda.is_available():
            self.assertEqual(choose_device("cuda").type, "cuda")
        else:
            with self.assertRaises(RuntimeError):
                choose_device("cuda")

    def test_default_policy_checkpoint_uses_current_v10_bridge_checkpoint(self):
        import run_v11_rollout_value_calibrator as training

        self.assertIn(
            "pi_jwm_v10_policy_bridge_gpu_20260620",
            str(training.DEFAULT_POLICY_CHECKPOINT),
        )

    def test_build_calibrated_action_applies_activity_mask_scale_and_true_first(self):
        from run_v11_rollout_value_calibrator import build_calibrated_raw_action

        policy_value = torch.tensor(
            [[[[0.9, 2.1], [3.9, 8.1]], [[1.2, 4.7], [6.2, 9.1]]]],
            dtype=torch.float32,
        )
        calibrated_value = torch.tensor(
            [[[[1.0, 2.0], [4.0, 8.0]], [[1.0, 5.0], [6.0, 10.0]]]],
            dtype=torch.float32,
        )
        fixed_activity = torch.tensor(
            [[[[True, False], [False, True]], [[True, True], [False, False]]]]
        )
        true_raw = torch.tensor(
            [[[[11.0, 12.0], [13.0, 14.0]], [[21.0, 22.0], [23.0, 24.0]]]],
            dtype=torch.float32,
        )

        action = build_calibrated_raw_action(
            calibrated_value,
            fixed_activity,
            value_scale=1.5,
            true_raw=true_raw,
            true_first=True,
        )

        self.assertTrue(torch.equal(action[:, 0], true_raw[:, 0]))
        self.assertEqual(float(action[0, 1, 0, 0]), 1.5)
        self.assertEqual(float(action[0, 1, 0, 1]), 7.5)
        self.assertEqual(float(action[0, 1, 1, 0]), 0.0)
        self.assertEqual(float(action[0, 1, 1, 1]), 0.0)

    def test_mix_adaptive_new_action_by_step_gate_uses_new_only_after_threshold(self):
        from run_v11_rollout_value_calibrator import mix_adaptive_new_action_by_step_gate

        old_action = torch.zeros((1, 3, 2, 6), dtype=torch.float32)
        new_action = torch.ones_like(old_action)
        new_action[:, 0, :, 2] = 100.0
        new_action[:, 0, :, 4] = 10.0
        new_action[:, 1, :, 2] = 100.0
        new_action[:, 1, :, 4] = 25.0
        new_action[:, 2, :, 2] = 250.0
        new_action[:, 2, :, 4] = 25.0
        true_raw = torch.full_like(old_action, 9.0)

        mixed = mix_adaptive_new_action_by_step_gate(
            old_action,
            new_action,
            gate_feature="step_rb_cpu_total",
            gate_threshold=450.0,
            true_raw=true_raw,
            true_first=True,
        )

        self.assertTrue(torch.equal(mixed[:, 0], true_raw[:, 0]))
        self.assertTrue(torch.equal(mixed[:, 1], old_action[:, 1]))
        self.assertTrue(torch.equal(mixed[:, 2], new_action[:, 2]))

    def test_summarize_baseline_reproduction_reports_pass_fail_and_delta(self):
        from run_v11_rollout_value_calibrator import summarize_baseline_reproduction

        summary = summarize_baseline_reproduction(
            actual_val_active_rmse=234.12,
            expected_val_active_rmse=234.10,
            tolerance=0.05,
        )

        self.assertAlmostEqual(summary["delta"], 0.02, places=6)
        self.assertTrue(summary["passed"])

        failed = summarize_baseline_reproduction(
            actual_val_active_rmse=234.30,
            expected_val_active_rmse=234.10,
            tolerance=0.05,
        )

        self.assertFalse(failed["passed"])

    def test_normalize_and_inverse_normalize_action_tensor_round_trip(self):
        from run_v11_rollout_value_calibrator import inverse_normalize_action_tensor, normalize_action_tensor

        raw = torch.tensor([[[[1.0, 3.0], [2.0, 5.0]]]])
        mean = np.array([[[[1.0, 2.0], [1.0, 2.0]]]], dtype=np.float32)
        std = np.array([[[[2.0, 3.0], [2.0, 3.0]]]], dtype=np.float32)

        normalized = normalize_action_tensor(raw, (mean, std))
        restored = inverse_normalize_action_tensor(normalized, (mean, std))

        self.assertTrue(torch.allclose(restored, raw))

    def test_replace_future_actions_preserves_other_batch_fields(self):
        from pi_jwm.v6_dual_graph import V6DualGraphBatch
        from run_v11_rollout_value_calibrator import replace_future_actions

        batch = V6DualGraphBatch(
            node_history=torch.randn(2, 3, 4, 5),
            physical_edge_history=torch.randn(2, 3, 6, 8),
            info_edge_history=torch.randn(2, 3, 6, 5),
            action_history=torch.randn(2, 3, 6, 2),
            future_actions=torch.randn(2, 2, 6, 2),
            task_history=torch.randn(2, 3, 7),
            link_rate_baseline=torch.randn(2, 2, 6, 1),
        )
        future = torch.ones_like(batch.future_actions)

        replaced = replace_future_actions(batch, future)

        self.assertTrue(torch.equal(replaced.future_actions, future))
        self.assertIs(replaced.node_history, batch.node_history)
        self.assertIs(replaced.link_rate_baseline, batch.link_rate_baseline)

    def test_positive_value_bc_loss_uses_only_positive_targets(self):
        from run_v11_rollout_value_calibrator import compute_positive_value_bc_loss

        predicted = torch.tensor([0.0, 2.0, 8.0, 50.0])
        target = torch.tensor([0.0, 4.0, 10.0, 0.0])

        loss = compute_positive_value_bc_loss(predicted, target)

        expected = torch.nn.functional.huber_loss(
            torch.tensor([2.0, 8.0]),
            torch.tensor([4.0, 10.0]),
        )
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_weighted_objective_can_isolate_bridge_loss(self):
        from run_v11_rollout_value_calibrator import compute_weighted_objective

        bc = torch.tensor(2.0)
        bridge = torch.tensor(3.0)
        aggregate = torch.tensor(5.0)

        loss = compute_weighted_objective(
            bc,
            bridge,
            aggregate,
            bc_loss_weight=0.0,
            bridge_loss_weight=1.0,
            aggregate_loss_weight=0.0,
        )

        self.assertEqual(float(loss), 3.0)


if __name__ == "__main__":
    unittest.main()
