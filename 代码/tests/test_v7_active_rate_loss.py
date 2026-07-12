import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V7ActiveRateLossTest(unittest.TestCase):
    def test_active_only_rate_loss_ignores_inactive_edges(self):
        from run_world_model_v6_dual_graph_rollout import compute_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.zeros(1, 1, 2, 1),
            "link_rate": torch.tensor([[[[0.0], [10.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[1.0], [0.0]]]]),
            "link_rate": torch.tensor([[[[2.0], [0.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }

        _, parts = compute_loss(outputs, target, rate_loss_mode="active_only")

        self.assertAlmostEqual(parts["rate"], 4.0)
        self.assertAlmostEqual(parts["active_rate_loss"], 4.0)
        self.assertAlmostEqual(parts["inactive_rate_loss"], 100.0)

    def test_active_mixed_rate_loss_keeps_small_inactive_penalty(self):
        from run_world_model_v6_dual_graph_rollout import compute_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.zeros(1, 1, 2, 1),
            "link_rate": torch.tensor([[[[0.0], [10.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[1.0], [0.0]]]]),
            "link_rate": torch.tensor([[[[2.0], [0.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }

        _, parts = compute_loss(outputs, target, rate_loss_mode="active_mixed", inactive_rate_weight=0.1)

        self.assertAlmostEqual(parts["rate"], 14.0)
        self.assertAlmostEqual(parts["active_rate_loss"], 4.0)
        self.assertAlmostEqual(parts["inactive_rate_loss"], 100.0)

    def test_rate_teacher_loss_uses_teacher_mask_only(self):
        from run_world_model_v6_dual_graph_rollout import compute_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.zeros(1, 1, 3, 1),
            "link_rate": torch.tensor([[[[1.0], [3.0], [10.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[1.0], [1.0], [0.0]]]]),
            "link_rate": torch.tensor([[[[1.0], [3.0], [0.0]]]]),
            "link_rate_teacher": torch.tensor([[[[2.0], [1.0], [0.0]]]]),
            "link_rate_teacher_mask": torch.tensor([[[[1.0], [1.0], [0.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }

        _, parts = compute_loss(
            outputs,
            target,
            rate_loss_mode="active_only",
            rate_teacher_weight=0.5,
        )

        self.assertAlmostEqual(parts["rate"], 0.0)
        self.assertAlmostEqual(parts["rate_teacher"], 2.5)

    def test_rate_teacher_loss_is_zero_when_mask_is_empty(self):
        from run_world_model_v6_dual_graph_rollout import compute_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.zeros(1, 1, 1, 1),
            "link_rate": torch.tensor([[[[5.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[0.0]]]]),
            "link_rate": torch.tensor([[[[0.0]]]]),
            "link_rate_teacher": torch.tensor([[[[100.0]]]]),
            "link_rate_teacher_mask": torch.tensor([[[[0.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }

        _, parts = compute_loss(
            outputs,
            target,
            rate_loss_mode="active_mixed",
            inactive_rate_weight=0.1,
            rate_teacher_weight=1.0,
        )

        self.assertAlmostEqual(parts["rate_teacher"], 0.0)

    def test_active_rate_auxiliary_loss_uses_active_edges_only(self):
        from run_world_model_v6_dual_graph_rollout import compute_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.zeros(1, 1, 3, 1),
            "link_rate": torch.tensor([[[[1.0], [3.0], [99.0]]]]),
            "link_active_rate_aux": torch.tensor([[[[2.0], [7.0], [100.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[1.0], [1.0], [0.0]]]]),
            "link_rate": torch.tensor([[[[1.0], [5.0], [0.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }

        _, parts = compute_loss(
            outputs,
            target,
            rate_loss_mode="active_mixed",
            active_rate_auxiliary_weight=0.7,
        )

        self.assertAlmostEqual(parts["active_rate_auxiliary"], 2.5)

    def test_active_rate_auxiliary_loss_is_zero_without_active_edges(self):
        from run_world_model_v6_dual_graph_rollout import compute_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.zeros(1, 1, 1, 1),
            "link_rate": torch.tensor([[[[0.0]]]]),
            "link_active_rate_aux": torch.tensor([[[[100.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[0.0]]]]),
            "link_rate": torch.tensor([[[[0.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }

        _, parts = compute_loss(
            outputs,
            target,
            active_rate_auxiliary_weight=1.0,
        )

        self.assertAlmostEqual(parts["active_rate_auxiliary"], 0.0)

    def test_aux_soft_zero_rate_output_uses_activity_probability_and_zero_baseline(self):
        from run_world_model_v6_dual_graph_rollout import compute_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.tensor([[[[0.0]]]]),
            "link_rate": torch.tensor([[[[99.0]]]]),
            "link_active_rate_aux": torch.tensor([[[[4.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[1.0]]]]),
            "link_rate": torch.tensor([[[[1.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }

        _, parts = compute_loss(
            outputs,
            target,
            rate_loss_mode="active_only",
            rate_output_mode="aux_soft_zero",
            inactive_rate_value=0.0,
        )

        self.assertAlmostEqual(parts["rate"], 1.0)

    def test_aux_oracle_zero_rate_output_uses_true_activity_and_zero_baseline(self):
        from run_world_model_v6_dual_graph_rollout import compute_loss

        outputs = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity_logit": torch.zeros(1, 1, 2, 1),
            "link_rate": torch.tensor([[[[99.0], [99.0]]]]),
            "link_active_rate_aux": torch.tensor([[[[4.0], [10.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }
        target = {
            "node": torch.zeros(1, 1, 1, 1),
            "link_activity": torch.tensor([[[[1.0], [0.0]]]]),
            "link_rate": torch.tensor([[[[1.0], [-2.0]]]]),
            "task": torch.zeros(1, 1, 1),
        }

        _, parts = compute_loss(
            outputs,
            target,
            rate_loss_mode="active_mixed",
            inactive_rate_weight=0.1,
            rate_output_mode="aux_oracle_zero",
            inactive_rate_value=-2.0,
        )

        self.assertAlmostEqual(parts["active_rate_loss"], 9.0)
        self.assertAlmostEqual(parts["inactive_rate_loss"], 0.0)
        self.assertAlmostEqual(parts["rate"], 9.0)

    def test_aux_hard_zero_rate_output_uses_thresholded_activity_prediction(self):
        from run_world_model_v6_dual_graph_rollout import select_link_rate_output

        outputs = {
            "link_activity_logit": torch.tensor([[[[2.0], [-2.0]]]]),
            "link_rate": torch.tensor([[[[99.0], [99.0]]]]),
            "link_active_rate_aux": torch.tensor([[[[4.0], [10.0]]]]),
        }

        selected = select_link_rate_output(
            outputs,
            rate_output_mode="aux_hard_zero",
            inactive_rate_value=-2.0,
            activity_threshold=0.5,
        )

        expected = torch.tensor([[[[4.0], [-2.0]]]])
        torch.testing.assert_close(selected, expected)

    def test_default_inactive_rate_value_matches_normalized_raw_zero(self):
        import numpy as np

        from run_world_model_v6_dual_graph_rollout import compute_normalized_inactive_rate_value

        stats = {
            "rate_target_transform": "raw",
            "y_link_rate": (
                np.array([[[[10.0]]]], dtype=np.float32),
                np.array([[[[2.0]]]], dtype=np.float32),
            ),
        }

        self.assertAlmostEqual(compute_normalized_inactive_rate_value(stats), -5.0)


if __name__ == "__main__":
    unittest.main()
