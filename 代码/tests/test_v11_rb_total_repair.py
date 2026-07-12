import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11RbTotalRepairTest(unittest.TestCase):
    def test_apply_global_scale_changes_only_active_rb_total(self):
        from run_v11_rb_total_repair import RbRepairRule, apply_rb_total_repair

        actions = np.ones((1, 2, 2, 6), dtype=np.float32)
        actions[..., 2] = np.array([[[10.0, 0.0], [20.0, 30.0]]], dtype=np.float32)

        repaired = apply_rb_total_repair(
            actions,
            RbRepairRule(name="x", mode="global_scale", scale=0.5),
            preserve_step0=False,
        )

        self.assertTrue(np.allclose(repaired[..., 2], np.array([[[5.0, 0.0], [10.0, 15.0]]], dtype=np.float32)))
        for dim in (0, 1, 3, 4, 5):
            self.assertTrue(np.allclose(repaired[..., dim], actions[..., dim]))

    def test_fit_step_scales_uses_positive_predicted_rb_values(self):
        from run_v11_rb_total_repair import fit_step_scale_rule

        baseline = np.zeros((2, 2, 1, 6), dtype=np.float32)
        truth = np.zeros_like(baseline)
        baseline[:, 0, 0, 2] = [10.0, 20.0]
        truth[:, 0, 0, 2] = [20.0, 40.0]
        baseline[:, 1, 0, 2] = [10.0, 20.0]
        truth[:, 1, 0, 2] = [5.0, 10.0]

        rule = fit_step_scale_rule(baseline, truth)

        self.assertEqual(rule.mode, "step_scale")
        self.assertTrue(np.allclose(rule.step_scales, np.array([2.0, 0.5], dtype=np.float32)))

    def test_apply_step_scale_preserves_true_first_by_default(self):
        from run_v11_rb_total_repair import RbRepairRule, apply_rb_total_repair

        actions = np.ones((1, 3, 1, 6), dtype=np.float32)
        actions[..., 2] = 10.0
        rule = RbRepairRule(name="step", mode="step_scale", step_scales=np.array([3.0, 2.0, 0.5], dtype=np.float32))

        repaired = apply_rb_total_repair(actions, rule, preserve_step0=True)

        self.assertEqual(repaired[0, 0, 0, 2], 10.0)
        self.assertEqual(repaired[0, 1, 0, 2], 20.0)
        self.assertEqual(repaired[0, 2, 0, 2], 5.0)

    def test_zero_below_threshold_only_removes_small_rb_total_after_step0(self):
        from run_v11_rb_total_repair import RbRepairRule, apply_rb_total_repair

        actions = np.ones((1, 3, 3, 6), dtype=np.float32)
        actions[..., 2] = np.array([[[1.0, 2.0, 3.0], [0.5, 5.0, 9.0], [1.0, 4.0, 8.0]]], dtype=np.float32)
        rule = RbRepairRule(name="zero", mode="zero_below_threshold", threshold=4.0)

        repaired = apply_rb_total_repair(actions, rule, preserve_step0=True)

        self.assertTrue(np.allclose(repaired[:, 0, :, 2], actions[:, 0, :, 2]))
        self.assertTrue(np.allclose(repaired[:, 1, :, 2], np.array([[0.0, 5.0, 9.0]], dtype=np.float32)))
        self.assertTrue(np.allclose(repaired[:, 2, :, 2], np.array([[0.0, 4.0, 8.0]], dtype=np.float32)))
        for dim in (0, 1, 3, 4, 5):
            self.assertTrue(np.allclose(repaired[..., dim], actions[..., dim]))


if __name__ == "__main__":
    unittest.main()
