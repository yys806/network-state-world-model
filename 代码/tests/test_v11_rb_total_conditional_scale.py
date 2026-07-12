import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11RbTotalConditionalScaleTest(unittest.TestCase):
    def test_global_scale_changes_only_active_rb_after_step0(self):
        from diagnose_v11_rb_total_conditional_scale import ScaleCandidate, apply_conditional_scale

        actions = np.ones((1, 3, 2, 6), dtype=np.float32)
        actions[..., 2] = np.array([[[10.0, 0.0], [20.0, 30.0], [40.0, 50.0]]], dtype=np.float32)

        repaired = apply_conditional_scale(actions, ScaleCandidate(name="g", mode="global", scale=1.5))

        self.assertTrue(np.allclose(repaired[:, 0, :, 2], actions[:, 0, :, 2]))
        self.assertTrue(np.allclose(repaired[:, 1:, :, 2], np.array([[[30.0, 45.0], [60.0, 75.0]]], dtype=np.float32)))
        for dim in (0, 1, 3, 4, 5):
            self.assertTrue(np.allclose(repaired[..., dim], actions[..., dim]))

    def test_step_scale_uses_separate_step1_step2_values(self):
        from diagnose_v11_rb_total_conditional_scale import ScaleCandidate, apply_conditional_scale

        actions = np.ones((1, 3, 1, 6), dtype=np.float32)
        actions[..., 2] = 10.0

        repaired = apply_conditional_scale(
            actions,
            ScaleCandidate(name="s", mode="step", step1_scale=1.2, step2_scale=0.8),
        )

        self.assertTrue(np.allclose(repaired[0, :, 0, 2], np.array([10.0, 12.0, 8.0], dtype=np.float32)))

    def test_gate_scale_uses_step_feature_threshold(self):
        from diagnose_v11_rb_total_conditional_scale import ScaleCandidate, apply_conditional_scale

        actions = np.zeros((1, 3, 2, 6), dtype=np.float32)
        actions[..., 2] = 10.0
        actions[0, 1, :, 4] = 100.0
        actions[0, 2, :, 4] = 5.0

        repaired = apply_conditional_scale(
            actions,
            ScaleCandidate(
                name="gate",
                mode="gate",
                gate_feature="step_rb_cpu_total",
                gate_threshold=100.0,
                low_scale=1.0,
                high_scale=1.2,
            ),
        )

        self.assertTrue(np.allclose(repaired[0, 0, :, 2], np.array([10.0, 10.0], dtype=np.float32)))
        self.assertTrue(np.allclose(repaired[0, 1, :, 2], np.array([12.0, 12.0], dtype=np.float32)))
        self.assertTrue(np.allclose(repaired[0, 2, :, 2], np.array([10.0, 10.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
