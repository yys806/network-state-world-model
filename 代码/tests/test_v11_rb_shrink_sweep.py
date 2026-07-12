import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11RbShrinkSweepTest(unittest.TestCase):
    def test_apply_rb_total_scale_only_changes_rb_total_and_preserves_true_first(self):
        from sweep_v11_rb_shrink_bridge import apply_rb_total_scale

        actions = torch.ones((1, 3, 2, 6), dtype=torch.float32)
        actions[..., 2] = 10.0
        true_raw = torch.full_like(actions, 7.0)

        scaled = apply_rb_total_scale(actions, rb_scale=0.8, true_raw=true_raw, true_first=True)

        self.assertTrue(torch.equal(scaled[:, 0], true_raw[:, 0]))
        self.assertTrue(torch.allclose(scaled[:, 1:, :, 2], torch.full((1, 2, 2), 8.0)))
        for dim in (0, 1, 3, 4, 5):
            self.assertTrue(torch.allclose(scaled[:, 1:, :, dim], actions[:, 1:, :, dim]))

    def test_apply_rb_total_scale_can_use_step_mask(self):
        from sweep_v11_rb_shrink_bridge import apply_rb_total_scale

        actions = torch.ones((1, 3, 2, 6), dtype=torch.float32)
        actions[..., 2] = 10.0
        step_mask = torch.tensor([[False, True, False]])

        scaled = apply_rb_total_scale(actions, rb_scale=0.5, step_mask=step_mask, true_first=False)

        self.assertTrue(torch.allclose(scaled[:, 0, :, 2], torch.full((1, 2), 10.0)))
        self.assertTrue(torch.allclose(scaled[:, 1, :, 2], torch.full((1, 2), 5.0)))
        self.assertTrue(torch.allclose(scaled[:, 2, :, 2], torch.full((1, 2), 10.0)))

    def test_apply_rb_total_scale_rejects_negative_scale(self):
        from sweep_v11_rb_shrink_bridge import apply_rb_total_scale

        with self.assertRaises(ValueError):
            apply_rb_total_scale(torch.ones((1, 1, 1, 6)), rb_scale=-0.1)


if __name__ == "__main__":
    unittest.main()
