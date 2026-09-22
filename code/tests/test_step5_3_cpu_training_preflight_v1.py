"""Focused contract tests for the bounded STEP 5.3 preflight helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "code" / "src", ROOT / "code" / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.step5_2_training_loop_v1 import CurriculumConfig  # noqa: E402
from step5_3_cpu_training_preflight_v1 import DEVELOPMENT_GATE, _config, _raw_metrics, _relative_drop  # noqa: E402


class Step53PreflightTests(unittest.TestCase):
    def test_threshold_is_fixed_development_only_and_not_zero(self):
        self.assertGreater(DEVELOPMENT_GATE["relative_family_loss_drop"], 0.0)
        self.assertEqual(DEVELOPMENT_GATE["relative_family_loss_drop"], DEVELOPMENT_GATE["relative_prior_loss_drop"])
        self.assertEqual(DEVELOPMENT_GATE["max_steps_phase_a"], 30)
        self.assertEqual(DEVELOPMENT_GATE["max_steps_phase_b"], 30)
        self.assertEqual(DEVELOPMENT_GATE["max_steps_phase_c"], 40)

    def test_relative_drop_is_directional_and_scale_safe(self):
        self.assertAlmostEqual(_relative_drop(100.0, 90.0), 0.1)
        self.assertAlmostEqual(_relative_drop(0.0, 0.0), 0.0)
        self.assertLess(_relative_drop(100.0, 110.0), 0.0)

    def test_raw_metric_bridge_inverse_transforms_target_only(self):
        stats = {"position_std": 2.0, "speed_mean": 10.0, "speed_std": 4.0, "csi_mean": 98.0, "csi_std": 2.0}
        target_motion = torch.tensor([[[0.5, -1.0, 1.5, 0.25]]])
        target_csi = torch.tensor([[[1.0, -1.0]]])
        target_tensors = {
            "target_vehicle_motion_normalized": target_motion,
            "target_vehicle_motion_mask": torch.ones_like(target_motion, dtype=torch.bool),
            "target_comm_csi_normalized": target_csi,
            "target_comm_csi_mask": torch.ones_like(target_csi, dtype=torch.bool),
        }
        trainer = SimpleNamespace(data=SimpleNamespace(target_tensors=target_tensors, target_normalization=stats))
        # Decoder outputs are already raw: [1, -2, 3] m, 11 m/s and [100, 96] dB.
        result = {"motion_predictions": torch.tensor([[[1.0, -2.0, 3.0, 11.0]]]), "csi_predictions": torch.tensor([[[100.0, 96.0]]])}
        metrics = _raw_metrics(trainer, result, 1, [0])
        self.assertAlmostEqual(metrics["motion_delta_x_m_mae"], 0.0)
        self.assertAlmostEqual(metrics["motion_delta_y_m_mae"], 0.0)
        self.assertAlmostEqual(metrics["motion_delta_z_m_mae"], 0.0)
        self.assertAlmostEqual(metrics["motion_next_speed_mps_mae"], 0.0)
        self.assertAlmostEqual(metrics["csi_raw_mae_db"], 0.0)

    def test_phase_c_schedule_is_normal_path(self):
        config = _config(seed=1, horizon=2, stage1_steps=10, beta=1.0, max_steps=40, curriculum=CurriculumConfig(horizons=(1, 2), start_steps=(0, 10)), kl_warmup_steps=20)
        self.assertEqual(config.curriculum.horizon_for_step(0, 2), 1)
        self.assertEqual(config.curriculum.horizon_for_step(10, 2), 2)
        self.assertEqual(config.kl_schedule.beta_at(0), 0.0)
        self.assertEqual(config.kl_schedule.beta_at(20), 1.0)


if __name__ == "__main__":
    unittest.main()
