from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "code" / "src", ROOT / "code" / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from step5_3d_csi_scale_optimization_diagnosis_v1 import MAX_STEPS, LOG_EVERY, SAMPLES, config  # noqa: E402


class Step53DCsiDiagnosisTests(unittest.TestCase):
    def test_budget_and_subset_are_frozen(self):
        self.assertEqual(MAX_STEPS, 200)
        self.assertEqual(LOG_EVERY, 20)
        self.assertEqual(SAMPLES, [0, 1])

    def test_diagnostic_uses_normal_training_schedule(self):
        cfg = config()
        self.assertEqual(cfg.curriculum.horizons, (1, 2))
        self.assertEqual(cfg.curriculum.start_steps, (0, 10))
        self.assertEqual(cfg.stage1_steps, 10)
        self.assertEqual(cfg.kl_schedule.warmup_steps, 20)
        self.assertEqual(cfg.kl_schedule.beta_at(0), 0.0)
        self.assertEqual(cfg.kl_schedule.beta_at(20), 1.0)


if __name__ == "__main__":
    unittest.main()
