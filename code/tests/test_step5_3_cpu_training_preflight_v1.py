"""Focused contract tests for the bounded STEP 5.3 preflight helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "code" / "src", ROOT / "code" / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from step5_3_cpu_training_preflight_v1 import DEVELOPMENT_GATE, _relative_drop  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
