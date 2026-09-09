from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
for path in (SCRIPTS_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class ThresholdProtocolAuditTests(unittest.TestCase):
    def test_weighted_probability_fixed_point_uses_high_raw_threshold(self):
        from run_formal_p4_threshold_protocol_audit_v1 import POS_WEIGHT

        ordinary_probability = 0.5
        raw = POS_WEIGHT * ordinary_probability / (1.0 - ordinary_probability + POS_WEIGHT * ordinary_probability)
        self.assertAlmostEqual(raw, 50.0 / 51.0, places=12)

    def test_f1_counts_are_deterministic(self):
        from run_formal_p4_threshold_protocol_audit_v1 import _f1

        row = _f1(np.array([0.9, 0.2, 0.8]), np.array([True, False, False]), 0.5)
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (1, 1, 0))
        self.assertAlmostEqual(row["f1"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
