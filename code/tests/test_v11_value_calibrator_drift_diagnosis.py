import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11ValueCalibratorDriftDiagnosisTest(unittest.TestCase):
    def test_summarize_delta_returns_count_means_and_totals(self):
        from diagnose_v11_value_calibrator_drift import summarize_delta

        baseline = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        calibrated = np.array([2.0, 1.0, 7.0], dtype=np.float32)

        row = summarize_delta("x", baseline, calibrated)

        self.assertEqual(row["name"], "x")
        self.assertEqual(row["count"], 3)
        self.assertAlmostEqual(row["baseline_mean"], 7.0 / 3.0, places=6)
        self.assertAlmostEqual(row["calibrated_mean"], 10.0 / 3.0, places=6)
        self.assertAlmostEqual(row["delta_mean"], 1.0, places=6)
        self.assertAlmostEqual(row["abs_delta_mean"], 5.0 / 3.0, places=6)

    def test_summarize_delta_handles_empty_values(self):
        from diagnose_v11_value_calibrator_drift import summarize_delta

        row = summarize_delta("empty", np.array([], dtype=np.float32), np.array([], dtype=np.float32))

        self.assertEqual(row["count"], 0)
        self.assertTrue(np.isnan(row["baseline_mean"]))
        self.assertTrue(np.isnan(row["delta_mean"]))


if __name__ == "__main__":
    unittest.main()
