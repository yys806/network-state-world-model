import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11BridgeOperatingPointDiagnosisTest(unittest.TestCase):
    def test_bin_rmse_summary_reports_old_new_and_delta(self):
        from diagnose_v11_bridge_operating_point import summarize_rate_bins

        truth = np.array([0.0, 10.0, 50.0, 120.0], dtype=np.float32)
        active = np.array([False, True, True, True])
        old_pred = np.array([0.0, 20.0, 70.0, 90.0], dtype=np.float32)
        new_pred = np.array([0.0, 15.0, 55.0, 100.0], dtype=np.float32)

        rows = summarize_rate_bins(truth, active, old_pred, new_pred, bins=[0.0, 60.0, float("inf")])

        self.assertEqual(rows[0]["bin"], "0-60")
        self.assertEqual(rows[0]["count"], 2)
        self.assertAlmostEqual(rows[0]["old_rmse"], np.sqrt((10.0**2 + 20.0**2) / 2.0), places=6)
        self.assertAlmostEqual(rows[0]["new_rmse"], 5.0, places=6)
        self.assertLess(rows[0]["delta_rmse"], 0.0)
        self.assertAlmostEqual(rows[0]["old_sse"], 500.0, places=6)
        self.assertAlmostEqual(rows[0]["new_sse"], 50.0, places=6)
        self.assertAlmostEqual(rows[0]["delta_sse"], -450.0, places=6)
        self.assertAlmostEqual(rows[0]["true_mean"], 30.0, places=6)
        self.assertAlmostEqual(rows[0]["old_pred_mean"], 45.0, places=6)
        self.assertAlmostEqual(rows[0]["new_pred_mean"], 35.0, places=6)
        self.assertEqual(rows[1]["bin"], "60-inf")

    def test_top_improvement_rows_sort_by_squared_error_reduction(self):
        from diagnose_v11_bridge_operating_point import top_active_improvements

        truth = np.array([[[10.0, 20.0, 30.0]]], dtype=np.float32)
        active = np.ones_like(truth, dtype=bool)
        old_pred = np.array([[[30.0, 19.0, 100.0]]], dtype=np.float32)
        new_pred = np.array([[[11.0, 25.0, 80.0]]], dtype=np.float32)

        rows = top_active_improvements(truth, active, old_pred, new_pred, top_k=2)

        self.assertEqual(rows[0]["edge"], 2)
        self.assertGreater(rows[0]["squared_error_reduction"], rows[1]["squared_error_reduction"])

    def test_top_regression_rows_sort_by_squared_error_increase(self):
        from diagnose_v11_bridge_operating_point import top_active_regressions

        truth = np.array([[[10.0, 20.0, 30.0]]], dtype=np.float32)
        active = np.ones_like(truth, dtype=bool)
        old_pred = np.array([[[11.0, 25.0, 80.0]]], dtype=np.float32)
        new_pred = np.array([[[30.0, 19.0, 100.0]]], dtype=np.float32)

        rows = top_active_regressions(truth, active, old_pred, new_pred, top_k=2)

        self.assertEqual(rows[0]["edge"], 2)
        self.assertGreater(rows[0]["squared_error_increase"], rows[1]["squared_error_increase"])

    def test_compare_three_predictions_reports_two_deltas(self):
        from diagnose_v11_bridge_operating_point import summarize_three_way_overall

        truth = np.array([10.0, 20.0], dtype=np.float32)
        active = np.array([True, True])
        old_pred = np.array([20.0, 30.0], dtype=np.float32)
        global_pred = np.array([15.0, 25.0], dtype=np.float32)
        adaptive_pred = np.array([12.0, 22.0], dtype=np.float32)

        row = summarize_three_way_overall(truth, active, old_pred, global_pred, adaptive_pred, split="unit")

        self.assertEqual(row["split"], "unit")
        self.assertAlmostEqual(row["old_rmse"], 10.0, places=6)
        self.assertAlmostEqual(row["global_rmse"], 5.0, places=6)
        self.assertAlmostEqual(row["adaptive_rmse"], 2.0, places=6)
        self.assertAlmostEqual(row["adaptive_vs_old_delta"], -8.0, places=6)
        self.assertAlmostEqual(row["adaptive_vs_global_delta"], -3.0, places=6)


if __name__ == "__main__":
    unittest.main()
