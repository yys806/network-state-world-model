import unittest
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class V6MetricsTest(unittest.TestCase):
    def test_regression_metrics_returns_rmse_and_mae(self):
        from pi_jwm.v6_metrics import regression_metrics

        pred = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        true = np.array([1.0, 4.0, 4.0], dtype=np.float32)

        metrics = regression_metrics(pred, true)

        self.assertAlmostEqual(metrics["rmse"], (4.0 / 3.0) ** 0.5)
        self.assertAlmostEqual(metrics["mae"], 2.0 / 3.0)

    def test_activity_metrics_uses_thresholded_probabilities(self):
        from pi_jwm.v6_metrics import activity_metrics

        prob = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
        true = np.array([1, 0, 1, 0], dtype=np.float32)

        metrics = activity_metrics(prob, true, threshold=0.5)

        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)

    def test_active_rate_metrics_filters_to_true_active_edges(self):
        from pi_jwm.v6_metrics import active_rate_metrics

        pred = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        true = np.array([12.0, 200.0, 36.0], dtype=np.float32)
        active = np.array([1, 0, 1], dtype=np.float32)

        metrics = active_rate_metrics(pred, true, active)

        self.assertEqual(metrics["active_count"], 2)
        self.assertAlmostEqual(metrics["active_rmse"], ((4.0 + 36.0) / 2.0) ** 0.5)
        self.assertAlmostEqual(metrics["active_mae"], 4.0)


if __name__ == "__main__":
    unittest.main()
