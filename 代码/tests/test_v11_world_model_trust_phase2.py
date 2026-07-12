import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from diagnose_v11_world_model_trust_phase2 import sensitivity_rows, slice_metrics  # noqa: E402


class WorldModelTrustPhase2Tests(unittest.TestCase):
    def test_slice_metrics_reports_high_load_bucket(self):
        pred = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
        true = np.array([[[[1.0], [1.0]], [[1.0], [1.0]]]], dtype=np.float32)
        active = np.ones_like(pred, dtype=np.float32)
        prob = np.full_like(pred, 0.75)
        rows = slice_metrics(
            {
                "link_rate_pred": pred,
                "link_rate_true": true,
                "link_activity_true": active,
                "link_activity_prob": prob,
            },
            np.array([[1.0, 10.0]], dtype=np.float32),
        )
        labels = {row["slice"] for row in rows}
        self.assertIn("all", labels)
        self.assertIn("p95_p100", labels)

    def test_sensitivity_rows_compare_normal_to_ablated(self):
        normal = {
            "link_activity_prob": np.ones((1, 2, 2, 1), dtype=np.float32),
            "link_rate_pred": np.ones((1, 2, 2, 1), dtype=np.float32) * 5,
            "link_activity_true": np.ones((1, 2, 2, 1), dtype=np.float32),
        }
        ablated = {
            "link_activity_prob": np.zeros((1, 2, 2, 1), dtype=np.float32),
            "link_rate_pred": np.zeros((1, 2, 2, 1), dtype=np.float32),
            "link_activity_true": np.ones((1, 2, 2, 1), dtype=np.float32),
        }
        rows = sensitivity_rows(normal, ablated, np.array([[1.0, 2.0]], dtype=np.float32), "zero_future_actions")
        by_scope = {row["scope"]: row for row in rows}
        self.assertEqual(by_scope["all_edges"]["mean_abs_activity_prob_delta"], 1.0)
        self.assertEqual(by_scope["true_active_edges"]["mean_abs_link_rate_delta"], 5.0)


if __name__ == "__main__":
    unittest.main()
