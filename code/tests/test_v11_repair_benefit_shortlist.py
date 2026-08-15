import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11RepairBenefitShortlistTest(unittest.TestCase):
    def test_conformal_lower_bound_subtracts_calibration_residual_quantile(self):
        from diagnose_v11_repair_benefit_shortlist import conformal_lower_bound

        pred = np.array([1.0, 3.0], dtype=np.float32)
        calib_pred = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        calib_target = np.array([0.5, 2.5, 2.0], dtype=np.float32)

        lower = conformal_lower_bound(pred, calib_pred, calib_target, residual_quantile=0.5)

        self.assertTrue(np.allclose(lower, np.array([0.5, 2.5], dtype=np.float32)))

    def test_select_positive_topk_indices_respects_group_scope_and_threshold(self):
        from diagnose_v11_repair_benefit_shortlist import select_positive_topk_indices

        coords = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 2, 0],
                [1, 1, 0],
            ],
            dtype=np.int64,
        )
        scores = np.array([0.2, -0.1, 0.5, 0.3], dtype=np.float32)

        selected = select_positive_topk_indices(coords, scores, top_k=1, scope="per_sample_step", min_score=0.0)

        self.assertEqual(set(selected.tolist()), {0, 2, 3})

    def test_predict_support_values_uses_nearest_neighbor_median_with_fallback(self):
        from diagnose_v11_repair_benefit_shortlist import predict_support_values

        train_x = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        train_y = np.array([10.0, 20.0, 100.0], dtype=np.float32)
        query_x = np.array([[0.2], [1.6]], dtype=np.float32)
        fallback = np.array([7.0, 8.0], dtype=np.float32)

        pred = predict_support_values(train_x, train_y, query_x, k=2, fallback_values=fallback)
        empty_pred = predict_support_values(train_x[:0], train_y[:0], query_x, k=2, fallback_values=fallback)

        self.assertTrue(np.allclose(pred, np.array([15.0, 60.0], dtype=np.float32)))
        self.assertTrue(np.allclose(empty_pred, fallback))

    def test_diagnostic_oracle_candidate_names_are_explicit(self):
        from diagnose_v11_repair_benefit_shortlist import diagnostic_candidate_name

        name = diagnostic_candidate_name("oracle_rank", "support_value", "per_sample", 4, 8)

        self.assertIn("diagnostic_only", name)
        self.assertIn("oracle_rank", name)
        self.assertIn("support8", name)


if __name__ == "__main__":
    unittest.main()
