import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V8ActiveRateCalibrationTest(unittest.TestCase):
    def test_ridge_residual_calibrator_learns_train_val_correction_without_test_leakage(self):
        from calibrate_v8_active_rate_residuals import fit_ridge_residual_calibrator, predict_calibrated_rates

        train_rows = [
            _row(true_rate=200.0, pred_rate=100.0, action_l1=1.0),
            _row(true_rate=300.0, pred_rate=200.0, action_l1=2.0),
            _row(true_rate=400.0, pred_rate=300.0, action_l1=3.0),
        ]
        val_rows = [
            _row(true_rate=250.0, pred_rate=150.0, action_l1=1.5),
            _row(true_rate=350.0, pred_rate=250.0, action_l1=2.5),
        ]
        poisoned_test_rows = [
            _row(true_rate=2000.0, pred_rate=500.0, action_l1=9.0),
        ]

        calibrator, tuning_rows = fit_ridge_residual_calibrator(
            train_rows,
            val_rows,
            alphas=[0.1, 1.0, 10.0],
            include_edge_id=False,
            num_edges=4,
            name="ridge_basic",
        )
        calibrated = predict_calibrated_rates(calibrator, poisoned_test_rows)

        self.assertEqual(calibrator["name"], "ridge_basic")
        self.assertEqual(len(tuning_rows), 3)
        self.assertGreater(calibrated[0], 580.0)
        self.assertLess(calibrated[0], 650.0)

    def test_select_best_candidate_uses_validation_rmse(self):
        from calibrate_v8_active_rate_residuals import select_best_candidate

        rows = [
            {"model": "looks_good_on_test", "val_rmse": 50.0, "test_rmse": 1.0},
            {"model": "best_on_val", "val_rmse": 20.0, "test_rmse": 100.0},
        ]

        best = select_best_candidate(rows)

        self.assertEqual(best["model"], "best_on_val")

    def test_evaluate_rate_predictions_reports_rmse_mae_bias_and_under_rate(self):
        from calibrate_v8_active_rate_residuals import evaluate_rate_predictions

        true = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        pred = np.array([90.0, 220.0, 240.0], dtype=np.float32)

        metrics = evaluate_rate_predictions(true, pred)

        self.assertEqual(metrics["count"], 3)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt((100.0 + 400.0 + 3600.0) / 3.0))
        self.assertAlmostEqual(metrics["mae"], 30.0)
        self.assertAlmostEqual(metrics["bias"], -50.0 / 3.0)
        self.assertAlmostEqual(metrics["under_prediction_rate"], 2.0 / 3.0)

    def test_resolve_seed_splits_supports_expanded_active_heavy_split(self):
        from calibrate_v8_active_rate_residuals import parse_seed_list, resolve_seed_splits

        sample_seed = np.repeat(np.arange(20), 2)

        train_idx, val_idx, test_idx, spec = resolve_seed_splits(
            sample_seed,
            train_seeds=parse_seed_list("0 1 2 3"),
            val_seeds=parse_seed_list("4,5"),
            test_seeds=parse_seed_list("6"),
        )

        self.assertEqual(train_idx.tolist(), list(range(0, 8)))
        self.assertEqual(val_idx.tolist(), list(range(8, 12)))
        self.assertEqual(test_idx.tolist(), list(range(12, 14)))
        self.assertEqual(spec["train_seeds"], [0, 1, 2, 3])
        self.assertEqual(spec["val_seeds"], [4, 5])
        self.assertEqual(spec["test_seeds"], [6])

def _row(true_rate, pred_rate, action_l1, edge_id=1, horizon=1):
    return {
        "true_rate": true_rate,
        "pred_rate": pred_rate,
        "activity_prob": 0.99,
        "last_rate": 0.0,
        "action_l1": action_l1,
        "action_nonzero": 1,
        "horizon": horizon,
        "distance_3d": 10.0,
        "edge_id": edge_id,
    }


if __name__ == "__main__":
    unittest.main()
