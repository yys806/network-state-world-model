import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V8ActiveRateDiagnosisTest(unittest.TestCase):
    def test_resolve_diagnosis_indices_defaults_to_validation_seeds(self):
        from analyze_v8_active_rate_errors import resolve_diagnosis_indices

        sample_seed = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        indices, split_spec = resolve_diagnosis_indices(
            sample_seed,
            train_seeds=[0, 1],
            val_seeds=[2, 3],
            test_seeds=[4],
        )

        np.testing.assert_array_equal(indices, np.array([2, 3], dtype=np.int64))
        self.assertEqual(split_spec["diagnosis_split"], "val")
        self.assertEqual(split_spec["val_seeds"], [2, 3])

    def test_prepare_arrays_for_experiment_adds_event_memory_features(self):
        from analyze_v8_active_rate_errors import prepare_arrays_for_experiment

        with tempfile.TemporaryDirectory() as temp_dir:
            exp_dir = Path(temp_dir)
            summary_path = exp_dir / "v8_full_training_summary.json"
            summary_path.write_text(
                """
                {
                  "config": {
                    "use_event_memory_features": true
                  }
                }
                """.strip(),
                encoding="utf-8",
            )
            arrays = {
                "x_link": np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
                "link_features": np.array(["active", "rate_sum"]),
            }

            prepared = prepare_arrays_for_experiment(exp_dir, arrays)

        self.assertIn("active_frequency", [str(item) for item in prepared["link_features"]])
        self.assertGreater(prepared["x_link"].shape[-1], arrays["x_link"].shape[-1])

    def test_flatten_active_rate_rows_keeps_only_true_active_links(self):
        from analyze_v8_active_rate_errors import flatten_active_rate_rows

        predictions = {
            "link_rate_pred": np.array([[[[10.0], [20.0]], [[30.0], [40.0]]]], dtype=np.float32),
            "link_active_rate_aux_pred": np.array([[[[11.0], [21.0]], [[31.0], [41.0]]]], dtype=np.float32),
            "link_rate_true": np.array([[[[12.0], [0.0]], [[33.0], [44.0]]]], dtype=np.float32),
            "link_activity_true": np.array([[[[1.0], [0.0]], [[1.0], [1.0]]]], dtype=np.float32),
            "link_activity_prob": np.array([[[[0.9], [0.1]], [[0.8], [0.7]]]], dtype=np.float32),
        }
        arrays = {
            "sample_seed": np.array([9], dtype=np.int32),
            "edge_src_idx": np.array([0, 1], dtype=np.int32),
            "edge_dst_idx": np.array([1, 0], dtype=np.int32),
            "edge_a_future": np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32),
            "x_link": np.array([[[[1.0, 5.0], [1.0, 6.0]]]], dtype=np.float32),
            "link_features": np.array(["active", "rate_sum"]),
        }
        physical_last = np.array([[[1.0, 2.0, 3.0, 10.0], [1.0, 2.0, 3.0, 20.0]]], dtype=np.float32)

        rows = flatten_active_rate_rows(
            model_name="toy",
            predictions=predictions,
            arrays=arrays,
            sample_indices=np.array([0], dtype=np.int64),
            physical_edge_last=physical_last,
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual([row["edge_id"] for row in rows], [0, 0, 1])
        self.assertEqual([row["horizon"] for row in rows], [1, 2, 2])
        self.assertAlmostEqual(rows[0]["residual"], -2.0)
        self.assertAlmostEqual(rows[1]["abs_error"], 3.0)
        self.assertAlmostEqual(rows[2]["action_l1"], 15.0)
        self.assertAlmostEqual(rows[2]["last_rate"], 6.0)
        self.assertAlmostEqual(rows[2]["distance_3d"], 20.0)

    def test_summarize_bucket_metrics_reports_rmse_and_bias(self):
        from analyze_v8_active_rate_errors import summarize_bucket_metrics

        rows = [
            {"model": "a", "bucket": "low", "true_rate": 10.0, "pred_rate": 8.0, "residual": -2.0, "abs_error": 2.0},
            {"model": "a", "bucket": "low", "true_rate": 20.0, "pred_rate": 25.0, "residual": 5.0, "abs_error": 5.0},
            {"model": "a", "bucket": "high", "true_rate": 100.0, "pred_rate": 80.0, "residual": -20.0, "abs_error": 20.0},
        ]

        summary = summarize_bucket_metrics(rows, group_keys=("model", "bucket"))
        low = next(row for row in summary if row["bucket"] == "low")
        high = next(row for row in summary if row["bucket"] == "high")

        self.assertEqual(low["count"], 2)
        self.assertAlmostEqual(low["rmse"], np.sqrt((4.0 + 25.0) / 2.0))
        self.assertAlmostEqual(low["bias"], 1.5)
        self.assertAlmostEqual(low["mean_true_rate"], 15.0)
        self.assertEqual(high["count"], 1)
        self.assertAlmostEqual(high["under_prediction_rate"], 1.0)

    def test_diagnose_hurdle_predictions_separates_gate_amplitude_and_inactive_mass(self):
        from analyze_v8_active_rate_errors import diagnose_hurdle_predictions

        predictions = {
            "link_rate_pred": np.array([[[[45.0], [10.0], [240.0], [2.0]]]], dtype=np.float32),
            "link_positive_rate_pred": np.array([[[[90.0], [50.0], [300.0], [20.0]]]], dtype=np.float32),
            "link_rate_true": np.array([[[[100.0], [0.0], [400.0], [0.0]]]], dtype=np.float32),
            "link_activity_true": np.array([[[[1.0], [0.0], [1.0], [0.0]]]], dtype=np.float32),
            "link_activity_prob": np.array([[[[0.5], [0.2], [0.8], [0.1]]]], dtype=np.float32),
        }

        diagnosis = diagnose_hurdle_predictions(predictions)

        expected_final_rmse = np.sqrt((55.0**2 + 160.0**2) / 2.0)
        expected_positive_rmse = np.sqrt((10.0**2 + 100.0**2) / 2.0)
        self.assertAlmostEqual(diagnosis["active_final_rmse"], expected_final_rmse)
        self.assertAlmostEqual(diagnosis["active_positive_rmse"], expected_positive_rmse)
        self.assertAlmostEqual(
            diagnosis["gate_suppression_gap"],
            expected_final_rmse - expected_positive_rmse,
        )
        self.assertAlmostEqual(diagnosis["inactive_rate_mass_sum"], 12.0)
        self.assertAlmostEqual(diagnosis["inactive_rate_mass_mean"], 6.0)
        self.assertGreater(diagnosis["top20_active_sse_share"], 0.8)
        self.assertEqual(diagnosis["recommended_method"], "lds")

    def test_summarize_bucket_metrics_reports_sse_share_within_model(self):
        from analyze_v8_active_rate_errors import summarize_bucket_metrics

        rows = [
            {"model": "a", "bucket": "low", "true_rate": 10.0, "pred_rate": 9.0, "residual": -1.0, "abs_error": 1.0},
            {"model": "a", "bucket": "high", "true_rate": 100.0, "pred_rate": 97.0, "residual": -3.0, "abs_error": 3.0},
            {"model": "b", "bucket": "low", "true_rate": 10.0, "pred_rate": 8.0, "residual": -2.0, "abs_error": 2.0},
        ]

        summary = summarize_bucket_metrics(rows, group_keys=("model", "bucket"))
        a_low = next(row for row in summary if row["model"] == "a" and row["bucket"] == "low")
        a_high = next(row for row in summary if row["model"] == "a" and row["bucket"] == "high")
        b_low = next(row for row in summary if row["model"] == "b" and row["bucket"] == "low")

        self.assertAlmostEqual(a_low["sse_share"], 0.1)
        self.assertAlmostEqual(a_high["sse_share"], 0.9)
        self.assertAlmostEqual(b_low["sse_share"], 1.0)


if __name__ == "__main__":
    unittest.main()
