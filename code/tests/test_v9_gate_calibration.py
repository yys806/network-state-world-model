import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V9GateCalibrationTest(unittest.TestCase):
    def test_calibrate_activity_gate_applies_temperature_and_power(self):
        from evaluate_v9_hurdle_gate_calibration import calibrate_activity_gate

        prob = np.array([0.25, 0.5, 0.75], dtype=np.float64)

        sharpened = calibrate_activity_gate(prob, temperature=0.5, power=1.0)
        softened = calibrate_activity_gate(prob, temperature=2.0, power=1.0)
        powered = calibrate_activity_gate(prob, temperature=1.0, power=2.0)

        self.assertLess(sharpened[0], prob[0])
        self.assertAlmostEqual(sharpened[1], 0.5, places=6)
        self.assertGreater(sharpened[2], prob[2])
        self.assertGreater(softened[0], prob[0])
        self.assertLess(softened[2], prob[2])
        np.testing.assert_allclose(powered, prob**2)

    def test_gate_calibrated_rate_reports_active_and_link_metrics(self):
        from evaluate_v9_hurdle_gate_calibration import evaluate_gate_calibrated_rate

        activity_prob = np.array([[[[0.2], [0.8], [0.5]]]], dtype=np.float64)
        positive_rate = np.array([[[[10.0], [20.0], [40.0]]]], dtype=np.float64)
        true_rate = np.array([[[[0.0], [20.0], [40.0]]]], dtype=np.float64)
        true_activity = np.array([[[[0.0], [1.0], [1.0]]]], dtype=np.float64)

        base = evaluate_gate_calibrated_rate(
            activity_prob,
            positive_rate,
            true_rate,
            true_activity,
            temperature=1.0,
            power=1.0,
        )
        bypass = evaluate_gate_calibrated_rate(
            activity_prob,
            positive_rate,
            true_rate,
            true_activity,
            temperature=1.0,
            power=0.0,
        )

        self.assertIn("active_rate", base)
        self.assertIn("link_rate", base)
        self.assertGreater(base["active_rate"]["active_rmse"], 0.0)
        self.assertAlmostEqual(bypass["active_rate"]["active_rmse"], 0.0, places=6)
        self.assertLess(bypass["link_rate"]["rmse"], base["link_rate"]["rmse"])

    def test_gate_calibrated_rate_can_scale_positive_rate(self):
        from evaluate_v9_hurdle_gate_calibration import evaluate_gate_calibrated_rate

        activity_prob = np.array([[[[1.0], [0.1]]]], dtype=np.float64)
        positive_rate = np.array([[[[20.0], [20.0]]]], dtype=np.float64)
        true_rate = np.array([[[[10.0], [0.0]]]], dtype=np.float64)
        true_activity = np.array([[[[1.0], [0.0]]]], dtype=np.float64)

        base = evaluate_gate_calibrated_rate(
            activity_prob,
            positive_rate,
            true_rate,
            true_activity,
            temperature=1.0,
            power=1.0,
            positive_rate_scale=1.0,
        )
        scaled = evaluate_gate_calibrated_rate(
            activity_prob,
            positive_rate,
            true_rate,
            true_activity,
            temperature=1.0,
            power=1.0,
            positive_rate_scale=0.5,
        )

        self.assertLess(scaled["link_rate"]["rmse"], base["link_rate"]["rmse"])
        self.assertAlmostEqual(scaled["positive_rate_scale"], 0.5)

    def test_gate_calibrated_rate_can_use_hard_rate_gate(self):
        from evaluate_v9_hurdle_gate_calibration import evaluate_gate_calibrated_rate

        activity_prob = np.array([[[[0.9], [0.2]]]], dtype=np.float64)
        positive_rate = np.array([[[[10.0], [100.0]]]], dtype=np.float64)
        true_rate = np.array([[[[10.0], [0.0]]]], dtype=np.float64)
        true_activity = np.array([[[[1.0], [0.0]]]], dtype=np.float64)

        soft = evaluate_gate_calibrated_rate(
            activity_prob,
            positive_rate,
            true_rate,
            true_activity,
            temperature=1.0,
            power=1.0,
            threshold=0.5,
            rate_gate_mode="soft",
        )
        hard = evaluate_gate_calibrated_rate(
            activity_prob,
            positive_rate,
            true_rate,
            true_activity,
            temperature=1.0,
            power=1.0,
            threshold=0.5,
            rate_gate_mode="hard",
        )

        self.assertLess(hard["link_rate"]["rmse"], soft["link_rate"]["rmse"])
        self.assertEqual(hard["rate_gate_mode"], "hard")

    def test_selective_gate_softens_only_selected_edges(self):
        from evaluate_v9_hurdle_gate_calibration import calibrate_selective_activity_gate

        activity_prob = np.array([[[[0.04], [0.5], [0.8]]]], dtype=np.float64)
        positive_rate = np.array([[[[100.0], [10.0], [120.0]]]], dtype=np.float64)

        selective = calibrate_selective_activity_gate(
            activity_prob,
            positive_rate,
            temperature=1.0,
            power=0.5,
            min_activity_prob=0.1,
            min_positive_rate=50.0,
        )

        np.testing.assert_allclose(selective[..., 0, :], activity_prob[..., 0, :])
        np.testing.assert_allclose(selective[..., 1, :], activity_prob[..., 1, :])
        self.assertGreater(float(selective[..., 2, :].item()), float(activity_prob[..., 2, :].item()))

    def test_constrained_calibration_score_prefers_reportable_point(self):
        from evaluate_v9_hurdle_gate_calibration import constrained_calibration_score

        fast_but_bad = {
            "active_rate": {"active_rmse": 280.0},
            "link_rate": {"rmse": 110.0},
            "activity": {"f1": 0.010},
        }
        slower_but_reportable = {
            "active_rate": {"active_rmse": 291.0},
            "link_rate": {"rmse": 89.0},
            "activity": {"f1": 0.030},
        }

        bad_score = constrained_calibration_score(
            fast_but_bad,
            min_f1=0.027,
            max_link_rmse=90.0,
            f1_penalty_weight=1000.0,
            link_penalty_weight=10.0,
        )
        good_score = constrained_calibration_score(
            slower_but_reportable,
            min_f1=0.027,
            max_link_rmse=90.0,
            f1_penalty_weight=1000.0,
            link_penalty_weight=10.0,
        )

        self.assertLess(good_score, bad_score)

    def test_select_best_calibration_can_use_constrained_active_rate(self):
        from evaluate_v9_hurdle_gate_calibration import select_best_row

        rows = [
            {
                "temperature": 1.0,
                "power": 0.5,
                "active_rate_rmse": 280.0,
                "link_rate_rmse": 110.0,
                "f1": 0.010,
                "score": 280.0,
            },
            {
                "temperature": 1.0,
                "power": 1.0,
                "active_rate_rmse": 291.0,
                "link_rate_rmse": 89.0,
                "f1": 0.030,
                "score": 291.0,
            },
        ]

        best = select_best_row(
            rows,
            selection_metric="constrained_active_rate",
            min_f1=0.027,
            max_link_rmse=90.0,
            f1_penalty_weight=1000.0,
            link_penalty_weight=10.0,
        )

        self.assertEqual(best["power"], 1.0)
        self.assertTrue(best["meets_constraints"])

    def test_select_best_calibration_can_use_explicit_thresholds(self):
        from evaluate_v9_hurdle_gate_calibration import select_best_calibration

        predictions = {
            "link_activity_prob": np.array([[[[0.2], [0.6], [0.9]]]], dtype=np.float64),
            "link_positive_rate_pred": np.array([[[[5.0], [10.0], [20.0]]]], dtype=np.float64),
            "link_rate_true": np.array([[[[0.0], [10.0], [20.0]]]], dtype=np.float64),
            "link_activity_true": np.array([[[[0.0], [1.0], [1.0]]]], dtype=np.float64),
        }

        best, rows = select_best_calibration(
            predictions,
            temperatures=[1.0],
            powers=[1.0],
            selection_metric="active_rate_rmse",
            thresholds=[0.8],
        )

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(best["threshold"], 0.8)

    def test_resolve_checkpoint_paths_supports_globbed_metric_checkpoints(self):
        from evaluate_v9_hurdle_gate_calibration import resolve_checkpoint_paths

        with self.subTest("globbed checkpoints"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                exp_dir = Path(tmpdir)
                checkpoint_dir = exp_dir / "checkpoints"
                checkpoint_dir.mkdir()
                (checkpoint_dir / "v8_dual_best.pt").write_bytes(b"main")
                (checkpoint_dir / "v8_dual_best_val_active_rate_rmse.pt").write_bytes(b"active")
                (checkpoint_dir / "v8_dual_last.pt").write_bytes(b"last")

                paths = resolve_checkpoint_paths(exp_dir, "v8_dual_best*.pt")

        self.assertEqual([path.name for path in paths], ["v8_dual_best.pt", "v8_dual_best_val_active_rate_rmse.pt"])


if __name__ == "__main__":
    unittest.main()
