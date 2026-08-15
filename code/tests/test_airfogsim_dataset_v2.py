from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_dataset_v2 import (
    aggregate_metric_reports,
    build_multiseed_window_index,
    validate_multiseed_window_index,
)


class AirFogSimDatasetV2Tests(unittest.TestCase):
    def test_windows_are_contiguous_and_never_cross_seed_splits(self):
        times_by_seed = {
            0: [round(index * 0.1, 1) for index in range(1, 13)],
            1: [round(index * 0.1, 1) for index in range(1, 13)],
        }
        splits = {0: "dev_train", 1: "dev_validation"}

        rows = build_multiseed_window_index(
            times_by_seed,
            splits,
            history_steps=3,
            horizon_steps=2,
        )
        validation = validate_multiseed_window_index(
            rows,
            times_by_seed,
            splits,
            history_steps=3,
            horizon_steps=2,
        )

        self.assertEqual(16, len(rows))
        self.assertEqual({"dev_train", "dev_validation"}, {row["split"] for row in rows})
        self.assertTrue(validation["window_index_valid"])
        self.assertEqual([], validation["failed_checks"])
        self.assertTrue(all(row["seed"] == 0 for row in rows if row["split"] == "dev_train"))
        self.assertEqual(0.3, rows[0]["decision_time"])
        self.assertEqual(0.4, rows[0]["label_start_time"])
        self.assertEqual(0.5, rows[0]["label_end_time"])

    def test_nonuniform_time_grid_is_rejected_before_windowing(self):
        with self.assertRaisesRegex(ValueError, "uniform time grid"):
            build_multiseed_window_index(
                {0: [0.1, 0.2, 0.4, 0.5, 0.6]},
                {0: "dev_train"},
                history_steps=2,
                horizon_steps=1,
            )

    def test_metric_aggregation_preserves_seed_evidence_and_status(self):
        reports = {
            0: {
                "metrics": [
                    {
                        "name": "task_completion_rate",
                        "value": 0.5,
                        "numerator": 1,
                        "denominator": 2,
                        "sample_count": 2,
                        "unit": "ratio",
                        "status": "available",
                        "source": "task",
                    },
                    {
                        "name": "action_regret",
                        "value": None,
                        "numerator": None,
                        "denominator": None,
                        "sample_count": 0,
                        "unit": "not available",
                        "status": "not_computable",
                        "source": "required",
                    },
                ]
            },
            1: {
                "metrics": [
                    {
                        "name": "task_completion_rate",
                        "value": 0.75,
                        "numerator": 3,
                        "denominator": 4,
                        "sample_count": 4,
                        "unit": "ratio",
                        "status": "available",
                        "source": "task",
                    }
                ]
            },
        }

        rows, summary = aggregate_metric_reports(reports, {0: "dev_train", 1: "dev_validation"})

        completion = summary["task_completion_rate"]
        self.assertEqual(0.625, completion["mean"])
        self.assertEqual(2, completion["available_seed_count"])
        self.assertEqual(1, summary["action_regret"]["not_computable_seed_count"])
        self.assertIn("numerator", rows[0])
        self.assertEqual({0, 1}, {row["seed"] for row in rows if row["name"] == "task_completion_rate"})


if __name__ == "__main__":
    unittest.main()
