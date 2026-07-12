import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11CoupledTokenAnalysisTests(unittest.TestCase):
    def test_confusion_rows_count_true_pred_pairs(self):
        from analyze_v11_coupled_token_policy import make_confusion_rows

        token_true = np.array([0, 1, 1, 2, 2, 2])
        token_pred = np.array([0, 1, 2, 2, 1, 2])
        rows = make_confusion_rows(token_true, token_pred, group_name="rb", vocab_size=3)

        by_pair = {(row["true_token"], row["pred_token"]): row["count"] for row in rows}
        self.assertEqual(by_pair[(0, 0)], 1)
        self.assertEqual(by_pair[(1, 1)], 1)
        self.assertEqual(by_pair[(1, 2)], 1)
        self.assertEqual(by_pair[(2, 1)], 1)
        self.assertEqual(by_pair[(2, 2)], 2)
        self.assertAlmostEqual(sum(row["count"] for row in rows), 6)

    def test_histogram_rows_include_vocab_values(self):
        from analyze_v11_coupled_token_policy import make_token_histogram_rows

        vocab_values = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 25.0],
                [0.0, 2.0, 50.0],
            ],
            dtype=np.float32,
        )
        rows = make_token_histogram_rows(
            token_ids=np.array([1, 1, 2, 2, 2]),
            vocab_values=vocab_values,
            group_name="rb",
            prefix="pred",
        )

        by_token = {row["token"]: row for row in rows}
        self.assertEqual(by_token[1]["pred_count"], 2)
        self.assertEqual(by_token[2]["pred_count"], 3)
        self.assertEqual(by_token[1]["value"], "0:0.000000;1:1.000000;2:25.000000")

    def test_aggregate_rows_compare_action_totals_by_horizon_and_dim(self):
        from analyze_v11_coupled_token_policy import make_aggregate_rows

        true_value = np.zeros((2, 2, 3, 6), dtype=np.float32)
        pred_value = np.zeros_like(true_value)
        true_value[:, 0, :, 2] = 10.0
        pred_value[:, 0, :, 2] = 8.0
        true_value[:, 1, :, 4] = 2.0
        pred_value[:, 1, :, 4] = 3.0
        features = ["offload_count", "rb_task_count", "rb_total", "cpu_task_count", "cpu_total", "return_count"]

        rows = make_aggregate_rows(true_value, pred_value, features)
        by_key = {(row["horizon"], row["dim"]): row for row in rows}

        self.assertEqual(by_key[(0, 2)]["true_total"], 60.0)
        self.assertEqual(by_key[(0, 2)]["pred_total"], 48.0)
        self.assertEqual(by_key[(0, 2)]["total_error"], -12.0)
        self.assertEqual(by_key[(1, 4)]["true_total"], 12.0)
        self.assertEqual(by_key[(1, 4)]["pred_total"], 18.0)
        self.assertEqual(by_key[(1, 4)]["active_count_true"], 6)

    def test_write_csv_accepts_rows_with_different_count_columns(self):
        from analyze_v11_coupled_token_policy import write_csv

        path = PROJECT_ROOT / "artifacts" / "reports" / "v11_next_plan_20260620" / "_tmp_union_columns.csv"
        rows = [
            {"group": "rb", "token": 1, "true_count": 2},
            {"group": "rb", "token": 1, "pred_count": 3},
        ]

        write_csv(path, rows)
        content = path.read_text(encoding="utf-8")

        self.assertIn("true_count", content.splitlines()[0])
        self.assertIn("pred_count", content.splitlines()[0])
        self.assertIn("2", content)
        self.assertIn("3", content)
        path.unlink()

    def test_apply_activity_mask_zeroes_inactive_decoded_values(self):
        from analyze_v11_coupled_token_policy import apply_activity_mask

        value = np.ones((1, 1, 3, 2), dtype=np.float32)
        prob = np.array([[[[0.9, 0.1], [0.2, 0.8], [0.7, 0.7]]]], dtype=np.float32)

        masked = apply_activity_mask(value, prob, threshold=0.5)

        expected = np.array([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=np.float32)
        self.assertTrue(np.array_equal(masked, expected))


if __name__ == "__main__":
    unittest.main()
