import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11CandidateSubgroupDiagnosisTest(unittest.TestCase):
    def test_summarize_candidate_groups_uses_active_mask(self):
        from diagnose_v11_candidate_subgroups import summarize_candidate_groups

        truth = np.array([[10.0, 100.0], [20.0, 200.0]], dtype=np.float32)
        active = np.array([[True, False], [True, True]])
        pred = np.array([[13.0, 500.0], [26.0, 194.0]], dtype=np.float32)
        groups = np.array([["a", "a"], ["a", "b"]], dtype=object)

        rows = summarize_candidate_groups(
            candidate="unit",
            split="val",
            truth=truth,
            active=active,
            pred=pred,
            group_name="bucket",
            group_labels=groups,
        )

        by_group = {row["group"]: row for row in rows}
        self.assertEqual(by_group["a"]["active_count"], 2)
        self.assertAlmostEqual(by_group["a"]["active_rmse"], np.sqrt((3.0**2 + 6.0**2) / 2.0), places=6)
        self.assertEqual(by_group["b"]["active_count"], 1)
        self.assertAlmostEqual(by_group["b"]["active_rmse"], 6.0, places=6)

    def test_robust_ranking_uses_min_count_for_worst_group(self):
        from diagnose_v11_candidate_subgroups import robust_rank_candidates

        overall_rows = [
            {"candidate": "a", "split": "val", "active_rmse": 10.0, "link_rmse": 3.0},
            {"candidate": "b", "split": "val", "active_rmse": 9.0, "link_rmse": 3.0},
            {"candidate": "a", "split": "test", "active_rmse": 11.0, "link_rmse": 3.0},
            {"candidate": "b", "split": "test", "active_rmse": 12.0, "link_rmse": 3.0},
        ]
        subgroup_rows = [
            {"candidate": "a", "split": "val", "group_name": "step", "group": "0", "active_count": 5, "active_rmse": 50.0},
            {"candidate": "a", "split": "val", "group_name": "step", "group": "1", "active_count": 1, "active_rmse": 500.0},
            {"candidate": "b", "split": "val", "group_name": "step", "group": "0", "active_count": 5, "active_rmse": 80.0},
        ]

        rows = robust_rank_candidates(overall_rows, subgroup_rows, min_group_count=2)

        self.assertEqual(rows[0]["candidate"], "b")
        self.assertEqual(rows[0]["val_worst_group_active_rmse"], 80.0)
        self.assertEqual(rows[1]["candidate"], "a")
        self.assertEqual(rows[1]["val_worst_group_active_rmse"], 50.0)

    def test_bucket_labels_are_stable_for_rate_ranges(self):
        from diagnose_v11_candidate_subgroups import make_rate_bucket_labels

        truth = np.array([[0.0, 49.9, 50.0, 250.0]], dtype=np.float32)
        labels = make_rate_bucket_labels(truth, bins=[0.0, 50.0, 250.0, float("inf")])

        self.assertEqual(labels.tolist(), [["0-50", "0-50", "50-250", "250-inf"]])


if __name__ == "__main__":
    unittest.main()
