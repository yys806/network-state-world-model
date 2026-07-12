import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11StrategySelectorTest(unittest.TestCase):
    def test_analyze_pairs_val_and_test_and_groups_alpha(self):
        from analyze_v11_strategy_selector import analyze

        rows = [
            {
                "candidate": "a",
                "split": "val",
                "family": "graph_support_generator",
                "active_rate_rmse": "10",
                "link_rmse": "80",
                "activity_f1": "0.02",
                "selection_group_mode": "g",
                "top_k": "4",
                "alpha": "1.0",
                "new_edge_value_cap": "1.0",
            },
            {
                "candidate": "a",
                "split": "test",
                "family": "graph_support_generator",
                "active_rate_rmse": "30",
                "link_rmse": "81",
                "activity_f1": "0.02",
            },
            {
                "candidate": "b",
                "split": "val",
                "family": "graph_support_generator",
                "active_rate_rmse": "20",
                "link_rmse": "79",
                "activity_f1": "0.03",
                "selection_group_mode": "g",
                "top_k": "8",
                "alpha": "0.6",
                "new_edge_value_cap": "4.0",
            },
            {
                "candidate": "b",
                "split": "test",
                "family": "graph_support_generator",
                "active_rate_rmse": "15",
                "link_rmse": "82",
                "activity_f1": "0.03",
            },
        ]

        report = analyze(rows)

        self.assertEqual(report["paired_count"], 2)
        self.assertEqual(report["best_val_active"]["candidate"], "a")
        self.assertEqual(report["best_test_active"]["candidate"], "b")
        self.assertLess(report["val_test_active_pearson"], 0.0)
        self.assertEqual(report["group_by_alpha"][0]["alpha"], "0.6")


if __name__ == "__main__":
    unittest.main()
