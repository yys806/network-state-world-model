import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11RbTotalRiskControlledScaleTest(unittest.TestCase):
    def test_selects_best_active_rate_under_link_risk_constraint(self):
        from diagnose_v11_rb_total_risk_controlled_scale import select_risk_controlled_candidate

        rows = [
            {"candidate": "identity", "split": "val", "active_rate_rmse": 234.0, "link_rmse": 80.0},
            {"candidate": "aggressive", "split": "val", "active_rate_rmse": 226.0, "link_rmse": 96.0},
            {"candidate": "conservative", "split": "val", "active_rate_rmse": 229.0, "link_rmse": 84.0},
        ]

        selected = select_risk_controlled_candidate(rows, max_link_delta=6.0, min_active_improvement=1.0)

        self.assertEqual(selected["candidate"], "conservative")

    def test_falls_back_to_identity_when_no_candidate_satisfies_constraints(self):
        from diagnose_v11_rb_total_risk_controlled_scale import select_risk_controlled_candidate

        rows = [
            {"candidate": "identity", "split": "val", "active_rate_rmse": 234.0, "link_rmse": 80.0},
            {"candidate": "risky", "split": "val", "active_rate_rmse": 220.0, "link_rmse": 95.0},
        ]

        selected = select_risk_controlled_candidate(rows, max_link_delta=4.0, min_active_improvement=1.0)

        self.assertEqual(selected["candidate"], "identity")


if __name__ == "__main__":
    unittest.main()
