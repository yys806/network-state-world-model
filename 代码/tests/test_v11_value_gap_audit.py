import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class V11ValueGapAuditTest(unittest.TestCase):
    def test_relative_gap_computes_needed_improvement_and_oracle_headroom(self):
        from diagnose_v11_value_gap_audit import relative_gap

        gap = relative_gap(current=223.0, oracle=122.0, target=200.0)

        self.assertAlmostEqual(gap["needed_to_target"], 23.0)
        self.assertAlmostEqual(gap["oracle_headroom"], 78.0)
        self.assertAlmostEqual(gap["current_to_oracle"], 101.0)
        self.assertTrue(gap["oracle_can_meet_target"])

    def test_classify_value_bottleneck_when_value_corr_is_low(self):
        from diagnose_v11_value_gap_audit import classify_bottleneck

        verdict = classify_bottleneck(
            learned_rmse=223.0,
            oracle_rmse=122.0,
            value_pearson=0.03,
            support_corr=0.08,
            target_rmse=200.0,
        )

        self.assertEqual(verdict["primary_bottleneck"], "value_magnitude_representation")
        self.assertIn("value predictor correlation is near zero", verdict["reasons"])

    def test_classify_support_bottleneck_when_support_corr_is_low_but_value_ok(self):
        from diagnose_v11_value_gap_audit import classify_bottleneck

        verdict = classify_bottleneck(
            learned_rmse=223.0,
            oracle_rmse=122.0,
            value_pearson=0.35,
            support_corr=0.02,
            target_rmse=200.0,
        )

        self.assertEqual(verdict["primary_bottleneck"], "support_ranking_generalization")


if __name__ == "__main__":
    unittest.main()
