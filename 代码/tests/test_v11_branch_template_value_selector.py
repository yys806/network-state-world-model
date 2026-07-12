import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))


class V11BranchTemplateValueSelectorTest(unittest.TestCase):
    def test_objective_penalizes_link_budget_and_high_load(self):
        from compare_v11_branch_template_value_selector import objective_score

        clean = {
            'active_rate_rmse': 210.0,
            'link_rmse': 86.0,
            'activity_f1': 0.03,
            'high_load_active_rate_rmse': 215.0,
        }
        risky = {
            'active_rate_rmse': 210.0,
            'link_rmse': 100.0,
            'activity_f1': 0.01,
            'high_load_active_rate_rmse': 260.0,
        }

        self.assertLess(objective_score(clean), objective_score(risky))

    def test_validation_selection_returns_matched_test_not_test_only(self):
        from compare_v11_branch_template_value_selector import select_matched_test_by_validation

        rows = [
            {'candidate': 'a', 'split': 'val', 'active_rate_rmse': 205.0, 'link_rmse': 86.0, 'activity_f1': 0.03},
            {'candidate': 'a', 'split': 'test', 'active_rate_rmse': 230.0, 'link_rmse': 86.0, 'activity_f1': 0.03},
            {'candidate': 'b', 'split': 'val', 'active_rate_rmse': 210.0, 'link_rmse': 86.0, 'activity_f1': 0.03},
            {'candidate': 'b', 'split': 'test', 'active_rate_rmse': 190.0, 'link_rmse': 86.0, 'activity_f1': 0.03},
        ]

        selected = select_matched_test_by_validation(rows)

        self.assertEqual(selected['val']['candidate'], 'a')
        self.assertEqual(selected['test']['candidate'], 'a')
        self.assertEqual(float(selected['test']['active_rate_rmse']), 230.0)

    def test_validation_selection_skips_missing_or_nan_test_metrics(self):
        from compare_v11_branch_template_value_selector import select_matched_test_by_validation

        rows = [
            {'candidate': 'bad', 'split': 'val', 'active_rate_rmse': 100.0, 'link_rmse': 80.0, 'activity_f1': 0.03},
            {'candidate': 'bad', 'split': 'test', 'active_rate_rmse': 'nan', 'link_rmse': 80.0, 'activity_f1': 0.03},
            {'candidate': 'good', 'split': 'val', 'active_rate_rmse': 120.0, 'link_rmse': 80.0, 'activity_f1': 0.03},
            {'candidate': 'good', 'split': 'test', 'active_rate_rmse': 130.0, 'link_rmse': 80.0, 'activity_f1': 0.03},
        ]

        selected = select_matched_test_by_validation(rows)

        self.assertEqual(selected['val']['candidate'], 'good')
        self.assertEqual(float(selected['test']['active_rate_rmse']), 130.0)

    def test_oracle_gap_marks_sub200_potential_when_oracle_is_strong(self):
        from compare_v11_branch_template_value_selector import oracle_gap_summary

        summary = oracle_gap_summary(
            learned_active_rmse=217.0,
            oracle_active_rmse=122.0,
            autonomous_reference=217.237962,
            target_rmse=200.0,
        )

        self.assertTrue(summary['oracle_has_sub200_potential'])
        self.assertGreater(summary['learned_to_oracle_gap'], 90.0)
        self.assertGreater(summary['needed_improvement_to_target'], 16.0)

    def test_recommendation_focuses_on_support_when_oracle_gap_is_large(self):
        from compare_v11_branch_template_value_selector import recommend_next_action

        recommendation = recommend_next_action(
            matched_test_active_rmse=219.0,
            matched_test_link_rmse=86.0,
            oracle_active_rmse=122.0,
            target_rmse=200.0,
        )

        self.assertEqual(recommendation['decision'], 'continue_cpu_support_value_research')
        self.assertIn('support/value ranking', recommendation['reason'])

    def test_true_value_oracle_scope_rows_are_diagnostic(self):
        from compare_v11_branch_template_value_selector import is_diagnostic_row

        row = {
            'candidate': 'pred_rank__true_value__per_sample__top64',
            'experiment': 'pi_jwm_v11_rb_total_oracle_value_scope_diagnostic_20260622',
            'source_csv': 'artifacts/experiments/pi_jwm_v11_rb_total_oracle_value_scope_diagnostic_20260622/oracle_value_scope_results.csv',
        }

        self.assertTrue(is_diagnostic_row(row))

    def test_smoke_rows_are_low_confidence_by_default(self):
        from compare_v11_branch_template_value_selector import is_smoke_row

        row = {
            'experiment': 'pi_jwm_v11_rb_total_latent_identifiability_smoke_20260622',
            'source_csv': 'artifacts/experiments/pi_jwm_v11_rb_total_latent_identifiability_smoke_20260622/latent_identifiability_results.csv',
        }

        self.assertTrue(is_smoke_row(row))


if __name__ == '__main__':
    unittest.main()
