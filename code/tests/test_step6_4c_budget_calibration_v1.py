"""Calibration orchestration oracles; no model or GPU execution."""
import unittest
from pi_jwm.step6_4c_budget_calibration_v1 import select_cohort, compare_pairs, validate_identity
from run_step6_4c_mh_budget_calibration_v1 import result_identity, raw_path, validate_scientific_result


class CalibrationTests(unittest.TestCase):
    def test_selection_is_balanced_order_invariant_and_outcome_blind(self):
        rows = [{"sample_id": f"{s}-{i}", "stratum": f"s{s}",
                 "concrete_count": i, "eligible_compute_tasks": s % 2}
                for s in range(7) for i in range(9)]
        a = select_cohort(rows)
        poisoned = [dict(r, best_objective=[999], scoreable=False) for r in reversed(rows)]
        self.assertEqual(a, select_cohort(poisoned))
        self.assertEqual(len(a), 16)
        self.assertEqual([sum(r['stratum'] == f's{s}' for r in a) for s in range(7)],
                         [3, 3, 2, 2, 2, 2, 2])
        with self.assertRaises(ValueError):
            select_cohort(rows + [rows[0]])

    def test_six_categories_and_missing_pair_rejected(self):
        good, bad = [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]
        pairs = [(good, None), (None, good), (good, bad), (bad, good),
                 (good, good), (None, None)]
        left = {(str(i), 6311): a for i, (a, b) in enumerate(pairs)}
        right = {(str(i), 6311): b for i, (a, b) in enumerate(pairs)}
        result = compare_pairs(left, right, expected_count=6)
        self.assertEqual(list(result['six_categories'].values()), [1] * 6)
        self.assertEqual((result['win'], result['tie'], result['loss']), (2, 2, 2))
        right.pop(('0', 6311))
        with self.assertRaises(ValueError):
            compare_pairs(left, right, expected_count=6)

    def test_resume_requires_exact_identity(self):
        expected = {'seed': 6311, 'budget': 256, 'locked_test': False,
                    'execution_config_id': 'new', 'source_sha256': {'solver': 'sha'}}
        validate_identity(dict(expected), expected)
        for key in expected:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_identity(dict(expected, **{key: 'wrong'}), expected)

    def test_namespace_budget_isolation_and_scorer_stop(self):
        self.assertNotEqual(raw_path('a',6311,256),raw_path('a',6311,512))
        self.assertIn('calibration',str(raw_path('a',6311,256)))
        config={'execution_config_id':'new','source_sha256':{'solver':'sha'},'checkpoint_sha256':'pt'}
        a=result_identity(config,'a',6311,256)
        b=result_identity(config,'a',6311,512)
        with self.assertRaises(ValueError):
            validate_identity(a,b)
        row=dict(a,outcome={'method':'MH-CEM','seed':6311,'budget':256,'best_objective':None,
                           'budget_receipt':{'B_WM':256,'N_unique_transition_evals':256}},
                 support_horizon_counts={},score_residuals={})
        validate_scientific_result(row)
        row['support_horizon_counts']['SCORER_EXCEPTION']=1
        with self.assertRaises(ValueError):
            validate_scientific_result(row)
        row['support_horizon_counts']={}
        row['outcome']['best_objective']=[0,float('nan'),0,0,0]
        with self.assertRaises(ValueError):
            validate_scientific_result(row)


if __name__ == '__main__':
    unittest.main()
