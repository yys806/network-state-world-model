"""Resume and cluster-shape failure gates without a model forward."""
import unittest
from run_step6_4d_full_cohort_b512_v1 import validate_row, new_identity
from pi_jwm.step6_4d_full_cohort_b512_v1 import full_pair_analysis, SEEDS


class RunnerGateTests(unittest.TestCase):
    def test_resume_rejects_identity_parameter_and_nested_config_changes(self):
        c = {'execution_config_id': 'new', 'source_sha256': {'solver': 'sha'}, 'checkpoint_sha256': 'pt'}
        identity = new_identity(c, 'a', 6311, 'a'*40)
        row = dict(identity, parameter_digest='frozen', support_horizon_counts={}, score_residuals={},
                   outcome={'method': 'MH-CEM', 'seed': 6311, 'budget': 512, 'iterations': 4,
                            'elite_ratio': .1, 'best_objective': None, 'h4_scoreable_count': 0,
                            'budget_receipt': {'B_WM': 512, 'N_unique_transition_evals': 512}})
        validate_row(row, identity, 'frozen')
        for key in ('seed','budget','batch_size','execution_device','execution_config_id','checkpoint_sha256','protocol_freeze_commit'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_row(dict(row, **{key: 'wrong'}), identity, 'frozen')
        with self.assertRaises(ValueError): validate_row(row, identity, 'changed-parameters')
        for key, value in [('iterations',3), ('elite_ratio',.2)]:
            changed = dict(row, outcome=dict(row['outcome'], **{key:value}))
            with self.assertRaises(ValueError): validate_row(changed, identity, 'frozen')

    def test_full_cluster_bootstrap_keeps_all_five_seeds(self):
        a = {(f'a{i}', s): [0,0,0,0,0] for i in range(64) for s in SEEDS}
        b = {k: [0,0,0,0,1] for k in a}
        pair, components = full_pair_analysis(a,b)
        self.assertEqual((pair['win'],pair['tie'],pair['loss']), (320,0,0))
        self.assertEqual(pair['cluster_bootstrap']['ci95_lower'], 1)
        self.assertEqual(pair['cluster_bootstrap']['ci95_upper'], 1)
        self.assertEqual(components['components']['J_Effort']['B512_better'], 320)
        bad = dict(a); bad[('a0',6360)] = bad.pop(('a0',6311))
        with self.assertRaises(ValueError): full_pair_analysis(bad,bad)


if __name__ == '__main__': unittest.main()
