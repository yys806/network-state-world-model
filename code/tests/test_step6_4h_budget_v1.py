import sys
import unittest
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts')]
from step6_4h_s_cem_budget_v1 import qualification, planned_cases, statistics
from unittest.mock import patch
from pi_jwm.step6_4d_full_cohort_b512_v1 import full_pair_analysis
from test_step6_4g_runner_v1 import synthetic_row
from run_step6_4g_requalification_v1 import validate_result
import copy


class BudgetGateTests(unittest.TestCase):
    def test_no_research_decision_even_when_pass(self):
        stats = dict(paired_cases=320, provenance=True, budget=True,
                     scorer_errors=0, only_left=0, only_right=0, primary_loss=0,
                     mean_512=10, mean_1024=20, median_512=9, median_1024=18)
        self.assertEqual(qualification(stats)['verdict'], 'PASS')
        self.assertEqual(qualification(stats)['final_budget'], 'RESEARCHER_DECISION_PENDING')
        for field, value in [('paired_cases',319), ('provenance',False),
                             ('budget',False), ('scorer_errors',1), ('only_left',1),
                             ('only_right',1), ('primary_loss',1), ('mean_512',20),
                             ('median_512',18)]:
            with self.subTest(field=field):
                self.assertEqual(qualification({**stats, field:value})['verdict'], 'FAIL')

    def test_fixed_complete_cohort_no_other_method_or_budget(self):
        anchors = [f'a{i}' for i in range(64)]
        rows = planned_cases(anchors)
        self.assertEqual(len(rows),320)
        self.assertEqual({r['budget'] for r in rows},{512})
        self.assertEqual({r['method'] for r in rows},{'S-CEM'})
        self.assertEqual({r['seed'] for r in rows},set(range(6311,6316)))
        with self.assertRaises(ValueError): planned_cases(anchors[:-1])

    def test_six_categories_and_first_difference_oracle(self):
        left={};right={}
        for i in range(64):
            for seed in range(6311,6316):
                key=(f'a{i}',seed);category=(i*5+seed-6311)%6
                a=[0,0,0,0,0];b=a.copy()
                if category==0: b=None
                elif category==1: a=None
                elif category==2: b[3]=1
                elif category==3: a[2]=1
                elif category==5: a=b=None
                left[key]=a;right[key]=b
        paired,components=full_pair_analysis(left,right)
        self.assertEqual(sum(paired['six_categories'].values()),320)
        self.assertTrue(all(v>0 for v in paired['six_categories'].values()))
        self.assertEqual(components['PRIMARY_OBJECTIVE_DEGRADATION_COUNT'],53)
        self.assertEqual(components['components']['J_Burden']['B512_better'],53)
        self.assertEqual(components['BURDEN_EFFORT_ONLY_DEGRADATION_COUNT'],0)
        with self.assertRaises(ValueError): full_pair_analysis(dict(list(left.items())[:-1]),right)

    def test_resume_budget_scorer_and_float_failures(self):
        row=synthetic_row('S-CEM',6311,512,4,.2)
        row.update(phase='H',execution_config_id='new-H-only')
        expected={k:row[k] for k in ('phase','execution_config_id','budget','iterations','elite_ratio','seed','sample_id','checkpoint_sha256','gpu_model','batch_size','precision','source_sha256')}
        validate_result(row,expected)
        for k in expected:
            changed=copy.deepcopy(row);changed[k]='wrong'
            with self.assertRaises(ValueError): validate_result(changed,expected)

        for mutate in (lambda r:r['outcome']['budget_receipt'].__setitem__('N_unique_transition_evals',511),
                       lambda r:r['support_horizon_counts'].__setitem__('SCORER_EXCEPTION',1),
                       lambda r:r['outcome'].__setitem__('best_objective',[float('nan')]*5)):
            changed=copy.deepcopy(row);mutate(changed)
            with self.assertRaises(ValueError): validate_result(changed,expected)

    def test_summary_qualification_from_synthetic_raw_receipts(self):
        anchors=[f'a{i}' for i in range(64)];left=[];right=[]
        for c in planned_cases(anchors):
            for budget,rows in ((512,left),(1024,right)):
                r=synthetic_row('S-CEM',c['seed'],budget,4,.2);r['sample_id']=c['sample_id']
                r['outcome']['budget_receipt']['wall_clock_seconds_diagnostic_only']=budget/512
                r['case_total_seconds']=budget/512+1
                r['score_residuals']={'H4_SUPPORT_BOUNDARY:UNSUPPORTED_FUTURE_RETURN_BIRTH':8}
                rows.append(r)
        with patch('step6_4h_s_cem_budget_v1.read',return_value={'anchors':anchors}):
            result=statistics(left,right)
        self.assertEqual(result['qualification.json']['verdict'],'PASS')
        self.assertEqual(result['paired.json']['six_categories']['both_unscoreable'],320)
        self.assertEqual(result['summary.json']['512']['return_birth_boundary'],2560)
        self.assertEqual(result['runtime_tradeoff.json']['mean_reduction'],.5)

if __name__ == '__main__': unittest.main()
