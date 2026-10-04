"""Phase transitions and researcher frozen fallback/budget decisions."""
import sys, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts'),str(ROOT/'code/tests')]
from test_step6_4e_fallback_v1 import FallbackTests
from pi_jwm.step6_4g_requalification_v1 import (
    phase_cases, require_phase_parent, budget_requalification_gate,
    dispatch_frozen_fallback, FallbackCounters)

class RequalificationTests(unittest.TestCase):
    def configs(self):return {'S-CEM':{'K':3,'elite_ratio':.2},'MH-CEM':{'K':4,'elite_ratio':.1}}
    def test_full_grid_and_phase_condition(self):
        train=tuple('t'+str(i) for i in range(32));val=tuple('v'+str(i) for i in range(64))
        self.assertEqual(len(phase_cases('T',train,val)),768)
        self.assertEqual(len(phase_cases('A',train,val,self.configs())),960)
        self.assertEqual(len(phase_cases('B',train,val,self.configs())),320)
        self.assertEqual({c['budget'] for c in phase_cases('A',train,val,self.configs())},{1024})
        with self.assertRaises(ValueError):require_phase_parent('A',None,None)
        with self.assertRaises(ValueError):require_phase_parent('B',{'verdict':'PASS'}, {'verdict':'PASS','selected_method':'S-CEM'})
        require_phase_parent('B',{'verdict':'PASS'},{'verdict':'PASS','selected_method':'MH-CEM'})
    def test_budget_gate_fails_each_frozen_condition(self):
        good={'paired_cases':320,'scorer_exceptions':0,'scorer_inconsistencies':0,
          'only_B512_scoreable':0,'only_B1024_scoreable':0,'PRIMARY_OBJECTIVE_DEGRADATION_COUNT':0,
          'mean_512':50.,'mean_1024':100.,'median_512':40.,'median_1024':80.,'budget_accounting_passed':True}
        self.assertTrue(budget_requalification_gate(good)['passed'])
        for key,bad in [('paired_cases',319),('scorer_exceptions',1),('scorer_inconsistencies',1),
          ('only_B512_scoreable',1),('only_B1024_scoreable',1),('PRIMARY_OBJECTIVE_DEGRADATION_COUNT',1),
          ('mean_512',100.),('median_512',80.),('budget_accounting_passed',False)]:
            r=budget_requalification_gate({**good,key:bad});self.assertFalse(r['passed'],key)
            self.assertEqual(r['closed_loop_budget'],'RESEARCHER_DECISION_PENDING')
    def test_fallback_A_only_for_valid_no_scoreable_and_critical_C(self):
        domain,decision,tasks=FallbackTests().inputs()
        counters=FallbackCounters()
        # Setter execution deliberately injected success; this is dispatch logic,
        # actual native bridge has separate real 6.4F evidence.
        with patch('pi_jwm.step6_4g_requalification_v1.execute_fallback',return_value={'action_executed':True,'terminate_episode':False}):
            r=dispatch_frozen_fallback('NO_SCOREABLE_H4',domain,decision,tasks,None,None,None,None,counters=counters)
        self.assertFalse(r['terminate_episode']);self.assertEqual(counters.A_execution_count,1)
        for reason in ('DOMAIN_EMPTY','REQUIRED_LIVE_OBSERVATION_MISSING','LIVE_SLOT_UNSUPPORTED','ACTION_BRIDGE_REJECTED','SETTER_FAILURE'):
            r=dispatch_frozen_fallback(reason,counters=counters)
            self.assertTrue(r['terminate_episode']);self.assertEqual(r['selected_candidate'],'FAIL_CLOSED_TERMINATION')
        self.assertEqual(counters.fallback_trigger_count,6);self.assertEqual(counters.C_termination_count,5)
    def test_preparation_or_setter_failure_C_no_second_attempt(self):
        domain,decision,tasks=FallbackTests().inputs()
        with patch('pi_jwm.step6_4g_requalification_v1.prepare_fallback',side_effect=RuntimeError('prep')) as p:
            r=dispatch_frozen_fallback('NO_SCOREABLE_H4',domain,decision,tasks)
        self.assertTrue(r['terminate_episode']);self.assertEqual(p.call_count,1)
        with patch('pi_jwm.step6_4g_requalification_v1.execute_fallback',return_value={'action_executed':False,'terminate_episode':True,'reason':'SETTER_FAILURE'}) as e:
            r=dispatch_frozen_fallback('NO_SCOREABLE_H4',domain,decision,tasks)
        self.assertTrue(r['terminate_episode']);self.assertEqual(e.call_count,1)
        self.assertEqual(r['selected_candidate'],'FAIL_CLOSED_TERMINATION')
        missing=dict(decision);missing.pop('n_rb')
        r=dispatch_frozen_fallback('NO_SCOREABLE_H4',domain,missing,tasks)
        self.assertTrue(r['terminate_episode'])

if __name__=='__main__':unittest.main()
