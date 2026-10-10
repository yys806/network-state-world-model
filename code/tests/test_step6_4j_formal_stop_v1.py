import json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
from pi_jwm.step6_4j_pilot_v1 import FormalPilotGate, formal_result


def executed(i, episode='a'):
    fresh={'capture_event_id':f'fresh-{episode}-{i}', 'simulation_time_s':i+.1,
           'capture_method':'fresh_direct_real_environment_read', 'capture_phase':'loop_start_decision',
           'frame_index':i+1, 'slot_transfer_events':[], 'communication_observation':{}}
    return {'status':'EXECUTED','dispatch':'FALLBACK_A','search_attempt':i+1,
            'execution_identity':{'device':'cuda','precision':'FP32','batch_size':16,'execution_config_id':'id'},
            'budget_receipt':{'B_WM':512,'N_unique_transition_evals':512},'internal_search_seconds':.2,
            'decision_token':[f'old-{episode}-{i}',i,i], 'setter_attempted':True,
            'environment_step_attempted':True,'executed_horizon':1,'simulation_time_before':i,
            'simulation_time_after':i+.1,'fresh_observation':fresh,'history_outcome':fresh,
            'action':{'comm':{'entries':[]}},'history_action':{'comm':{'entries':[]}},
            'root':{'state':f'root-{episode}-{i}'},'planning_validation':'PASS','action_history_validation':'PASS'}

class FormalGateTests(unittest.TestCase):
    def gate(self):return FormalPilotGate(['a','b'],'id')
    def test_first_success_and_16_completed(self):
        g=self.gate(); rows={x:[] for x in ('a','b')}
        for ep in rows:
            for i in range(8):
                n=g.begin_search(ep);r=executed(i,ep);r['search_attempt']=n
                rows[ep].append(r);g.accept(ep,r)
        result=formal_result(['a','b'],rows,g.attempts,None,'id')
        self.assertEqual((result['status'],result['exit_code']),('COMPLETED',0))
        self.assertEqual(result['env_step_count'],16);self.assertTrue(g.qualified)
        with self.assertRaisesRegex(RuntimeError,'BUDGET'):g.begin_search('b')
    def test_first_failure_never_starts_second_episode(self):
        for reason in ('BUDGET_INCOMPLETE_OR_MISMATCH','PILOT_SINGLE_PLAN_TIMEOUT','ENV_STEP_FAILURE','SCORER_ERROR'):
            g=self.gate();g.begin_search('a');g.accept('a',{'status':'STOPPED','reason':reason})
            self.assertFalse(g.qualified);self.assertEqual(g.stop['status'],'BLOCKED')
            with self.assertRaisesRegex(RuntimeError,'PILOT_STOPPED'):g.begin_search('b')
            self.assertEqual(len([x for x in g.attempts if x['episode']=='b']),0)
    def test_partial_only_after_qualification(self):
        g=self.gate();g.begin_search('a');g.accept('a',executed(0))
        g.resource_stop('PILOT_TIME_LIMIT')
        r=formal_result(['a','b'],{'a':[executed(0)],'b':[]},g.attempts,g.stop,'id')
        self.assertEqual((r['status'],r['exit_code']),('PARTIAL',2))
        g=self.gate();g.resource_stop('PILOT_TIME_LIMIT');self.assertEqual(g.stop['status'],'BLOCKED')
    def test_bad_fresh_identity_budget_and_duplicate_are_blocked(self):
        for key,value in [('budget_receipt',{'B_WM':512,'N_unique_transition_evals':511}),('planning_validation','FAIL'),('history_outcome',{}),('internal_search_seconds',float('nan'))]:
            g=self.gate();g.begin_search('a');r=executed(0);r[key]=value;g.accept('a',r)
            self.assertEqual(g.stop['status'],'BLOCKED')
        g=self.gate();g.begin_search('a');g.accept('a',executed(0));g.begin_search('a');g.accept('a',executed(0))
        self.assertEqual(g.stop['status'],'BLOCKED')
    def test_no_partial_without_approved_reason_and_no_claim_from_summary(self):
        r=formal_result(['a','b'],{'a':[executed(0)],'b':[]},[{'episode':'a','number':1}],None,'id')
        self.assertEqual((r['status'],r['exit_code']),('BLOCKED',1))
    def test_legal_fallback_c_is_partial_after_qualification(self):
        g=self.gate();g.begin_search('a');g.accept('a',executed(0));g.begin_search('a')
        r={'status':'STOPPED','reason':'ACTION_BRIDGE_REJECTED','fallback_reason':'NO_SCOREABLE_H4','dispatch':'FAIL_CLOSED_C','setter_attempted':False,'environment_step_attempted':False,'budget_receipt':{'B_WM':512,'N_unique_transition_evals':512},'planning_validation':'PASS','action_history_validation':'PASS'}
        g.accept('a',r);self.assertEqual(g.stop['status'],'PARTIAL')
    def test_final_rejects_omitted_failed_search(self):
        g=self.gate();g.begin_search('a');g.accept('a',executed(0));g.begin_search('a')
        r=formal_result(['a','b'],{'a':[executed(0)],'b':[]},g.attempts,{'status':'PARTIAL','reason':'PILOT_TIME_LIMIT'},'id')
        self.assertEqual(r['status'],'BLOCKED')

if __name__=='__main__':unittest.main()
