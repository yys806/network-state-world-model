import json, sys, tempfile, unittest, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
from pi_jwm.step6_4j_pilot_v1 import FormalPilotGate, formal_result, audit_formal_result


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
        r=formal_result(['a','b'],{'a':[executed(0)],'b':[]},[{'episode':'a','search_attempt':1}],None,'id')
        self.assertEqual((r['status'],r['exit_code']),('BLOCKED',1))
    def test_legal_fallback_c_is_partial_after_qualification(self):
        g=self.gate();g.begin_search('a');g.accept('a',executed(0));g.begin_search('a')
        r={'status':'STOPPED','reason':'ACTION_BRIDGE_REJECTED','fallback_reason':'NO_SCOREABLE_H4','dispatch':'FAIL_CLOSED_C','setter_attempted':False,'environment_step_attempted':False,'budget_receipt':{'B_WM':512,'N_unique_transition_evals':512},'planning_validation':'PASS','action_history_validation':'PASS'}
        g.accept('a',r);self.assertEqual(g.stop['status'],'PARTIAL')
    def test_final_rejects_omitted_failed_search(self):
        g=self.gate();g.begin_search('a');g.accept('a',executed(0));g.begin_search('a')
        r=formal_result(['a','b'],{'a':[executed(0)],'b':[]},g.attempts,{'status':'PARTIAL','reason':'PILOT_TIME_LIMIT'},'id')
        self.assertEqual(r['status'],'BLOCKED')

    def write_disk(self, root, rows, attempts, stop=None, omit_episode=None, corrupt=None):
        root.mkdir(parents=True, exist_ok=True)
        receipt={'status':'RUNNING','search_attempts':attempts,'stop':stop,'episodes':[]}
        (root/'pilot_attempt.json').write_text(json.dumps(receipt), encoding='utf-8')
        for ep, ep_rows in rows.items():
            if ep == omit_episode: continue
            folder=root/ep; folder.mkdir()
            events=[]
            for row in ep_rows:
                events.append({'event':'SEARCH_INTENT','search_attempt':row['search_attempt']})
                budget=row.get('budget_receipt',{}).get('N_unique_transition_evals')
                if budget is not None: events.append({'event':'WM_TRANSITION_ATTEMPT','search_attempt':row['search_attempt'],'count':budget,'cumulative':budget})
                events.append({'event':'FINAL_DECISION','row':row})
                (folder/f"decision_{row['search_attempt']:02d}.json").write_text(json.dumps(row), encoding='utf-8')
            (folder/'journal.jsonl').write_text('\n'.join(json.dumps(x) for x in events), encoding='utf-8')
            (folder/'episode_receipt.json').write_text(json.dumps({'decision_count':len(ep_rows)}), encoding='utf-8')
        if corrupt:
            (root/corrupt).write_text('{broken', encoding='utf-8')

    def test_disk_completed_rebuilds_from_runner_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows={'a':[],'b':[]}; attempts=[]
            for ep in rows:
                for i in range(8):
                    n=len(attempts)+1; row=executed(i,ep); row['search_attempt']=n
                    rows[ep].append(row); attempts.append({'episode':ep,'search_attempt':n})
            self.write_disk(Path(tmp),rows,attempts)
            r=audit_formal_result(Path(tmp),['a','b'],'id')
            self.assertEqual((r['status'],r['exit_code'],r['env_step_count']),('COMPLETED',0,16))

    def test_disk_first_failure_has_no_second_episode_and_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            row={'status':'STOPPED','reason':'PILOT_SINGLE_PLAN_TIMEOUT','search_attempt':1}
            self.write_disk(Path(tmp),{'a':[row],'b':[]},[{'episode':'a','search_attempt':1}],omit_episode='b')
            r=audit_formal_result(Path(tmp),['a','b'],'id')
            self.assertEqual((r['status'],r['exit_code']),('BLOCKED',1))

    def test_disk_qualified_resource_stop_allows_unstarted_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            row=executed(0,'a'); row['search_attempt']=1
            self.write_disk(Path(tmp),{'a':[row],'b':[]},[{'episode':'a','search_attempt':1}],stop={'status':'PARTIAL','reason':'PILOT_TIME_LIMIT'},omit_episode='b')
            r=audit_formal_result(Path(tmp),['a','b'],'id')
            self.assertEqual((r['status'],r['exit_code']),('PARTIAL',2))

    def test_disk_missing_duplicate_corrupt_and_budget_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            row=executed(0,'a');row['search_attempt']=1
            self.write_disk(Path(tmp),{'a':[row],'b':[]},[{'episode':'a','search_attempt':1}],omit_episode='b')
            (Path(tmp)/'a'/'journal.jsonl').unlink()
            r=audit_formal_result(Path(tmp),['a','b'],'id');self.assertEqual(r['status'],'BLOCKED')
        with tempfile.TemporaryDirectory() as tmp:
            row=executed(0,'a');row['search_attempt']=1
            self.write_disk(Path(tmp),{'a':[row,row],'b':[]},[{'episode':'a','search_attempt':1}],omit_episode='b')
            r=audit_formal_result(Path(tmp),['a','b'],'id');self.assertEqual((r['status'],r['exit_code']),('BLOCKED',1))

    def test_disk_partial_fallback_c_and_incomplete_without_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            good=executed(0,'a');good['search_attempt']=1
            bad={'status':'STOPPED','reason':'ACTION_BRIDGE_REJECTED','fallback_reason':'NO_SCOREABLE_H4','dispatch':'FAIL_CLOSED_C','setter_attempted':False,'environment_step_attempted':False,'budget_receipt':{'B_WM':512,'N_unique_transition_evals':512},'planning_validation':'PASS','search_attempt':2}
            self.write_disk(Path(tmp),{'a':[good,bad],'b':[]},[{'episode':'a','search_attempt':1},{'episode':'a','search_attempt':2}],stop={'status':'PARTIAL','reason':'LEGAL_FALLBACK_C'},omit_episode='b')
            r=audit_formal_result(Path(tmp),['a','b'],'id');self.assertEqual((r['status'],r['exit_code']),('PARTIAL',2))
        with tempfile.TemporaryDirectory() as tmp:
            row=executed(0,'a');row['search_attempt']=1
            self.write_disk(Path(tmp),{'a':[row],'b':[]},[{'episode':'a','search_attempt':1}],omit_episode='b')
            r=audit_formal_result(Path(tmp),['a','b'],'id');self.assertEqual((r['status'],r['exit_code']),('BLOCKED',1))

    def run_cli(self, root):
        return subprocess.run([sys.executable, str(ROOT/'code/scripts/audit_step6_4j_formal_v1.py'), '--root', str(root), '--episodes', 'a', 'b', '--execution-config-id', 'id'], capture_output=True, text=True)

    def test_cli_exit_codes_are_rebuilt_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows={'a':[],'b':[]};attempts=[]
            for ep in rows:
                for i in range(8):
                    n=len(attempts)+1;row=executed(i,ep);row['search_attempt']=n;rows[ep].append(row);attempts.append({'episode':ep,'search_attempt':n})
            self.write_disk(Path(tmp),rows,attempts)
            out=self.run_cli(Path(tmp));self.assertEqual((json.loads(out.stdout)['status'],out.returncode),('COMPLETED',0))
        with tempfile.TemporaryDirectory() as tmp:
            row=executed(0,'a');row['search_attempt']=1
            self.write_disk(Path(tmp),{'a':[row],'b':[]},[{'episode':'a','search_attempt':1}],stop={'status':'PARTIAL','reason':'PILOT_TIME_LIMIT'},omit_episode='b')
            out=self.run_cli(Path(tmp));self.assertEqual((json.loads(out.stdout)['status'],out.returncode),('PARTIAL',2))
        with tempfile.TemporaryDirectory() as tmp:
            row={'status':'STOPPED','reason':'PILOT_SINGLE_PLAN_TIMEOUT','search_attempt':1}
            self.write_disk(Path(tmp),{'a':[row],'b':[]},[{'episode':'a','search_attempt':1}],omit_episode='b')
            out=self.run_cli(Path(tmp));self.assertEqual((json.loads(out.stdout)['status'],out.returncode),('BLOCKED',1))

    def test_cli_rejects_missing_duplicate_corrupt_and_budget_evidence(self):
        cases=[]
        for kind in ('missing_journal','duplicate_receipt','corrupt_journal','budget_mismatch'):
            with tempfile.TemporaryDirectory() as tmp:
                row=executed(0,'a');row['search_attempt']=1
                self.write_disk(Path(tmp),{'a':[row],'b':[]},[{'episode':'a','search_attempt':1}],omit_episode='b')
                if kind=='missing_journal': (Path(tmp)/'a'/'journal.jsonl').unlink()
                elif kind=='duplicate_receipt':
                    (Path(tmp)/'a'/'decision_02.json').write_text(json.dumps(row),encoding='utf-8')
                elif kind=='corrupt_journal': (Path(tmp)/'a'/'journal.jsonl').write_text('{bad',encoding='utf-8')
                else:
                    row['budget_receipt']['N_unique_transition_evals']=511
                    (Path(tmp)/'a'/'decision_01.json').write_text(json.dumps(row),encoding='utf-8')
                out=self.run_cli(Path(tmp)); cases.append((kind,json.loads(out.stdout)['status'],out.returncode))
        self.assertEqual(cases,[('missing_journal','BLOCKED',1),('duplicate_receipt','BLOCKED',1),('corrupt_journal','BLOCKED',1),('budget_mismatch','BLOCKED',1)])

if __name__=='__main__':unittest.main()
