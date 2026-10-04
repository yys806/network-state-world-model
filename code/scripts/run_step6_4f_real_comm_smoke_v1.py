"""New provenance, patched eligibility, two separately frozen real one-step paths.

Reuse the same 6.4E observation/one-step instrumentation. Never overwrite 6.4E.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,gzip,json,sys
from pathlib import Path
from types import SimpleNamespace
import run_step6_4e_comm_fallback_audit_v1 as mechanism
from pi_jwm.step6_4b_live_bridge_v1 import validate_command
ROOT=mechanism.ROOT
OUT=ROOT/'code/artifacts/protocols/pi_jwm_step6_4f_comm_eligibility_v1_20261004'
mechanism.OUT=OUT
def freeze():
    parent=ROOT/'code/artifacts/protocols/pi_jwm_step6_4e_comm_fallback_v1_20261004/01_fixture_and_execution_freeze.json'
    old=json.loads(parent.read_text(encoding='utf-8'));p=ROOT/old['raw_path']
    assert mechanism.sha(p)==old['raw_sha256']
    raw=json.loads(gzip.decompress(p.read_bytes()));frame=old['anchor']
    prefix={**raw,'decisions':raw['decisions'][:frame+1],'steps':raw['steps'][:frame]}
    sample,domain=mechanism.current_domain(prefix);canonical,comm=mechanism.actions(domain)
    rows={t['task_id']:t for t in prefix['decisions'][-1]['tasks']}
    assert all(rows[r['task_id']]['lifecycle'] in ('offloading','transmitting') for r in comm.comm)
    assert 'Task_1' not in domain.wireless_task_to_relation and 'Task_6' not in domain.wireless_task_to_relation
    validate_command(comm,domain.context,prefix['decisions'][-1],{k:SimpleNamespace() for k in rows})
    poisoned=copy.deepcopy(prefix);poisoned['future_action']=object();poisoned['target']=object()
    poisoned['environment']['future_schedule']=object()
    for d in poisoned['decisions']:d['internal_metadata']={'future_schedule':object()}
    ps,pd=mechanism.current_domain(poisoned)
    assert sample==ps and mechanism.actions(pd)[0].frame()==canonical.frame() and mechanism.actions(pd)[1].frame()==comm.frame()
    sources=['code/scripts/run_step6_4f_real_comm_smoke_v1.py','code/scripts/run_step6_4e_comm_fallback_audit_v1.py','code/scripts/collect_step5_5_formal_raw_v1.py',
      'code/src/pi_jwm/step6_4f_comm_eligibility_v1.py','code/src/pi_jwm/step6_3b_candidate_grammar_v1.py','code/src/pi_jwm/step6_3c_candidate_domain_v1.py','code/src/pi_jwm/step6_4b_live_bridge_v1.py','code/src/pi_jwm/step6_4e_fallback_v1.py']
    value={k:old[k] for k in ('sample_id','raw_path','raw_sha256','simulator_seed','policy_seed','anchor')}
    value.update(verdict='FROZEN_BEFORE_EXECUTION',selection='Priority previous TRAIN anchor0041; current patched eligibility and live validation succeed before execution; no outcome selection.',
        parent_6_4e_fixture_sha256=mechanism.sha(parent),source_sha256={p:mechanism.sha(ROOT/p) for p in sources},normalization_sha256=mechanism.sha(mechanism.NORM),catalog_sha256=mechanism.sha(mechanism.CAT),
        live_sample_sha256=mechanism.digest(sample),wireless_task_to_relation=dict(domain.wireless_task_to_relation),domain_count=domain.exact_unique_single_step_count,
        canonical_fallback_action=canonical.frame(),nonempty_comm_action=comm.frame(),current_time=prefix['decisions'][-1]['simulation_time_s'],
        scenarios=['real_nonempty_comm_one_step','simulated_NO_SCOREABLE_H4_canonical_fallback_one_step'],steps_per_scenario=1,no_retry=True,
        WM_forward_count=0,search_transition_count=0,GPU='NOT_USED',locked_test=False,PLANNER_V1_CLOSED_LOOP_B_WM=512,FINAL_FALLBACK_POLICY='RESEARCHER_DECISION_PENDING')
    value['execution_config_id']=mechanism.digest(value)
    mechanism.write('01_fixture_and_execution_freeze.json',value)
    mechanism.write('02_future_target_poison_receipt.json',{'verdict':'PASS','live_sample_and_current_actions_identical':True,'future_target_read':False,'normalization_refit':False,'WM_forward_count':0})
    print(json.dumps({'freeze':'PASS','sample':value['sample_id'],'eligible':list(domain.wireless_task_to_relation),'comm':comm.frame(),'config':value['execution_config_id']}))
if __name__=='__main__':
    mechanism.torch.set_num_threads(1)
    if '--freeze' in sys.argv:freeze()
    elif '--execute' in sys.argv:mechanism.execute()
    else:raise SystemExit('Explicit --freeze or --execute required')
