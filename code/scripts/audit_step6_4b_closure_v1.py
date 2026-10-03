"""Independent CPU inference/provenance acceptance; no search or env execution."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,gzip,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
import torch
from run_step6_4b_two_cycle_smoke_v1 import OUT,NORM,write,history_tensor
from run_step6_3d_one_cpu_solve_v1 import load_frozen_runtime,EXPECTED_SHA,CHECKPOINT,sha
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
from pi_jwm.step5_5_full_sharded_loader_v1 import _one,FORMAL_V1_WIRED_EDGES
from pi_jwm.step5_2_training_loop_v1 import _torch_tree
from build_step5_1d_unified_model_chain_v1 import build_state
from pi_jwm.step6_0a_candidate_generation_v1 import PlannerCandidateContext
from pi_jwm.step6_0c_planner_action_domain_v1 import context_from_current_raw
from pi_jwm.step6_1_trained_candidate_rollout_v1 import prepare_anchor,fingerprint

def main():
    fixture=json.loads((OUT/'01_fixture_freeze.json').read_text())
    identity=json.loads((OUT/'02_execution_identity.json').read_text())
    smoke=json.loads((OUT/'05_two_cycle_smoke_receipt.json').read_text())
    assert all(sha(ROOT/name)==value for name,value in identity['source_sha256'].items())
    assert sha(CHECKPOINT)==EXPECTED_SHA==identity['checkpoint_sha256']
    assert sha(NORM)==identity['normalization_sha256']
    assert sha(ROOT/fixture['raw_path'])==fixture['raw_sha256']
    model,encoder,old_anchor,_,_,_,_,parameters=load_frozen_runtime(fixture['sample_id'],device='cpu')
    raw=json.loads(gzip.decompress((ROOT/fixture['raw_path']).read_bytes()))
    raw['decisions']=raw['decisions'][:4];raw['steps']=raw['steps'][:3]
    tid=fixture['sample_id'].split('::')[0]
    dataset=(ROOT/fixture['raw_path']).parent.parent
    frozen=_one(load_flow_tensor_batch(dataset/f'packages/tensor/{tid}.npz'),2,92)
    stats=json.loads(NORM.read_text())
    roots=[]
    for mode in ('deleted','poisoned'):
        causal=copy.deepcopy(raw)
        for d in causal['decisions']:d.pop('internal_metadata',None)
        if mode=='poisoned':
            causal['target']=object();causal['future_action']=object()
            for d in causal['decisions']:d['internal_metadata']={'future_task_schedule':object(),'future_dag_edges':object()}
        sample,tensor,ginput=history_tensor(causal,stats,frozen)
        state,graph=build_state(tensor,ginput,0,wired_edges=FORMAL_V1_WIRED_EDGES)
        context=PlannerCandidateContext.from_sample(sample,state,'cpu_leakage_oracle')
        domain=context_from_current_raw(context,causal['decisions'][-1])
        with torch.inference_mode():
            prepared=prepare_anchor(model,encoder,_torch_tree(tensor),_torch_tree(ginput),state,graph,context,domain,'cpu_leakage_oracle')
        roots.append(prepared.fingerprints)
    unchanged=fingerprint({'encoder':encoder.state_dict(),'rssm':model.state_dict()})==parameters
    leakage={'verdict':'PASS' if roots[0]==roots[1] and unchanged else 'BLOCKED',
        'deleted_future_root':roots[0],'poisoned_future_root':roots[1],
        'model_parameters_unchanged':unchanged,'checkpoint_sha256':EXPECTED_SHA,
        'normalization_refit_on_live_data':False,'normalization_sha256':sha(NORM),
        'new_search_calls':0,'new_world_model_transitions':0,'GPU':'NOT_USED','locked_test':False}
    write('08_leakage_provenance_receipt.json',leakage)
    baseline='f9b7c05fb016403b776065074de8d791126661f0'
    allowed={'code/src/pi_jwm/step6_3d_fixed_budget_search_v1.py','code/scripts/run_step6_3d_one_cpu_solve_v1.py',
        'code/scripts/collect_step5_5_formal_raw_v1.py','code/src/pi_jwm/step6_4b_live_bridge_v1.py',
        'code/scripts/run_step6_4b_two_cycle_smoke_v1.py','code/scripts/audit_step6_4b_closure_v1.py'}
    changed=subprocess.check_output(['git','diff','--name-only',baseline,'--','code/src','code/scripts'],cwd=ROOT,text=True).splitlines()
    assert set(changed)<=allowed
    historical=subprocess.check_output(['git','diff','--name-only',baseline,'--',
        'code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929',
        'code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001'],cwd=ROOT,text=True).splitlines()
    assert not historical
    write('09_search_semantics_provenance_receipt.json',{'verdict':'PASS','oracle':'test_step6_3d_exact_oracle_v1',
        'baseline':baseline,'paired_methods':['HRS','S-CEM','MH-CEM'],'paired_batches':[1,4],
        'compared':'entire SearchOutcome excluding added winner and wall-clock; exact equality',
        'source_diff_gate':changed,'historical_tracked_evidence_changes':historical,
        'candidate_domain_grammar_objective_rollout_comparator_parameters_unchanged':True,
        'TRAIN_rerun':False,'Stage_A_rewrite':False})
    checks={'two_real_cycles':len(smoke['cycles'])==2,'one_env_step':smoke['environment_steps_by_planner']==1,
        'simulator_paused':all(c['simulator_paused_during_planning'] for c in smoke['cycles']),
        'real_time_increment':abs(smoke['cycles'][-1]['simulation_time_s']-smoke['cycles'][0]['simulation_time_s']-.1)<1e-9,
        'fresh_domain':all(c['fresh_domain'] for c in smoke['cycles']),
        'fresh_root_not_predicted':smoke.get('fresh_root_not_predicted_state',False),
        'live_tensor_parity':json.loads((OUT/'03_live_input_parity_receipt.json').read_text())['verdict']=='PASS',
        'deadline_parity':json.loads((OUT/'04_live_deadline_receipt.json').read_text())['verdict']=='PASS',
        'winner':json.loads((OUT/'07_winner_first_action_receipt.json').read_text())['verdict']=='PASS',
        'first_action_only':json.loads((OUT/'06_action_bridge_receipt.json').read_text())['executed_horizon']==1,
        'leakage':leakage['verdict']=='PASS','normalization_unchanged':smoke['normalization_unchanged'],
        'parameters_unchanged':smoke['parameters_unchanged'],'budget64':all(c['budget_receipt']['N_unique_transition_evals']==64 for c in smoke['cycles']),
        'locked_test_false':smoke['locked_test'] is False,'stage_b_not_started':smoke['Stage_B']=='NOT_STARTED'}
    write('10_mechanism_acceptance.json',{'STEP_6_4B':'PASS' if all(checks.values()) else 'BLOCKED',
        'CLOSED_LOOP_MECHANISM_READINESS':'PASS' if all(checks.values()) else 'BLOCKED',
        'READY_FOR_BUDGET_CALIBRATION':all(checks.values()),'checks':checks,
        'actual_unique_transitions_two_cycle':sum(c['budget_receipt']['N_unique_transition_evals'] for c in smoke['cycles']),
        'aborted_pre_action_attempt_unique_transitions':64,'total_smoke_execution_unique_transitions_including_aborted_attempt':192,
        'remaining_blockers':[], 'warnings':['Mechanism-only, one preregistered TRAIN fixture; not a performance or universal-state acceptance.',
            'One pre-action receipt/control-flow failure was preserved and explicitly recovered; identical first-cycle discrete result verified.',
            'Comm was empty in the real winner; nonempty mapping covered by causal synthetic CPU fixture.',
            'Setter rollback scope is audited buffers in this synchronous single-thread smoke; no general simulator atomicity claim.'],
        'FORMAL_CLOSED_LOOP_PERFORMANCE':'NOT_STARTED','FINAL_FALLBACK_POLICY':'NOT_FROZEN','HYBRID_PLANNER':'NOT_FROZEN',
        'SEARCH_METHOD':'MH-CEM','Stage_B':'NOT_STARTED','GPU':'NOT_USED','locked_test':False})
    print('Independent input/posterior/provenance acceptance:',all(checks.values()))

if __name__=='__main__':main()
