"""CPU/static closure of 6.4E, preserving the rejected real execution attempt.

Never retries an environment episode or modifies CandidateDomain/WM/search.
"""
import copy,gzip,json,subprocess,sys
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
from run_step6_4e_comm_fallback_audit_v1 import ROOT,OUT,NORM,CAT,current_domain,actions,sha,write
from pi_jwm.step6_4b_live_bridge_v1 import validate_command,SmokeFailure
from pi_jwm.step6_4e_fallback_v1 import prepare_fallback,execute_fallback,FallbackReason as R,FallbackCandidate as C

def main():
    freeze=json.loads((OUT/'01_fixture_and_execution_freeze.json').read_text())
    raw=json.loads(gzip.decompress((ROOT/freeze['raw_path']).read_bytes()))
    frame=freeze['anchor'];prefix={**raw,'decisions':raw['decisions'][:frame+1],'steps':raw['steps'][:frame]}
    sample,domain=current_domain(prefix);canonical,comm=actions(domain);decision=prefix['decisions'][-1]
    runtime={t['task_id']:SimpleNamespace() for t in decision['tasks']}
    assert canonical.frame()==freeze['canonical_fallback_action'] and comm.frame()==freeze['nonempty_comm_action']
    mismatch=[]
    task_rows={t['task_id']:t for t in decision['tasks']}
    for task,relation in domain.wireless_task_to_relation.items():
        lifecycle=task_rows[task]['lifecycle']
        if lifecycle not in ('offloading','transmitting'):
            mismatch.append({'task_id':task,'relation_index':relation,'current_lifecycle':lifecycle,'bridge_eligible':False})
    assert mismatch
    try:validate_command(comm,domain.context,decision,runtime)
    except SmokeFailure as exc:assert exc.reason=='ACTION_BRIDGE_REJECTED'
    else:raise AssertionError('Expected rejected failed task')
    write('06_current_domain_bridge_conflict.json',{'verdict':'BLOCKED','Documented Intent':'Current Domain action must map to currently eligible real simulator task.',
        'Actual Implementation':'Existing wireless Flow binding can retain failed task; Comm grammar admits it; live bridge rejects non-offloading/non-transmitting task.',
        'Evidence':mismatch,'attempted_comm':comm.frame()['comm'],'source_paths':['code/src/pi_jwm/step6_3b_candidate_grammar_v1.py:_wireless_bindings','code/src/pi_jwm/step6_4b_live_bridge_v1.py:validate_command'],
        'Conflict':'Flow presence/active masks and actual task lifecycle do not establish the same current execution eligibility.',
        'Status':'Awaiting Researcher Decision','scientific_definition_changed':False,'fixture_replaced':False,'execution_retry':False,
        'new_planner_setter_calls':0,'authorized_decision_steps':0,'replay_prior_behavior_steps':frame})
    A=prepare_fallback(R.NO_SCOREABLE_H4,C.CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION,domain,decision,runtime)
    assert not A.terminate_episode
    empty=prepare_fallback(R.NO_SCOREABLE_H4,C.CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION,replace(domain,modes=(),empty_reason='strict negative fixture'),decision,runtime)
    missing=copy.deepcopy(decision);missing.pop('n_rb')
    missing_result=prepare_fallback(R.NO_SCOREABLE_H4,C.CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION,domain,missing,runtime)
    C_result=prepare_fallback(R.NO_SCOREABLE_H4,C.FAIL_CLOSED_TERMINATION)
    B=prepare_fallback(R.NO_SCOREABLE_H4,C.PLANNER_V1_PROJECTED_EXISTING_BEHAVIOR,domain,decision,runtime)
    offer={'frame_index':frame,'simulation_time_s':decision['simulation_time_s'],'action':comm}
    bad_bridge=prepare_fallback(R.NO_SCOREABLE_H4,C.PLANNER_V1_PROJECTED_EXISTING_BEHAVIOR,domain,decision,runtime,behavior_offer=offer)
    old_callback=object();env=SimpleNamespace(simulation_time=decision['simulation_time_s'],activated_offloading_tasks_with_RB_Nos={'old':[2]},alloc_cpu_callback=old_callback,uav_mobility_patterns={})
    def setter_failure(e,c):e.alloc_cpu_callback=c;raise RuntimeError('injected CPU setter failure')
    setters=execute_fallback(A,env,SimpleNamespace(setCommunicationWithRB=lambda e,k,v:None),SimpleNamespace(setComputingCallBack=setter_failure),None)
    assert setters['terminate_episode'] and env.alloc_cpu_callback is old_callback and env.activated_offloading_tasks_with_RB_Nos=={'old':[2]}
    assert empty.reason==R.DOMAIN_EMPTY and missing_result.reason==R.REQUIRED_LIVE_OBSERVATION_MISSING
    assert C_result.terminate_episode and B.terminate_episode and bad_bridge.reason==R.ACTION_BRIDGE_REJECTED
    write('07_fallback_negative_path_receipt.json',{'verdict':'PASS','NO_SCOREABLE_H4_dispatch_prepare':'PASS_CURRENT_CAUSAL_REPLAY_ONLY',
        'NO_SCOREABLE_H4_real_execution':'NOT_EXECUTED_AFTER_COMM_BLOCK','DOMAIN_EMPTY':'PASS','missing_required_field':'PASS','bridge_failure':'PASS',
        'malformed_behavior_offer':'covered_by_focused_unit_test','setter_failure':setters,'fail_closed_termination':'PASS',
        'native_buffers_restored':True,'environment_time_unchanged':float(env.simulation_time)==float(decision['simulation_time_s']),
        'environment_steps':0,'injected_failures_are_synthetic':True,'arbitrary_setter_atomicity_guaranteed':False})
    behavior=ROOT/'code/scripts/collect_step5_5_formal_raw_v1.py'
    source=behavior.read_text(encoding='utf-8')
    assert 'def collect_trajectory' in source and 'setTaskOffloading' in source and 'setCommunicationWithRB' in source
    candidates={
      'A':{'name':C.CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION.value,'all_state_constructible':'NO','CandidateDomain_legal':'YES_IF_LAZY_BOUND_AND_LIVE_VALIDATION_PASS','Route_v1_compatible':True,'needs_WM_score':False,'affected_by_future_Return_birth':'NO_FOR_CURRENT_ONE_STEP_CONSTRUCTION; H4 limitation unchanged','uses_Future_Target':False,'real_AirFogSim_executable':'NOT_VERIFIED_THIS_STEP','deterministic':True,'known_safety_evidence':'CPU prevalidation and fail-closed only; no operational safety proof','known_performance_evidence':None,'failure_modes':['empty domain','missing live input','stale Flow/task lifecycle','bridge rejection','setter failure'],'replay_action':canonical.frame()},
      'B':{'name':C.PLANNER_V1_PROJECTED_EXISTING_BEHAVIOR.value,'all_state_constructible':'NO','CandidateDomain_legal':'CONDITIONAL_CURRENT_OFFER_ADMISSION_REQUIRED','Route_v1_compatible':'PROJECTION_REMOVES_ROUTE_ONLY','needs_WM_score':False,'affected_by_future_Return_birth':'No H4 score used; current support still required','uses_Future_Target':'Not allowed by interface; actual detached behavior provider missing','real_AirFogSim_executable':'NOT_VERIFIED_PROVIDER_MISSING','deterministic':False,'known_safety_evidence':None,'known_performance_evidence':None,'failure_modes':['inline behavior has Route and RNG/control flow coupling','no side-effect-free current Comm/Comp/Mob offer provider','projected action may be outside frozen Domain'],'source_sha256':sha(behavior),'current_offer_provider':'MISSING'},
      'C':{'name':C.FAIL_CLOSED_TERMINATION.value,'all_state_constructible':'YES_AS_TERMINATION_NOT_ACTION','CandidateDomain_legal':'NOT_APPLICABLE_NO_ACTION','Route_v1_compatible':True,'needs_WM_score':False,'affected_by_future_Return_birth':False,'uses_Future_Target':False,'real_AirFogSim_executable':'NO_ACTION_BY_DESIGN','deterministic':True,'known_safety_evidence':'No new setters or env.step; native buffer rollback on injected setter failure','known_performance_evidence':None,'failure_modes':['episode unavailable after termination','arbitrary external setter side effects not globally reversible']}}
    write('08_fallback_candidate_audit.json',{'FALLBACK_INTERFACE':'READY_CPU_CONTRACT','candidates':candidates,'RECOMMENDED_FALLBACK':'Conditional A for NO_SCOREABLE_H4 after real eligibility conflict closure; C for critical failures','recommendation_only':True,'FINAL_FALLBACK_POLICY':'RESEARCHER_DECISION_PENDING'})
    protected=['code/src/pi_jwm/step6_3d_fixed_budget_search_v1.py','code/src/pi_jwm/step6_3d_structured_proposal_v1.py','code/src/pi_jwm/step6_3d_method_selection_v1.py','code/src/pi_jwm/step6_3c_candidate_domain_v1.py','code/src/pi_jwm/step6_3b_candidate_grammar_v1.py','code/src/pi_jwm/step6_4b_live_bridge_v1.py']
    import hashlib
    hashes={}
    for path in protected:
        original=subprocess.check_output(['git','show','9c9cdfc9845ac5485ca07493aa3f332e43b3aa50:'+path],cwd=ROOT)
        current=(ROOT/path).read_bytes().replace(b'\r\n',b'\n')
        assert original==current,'Protected source changed: '+path
        hashes[path]=hashlib.sha256(current).hexdigest()
    write('09_budget_freeze_and_source_gate.json',{'verdict':'PASS','Researcher_Decision':'Latest explicit user authorization STEP6.4E','PLANNER_V1_CLOSED_LOOP_B_WM':512,'SEARCH_METHOD':'MH-CEM','K':4,'rho':.1,
        'budget_scope':'Planner v1 pure-search only; future hybrid budget not frozen','B1024':'HIGH_BUDGET_REFERENCE_DIAGNOSTIC','equivalence_or_optimality_claim':False,
        'known_tradeoff':'Observed burden/effort loss; no observed first-three-component degradation in 6.4D cohort; not closed-loop performance',
        'Stage_B':'NOT_STARTED','Stage_B_disposition':'DEFERRED / NOT_REQUIRED_FOR_CURRENT_MAINLINE','protected_source_SHA256_LF':hashes,
        'normalization_sha256':sha(NORM),'catalog_sha256':sha(CAT),'checkpoint_loaded':False,'checkpoint_modified':False,'WM_forward_count':0,'search_transition_count':0,'GPU':'NOT_USED','locked_test':False})
    write('10_pre_formal_readiness_receipt.json',{'STEP_6_4E':'BLOCKED','CLOSED_LOOP_PRE_FORMAL_READINESS':'BLOCKED','PLANNER_V1_CLOSED_LOOP_B_WM':512,
        'REAL_NONEMPTY_COMM_EXECUTION':'BLOCKED','FALLBACK_INTERFACE':'READY_CPU_CONTRACT','NO_SCOREABLE_H4_path':'CPU_DISPATCH_PASS_REAL_EXECUTION_NOT_OBTAINED',
        'DOMAIN_EMPTY_path':'PASS','missing_bridge_setter_failure':'PASS_FAIL_CLOSED','Future_Target_leakage':'PASS','authorized_environment_steps':0,
        'remaining_blockers':['Current wireless Flow binding includes failed Task_1/Task_6; frozen nonempty Comm action rejected by live task eligibility','No real nonempty Comm command consumption evidence','Nonterminating fallback real one-step evidence not obtained; scheduled second episode not started after rejection','Existing behavior projected provider missing (candidate B unavailable)'],
        'scientific_conflict_status':'Awaiting Researcher Decision; no support/Domain/Objective change or sample replacement',
        'FINAL_FALLBACK_POLICY':'RESEARCHER_DECISION_PENDING','FORMAL_CLOSED_LOOP_PERFORMANCE':'NOT_STARTED','HYBRID_PLANNER':'NOT_FROZEN','Stage_B':'NOT_STARTED','GPU':'NOT_USED','locked_test':False})
    print(json.dumps({'STEP_6_4E':'BLOCKED','domain_failed_task_bindings':mismatch,'fallback_CPU_audit':'PASS','real_execution_retried':False}))
if __name__=='__main__':main()
