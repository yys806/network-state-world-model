"""Pre-frozen TRAIN replay, two independent one-step execution mechanisms only.

No search, model forward, performance experiment, future target or default policy.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
import copy,gzip,hashlib,json,sys,traceback
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts'),str(ROOT/'code/scripts/small_experiments'),str(ROOT/'code/reference/AirFogSim')]
import torch
from pi_jwm.step6_4b_live_bridge_v1 import build_live_sample,validate_command,apply_commands,SmokeFailure
from pi_jwm.step6_4e_fallback_v1 import prepare_fallback,execute_fallback,FallbackReason,FallbackCandidate
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_3b_candidate_support_v1 import TrainStructuralSupportCatalog
from pi_jwm.step6_0a_candidate_generation_v1 import PlannerCandidateContext,CandidateActionSequence,Backend
from pi_jwm.step6_0c_planner_action_domain_v1 import context_from_current_raw
from pi_jwm.step6_3b_candidate_grammar_v1 import admit_structured_candidate
from build_step5_5_formal_dataset_v1 import normalize
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import build_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import build_typed_dual_graph_batch,PhysicalTopologyConfig
from pi_jwm.step5_5_full_sharded_loader_v1 import FORMAL_V1_WIRED_EDGES
from build_step5_1d_unified_model_chain_v1 import build_state
OUT=ROOT/'code/artifacts/protocols/pi_jwm_step6_4e_comm_fallback_v1_20261004'
NORM=ROOT/'code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/packages/normalization/stats.json'
CAT=ROOT/'code/artifacts/protocols/pi_jwm_step6_3b_candidate_grammar_v1_20260929/02_train_structural_support_catalog.json'
OLD=ROOT/'code/artifacts/protocols/pi_jwm_step6_4b_two_cycle_smoke_v1_20261003/01_fixture_freeze.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
def write(name,v):
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/name
    if p.exists():raise RuntimeError('Receipt exists; no overwrite/retry: '+name)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,sort_keys=True,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8');tmp.replace(p)
def current_domain(prefix):
    sample=build_live_sample(prefix)
    stats=json.loads(NORM.read_text(encoding='utf-8'))
    tensor=build_flow_tensor_batch(normalize([sample],stats),stats=stats['flow_step4_2c'])
    graph=build_typed_dual_graph_batch(tensor,PhysicalTopologyConfig(development_only=False,research_frozen=True))
    state,_=build_state(tensor,graph,0,wired_edges=FORMAL_V1_WIRED_EDGES)
    current=prefix['decisions'][-1]
    context=PlannerCandidateContext.from_sample(sample,state,f"real-current:{current['trajectory_id']}:{current['frame_index']}")
    operational=context_from_current_raw(context,current)
    domain=CandidateDomain.from_state(context,operational,state,operational.mobility_states,TrainStructuralSupportCatalog.from_json(CAT))
    return sample,domain
def actions(domain):
    if domain.is_empty:raise SmokeFailure('DOMAIN_EMPTY')
    canonical=next(domain.iter_bound()).action
    nonempty=replace(domain,modes=tuple(m for m in domain.modes if m.widths))
    if nonempty.is_empty:raise SmokeFailure('NO_LEGAL_NONEMPTY_COMM')
    comm=next(nonempty.iter_bound()).action
    assert comm.comm and not comm.route and not canonical.route
    for step in (canonical,comm):
        one=CandidateActionSequence((step,),'step6.4e',Backend.RULE_FALLBACK,None,domain.context.causal_provenance)
        assert admit_structured_candidate(one,domain.context,domain.operational_domain,(domain.state,),domain.catalog).admitted
    return canonical,comm
def freeze():
    dataset=NORM.parents[2]
    splitpath=dataset/'split_manifest.json'
    selected=None;examined=[]
    for trajectory in sorted(json.loads(splitpath.read_text(encoding='utf-8'))['dev_train']):
        rawpath=dataset/'raw'/f'{trajectory}.json.gz'
        raw=json.loads(gzip.decompress(rawpath.read_bytes()))
        for d in raw['decisions'][1:]:
            anchor=int(d['frame_index'])
            if not any(t['lifecycle'] in ('offloading','transmitting') for t in d['tasks']):continue
            prefix={**raw,'decisions':raw['decisions'][:anchor+1],'steps':raw['steps'][:anchor]}
            try:sample,domain=current_domain(prefix);canonical,comm=actions(domain)
            except (SmokeFailure,ValueError) as exc:
                examined.append({'trajectory':trajectory,'anchor':anchor,'reason':str(exc)});continue
            selected=(trajectory,anchor,rawpath,raw);break
        if selected:break
    if selected is None:raise SmokeFailure('NO_TRAIN_CURRENT_NONEMPTY_COMM_FIXTURE')
    trajectory,anchor,rawpath,raw=selected
    old={'sample_id':f'{trajectory}::anchor-{anchor:04d}','anchor':anchor,
         'raw_path':str(rawpath.relative_to(ROOT)).replace('\\','/'),
         'simulator_seed':raw['environment']['seed'],'policy_seed':raw['environment']['policy_seed']}
    poisoned=copy.deepcopy(prefix)
    poisoned['target']={'poison':999999};poisoned['future_action']={'poison':999999}
    poisoned['environment']['future_schedule']={'poison':999999}
    for d in poisoned['decisions']:d.update(target={'poison':999999},future_action={'poison':999999},internal_metadata={'future_schedule':999999})
    ps,pd=current_domain(poisoned)
    assert sample==ps and actions(pd)[0].frame()==canonical.frame() and actions(pd)[1].frame()==comm.frame()
    sources=[Path(__file__).resolve(),ROOT/'code/src/pi_jwm/step6_4e_fallback_v1.py',ROOT/'code/src/pi_jwm/step6_4b_live_bridge_v1.py',ROOT/'code/scripts/collect_step5_5_formal_raw_v1.py',ROOT/'code/src/pi_jwm/step6_3c_candidate_domain_v1.py',ROOT/'code/src/pi_jwm/step6_3b_candidate_grammar_v1.py']
    receipt={'verdict':'FROZEN_BEFORE_EXECUTION','sample_id':old['sample_id'],'split':'dev_train','raw_path':old['raw_path'],'raw_sha256':sha(rawpath),
        'simulator_seed':old['simulator_seed'],'policy_seed':old['policy_seed'],'anchor':old['anchor'],
        'selection':'Sort dev_train trajectory IDs then frame ascending; first current offloading/transmitting observation with a legal nonempty Comm Domain action. Inspect only causal prefixes; no objective/scoreability/new outcome selection.',
        'split_manifest_sha256':sha(splitpath),'rejected_current_prefixes':examined,
        'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sources},'normalization_sha256':sha(NORM),'catalog_sha256':sha(CAT),
        'live_sample_sha256':digest(sample),'wireless_task_to_relation':dict(domain.wireless_task_to_relation),
        'current_time':prefix['decisions'][-1]['simulation_time_s'],'domain_count':domain.exact_unique_single_step_count,
        'canonical_fallback_action':canonical.frame(),'nonempty_comm_action':comm.frame(),
        'scenarios':['real_nonempty_comm_one_step','simulated_NO_SCOREABLE_H4_canonical_fallback_one_step'],
        'steps_per_scenario':1,'no_retry':True,'WM_forward_count':0,'search_transition_count':0,'GPU':'NOT_USED','locked_test':False,
        'PLANNER_V1_CLOSED_LOOP_B_WM':512,'FINAL_FALLBACK_POLICY':'RESEARCHER_DECISION_PENDING'}
    receipt['execution_config_id']=digest(receipt)
    write('01_fixture_and_execution_freeze.json',receipt)
    write('02_future_target_poison_receipt.json',{'verdict':'PASS','live_sample_and_current_actions_identical':True,'future_target_read':False,'normalization_refit':False,'WM_forward_count':0})
    print(json.dumps({'freeze':'PASS','wireless_tasks':list(domain.wireless_task_to_relation),'comm':comm.frame(),'canonical':canonical.frame()}))
class EpisodeStopped(BaseException):pass
def execute():
    recovery='--continue-pre-environment-import' in sys.argv
    if recovery:
        failure=json.loads((OUT/'03a_pre_environment_import_failure.json').read_text(encoding='utf-8'))
        assert failure['environment_created'] is False and failure['action_setter_calls']==0 and failure['environment_steps']==0
        assert failure['exception']=='UnicodeEncodeError'
        assert not (OUT/'04_real_nonempty_comm_one_step_receipt.json').exists() and not (OUT/'05_simulated_NO_SCOREABLE_H4_canonical_fallback_one_step_receipt.json').exists()
    frozen=json.loads((OUT/('01b_pre_environment_encoding_execution_identity.json' if recovery else '01_fixture_and_execution_freeze.json')).read_text(encoding='utf-8'))
    for path,value in frozen['source_sha256'].items():assert sha(ROOT/path)==value,'Source drift: '+path
    assert sha(NORM)==frozen['normalization_sha256'] and sha(CAT)==frozen['catalog_sha256']
    assert sha(ROOT/frozen['raw_path'])==frozen['raw_sha256']
    # A durable marker refuses repeat even after an interrupted run.
    write('03b_first_environment_execution_started.json' if recovery else '03_execution_started.json',{'execution_config_id':frozen['execution_config_id'],'no_retry':True,'pre_environment_import_continuation':recovery})
    import collect_step5_5_formal_raw_v1 as collector
    from airfogsim.scheduler.computation_sched import ComputationScheduler
    for index,scenario in enumerate(frozen['scenarios']):
        receipt={'verdict':'BLOCKED','scenario':scenario,'environment_steps':0,'WM_forward_count':0,'GPU':'NOT_USED','locked_test':False,'execution_config_id':frozen['execution_config_id']}
        def hook(env,decisions,steps,config,communication):
            if decisions[-1]['frame_index']!=frozen['anchor']:return
            try:
                sample,domain=current_domain({'environment':{'seed':frozen['simulator_seed'],'wired_edges':collector.WIRED_EDGES},'decisions':decisions,'steps':steps})
                assert digest(sample)==frozen['live_sample_sha256'],'Live replay mismatch'
                canonical,comm=actions(domain);action=comm if index==0 else canonical
                assert action.frame()==frozen['nonempty_comm_action' if index==0 else 'canonical_fallback_action']
                tasks=collector.step23._all_runtime_tasks(env)
                commands=validate_command(action,domain.context,decisions[-1],tasks)
                start=float(env.simulation_time)
                route_buffer={k:(v.getCurrentNodeId(),tuple(v.getToOffloadRoute())) for k,v in tasks.items()}
                return_buffer=copy.deepcopy(env.task_return_routes)
                if index==0:apply_commands(env,commands,communication,ComputationScheduler,collector.step23.TrafficScheduler)
                else:
                    prepared=prepare_fallback(FallbackReason.NO_SCOREABLE_H4,FallbackCandidate.CURRENT_DOMAIN_DETERMINISTIC_LEGAL_ACTION,domain,decisions[-1],tasks)
                    assert not prepared.terminate_episode and prepared.action.frame()==action.frame()
                    receipt['dispatch']=execute_fallback(prepared,env,communication,ComputationScheduler,collector.step23.TrafficScheduler)
                    assert receipt['dispatch']['action_executed']
                assert float(env.simulation_time)==start
                assert {k:(v.getCurrentNodeId(),tuple(v.getToOffloadRoute())) for k,v in tasks.items()}==route_buffer and env.task_return_routes==return_buffer
                assert env.activated_offloading_tasks_with_RB_Nos==commands['rb']
                consumed=[];original=env._allocate_communication_RBs
                def traced(requests):
                    profiles=original(requests)
                    for tid,p in profiles.items():
                        task=p['task'];consumed.append({'task_id':tid,'rb_indices':list(p['RB_Nos']),
                            'source_node_id':task.getCurrentNodeId(),'destination_node_id':task.getToOffloadRoute()[0],
                            'tx_idx':p['tx_idx'],'rx_idx':p['rx_idx'],'channel_type':p['channel_type']})
                    return profiles
                env._allocate_communication_RBs=traced
                before={k:float(v.getComputedSize()) for k,v in tasks.items()}
                event_start=len(env.pi_jwm_transfer_events)
                env.step();receipt['environment_steps']=1
                end=float(env.simulation_time)
                assert abs(end-start-.1)<1e-9,'Not exactly one .1s decision step'
                events=[dict(v) for v in env.pi_jwm_transfer_events[event_start:]]
                outcomes=collector.aggregate_slot_outcomes(transfer_events=events,computed_before=before,
                    computed_after={k:float(v.getComputedSize()) for k,v in tasks.items()},transport_observation=env.pi_jwm_transfer_observation)
                fresh=collector.step23._capture(env,frame=frozen['anchor']+1,phase='loop_start_decision',event_index=2*(frozen['anchor']+1),previous_speed_by_entity={r['entity_id']:float(r['speed_mps']) for r in decisions[-1]['entities']},delta_t_s=end-start)
                if index==0:
                    assert consumed and all(any(v['task_id']==tid and v['rb_indices']==rbs for v in consumed) for tid,rbs in commands['rb'].items()),'Comm command not consumed'
                    for v in consumed:
                        assert v['tx_idx']==env._getNodeIdxById(v['source_node_id']) and v['rx_idx']==env._getNodeIdxById(v['destination_node_id'])
                receipt.update(verdict='PASS',start_time_s=start,finish_time_s=end,action=action.frame(),commands_before_step=commands,
                    consumed_wireless_profiles=consumed,slot_transfer_events=events,observable_outcomes=outcomes,
                    fresh_observation_time_s=fresh['simulation_time_s'],fresh_observation_sha256=digest(fresh),
                    route_command_count=0,behavior_policy_overwrite=False,selection_during_step=False,simulation_paused_during_preparation=True,
                    future_target_read=False,normalization_unchanged=sha(NORM)==frozen['normalization_sha256'],performance_claim=False)
            except Exception as exc:
                receipt.update(blocker=getattr(exc,'reason',type(exc).__name__),detail=str(exc),traceback=traceback.format_exc())
            finally:write(f'{4+index:02d}_{scenario}_receipt.json',receipt)
            raise EpisodeStopped()
        try:collector.collect_trajectory(frozen['simulator_seed'],frozen['policy_seed'],frozen['sample_id'].split('::')[0],on_decision=hook)
        except EpisodeStopped:pass
        except Exception as exc:
            receipt.update(blocker=type(exc).__name__,detail=str(exc),traceback=traceback.format_exc())
            if not (OUT/f'{4+index:02d}_{scenario}_receipt.json').exists():write(f'{4+index:02d}_{scenario}_receipt.json',receipt)
        print(json.dumps({'scenario':scenario,'verdict':receipt['verdict'],'blocker':receipt.get('blocker'),'steps':receipt['environment_steps']}))
        if receipt['verdict']!='PASS':break
if __name__=='__main__':
    torch.set_num_threads(1)
    if '--freeze' in sys.argv:freeze()
    elif '--execute' in sys.argv:execute()
    else:raise SystemExit('Choose --freeze or --execute; no automatic experiment')
