"""CPU-only, pre-frozen TRAIN fixture, B64 mechanism smoke. Never retunes."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
import copy,gzip,hashlib,json,sys,time,traceback
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts'),str(ROOT/'code/scripts/small_experiments'),str(ROOT/'code/reference/AirFogSim')]
import numpy as np
import torch
from pi_jwm.step6_4b_live_bridge_v1 import build_live_sample,live_deadline_sidecar,validate_command,apply_commands,SmokeFailure,require_nonempty_domain
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from build_step5_5_formal_dataset_v1 import normalize
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import build_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import build_typed_dual_graph_batch,PhysicalTopologyConfig
from pi_jwm.step5_5_full_sharded_loader_v1 import FORMAL_V1_WIRED_EDGES,FullFormalShardDataset
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_2_training_loop_v1 import _torch_tree,_jsonable
from pi_jwm.step6_0a_candidate_generation_v1 import PlannerCandidateContext,CandidateActionSequence
from pi_jwm.step6_0c_planner_action_domain_v1 import context_from_current_raw
from pi_jwm.step6_3b_candidate_grammar_v1 import admit_structured_candidate
from pi_jwm.step6_3d_fixed_budget_search_v1 import solve_fixed_budget
from pi_jwm.step6_1_trained_candidate_rollout_v1 import prepare_anchor,rollout_one_step,fingerprint
from pi_jwm.step6_2b_planner_objective_scorer_v1 import score_candidate_set
from build_step5_1d_unified_model_chain_v1 import build_state
from run_step6_3d_one_cpu_solve_v1 import load_frozen_runtime,load_selected,objective_side,DATASET,CHECKPOINT,EXPECTED_SHA,sha
from pi_jwm.step6_4i_episode_v1 import EpisodeController,PlanPacket
OUT=ROOT/'code/artifacts/protocols/pi_jwm_step6_4i_closed_loop_readiness_v1_20261009_r2'
NORM=ROOT/'code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/packages/normalization/stats.json'

def write(name,value):
    def extra(x):
        if isinstance(x,(set,frozenset)):return sorted(x)
        raise TypeError(type(x).__name__)
    p=OUT/name;temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(_jsonable(value),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False,default=extra)+'\n',encoding='utf-8');temp.replace(p)

def history_tensor(prefix,stats,frozen):
    sample=build_live_sample(prefix)
    tensor=build_flow_tensor_batch(normalize([sample],stats),stats=stats['flow_step4_2c'])
    # Only padding to the already frozen TRAIN slot bounds; never new entities.
    for key,value in list(tensor.items()):
        if not isinstance(value,np.ndarray) or key.startswith(('target_','future_')):continue
        ref=frozen.get(key)
        if not isinstance(ref,np.ndarray) or ref.ndim!=value.ndim:continue
        if any(a>b for a,b in zip(value.shape,ref.shape)):
            raise SmokeFailure('LIVE_SLOT_UNSUPPORTED',f'{key}: {value.shape} exceeds {ref.shape}')
        if value.shape!=ref.shape:
            categorical=key.endswith(('type_index','lifecycle_index','transport_index','kind_index','status_index'))
            fill=-1 if not categorical and (key.endswith(('index','indices','endpoints','dag_edges','epoch','revision'))) else 0
            padded=np.full(ref.shape,fill,dtype=value.dtype)
            padded[tuple(slice(0,n) for n in value.shape)]=value
            tensor[key]=padded
    tensor['contract']={**frozen['contract'],'horizon_steps':0,
        'max_target_entity':0,'max_target_task':0,'max_target_flow':0,'max_target_logical_flow':0,
        'target_entity_capacity':0,'target_task_capacity':0}
    graph=build_typed_dual_graph_batch(tensor,PhysicalTopologyConfig(development_only=False,research_frozen=True))
    return sample,tensor,graph

class EpisodeStopped(BaseException):pass

def freeze():
    if OUT.exists():raise SystemExit('Existing Step evidence; do not overwrite or repeat')
    parent=ROOT/'code/artifacts/protocols/pi_jwm_step6_4b_two_cycle_smoke_v1_20261003/01_fixture_freeze.json'
    f=json.loads(parent.read_text());assert sha(ROOT/f['raw_path'])==f['raw_sha256']
    OUT.mkdir(parents=True)
    sources=['code/scripts/run_step6_4i_cpu_smoke_v1.py','code/src/pi_jwm/step6_4i_episode_v1.py',
      'code/src/pi_jwm/step6_4b_live_bridge_v1.py','code/src/pi_jwm/step6_4e_fallback_v1.py',
      'code/src/pi_jwm/step6_4f_comm_eligibility_v1.py','code/src/pi_jwm/step6_3c_candidate_domain_v1.py',
      'code/src/pi_jwm/step6_3b_candidate_grammar_v1.py','code/src/pi_jwm/step6_3d_fixed_budget_search_v1.py',
      'code/src/pi_jwm/step6_3d_structured_proposal_v1.py','code/src/pi_jwm/step6_2b_planner_objective_scorer_v1.py',
      'code/src/pi_jwm/step6_1_trained_candidate_rollout_v1.py','code/scripts/collect_step5_5_formal_raw_v1.py']
    f.update(method='S-CEM',K=4,rho=.2,B_WM=64,batch_size=1,cycles=2,
      selection='Reuse predetermined TRAIN anchor0003 from accepted6.4B; repaired eligibility; no selection using new S-CEM outcome',
      parent_fixture_sha256=sha(parent),source_sha256={p:sha(ROOT/p) for p in sources},
      checkpoint_sha256=sha(CHECKPOINT),normalization_sha256=sha(NORM),
      engineering_only=True,formal_budget=512,device='cpu',precision='FP32',locked_test=False)
    assert f['checkpoint_sha256']==EXPECTED_SHA
    f['execution_config_id']=hashlib.sha256(json.dumps(f,sort_keys=True).encode()).hexdigest()
    write('01_fixture_and_execution_identity.json',f)
    print(json.dumps({'freeze':'PASS','sample_id':f['sample_id'],'execution_config_id':f['execution_config_id']}))

def smoke():
    if (OUT/'02_attempt.json').exists():raise SystemExit('Attempt already recorded; no retry or overwrite')
    f=json.loads((OUT/'01_fixture_and_execution_identity.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in f['source_sha256'].items())
    assert sha(CHECKPOINT)==f['checkpoint_sha256'] and sha(NORM)==f['normalization_sha256']
    write('02_attempt.json',{'execution_config_id':f['execution_config_id'],'GPU':'NOT_USED','started':True,'no_retry':True})
    torch.set_num_threads(1)
    stats=json.loads(NORM.read_text())
    model,encoder,_,_,catalog,protocol,meta,parameters=load_frozen_runtime(f['sample_id'],device='cpu')
    interface=FormalTrainingInterface.from_manifest(DATASET)
    _,frozen,*_=load_selected(f['sample_id'],interface,FullFormalShardDataset(interface))
    raw=json.loads(gzip.decompress((ROOT/f['raw_path']).read_bytes()))
    prefix={**raw,'decisions':raw['decisions'][:4],'steps':raw['steps'][:3]}
    clean=history_tensor(prefix,stats,frozen)
    poisoned=copy.deepcopy(prefix);poisoned['target']=object();poisoned['future_action']=object()
    for d in poisoned['decisions']:d['internal_metadata']={'future_schedule':object()}
    poison=history_tensor(poisoned,stats,frozen)
    for k,v in clean[1].items():
        if isinstance(v,np.ndarray) and not k.startswith(('target_','future_')):assert np.array_equal(v,poison[1][k])
    write('03_live_input_and_leakage.json',{'verdict':'PASS','future_poison_invariant':True,'future_targets_built':False,'normalization_refit':False,'live_builder_reused':'6.4B exact history_tensor body'})
    receipt={'verdict':'BLOCKED','GPU':'NOT_USED','locked_test':False,'engineering_only':True,'budget_per_cycle':64,
      'method':'S-CEM','K':4,'rho':.2,'execution_config_id':f['execution_config_id'],'cycles':[],'environment_steps':0}
    journal=[]
    def log(row):journal.append(row);write('04_episode_journal.json',journal)
    controller=EpisodeController(log)
    import collect_step5_5_formal_raw_v1 as collector
    from airfogsim.scheduler.computation_sched import ComputationScheduler
    def hook(env,decisions,steps,config,communication):
        if decisions[-1]['frame_index']!=f['anchor']:return
        envconfig={'seed':f['simulator_seed'],'wired_edges':collector.WIRED_EDGES}
        try:
            assert build_live_sample({'environment':envconfig,'decisions':decisions,'steps':steps})==build_live_sample(prefix)
            for cycle in range(2):
                current=decisions[-1];tasks=collector.step23._all_runtime_tasks(env)
                def planner():
                    sample,tensor,ginput=history_tensor({'environment':envconfig,'decisions':decisions,'steps':steps},stats,frozen)
                    state,gstate=build_state(tensor,ginput,0,wired_edges=FORMAL_V1_WIRED_EDGES)
                    provenance=f"real:{current['trajectory_id']}:{current['frame_index']}:{current['capture_event_id']}"
                    context=PlannerCandidateContext.from_sample(sample,state,provenance)
                    operational=context_from_current_raw(context,current)
                    domain=CandidateDomain.from_state(context,operational,state,operational.mobility_states,catalog)
                    require_nonempty_domain(domain)
                    sidecar=live_deadline_sidecar(env,current,tasks)
                    side=objective_side(sample,state,context,current,sidecar)
                    traces={};support={}
                    def score(candidate,trace):
                        scored=score_candidate_set(state,side,((candidate,trace),),slot_duration_s=protocol.training.rssm.slot_duration_s)
                        support[scored.status]=support.get(scored.status,0)+1
                        if scored.status!='SCOREABLE' or scored.H_eff!=4 or not scored.scores or scored.scores[0].H_sup!=4:return None
                        traces[candidate.fingerprint]=trace;return scored.scores[0]
                    timer=time.perf_counter()
                    with torch.inference_mode():
                        prepared=prepare_anchor(model,encoder,_torch_tree(tensor),_torch_tree(ginput),state,gstate,context,operational,provenance)
                        result=solve_fixed_budget(method='S-CEM',seed=6301,b_wm=64,iterations=4,elite_ratio=.2,batch_size=1,
                            anchor=prepared,catalog=catalog,transition=lambda n,b:rollout_one_step(model,context,operational,n.latent,n.state,n.graph,n.mobility_control,b),score_h4=score)
                    predicted=None if result.best_fingerprint not in traces else fingerprint(traces[result.best_fingerprint].states[0])
                    receipt['cycles'].append({'root':prepared.fingerprints,'sidecar':sidecar,'score_statuses':support,
                      'complete_h4':result.complete_sequence_count,'distinct_scoreable_h4':result.h4_scoreable_count,
                      'winner_sequence':None if result.winner_sequence is None else asdict(result.winner_sequence),
                      'best_fingerprint':result.best_fingerprint,'best_objective':result.best_objective,
                      'budget':result.budget_receipt,'fresh_candidate_domain':True})
                    return PlanPacket(result,domain,dict(prepared.fingerprints),predicted,time.perf_counter()-timer)
                previous_speeds={r['entity_id']:float(r['speed_mps']) for r in current['entities']}
                before={k:float(v.getComputedSize()) for k,v in tasks.items()}
                event_start=len(env.pi_jwm_transfer_events);start=float(env.simulation_time)
                def capture():
                    end=float(env.simulation_time);events=[dict(v) for v in env.pi_jwm_transfer_events[event_start:]]
                    outcomes=collector.aggregate_slot_outcomes(transfer_events=events,computed_before=before,
                      computed_after={k:float(v.getComputedSize()) for k,v in tasks.items()},transport_observation=env.pi_jwm_transfer_observation)
                    observed=collector.step23._capture(env,frame=current['frame_index'],phase='post_env_step_outcome',event_index=2*current['frame_index']+1,previous_speed_by_entity=previous_speeds,delta_t_s=end-start)
                    observed.update(outcomes);observed['slot_transfer_events']=events
                    current_row=journal[-1];action=current_row['action'];commands=current_row['commands']
                    real_action={'route':collector.step23._family([],'EXPLICIT_NOOP_ONLY'),
                      'comm':action['comm'],
                      'comp':action['comp'],
                      'mobility':collector.step23._family([{'uav_id':k,'azimuth_rad':v['angle'],'elevation_rad':v['phi'],'speed_mps':v['speed']} for k,v in commands['mobility'].items()],'no_present_uav'),'vehicle_motion':'SUMO external'}
                    steps.append({'frame_index':current['frame_index'],'action':real_action,'outcome':observed})
                    fresh=collector.step23._capture(env,frame=current['frame_index']+1,phase='loop_start_decision',event_index=2*(current['frame_index']+1),previous_speed_by_entity=previous_speeds,delta_t_s=end-start)
                    rt=collector.step23._all_runtime_tasks(env)
                    for t in fresh['tasks']:t['required_returned_size']=float(rt[t['task_id']].getReturnedSize())
                    decisions.append(fresh);return fresh
                row=controller.cycle(env,current,tasks,planner,
                  lambda e,c:apply_commands(e,c,communication,ComputationScheduler,collector.step23.TrafficScheduler),env.step,capture)
                receipt['cycles'][-1]['execution']=row
                if row['status']!='EXECUTED':receipt['blocker']=row['reason'];break
                receipt['environment_steps']+=1
            if receipt['environment_steps']==2:
                receipt['fresh_root_not_predicted']=receipt['cycles'][1]['root']['state']!=receipt['cycles'][0]['execution']['predicted_h1_state']
                assert receipt['fresh_root_not_predicted']
                assert receipt['cycles'][0]['root']!=receipt['cycles'][1]['root']
                receipt.update(verdict='PASS',normal_winner_path=any(r['execution']['dispatch']=='WINNER' for r in receipt['cycles']),no_behavior_overwrite=True)
        except Exception as exc:receipt.update(blocker=getattr(exc,'reason',type(exc).__name__),detail=str(exc),traceback=traceback.format_exc())
        finally:
            receipt['parameters_unchanged']=fingerprint({'encoder':encoder.state_dict(),'rssm':model.state_dict()})==parameters
            receipt['normalization_unchanged']=sha(NORM)==f['normalization_sha256']
            write('05_two_cycle_real_feedback.json',receipt)
        raise EpisodeStopped()
    try:collector.collect_trajectory(f['simulator_seed'],f['policy_seed'],f['sample_id'].split('::')[0],on_decision=hook)
    except EpisodeStopped:pass
    except Exception as exc:
        receipt.update(blocker=type(exc).__name__,detail=str(exc),traceback=traceback.format_exc());write('05_two_cycle_real_feedback.json',receipt)
    print(json.dumps({'verdict':receipt['verdict'],'steps':receipt['environment_steps'],'blocker':receipt.get('blocker')}))
    return 0 if receipt['verdict']=='PASS' else 1

if __name__=='__main__':
    if '--freeze' in sys.argv:freeze()
    elif '--smoke' in sys.argv:raise SystemExit(smoke())
    else:raise SystemExit('CPU-only explicit --freeze / --smoke; no formal episode CLI')
