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
OUT=ROOT/'code/artifacts/protocols/pi_jwm_step6_4b_two_cycle_smoke_v1_20261003'
NORM=ROOT/'code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/packages/normalization/stats.json'

def write(name,value):
    def extra(value):
        if isinstance(value,(set,frozenset)):return sorted(value)
        raise TypeError(f'Unsupported receipt type: {type(value).__name__}')
    (OUT/name).write_text(json.dumps(_jsonable(value),ensure_ascii=False,indent=2,sort_keys=True,default=extra)+'\n',encoding='utf-8')

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

def main():
    torch.set_num_threads(1)
    freeze=json.loads((OUT/'01_fixture_freeze.json').read_text())
    recovery_reference=None
    if (OUT/'05_two_cycle_smoke_receipt.json').exists():
        previous=json.loads((OUT/'05_two_cycle_smoke_receipt.json').read_text())
        if previous['cycles'] or previous['environment_steps_by_planner']:
            # Only an explicit engineering recovery before any setter; never
            # recover a no-scoreable, legality, setter or environmental failure.
            if '--recover-pre-action-receipt' not in sys.argv or previous.get('blocker')!='RuntimeError' or previous['environment_steps_by_planner'] or (OUT/'06_action_bridge_receipt.json').exists():
                raise SystemExit('Existing search/smoke outcome: no automatic repeat')
            if len(previous['cycles'])!=1 or previous['cycles'][0]['h4_scoreable_count']<=0:
                raise SystemExit('Not a pre-action receipt recovery')
            recovery_reference=previous['cycles'][0]
        attempts=OUT/'preflight_attempts';attempts.mkdir(exist_ok=True)
        attempt=len(list(attempts.glob('*_receipt.json')))+1
        for name in ('02_execution_identity.json','03_live_input_parity_receipt.json','05_two_cycle_smoke_receipt.json'):
            p=OUT/name
            if p.exists():(attempts/f'{attempt}_{name}').write_bytes(p.read_bytes())
    stats=json.loads(NORM.read_text())
    receipt={'STEP_6_4B':'BLOCKED','GPU':'NOT_USED','locked_test':False,'Stage_B':'NOT_STARTED',
        'latency_mode':'SYNCHRONOUS_PAUSED_SIMULATION','failure_handling':'FAIL_CLOSED_AND_STOP_EPISODE',
        'budget_per_cycle':64,'batch_size':1,'cycles':[],'environment_steps_by_planner':0}
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [Path(__file__).resolve(),ROOT/'code/src/pi_jwm/step6_4b_live_bridge_v1.py',ROOT/'code/src/pi_jwm/step6_3d_fixed_budget_search_v1.py',ROOT/'code/scripts/collect_step5_5_formal_raw_v1.py']}
    identity={'fixture_sha256':sha(OUT/'01_fixture_freeze.json'),'checkpoint_sha256':sha(CHECKPOINT),
        'normalization_sha256':sha(NORM),'source_sha256':sources,'execution_device':'cpu','precision':'FP32','batch_size':1,
        'B_WM':64,'K':4,'rho':0.1,'seed':6301,'locked_test':False}
    assert identity['checkpoint_sha256']==EXPECTED_SHA
    identity['execution_config_id']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    write('02_execution_identity.json',identity)
    model,encoder,old_anchor,_,catalog,protocol,meta,parameters=load_frozen_runtime(freeze['sample_id'],device='cpu')
    interface=FormalTrainingInterface.from_manifest(DATASET)
    old_sample,frozen,*_=load_selected(freeze['sample_id'],interface,FullFormalShardDataset(interface))
    raw=json.loads(gzip.decompress((ROOT/freeze['raw_path']).read_bytes()))
    prefix={**raw,'decisions':raw['decisions'][:4],'steps':raw['steps'][:3]}
    sample,tensor,graph=history_tensor(prefix,stats,frozen)
    differences=[]
    for key,value in tensor.items():
        if isinstance(value,np.ndarray) and not key.startswith(('target_','future_')) and key in frozen:
            if not np.array_equal(value,frozen[key]):differences.append(key)
    write('03_live_input_parity_receipt.json',{'verdict':'PASS' if not differences else 'BLOCKED','differing_arrays':differences,
        'future_targets_built':False,'normalization_refit':False,'input_policy':'history_causal_observable_object_union'})
    if differences:
        receipt['blocker']='LIVE_INPUT_FROZEN_PARITY_MISMATCH';receipt['differing_arrays']=differences
        write('05_two_cycle_smoke_receipt.json',receipt);return
    import collect_step5_5_formal_raw_v1 as collector
    from airfogsim.scheduler.computation_sched import ComputationScheduler
    def plan(env,decisions,steps,config):
        before_time=float(env.simulation_time)
        current=decisions[-1]
        sample,tensor,ginput=history_tensor({'environment':config,'decisions':decisions,'steps':steps},stats,frozen)
        state,gstate=build_state(tensor,ginput,0,wired_edges=FORMAL_V1_WIRED_EDGES)
        provenance=f"live:{current['trajectory_id']}:{current['frame_index']}:{current['capture_event_id']}"
        context=PlannerCandidateContext.from_sample(sample,state,provenance)
        domain=context_from_current_raw(context,current)
        require_nonempty_domain(CandidateDomain.from_state(context,domain,state,domain.mobility_states,catalog))
        sidecar=live_deadline_sidecar(env,current,collector.step23._all_runtime_tasks(env))
        if len(receipt['cycles'])==0:
            official=json.loads((ROOT/'code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/19_formal_selected_deadline_sidecars.json').read_text())[freeze['sample_id']]
            if sidecar['tasks']!=official['tasks']:raise SmokeFailure('LIVE_DEADLINE_SIDECAR_MISMATCH')
            write('04_live_deadline_receipt.json',{'verdict':'PASS','live':sidecar,'official_tasks':official['tasks']})
        side=objective_side(sample,state,context,current,sidecar)
        traces={}
        def score(candidate,trace):
            scored=score_candidate_set(state,side,((candidate,trace),),slot_duration_s=protocol.training.rssm.slot_duration_s)
            if scored.status!='SCOREABLE' or scored.H_eff!=4 or not scored.scores or scored.scores[0].H_sup!=4:return None
            traces[candidate.fingerprint]=trace
            return scored.scores[0]
        start=time.perf_counter()
        with torch.inference_mode():
            prepared=prepare_anchor(model,encoder,_torch_tree(tensor),_torch_tree(ginput),state,gstate,context,domain,provenance)
            result=solve_fixed_budget(method='MH-CEM',seed=6301,b_wm=64,iterations=4,elite_ratio=0.1,batch_size=1,
                anchor=prepared,catalog=catalog,transition=lambda n,b:rollout_one_step(model,context,domain,n.latent,n.state,n.graph,n.mobility_control,b),score_h4=score)
        assert float(env.simulation_time)==before_time
        row={'simulation_time_s':before_time,'root':prepared.fingerprints,'best_objective':result.best_objective,
            'best_fingerprint':result.best_fingerprint,'h4_scoreable_count':result.h4_scoreable_count,
            'complete_h4':result.complete_sequence_count,'budget_receipt':result.budget_receipt,'wall_clock_seconds':time.perf_counter()-start,
            'fresh_domain':True,'simulator_paused_during_planning':True}
        receipt['cycles'].append(row)
        if len(receipt['cycles'])==1:
            if recovery_reference is not None:
                keys=('best_fingerprint','best_objective','h4_scoreable_count','complete_h4','root')
                if any(_jsonable(row[k])!=recovery_reference[k] for k in keys):
                    raise SmokeFailure('PRE_ACTION_RECOVERY_SEARCH_RESULT_DRIFT')
                receipt['pre_action_receipt_recovery_discrete_result_identical']=True
            write('07_winner_first_action_receipt.json',{'verdict':'PASS' if result.winner_sequence else 'NO_SCOREABLE_H4',
                'winner_sequence':None if result.winner_sequence is None else asdict(result.winner_sequence),
                'winner_first_action':None if result.winner_first_action is None else result.winner_first_action.frame(),
                'best_fingerprint':result.best_fingerprint,'best_objective':result.best_objective})
        if len(receipt['cycles'])==1 and result.winner_sequence is None:raise SmokeFailure('NO_SCOREABLE_H4','SUCCESS_PATH_NOT_OBTAINED_AT_SMOKE_BUDGET')
        return result,context,domain,state,traces
    def hook(env,decisions,steps,config,communication):
        if decisions[-1]['frame_index']!=3:return
        envconfig={'seed':freeze['simulator_seed'],'wired_edges':collector.WIRED_EDGES}
        try:
            # Match actual replay before any planner action; future audit metadata excluded.
            if build_live_sample({'environment':envconfig,'decisions':decisions,'steps':steps})!=build_live_sample(prefix):
                raise SmokeFailure('REAL_REPLAY_CURRENT_OBSERVATION_MISMATCH')
            result,context,domain,state,traces=plan(env,decisions,steps,envconfig)
            one=CandidateActionSequence((result.winner_first_action,),result.winner_sequence.candidate_id,result.winner_sequence.generator_backend,
                result.winner_sequence.seed,context.causal_provenance)
            admission=admit_structured_candidate(one,context,domain,(state,),catalog)
            if not admission.admitted:raise SmokeFailure('ACTION_BRIDGE_REJECTED',admission.reason_codes)
            commands=validate_command(result.winner_first_action,context,decisions[-1],collector.step23._all_runtime_tasks(env))
            write('06_action_bridge_receipt.json',{'verdict':'PASS','commands':commands,'executed_horizon':1,
                'winner_sequence':asdict(result.winner_sequence),'grammar_admission':asdict(admission)})
            apply_commands(env,commands,communication,ComputationScheduler,collector.step23.TrafficScheduler)
            previous_speeds={r['entity_id']:float(r['speed_mps']) for r in decisions[-1]['entities']}
            tasks=collector.step23._all_runtime_tasks(env);before={k:float(v.getComputedSize()) for k,v in tasks.items()}
            event_start=len(env.pi_jwm_transfer_events);start=float(env.simulation_time)
            env.step();receipt['environment_steps_by_planner']=1
            collector._repair_duplicate_task_references(env.task_manager,3)
            end=float(env.simulation_time);events=[dict(v) for v in env.pi_jwm_transfer_events[event_start:]]
            outcomes=collector.aggregate_slot_outcomes(transfer_events=events,computed_before=before,
                computed_after={k:float(v.getComputedSize()) for k,v in tasks.items()},transport_observation=env.pi_jwm_transfer_observation)
            observed=collector.step23._capture(env,frame=3,phase='post_env_step_outcome',event_index=7,previous_speed_by_entity=previous_speeds,delta_t_s=end-start)
            observed.update(outcomes);observed['slot_transfer_events']=events
            first=result.winner_first_action
            action={'route':collector.step23._family([], 'EXPLICIT_NOOP_ONLY'),
                'comm':collector.step23._family(list(first.comm),'no_eligible'),
                'comp':collector.step23._family(list(first.comp),'no_eligible'),
                'mobility':collector.step23._family([{'uav_id':k,'azimuth_rad':v['angle'],'elevation_rad':v['phi'],'speed_mps':v['speed']} for k,v in commands['mobility'].items()],'no_present_uav'),
                'vehicle_motion':'SUMO external'}
            steps.append({'frame_index':3,'action':action,'outcome':observed})
            fresh=collector.step23._capture(env,frame=4,phase='loop_start_decision',event_index=8,previous_speed_by_entity=previous_speeds,delta_t_s=end-start)
            runtime=collector.step23._all_runtime_tasks(env)
            for t in fresh['tasks']:t['required_returned_size']=float(runtime[t['task_id']].getReturnedSize())
            decisions.append(fresh)
            predicted=traces[result.best_fingerprint]
            receipt['cycle1_predicted_h1_state']=fingerprint(predicted.states[0])
            plan(env,decisions,steps,envconfig)
            receipt['fresh_root_not_predicted_state']=receipt['cycles'][1]['root']['state']!=receipt['cycle1_predicted_h1_state']
            if not receipt['fresh_root_not_predicted_state']:raise SmokeFailure('ROOT_PREDICTION_REUSE_OR_IDENTITY_COLLISION')
            receipt.update(STEP_6_4B='PASS',TWO_CYCLE_REAL_FEEDBACK_SMOKE='PASS',CLOSED_LOOP_MECHANISM_READINESS='PASS')
        except Exception as exc:
            receipt['blocker']=getattr(exc,'reason',type(exc).__name__)
            receipt['detail']=str(exc);receipt['traceback']=traceback.format_exc()
        finally:
            receipt['parameters_unchanged']=fingerprint({'encoder':encoder.state_dict(),'rssm':model.state_dict()})==parameters
            receipt['normalization_unchanged']=sha(NORM)==identity['normalization_sha256']
            write('05_two_cycle_smoke_receipt.json',receipt)
        raise EpisodeStopped()
    try:
        collector.collect_trajectory(freeze['simulator_seed'],freeze['policy_seed'],freeze['sample_id'].split('::')[0],on_decision=hook)
    except EpisodeStopped:pass
    except Exception as exc:
        receipt['blocker']=type(exc).__name__;receipt['detail']=str(exc);receipt['traceback']=traceback.format_exc()
        write('05_two_cycle_smoke_receipt.json',receipt)
    print(json.dumps({'verdict':receipt['STEP_6_4B'],'blocker':receipt.get('blocker'),'cycles':len(receipt['cycles'])}))

if __name__=='__main__':main()
