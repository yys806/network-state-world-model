"""STEP 6.4J H1 Return-birth action sensitivity diagnostic."""
from __future__ import annotations
import json, os, sys, time, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('PYTHONIOENCODING','utf-8')
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts'),str(ROOT/'code/reference/AirFogSim')]
OUT=ROOT/'code/artifacts/audit/pi_jwm_step6_4j_h1_return_action_sensitivity_20261011_v2'
NORM=ROOT/'code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/packages/normalization/stats.json'
from run_step6_3d_one_cpu_solve_v1 import load_frozen_runtime,load_selected,DATASET,EXPECTED_SHA,CHECKPOINT
from run_step6_4i_cpu_smoke_v1 import history_tensor
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset,FORMAL_V1_WIRED_EDGES
from pi_jwm.step5_2_training_loop_v1 import _torch_tree
from build_step5_1d_unified_model_chain_v1 import build_state
from pi_jwm.step6_0a_candidate_generation_v1 import PlannerCandidateContext,CandidateActionSequence,Backend
from pi_jwm.step6_0c_planner_action_domain_v1 import context_from_current_raw,annotate_domain
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_1_trained_candidate_rollout_v1 import prepare_anchor,rollout_candidate_sequential,fingerprint
from pi_jwm.step6_2b_planner_objective_scorer_v1 import score_candidate_set,_return_slot,_compute_host
from pi_jwm.step6_2a_planner_objective_side_state_v1 import planner_derived_return_birth_required
from pi_jwm.step6_4b_live_bridge_v1 import live_deadline_sidecar
class StopCapture(BaseException): pass

def atomic(path,value):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,default=str)+'\n',encoding='utf8'); tmp.replace(path)
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def action_key(c): return c.fingerprint
def task_state(side,state,task):
 ti=task.task_index; rem=float(state['task_work_remaining'][0,ti]); existing=_return_slot(state,ti); host=None
 if rem<=0 and not existing: host=_compute_host(state,ti)
 return {'task_id':task.task_id,'task_index':ti,'work_remaining':rem,'return_exists':existing,'compute_host_index':host,'return_destination_index':task.return_destination_index,'required_returned_size':task.required_returned_size}
def classify(task,anchor,pred):
 if anchor['work_remaining']>0 and pred['work_remaining']<=0: return 'A'
 try: anchor_birth=planner_derived_return_birth_required(task,anchor['work_remaining']<=0,anchor['return_exists'],anchor['compute_host_index'])
 except Exception: anchor_birth=False
 if anchor['work_remaining']<=0 and anchor_birth: return 'B'
 return 'C'
def run_episode(ep,max_actions=12):
 import collect_step5_5_formal_raw_v1 as collector
 import torch
 stats=json.loads(NORM.read_text()); interface=FormalTrainingInterface.from_manifest(DATASET); shards=FullFormalShardDataset(interface); sid=ep['sample_id']
 tload=time.perf_counter(); model,encoder,_,_,catalog,protocol,meta,param=load_frozen_runtime(sid,device='cpu'); _,frozen,*_=load_selected(sid,interface,shards)
 rec={'schema':'PI-JWM-STEP-6.4J-H1-RETURN-ACTION-SENSITIVITY-v1','diagnostic_label':'NEW_CPU_DIAGNOSTIC','sample_id':sid,'trajectory_id':ep['trajectory_id'],'target_frame':ep['initial_frame'],'GPU':'NOT_USED','locked_test':False,'R39_EXACT_EQUIVALENCE':'UNPROVEN','identity':{'source_commit':'3cde78e551915a98290e53e20521183f3aa099c6','checkpoint_sha256':EXPECTED_SHA,'normalization_sha256':digest(NORM),'diagnostic_script_sha256':digest(Path(__file__)),'model_load_seconds':time.perf_counter()-tload},'actions':[]}
 def hook(env,decisions,steps,config,communication):
  if int(decisions[-1]['frame_index'])!=int(ep['initial_frame']): return
  root_t=time.perf_counter(); current=decisions[-1]; raw={'environment':{'seed':ep['simulator_seed'],'policy_seed':ep['policy_seed'],'wired_edges':collector.WIRED_EDGES},'decisions':decisions,'steps':steps}
  sample,tensor,ginput=history_tensor(raw,stats,frozen); state,gstate=build_state(tensor,ginput,0,wired_edges=FORMAL_V1_WIRED_EDGES); tasks=collector.step23._all_runtime_tasks(env); provenance=f'real:{current["trajectory_id"]}:{current["frame_index"]}:{current["capture_event_id"]}'; context=PlannerCandidateContext.from_sample(sample,state,provenance); operational=context_from_current_raw(context,current); domain=CandidateDomain.from_state(context,operational,state,operational.mobility_states,catalog); side=__import__('run_step6_3d_one_cpu_solve_v1',fromlist=['objective_side']).objective_side(sample,state,context,current,live_deadline_sidecar(env,current,tasks))
  with torch.no_grad(): prepared=prepare_anchor(model,encoder,_torch_tree(tensor),_torch_tree(ginput),state,gstate,context,operational,provenance)
  rootfp=dict(prepared.fingerprints); target_ids=[t.task_id for t in side.tasks if t.task_id in {'Task_5','Task_15'}]
  raw_tasks={str(t['task_id']):t for t in current.get('tasks',[])}; anchor_tasks={t.task_id:task_state(side,state,t) for t in side.tasks if t.task_id in target_ids}
  rec['root']={'capture':{'frame':current['frame_index'],'simulation_time_s':current['simulation_time_s'],'capture_event_id':current['capture_event_id'],'decision_count':len(decisions),'step_count':len(steps),'causal_prefix_sha256':hashlib.sha256(json.dumps(raw,sort_keys=True,default=str).encode()).hexdigest(),'causal_prefix':raw},'fingerprints':rootfp,'domain_count':domain.exact_unique_single_step_count,'target_tasks':target_ids,'anchor_task_state':anchor_tasks,'raw_target_lifecycle':{k:{x:raw_tasks[k].get(x) for x in ('lifecycle','current_node_id','return_destination_id','required_returned_size','computed_cpu_work','task_cpu_work','transmitted_size')} for k in target_ids},'objective_side_task_count':len(side.tasks),'scorer_cohort_size':sum(1 for t in side.tasks if not t.terminal_at_anchor and bool(state['task_presence'][0,t.task_index]) and not bool(state['task_completed'][0,t.task_index])),'objective_side_source':side.source,'root_prepare_seconds':time.perf_counter()-root_t}
  atomic(OUT/f'{ep["trajectory_id"]}_root.json',rec['root'])
  seen=set(); candidates=[]
  for bound in domain.iter_bound():
   raw_candidate=CandidateActionSequence((bound.action,),f'h1_candidate_{len(candidates)+1}',Backend.RULE_FALLBACK,6311,provenance,generation_metadata={'planner_action_domain':'V1','diagnostic':'H1_RETURN_ACTION_SENSITIVITY'})
   try: candidate=annotate_domain(raw_candidate,context,operational)
   except Exception as exc: rec.setdefault('invalid_actions',[]).append({'action_frame':bound.action.frame(),'error_type':type(exc).__name__,'error':str(exc)}); continue
   if candidate.fingerprint in seen: continue
   seen.add(candidate.fingerprint); candidates.append(candidate)
   if len(candidates)>=max_actions: break
  for idx,candidate in enumerate(candidates,1):
   started=time.perf_counter(); row={'action_index':idx,'candidate_fingerprint':candidate.fingerprint,'root_fingerprint':rootfp,'action':candidate.steps[0].frame(),'action_components':{'comm':len(candidate.steps[0].comm),'comp':len(candidate.steps[0].comp),'mobility':len(candidate.steps[0].mob),'route':len(candidate.steps[0].route)},'same_root':True}
   if candidates: row['changed_components_vs_baseline']=[name for name in ('comm','comp','mobility','route') if getattr(candidate.steps[0],name)!=getattr(candidates[0].steps[0],name)]
   try:
    with torch.no_grad(): trace=rollout_candidate_sequential(model,prepared,candidate)
    pred=trace.states[0]; row['rollout_seconds']=time.perf_counter()-started; row['completed_h1']=len(trace.states)==1; row['trace_output_fingerprints']=trace.output_fingerprints
    row['target_predicted_state']={t.task_id:task_state(side,pred,t) for t in side.tasks if t.task_id in target_ids}
    row['target_classification']={t.task_id:classify(t,anchor_tasks[t.task_id],row['target_predicted_state'][t.task_id]) for t in side.tasks if t.task_id in target_ids}
    try:
     scored=score_candidate_set(state,side,((candidate,trace),),slot_duration_s=protocol.training.rssm.slot_duration_s); row.update({'scorer_status':scored.status,'H_eff':scored.H_eff,'H_sup':[s.H_sup for s in scored.scores],'support_horizons':scored.support_horizons,'scorer_diagnostics':scored.diagnostics,'scoreable_h1':bool(scored.status=='SCOREABLE' and scored.H_eff==1 and scored.scores and scored.scores[0].H_sup==1)})
    except Exception as exc: row.update({'scorer_exception_type':type(exc).__name__,'scorer_exception':str(exc)})
   except Exception as exc: row.update({'rollout_exception_type':type(exc).__name__,'rollout_exception':str(exc),'rollout_seconds':time.perf_counter()-started})
   rec['actions'].append(row); atomic(OUT/f'{ep["trajectory_id"]}_actions.json',rec)
  rec['final']={'root_status':'LIVE_ROOT_CAPTURE_VALID','legal_distinct_actions':len(candidates),'invalid_actions':len(rec.get('invalid_actions',[])),'successful_h1_predictions':sum(bool(r.get('completed_h1')) for r in rec['actions']),'return_birth_triggers':sum('UNSUPPORTED_FUTURE_RETURN_BIRTH' in str(r.get('scorer_diagnostics',{})) for r in rec['actions']),'h1_support_pass':sum(r.get('H_sup')==[1] for r in rec['actions']),'h1_scoreable':sum(bool(r.get('scoreable_h1')) for r in rec['actions']),'rollout_exceptions':sum('rollout_exception' in r for r in rec['actions']),'scorer_exceptions':sum('scorer_exception' in r for r in rec['actions']),'parameters_unchanged':fingerprint({'encoder':encoder.state_dict(),'rssm':model.state_dict()})==param,'wm_transitions':sum(1 for r in rec['actions'] if r.get('completed_h1')),'environment_advanced_during_capture':False}; atomic(OUT/f'{ep["trajectory_id"]}_final.json',rec); raise StopCapture()
 try: collector.collect_trajectory(ep['simulator_seed'],ep['policy_seed'],ep['trajectory_id'],on_decision=hook)
 except StopCapture: pass
 except BaseException as exc: rec['error']={'type':type(exc).__name__,'error':str(exc)}; atomic(OUT/f'{ep["trajectory_id"]}_final.json',rec)
 return rec
def main():
 OUT.mkdir(parents=True,exist_ok=False); manifest=json.loads((ROOT/'code/artifacts/protocols/pi_jwm_step6_4j_s_cem_gpu_pilot_v1_20261010_r39/01_manifest.json').read_text())
 for ep in sorted(manifest['episodes'],key=lambda x:0 if x['initial_frame']==6 else 1):
  t=time.perf_counter(); print(json.dumps({'start':ep['sample_id']},ensure_ascii=False),flush=True); r=run_episode(ep); print(json.dumps({'done':ep['sample_id'],'seconds':time.perf_counter()-t,'actions':len(r.get('actions',[]))},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
