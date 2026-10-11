"""STEP 6.4J additive live-root H4 diagnostic; no search or semantic changes."""
from __future__ import annotations
import argparse, json, os, sys, time, traceback, hashlib
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"): sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts'),str(ROOT/'code/reference/AirFogSim')]
OUT=ROOT/'code/artifacts/audit/pi_jwm_step6_4j_live_root_capture_20261011_v4'
NORM=ROOT/'code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/packages/normalization/stats.json'
from run_step6_3d_one_cpu_solve_v1 import load_frozen_runtime,load_selected,DATASET,objective_side
from run_step6_4i_cpu_smoke_v1 import history_tensor
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset,FORMAL_V1_WIRED_EDGES
from pi_jwm.step5_2_training_loop_v1 import _torch_tree
from build_step5_1d_unified_model_chain_v1 import build_state
from pi_jwm.step6_0a_candidate_generation_v1 import PlannerCandidateContext,CandidateActionSequence,Backend
from pi_jwm.step6_0c_planner_action_domain_v1 import context_from_current_raw
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_1_trained_candidate_rollout_v1 import prepare_anchor,rollout_candidate_sequential,fingerprint
from pi_jwm.step6_2b_planner_objective_scorer_v1 import score_candidate_set
from pi_jwm.step6_4b_live_bridge_v1 import live_deadline_sidecar
class StopCapture(BaseException): pass

def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix('.tmp'); t.write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,default=str)+'\n',encoding='utf8'); t.replace(p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run_episode(ep, max_candidates=2):
 import collect_step5_5_formal_raw_v1 as collector
 from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset
 stats=json.loads(NORM.read_text()); interface=FormalTrainingInterface.from_manifest(DATASET); shards=FullFormalShardDataset(interface)
 sid=ep['sample_id']; model,encoder,_,_,catalog,protocol,meta,param=load_frozen_runtime(sid,device='cpu'); _,frozen,*_=load_selected(sid,interface,shards)
 rec={'schema':'PI-JWM-STEP-6.4J-LIVE-ROOT-CAPTURE-v1','diagnostic_label':'NEW_CPU_DIAGNOSTIC','sample_id':sid,'trajectory_id':ep['trajectory_id'],'target_frame':ep['initial_frame'],'GPU':'NOT_USED','locked_test':False,'R39_EXACT_EQUIVALENCE':'UNPROVEN','stages':{},'candidates':[]}
 atomic(OUT/f'{ep["trajectory_id"]}_identity.json',{'status':'RUNNING','sample_id':sid,'checkpoint_sha256':sha(ROOT/'code/artifacts/formal_training/pi_jwm_formal_training_v1_20260924/checkpoints/best.pt') if (ROOT/'code/artifacts/formal_training/pi_jwm_formal_training_v1_20260924/checkpoints/best.pt').exists() else None,'normalization_sha256':sha(NORM),'source_commit':'0b408e692e095bf78ad85023bd32bee5f9688565'})
 def hook(env,decisions,steps,config,communication):
  if int(decisions[-1]['frame_index'])!=int(ep['initial_frame']): return
  t0=time.perf_counter(); current=decisions[-1]; raw={'environment':{'seed':ep['simulator_seed'],'policy_seed':ep['policy_seed'],'wired_edges':collector.WIRED_EDGES},'decisions':decisions,'steps':steps}
  rec['capture']={'simulator_seed':ep['simulator_seed'],'policy_seed':ep['policy_seed'],'frame':current['frame_index'],'simulation_time_s':current['simulation_time_s'],'capture_event_id':current['capture_event_id'],'decision_count':len(decisions),'step_count':len(steps),'causal_prefix_sha256':hashlib.sha256(json.dumps(raw,sort_keys=True,default=str).encode()).hexdigest()}
  atomic(OUT/f'{ep["trajectory_id"]}_capture.json',rec['capture']); rec['stages']['root_capture']='PASS'
  sample,tensor,ginput=history_tensor(raw,stats,frozen); state,gstate=build_state(tensor,ginput,0,wired_edges=FORMAL_V1_WIRED_EDGES); tasks=collector.step23._all_runtime_tasks(env); provenance=f'real:{current["trajectory_id"]}:{current["frame_index"]}:{current["capture_event_id"]}'; context=PlannerCandidateContext.from_sample(sample,state,provenance); operational=context_from_current_raw(context,current); domain=CandidateDomain.from_state(context,operational,state,operational.mobility_states,catalog); side=objective_side(sample,state,context,current,live_deadline_sidecar(env,current,tasks)); 
  import torch
  with torch.no_grad(): prepared=prepare_anchor(model,encoder,_torch_tree(tensor),_torch_tree(ginput),state,gstate,context,operational,provenance)
  rootfp=dict(prepared.fingerprints); rec['root']={'fingerprints':rootfp,'domain_count':domain.exact_unique_single_step_count,'domain_empty':domain.is_empty,'empty_reason':domain.empty_reason,'cohort_size':len(context.static.get('input_task_index',{})),'objective_side_keys':sorted(side.__dict__.keys()) if hasattr(side,'__dict__') else sorted(side.keys())}; rec['stages']['root_identity']='LIVE_ROOT_CAPTURE_VALID' if not domain.is_empty else 'LIVE_ROOT_CAPTURE_VALID_DOMAIN_EMPTY'; atomic(OUT/f'{ep["trajectory_id"]}_root.json',rec['root'])
  if domain.is_empty: raise StopCapture()
  bounds=[]
  for b in domain.iter_bound():
   bounds.append(b)
   if len(bounds)>=max_candidates: break
  for i,bound in enumerate(bounds,1):
   candidate=CandidateActionSequence(tuple(bound.action for _ in range(4)),f'live_diag_{i}',Backend.RULE_FALLBACK,6311,provenance,generation_metadata={'planner_action_domain':'V1','diagnostic':'NEW_CPU_DIAGNOSTIC'})
   started=time.perf_counter(); row={'candidate_index':i,'candidate_fingerprint':candidate.fingerprint,'root_fingerprint':rootfp,'horizon':4,'action_prefix':[s.frame() for s in candidate.steps],'wm_unique_transition_budget':4}
   try:
    with __import__('torch').inference_mode(): trace=rollout_candidate_sequential(model,prepared,candidate)
    row['rollout_seconds']=time.perf_counter()-started; row['trace_initial']=trace.initial_fingerprints; row['trace_outputs']=trace.output_fingerprints; row['completed_horizons']=len(trace.states)
    try:
     scored=score_candidate_set(state,side,((candidate,trace),),slot_duration_s=protocol.training.rssm.slot_duration_s)
     row.update({'scorer_status':scored.status,'H_eff':scored.H_eff,'H_sup':[s.H_sup for s in scored.scores],'support_horizons':scored.support_horizons,'diagnostics':scored.diagnostics,'scoreable_h4':bool(scored.status=='SCOREABLE' and scored.H_eff==4 and scored.scores and scored.scores[0].H_sup==4)})
    except Exception as exc: row.update({'scorer_exception':type(exc).__name__,'scorer_error':str(exc)})
   except Exception as exc: row.update({'rollout_exception':type(exc).__name__,'rollout_error':str(exc),'rollout_seconds':time.perf_counter()-started})
   rec['candidates'].append(row); atomic(OUT/f'{ep["trajectory_id"]}_candidates.json',rec)
  rec['stages']['diagnostic']='PASS'; rec['parameter_unchanged']=fingerprint({'encoder':encoder.state_dict(),'rssm':model.state_dict()})==param; atomic(OUT/f'{ep["trajectory_id"]}_final.json',rec); raise StopCapture()
 try: collector.collect_trajectory(ep['simulator_seed'],ep['policy_seed'],ep['trajectory_id'],on_decision=hook)
 except StopCapture: pass
 except BaseException as exc: rec['stages']['error']=type(exc).__name__+':'+str(exc); atomic(OUT/f'{ep["trajectory_id"]}_final.json',rec); return rec
 return rec

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 manifest=json.loads((ROOT/'code/artifacts/protocols/pi_jwm_step6_4j_s_cem_gpu_pilot_v1_20261010_r39/01_manifest.json').read_text())
 for ep in sorted(manifest['episodes'],key=lambda x:0 if x['initial_frame']==6 else 1):
  t=time.perf_counter(); print(json.dumps({'start':ep['sample_id']},ensure_ascii=False),flush=True); r=run_episode(ep); print(json.dumps({'done':ep['sample_id'],'seconds':time.perf_counter()-t,'candidates':len(r.get('candidates',[]))},ensure_ascii=False),flush=True)
if __name__=='__main__': main()




