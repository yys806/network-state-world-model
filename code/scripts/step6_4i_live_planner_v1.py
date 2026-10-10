"""Live S-CEM provider: fresh history-only posterior per solve, no warm start.

 No CLI or implicit GPU launch. Formal CUDA configuration needs an explicitly
 approved execution identity; this Step executes CPU engineering smoke only.
 """
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
import numpy as np
import torch
from pi_jwm.step6_4b_live_bridge_v1 import build_live_sample,live_deadline_sidecar,require_nonempty_domain,SmokeFailure
from build_step5_5_formal_dataset_v1 import normalize
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import build_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import build_typed_dual_graph_batch,PhysicalTopologyConfig
from pi_jwm.step5_5_full_sharded_loader_v1 import FORMAL_V1_WIRED_EDGES
from pi_jwm.step5_2_training_loop_v1 import _torch_tree
from pi_jwm.step6_0a_candidate_generation_v1 import PlannerCandidateContext
from pi_jwm.step6_0c_planner_action_domain_v1 import context_from_current_raw
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_3d_fixed_budget_search_v1 import solve_fixed_budget
from pi_jwm.step6_1_trained_candidate_rollout_v1 import prepare_anchor,rollout_one_step,rollout_one_step_batch,fingerprint
from pi_jwm.step6_2b_planner_objective_scorer_v1 import score_candidate_set
from pi_jwm.step6_4i_episode_v1 import PlanPacket
from build_step5_1d_unified_model_chain_v1 import build_state
from run_step6_3d_one_cpu_solve_v1 import objective_side,_to_device,offload_prepared_anchor,gpu_transition_one_with_cpu_storage,gpu_transition_batch_with_cpu_storage,EXPECTED_SHA

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

class LiveSCEMPlanner:
 def __init__(self,*,model,encoder,stats,slot_template,catalog,protocol,device='cpu',engineering_only=True,approved_execution=None):
  if engineering_only:
   if device!='cpu':raise PermissionError('CPU mechanism mode only')
   self.budget=64;self.batch=1
  else:
   expected={'status':'APPROVED_BY_RESEARCHER','method':'S-CEM','K':4,'rho':.2,'B_WM':512,'batch_size':16,'precision':'FP32','gpu_model':'NVIDIA GeForce RTX 3080 Ti','checkpoint_sha256':EXPECTED_SHA}
   if device!='cuda' or not approved_execution or any(approved_execution.get(k)!=v for k,v in expected.items()):raise PermissionError('formal GPU execution identity/authorization pending')
   self.budget=512;self.batch=16
  self.model=model;self.encoder=encoder;self.stats=stats;self.template=slot_template;self.catalog=catalog;self.protocol=protocol;self.device=device
 def plan(self,raw,env,runtime_tasks,*,seed,budget_progress=None):
  before=float(env.simulation_time);current=raw['decisions'][-1]
  sample,tensor,ginput=history_tensor(raw,self.stats,self.template)
  state,gstate=build_state(tensor,ginput,0,wired_edges=FORMAL_V1_WIRED_EDGES)
  provenance=f"real:{current['trajectory_id']}:{current['frame_index']}:{current['capture_event_id']}"
  context=PlannerCandidateContext.from_sample(sample,state,provenance)
  operational=context_from_current_raw(context,current)
  domain=CandidateDomain.from_state(context,operational,state,operational.mobility_states,self.catalog);require_nonempty_domain(domain)
  side=objective_side(sample,state,context,current,live_deadline_sidecar(env,current,runtime_tasks))
  traces={}
  def score(candidate,trace):
   scored=score_candidate_set(state,side,((candidate,trace),),slot_duration_s=self.protocol.training.rssm.slot_duration_s)
   if scored.status!='SCOREABLE' or scored.H_eff!=4 or not scored.scores or scored.scores[0].H_sup!=4:return None
   traces[candidate.fingerprint]=trace;return scored.scores[0]
  timer=time.perf_counter()
  with torch.inference_mode():
   device=torch.device(self.device)
   prepared=prepare_anchor(self.model,self.encoder,_to_device(_torch_tree(tensor),device),_to_device(_torch_tree(ginput),device),_to_device(state,device),_to_device(gstate,device),context,operational,provenance)
   if self.device=='cuda':
    prepared=offload_prepared_anchor(prepared)
    def one(n,b):
     if budget_progress:budget_progress(1)
     return gpu_transition_one_with_cpu_storage(self.model,context,operational,n,b)
    def batch(requests):
     if budget_progress:budget_progress(len(requests))
     return gpu_transition_batch_with_cpu_storage(self.model,context,operational,requests)
   else:
    def one(n,b):
     if budget_progress:budget_progress(1)
     return rollout_one_step(self.model,context,operational,n.latent,n.state,n.graph,n.mobility_control,b)
    def batch(requests):
     if budget_progress:budget_progress(len(requests))
     return rollout_one_step_batch(self.model,context,operational,requests)
   result=solve_fixed_budget(method='S-CEM',seed=seed,b_wm=self.budget,iterations=4,elite_ratio=.2,batch_size=self.batch,anchor=prepared,catalog=self.catalog,transition=one,transition_batch=batch,score_h4=score)
  if float(env.simulation_time)!=before:raise SmokeFailure('SIMULATION_ADVANCED_DURING_PLANNING')
  predicted=None if result.best_fingerprint not in traces else fingerprint(traces[result.best_fingerprint].states[0])
  return PlanPacket(result,domain,dict(prepared.fingerprints),predicted,time.perf_counter()-timer)
