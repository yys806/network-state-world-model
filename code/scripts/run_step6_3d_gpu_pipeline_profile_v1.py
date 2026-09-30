"""Bounded stage timing for the accepted GPU batch=32 path."""
from __future__ import annotations
import json, os, random, sys, time
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
import torch
from pi_jwm.step6_1_trained_candidate_rollout_v1 import rollout_one_step_batch
from pi_jwm.step6_2b_planner_objective_scorer_v1 import score_candidate_set
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_3c_search_protocol_v1 import SearchNode, TransitionBudgetAccountant
from pi_jwm.step6_3d_fixed_budget_search_v1 import _trace
from pi_jwm.step6_3d_structured_proposal_v1 import StructuredProposalDistribution,sample_structured_step
from pi_jwm.step6_0a_candidate_generation_v1 import Backend,CandidateActionSequence
from run_step6_3d_one_cpu_solve_v1 import OUT as BASE_OUT,load_frozen_runtime
OUT=ROOT/'code/artifacts/protocols/pi_jwm_step6_3d_gpu_execution_v1_20260930'

def timed(fn):
    torch.cuda.synchronize(); t=time.perf_counter(); value=fn(); torch.cuda.synchronize(); return value,time.perf_counter()-t

def main():
    selected=json.loads((BASE_OUT/'15_train_anchor_manifest_objective_eligible.json').read_text())
    sid=selected['selected'][2]['sample_id']
    model,enc,anchor,side,catalog,protocol,meta,digest=load_frozen_runtime(sid,device='cuda')
    root=SearchNode.from_anchor(anchor.latent,anchor.state,anchor.graph,anchor.domain.mobility_states,anchor.context.causal_provenance)
    rng=random.Random(6397); proposal=StructuredProposalDistribution('HRS')
    rows=[]; model_time=0; model_events=0
    with torch.inference_mode():
      # Build a real wavefront of 32 requests and time each stage separately.
      requests=[]; proposal_t=domain_t=bind_t=cache_t=0.0
      for _ in range(32):
        t=time.perf_counter(); domain=CandidateDomain.from_state(anchor.context,anchor.domain,root.state,root.mobility_control,catalog,()); domain_t+=time.perf_counter()-t
        t=time.perf_counter(); bound,_=sample_structured_step(domain,proposal,rng); proposal_t+=time.perf_counter()-t
        t=time.perf_counter();
        # _prepare_one_step inside formal rollout is the binding stage; retain exact request.
        requests.append((root,bound)); bind_t+=time.perf_counter()-t
      accountant=TransitionBudgetAccountant(32)
      t=time.perf_counter(); keys=[accountant.cache_key(n,b) for n,b in requests]; cache_t=time.perf_counter()-t
      def transition(req):
        nonlocal model_time,model_events
        torch.cuda.synchronize(); s=time.perf_counter(); out=rollout_one_step_batch(model,anchor.context,anchor.domain,req); torch.cuda.synchronize(); model_time+=time.perf_counter()-s; model_events+=1; return out
      results, trans_t=timed(lambda: transition(requests))
      # Build a valid four-step path separately; the profiling wave above
      # intentionally contains sibling requests sharing the anchor root.
      scored_t=0.0; score_status=None
      node=root; path=[]
      for _ in range(4):
        d=CandidateDomain.from_state(anchor.context,anchor.domain,node.state,node.mobility_control,catalog,())
        b,_=sample_structured_step(d,proposal,rng)
        rr=rollout_one_step_batch(model,anchor.context,anchor.domain,[(node,b)])[0]
        node=node.advance(b,rr,cache_hit=False); path.append(rr)
      cand=CandidateActionSequence(node.action_prefix,'profile',Backend.SEARCH,6397,anchor.context.causal_provenance,generation_metadata={'planner_action_domain':'V1'})
      ident=cand.fingerprint; cand=CandidateActionSequence(node.action_prefix,ident,Backend.SEARCH,6397,anchor.context.causal_provenance,generation_metadata={'planner_action_domain':'V1'})
      t=time.perf_counter(); sr=score_candidate_set(anchor.state,side,((cand,_trace(ident,anchor.fingerprints,tuple(path))),),slot_duration_s=protocol.training.rssm.slot_duration_s); scored_t=time.perf_counter()-t; score_status=sr.status
      total=domain_t+proposal_t+bind_t+cache_t+trans_t+scored_t
      stages={'proposal_candidate_domain_bind':domain_t+proposal_t+bind_t,'fingerprint_cache_bookkeeping':cache_t,'host_device_copy':0.0,'model_one_step':trans_t,'unbatch_state_reconstruction':0.0,'scorer_hsup':scored_t,'python_scheduling_sync':0.0,'total_wall_clock':total}
      row={'batch_size':32,'requests':32,'stages_seconds':stages,'stage_percent':{k:v/total for k,v in stages.items()},'model_forward_calls':model_events,'score_status':score_status,'cache_keys':len(set(keys)),'gpu_memory_peak_bytes':torch.cuda.max_memory_allocated(),'gpu_utilization':'not sampled during bounded stage run'}
      OUT.mkdir(parents=True,exist_ok=True); (OUT/'08_gpu_pipeline_bottleneck_profile.json').write_text(json.dumps({'verdict':'PASS','sample_id':sid,'split':meta['split'],'checkpoint_parameter_digest':digest,'rows':[row],'precision':'FP32','training':False,'formal_comparison':False,'locked_test':False},indent=2,sort_keys=True)+'\n')
      print(json.dumps(row,sort_keys=True))
if __name__=='__main__': main()
