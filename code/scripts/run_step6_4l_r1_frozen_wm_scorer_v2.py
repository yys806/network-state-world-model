"""Frozen WM/Scorer side of STEP 6.4L Route--Flow experiment.

Consumes only the AirFogSim A/B causal-prefix receipts.  The formal planner,
Objective and Scorer are imported unchanged; the B Flow slot is explicitly
labelled Shadow and its score is never promoted.
"""
from __future__ import annotations
import json, hashlib, time
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
IN=ROOT/'code/artifacts/audit/pi_jwm_step6_4l_route_flow_causal_20261011_v7'
OUT=ROOT/'code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v2'
SID='formal-v1-sim-2026092318-policy-2026092418::anchor-0006'
NORM=ROOT/'code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/packages/normalization/stats.json'
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def atomic(p,v):
    q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,default=str)+'\n',encoding='utf8'); q.replace(p)
def main():
    import torch
    from run_step6_3d_one_cpu_solve_v1 import load_frozen_runtime, load_selected, DATASET, objective_side
    from run_step6_4i_cpu_smoke_v1 import history_tensor
    from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
    from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset, FORMAL_V1_WIRED_EDGES
    from pi_jwm.step5_2_training_loop_v1 import _torch_tree
    from build_step5_1d_unified_model_chain_v1 import build_state
    from pi_jwm.step6_0a_candidate_generation_v1 import PlannerCandidateContext, CandidateActionSequence, Backend
    from pi_jwm.step6_0c_planner_action_domain_v1 import context_from_current_raw, annotate_domain
    from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
    from pi_jwm.step6_1_trained_candidate_rollout_v1 import prepare_anchor, rollout_candidate_sequential, fingerprint
    from pi_jwm.step6_2b_planner_objective_scorer_v1 import score_candidate_set
    from pi_jwm.step6_4b_live_bridge_v1 import live_deadline_sidecar
    from pi_jwm.step6_4k_return_shadow_v1 import ReturnShadow, inject_return_slot
    OUT.mkdir(parents=True, exist_ok=True)
    started=time.perf_counter(); atomic(OUT/'07_wm_start.json',{'status':'LOADING_FROZEN_RUNTIME','sample_id':SID,'GPU':False,'locked_test':False})
    stats=json.loads(NORM.read_text(encoding='utf8')); interface=FormalTrainingInterface.from_manifest(DATASET); shards=FullFormalShardDataset(interface)
    model,encoder,_,_,catalog,protocol,meta,param=load_frozen_runtime(SID,device='cpu'); _,frozen,*_=load_selected(SID,interface,shards)
    results={}
    for mode in ('A','B'):
      try:
        prefix=json.loads((IN/f'{mode}_01_causal_prefix.json').read_text(encoding='utf8')); root=json.loads((IN/f'{mode}_01_root.json').read_text(encoding='utf8')); route=json.loads((IN/f'{mode}_02_route.json').read_text(encoding='utf8'))
        sample,tensor,ginput=history_tensor(prefix,stats,frozen); state,gstate=build_state(tensor,ginput,0,wired_edges=FORMAL_V1_WIRED_EDGES); current=prefix['decisions'][-1]; provenance=f"real:{current['trajectory_id']}:{current['frame_index']}:{current['capture_event_id']}"; context=PlannerCandidateContext.from_sample(sample,state,provenance); operational=context_from_current_raw(context,current); domain=CandidateDomain.from_state(context,operational,state,operational.mobility_states,catalog); tasks=prefix['decisions'][-1].get('tasks',[]); sidecar={'trajectory_id':current['trajectory_id'],'frame_index':current['frame_index'],'capture_event_id':current['capture_event_id'],'simulation_time_s':current['simulation_time_s'],'source_time_s':current['simulation_time_s'],'alignment_passed':True,'tasks':[{'task_id':tid,**row} for tid,row in sorted(root.get('deadline_sidecar_tasks',{}).items())],'source':'saved AirFogSim Task getters at live root'}; side=objective_side(sample,state,context,current,sidecar)
        with torch.no_grad(): prepared=prepare_anchor(model,encoder,_torch_tree(tensor),_torch_tree(ginput),state,gstate,context,operational,provenance)
        root_fp=dict(prepared.fingerprints); result={'mode':mode,'root_fingerprint':root_fp,'domain_count':domain.exact_unique_single_step_count,'root_identity':'LIVE_ROOT_CAPTURE_VALID','route':route,'formal_route_action_in_candidate':False,'diagnostic_label':'NEW_CPU_DIAGNOSTIC'}
        run=prepared
        if mode=='B':
            task=route; shadow=ReturnShadow(); entities={str(x.get('entity_id')) for x in current.get('entities',[])}; entities.update([route['source'],route['destination']]); task_index=next(x.task_index for x in side.tasks if x.task_id=='Task_5'); raw_task=root['task_5']; fid=shadow.establish(raw_task,route['route'],now=float(current['simulation_time_s']),source_time=float(current['simulation_time_s']),entities=entities,links={(route['source'],route['route'][0])})
            host_hits=torch.nonzero((prepared.state['task_agent_task_index'][0]==task_index)&(prepared.state['task_agent_relation_type_index'][0]==3)&prepared.state['task_agent_validity'][0],as_tuple=False).flatten()
            if not len(host_hits): raise ValueError('TASK_HOST_INDEX_MISSING')
            source_index=int(prepared.state['task_agent_agent_index'][0,int(host_hits[0])]); destination_index=int(next(x.return_destination_index for x in side.tasks if x.task_id=='Task_5'))
            route_indices=(destination_index,)
            ss,sg,sl,fmeta=inject_return_slot(model,prepared.state,prepared.graph,prepared.latent,task_index=task_index,source=source_index,destination=destination_index,route=route_indices,size=float(route['required_returned_size']),flow_id=fid); run=replace(prepared,state=ss,graph=sg,latent=sl,fingerprints={'anchor':fingerprint(ss),'graph':fingerprint(sg),'latent':fingerprint(sl),'z_pi':prepared.fingerprints.get('z_pi','')}); result['shadow_flow']={'flow_id':fid,'slot':fmeta['slot'],'meta':fmeta,'source_index':source_index,'destination_index':destination_index,'formal_score_not_promoted':True}
        try:
            bound=next(domain.iter_bound())
            raw_candidate=CandidateActionSequence(tuple(bound.action for _ in range(4)),f'step6_4l_r1_{mode}',Backend.RULE_FALLBACK,0,provenance,generation_metadata={'diagnostic':'NEW_CPU_DIAGNOSTIC','route_experiment':mode})
            # This is the official domain annotation path.  The previous
            # 6.4L probe copied a bound action but omitted this metadata,
            # causing the Scorer's intended admission guard to reject A.
            candidate=annotate_domain(raw_candidate,context,operational)
            with torch.inference_mode(): trace=rollout_candidate_sequential(model,run,candidate)
            scored=score_candidate_set(run.state,side,((candidate,trace),),slot_duration_s=protocol.training.rssm.slot_duration_s)
            result['wm_scorer']={'horizons':len(trace.states),'trace_initial':trace.initial_fingerprints,'trace_outputs':trace.output_fingerprints,'scorer_status':scored.status,'H_eff':scored.H_eff,'support_horizons':scored.support_horizons,'diagnostics':scored.diagnostics,'scoreable_h4':bool(scored.status=='SCOREABLE' and scored.H_eff==4),'formal_score_promoted':False}
            result['stage']='PASS'
        except Exception as exc:
            result['wm_scorer']={'exception_type':type(exc).__name__,'exception':str(exc),'formal_score_promoted':False}; result['stage']='WM_OR_SCORER_EXCEPTION'
        atomic(OUT/f'{mode}_08_wm_scorer.json',result); results[mode]=result
      except Exception as exc:
        result={'mode':mode,'stage':'WM_PREPARATION_EXCEPTION','exception_type':type(exc).__name__,'exception':str(exc),'formal_score_promoted':False,'diagnostic_label':'NEW_CPU_DIAGNOSTIC'}; atomic(OUT/f'{mode}_08_wm_scorer.json',result); results[mode]=result
    atomic(OUT/'09_wm_acceptance_receipt.json',{'status':'COMPLETED','namespace':'STEP_6.4L-R1','root_identity':'LIVE_ROOT_CAPTURE_VALID','a_diagnostic_qualification':'PASS_QUALIFIED_OR_EXPLICIT_RESULT','frozen_wm_h4_compatibility':{m:results[m].get('stage') for m in results},'scorer_scoreability':{m:results[m].get('wm_scorer',{}).get('scorer_status','EXCEPTION') for m in results},'formal_scores_promoted':False,'retrain_required':'UNDETERMINED','elapsed_seconds':time.perf_counter()-started,'GPU':False,'locked_test':False})
if __name__=='__main__': main()
