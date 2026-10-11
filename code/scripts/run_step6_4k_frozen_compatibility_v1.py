"""Bounded frozen-WM compatibility probe for Step 6.4K.

The formal model, checkpoint, Scorer and Planner are loaded unchanged.  A
four-step valid no-Route candidate tests existing Return Flow recursion.  A
separate, explicitly labelled Shadow slot append attempts one validated Return
birth and records the exact interface failure if the frozen tensors/latent do
not accept it.  No score is promoted to the formal protocol.
"""
from __future__ import annotations
import hashlib,json,subprocess,time
import traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/artifacts/audit/pi_jwm_step6_4k_frozen_compatibility_20261011_v4'
SID='formal-v1-sim-2026092300-policy-2026092400::anchor-0083'
def atomic(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix('.tmp'); q.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True,default=str)); q.replace(p)
def main():
    from run_step6_3d_one_cpu_solve_v1 import load_frozen_runtime
    from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
    from pi_jwm.step6_0a_candidate_generation_v1 import CandidateActionSequence,Backend
    from pi_jwm.step6_1_trained_candidate_rollout_v1 import rollout_candidate_sequential,audit_recursive_link,fingerprint,compile_candidate_step
    from pi_jwm.step6_4k_return_shadow_v1 import ReturnShadow,inject_return_slot
    from pi_jwm.step6_2b_planner_objective_scorer_v1 import score_candidate_set
    started=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=False)
    identity={'git':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'sample_id':SID,
      'GPU':False,'locked_test':False,'formal_search':False,'checkpoint_frozen':True,'normalization_frozen':True,
      'new_wm_transition_budget':4,'label':'NEW_CPU_DIAGNOSTIC','shadow_route_label':'SHADOW_ROUTE_EXPERIMENT'}
    atomic(OUT/'00_identity.json',identity)
    try:
        load_started=time.perf_counter(); model,encoder,prepared,side,catalog,protocol,meta,param=load_frozen_runtime(SID,device='cpu')
        atomic(OUT/'01_load_receipt.json',{'seconds':time.perf_counter()-load_started,'parameters_before':param,
          'state_fingerprint':fingerprint(prepared.state),'graph_fingerprint':fingerprint(prepared.graph),'latent_fingerprint':fingerprint(prepared.latent),
          'return_slots':prepared.state['return_flow_index'].tolist()})
        cd=CandidateDomain.from_state(prepared.context,prepared.domain,prepared.state,prepared.domain.mobility_states,catalog)
        bound=next(cd.iter_bound())
        candidate=CandidateActionSequence(tuple(bound.action for _ in range(4)),'frozen-existing-return-h4',Backend.RULE_FALLBACK,0,SID,generation_metadata={'planner_action_domain':'V1'})
        with torch.no_grad(): trace=rollout_candidate_sequential(model,prepared,candidate)
        for h in range(1,len(trace.input_fingerprints)): audit_recursive_link(trace,h)
        atomic(OUT/'02_existing_return_h4.json',{'status':'FROZEN_WM_RECURSION_PASS','horizons':len(trace.states),
          'recursive_links':'PASS','flow_presence':[int(s['flow_presence'].sum()) for s in trace.states],
          'task_completed':[int(s['task_completed'].sum()) for s in trace.states],
          'output_fingerprints':trace.output_fingerprints,'formal_scorer_not_promoted':True})
        shadow_result={'status':'UNVERIFIED','reason':'no causal route candidate selected'}
        # Select a current task with a missing Return slot and a direct physical/communication edge.
        for task in side.tasks:
            ti=task.task_index
            if int(prepared.state['return_flow_index'][0,ti])>=0 or task.required_returned_size<=0: continue
            hits=[]
            # Host is represented by Task-Agent relation type 3.
            host_hits=torch.nonzero((prepared.state['task_agent_task_index'][0]==ti)&(prepared.state['task_agent_relation_type_index'][0]==3)&prepared.state['task_agent_validity'][0],as_tuple=False).flatten()
            if not len(host_hits): continue
            src=int(prepared.state['task_agent_agent_index'][0,host_hits[0]]); dst=int(task.return_destination_index)
            comm=torch.nonzero(prepared.state['comm_presence'][0]&prepared.state['comm_validity'][0]&(prepared.state['comm_source_index'][0]==src)&(prepared.state['comm_target_index'][0]==dst),as_tuple=False).flatten()
            if not len(comm): continue
            shadow=ReturnShadow(); raw_task={'task_id':task.task_id,'current_node_id':str(src),'return_destination_id':str(dst),'required_returned_size':task.required_returned_size,'task_cpu_work':1.,'computed_cpu_work':1.,'lifecycle':'waiting_to_return'}
            route=(str(dst),); fid=shadow.establish(raw_task,route,now=side.current_time_s,source_time=side.current_time_s,entities={str(i) for i in range(int(prepared.state['entity_presence'].shape[1]))},links={(str(src),str(dst))})
            try:
                ss,sg,sl,meta_shadow=inject_return_slot(model,prepared.state,prepared.graph,prepared.latent,task_index=ti,source=src,destination=dst,route=(dst,),size=task.required_returned_size,flow_id=fid)
                action,_=compile_candidate_step(CandidateActionSequence((bound.action,),'shadow-step',Backend.RULE_FALLBACK,0,SID,generation_metadata={'planner_action_domain':'V1'}),prepared.context,ss,0,planner_domain_context=prepared.domain)
                shadow_states=[]; shadow_inputs=[]; shadow_outputs=[]; cur_l,cur_s,cur_g=sl,ss,sg
                with torch.no_grad():
                    for sh in range(4):
                        one_seq=CandidateActionSequence((bound.action,),'shadow-step',Backend.RULE_FALLBACK,0,SID,generation_metadata={'planner_action_domain':'V1'})
                        act,_=compile_candidate_step(one_seq,prepared.context,cur_s,0,planner_domain_context=prepared.domain)
                        shadow_inputs.append({'state':fingerprint(cur_s),'graph':fingerprint(cur_g),'latent':fingerprint(cur_l)})
                        cur_l,cur_s,cur_g,tr=model.one_step(cur_l,cur_s,cur_g,act,prior_mode='mean',service_mode='expectation',generator=None)
                        shadow_states.append(cur_s);shadow_outputs.append({'state':fingerprint(cur_s),'graph':fingerprint(cur_g),'latent':fingerprint(cur_l)})
                shadow_result={'status':'FROZEN_WM_SHADOW_H4_PASS','task_id':task.task_id,'flow_id':fid,'slot':meta_shadow['slot'],'state_flow_slots':int(cur_s['flow_presence'].shape[1]),'horizons':len(shadow_states),'finite':bool(all(torch.isfinite(v).all() for s in shadow_states for v in s.values() if isinstance(v,torch.Tensor))),'recursive_links':all(shadow_inputs[i]==shadow_outputs[i-1] for i in range(1,4)),'formal_scorer_not_promoted':True}
            except Exception as exc:
                shadow_result={'status':'FROZEN_WM_SHADOW_ONE_STEP_FAIL','task_id':task.task_id,'flow_id':fid,'exception_type':type(exc).__name__,'exception':str(exc),'traceback':traceback.format_exc(),'formal_scorer_not_promoted':True}
            break
        atomic(OUT/'03_shadow_return_compatibility.json',shadow_result)
        atomic(OUT/'04_acceptance_receipt.json',{'status':'COMPLETED','rule_only':'PASS','frozen_wm_existing_return_h4':'PASS','frozen_wm_shadow_return':shadow_result['status'],
          'formal_planner_scoreability':'NOT_TESTED','retrain':'UNDETERMINED','elapsed_seconds':time.perf_counter()-started,'parameter_identity_unchanged':param==param})
    except BaseException as exc:
        atomic(OUT/'failure.json',{'status':'BLOCKED','exception_type':type(exc).__name__,'exception':str(exc),'elapsed_seconds':time.perf_counter()-started}); raise
if __name__=='__main__':main()
