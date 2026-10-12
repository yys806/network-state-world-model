"""Gate 2/3: two supplied candidates from one audited real pre-Route root."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,sys,time,threading
from pathlib import Path
from dataclasses import replace,asdict
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
IN=ROOT/'code/artifacts/audit/pi_jwm_step6_4n_20261012_v9'
OUT=ROOT/'code/artifacts/audit/pi_jwm_step6_4n_20261012_wm_v5'

def atomic(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,default=str)+'\n',encoding='utf8');tmp.replace(p)

def main():
    import torch
    from run_step6_3d_one_cpu_solve_v1 import load_frozen_runtime,load_selected,DATASET,CHECKPOINT,EXPECTED_SHA,objective_side,sha
    from run_step6_4i_cpu_smoke_v1 import history_tensor,NORM
    from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
    from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset,FORMAL_V1_WIRED_EDGES
    from pi_jwm.step5_2_training_loop_v1 import _torch_tree
    from build_step5_1d_unified_model_chain_v1 import build_state
    from pi_jwm.step6_0a_candidate_generation_v1 import PlannerCandidateContext,CandidateActionStep,CandidateActionSequence,Backend
    from pi_jwm.step6_0c_planner_action_domain_v1 import context_from_current_raw
    from pi_jwm.step6_1_trained_candidate_rollout_v1 import prepare_anchor,fingerprint,clone_tree
    from pi_jwm.step6_4n_dynamic_v2 import compact_anchor_support,DynamicScorerV2,FLOW_KEYS
    from pi_jwm.step6_4n_rollout_v2 import rollout_dynamic_v2
    from pi_jwm.step6_4b_live_bridge_v1 import causal_prefix
    from pi_jwm.step4_2a_graph_input_extension_v1 import amend_raw_graph_inputs
    from pi_jwm.step4_2c_b_causal_flow_ledger_raw_v1 import amend_raw_with_causal_flow_ledger
    from audit_step6_4n_gate1_v2 import audit
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'wm_attempt.json').exists():raise ValueError('WM_NAMESPACE_ALREADY_ATTEMPTED')
    if audit()['status']!='PASS':raise ValueError('GATE1_NOT_PASS')
    import hashlib
    sources=(Path(__file__),ROOT/'code/src/pi_jwm/step6_4n_dynamic_v2.py',ROOT/'code/src/pi_jwm/step6_4n_rollout_v2.py',ROOT/'code/src/pi_jwm/step6_4k_return_shadow_v1.py')
    started=time.perf_counter();atomic(OUT/'wm_attempt.json',{'CPU':True,'max_transitions':8,'root_count':1,'H':4,'locked_test':False,'initialization':'ZERO_COLD_START','executed_source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}})
    def timeout():
        atomic(OUT/'wm_timeout.json',{'status':'BLOCKED','reason':'CPU_WM_300S'});os._exit(124)
    timer=threading.Timer(300,timeout);timer.daemon=True;timer.start()
    try:
        sid='formal-v1-sim-2026092318-policy-2026092418::anchor-0006'
        model,encoder,_,_,catalog,protocol,_,_=load_frozen_runtime(sid,device='cpu')
        param_before=fingerprint(model.state_dict());enc_before=fingerprint(encoder.state_dict())
        interface=FormalTrainingInterface.from_manifest(DATASET);_,frozen,*_=load_selected(sid,interface,FullFormalShardDataset(interface))
        prefix=json.loads((IN/'V2_NOOP/01_prefix.json').read_text(encoding='utf8'))
        root=json.loads((IN/'V2_NOOP/02_root.json').read_text(encoding='utf8'))
        # Static simulator configuration, independently frozen before outcomes.
        from pi_jwm.step5_5_full_sharded_loader_v1 import FORMAL_V1_WIRED_EDGES
        prefix['environment']={'seed':2026092318,'wired_edges':FORMAL_V1_WIRED_EDGES}
        stats=json.loads(NORM.read_text(encoding='utf8'))
        sample,tensor,gin=history_tensor(prefix,stats,frozen)
        poisoned=clone_tree(prefix);poisoned['target']=object();poisoned['future_action']=object()
        ps,pt,pg=history_tensor(poisoned,stats,frozen)
        poison_pass=fingerprint(_torch_tree(tensor))==fingerprint(_torch_tree(pt)) and fingerprint(_torch_tree(gin))==fingerprint(_torch_tree(pg))
        state,graph=build_state(tensor,gin,0,wired_edges=FORMAL_V1_WIRED_EDGES)
        current=prefix['decisions'][-1];context=PlannerCandidateContext.from_sample(sample,state,sid);domain=context_from_current_raw(context,current)
        sidecar={k:current[k] for k in ('trajectory_id','frame_index','capture_event_id','simulation_time_s')}
        sidecar.update(source_time_s=current['simulation_time_s'],alignment_passed=True,tasks=[{'task_id':tid,**r} for tid,r in root['deadline_sidecar_tasks'].items()])
        side=objective_side(sample,state,context,current,sidecar)
        with torch.no_grad():
            anchor=prepare_anchor(model,encoder,_torch_tree(tensor),_torch_tree(gin),state,graph,context,domain,sid)
            ss,sg,sl=compact_anchor_support(model,anchor.state,anchor.graph,anchor.latent)
        # Both candidates see this same causal support, with padding removed
        # before any candidate action. All owned tensor/hidden values retained.
        n=ss['flow_known'].shape[1]
        compact_preserves=all(torch.equal(anchor.state[k][:,:n],ss[k]) for k in FLOW_KEYS) and torch.equal(anchor.latent['h']['flow'][:,:n],sl['h']['flow'])
        fp={'state':fingerprint(ss),'graph':fingerprint(sg),'latent':fingerprint(sl),'z_pi':anchor.fingerprints['z_pi']};fp['anchor']=fingerprint(fp)
        anchor=replace(anchor,state=ss,graph=sg,latent=sl,fingerprints=fp,context=replace(context,current_state=ss))
        amended,receipt=amend_raw_with_causal_flow_ledger(amend_raw_graph_inputs(causal_prefix(prefix)))
        last=amended['decisions'][-1]
        snapshot={'logical_flow_rows':last['logical_flow_rows'],'carrying_rows':last['carrying_rows']}
        torch.save({'state':ss,'graph':sg,'latent':sl,'side':side,'sample':sample,'current':current,'snapshot':snapshot,'fingerprints':fp},OUT/'common_anchor.pt')
        indices=sample['static']['input_entity_index'];ti=int(indices['task']['Task_5']);src=int(indices['physical']['UAV_0']);dst=int(indices['physical']['UAV_1'])
        hits=torch.nonzero(ss['comm_presence'][0]&ss['comm_validity'][0]&ss['comm_wireless_mask'][0]&(ss['comm_source_index'][0]==src)&(ss['comm_target_index'][0]==dst)).flatten()
        if len(hits)!=1:raise ValueError('TASK5_RELATION_NOT_UNIQUE')
        ri=int(hits[0])
        noop=CandidateActionStep()
        route={'task_id':'Task_5','task_index':ti,'route_kind':'return','route_node_ids':['UAV_1'],'route_node_indices':[dst]}
        comm={'task_id':'Task_5','task_index':ti,'relation_index':ri,'rb_indices':[0]}
        candidates=(CandidateActionSequence((noop,)*4,'A_NOOP',Backend.RULE_FALLBACK,None,sid),CandidateActionSequence((CandidateActionStep(route=(route,),comm=(comm,)),noop,noop,noop),'B_ROUTE_RB',Backend.RULE_FALLBACK,None,sid))
        pairs=[];facts={}
        def finite(v):
            if isinstance(v,torch.Tensor):return bool(torch.isfinite(v).all())
            if isinstance(v,dict):return all(finite(x) for x in v.values())
            if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
            return True
        for c in candidates:
            with torch.inference_mode():tr,meta=rollout_dynamic_v2(model,anchor,c,current_raw=current,ledger_snapshot=snapshot,catalog=catalog,slot_duration_s=.1,tti_threshold=.5)
            torch.save({'candidate':c,'trace':tr,'meta':meta},OUT/f'{c.candidate_id}_trace.pt')
            conservation=[];aligned=[]
            realmode='V2_NOOP' if c.candidate_id=='A_NOOP' else 'V2_ROUTE_RB'
            for h,s in enumerate(tr.states,1):
                slot=int(s['return_flow_index'][0,ti]);remaining=float(s['flow_remaining'][0,slot]) if slot>=0 else float(root['task_raw']['required_returned_size'])
                total=float(s['flow_total'][0,slot]) if slot>=0 else float(root['task_raw']['required_returned_size'])
                real=json.loads((IN/realmode/f'04_h{h}.json').read_text(encoding='utf8'))
                rr=real['ledger']['logical_flow_rows'];rremaining=float(rr[0]['e2e_remaining']) if rr else float(root['task_raw']['required_returned_size'])
                done=any(x['task_id']=='Task_5' for x in real['buckets']['_done_tasks'])
                conservation.append({'h':h,'flow_id':meta['births'][0]['flow_id'] if meta['births'] else None,'slot':slot,'total':total,'remaining':remaining,'delivered':total-remaining,'known':slot>=0 and bool(s['flow_known'][0,slot]),'finite':finite((s,tr.graphs[h-1],tr.latents[h-1])),'flow_hidden_fingerprint':fingerprint(tr.latents[h-1]['h']['flow'])})
                aligned.append({'h':h,'task_id':'Task_5','predicted_remaining':remaining,'real_remaining':rremaining,'absolute_remaining_error':abs(remaining-rremaining),'predicted_completed':bool(s['task_completed'][0,ti]),'real_completed':done,'predicted_failure':meta['tti_failure'],'real_failure_code':real['after']['failure_reason_code']})
            facts[c.candidate_id]={'initial':tr.initial_fingerprints,'inputs':tr.input_fingerprints,'outputs':tr.output_fingerprints,'mappings':tr.mappings,'meta':meta,'conservation':conservation,'alignment':aligned}
            pairs.append((c,tr,meta))
            atomic(OUT/f'{c.candidate_id}_wm.json',facts[c.candidate_id])
        result=DynamicScorerV2(ss,side,current,slot_duration_s=.1).score_set(pairs)
        atomic(OUT/'scorer_result.json',result)
        identity={'checkpoint':str(CHECKPOINT.relative_to(ROOT)),'checkpoint_sha':sha(CHECKPOINT),'expected_checkpoint_sha':EXPECTED_SHA,'normalization_sha':sha(NORM),'model_parameters_before':param_before,'model_parameters_after':fingerprint(model.state_dict()),'encoder_parameters_before':enc_before,'encoder_parameters_after':fingerprint(encoder.state_dict()),'future_poison_pass':poison_pass,'compact_preserves_owned_state_hidden':compact_preserves,'pre_route_anchor':fp,'transitions':8,'GPU':False,'locked_test':False,'formal_route_search':False}
        atomic(OUT/'wm_completion.json',identity)
    except BaseException as exc:
        atomic(OUT/'wm_failure.json',{'status':'BLOCKED','type':type(exc).__name__,'reason':str(exc)})
        raise
    finally:timer.cancel()

if __name__=='__main__':main()
