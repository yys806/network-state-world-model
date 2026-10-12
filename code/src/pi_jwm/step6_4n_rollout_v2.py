"""Action-triggered Return birth before frozen one_step; CPU opt-in only."""
from dataclasses import replace
import math
import torch
from .step6_1_trained_candidate_rollout_v1 import clone_tree,fingerprint,CandidateRolloutTrace
from .step6_0a_candidate_generation_v1 import CandidateActionStep,validate_step
from .step6_4n_dynamic_v2 import append_validated_return
from .step4_2c_b_causal_flow_ledger_raw_v1 import CausalFlowLedger,parse_flow_id
from .step3_3_model_input_tensor_v1 import LIFECYCLE_VOCAB

def restore_ledger(snapshot):
    ledger=CausalFlowLedger()
    flows=snapshot['logical_flow_rows'];carrying=snapshot['carrying_rows']
    ids=[r['flow_id'] for r in flows]
    if len(set(ids))!=len(ids):raise ValueError('DUPLICATE_FLOW_IDENTITY')
    if sorted(int(r['flow_index']) for r in flows)!=list(range(len(flows))):raise ValueError('LEDGER_SLOT_INVALID')
    if set(ids)!={r['flow_id'] for r in carrying}:raise ValueError('LEDGER_CARRYING_MISSING')
    for r in flows:
        if parse_flow_id(r['flow_id'])!=(r['task_id'],r['flow_type'],int(r['epoch'])):raise ValueError('FLOW_IDENTITY_MISMATCH')
        ledger._assert_flow(r)
    ledger._flows={r['flow_id']:clone_tree(r) for r in flows}
    ledger._carrying={r['flow_id']:clone_tree(r) for r in carrying}
    ledger._flow_index={r['flow_id']:int(r['flow_index']) for r in flows}
    return ledger

def advance_stage_clock(*, clock, tick_time, useful_progress, unfinished,
                        next_check_time, threshold):
    if not all(math.isfinite(float(x)) for x in (clock,tick_time,useful_progress,next_check_time,threshold)) or useful_progress<0 or threshold<0:
        raise ValueError('TTI_CLOCK_INVALID')
    if unfinished and tick_time-clock>threshold:
        return clock,'TASK_FAIL_OUT_OF_TTI'
    updated=tick_time if useful_progress>0 else clock
    return updated, 'TASK_FAIL_OUT_OF_TTI' if unfinished and next_check_time-updated>threshold else None

def rollout_dynamic_v2(model,anchor,candidate,*,current_raw,ledger_snapshot,catalog,
                       slot_duration_s=.1,tti_threshold=.5,max_flow_slots=4096):
    if model.training or torch.is_grad_enabled():raise ValueError('FROZEN_NO_GRAD_REQUIRED')
    if any(k in current_raw for k in ('future_target','future_outcome','target_tensor')):raise ValueError('FUTURE_TARGET_FORBIDDEN')
    if not 0<slot_duration_s<math.inf or not 0<=tti_threshold<math.inf:raise ValueError('CLOCK_INVALID')
    from build_step5_1d_unified_model_chain_v1 import build_action
    s,g,l=clone_tree(anchor.state),clone_tree(anchor.graph),clone_tree(anchor.latent)
    static=clone_tree(anchor.context.static);ledger=restore_ledger(ledger_snapshot)
    taskmap=static['input_entity_index']['task'];entities=static['input_entity_index']['physical']
    flowmap=static['input_entity_index']['logical_flow']
    if ledger._flow_index!=flowmap:raise ValueError('LEDGER_TENSOR_MAP_MISMATCH')
    raw={r['task_id']:r for r in current_raw['tasks']}
    inputs=[];outputs=[];actions=[];mappings=[];latents=[];states=[];graphs=[];traces=[];births=[];clocks={};failure=None;events=[]
    now=float(current_raw['simulation_time_s'])
    pre=fingerprint(s)
    for h,step in enumerate(candidate.steps):
        if failure:break  # explicit terminal rejection, no retry or neural step
        incoming={'state':fingerprint(s),'graph':fingerprint(g),'latent':fingerprint(l)}
        route_effect=[]
        for row in step.route:
            if h!=0 or row.get('route_kind')!='return':raise ValueError('FUTURE_INPUT_REROUTE_UNSUPPORTED')
            tid=str(row['task_id']);r=raw[tid]
            if r['lifecycle']!='waiting_to_return' or float(r['computed_cpu_work'])<float(r['task_cpu_work']):raise ValueError('RETURN_NOT_EXACT_ANCHOR_WAITING')
            if any(f['task_id']==tid and f['flow_type']=='Return' for f in ledger.all_flow_rows()):raise ValueError('DUPLICATE_RETURN_BIRTH')
            src,dst=str(r['current_node_id']),str(r['return_destination_id'])
            route=list(row['route_node_ids']);size=float(r['required_returned_size'])
            if route!=[dst] or src==dst or size<=0 or not math.isfinite(size):raise ValueError('RETURN_NOT_REQUIRED_OR_SINGLE_HOP')
            ti=int(taskmap[tid]);si,di=int(entities[src]),int(entities[dst])
            if int(row['task_index'])!=ti or tuple(row['route_node_indices'])!=(di,):raise ValueError('ROUTE_INDEX_MISMATCH')
            hits=torch.nonzero(s['comm_presence'][0]&s['comm_validity'][0]&s['comm_wireless_mask'][0]&(s['comm_source_index'][0]==si)&(s['comm_target_index'][0]==di)).flatten()
            if len(hits)!=1:raise ValueError('RETURN_WIRELESS_BINDING_NOT_UNIQUE')
            fid=ledger.create_return_flow(r,logical_destination=dst,total_data=size,route=route,logical_destination_source='v2_current_task_return_destination',logical_destination_capture_phase='v2_candidate_route_before_learned_dynamics')
            birth=append_validated_return(model,s,g,l,task_index=ti,source_index=si,destination_index=di,route=(di,),size=size,flow_id=fid,max_flow_slots=max_flow_slots)
            if ledger.flow(fid)['flow_index']!=birth.slot:raise ValueError('LEDGER_TENSOR_SLOT_MISMATCH')
            s,g,l=birth.state,birth.graph,birth.latent
            s['task_lifecycle_index'][0,ti]=LIFECYCLE_VOCAB.index('transmitting')
            g=model.rebuild_graph(s,g)
            flowmap[fid]=birth.slot;clocks[tid]=now+h*slot_duration_s
            route_effect.append({**birth.metadata,'flow_id':fid,'slot':birth.slot,'epoch':0,'task_id':tid,'relation_index':int(hits[0]),'source_index':si,'destination_index':di,'official_ledger':ledger.flow(fid)})
            births.extend(route_effect[-1:])
        for tid,clock in clocks.items():
            ti=int(taskmap[tid]);slot=int(s['return_flow_index'][0,ti])
            if bool(s['flow_presence'][0,slot]) and now+h*slot_duration_s-clock>tti_threshold:
                failure={'task_id':tid,'horizon':h+1,'reason':'TASK_FAIL_OUT_OF_TTI','clock':clock,'check_time':now+h*slot_duration_s}
                break
        if failure:break
        seen=set()
        for row in step.comm:
            tid=str(row['task_id']);ti=int(row['task_index']);ri=int(row['relation_index'])
            if tid in seen or taskmap.get(tid)!=ti:raise ValueError('COMM_TASK_NOT_UNIQUE')
            seen.add(tid)
            lifecycle=LIFECYCLE_VOCAB[int(s['task_lifecycle_index'][0,ti])]
            if not bool(s['task_presence'][0,ti]) or bool(s['task_completed'][0,ti]) or lifecycle not in ('offloading','transmitting'):raise ValueError('COMM_FAILED_DONE_STALE')
            slots=torch.nonzero(s['flow_known'][0]&s['flow_presence'][0]&s['carrying_active'][0]&(s['flow_task_index'][0]==ti)).flatten()
            if len(slots)!=1 or ri<0 or ri>=s['comm_presence'].shape[1] or int(s['flow_comm_relation_index'][0,int(slots[0])])!=ri:raise ValueError('COMM_FLOW_BINDING_INVALID')
            if not bool(s['comm_presence'][0,ri]&s['comm_validity'][0,ri]&s['comm_wireless_mask'][0,ri]):raise ValueError('COMM_RELATION_INVALID')
            if int(s['carrying_hop_source_index'][0,int(slots[0])])!=int(s['comm_source_index'][0,ri]) or int(s['carrying_hop_destination_index'][0,int(slots[0])])!=int(s['comm_target_index'][0,ri]):raise ValueError('COMM_ENDPOINT_MISMATCH')
            rb=tuple(row['rb_indices'])
            if any(type(x)!=int for x in rb) or not rb or len(set(rb))!=len(rb) or rb!=tuple(range(rb[0],rb[0]+len(rb))) or (rb[0],len(rb)) not in catalog.comm_start_width_pairs:raise ValueError('COMM_RB_TRAIN_SUPPORT_INVALID')
        consumed=CandidateActionStep(route=(),comm=step.comm,comp=step.comp,mob=step.mob)
        context=replace(anchor.context,static=static,current_state=s)
        validate_step(consumed,context)
        # Current allocation belongs to this decision; never carry stale RBs.
        s['rb_active_mask'].zero_()
        sample={'static':static,'history':clone_tree(list(context.history)),'future_action':[consumed.frame()]}
        action,mapping=build_action(sample,s,0)
        if any(int(m['relation_index'])!=int(r['relation_index']) for m,r in zip(mapping['comm'],step.comm)):raise ValueError('COMM_ADAPTER_MAPPING_MISMATCH')
        before_remaining={tid:float(s['flow_remaining'][0,int(s['return_flow_index'][0,int(taskmap[tid])])]) for tid in clocks}
        service_relations={tid:int(s['flow_comm_relation_index'][0,int(s['return_flow_index'][0,int(taskmap[tid])])]) for tid in clocks}
        inputs.append(incoming)
        learned_input={'state':fingerprint(s),'graph':fingerprint(g),'latent':fingerprint(l)}
        l,s,g,trace=model.one_step(l,s,g,action,prior_mode='mean',service_mode='expectation',generator=None,route_rule_metadata=[])
        for tid,before in before_remaining.items():
            ti=int(taskmap[tid]);slot=int(s['return_flow_index'][0,ti]);after=float(s['flow_remaining'][0,slot])
            if after>before+1e-8 or after<0:raise ValueError('FLOW_CONSERVATION_FAILED')
            progress=before-after
            clocks[tid],reason=advance_stage_clock(clock=clocks[tid],tick_time=now+h*slot_duration_s,useful_progress=progress,unfinished=bool(s['flow_presence'][0,slot]),next_check_time=now+(h+1)*slot_duration_s,threshold=tti_threshold)
            if reason:failure={'task_id':tid,'horizon':h+1,'reason':reason,'clock':clocks[tid],'check_time':now+(h+1)*slot_duration_s}
            events.append({'task_id':tid,'horizon':h+1,'before':before,'after':after,'useful_progress':progress,'clock':clocks[tid],'terminal':reason,'relation_index_before_service':service_relations[tid]})
        outputs.append({'state':fingerprint(s),'graph':fingerprint(g),'latent':fingerprint(l)})
        mapping['route_effect']=route_effect;mapping['incoming']=incoming;mapping['learned_input']=learned_input
        actions.append(action);mappings.append(mapping);latents.append(clone_tree(l));states.append(clone_tree(s));graphs.append(clone_tree(g));traces.append(trace)
    tr=CandidateRolloutTrace(candidate.candidate_id,dict(anchor.fingerprints),tuple(inputs),tuple(outputs),tuple(actions),tuple(mappings),tuple(latents),tuple(states),tuple(graphs),tuple(traces))
    meta={'source':'V2_CAUSAL_PREDICTED_TRANSITION','pre_route_state_fingerprint':pre,'births':births,'tti_failure':failure,'events':events,'flow_id_to_slot':flowmap,'future_truth_used':False,'initialization':'ZERO_COLD_START','formal_pool_admitted':False}
    return tr,meta
