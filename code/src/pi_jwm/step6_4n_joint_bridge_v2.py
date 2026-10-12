"""Whole-decision opt-in validation and native transaction; no action policy."""
from contextlib import contextmanager
from types import MethodType
import copy

from .step6_4n_runtime_v2 import validate_joint_return,stage_aware_return_tti
from .step6_4b_live_bridge_v1 import validate_command
from .step6_0a_candidate_generation_v1 import CandidateActionStep

def validate_joint_commands_v2(env,step,context,decision,runtime_tasks,catalog):
    if getattr(env,'_pi_jwm_v2_quarantined',False):raise ValueError('V2_ENV_QUARANTINED')
    if any(k in decision for k in ('future_target','future_outcome','target_tensor')):raise ValueError('FUTURE_TARGET_FORBIDDEN')
    if int(decision['n_rb'])!=int(env.channel_manager.n_RB):raise ValueError('RB_RUNTIME_COUNT_MISMATCH')
    raw={str(r['task_id']):r for r in decision['tasks']}
    if len(raw)!=len(decision['tasks']):raise ValueError('DUPLICATE_CURRENT_TASK')
    indices=context.static['input_entity_index'];taskmap=indices['task'];physical=indices['physical'];s=context.current_state
    links={(str(r['source_id']),str(r['target_id'])) for r in decision['channel_rows'] if r.get('channel_type') in {'U2U','U2V','U2I','V2U','V2V','V2I','I2U','I2V','I2I'}}
    # Reuse existing exact Comp/CPU/Mob checks, after removing only the
    # v2-owned Route/Comm families. This validates the entire declaration
    # before touching any real buffer.
    base=validate_command(CandidateActionStep(comp=step.comp,mob=step.mob),context,decision,runtime_tasks)
    if any(float(r['speed_mps'])>15. for r in step.mob):raise ValueError('MOB_OUTSIDE_FROZEN_SPEED_BOUND')
    routes={};post={}
    for r in step.route:
        tid=str(r['task_id']);ti=taskmap.get(tid)
        if tid in routes or r.get('route_kind')!='return' or ti is None or type(r['task_index'])!=int or r['task_index']!=ti:raise ValueError('ROUTE_DUPLICATE_OR_UNSUPPORTED')
        if int(s['return_flow_index'][0,ti])>=0 or any(str(fid).startswith(f'flow::{tid}::Return::') for fid in indices['logical_flow']):raise ValueError('RETURN_ALREADY_KNOWN')
        if tid in env.task_return_routes:raise ValueError('RETURN_ROUTE_ALREADY_PENDING')
        declared=tuple(r['route_node_ids'])
        expected=tuple(physical[str(x)] for x in declared)
        if tuple(r['route_node_indices'])!=expected:raise ValueError('ROUTE_INDEX_MISMATCH')
        admitted=validate_joint_return(env,raw[tid],declared,(),links)
        src,dst=physical[admitted['source']],physical[admitted['destination']]
        import torch
        hits=torch.nonzero(s['comm_presence'][0]&s['comm_validity'][0]&s['comm_wireless_mask'][0]&(s['comm_source_index'][0]==src)&(s['comm_target_index'][0]==dst)).flatten()
        if len(hits)!=1:raise ValueError('POST_ROUTE_RELATION_NOT_UNIQUE')
        post[tid]=int(hits[0]);routes[tid]=list(declared)
    rb={}
    for r in step.comm:
        tid=str(r['task_id']);ti=taskmap.get(tid);ri=r['relation_index'];ids=list(r['rb_indices'])
        if tid in rb or tid not in raw or ti is None or type(r['task_index'])!=int or r['task_index']!=ti or type(ri)!=int or not 0<=ri<s['comm_presence'].shape[1]:raise ValueError('COMM_BINDING_INVALID')
        if tid in env.activated_offloading_tasks_with_RB_Nos:raise ValueError('COMM_ALREADY_BOUND')
        if not ids or any(type(x)!=int or not 0<=x<int(decision['n_rb']) for x in ids) or len(set(ids))!=len(ids):raise ValueError('COMM_RB_INVALID')
        if ids!=list(range(ids[0],ids[0]+len(ids))) or (ids[0],len(ids)) not in catalog.comm_start_width_pairs:raise ValueError('COMM_RB_OUTSIDE_TRAIN_PATTERN')
        if not bool(s['comm_presence'][0,ri]&s['comm_validity'][0,ri]&s['comm_wireless_mask'][0,ri]):raise ValueError('COMM_RELATION_INVALID')
        if tid in post:
            if ri!=post[tid]:raise ValueError('POST_ROUTE_COMM_RELATION_MISMATCH')
        else:
            from .step6_4f_comm_eligibility_v1 import tensor_comm_task_eligible
            life=str(raw[tid]['lifecycle'])
            if life not in ('offloading','transmitting','returning') or not tensor_comm_task_eligible(s,ti):raise ValueError('COMM_DONE_FAILED_STALE')
            if tid not in runtime_tasks:raise ValueError('COMM_RUNTIME_TASK_MISSING')
            task=runtime_tasks[tid]
            # Native getter extends _offloading_tasks lists with returning
            # tasks, so never call it from zero-mutation prevalidation.
            transmitting=[x for name in ('_offloading_tasks','_returning_tasks')
                for xs in getattr(env.task_manager,name,{}).values() for x in xs]
            if not any(task is x for x in transmitting):raise ValueError('COMM_RUNTIME_BUCKET_STALE')
            import torch
            slots=torch.nonzero(s['flow_known'][0]&s['flow_presence'][0]&s['carrying_active'][0]&(s['flow_task_index'][0]==ti)).flatten()
            if len(slots)!=1:raise ValueError('COMM_FLOW_NOT_UNIQUE')
            f=int(slots[0]);src=int(s['carrying_hop_source_index'][0,f]);dst=int(s['carrying_hop_destination_index'][0,f])
            if int(s['flow_comm_relation_index'][0,f])!=ri or int(s['comm_source_index'][0,ri])!=src or int(s['comm_target_index'][0,ri])!=dst:raise ValueError('COMM_ENDPOINT_INVALID')
            if physical.get(str(task.getCurrentNodeId()))!=src or not task.getToOffloadRoute() or physical.get(str(task.getToOffloadRoute()[0]))!=dst:raise ValueError('COMM_RUNTIME_ENDPOINT_STALE')
        rb[tid]=ids
    return {'mode':'DYNAMIC_RETURN_V2','routes':routes,'rb':rb,'cpu':base['cpu'],'mobility':base['mobility'],'post_route_relations':post,'joint_validation_before_any_setter':True,'RB_reuse':'allowed_native_interference_not_exclusive_pool'}

def apply_joint_commands_v2(env,commands,*,route_scheduler,communication,computation,traffic):
    if commands.get('mode')!='DYNAMIC_RETURN_V2' or commands.get('joint_validation_before_any_setter') is not True:raise ValueError('JOINT_VALIDATION_REQUIRED')
    if getattr(env,'_pi_jwm_v2_quarantined',False):raise ValueError('V2_ENV_QUARANTINED')
    try:
        # No buffer is changed until all four families have passed validation.
        for tid,path in sorted(commands['routes'].items()):route_scheduler.setTaskReturnRoute(env,tid,list(path))
        for tid,ids in sorted(commands['rb'].items()):communication.setCommunicationWithRB(env,tid,list(ids))
        callback=lambda *args,**kwargs:dict(commands['cpu'])
        computation.setComputingCallBack(env,callback)
        traffic.setUAVMobilityPatterns(env,copy.deepcopy(commands['mobility']))
        if any(env.task_return_routes.get(tid)!=path for tid,path in commands['routes'].items()) or any(env.activated_offloading_tasks_with_RB_Nos.get(tid)!=ids for tid,ids in commands['rb'].items()):raise ValueError('JOINT_SETTER_WRITEBACK_MISMATCH')
        if getattr(env,'alloc_cpu_callback',None) is not callback or getattr(env,'uav_mobility_patterns',None)!=commands['mobility']:raise ValueError('JOINT_COMP_MOB_WRITEBACK_MISMATCH')
    except Exception as exc:
        env._pi_jwm_v2_quarantined=True
        def stop(*args,**kwargs):raise RuntimeError('V2_ENV_QUARANTINED_NO_STEP_OR_RETRY')
        env.step=stop
        raise RuntimeError(f'JOINT_PARTIAL_SETTER_FAILURE:{exc}') from exc
    return {'all_setters_committed':True,'automatic_retry':False}

class DynamicCandidateDomainV2:
    """Explicit bounded declarations, never a policy or a formal search domain.

    Enumerate NOOP, Route without service and Route with each supplied TRAIN
    RB block. All use the same pre-Route root; rules only reject declarations.
    Comp/Mob declarations are supplied by the caller and validated jointly.
    Future birth, Input, multi-hop and reroute never enter this domain.
    """
    def __init__(self,env,context,decision,runtime_tasks,catalog):
        self.env,self.context,self.decision,self.tasks,self.catalog=env,context,decision,runtime_tasks,catalog

    def validate(self,step):
        return validate_joint_commands_v2(self.env,step,self.context,self.decision,self.tasks,self.catalog)

    def declared_return_controls(self,route_row,relation_index,rb_blocks,*,comp=(),mob=()):
        steps=[CandidateActionStep(comp=comp,mob=mob),CandidateActionStep(route=(route_row,),comp=comp,mob=mob)]
        for block in rb_blocks:
            steps.append(CandidateActionStep(route=(route_row,),comm=({'task_id':route_row['task_id'],
                'task_index':route_row['task_index'],'relation_index':relation_index,'rb_indices':list(block)},),comp=comp,mob=mob))
        # Fail closed on any unsupported supplied declaration, no silent pruning.
        for step in steps:self.validate(step)
        return tuple(steps)

@contextmanager
def stage_aware_input_return_tti_v2(manager):
    """Input keeps native stage start; only positive Input activity refreshes.

    New Input Route optimization is unsupported; this watches already current
    Input transmissions. Return first start and activity use the existing v2
    instance adapter. Restoration preserves original bound methods.
    """
    originals={};input_clocks={}
    def watch_inputs():
        for tasks in manager._offloading_tasks.values():
            for task in tasks:
                if id(task) in originals:continue
                old=task.transmit_to_Node;had='transmit_to_Node' in vars(task)
                originals[id(task)]=(task,old,had)
                input_clocks[str(task.getTaskId())]=float(task.getLastTransmissionTime())
                def transmit(self,node_id,trans_data,current_time,fast_return=False,_old=old):
                    before=float(self.getLastTransmissionTime());is_return=bool(self.isReturning())
                    result=_old(node_id,trans_data,current_time,fast_return=fast_return)
                    if not is_return:
                        if float(trans_data)<=0:self._last_transmission_time=before
                        input_clocks[str(self.getTaskId())]=float(self.getLastTransmissionTime())
                    return result
                task.transmit_to_Node=MethodType(transmit,task)
    try:
        with stage_aware_return_tti(manager) as evidence:
            native_check=manager.checkTasks
            def checked(self,time):watch_inputs();return native_check(time)
            watch_inputs();manager.checkTasks=MethodType(checked,manager)
            evidence['input_clocks']=input_clocks
            try:yield evidence
            finally:manager.checkTasks=native_check
    finally:
        for task,old,had in originals.values():
            if had:task.transmit_to_Node=old
            else:del task.transmit_to_Node
