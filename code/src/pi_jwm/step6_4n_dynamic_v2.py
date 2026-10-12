"""Explicitly isolated Dynamic Flow/Scorer v2 interfaces.

These interfaces append a validated Return slot to a cloned frozen state and
score supplied predicted states against one common anchor. They never choose
an action and never mutate the formal v1 planner or checkpoint.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .step6_4k_return_shadow_v1 import inject_return_slot
from .step6_1_trained_candidate_rollout_v1 import fingerprint, clone_tree

def compact_anchor_support(model, state, graph, latent):
    """Remove only trailing unowned padding, before branching the common root.

    Logical slot indices and every owned object value remain unchanged.
    This avoids assigning Ledger index 1 to tensor index 32 merely because
    the historical batch had 31 padding rows.
    """
    import torch
    s,g,l=clone_tree(state),clone_tree(graph),clone_tree(latent)
    known=torch.nonzero(s['flow_known'][0]).flatten()
    n=int(known[-1])+1 if len(known) else 0
    if list(map(int,known))!=list(range(n)):
        raise ValueError('LEDGER_SUPPORT_NOT_CONTIGUOUS')
    for k in FLOW_KEYS:
        s[k]=s[k][:,:n].clone()
    l['h']['flow']=l['h']['flow'][:,:n].clone()
    g['flow_type_index']=s['flow_type_index']
    g=model.rebuild_graph(s,g)
    return s,g,l

FLOW_KEYS=('flow_presence','flow_known','flow_task_index','flow_total','flow_remaining',
    'flow_source_index','flow_destination_index','flow_type_index','flow_status_index',
    'flow_identity_index','flow_route_revision','current_holder_index','current_hop_index',
    'hop_progress','hop_remaining','route_node_indices','route_node_mask','carrying_active',
    'carrying_hop_source_index','carrying_hop_destination_index','flow_comm_relation_index')

SCHEMA = 'PI-JWM-STEP-6.4N-DYNAMIC-V2'

@dataclass(frozen=True)
class DynamicBirthResult:
    flow_id: str
    slot: int
    state: Mapping[str, Any]
    graph: Mapping[str, Any]
    latent: Mapping[str, Any]
    metadata: Mapping[str, Any]
    anchor_fingerprint: str

def append_validated_return(model, state, graph, latent, *, task_index: int,
                            source_index: int, destination_index: int,
                            route: Sequence[int], size: float, flow_id: str,
                            max_flow_slots: int = 4096) -> DynamicBirthResult:
    if not isinstance(flow_id, str) or not flow_id.startswith('flow::'):
        raise ValueError('DYNAMIC_FLOW_ID_INVALID')
    if any(not isinstance(x, int) or x < 0 for x in (task_index, source_index, destination_index)):
        raise ValueError('DYNAMIC_FLOW_INDEX_INVALID')
    if not route or any(not isinstance(x, int) or x < 0 for x in route):
        raise ValueError('DYNAMIC_FLOW_ROUTE_INVALID')
    if len(route)!=1 or task_index>=state['task_presence'].shape[1] or not bool(state['task_presence'][0,task_index]):
        raise ValueError('DYNAMIC_RETURN_SUPPORT_INVALID')
    if max(source_index,destination_index)>=state['entity_presence'].shape[1] or not all(bool(state['entity_presence'][0,x]) for x in (source_index,destination_index)):
        raise ValueError('DYNAMIC_FLOW_ENDPOINT_INVALID')
    from .step4_2c_b_causal_flow_ledger_raw_v1 import parse_flow_id
    if parse_flow_id(flow_id)[1:]!=('Return',0):
        raise ValueError('DYNAMIC_FLOW_EPOCH_UNSUPPORTED')
    before = fingerprint((state, graph, latent))
    next_state, next_graph, next_latent, meta = inject_return_slot(
        model, state, graph, latent, task_index=task_index, source=source_index,
        destination=destination_index, route=route, size=size, flow_id=flow_id,
        max_flow_slots=max_flow_slots)
    if int(meta['slot']) != int(state['flow_presence'].shape[1]):
        raise ValueError('DYNAMIC_FLOW_SLOT_NOT_APPENDED')
    if fingerprint(state) == fingerprint(next_state):
        raise ValueError('DYNAMIC_FLOW_STATE_NOT_UPDATED')
    meta = {**meta, 'schema': SCHEMA, 'birth_timing': 'before_current_step_one_step',
            'future_truth_used': False, 'anchor_fingerprint': before}
    return DynamicBirthResult(flow_id, int(meta['slot']), next_state, next_graph,
                              next_latent, meta, before)

class DynamicScorerV2:
    """Five original objectives; v2 unmet-known-Return burden only.

    Constructor consumes exact current Raw for demand eligibility. No future
    outcome or tensor-rounded computation state can establish new demand.
    TTI side rows must come from the causal v2 transition, never real targets.
    """
    def __init__(self, anchor_state, side, current_raw, *, slot_duration_s):
        from .step6_2b_planner_objective_scorer_v1 import _anchor_flows, _tx_burden, _flag, _scalar
        from .step6_2a_planner_objective_side_state_v1 import FORBIDDEN
        if FORBIDDEN.intersection(current_raw): raise ValueError('FUTURE_TARGET_FORBIDDEN')
        if not math.isfinite(slot_duration_s) or slot_duration_s<=0: raise ValueError('INVALID_DT')
        self.anchor=clone_tree(anchor_state);self.side=side;self.dt=slot_duration_s
        self.anchor_fp=fingerprint(anchor_state)
        raw={str(x['task_id']):x for x in current_raw['tasks']}
        if len(raw)!=len(current_raw['tasks']):raise ValueError('DUPLICATE_TASK')
        self.cohort=tuple(t for t in side.tasks if not t.terminal_at_anchor and _flag(anchor_state,'task_presence',t.task_index) and not _flag(anchor_state,'task_completed',t.task_index))
        self.components={};self.known={}
        for t in self.cohort:
            r=raw[t.task_id]; flows=_anchor_flows(anchor_state,t.task_index)
            work=_scalar(anchor_state,'task_work_remaining',t.task_index)
            slot=int(anchor_state['return_flow_index'][0,t.task_index])
            confirmed=(str(r['lifecycle'])=='waiting_to_return' and float(r['computed_cpu_work'])>=float(r['task_cpu_work']) and float(r['required_returned_size'])>0 and str(r['current_node_id'])!=str(r['return_destination_id']) and slot<0)
            demand=float(r['required_returned_size']) if confirmed else 0.
            if confirmed:
                from .step4_2c_c_flow_sample_tensor_v1 import FLOW_TYPE_VOCAB
                if any(int(anchor_state['flow_type_index'][0,f])==FLOW_TYPE_VOCAB.index('Return') for f in flows):raise ValueError('DOUBLE_RETURN_BURDEN')
                self.known[t.task_id]=demand
            tx0=_tx_burden(anchor_state,flows)+demand
            self.components[t.task_id]=(flows,tx0,work,demand)

    def score_set(self, pairs):
        from .step6_2b_planner_objective_scorer_v1 import (_tx_burden, _single_hop_flow_burden, _scalar, _flag, _effort, _compute_host)
        from .step4_2c_c_flow_sample_tensor_v1 import FLOW_TYPE_VOCAB
        if not pairs:return {'status':'OBJECTIVE_UNSCOREABLE_EMPTY_SET','H_eff':0,'scores':[],'rejections':{}}
        horizon=pairs[0][0].horizon
        if any(c.horizon!=horizon or len(tr.states)!=horizon for c,tr,meta in pairs):raise ValueError('COMMON_HORIZON_REQUIRED')
        if len({c.fingerprint for c,tr,m in pairs})!=len(pairs):raise ValueError('DUPLICATE_CANDIDATE')
        support={};reject={}
        for c,tr,meta in pairs:
            if meta['pre_route_state_fingerprint']!=self.anchor_fp or c.candidate_id!=tr.candidate_id:raise ValueError('COMMON_ANCHOR_REQUIRED')
            if meta.get('source')!='V2_CAUSAL_PREDICTED_TRANSITION':raise ValueError('CAUSAL_TTI_EVIDENCE_REQUIRED')
            support[c.candidate_id]=horizon
            for h,s in enumerate(tr.states,1):
                for t in self.cohort:
                    slot=int(s['return_flow_index'][0,t.task_index])
                    finished=_scalar(s,'task_work_remaining',t.task_index)<=0
                    if finished and slot<0 and t.required_returned_size>0 and t.task_id not in self.known:
                        host=_compute_host(s,t.task_index)
                        if host!=t.return_destination_index:
                            support[c.candidate_id]=min(support[c.candidate_id],h-1)
            if meta.get('tti_failure'):
                reject[c.candidate_id]={'status':'HARD_REJECT_TTI','reason':meta['tti_failure'],'terminal':'TTI_FAILED'}
        H=min(support.values())
        if not self.cohort or H==0:return {'status':'OBJECTIVE_UNSCOREABLE','H_eff':H,'scores':[],'support':support,'rejections':reject}
        scores=[]
        for c,tr,meta in pairs:
            if c.candidate_id in reject:continue
            statuses={t.task_id:'ACTIVE' for t in self.cohort};failed=set();rows=[];previous={}
            for h,s in enumerate(tr.states[:H],1):
                now=self.side.current_time_s+h*self.dt
                task_rows=[]
                for t in self.cohort:
                    flows,tx0,work0,demand=self.components[t.task_id];ti=t.task_index
                    tx=_tx_burden(s,flows);unmet=0.
                    if demand:
                        slot=int(s['return_flow_index'][0,ti])
                        unmet=demand
                        if slot>=0:
                            if slot in flows or slot>=s['flow_known'].shape[1] or not _flag(s,'flow_known',slot) or int(s['flow_task_index'][0,slot])!=ti or int(s['flow_type_index'][0,slot])!=FLOW_TYPE_VOCAB.index('Return'):raise ValueError('DYNAMIC_RETURN_IDENTITY_INVALID')
                            if not math.isclose(_scalar(s,'flow_total',slot),demand,rel_tol=1e-6,abs_tol=1e-8):raise ValueError('DYNAMIC_RETURN_TOTAL_MISMATCH')
                            # The neural interface stores size in FP32. Keep
                            # demand in common Raw FP64 units and transfer the
                            # predicted fractional progress, so birth itself
                            # cannot gain/lose burden from the dtype cast.
                            unmet=demand*(_single_hop_flow_burden(s,slot)/_scalar(s,'flow_total',slot))
                            before=previous.get(t.task_id,demand)
                            if unmet>before+1e-8 or unmet<0:raise ValueError('DYNAMIC_RETURN_CONSERVATION_INVALID')
                            progress=before-unmet
                            if progress>0:
                                event=next((e for e in meta.get('events',()) if e['task_id']==t.task_id and e['horizon']==h),None)
                                if event is None or not math.isclose(float(event['useful_progress']),progress,rel_tol=1e-5,abs_tol=1e-8):raise ValueError('DYNAMIC_RETURN_PROGRESS_EVIDENCE_MISSING')
                                relation=int(event['relation_index_before_service'])
                                if not any(str(r['task_id'])==t.task_id and int(r['relation_index'])==relation and r['rb_indices'] for r in c.steps[h-1].comm):raise ValueError('RETURN_PROGRESS_WITHOUT_COMM_SERVICE')
                            previous[t.task_id]=unmet
                        tx+=unmet
                    work=_scalar(s,'task_work_remaining',ti)
                    if work<0:raise ValueError('NEGATIVE_WORK')
                    completed=_flag(s,'task_completed',ti)
                    if demand and unmet>0 and completed:raise ValueError('COMPLETED_WITH_UNMET_RETURN')
                    no_return=int(s['return_flow_index'][0,ti])<0 and (t.required_returned_size==0 or _compute_host(s,ti)==t.return_destination_index)
                    if statuses[t.task_id]=='ACTIVE':
                        if completed:statuses[t.task_id]='SUCCESSFULLY_COMPLETED' if now<=t.absolute_deadline_s+(1e-5 if no_return else 0.) else 'DEADLINE_FAILED'
                        elif now>t.absolute_deadline_s:statuses[t.task_id]='DEADLINE_FAILED'
                    status=statuses[t.task_id];ddl=status=='DEADLINE_FAILED'
                    if ddl:failed.add(t.task_id)
                    mtx,mcomp=tx0>0,work0>0
                    if status=='SUCCESSFULLY_COMPLETED':burden=0.
                    elif not(mtx or mcomp):raise ValueError('BURDEN_SEMANTICS_BLOCKED')
                    else:burden=((tx/tx0 if mtx else 0.)+(work/work0 if mcomp else 0.))/(int(mtx)+int(mcomp))
                    task_rows.append({'task_id':t.task_id,'task_index':ti,'status':status,'unfinished':int(status!='SUCCESSFULLY_COMPLETED'),'deadline_violation':int(ddl),'anchor_tx_burden':tx0,'tx_burden':tx,'anchor_work_remaining':work0,'work_remaining':work,'known_return_demand':demand,'unmet_return':unmet,'burden':burden})
                efforts=_effort(c,self.side,h,self.anchor)
                if not efforts:raise ValueError('NO_EFFORT_COMPONENT')
                rows.append({'horizon':h,'tasks':task_rows,'effort_components':efforts,'effort':sum(efforts.values())/len(efforts)})
            denom=len(self.cohort)*H
            obj=(len(failed),sum(t['deadline_violation'] for r in rows for t in r['tasks'])/denom,sum(t['unfinished'] for r in rows for t in r['tasks'])/denom,sum(t['burden'] for r in rows for t in r['tasks'])/denom,sum(r['effort'] for r in rows)/H)
            if not all(math.isfinite(x) for x in obj):raise ValueError('NONFINITE_OBJECTIVE')
            scores.append({'candidate_id':c.candidate_id,'fingerprint':c.fingerprint,'objective':obj,'rows':rows,'H_eff':H,'status':'SCOREABLE'})
        scores.sort(key=lambda r:(*r['objective'],r['fingerprint']))
        return {'status':'SCOREABLE' if scores else 'OBJECTIVE_UNSCOREABLE_ALL_REJECTED','H_eff':H,'scores':scores,'support':support,'rejections':reject,'cohort':[t.task_id for t in self.cohort],'effort_denominators':self.side.effort_denominators,'semantic_change':'OBJECTIVE_V2_SEMANTIC_CHANGE','future_truth_used':False}
