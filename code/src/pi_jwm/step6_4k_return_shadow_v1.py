"""Isolated research prototype; NEVER invoked by the formal planner/model.

Birth requires a causal, validated Return route and exact current completion.
Logical identity and event updates reuse the existing Ledger. Tensor/latent
adaptation is explicitly a Shadow interface proposal, not a trained contract.
"""
from __future__ import annotations
import copy
import math
from .step4_2c_b_causal_flow_ledger_raw_v1 import CausalFlowLedger, parse_flow_id


def exact_completion(task):
    values=[float(task[k]) for k in ('task_cpu_work', 'computed_cpu_work')]
    if not all(math.isfinite(x) and x >= 0 for x in values): raise ValueError('INVALID_CPU_WORK')
    return values[1] >= values[0]


class ReturnShadow:
    def __init__(self, snapshot=None):
        self.ledger=CausalFlowLedger()
        self.birth_log=[]
        if snapshot is not None:
            flows=copy.deepcopy(snapshot['logical_flow_rows']); carry=copy.deepcopy(snapshot['carrying_rows'])
            if any(not {'flow_id','flow_index','task_id','flow_type','epoch','total_data','e2e_remaining','e2e_delivered'} <= r.keys() for r in flows):
                raise ValueError('FLOW_SCHEMA_MISSING')
            ids=[r['flow_id'] for r in flows]; slots=[int(r['flow_index']) for r in flows]
            if len(set(ids)) != len(ids) or len(set(slots)) != len(slots): raise ValueError('DUPLICATE_FLOW_IDENTITY')
            if sorted(slots) != list(range(len(slots))): raise ValueError('NONCONTIGUOUS_LEDGER_INDEX')
            if set(ids) != {r['flow_id'] for r in carry}: raise ValueError('CARRYING_IDENTITY_MISSING')
            for r in flows:
                if parse_flow_id(r['flow_id']) != (r['task_id'], r['flow_type'], int(r['epoch'])): raise ValueError('FLOW_IDENTITY_MISMATCH')
                CausalFlowLedger._assert_flow(r)
            index={r['flow_id']:r['flow_index'] for r in flows}
            if index != snapshot['flow_index']: raise ValueError('FLOW_INDEX_CONFLICT')
            # Snapshot restore is confined to Shadow; official Ledger unchanged.
            self.ledger._flows={r['flow_id']:r for r in flows}
            self.ledger._carrying={r['flow_id']:r for r in carry}
            self.ledger._flow_index=index

    def snapshot(self): return self.ledger.snapshot()

    def establish(self, task, route, *, now, source_time, entities, links):
        if not all(math.isfinite(float(x)) for x in (now,source_time)) or source_time>now: raise ValueError('FUTURE_ROUTE_SOURCE')
        if task['lifecycle'] not in ('waiting_to_return','returning') or not exact_completion(task): raise ValueError('TASK_NOT_LEGALLY_WAITING_RETURN')
        src,dst=str(task['current_node_id']),str(task['return_destination_id'])
        size=float(task['required_returned_size'])
        if not math.isfinite(size) or size<0: raise ValueError('INVALID_RETURN_SIZE')
        if src not in entities or dst not in entities: raise ValueError('RETURN_ENDPOINT_MISSING')
        if size==0 or src==dst: return None
        route=list(map(str,route))
        if not route or route[-1]!=dst or len(set(route))!=len(route) or src in route: raise ValueError('INVALID_RETURN_ROUTE')
        if any(x not in entities for x in route): raise ValueError('ROUTE_NODE_MISSING')
        if any(pair not in links for pair in zip([src]+route[:-1],route)): raise ValueError('ROUTE_LINK_UNAVAILABLE')
        known=[r for r in self.ledger.all_flow_rows() if r['task_id']==str(task['task_id']) and r['flow_type']=='Return']
        if known:
            r=max(known,key=lambda r:r['epoch']); fid=r['flow_id']
            if r['logical_destination']!=dst: raise ValueError('DESTINATION_CHANGE_REQUIRES_NEW_EPOCH_PROTOCOL')
            if r['status']=='COMPLETED': return fid # no resurrection
            c=self.ledger.carrying(fid)
            if c['current_holder']!=src: raise ValueError('RETURN_HOLDER_MISMATCH')
            if route!=c['route'][c['current_hop_index']:]:
                if c['hop_progress']>0: raise ValueError('REROUTE_NOT_AT_CLEAN_BOUNDARY')
                self.ledger.reroute(fid,new_route=route,new_logical_destination=dst)
            return fid
        fid=self.ledger.create_return_flow(task,logical_destination=dst,total_data=size,route=route,
            logical_destination_source='shadow_current_task_return_destination',
            logical_destination_capture_phase='SHADOW_ROUTE_EXPERIMENT_after_validated_route_before_service')
        self.birth_log.append({'flow_id':fid,'task_id':task['task_id'],'now':now,'source_time':source_time,'route':route,'label':'SHADOW_ROUTE_EXPERIMENT'})
        return fid

    def transfer(self, fid, *, source, target, amount, hop_completed):
        f=self.ledger.flow(fid); c=self.ledger.carrying(fid)
        amount=float(amount)
        if f['status']!='ACTIVE' or source!=c['hop_source'] or target!=c['hop_destination']: raise ValueError('TRANSFER_IDENTITY_MISMATCH')
        if not math.isfinite(amount) or amount<0 or amount>c['hop_remaining']: raise ValueError('TRANSFER_CONSERVATION_FAILED')
        if hop_completed != (amount==c['hop_remaining']): raise ValueError('HOP_COMPLETION_INCONSISTENT')
        event={'task_id':f['task_id'],'phase':'return','source_id':source,'target_id':target,
               'delivered_data':amount,'stage_or_hop_completed':hop_completed}
        return self.ledger.apply_transfer_event(event,observed_task_current_node_id=target if hop_completed else source)


def inject_return_slot(model, state, graph, latent, *, task_index, source, destination,
                       route, size, flow_id, max_flow_slots=4096):
    """Append support without overwriting slots. Cold zero h is a proposal.

    Physical/Comm support is NOT expanded: all causal route edges must exist.
    The graph is rebuilt with frozen weights; new Flow type is synchronized.
    This introduces a distribution change even if recursion is executable.
    """
    import torch
    route=list(route)
    from .step6_1_trained_candidate_rollout_v1 import clone_tree
    from .step4_2c_c_flow_sample_tensor_v1 import FLOW_TYPE_VOCAB,FLOW_STATUS_VOCAB
    s,g,l=clone_tree(state),clone_tree(graph),clone_tree(latent)
    if s['flow_presence'].shape[0]!=1: raise ValueError('SHADOW_SINGLE_ROOT_ONLY')
    if int(s['return_flow_index'][0,task_index])>=0: raise ValueError('RETURN_SLOT_ALREADY_KNOWN')
    slot=s['flow_presence'].shape[1]
    if slot>=max_flow_slots: raise ValueError('SHADOW_FLOW_CAPACITY_EXHAUSTED')
    if not route or route[-1]!=destination or len(set(route))!=len(route) or source in route: raise ValueError('INVALID_RETURN_ROUTE')
    if size<=0 or not math.isfinite(size): raise ValueError('INVALID_RETURN_SIZE')
    relation=-1
    for i,(src,dst) in enumerate(zip([source]+route[:-1],route)):
        hits=torch.nonzero(s['comm_presence'][0]&s['comm_validity'][0]&(s['comm_source_index'][0]==src)&(s['comm_target_index'][0]==dst)).flatten()
        if not len(hits): raise ValueError('SHADOW_ROUTE_COMM_SUPPORT_MISSING')
        if i==0: relation=int(hits[0])
    keys=['flow_presence','flow_known','flow_task_index','flow_total','flow_remaining',
          'flow_source_index','flow_destination_index','flow_type_index','flow_status_index',
          'flow_identity_index','flow_route_revision','current_holder_index','current_hop_index',
          'hop_progress','hop_remaining','route_node_indices','route_node_mask','carrying_active',
          'carrying_hop_source_index','carrying_hop_destination_index','flow_comm_relation_index']
    width=max(s['route_node_indices'].shape[-1],len(route))
    for key in ('route_node_indices','route_node_mask'):
        if width>s[key].shape[-1]:
            extra=torch.full((*s[key].shape[:-1],width-s[key].shape[-1]),False if key.endswith('mask') else -1,dtype=s[key].dtype)
            s[key]=torch.cat((s[key],extra),dim=-1)
    for key in keys:
        new=torch.zeros((1,1,*s[key].shape[2:]),dtype=s[key].dtype,device=s[key].device)
        s[key]=torch.cat((s[key],new),dim=1)
    values={'flow_presence':True,'flow_known':True,'flow_task_index':task_index,
            'flow_total':size,'flow_remaining':size,'flow_source_index':source,'flow_destination_index':destination,
            'flow_type_index':FLOW_TYPE_VOCAB.index('Return'),'flow_status_index':FLOW_STATUS_VOCAB.index('ACTIVE'),
            'flow_identity_index':slot,'flow_route_revision':0,'current_holder_index':source,'current_hop_index':0,
            'hop_progress':0.,'hop_remaining':size,'carrying_active':True,'carrying_hop_source_index':source,
            'carrying_hop_destination_index':route[0],'flow_comm_relation_index':relation}
    for k,v in values.items(): s[k][0,slot]=v
    s['route_node_indices'][0,slot].fill_(-1);s['route_node_indices'][0,slot,:len(route)]=torch.tensor(route)
    s['route_node_mask'][0,slot].fill_(False);s['route_node_mask'][0,slot,:len(route)]=True
    s['return_flow_index'][0,task_index]=slot
    s['task_requires_return'][0,task_index]=True;s['task_return_requirement_known'][0,task_index]=True
    l['h']['flow']=torch.cat((l['h']['flow'],torch.zeros((1,1,model.config.d_h),dtype=l['h']['flow'].dtype)),dim=1)
    g['flow_type_index']=s['flow_type_index']
    g=model.rebuild_graph(s,g)
    return s,g,l,{'slot':slot,'flow_id':flow_id,'latent_initialization':'ZERO_COLD_START_SHADOW_PROPOSAL','shape_adaptation':'APPEND_NO_OVERWRITE','route':route}
