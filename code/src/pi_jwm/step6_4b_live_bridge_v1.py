"""History-only inference and fail-closed simulator commands. No future builder."""
from __future__ import annotations
import copy
import math
from typing import Mapping
from .model_ready_sample_contract_v1 import (
    TensorContract, ACTION_FAMILIES, _observation_entities, _observation_tasks,
    _action_tensor, _outcome_rows, _flow_id,
)
from .step4_2a_graph_input_extension_v1 import (
    amend_raw_graph_inputs, _communication_rows, _task_agent_rows, _wrapped, _vector_wrapped,
)
from .step4_2c_b_causal_flow_ledger_raw_v1 import amend_raw_with_causal_flow_ledger
from .step4_2c_c_flow_sample_tensor_v1 import SAMPLE_SCHEMA_VERSION, _sample_rows, _index_by_raw_flow

class SmokeFailure(ValueError):
    def __init__(self, reason, detail=''):
        self.reason=reason
        super().__init__(reason+(':'+str(detail) if detail else ''))

def require_nonempty_domain(domain):
    if domain.exact_unique_single_step_count <= 0:
        raise SmokeFailure('DOMAIN_EMPTY')

DECISION_FIELDS=('trajectory_id','frame_index','simulation_time_s','capture_event_id','capture_phase',
    'capture_method','entities','tasks','channel_rows','dag_edges','n_rb',
    'node_cpu_capacity_observation_rows','node_cpu_capacity_per_s')

def causal_prefix(raw):
    ds=raw.get('decisions',[])
    if len(ds)<2:raise SmokeFailure('REQUIRED_LIVE_OBSERVATION_MISSING','two real observations required')
    anchor=int(ds[-1]['frame_index'])
    decisions=[]
    for d in ds:
        for key in ('entities','tasks','channel_rows','dag_edges','node_cpu_capacity_observation_rows','n_rb'):
            if key not in d:raise SmokeFailure('REQUIRED_LIVE_OBSERVATION_MISSING',key)
        decisions.append({k:copy.deepcopy(d[k]) for k in DECISION_FIELDS if k in d})
    steps=[{'frame_index':s['frame_index'],'action':copy.deepcopy(s['action']),
            'outcome':{k:copy.deepcopy(v) for k,v in s['outcome'].items() if k not in ('internal_metadata','target','future_action')}}
           for s in raw.get('steps',[]) if int(s['frame_index'])<anchor]
    env={k:copy.deepcopy(raw.get('environment',{}).get(k)) for k in ('seed','wired_edges')}
    return {'environment':env,'decisions':decisions,'steps':steps}

def build_live_sample(raw):
    raw=causal_prefix(raw)
    raw,receipt=amend_raw_with_causal_flow_ledger(amend_raw_graph_inputs(raw))
    if not receipt['passed']:raise SmokeFailure('LIVE_SLOT_UNSUPPORTED','causal flow ledger')
    ds=raw['decisions'][-2:]; anchor=int(ds[-1]['frame_index']); steps={int(s['frame_index']):s for s in raw['steps']}
    if int(ds[0]['frame_index'])!=anchor-1:raise SmokeFailure('REQUIRED_LIVE_OBSERVATION_MISSING','noncontinuous history')
    if len({d['trajectory_id'] for d in ds})!=1:raise SmokeFailure('LIVE_SLOT_UNSUPPORTED','trajectory identity')
    ni={x:i for i,x in enumerate(sorted({str(v['entity_id']) for d in ds for v in d['entities']}))}
    ti={x:i for i,x in enumerate(sorted({str(v['task_id']) for d in ds for v in d['tasks']}))}
    past=steps.get(anchor-1)
    if past is None:raise SmokeFailure('REQUIRED_LIVE_OBSERVATION_MISSING','past action/outcome')
    fi={x:i for i,x in enumerate(sorted({_flow_id(e) for e in past['outcome'].get('slot_transfer_events',[])}))}
    li=_index_by_raw_flow([v for d in ds for v in d['logical_flow_rows']])
    history=[]
    for d in ds:
        frame=d['frame_index']; entities=_observation_entities(d,list(ni),ni); tasks=_observation_tasks(d,list(ti),ti)
        obs={'entities':entities,'tasks':tasks,'relation_endpoints':[
            {'relation_id':str(v['physical_edge_id']),'source_entity_index':ni[v['source_id']],
             'target_entity_index':ni[v['target_id']],'validity_mask':True} for v in d['channel_rows'] if v['source_id'] in ni and v['target_id'] in ni],
            'dag_relations':{'rows':[{'dag_edge_id':str(v['dag_edge_id']),'source_task_index':ti[v['source_task_id']],
              'target_task_index':ti[v['target_task_id']],'validity_mask':True} for v in d['dag_edges'] if v['source_task_id'] in ti and v['target_task_id'] in ti],
              'observed_mask':True,'source':'airfogsim_full_dual_graph_observer_v1._extract_dag_edges'}}
        row={'frame_index':frame,'simulation_time_s':d['simulation_time_s'],**obs,'observation':obs}
        if frame<anchor:
            s=steps[frame];row['action']={k:_action_tensor(s['action'],k,ti,ni) for k in ACTION_FAMILIES}
            row['outcome']=_outcome_rows(s['outcome'],list(ni),list(ti),ni,ti,fi)
            row['outcome'].update(frame_index=s['outcome']['frame_index'],simulation_time_s=s['outcome']['simulation_time_s'])
        es={v['entity_id']:v for v in d['entities']};ts={v['task_id']:v for v in d['tasks']}
        for v in entities:
            source=es.get(v['entity_id']);v['position_m']=_vector_wrapped(source.get('position_m') if source else None,source is not None,unit='m',width=3)
            v.setdefault('feature_mask',{})['position_m']=list(v['position_m']['feature_mask'])
        for v in tasks:
            s=ts.get(v['task_id']);present=s is not None
            v['arrival_time_s']={**_wrapped(s.get('arrival_time_s') if s else None,present,unit='s'),'role':'causal_derivation_source_not_separate_tensor_feature'}
            for name,unit in [('task_cpu_work','AirFogSim CPU-work-unit'),('computed_cpu_work','AirFogSim CPU-work-unit'),('transmitted_size','AirFogSim data-unit')]:
                v[name]=_wrapped(s.get(name) if s else None,present,unit=unit);v.setdefault('feature_mask',{})[name]=v[name]['feature_mask']
            elapsed=None if s is None else float(d['simulation_time_s'])-float(s['arrival_time_s'])
            if elapsed is not None and elapsed< -1e-9:raise SmokeFailure('REQUIRED_LIVE_OBSERVATION_MISSING','future task')
            v['elapsed_time_s']=_wrapped(None if elapsed is None else max(0,elapsed),present,unit='s');v['feature_mask']['elapsed_time_s']=v['elapsed_time_s']['feature_mask']
        row['communication_relations']=_communication_rows(d,ni);row['task_agent_relations']=_task_agent_rows(d,ni,ti)
        obs['communication_relations']=row['communication_relations'];obs['task_agent_relations']=row['task_agent_relations']
        row['logical_flows'],row['carrying_states']=_sample_rows(d['logical_flow_rows'],d['carrying_rows'],li,ni,ti,target=False)
        history.append(row)
    capability=[]
    for entity,slot in ni.items():
        observed={float(v['capacity_per_s']) for d in ds for v in d['node_cpu_capacity_observation_rows'] if v['node_id']==entity and v.get('observed_mask') and v.get('capacity_per_s') is not None}
        if len(observed)>1:raise SmokeFailure('REQUIRED_LIVE_OBSERVATION_MISSING','static CPU changed')
        value=next(iter(observed)) if observed else None
        capability.append({'entity_id':entity,'agent_index':slot,'cpu_capacity_per_s':{**_wrapped(value,True,unit='AirFogSim CPU-work-unit/s'),'observed_mask':value is not None,'semantic_role':'information_agent.static_capability'}})
    empty={k:{} for k in ('physical','task','flow','logical_flow')}
    return {'schema_version':SAMPLE_SCHEMA_VERSION,'contract':{**TensorContract(history_steps=2,horizon_steps=0).to_dict(),'schema_version':SAMPLE_SCHEMA_VERSION},
        'history':history,'static':{'input_entity_index':{'physical':ni,'task':ti,'flow':fi,'logical_flow':li},
           'input_entity_type_by_index':{str(slot):next((v['entity_type'] for d in ds for v in d['entities'] if v['entity_id']==entity),None) for entity,slot in ni.items()},
           'target_index':empty,'target_only_objects':{k:[] for k in empty},'agent_static_capability':capability},
        # Zero-length namespaces are absent supervision, not synthetic Future Target values.
        'future_action':[],'target':[],
        'metadata':{'trajectory_id':ds[-1]['trajectory_id'],'anchor_decision_frame':anchor,'history_frame_indices':[d['frame_index'] for d in ds],
          'future_action_frame_indices':[],'sample_id':f"{ds[-1]['trajectory_id']}::anchor-{anchor:04d}",'split':'dev_train_mechanism_smoke','metadata_is_model_input':False}}

def live_deadline_sidecar(env,decision,runtime_tasks):
    rows=[]
    for row in decision['tasks']:
        task=runtime_tasks.get(str(row['task_id']))
        if task is None:raise SmokeFailure('REQUIRED_LIVE_OBSERVATION_MISSING','deadline task missing')
        arrival=float(task.getTaskArrivalTime());deadline=float(task.getTaskDeadline())
        if not math.isfinite(deadline) or deadline<0 or not math.isclose(arrival,float(row['arrival_time_s']),abs_tol=1e-9):raise SmokeFailure('REQUIRED_LIVE_OBSERVATION_MISSING','deadline alignment')
        rows.append({'task_id':str(row['task_id']),'arrival_time_s':arrival,'deadline_s':deadline})
    return {**{k:decision[k] for k in ('trajectory_id','frame_index','capture_event_id','simulation_time_s')},'source_time_s':float(env.simulation_time),
        'alignment_passed':True,'tasks':sorted(rows,key=lambda v:v['task_id']),'source':'live Task getters before action; no future replay'}

def validate_command(step,context,decision,runtime_tasks):
    if step is None:raise SmokeFailure('NO_SCOREABLE_H4')
    if step.route:raise SmokeFailure('ACTION_BRIDGE_REJECTED','Route must remain explicit NOOP')
    from .step6_0a_candidate_generation_v1 import validate_step
    try:
        validate_step(step,context)
    except (ValueError,KeyError,IndexError) as exc:
        raise SmokeFailure('ACTION_BRIDGE_REJECTED',str(exc)) from exc
    tasks={v['task_id']:v for v in decision['tasks']};entities={v['entity_id']:v for v in decision['entities']}
    capacities={v['node_id']:v for v in decision['node_cpu_capacity_observation_rows'] if v.get('observed_mask')}
    inverse={int(v):k for k,v in context.static['input_entity_index']['physical'].items()}
    rb={};cpu={};patterns={};totals={}
    for row in step.comm:
        task=runtime_tasks.get(row['task_id']);ids=list(row['rb_indices'])
        if task is None or row['task_id'] not in tasks or not ids or len(set(ids))!=len(ids) or any(type(v)!=int or not 0<=v<int(decision['n_rb']) for v in ids):raise SmokeFailure('ACTION_BRIDGE_REJECTED','RB/task mapping')
        if tasks[row['task_id']]['lifecycle'] not in ('offloading','transmitting'):raise SmokeFailure('ACTION_BRIDGE_REJECTED','communication not current eligible')
        rb.setdefault(row['task_id'],[]).extend(ids)
    for row in step.comp:
        task=runtime_tasks.get(row['task_id']);node=row['node_id'];amount=float(row['allocated_cpu_per_s'])
        if task is None or row['task_id'] not in tasks or tasks[row['task_id']]['lifecycle']!='computing' or task.getAssignedTo()!=node or node not in entities or node not in capacities or not math.isfinite(amount) or amount<0 or row['task_id'] in cpu:raise SmokeFailure('ACTION_BRIDGE_REJECTED','CPU task/node/presence/unit')
        cpu[row['task_id']]=amount;totals[node]=totals.get(node,0)+amount
    if any(total>float(capacities[node]['capacity_per_s'])+1e-9 for node,total in totals.items()):raise SmokeFailure('ACTION_BRIDGE_REJECTED','CPU budget')
    for row in step.mob:
        node=inverse.get(int(row['uav_index']));values=[float(row[k]) for k in ('azimuth_rad','elevation_rad','speed_mps')]
        if node not in entities or entities[node]['entity_type']!='uav' or node in patterns or not all(math.isfinite(v) for v in values) or values[-1]<0:raise SmokeFailure('ACTION_BRIDGE_REJECTED','UAV mapping/unit')
        patterns[node]=dict(zip(('angle','phi','speed'),values))
    return {'route':'EXPLICIT_NOOP_ONLY','rb':rb,'cpu':cpu,'mobility':patterns}

def apply_commands(env,commands,communication,computation,traffic):
    # These three native setters mutate only the documented decision buffers.
    old=(copy.deepcopy(env.activated_offloading_tasks_with_RB_Nos),env.alloc_cpu_callback,copy.deepcopy(env.uav_mobility_patterns))
    try:
        env.activated_offloading_tasks_with_RB_Nos={}
        for task,ids in sorted(commands['rb'].items()):communication.setCommunicationWithRB(env,task,ids)
        computation.setComputingCallBack(env,lambda _:dict(commands['cpu']))
        traffic.setUAVMobilityPatterns(env,commands['mobility'])
    except Exception as exc:
        env.activated_offloading_tasks_with_RB_Nos,env.alloc_cpu_callback,env.uav_mobility_patterns=old
        # No retry and no env.step; episode must terminate even after restored decision buffers.
        raise SmokeFailure('SETTER_FAILURE',str(exc)) from exc
