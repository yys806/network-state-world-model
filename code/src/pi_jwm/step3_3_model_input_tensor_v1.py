"""CPU fixed-shape collation for the frozen Step 3.1F JSON sample."""
from __future__ import annotations

import copy, json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from .model_ready_sample_contract_v1 import SCHEMA_VERSION as SAMPLE_SCHEMA_VERSION

SCHEMA_VERSION = "PI-JWM-Step-3.3-Model-Input-Tensor-Collation-v2"
ACTION_FAMILIES = ("route", "comm", "comp", "mobility")
ENTITY_FEATURES = ("speed_mps", "canonical_acceleration_mps2")
MOBILITY_FEATURES = ("azimuth_rad", "elevation_rad", "speed_mps")
LIFECYCLE_VOCAB = ("<PAD>", "unknown", "to_generate", "waiting_to_offload", "offloading", "computing", "transmitting", "completed", "failed")
ENTITY_TYPE_VOCAB = ("<PAD>", "unknown", "vehicle", "uav", "rsu", "edge", "cloud")
ROUTE_KIND_VOCAB = ("<PAD>", "unknown", "offload", "return")
TRANSPORT_VOCAB = ("<PAD>", "unknown", "wireless", "wired")
TENSOR_INPUT_GAP_TABLE = (
    {"field":"entity_type", "raw_source":"present", "sample":"static and history", "tensor":"entity_type_index", "status":"02_required_and_fixed", "follow_up":None},
    {"field":"position_m / spatial state", "raw_source":"present", "sample":"not exposed", "tensor":"absent", "status":"feature_selection_not_decided", "follow_up":"03/04 mapping"},
    {"field":"channel_rows.feature_values / CSI", "raw_source":"source-dependent", "sample":"endpoint_ids_only", "tensor":"absent", "status":"numeric feature not exposed", "follow_up":"03/04 mapping"},
    {"field":"node_cpu_capacity_per_s", "raw_source":"not reliable in current sample", "sample":"not exposed", "tensor":"absent", "status":"not synthesized", "follow_up":"03/04 mapping"},
    {"field":"current Flow total/rem state", "raw_source":"not exposed as stateful field", "sample":"past hop service event only", "tensor":"past_outcome_flow_service", "status":"not equivalent to current Flow state", "follow_up":"03/04 definition"},
    {"field":"raw_simulator_acceleration_mps2", "raw_source":"present", "sample":"audit metadata", "tensor":"absent", "status":"audit_only", "follow_up":"excluded from model feature"},
    {"field":"future_task_schedule", "raw_source":"internal metadata", "sample":"excluded", "tensor":"absent", "status":"metadata_only", "follow_up":None},
)

@dataclass(frozen=True)
class TensorContract:
    history_steps:int=2; horizon_steps:int=2; max_entity:int=0; max_task:int=0; max_flow:int=0
    max_target_entity:int=0; max_target_task:int=0; max_target_flow:int=0; max_relation:int=0; max_dag:int=0
    max_past_outcome_relation:int=0; max_past_outcome_dag:int=0
    max_route_hops:int=0; max_action_entries:int=0; n_rb:int=0
    entity_feature_order:tuple[str,...]=ENTITY_FEATURES; mobility_feature_order:tuple[str,...]=MOBILITY_FEATURES
    dtype:str="float32"; index_dtype:str="int64"
    input_index_policy:str="history_causal_observable_object_union"
    future_action_index_policy:str="anchor_visibility_then_history_union_input_index"
    sample_contract_version:str=SAMPLE_SCHEMA_VERSION
    padding_policy:str="numeric=0; index=-1; categorical=0; presence/mask=false"
    unknown_policy:str="categorical unknown code=1; padding code=0"
    lifecycle_vocab:tuple[str,...]=LIFECYCLE_VOCAB
    entity_type_vocab:tuple[str,...]=ENTITY_TYPE_VOCAB
    route_kind_vocab:tuple[str,...]=ROUTE_KIND_VOCAB
    transport_vocab:tuple[str,...]=TRANSPORT_VOCAB
    feature_units:tuple[tuple[str,str],...]=()
    dag_direction:str="source_task_j -> target_task_k; k depends on j"
    vehicle_motion_policy:str="SUMO_external_not_in_mobility_action"
    normalization_policy:str="reuse_step3_2_train_only_stats; no_refit_in_collation"
    def to_dict(self):
        d=asdict(self)
        for k in ("entity_feature_order","mobility_feature_order","lifecycle_vocab","entity_type_vocab","route_kind_vocab","transport_vocab"): d[k]=list(getattr(self,k))
        d["feature_units"]=dict(self.feature_units); d["schema_version"]=SCHEMA_VERSION; return d
    @classmethod
    def from_dict(cls,d):
        names={f.name for f in fields(cls)}; p={k:d[k] for k in names if k in d}
        for k in ("entity_feature_order","mobility_feature_order","lifecycle_vocab","entity_type_vocab","route_kind_vocab","transport_vocab"): p[k]=tuple(p[k])
        p["feature_units"]=tuple(sorted(dict(p.get("feature_units",{})).items())); return cls(**p)

def _triplet(row, field):
    if row.get("presence") is False: return 0.,0.,False
    mask=row.get("feature_mask",{}); raw=row.get(field); object_valid=not isinstance(mask,Mapping) or field not in mask or bool(mask.get(field,False))
    if isinstance(raw,Mapping):
        valid=object_valid and bool(raw.get("presence",True)) and bool(raw.get("feature_mask",False)) and raw.get("value") is not None
        return (float(raw["value"]),float(raw.get("normalized_value",raw["value"])),True) if valid else (0.,0.,False)
    valid=object_valid and raw is not None; return (float(raw),float(raw),True) if valid else (0.,0.,False)

def _id(index, object_id, numeric, label):
    key=str(object_id)
    if key not in index or int(index[key])!=int(numeric): raise ValueError(f"{label} ID/index mismatch: {key} != {numeric}")
    return int(numeric)

def _vocab(vocab,value):
    try:return list(vocab).index(str(value))
    except ValueError:return 1

def _maxima(samples):
    def mx(ns,key,frames):
        return max((int(v)+1 for s in samples for v in s["static"].get(ns,{}).get(key,{}).values()),default=0)
    rb=[int(v) for s in samples for a in s["future_action"] for e in a["comm"].get("entries",[]) for v in e.get("rb_indices",[])] + [int(v) for s in samples for f in s["history"] for e in f.get("action",{}).get("comm",{}).get("entries",[]) for v in e.get("rb_indices",[])]
    past_routes=(len(e.get("route_node_indices",[])) for s in samples for f in s["history"] for e in f.get("action",{}).get("route",{}).get("entries",[]))
    future_routes=(len(e.get("route_node_indices",[])) for s in samples for a in s["future_action"] for e in a["route"].get("entries",[]))
    past_actions=(len(f.get("action",{}).get(name,{}).get("entries",[])) for s in samples for f in s["history"] for name in ACTION_FAMILIES)
    future_actions=(len(a.get(name,{}).get("entries",[])) for s in samples for a in s["future_action"] for name in ACTION_FAMILIES)
    return {"max_entity":mx("input_entity_index","physical",None),"max_task":mx("input_entity_index","task",None),"max_flow":mx("input_entity_index","flow",None),"max_target_entity":mx("target_index","physical",None),"max_target_task":mx("target_index","task",None),"max_target_flow":mx("target_index","flow",None),"max_relation":max((len(f.get("relation_endpoints",[])) for s in samples for f in s["history"]),default=0),"max_dag":max((len(f.get("dag_relations",{}).get("rows",[])) for s in samples for f in s["history"]),default=0),"max_past_outcome_relation":max((len(f.get("outcome",{}).get("relation_endpoints",[])) for s in samples for f in s["history"]),default=0),"max_past_outcome_dag":max((len(f.get("outcome",{}).get("dag_relations",{}).get("rows",[])) for s in samples for f in s["history"]),default=0),"max_route_hops":max(max(past_routes,default=0),max(future_routes,default=0)),"max_action_entries":max(max(past_actions,default=0),max(future_actions,default=0)),"n_rb":max(rb,default=-1)+1}

def _labels(samples,field,source):
    rows=[]
    for s in samples:
        if source=="lifecycle": rows += [r for f in s["history"] for r in f.get("tasks",[])]
        elif source=="entity": rows += [r for f in s["history"] for r in f.get("entities",[])]
        elif source=="route": rows += [r for a in s["future_action"] for r in a["route"].get("entries",[])]
        else: rows += [r for t in s["target"] for r in t.get("flows",[])]
    return tuple(["<PAD>","unknown"]+sorted({str(r[field]) for r in rows if r.get(field) is not None}-{"<PAD>","unknown"}))

def _contract(samples,stats,contract):
    if not samples: raise ValueError("cannot tensorize empty samples")
    for s in samples:
        if s.get("schema_version")!=SAMPLE_SCHEMA_VERSION: raise ValueError("frozen sample schema mismatch")
    o=_maxima(samples)
    if contract is None:
        contract=TensorContract(history_steps=len(samples[0]["history"]),horizon_steps=len(samples[0]["future_action"]),feature_units=tuple(sorted((str(k),str(v.get("unit"))) for k,v in (stats or {}).get("features",{}).items() if v.get("unit"))),**o)
    for k,v in o.items():
        if v>getattr(contract,k): raise ValueError(f"capacity overflow: {k} observed={v} capacity={getattr(contract,k)}")
    return contract

def _fill_service(out,prefix,bi,ti,service,task_index):
    for source,name in (("wireless_delivered_data_by_task","wireless"),("wired_delivered_data_by_task","wired"),("delivered_data_by_task","total")):
        for task_id,value in (service.get(source) or {}).items():
            q=task_index.get(str(task_id),-1)
            if 0<=q<out[f"{prefix}_{name}_service"].shape[2] and value is not None:
                out[f"{prefix}_{name}_service"][bi,ti,q]=float(value); out[f"{prefix}_{name}_service_mask"][bi,ti,q]=True

def _fill_past_outcome(out,outcome,bi,ti,idx,c):
    for r in outcome.get("entities",[]):
        q=_id(idx["physical"],r["entity_id"],r["entity_index"],"past_outcome.entity"); out["past_outcome_entity_presence"][bi,ti,q]=bool(r.get("presence",False)); out["past_outcome_entity_type_index"][bi,ti,q]=_vocab(c.entity_type_vocab,r.get("entity_type")); _,out["past_outcome_entity_features"][bi,ti,q,0],out["past_outcome_entity_feature_mask"][bi,ti,q,0]=_triplet(r,"speed_mps")
    for r in outcome.get("tasks",[]):
        q=_id(idx["task"],r["task_id"],r["task_index"],"past_outcome.task"); out["past_outcome_task_presence"][bi,ti,q]=bool(r.get("presence",False)); out["past_outcome_task_lifecycle_index"][bi,ti,q]=_vocab(c.lifecycle_vocab,r.get("lifecycle")); _,out["past_outcome_task_features"][bi,ti,q,0],out["past_outcome_task_feature_mask"][bi,ti,q,0]=_triplet(r,"transmitted_size")
    for r in outcome.get("flows",[]):
        q=_id(idx["flow"],r["flow_id"],r["flow_index"],"past_outcome.flow"); val=r.get("service_volume"); out["past_outcome_flow_presence"][bi,ti,q]=True; out["past_outcome_flow_service"][bi,ti,q]=0. if val is None else float(val); out["past_outcome_flow_service_mask"][bi,ti,q]=val is not None; out["past_outcome_flow_task_index"][bi,ti,q]=_id(idx["task"],r["task_id"],r["task_index"],"past_outcome.flow.task"); out["past_outcome_flow_source_node_index"][bi,ti,q]=_id(idx["physical"],r["source_node_id"],r["source_node_index"],"past_outcome.flow.source"); out["past_outcome_flow_target_node_index"][bi,ti,q]=_id(idx["physical"],r["target_node_id"],r["target_node_index"],"past_outcome.flow.target"); out["past_outcome_flow_transport_index"][bi,ti,q]=_vocab(c.transport_vocab,r.get("transport")); out["past_outcome_flow_transport_mask"][bi,ti,q]=r.get("transport") is not None
    _fill_service(out,"past_outcome",bi,ti,outcome.get("communication_service",{}),idx["task"])
    for task_id,value in (outcome.get("served_cpu_work_by_task") or {}).items():
        q=idx["task"].get(str(task_id),-1)
        if 0<=q<c.max_task and value is not None: out["past_outcome_served_cpu_work"][bi,ti,q]=float(value); out["past_outcome_served_cpu_work_mask"][bi,ti,q]=True
    for ri,r in enumerate(outcome.get("relation_endpoints",[])):
        if ri>=c.max_past_outcome_relation: raise ValueError("past outcome relation capacity overflow")
        out["past_outcome_relation_endpoints"][bi,ti,ri]=[int(r["source_entity_index"]),int(r["target_entity_index"])]; out["past_outcome_relation_mask"][bi,ti,ri]=bool(r.get("validity_mask",False))
    for di,r in enumerate(outcome.get("dag_relations",{}).get("rows",[])):
        if di>=c.max_past_outcome_dag: raise ValueError("past outcome DAG capacity overflow")
        out["past_outcome_dag_edges"][bi,ti,di]=[int(r["source_task_index"]),int(r["target_task_index"])]; out["past_outcome_dag_mask"][bi,ti,di]=bool(r.get("validity_mask",False))

def build_tensor_batch(samples:Sequence[Mapping[str,Any]],*,stats=None,contract=None):
    samples=list(samples); c=_contract(samples,stats,contract); B,H,L=len(samples),c.history_steps,c.horizon_steps; E,T,F,R,D,A,Q,N=c.max_entity,c.max_task,c.max_flow,c.max_relation,c.max_dag,c.max_action_entries,c.max_route_hops,c.n_rb
    z=lambda shape,dtype=np.float32:np.zeros(shape,dtype=dtype); ix=lambda shape:np.full(shape,-1,dtype=np.int64)
    out={"schema_version":SCHEMA_VERSION,"contract":c,"sample_ids":[],"sample_static":[],"sample_metadata":[]}
    out.update({"entity_raw_features":z((B,H,E,2)),"entity_features":z((B,H,E,2)),"entity_feature_mask":z((B,H,E,2),bool),"entity_presence":z((B,H,E),bool),"entity_type_index":z((B,H,E),np.int64),"task_raw_features":z((B,H,T,1)),"task_features":z((B,H,T,1)),"task_feature_mask":z((B,H,T,1),bool),"task_presence":z((B,H,T),bool),"task_lifecycle_index":z((B,H,T),np.int64),"flow_features":z((B,H,F,1)),"flow_presence":z((B,H,F),bool),"flow_feature_mask":z((B,H,F,1),bool),"flow_task_index":ix((B,H,F)),"flow_source_node_index":ix((B,H,F)),"flow_target_node_index":ix((B,H,F)),"flow_transport_index":z((B,H,F),np.int64),"flow_transport_mask":z((B,H,F),bool),"relation_endpoints":ix((B,H,R,2)),"relation_mask":z((B,H,R),bool),"dag_edges":ix((B,H,D,2)),"dag_mask":z((B,H,D),bool)})
    for p in ("past","future"):
        n=H-1 if p=="past" else L
        out.update({f"{p}_action_present":z((B,n,4),bool),f"{p}_action_empty":z((B,n,4),bool),f"{p}_action_missing":z((B,n,4),bool),f"{p}_action_entry_mask":z((B,n,4,A),bool),f"{p}_task_index":ix((B,n,4,A)),f"{p}_target_node_index":ix((B,n,4,A)),f"{p}_task_node_index":ix((B,n,4,A)),f"{p}_uav_index":ix((B,n,4,A)),f"{p}_route_node_indices":ix((B,n,A,Q)),f"{p}_route_hop_mask":z((B,n,A,Q),bool),f"{p}_comm_rb_indices":ix((B,n,A,N)),f"{p}_comm_rb_mask":z((B,n,A,N),bool),f"{p}_mobility_features":z((B,n,A,3)),f"{p}_mobility_feature_mask":z((B,n,A,3),bool),f"{p}_comp_node_index":ix((B,n,A)),f"{p}_comp_allocated_cpu":z((B,n,A)),f"{p}_comp_allocated_cpu_mask":z((B,n,A),bool),f"{p}_route_kind_index":z((B,n,A),np.int64),f"{p}_route_kind_mask":z((B,n,A),bool)})
    P=H-1; PE,PT,PF=E,T,F; TE,TT,TF=c.max_target_entity,c.max_target_task,c.max_target_flow
    out.update({
        "past_outcome_entity_presence":z((B,P,PE),bool),"past_outcome_entity_features":z((B,P,PE,1)),"past_outcome_entity_feature_mask":z((B,P,PE,1),bool),"past_outcome_entity_type_index":z((B,P,PE),np.int64),
        "past_outcome_task_presence":z((B,P,PT),bool),"past_outcome_task_features":z((B,P,PT,1)),"past_outcome_task_feature_mask":z((B,P,PT,1),bool),"past_outcome_task_lifecycle_index":z((B,P,PT),np.int64),
        "past_outcome_flow_presence":z((B,P,PF),bool),"past_outcome_flow_service":z((B,P,PF)),"past_outcome_flow_service_mask":z((B,P,PF),bool),"past_outcome_flow_task_index":ix((B,P,PF)),"past_outcome_flow_source_node_index":ix((B,P,PF)),"past_outcome_flow_target_node_index":ix((B,P,PF)),"past_outcome_flow_transport_index":z((B,P,PF),np.int64),"past_outcome_flow_transport_mask":z((B,P,PF),bool),
        "past_outcome_wireless_service":z((B,P,PT)),"past_outcome_wired_service":z((B,P,PT)),"past_outcome_total_service":z((B,P,PT)),"past_outcome_wireless_service_mask":z((B,P,PT),bool),"past_outcome_wired_service_mask":z((B,P,PT),bool),"past_outcome_total_service_mask":z((B,P,PT),bool),"past_outcome_served_cpu_work":z((B,P,PT)),"past_outcome_served_cpu_work_mask":z((B,P,PT),bool),
        "past_outcome_relation_endpoints":ix((B,P,c.max_past_outcome_relation,2)),"past_outcome_relation_mask":z((B,P,c.max_past_outcome_relation),bool),"past_outcome_dag_edges":ix((B,P,c.max_past_outcome_dag,2)),"past_outcome_dag_mask":z((B,P,c.max_past_outcome_dag),bool),
        "target_entity_presence":z((B,L,TE),bool),"target_entity_features":z((B,L,TE,1)),"target_entity_feature_mask":z((B,L,TE,1),bool),"target_entity_type_index":z((B,L,TE),np.int64),
        "target_task_presence":z((B,L,TT),bool),"target_task_features":z((B,L,TT,1)),"target_task_feature_mask":z((B,L,TT,1),bool),"target_task_lifecycle_index":z((B,L,TT),np.int64),
        "target_flow_presence":z((B,L,TF),bool),"target_flow_index":ix((B,L,TF)),"target_flow_task_index":ix((B,L,TF)),"target_flow_source_node_index":ix((B,L,TF)),"target_flow_target_node_index":ix((B,L,TF)),"target_flow_transport_index":z((B,L,TF),np.int64),"target_flow_transport_mask":z((B,L,TF),bool),"target_flow_service":z((B,L,TF)),"target_flow_feature_mask":z((B,L,TF),bool),
        "target_wireless_service":z((B,L,TT)),"target_wired_service":z((B,L,TT)),"target_total_service":z((B,L,TT)),"target_wireless_service_mask":z((B,L,TT),bool),"target_wired_service_mask":z((B,L,TT),bool),"target_total_service_mask":z((B,L,TT),bool),
    })
    for bi,s in enumerate(samples):
        idx={k:{str(a):int(v) for a,v in d.items()} for k,d in s["static"]["input_entity_index"].items()}; tgt={k:{str(a):int(v) for a,v in d.items()} for k,d in s["static"].get("target_index",{}).items()}; out["sample_ids"].append(str(s["metadata"]["sample_id"])); out["sample_static"].append({"input_entity_index":copy.deepcopy(idx),"target_index":copy.deepcopy(tgt),"target_only_objects":copy.deepcopy(s["static"].get("target_only_objects",{})),"input_entity_type_by_index":copy.deepcopy(s["static"].get("input_entity_type_by_index",{}))}); out["sample_metadata"].append({"sample_id":str(s["metadata"]["sample_id"]),"trajectory_id":str(s["metadata"]["trajectory_id"]),"split":str(s["metadata"]["split"])})
        for hi,fm in enumerate(s["history"]):
            for r in fm.get("entities",[]):
                q=_id(idx["physical"],r["entity_id"],r["entity_index"],"entity"); out["entity_presence"][bi,hi,q]=bool(r.get("presence",False)); out["entity_type_index"][bi,hi,q]=_vocab(c.entity_type_vocab,r.get("entity_type"));
                for fi,field in enumerate(ENTITY_FEATURES): out["entity_raw_features"][bi,hi,q,fi],out["entity_features"][bi,hi,q,fi],out["entity_feature_mask"][bi,hi,q,fi]=_triplet(r,field)
            for r in fm.get("tasks",[]):
                q=_id(idx["task"],r["task_id"],r["task_index"],"task"); out["task_presence"][bi,hi,q]=bool(r.get("presence",False)); out["task_raw_features"][bi,hi,q,0],out["task_features"][bi,hi,q,0],out["task_feature_mask"][bi,hi,q,0]=_triplet(r,"task_size"); out["task_lifecycle_index"][bi,hi,q]=_vocab(c.lifecycle_vocab,r.get("lifecycle"))
            for r in fm.get("flows",[]):
                q=_id(idx["flow"],r["flow_id"],r["flow_index"],"flow"); out["flow_presence"][bi,hi,q]=True; val=r.get("service_volume"); out["flow_features"][bi,hi,q,0]=0. if val is None else float(val); out["flow_feature_mask"][bi,hi,q,0]=val is not None; out["flow_task_index"][bi,hi,q]=_id(idx["task"],r["task_id"],r["task_index"],"flow.task"); out["flow_source_node_index"][bi,hi,q]=_id(idx["physical"],r["source_node_id"],r["source_node_index"],"flow.source"); out["flow_target_node_index"][bi,hi,q]=_id(idx["physical"],r["target_node_id"],r["target_node_index"],"flow.target"); out["flow_transport_index"][bi,hi,q]=_vocab(c.transport_vocab,r.get("transport")); out["flow_transport_mask"][bi,hi,q]=r.get("transport") is not None
            for ri,r in enumerate(fm.get("relation_endpoints",[])):
                if ri>=R: raise ValueError("relation capacity overflow")
                out["relation_endpoints"][bi,hi,ri]=[int(r["source_entity_index"]),int(r["target_entity_index"])] ; out["relation_mask"][bi,hi,ri]=bool(r.get("validity_mask",False))
            for di,r in enumerate(fm.get("dag_relations",{}).get("rows",[])):
                if di>=D: raise ValueError("DAG capacity overflow")
                out["dag_edges"][bi,hi,di]=[int(r["source_task_index"]),int(r["target_task_index"])] ; out["dag_mask"][bi,hi,di]=bool(r.get("validity_mask",False))
            if hi<H-1:
                _actions(out,s["history"][hi].get("action",{}),bi,hi,True,idx,c)
                _fill_past_outcome(out,s["history"][hi].get("outcome",{}),bi,hi,idx,c)
        for li,a in enumerate(s["future_action"]): _actions(out,a,bi,li,False,idx,c)
        for li,tg in enumerate(s["target"]):
            for r in tg.get("entities",[]):
                q=int(r["target_index"]); out["target_entity_presence"][bi,li,q]=bool(r.get("presence",False)); out["target_entity_type_index"][bi,li,q]=_vocab(c.entity_type_vocab,r.get("entity_type")); _,out["target_entity_features"][bi,li,q,0],out["target_entity_feature_mask"][bi,li,q,0]=_triplet(r,"speed_mps")
            for r in tg.get("tasks",[]):
                q=int(r["target_index"]); out["target_task_presence"][bi,li,q]=bool(r.get("presence",False)); out["target_task_lifecycle_index"][bi,li,q]=_vocab(c.lifecycle_vocab,r.get("lifecycle")); out["target_task_features"][bi,li,q,0],_,out["target_task_feature_mask"][bi,li,q,0]=_triplet(r,"transmitted_size")
            for r in tg.get("flows",[]):
                q=int(r.get("target_index",r.get("flow_index",-1))); val=r.get("service_volume"); out["target_flow_presence"][bi,li,q]=True; out["target_flow_index"][bi,li,q]=q; out["target_flow_task_index"][bi,li,q]=int(r["task_index"]); out["target_flow_source_node_index"][bi,li,q]=int(r["source_node_index"]); out["target_flow_target_node_index"][bi,li,q]=int(r["target_node_index"]); out["target_flow_transport_index"][bi,li,q]=_vocab(c.transport_vocab,r.get("transport")); out["target_flow_transport_mask"][bi,li,q]=r.get("transport") is not None; out["target_flow_service"][bi,li,q]=0. if val is None else float(val); out["target_flow_feature_mask"][bi,li,q]=val is not None
            _fill_service(out,"target",bi,li,tg.get("communication_service",{}),tgt["task"])
    return out

def _actions(out,a,bi,ti,past,idx,c):
    p="past" if past else "future"; n=a
    for fi,fam in enumerate(ACTION_FAMILIES):
        rec=n.get(fam,{}); out[f"{p}_action_present"][bi,ti,fi]=bool(rec.get("field_present",False)); out[f"{p}_action_empty"][bi,ti,fi]=bool(rec.get("empty",False)) and not bool(rec.get("missing",False)); out[f"{p}_action_missing"][bi,ti,fi]=bool(rec.get("missing",False))
        for ai,e in enumerate(rec.get("entries",[])):
            if ai>=c.max_action_entries: raise ValueError("action entry capacity overflow")
            out[f"{p}_action_entry_mask"][bi,ti,fi,ai]=True
            if fam in ("route","comm","comp"): out[f"{p}_task_index"][bi,ti,fi,ai]=_id(idx["task"],e.get("task_id"),e.get("task_index"),f"{fam}.task")
            if fam=="route":
                if e.get("target_node_id") is not None: out[f"{p}_target_node_index"][bi,ti,fi,ai]=_id(idx["physical"],e["target_node_id"],e["target_node_index"],"route.target")
                if e.get("task_node_id") is not None: out[f"{p}_task_node_index"][bi,ti,fi,ai]=_id(idx["physical"],e["task_node_id"],e["task_node_index"],"route.task_node")
                out[f"{p}_route_kind_index"][bi,ti,ai]=_vocab(c.route_kind_vocab,e.get("route_kind")); out[f"{p}_route_kind_mask"][bi,ti,ai]=e.get("route_kind") is not None; ids=e.get("route_node_ids",[]); nums=e.get("route_node_indices",[])
                if len(ids)!=len(nums) or len(ids)>c.max_route_hops: raise ValueError("route hop capacity/ID mismatch")
                for hi,(oid,num) in enumerate(zip(ids,nums)): out[f"{p}_route_node_indices"][bi,ti,ai,hi]=_id(idx["physical"],oid,num,"route.hop"); out[f"{p}_route_hop_mask"][bi,ti,ai,hi]=True
            elif fam=="comm":
                for rb in e.get("rb_indices",[]):
                    if int(rb)<0 or int(rb)>=c.n_rb: raise ValueError("RB capacity overflow")
                    out[f"{p}_comm_rb_indices"][bi,ti,ai,int(rb)]=int(rb); out[f"{p}_comm_rb_mask"][bi,ti,ai,int(rb)]=True
            elif fam=="comp":
                if e.get("node_id") is not None: out[f"{p}_comp_node_index"][bi,ti,ai]=_id(idx["physical"],e["node_id"],e["node_index"],"comp.node")
                if e.get("allocated_cpu_per_s") is not None: out[f"{p}_comp_allocated_cpu"][bi,ti,ai]=float(e["allocated_cpu_per_s"]); out[f"{p}_comp_allocated_cpu_mask"][bi,ti,ai]=True
            else:
                out[f"{p}_uav_index"][bi,ti,fi,ai]=_id(idx["physical"],e.get("uav_id"),e.get("uav_index"),"mobility.uav")
                for mi,field in enumerate(MOBILITY_FEATURES):
                    if e.get(field) is not None: out[f"{p}_mobility_features"][bi,ti,ai,mi]=float(e[field]); out[f"{p}_mobility_feature_mask"][bi,ti,ai,mi]=True

def validate_tensor_batch_checks(tensor):
    c=tensor["contract"] if isinstance(tensor["contract"],TensorContract) else TensorContract.from_dict(tensor["contract"]); b=len(tensor["sample_ids"])
    checks={"fixed_shapes":tensor["entity_features"].shape==(b,c.history_steps,c.max_entity,2) and tensor["future_action_present"].shape==(b,c.horizon_steps,4),"static_per_sample":len(tensor.get("sample_static",[]))==b,"past_outcome_time_axis":tensor["past_outcome_entity_presence"].shape[1]==c.history_steps-1 and tensor["past_outcome_task_presence"].shape[1]==c.history_steps-1}
    namespace=True
    identity=True
    for static in tensor.get("sample_static",[]):
        for ns,values in static.get("target_only_objects",{}).items(): namespace &= set(values).isdisjoint(static["input_entity_index"].get(ns,{}))
        for ns,limit in (("physical",c.max_entity),("task",c.max_task),("flow",c.max_flow)):
            slots=list(static["input_entity_index"].get(ns,{}).values()); identity &= len(slots)==len(set(slots)) and all(0<=int(slot)<limit for slot in slots)
    checks["input_target_namespace_isolation"]=bool(namespace)
    checks["id_json_index_tensor_slot_alignment"]=bool(identity)
    index_padding=True
    for name,values in tensor.items():
        if isinstance(values,np.ndarray) and (name.endswith("_index") or name.endswith("_indices") or name.endswith("_endpoints") or name.endswith("_edges")): index_padding &= bool(np.all(values>=-1))
    checks["padding_index_is_minus_one_or_valid"]=bool(index_padding)
    vocab_checks=(("entity_type_index",c.entity_type_vocab),("task_lifecycle_index",c.lifecycle_vocab),("flow_transport_index",c.transport_vocab),("past_outcome_entity_type_index",c.entity_type_vocab),("past_outcome_task_lifecycle_index",c.lifecycle_vocab),("past_outcome_flow_transport_index",c.transport_vocab),("target_entity_type_index",c.entity_type_vocab),("target_task_lifecycle_index",c.lifecycle_vocab),("target_flow_transport_index",c.transport_vocab),("past_route_kind_index",c.route_kind_vocab),("future_route_kind_index",c.route_kind_vocab))
    categories=True
    for name,vocab in vocab_checks:
        if name in tensor and tensor[name].size: categories &= bool(np.all(tensor[name]<len(vocab)))
    checks["categorical_codes_in_vocabularies"]=bool(categories)
    endpoints_ok=True
    for endpoints,mask,limit in (("relation_endpoints","relation_mask",c.max_entity),("past_outcome_relation_endpoints","past_outcome_relation_mask",c.max_entity),("dag_edges","dag_mask",c.max_task),("past_outcome_dag_edges","past_outcome_dag_mask",c.max_task)):
        active=tensor[endpoints][tensor[mask]]
        endpoints_ok &= bool(np.all((active>=0)&(active<limit))) if active.size else True
    checks["relation_and_dag_endpoints_in_range"]=bool(endpoints_ok)
    checks["dag_direction_contract"]=c.dag_direction=="source_task_j -> target_task_k; k depends on j"
    placeholders=True
    for features,mask in (("entity_features","entity_feature_mask"),("past_outcome_entity_features","past_outcome_entity_feature_mask"),("past_outcome_task_features","past_outcome_task_feature_mask"),("target_entity_features","target_entity_feature_mask"),("target_task_features","target_task_feature_mask")):
        placeholders &= bool(np.all(tensor[features][~tensor[mask]]==0))
    checks["masked_numeric_placeholder_zero"]=bool(placeholders)
    action_ok=True
    for prefix in ("past","future"):
        active=tensor[f"{prefix}_action_entry_mask"]
        for fi,family in enumerate(ACTION_FAMILIES):
            fam_active=active[:,:,fi]
            if family in ("route","comm","comp"): action_ok &= bool(np.all((tensor[f"{prefix}_task_index"][:,:,fi][fam_active]>=0)&(tensor[f"{prefix}_task_index"][:,:,fi][fam_active]<c.max_task)))
        action_ok &= bool(np.all((tensor[f"{prefix}_route_node_indices"][tensor[f"{prefix}_route_hop_mask"]]>=0)&(tensor[f"{prefix}_route_node_indices"][tensor[f"{prefix}_route_hop_mask"]]<c.max_entity)))
        route_active=active[:,:,0]
        target_nodes=tensor[f"{prefix}_target_node_index"][:,:,0][route_active]; task_nodes=tensor[f"{prefix}_task_node_index"][:,:,0][route_active]
        action_ok &= bool(np.all((target_nodes>=0)&(target_nodes<c.max_entity)))
        action_ok &= bool(np.all((task_nodes==-1)|((task_nodes>=0)&(task_nodes<c.max_entity))))
        action_ok &= bool(np.all((tensor[f"{prefix}_comm_rb_indices"][tensor[f"{prefix}_comm_rb_mask"]]>=0)&(tensor[f"{prefix}_comm_rb_indices"][tensor[f"{prefix}_comm_rb_mask"]]<c.n_rb)))
        action_ok &= bool(np.all((tensor[f"{prefix}_uav_index"][:,:,3][active[:,:,3]]>=0)&(tensor[f"{prefix}_uav_index"][:,:,3][active[:,:,3]]<c.max_entity)))
        comp=tensor[f"{prefix}_comp_allocated_cpu_mask"]
        action_ok &= bool(np.all(tensor[f"{prefix}_comp_node_index"][comp]>=0) and np.all(np.isfinite(tensor[f"{prefix}_comp_allocated_cpu"][comp])))
    checks["action_references_semantically_valid"]=bool(action_ok)
    target_active=tensor["target_flow_presence"]
    checks["target_flow_references_in_target_namespace"]=bool(np.all((tensor["target_flow_task_index"][target_active]>=0)&(tensor["target_flow_task_index"][target_active]<c.max_target_task)) and np.all((tensor["target_flow_source_node_index"][target_active]>=0)&(tensor["target_flow_source_node_index"][target_active]<c.max_target_entity)) and np.all((tensor["target_flow_target_node_index"][target_active]>=0)&(tensor["target_flow_target_node_index"][target_active]<c.max_target_entity)))
    checks["current_y_t_excluded_by_axis_contract"]=tensor["past_outcome_entity_presence"].shape[1]==c.history_steps-1
    checks["categorical_padding_code_zero"]=bool(np.all(tensor["past_route_kind_index"][~tensor["past_route_kind_mask"]]==0) and np.all(tensor["future_route_kind_index"][~tensor["future_route_kind_mask"]]==0) and np.all(tensor["target_flow_transport_index"][~tensor["target_flow_transport_mask"]]==0))
    checks["no_silent_capacity_truncation"]=bool(tensor["entity_presence"].shape[2]==c.max_entity and tensor["task_presence"].shape[2]==c.max_task and tensor["target_flow_presence"].shape[2]==c.max_target_flow and tensor["past_action_entry_mask"].shape[3]==c.max_action_entries)
    checks["passed"]=bool(all(checks.values()))
    return checks

def validate_tensor_batch(tensor):
    return bool(validate_tensor_batch_checks(tensor)["passed"])

def save_tensor_batch(tensor,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); arrays={k:v for k,v in tensor.items() if isinstance(v,np.ndarray)}; c=tensor["contract"].to_dict() if isinstance(tensor["contract"],TensorContract) else tensor["contract"]
    for k,v in (("contract_json",c),("sample_ids_json",tensor["sample_ids"]),("sample_static_json",tensor["sample_static"]),("sample_metadata_json",tensor["sample_metadata"])): arrays[k]=np.asarray(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")))
    np.savez_compressed(path,**arrays)

def load_tensor_batch(path):
    with np.load(Path(path),allow_pickle=False) as d:
        excl={"contract_json","sample_ids_json","sample_static_json","sample_metadata_json"}; return {"schema_version":SCHEMA_VERSION,"contract":TensorContract.from_dict(json.loads(str(d["contract_json"]))),"sample_ids":json.loads(str(d["sample_ids_json"])),"sample_static":json.loads(str(d["sample_static_json"])),"sample_metadata":json.loads(str(d["sample_metadata_json"])),**{k:d[k] for k in d.files if k not in excl}}

__all__=["ACTION_FAMILIES","SCHEMA_VERSION","TENSOR_INPUT_GAP_TABLE","LIFECYCLE_VOCAB","ENTITY_TYPE_VOCAB","ROUTE_KIND_VOCAB","TRANSPORT_VOCAB","TensorContract","build_tensor_batch","load_tensor_batch","save_tensor_batch","validate_tensor_batch_checks","validate_tensor_batch"]
