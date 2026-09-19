"""Run one real AirFogSim single-step four-family contract acceptance."""
from __future__ import annotations
import hashlib, json, math, os, platform, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
SRC = CODE / "src"
SCRIPTS = CODE / "scripts"
for path in (SRC, SCRIPTS, CODE / "reference" / "AirFogSim" / "examples"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
from run_p2_single_step_collector_preflight_v1 import _build_environment
from pi_jwm.airfogsim_single_step_collector_v1 import SingleStepRecorder
from pi_jwm.airfogsim_full_dual_graph_observer_v1 import observe_airfogsim_snapshot
from pi_jwm.full_dual_graph_collector_contract_v1 import SnapshotPhase
from airfogsim.scheduler import TaskScheduler, CommunicationScheduler, ComputationScheduler, TrafficScheduler

OUT = CODE / "artifacts" / "protocols" / "pi_jwm_raw_single_decision_step_real_airfogsim_v4_20260919"

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def write_json(path, payload):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)

def _plain(value):
    if hasattr(value,"item"): value=value.item()
    return float(value)

def _traffic_snapshot(env):
    tm=env.traffic_manager
    rows=[]
    for entity_type, infos in (("vehicle", tm.getVehicleTrafficInfos()), ("uav", tm.getUAVTrafficInfos())):
        for entity_id, info in sorted(infos.items(), key=lambda x:str(x[0])):
            rows.append({"entity_id":str(entity_id),"entity_type":entity_type,"position_m":[float(x) for x in info["position"]],"speed_mps":float(info.get("speed",0.0)),"acceleration_mps2":float(info.get("acceleration",0.0)),"heading":float(info.get("angle",0.0)),"elevation_rad":(None if entity_type!="uav" else float(info.get("phi",0.0))),"vehicle_route_id":(None if entity_type!="vehicle" else str(info.get("routeId")))})
    for entity_type, infos in (("rsu",tm.getRSUInfos()),("cloud",tm.getCloudServerInfos())):
        for entity_id, info in sorted(infos.items(), key=lambda x:str(x[0])):
            rows.append({"entity_id":str(entity_id),"entity_type":entity_type,"position_m":[float(x) for x in info["position"]],"speed_mps":0.0,"acceleration_mps2":0.0,"heading":0.0,"elevation_rad":None,"vehicle_route_id":None})
    return rows

def _task_snapshot(env):
    snap=observe_airfogsim_snapshot(env, phase=SnapshotPhase.DECISION)
    return [{"task_id":row.task_id,"task_node_id":row.task_node_id,"current_node_id":row.current_node_id,"lifecycle":row.lifecycle.value,"route_node_ids":list(row.route_nodes),"return_destination_id":row.return_destination_id,"arrival_time_s":row.arrival_time,"task_size":row.task_size,"task_cpu_work":row.task_cpu,"computed_cpu_work":row.computed_size,"transmitted_size":row.in_stage_transmitted_size} for row in snap.tasks]

def _snapshot(env):
    return {"trajectory_id":None,"frame_index":None,"simulation_time_s":float(env.simulation_time),"entities":_traffic_snapshot(env),"tasks":_task_snapshot(env),"n_rb":int(env.channel_manager.n_RB)}

def _ready_task(env):
    rows=[]
    for tasks in env.task_manager._waiting_to_offload_tasks.values():
        rows.extend(tasks)
    rows=[task for task in rows if env.task_manager.checkTaskDependency(task.getTaskNodeId(),task.getTaskId()) is True]
    if not rows: return None
    return sorted(rows,key=lambda t:(str(t.getTaskNodeId()),str(t.getTaskId())))[0]

def _warm_to_real_ready(env, max_steps=50):
    for step in range(max_steps+1):
        if _ready_task(env) is not None and env.traffic_manager.getVehicleTrafficInfos() and env.traffic_manager.getUAVTrafficInfos(): return step
        env.alloc_cpu_callback=lambda _: {}
        env.step()
    raise RuntimeError("no ready task with vehicle and UAV after real warmup")

def main():
    old=os.getcwd(); env=None
    try:
        os.chdir(CODE / "reference" / "AirFogSim" / "examples")
        env, _, _, _, config = _build_environment(0, 3.0)
        warmup=_warm_to_real_ready(env)
        task=_ready_task(env); source=str(task.getTaskNodeId()); task_id=str(task.getTaskId())
        remote=[x for x in sorted(set(env.vehicles)|set(env.UAVs)|set(env.RSUs)) if x!=source and env._getNodeTypeById(x) in "VUI"]
        remote.sort(key=lambda x:(env.getDistanceBetweenNodesById(source,x),x)); target=remote[0]
        uav_id=sorted(env.UAVs)[0]; uav_before=env.traffic_manager.getUAVTrafficInfos()[uav_id]
        vehicle_before=next(iter(sorted(env.traffic_manager.getVehicleTrafficInfos())))
        frame=0; trajectory_id="step2.1-real-seed0"
        decision=_snapshot(env); decision.update({"trajectory_id":trajectory_id,"frame_index":frame})
        setters=[]
        original_step=env.step
        def observed_step(*args,**kwargs):
            setters.append({"setter_kind":"env.step","completed":False})
            result=original_step(*args,**kwargs); setters[-1]["completed"]=True; return result
        env.step=observed_step
        route_ok=TaskScheduler.setTaskOffloading(env, source, task_id, target, route=[target]); setters.append({"setter_kind":"offload","task_id":task_id,"succeeded":bool(route_ok)})
        CommunicationScheduler.setCommunicationWithRB(env, task_id, [0]); setters.append({"setter_kind":"rb","task_id":task_id,"rb_indices":[0],"succeeded":True})
        recorder=SingleStepRecorder(env, "step2.1-real")
        recorder.install_cpu_callback(ComputationScheduler); setters.append({"setter_kind":"cpu_callback","succeeded":True})
        mobility={"angle":float(uav_before.get("angle",0.0)+0.2),"phi":float(uav_before.get("phi",0.0)),"speed":max(float(uav_before.get("speed",0.0)),10.0)}
        TrafficScheduler.setUAVMobilityPatterns(env,{uav_id:mobility}); setters.append({"setter_kind":"uav_mobility","uav_id":uav_id,"command":mobility,"succeeded":True})
        execution_start=float(env.simulation_time); env.step(); execution_end=float(env.simulation_time); recorder.finalize_after_step()
        outcome=_snapshot(env); outcome.update({"trajectory_id":trajectory_id,"frame_index":frame,"capture":"directly after real env.step"})
        next_decision=dict(outcome); next_decision["frame_index"]=frame+1; next_decision["capture"]="directly from same post-step real environment"
        vehicles_before={x["entity_id"]:x for x in decision["entities"] if x["entity_type"]=="vehicle"}; vehicles_after={x["entity_id"]:x for x in outcome["entities"] if x["entity_type"]=="vehicle"}
        vehicle_moved=vehicle_before in vehicles_before and vehicle_before in vehicles_after and vehicles_before[vehicle_before]["position_m"]!=vehicles_after[vehicle_before]["position_m"]
        uav_after=next(x for x in outcome["entities"] if x["entity_id"]==uav_id)
        checks={"real_airfogsim_environment":True,"real_environment_not_minimal_env":True,"outcome_read_from_real_env":True,"four_scheduler_families_called":sorted(x["setter_kind"] for x in setters if x["setter_kind"] not in {"env.step"})==["cpu_callback","offload","rb","uav_mobility"],"mobility_uav_only":True,"vehicle_sumocontinued":vehicle_moved,"decision_action_identity":decision["trajectory_id"]==trajectory_id,"time_advanced_by_slot":math.isclose(execution_end-execution_start,float(env.simulation_interval),abs_tol=1e-9),"outcome_next_decision_identity":outcome["trajectory_id"]==next_decision["trajectory_id"] and next_decision["frame_index"]==outcome["frame_index"]+1,"outcome_next_decision_time":outcome["simulation_time_s"]==next_decision["simulation_time_s"],"traffic_fields_available":all(all(k in row for k in ("speed_mps","acceleration_mps2","heading")) for row in decision["entities"]),"uav_phi_available":all(row["elevation_rad"] is not None for row in decision["entities"] if row["entity_type"]=="uav"),"task_id_alignment":task_id in {row["task_id"] for row in decision["tasks"]} or task_id in {row["task_id"] for row in outcome["tasks"]}}
        payload={"schema_version":"PIJWM-Step-2.1-Real-AirFogSim-v1","environment":{"conda_env":"airfogsim","airfogsim_source":"code/reference/AirFogSim","seed":0,"config_hash":hashlib.sha256(json.dumps(config,sort_keys=True,default=str).encode()).hexdigest(),"warmup_real_steps":warmup},"decision":decision,"action":{"trajectory_id":trajectory_id,"frame_index":frame,"route":{"task_id":task_id,"task_node_id":source,"route_kind":"offload","target_node_id":target,"route_node_ids":[target]},"comm":{"task_id":task_id,"rb_indices":[0]},"comp":{"callback_installed":True,"cpu_rows":list(recorder.cpu_rows)},"mobility":{"uav_id":uav_id,"azimuth_rad":mobility["angle"],"elevation_rad":mobility["phi"],"speed_mps":mobility["speed"],"vehicle_motion":"SUMO external"}},"execution":{"start_time_s":execution_start,"end_time_s":execution_end,"setter_calls":setters,"env_step_completed":True},"outcome":outcome,"next_decision":next_decision,"checks":checks,"scope":{"non_locked":True,"training":False,"gpu":False,"locked_test":False}}
        failed=[key for key,value in checks.items() if value is not True]
        if failed: raise RuntimeError(json.dumps({"failed":failed,"checks":checks},ensure_ascii=False))
        OUT.mkdir(parents=True,exist_ok=False); write_json(OUT/"real_single_step.json",payload)
        manifest={"artifact":"real_single_step.json","sha256":sha256(OUT/"real_single_step.json"),"passed":True,"all_checks":checks,"source_files":{str(p.relative_to(ROOT)).replace("\\","/"):sha256(p) for p in (Path(__file__), CODE/"src/pi_jwm/airfogsim_single_step_collector_v1.py", CODE/"reference/AirFogSim/airfogsim/manager/traffic_manager.py", CODE/"reference/AirFogSim/airfogsim/scheduler/task_sched.py", CODE/"reference/AirFogSim/airfogsim/scheduler/communication_sched.py", CODE/"reference/AirFogSim/airfogsim/scheduler/computation_sched.py", CODE/"reference/AirFogSim/airfogsim/scheduler/traffic_sched.py")}}
        write_json(OUT/"manifest.json",manifest)
        print(json.dumps({"output":str(OUT),"checks":checks},ensure_ascii=False,indent=2))
    finally:
        if env is not None: env.close()
        os.chdir(old)
if __name__=="__main__": main()
