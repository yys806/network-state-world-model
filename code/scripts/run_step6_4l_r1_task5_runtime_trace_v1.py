"""STEP 6.4L-R1 bounded runtime trace for the real Task_5 return-route failure.

This is an audit-only wrapper around the existing AirFogSim collector.  It does
not alter a return value or simulator rule; it records method order, task
manager buckets, communication activation and the official failure code.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import MethodType

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts"),
               str(ROOT / "code/reference/AirFogSim/examples")]
OUT = ROOT / "code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v1"
SIM_SEED, POLICY_SEED, TARGET_FRAME = 2026092318, 2026092418, 6


class StopCapture(BaseException):
    pass


def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                              indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def task_snapshot(task):
    if task is None:
        return None
    def call(name, default=None):
        try:
            return getattr(task, name)()
        except Exception:
            return default
    return {
        "task_id": call("getTaskId"),
        "task_node_id": call("getTaskNodeId"),
        "current_node_id": call("getCurrentNodeId"),
        "assigned_to": call("getAssignedTo"),
        "lifecycle_flags": {
            "computed": call("isComputed"), "returning": call("isReturning"),
            "transmitting": call("isTransmitting"), "finished": call("isFinished"),
        },
        "route_remaining": list(call("getToOffloadRoute", []) or []),
        "return_destination": call("getToReturnNodeId"),
        "required_returned_size": call("getReturnedSize"),
        "transmitted_size": call("getTransmittedSize"),
        "task_deadline": call("getTaskDeadline"),
        "arrival_time": call("getTaskArrivalTime"),
        "last_transmission_time": call("getLastTransmissionTime"),
        "last_return_time": call("getLastReturnTime"),
        "failure_reason": call("getTaskFailureReason"),
        "failure_reason_code": getattr(task, "_failure_reason_code", None),
    }


def bucket_ids(manager):
    names = ("_waiting_to_return_tasks", "_returning_tasks", "_offloading_tasks",
             "_computing_tasks", "_done_tasks", "_out_of_ddl_tasks")
    out = {}
    for name in names:
        rows = []
        for node_id, tasks in getattr(manager, name, {}).items():
            rows.extend({"node_id": str(node_id), "task_id": str(t.getTaskId())}
                        for t in tasks)
        out[name] = sorted(rows, key=lambda x: (x["node_id"], x["task_id"]))
    return out


def task5(env):
    for obj in env.task_manager.getAllTasks():
        if str(obj.getTaskId()) == "Task_5":
            return obj
    return None


def node_types(env, src="UAV_0", dst="UAV_1"):
    return {"source": src, "destination": dst,
            "source_type": env._getNodeTypeById(src),
            "destination_type": env._getNodeTypeById(dst),
            "source_index": env._getNodeIdxById(src),
            "destination_index": env._getNodeIdxById(dst)}


def install_trace_wrappers(env, events):
    manager = env.task_manager
    original_update_task = env._updateTask
    original_update_wireless = env._updateWirelessCommunication
    original_allocate = env._allocate_communication_RBs
    original_start_return = manager.setTaskReturnRouteAndStartReturn
    original_fail = manager.failOffloadingTask

    def update_task(self):
        events.append({"event": "_updateTask.before", "time": float(self.simulation_time),
                       "task": task_snapshot(task5(self)), "buckets": bucket_ids(self.task_manager),
                       "route_table": dict(self.task_return_routes)})
        result = original_update_task()
        events.append({"event": "_updateTask.after", "time": float(self.simulation_time),
                       "task": task_snapshot(task5(self)), "buckets": bucket_ids(self.task_manager),
                       "route_table": dict(self.task_return_routes)})
        return result

    def update_wireless(self):
        events.append({"event": "_updateWirelessCommunication.before",
                       "time": float(self.simulation_time),
                       "task": task_snapshot(task5(self)),
                       "activated_request": dict(self.activated_offloading_tasks_with_RB_Nos)})
        result = original_update_wireless()
        events.append({"event": "_updateWirelessCommunication.after",
                       "time": float(self.simulation_time),
                       "task": task_snapshot(task5(self)),
                       "transfer_events": len(getattr(self, "pi_jwm_transfer_events", []))})
        return result

    def allocate(self, requested):
        out = original_allocate(requested)
        events.append({"event": "_allocate_communication_RBs",
                       "time": float(self.simulation_time), "requested": dict(requested),
                       "activated": {str(k): {kk: vv for kk, vv in v.items() if kk != "task"}
                                      for k, v in out.items()}})
        return out

    def start_return(self, task_id, route, current_time):
        events.append({"event": "setTaskReturnRouteAndStartReturn.before",
                       "time": float(current_time), "task_id": str(task_id),
                       "route": list(route), "buckets": bucket_ids(self),
                       "task": task_snapshot(next((t for t in self.getAllTasks()
                                                    if str(t.getTaskId()) == str(task_id)), None))})
        result = original_start_return(task_id, route, current_time)
        events.append({"event": "setTaskReturnRouteAndStartReturn.after",
                       "time": float(current_time), "task_id": str(task_id),
                       "route": list(route), "buckets": bucket_ids(self),
                       "task": task_snapshot(next((t for t in self.getAllTasks()
                                                    if str(t.getTaskId()) == str(task_id)), None))})
        return result

    def fail(self, task, failure=0):
        events.append({"event": "failOffloadingTask.before", "time": float(env.simulation_time),
                       "failure_argument": failure, "task": task_snapshot(task),
                       "buckets": bucket_ids(self)})
        result = original_fail(task, failure)
        events.append({"event": "failOffloadingTask.after", "time": float(env.simulation_time),
                       "failure_argument": failure, "result": result,
                       "task": task_snapshot(task), "buckets": bucket_ids(self)})
        return result

    env._updateTask = MethodType(update_task, env)
    env._updateWirelessCommunication = MethodType(update_wireless, env)
    env._allocate_communication_RBs = MethodType(allocate, env)
    manager.setTaskReturnRouteAndStartReturn = MethodType(start_return, manager)
    manager.failOffloadingTask = MethodType(fail, manager)


def main():
    import collect_step5_5_formal_raw_v1 as collector
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "PI-JWM-STEP-6.4L-R1-TASK5-RUNTIME-v1",
               "simulator_seed": SIM_SEED, "policy_seed": POLICY_SEED,
               "target_frame": TARGET_FRAME, "GPU": False, "locked_test": False,
               "formal_semantics_modified": False, "events": []}
    atomic(OUT / "00_start.json", receipt)

    def hook(env, decisions, steps, config, communication):
        frame = int(decisions[-1]["frame_index"])
        if frame == TARGET_FRAME:
            install_trace_wrappers(env, receipt["events"])
            obj = task5(env)
            before = {"task": task_snapshot(obj), "buckets": bucket_ids(env.task_manager),
                      "route_table": dict(env.task_return_routes),
                      "nodes": node_types(env),
                      "wireless_hop_before": bool(collector.step24._wireless_hop(env, obj))}
            route = [str(obj.getToReturnNodeId())]
            collector.step23.TaskScheduler.setTaskReturnRoute(env, "Task_5", route)
            after_setter = {"task": task_snapshot(obj), "buckets": bucket_ids(env.task_manager),
                            "route_table": dict(env.task_return_routes), "route": route,
                            "setter": "TaskScheduler.setTaskReturnRoute"}
            env._pi_jwm_diagnostic_disable_route_policy = True
            env._pi_jwm_diagnostic_route_entries = [{
                "task_id": "Task_5", "task_node_id": str(obj.getTaskNodeId()),
                "route_kind": "return", "target_node_id": route[-1],
                "route_node_ids": route}]
            receipt["root"] = {"frame": frame, "simulation_time_s": float(env.simulation_time),
                               "capture_event_id": decisions[-1]["capture_event_id"],
                               "before": before, "after_setter": after_setter}
            atomic(OUT / "01_root_and_setter.json", receipt["root"])
        elif frame == TARGET_FRAME + 1:
            obj = task5(env)
            receipt["post_step"] = {"frame": frame, "simulation_time_s": float(env.simulation_time),
                                    "capture_event_id": decisions[-1]["capture_event_id"],
                                    "task": task_snapshot(obj),
                                    "buckets": bucket_ids(env.task_manager),
                                    "route_table": dict(env.task_return_routes),
                                    "transfer_events": [dict(x) for x in env.pi_jwm_transfer_events],
                                    "task_manager_source": str(Path(collector.step23.task_manager.__file__).resolve())
                                    if hasattr(collector.step23, "task_manager") else None}
            atomic(OUT / "02_post_step.json", receipt["post_step"])
            atomic(OUT / "03_runtime_events.json", receipt["events"])
            receipt["elapsed_seconds"] = time.perf_counter() - started
            receipt["status"] = "COMPLETED"
            atomic(OUT / "04_acceptance_receipt.json", receipt)
            raise StopCapture()

    try:
        collector.collect_trajectory(SIM_SEED, POLICY_SEED, "step6_4l_r1_task5", on_decision=hook)
    except StopCapture:
        pass
    except BaseException as exc:
        receipt["status"] = "FAILED"
        receipt["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        receipt["elapsed_seconds"] = time.perf_counter() - started
        atomic(OUT / "failure_receipt.json", receipt)
        raise


if __name__ == "__main__":
    main()
