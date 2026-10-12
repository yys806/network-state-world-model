"""Opt-in STEP 6.4N runtime adapter.

The formal AirFogSim implementation is deliberately untouched. This module
wraps one environment instance while a bounded diagnostic step is running.
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, Mapping, Sequence

V2_SCHEMA = "PI-JWM-STEP-6.4N-RUNTIME-V2"


def _flatten(collection: Mapping[str, Sequence[Any]]) -> list[Any]:
    return [task for tasks in collection.values() for task in tasks]


def _task_row(env: Any, task_id: str) -> Any | None:
    manager = getattr(env, "task_manager", None)
    if manager is None:
        return None
    for name in ("_waiting_to_return_tasks", "_returning_tasks"):
        for task in _flatten(getattr(manager, name, {})):
            if str(task.getTaskId()) == str(task_id):
                return task
    return None


def _value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    getter = {
        "task_id": "getTaskId", "current_node_id": "getCurrentNodeId",
        "return_destination_id": "getToReturnNodeId", "required_returned_size": "getReturnedSize",
        "task_cpu_work": "getTaskCPU", "computed_cpu_work": "getComputedSize",
    }.get(name)
    if getter:
        try:
            return getattr(row, getter)()
        except Exception:
            return default
    if name == "lifecycle":
        if bool(row.isReturning()):
            return "returning"
        if bool(row.isComputed()):
            return "waiting_to_return"
        if bool(row.isComputing()):
            return "computing"
    return default


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not result == result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{name}:NONFINITE")
    return result


def validate_joint_return(env: Any, task: Mapping[str, Any] | Any,
                          route: Sequence[str], rb_indices: Sequence[int],
                          wireless_relations: set[tuple[Any, Any]]) -> dict[str, Any]:
    """Validate a complete Return+RB action without mutating ``env``."""
    task_id = str(_value(task, "task_id"))
    waiting = _flatten(getattr(env.task_manager, '_waiting_to_return_tasks', {}))
    raw_lifecycle = str(_value(task, "lifecycle", "")).lower() if isinstance(task, Mapping) else (
        'waiting_to_return' if task in waiting else 'unknown')
    if raw_lifecycle != "waiting_to_return":
        raise ValueError("RETURN_TASK_NOT_WAITING_TO_RETURN")
    cpu = _finite(_value(task, "task_cpu_work"), "task_cpu_work")
    computed = _finite(_value(task, "computed_cpu_work"), "computed_cpu_work")
    if computed < cpu:
        raise ValueError("RETURN_TASK_NOT_EXACTLY_COMPUTED")
    source = str(_value(task, "current_node_id"))
    destination = str(_value(task, "return_destination_id"))
    required = _finite(_value(task, "required_returned_size"), "required_returned_size")
    if required <= 0 or source == destination:
        raise ValueError("RETURN_NOT_REQUIRED")
    route_tuple = tuple(str(x) for x in route)
    if route_tuple != (destination,) or source in route_tuple:
        raise ValueError("RETURN_ROUTE_NOT_SINGLE_HOP")
    links = {(str(a), str(b)) for a, b in wireless_relations}
    if (source, destination) not in links:
        raise ValueError("RETURN_WIRELESS_RELATION_UNSUPPORTED")
    existing_route = _value(task, "route_node_ids", ()) or ()
    if tuple(existing_route):
        raise ValueError("RETURN_ROUTE_ALREADY_SET")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in rb_indices):
        raise ValueError("RETURN_RB_SUPPORT_INVALID")
    values = tuple(int(x) for x in rb_indices)
    n_rb = int(getattr(getattr(env, "channel_manager", None), "n_RB", 0))
    if len(set(values)) != len(values) or any(x < 0 or x >= n_rb for x in values):
        raise ValueError("RETURN_RB_SUPPORT_INVALID")
    runtime_task = _task_row(env, task_id)
    if runtime_task is None or runtime_task not in waiting:
        raise ValueError("RETURN_RUNTIME_LIFECYCLE_CHANGED")
    node_type = getattr(env, "_getNodeTypeById", lambda _: "?")
    node_index = getattr(env, '_getNodeIdxById', lambda _: -1)
    if any(node_type(x) not in ('U','V','I') or node_index(x) is None or int(node_index(x)) < 0 for x in (source,destination)):
        raise ValueError('RETURN_REAL_NODE_MISSING')
    # Raw admission must still agree with the current runtime object.
    for name, expected in (('current_node_id',source),('return_destination_id',destination)):
        if str(_value(runtime_task,name)) != expected:
            raise ValueError('RETURN_RUNTIME_ENDPOINT_CHANGED')
    if float(_value(runtime_task,'required_returned_size')) != required or float(_value(runtime_task,'computed_cpu_work')) < float(_value(runtime_task,'task_cpu_work')):
        raise ValueError('RETURN_RUNTIME_DEMAND_OR_COMPUTE_CHANGED')
    return {"task_id": task_id, "source": source, "destination": destination,
            "route": route_tuple, "rb_indices": values, "required_returned_size": required,
            "channel_type": f"{node_type(source)}2{node_type(destination)}",
            "mode": "DYNAMIC_RETURN_V2", "future_truth_used": False}


def commit_joint_return(env: Any, task_id: str, route: Sequence[str], rb_indices: Sequence[int],
                        route_setter: Callable[[Any, str, list[str]], Any],
                        rb_setter: Callable[[Any, str, list[int]], Any]) -> dict[str, Any]:
    """Validate then attempt Route and RB setters; quarantine on partial failure."""
    if getattr(env, '_pi_jwm_v2_quarantined', False):
        raise ValueError('V2_ENV_QUARANTINED_NO_RETRY')
    task = _task_row(env, task_id)
    if task is None:
        raise ValueError("RETURN_RUNTIME_TASK_MISSING")
    links = set(getattr(env, "_pi_jwm_v2_wireless_relations", set()))
    validated = validate_joint_return(env, task, route, rb_indices, links)
    route_table = getattr(env, "task_return_routes", None)
    if not isinstance(route_table, dict):
        raise ValueError("RETURN_ROUTE_BUFFER_MISSING")
    if str(task_id) in route_table:
        raise ValueError('RETURN_ROUTE_ALREADY_PENDING')
    rb_table = getattr(env, 'activated_offloading_tasks_with_RB_Nos', None)
    if not isinstance(rb_table, dict) or str(task_id) in rb_table:
        raise ValueError('RETURN_RB_BUFFER_MISSING_OR_ALREADY_BOUND')
    try:
        route_setter(env, task_id, list(validated["route"]))
        rb_setter(env, task_id, list(validated["rb_indices"]))
        if route_table.get(str(task_id)) != list(validated['route']) or rb_table.get(str(task_id)) != list(validated['rb_indices']):
            raise ValueError('SETTER_WRITEBACK_MISMATCH')
    except Exception as exc:
        env._pi_jwm_v2_quarantined = True
        if hasattr(env, 'step'):
            def blocked_step(*args, **kwargs):
                raise RuntimeError('V2_ENV_QUARANTINED_NO_STEP_OR_RETRY')
            env.step = blocked_step
        raise RuntimeError(f"PARTIAL_SETTER_FAILURE:{type(exc).__name__}:{exc}") from exc
    return {**validated, "route_setter_committed": True, "rb_setter_committed": True,
            "rollback_on_failure": False}


@contextmanager
def stage_aware_return_tti(task_manager: Any):
    """Instance-local stage clock; preserve every native checkTasks branch.

    Only the TTI getter during checkTasks is substituted. Native Task getters
    outside that call keep their original meaning and the Input clock is never
    overwritten. No-service calls do not refresh Return activity.
    """
    original_check = task_manager.checkTasks
    original_start = task_manager.setTaskReturnRouteAndStartReturn
    if inspect.getsource(original_check).count('task_info.getLastTransmissionTime()') != 1:
        raise ValueError('TTI_SOURCE_CONTRACT_CHANGED')
    originals = {}
    evidence = {"schema": V2_SCHEMA, "mode": "STAGE_AWARE_RETURN_TTI_V2",
                "return_clocks": {}, "transmission_events": [], "refreshed_by_service_only": True}

    def watch(task, initialized):
        tid = str(task.getTaskId())
        if tid in evidence['return_clocks']:
            return
        evidence['return_clocks'][tid] = float(initialized)
        old = task.transmit_to_Node
        originals[id(task)] = (task, old, 'transmit_to_Node' in vars(task))
        def transmit(self, node_id, trans_data, current_time, fast_return=False):
            before = float(self.getLastReturnTime())
            remaining = max(0., float(self.getReturnedSize()) - float(self.getTransmittedSize()))
            source = str(self.getCurrentNodeId())
            result = old(node_id, trans_data, current_time, fast_return=fast_return)
            evidence['transmission_events'].append({'task_id':tid,'phase':'return',
                'source_id':source,'target_id':str(node_id),'time':float(current_time),
                'service_argument':float(trans_data),'remaining_before':remaining,
                'delivered_data':min(remaining,max(0.,float(trans_data))),
                'stage_or_hop_completed':bool(result), 'fast_return':bool(fast_return),
                'source':'native_Task.transmit_to_Node_argument_and_return'})
            if float(trans_data) > 0:
                evidence['return_clocks'][tid] = float(current_time)
            elif not fast_return:
                self._last_return_time = before
            return result
        task.transmit_to_Node = MethodType(transmit, task)

    def start(manager, task_id, route, current_time):
        if str(task_id) in evidence['return_clocks']:
            raise ValueError('REPEATED_ROUTE_CLOCK_RESET_FORBIDDEN')
        rows = [t for t in _flatten(manager._waiting_to_return_tasks) if str(t.getTaskId()) == str(task_id)]
        if len(rows) != 1:
            raise ValueError('RETURN_TASK_NOT_UNIQUELY_WAITING')
        result = original_start(task_id, route, current_time)
        watch(rows[0], current_time)
        return result

    def wrapped(manager, current_time):
        temporary = []
        for task in _flatten(getattr(manager, "_returning_tasks", {})):
            watch(task, task.getLastReturnTime())
            old = getattr(task, "getLastTransmissionTime")
            temporary.append((task, old, 'getLastTransmissionTime' in vars(task)))
            def stage_clock(self):
                return evidence['return_clocks'][str(self.getTaskId())]
            task.getLastTransmissionTime = MethodType(stage_clock, task)
        try:
            return original_check(current_time)
        finally:
            for task, old, had in temporary:
                if had: task.getLastTransmissionTime = old
                else: del task.getLastTransmissionTime
    had_check = 'checkTasks' in vars(task_manager)
    had_start = 'setTaskReturnRouteAndStartReturn' in vars(task_manager)
    task_manager.checkTasks = MethodType(wrapped, task_manager)
    task_manager.setTaskReturnRouteAndStartReturn = MethodType(start, task_manager)
    try:
        yield evidence
    finally:
        for task, old, had in originals.values():
            if had: task.transmit_to_Node = old
            else: del task.transmit_to_Node
        if had_check: task_manager.checkTasks = original_check
        else: del task_manager.checkTasks
        if had_start: task_manager.setTaskReturnRouteAndStartReturn = original_start
        else: del task_manager.setTaskReturnRouteAndStartReturn


@dataclass(frozen=True)
class DynamicReturnBirth:
    flow_id: str
    slot: int
    task_id: str
    source_index: int
    destination_index: int
    relation_index: int
    route: tuple[int, ...]
    total_data: float
    latent_initialization: str = "ZERO_COLD_START"


def dynamic_birth_metadata(*, flow_id: str, slot: int, task_id: str,
                           source_index: int, destination_index: int,
                           relation_index: int, route: Sequence[int], total_data: float) -> DynamicReturnBirth:
    if min(slot, source_index, destination_index, relation_index) < 0:
        raise ValueError("DYNAMIC_FLOW_INDEX_INVALID")
    if not route or int(route[-1]) != destination_index:
        raise ValueError("DYNAMIC_FLOW_ROUTE_DESTINATION_MISMATCH")
    if _finite(total_data, "total_data") <= 0:
        raise ValueError("DYNAMIC_FLOW_SIZE_INVALID")
    return DynamicReturnBirth(flow_id, slot, task_id, source_index, destination_index,
                              relation_index, tuple(int(x) for x in route), float(total_data))
