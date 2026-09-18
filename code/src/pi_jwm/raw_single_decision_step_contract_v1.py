"""Raw one-decision-step contract at the PI-JWM/AirFogSim boundary.

This module freezes observable fields, four action families, time semantics,
identity checks, and the simulator setter mapping. It is independent of the
world model, graph construction, loss, training, and planner.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CONTRACT_VERSION = "PIJWM-Raw-Single-Decision-Step-v1"
ACTION_SPACE = ("route", "comm", "comp", "mobility")


RAW_FIELD_SPECS = (
    {
        "path": "decision.trajectory_id",
        "phase": "decision",
        "unit": "identifier",
        "id_domain": "trajectory",
        "source": "collector trajectory context",
        "required": True,
    },
    {
        "path": "decision.frame_index",
        "phase": "decision",
        "unit": "decision_step_index",
        "id_domain": None,
        "source": "collector frame counter",
        "required": True,
    },
    {
        "path": "decision.decision_time_s",
        "phase": "decision",
        "unit": "s",
        "id_domain": None,
        "source": "AirFogSimEnv.simulation_time",
        "required": True,
    },
    {
        "path": "decision.slot_duration_s",
        "phase": "decision",
        "unit": "s",
        "id_domain": None,
        "source": "AirFogSimEnv.simulation_interval",
        "required": True,
    },
    {
        "path": "decision.entities[].entity_id",
        "phase": "decision",
        "unit": "identifier",
        "id_domain": "physical_entity",
        "source": "AirFogSim entity dictionaries",
        "required": True,
    },
    {
        "path": "decision.entities[].entity_type",
        "phase": "decision",
        "unit": "enum(vehicle,uav,rsu,cloud)",
        "id_domain": None,
        "source": "AirFogSim entity collection",
        "required": True,
    },
    {
        "path": "decision.entities[].position_m",
        "phase": "decision",
        "unit": "m",
        "id_domain": None,
        "source": "entity.getPosition / traffic_manager current infos",
        "required": True,
    },
    {
        "path": "decision.entities[].speed_mps",
        "phase": "decision",
        "unit": "m/s",
        "id_domain": None,
        "source": "traffic_manager vehicle/UAV current infos",
        "required": True,
    },
    {
        "path": "decision.entities[].acceleration_mps2",
        "phase": "decision",
        "unit": "m/s^2",
        "id_domain": None,
        "source": "traffic_manager vehicle/UAV current infos",
        "required": True,
    },
    {
        "path": "decision.entities[].azimuth_rad",
        "phase": "decision",
        "unit": "rad",
        "id_domain": None,
        "source": "traffic_manager vehicle/UAV current infos",
        "required": True,
    },
    {
        "path": "decision.entities[].elevation_rad",
        "phase": "decision",
        "unit": "rad",
        "id_domain": None,
        "source": "traffic_manager UAV current infos; null for non-UAV",
        "required": False,
    },
    {
        "path": "decision.entities[].vehicle_route_id",
        "phase": "decision",
        "unit": "identifier",
        "id_domain": "sumo_route",
        "source": "traffic_manager vehicle current infos; null for non-vehicle",
        "required": False,
    },
    {
        "path": "decision.channel_rows[]",
        "phase": "decision",
        "unit": "dB per RB",
        "id_domain": "directed_entity_pair+channel_type",
        "source": "channel_manager.getCSI",
        "required": True,
    },
    {
        "path": "decision.tasks[]",
        "phase": "decision",
        "unit": "AirFogSim native task/data/CPU units with explicit masks",
        "id_domain": "task",
        "source": "task_manager lifecycle collections and Task getters",
        "required": True,
    },
    {
        "path": "decision.node_cpu_capacity_per_s",
        "phase": "decision",
        "unit": "AirFogSim CPU-work-unit/s",
        "id_domain": "physical_entity",
        "source": "entity.getFogProfile()['cpu']",
        "required": True,
    },
    {
        "path": "decision.n_rb",
        "phase": "decision",
        "unit": "RB count",
        "id_domain": None,
        "source": "channel_manager.n_RB",
        "required": True,
    },
    {
        "path": "execution.setter_calls[]",
        "phase": "execution",
        "unit": "event",
        "id_domain": "task or UAV",
        "source": "PI-JWM AirFogSim action adapter receipts",
        "required": True,
    },
    {
        "path": "outcome.delivered_data_by_task",
        "phase": "outcome",
        "unit": "AirFogSim native data unit/slot",
        "id_domain": "task",
        "source": "direct wireless/wired execution rows",
        "required": True,
    },
    {
        "path": "outcome.served_cpu_work_by_task",
        "phase": "outcome",
        "unit": "AirFogSim CPU-work-unit/slot",
        "id_domain": "task",
        "source": "CPU callback ledger and Task.getComputedSize delta",
        "required": True,
    },
    {
        "path": "outcome.entities/tasks",
        "phase": "outcome",
        "unit": "same as decision fields",
        "id_domain": "physical_entity/task",
        "source": "post-env.step direct snapshot",
        "required": True,
    },
)


ACTION_FIELD_SPECS = {
    "route": {
        "fields": (
            "task_id",
            "task_node_id",
            "route_kind",
            "target_node_id",
            "route_node_ids",
        ),
        "interfaces": (
            "TaskScheduler.setTaskOffloading",
            "TaskScheduler.setTaskReturnRoute",
        ),
    },
    "comm": {
        "fields": ("task_id", "rb_indices"),
        "interfaces": ("CommunicationScheduler.setCommunicationWithRB",),
    },
    "comp": {
        "fields": ("task_id", "node_id", "allocated_cpu_per_s"),
        "interfaces": ("ComputationScheduler.setComputingCallBack",),
    },
    "mobility": {
        "fields": ("uav_id", "azimuth_rad", "elevation_rad", "speed_mps"),
        "interfaces": ("TrafficScheduler.setUAVMobilityPatterns",),
        "boundary": "UAV only; vehicle motion is advanced externally by SUMO",
    },
}


class ContractError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class EntityState:
    entity_id: str
    entity_type: str
    present: bool
    position_m: tuple[float, float, float]
    speed_mps: float = 0.0
    acceleration_mps2: float = 0.0
    azimuth_rad: float = 0.0
    elevation_rad: float | None = None
    vehicle_route_id: str | None = None


@dataclass(frozen=True)
class TaskState:
    task_id: str
    task_node_id: str
    current_node_id: str
    lifecycle: str
    route_node_ids: tuple[str, ...]
    return_destination_id: str | None
    arrival_time_s: float
    task_size: float
    task_cpu_work: float
    computed_cpu_work: float


@dataclass(frozen=True)
class DecisionSnapshot:
    trajectory_id: str
    frame_index: int
    decision_time_s: float
    slot_duration_s: float
    entities: tuple[EntityState, ...]
    tasks: tuple[TaskState, ...]
    n_rb: int
    node_cpu_capacity_per_s: Mapping[str, float]
    source_phases: Mapping[str, str]


@dataclass(frozen=True)
class RouteAction:
    task_id: str
    task_node_id: str
    route_kind: str
    target_node_id: str
    route_node_ids: tuple[str, ...]


@dataclass(frozen=True)
class CommAction:
    task_id: str
    rb_indices: tuple[int, ...]


@dataclass(frozen=True)
class CompAction:
    task_id: str
    node_id: str
    allocated_cpu_per_s: float


@dataclass(frozen=True)
class UavMobilityAction:
    uav_id: str
    azimuth_rad: float
    elevation_rad: float
    speed_mps: float


@dataclass(frozen=True)
class ActionBundle:
    trajectory_id: str
    frame_index: int
    decision_time_s: float
    route: tuple[RouteAction, ...]
    comm: tuple[CommAction, ...]
    comp: tuple[CompAction, ...]
    mobility: tuple[UavMobilityAction, ...]


@dataclass(frozen=True)
class SetterReceipt:
    setter_kind: str
    subject_id: str | None
    succeeded: bool


@dataclass(frozen=True)
class ExecutionReceipt:
    execution_start_time_s: float
    execution_end_time_s: float
    setter_calls: tuple[SetterReceipt, ...]
    env_step_called: bool
    env_step_completed: bool


@dataclass(frozen=True)
class OutcomeSnapshot:
    outcome_time_s: float
    entities: tuple[EntityState, ...]
    tasks: tuple[TaskState, ...]
    delivered_data_by_task: Mapping[str, float]
    served_cpu_work_by_task: Mapping[str, float]


@dataclass(frozen=True)
class SingleDecisionStep:
    decision: DecisionSnapshot
    action: ActionBundle
    execution: ExecutionReceipt
    outcome: OutcomeSnapshot
    next_decision: DecisionSnapshot


@dataclass(frozen=True)
class AirFogSimSchedulerClasses:
    task: type
    communication: type
    computation: type
    traffic: type
    source_files: Mapping[str, str]
    source_load_notes: tuple[str, ...]


def _finite(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError("nonfinite_value", f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ContractError("nonfinite_value", f"{field} must be finite")
    return result


def _index(rows: tuple[Any, ...], attr: str, kind: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        identifier = getattr(row, attr)
        if not isinstance(identifier, str) or not identifier:
            raise ContractError("invalid_identity", f"{kind} id must be non-empty")
        if identifier in result:
            raise ContractError("duplicate_identity", f"duplicate {kind}: {identifier}")
        result[identifier] = row
    return result


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)


def validate_single_decision_step(step: SingleDecisionStep) -> SingleDecisionStep:
    if not isinstance(step, SingleDecisionStep):
        raise TypeError("step must be SingleDecisionStep")
    decision = step.decision
    action = step.action

    for field, phase in decision.source_phases.items():
        if phase not in {"decision", "history"}:
            code = "same_slot_outcome_leak" if phase == "outcome" else "invalid_source_phase"
            raise ContractError(code, f"decision field {field} has source phase {phase}")

    if decision.slot_duration_s <= 0:
        raise ContractError("invalid_slot_duration", "slot duration must be positive")
    if decision.n_rb < 0:
        raise ContractError("invalid_rb_count", "n_rb must be nonnegative")
    if (
        action.trajectory_id != decision.trajectory_id
        or action.frame_index != decision.frame_index
        or not _close(action.decision_time_s, decision.decision_time_s)
    ):
        raise ContractError("action_decision_identity_mismatch", "action is not bound to Decision_t")

    entities = _index(decision.entities, "entity_id", "entity")
    tasks = _index(decision.tasks, "task_id", "task")
    present_entities = {key for key, row in entities.items() if row.present}
    for entity in entities.values():
        if entity.entity_type not in {"vehicle", "uav", "rsu", "cloud"}:
            raise ContractError("invalid_entity_type", entity.entity_type)
        if len(entity.position_m) != 3 or any(
            not math.isfinite(float(value)) for value in entity.position_m
        ):
            raise ContractError("invalid_position", entity.entity_id)

    route_actions = _index(action.route, "task_id", "route action")
    for row in route_actions.values():
        if row.task_id not in tasks:
            raise ContractError("unknown_task", row.task_id)
        if row.task_node_id != tasks[row.task_id].task_node_id:
            raise ContractError("task_owner_mismatch", row.task_id)
        if row.route_kind not in {"offload", "return"}:
            raise ContractError("invalid_route_kind", row.route_kind)
        if row.target_node_id not in present_entities:
            raise ContractError("route_target_absent", row.target_node_id)
        if not row.route_node_ids or row.route_node_ids[-1] != row.target_node_id:
            raise ContractError("route_target_mismatch", row.task_id)
        if any(node_id not in present_entities for node_id in row.route_node_ids):
            raise ContractError("route_node_absent", row.task_id)

    comm_actions = _index(action.comm, "task_id", "communication action")
    for row in comm_actions.values():
        if row.task_id not in tasks:
            raise ContractError("unknown_task", row.task_id)
        if len(set(row.rb_indices)) != len(row.rb_indices):
            raise ContractError("duplicate_rb", row.task_id)
        if any(isinstance(rb, bool) or rb < 0 or rb >= decision.n_rb for rb in row.rb_indices):
            raise ContractError("rb_out_of_range", row.task_id)

    comp_actions = _index(action.comp, "task_id", "compute action")
    allocated_by_node: dict[str, float] = {}
    for row in comp_actions.values():
        if row.task_id not in tasks:
            raise ContractError("unknown_task", row.task_id)
        if row.node_id not in present_entities:
            raise ContractError("compute_node_absent", row.node_id)
        allocation = _finite(row.allocated_cpu_per_s, "allocated_cpu_per_s")
        if allocation < 0:
            raise ContractError("negative_cpu_allocation", row.task_id)
        allocated_by_node[row.node_id] = allocated_by_node.get(row.node_id, 0.0) + allocation
    for node_id, total in allocated_by_node.items():
        capacity = _finite(
            decision.node_cpu_capacity_per_s.get(node_id),
            f"CPU capacity for {node_id}",
        )
        if total > capacity + 1e-12:
            raise ContractError("cpu_capacity_exceeded", node_id)

    mobility_actions = _index(action.mobility, "uav_id", "mobility action")
    present_uavs = {
        key
        for key, row in entities.items()
        if row.present and row.entity_type == "uav"
    }
    for row in mobility_actions.values():
        if row.uav_id not in entities or entities[row.uav_id].entity_type != "uav":
            raise ContractError("mobility_non_uav", row.uav_id)
        for field, value in (
            ("azimuth_rad", row.azimuth_rad),
            ("elevation_rad", row.elevation_rad),
            ("speed_mps", row.speed_mps),
        ):
            _finite(value, field)
        if row.speed_mps < 0:
            raise ContractError("negative_mobility_speed", row.uav_id)
    missing_uavs = present_uavs - set(mobility_actions)
    extra_uavs = set(mobility_actions) - present_uavs
    if missing_uavs:
        raise ContractError("missing_uav_mobility_action", str(sorted(missing_uavs)))
    if extra_uavs:
        raise ContractError("mobility_absent_uav", str(sorted(extra_uavs)))

    expected_receipts = [
        *("offload" if row.route_kind == "offload" else "return_route" for row in action.route),
        *("rb" for _ in action.comm),
        "cpu_callback",
        *("uav_mobility" for _ in action.mobility),
    ]
    actual_receipts = [row.setter_kind for row in step.execution.setter_calls]
    if actual_receipts != expected_receipts or not all(
        row.succeeded for row in step.execution.setter_calls
    ):
        raise ContractError("setter_receipt_mismatch", "setter calls do not match Action_t")
    if not step.execution.env_step_called or not step.execution.env_step_completed:
        raise ContractError("env_step_incomplete", "AirFogSim env.step was not completed")

    expected_next_time = decision.decision_time_s + decision.slot_duration_s
    if not _close(step.execution.execution_start_time_s, decision.decision_time_s):
        raise ContractError("execution_start_time_mismatch", "execution did not start at Decision_t")
    if not _close(step.execution.execution_end_time_s, expected_next_time):
        raise ContractError("execution_end_time_mismatch", "execution did not end at t+1")
    if not _close(step.outcome.outcome_time_s, expected_next_time):
        raise ContractError("outcome_time_mismatch", "Outcome_t is not at t+1")
    if not _close(step.next_decision.decision_time_s, expected_next_time):
        raise ContractError("next_decision_time_mismatch", "Decision_{t+1} time is wrong")
    if step.next_decision.frame_index != decision.frame_index + 1:
        raise ContractError("next_frame_index_mismatch", "Decision_{t+1} frame is wrong")
    if step.next_decision.trajectory_id != decision.trajectory_id:
        raise ContractError("next_trajectory_mismatch", "trajectory changed within one step")
    if (
        step.outcome.entities != step.next_decision.entities
        or step.outcome.tasks != step.next_decision.tasks
    ):
        raise ContractError(
            "outcome_next_decision_mismatch",
            "post-step state does not equal the next decision snapshot",
        )
    return step


_SCHEDULER_PACKAGE = "_pi_jwm_airfogsim_scheduler_source"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load AirFogSim scheduler source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_airfogsim_scheduler_classes_from_source(
    reference_root: str | Path,
) -> AirFogSimSchedulerClasses:
    root = Path(reference_root).resolve()
    scheduler_root = root / "airfogsim" / "scheduler"
    paths = {
        "base": scheduler_root / "base_sched.py",
        "task": scheduler_root / "task_sched.py",
        "communication": scheduler_root / "communication_sched.py",
        "computation": scheduler_root / "computation_sched.py",
        "traffic": scheduler_root / "traffic_sched.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing AirFogSim scheduler sources: {missing}")

    for name in tuple(sys.modules):
        if name == _SCHEDULER_PACKAGE or name.startswith(f"{_SCHEDULER_PACKAGE}."):
            sys.modules.pop(name, None)
    package = types.ModuleType(_SCHEDULER_PACKAGE)
    package.__path__ = [str(scheduler_root)]
    package.__package__ = _SCHEDULER_PACKAGE
    sys.modules[_SCHEDULER_PACKAGE] = package
    source_load_notes: list[str] = []
    temporary_modules: list[str] = []
    if importlib.util.find_spec("shapely") is None:
        shapely_module = types.ModuleType("shapely")
        geometry_module = types.ModuleType("shapely.geometry")

        class _UnavailableGeometry:
            def __init__(self, *_args, **_kwargs) -> None:
                raise ModuleNotFoundError(
                    "shapely is required for AirFogSim non-fly-zone geometry helpers"
                )

        geometry_module.LineString = _UnavailableGeometry
        geometry_module.Point = _UnavailableGeometry
        shapely_module.geometry = geometry_module
        sys.modules["shapely"] = shapely_module
        sys.modules["shapely.geometry"] = geometry_module
        temporary_modules.extend(("shapely.geometry", "shapely"))
        source_load_notes.append(
            "shapely unavailable: non-fly-zone geometry helpers remain unavailable; "
            "the UAV mobility setter does not use them"
        )

    try:
        _load_module(f"{_SCHEDULER_PACKAGE}.base_sched", paths["base"])
        modules = {
            key: _load_module(f"{_SCHEDULER_PACKAGE}.{path.stem}", path)
            for key, path in paths.items()
            if key != "base"
        }
    finally:
        for module_name in temporary_modules:
            sys.modules.pop(module_name, None)
    return AirFogSimSchedulerClasses(
        task=modules["task"].TaskScheduler,
        communication=modules["communication"].CommunicationScheduler,
        computation=modules["computation"].ComputationScheduler,
        traffic=modules["traffic"].TrafficScheduler,
        source_files={key: str(path) for key, path in paths.items() if key != "base"},
        source_load_notes=tuple(source_load_notes),
    )


def apply_action_bundle_to_airfogsim(
    env: Any,
    action: ActionBundle,
    *,
    task_scheduler: type,
    communication_scheduler: type,
    computation_scheduler: type,
    traffic_scheduler: type,
) -> ExecutionReceipt:
    """Apply all four action families through AirFogSim's real scheduler APIs."""

    started = float(env.simulation_time)
    receipts: list[SetterReceipt] = []
    for row in action.route:
        if row.route_kind == "offload":
            ok = task_scheduler.setTaskOffloading(
                env,
                row.task_node_id,
                row.task_id,
                row.target_node_id,
                route=list(row.route_node_ids),
            )
            if ok is not True:
                raise ContractError("airfogsim_offload_rejected", row.task_id)
            receipts.append(SetterReceipt("offload", row.task_id, True))
        elif row.route_kind == "return":
            task_scheduler.setTaskReturnRoute(env, row.task_id, list(row.route_node_ids))
            receipts.append(SetterReceipt("return_route", row.task_id, True))
        else:
            raise ContractError("invalid_route_kind", row.route_kind)

    for row in action.comm:
        communication_scheduler.setCommunicationWithRB(env, row.task_id, list(row.rb_indices))
        receipts.append(SetterReceipt("rb", row.task_id, True))

    allocations = {row.task_id: float(row.allocated_cpu_per_s) for row in action.comp}

    def cpu_callback(computing_tasks):
        active_task_ids = {
            str(task.getTaskId())
            for tasks in computing_tasks.values()
            for task in tasks
        }
        return {task_id: allocations.get(task_id, 0.0) for task_id in active_task_ids}

    cpu_callback.__name__ = "pi_jwm_raw_action_contract_cpu_callback_v1"
    computation_scheduler.setComputingCallBack(env, cpu_callback)
    receipts.append(SetterReceipt("cpu_callback", None, True))

    patterns = {
        row.uav_id: {
            "angle": float(row.azimuth_rad),
            "phi": float(row.elevation_rad),
            "speed": float(row.speed_mps),
        }
        for row in action.mobility
    }
    if patterns:
        traffic_scheduler.setUAVMobilityPatterns(env, patterns)
        receipts.extend(
            SetterReceipt("uav_mobility", row.uav_id, True) for row in action.mobility
        )

    env.step()
    return ExecutionReceipt(
        execution_start_time_s=started,
        execution_end_time_s=float(env.simulation_time),
        setter_calls=tuple(receipts),
        env_step_called=True,
        env_step_completed=True,
    )
