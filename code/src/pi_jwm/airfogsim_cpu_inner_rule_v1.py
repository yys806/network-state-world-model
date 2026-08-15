"""AirFogSim callback adapter for the PI-JWM CPU inner rule."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .cpu_inner_rule_v1 import (
    CPU_INNER_RULE_VERSION,
    CpuRuleDecision,
    CpuTaskDemand,
    allocate_work_conserving_cpu,
)


_SOURCE_PACKAGE = "_pi_jwm_airfogsim_source"


@dataclass(frozen=True)
class AirFogSimCpuDecision:
    allocations: dict[str, float]
    decision: CpuRuleDecision
    source_task_classes: tuple[str, ...]


def _load_source_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load AirFogSim source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_airfogsim_task_class_from_source(reference_root: str | Path):
    """Load AirFogSim's real Task class without importing its GUI-heavy package root.

    This helper is for interface verification when the complete optional AirFogSim
    visualization stack is unavailable. It executes the repository's actual
    ``enum_const.py``, ``mission.py``, and ``task.py`` sources.
    """

    root = Path(reference_root).resolve()
    package_root = root / "airfogsim"
    source_files = {
        "enum": package_root / "enum_const.py",
        "mission": package_root / "entities" / "mission.py",
        "task": package_root / "entities" / "task.py",
    }
    missing = [str(path) for path in source_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing AirFogSim Task source dependencies: {missing}")

    cached = sys.modules.get(f"{_SOURCE_PACKAGE}.entities.task")
    package = sys.modules.get(_SOURCE_PACKAGE)
    if cached is not None and package is not None:
        cached_root = getattr(package, "__pi_jwm_reference_root__", None)
        if cached_root != str(root):
            raise RuntimeError(
                "AirFogSim Task source loader already initialized from a different root"
            )
        return cached.Task

    root_module = types.ModuleType(_SOURCE_PACKAGE)
    root_module.__path__ = [str(package_root)]
    root_module.__package__ = _SOURCE_PACKAGE
    root_module.__pi_jwm_reference_root__ = str(root)
    entities_name = f"{_SOURCE_PACKAGE}.entities"
    entities_module = types.ModuleType(entities_name)
    entities_module.__path__ = [str(package_root / "entities")]
    entities_module.__package__ = entities_name
    sys.modules[_SOURCE_PACKAGE] = root_module
    sys.modules[entities_name] = entities_module

    try:
        _load_source_module(f"{_SOURCE_PACKAGE}.enum_const", source_files["enum"])
        _load_source_module(f"{entities_name}.mission", source_files["mission"])
        task_module = _load_source_module(f"{entities_name}.task", source_files["task"])
    except Exception:
        for module_name in tuple(sys.modules):
            if module_name == _SOURCE_PACKAGE or module_name.startswith(f"{_SOURCE_PACKAGE}."):
                sys.modules.pop(module_name, None)
        raise
    return task_module.Task


def _finite(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _canonical_task_class(task: Any) -> str:
    module = task.__class__.__module__
    if module == f"{_SOURCE_PACKAGE}.entities.task":
        module = "airfogsim.entities.task"
    return f"{module}.{task.__class__.__name__}"


def allocate_airfogsim_cpu(
    env: Any,
    computing_tasks: Mapping[str, Iterable[Any]],
    *,
    slot_seconds: float | None = None,
) -> AirFogSimCpuDecision:
    """Adapt AirFogSim ``TaskManager.computeTasks`` inputs to the pure rule."""

    if not isinstance(computing_tasks, Mapping):
        raise TypeError("computing_tasks must be a mapping from node_id to tasks")
    if slot_seconds is None:
        if not hasattr(env, "simulation_interval"):
            raise ValueError("slot_seconds is required when env has no simulation_interval")
        slot_seconds = getattr(env, "simulation_interval")

    capacities: dict[str, float] = {}
    demands: list[CpuTaskDemand] = []
    source_task_classes: set[str] = set()
    for raw_node_id in sorted(computing_tasks, key=str):
        if not isinstance(raw_node_id, str) or not raw_node_id.strip():
            raise ValueError("computing_tasks node_id must be a non-empty string")
        node_id = raw_node_id
        node = env._getNodeById(node_id)
        if node is None:
            raise ValueError(f"AirFogSim node not found: {node_id}")
        fog_profile = node.getFogProfile()
        if not isinstance(fog_profile, Mapping) or "cpu" not in fog_profile:
            raise ValueError(f"missing CPU capacity for AirFogSim node: {node_id}")
        capacities[node_id] = _finite(
            fog_profile["cpu"],
            field=f"CPU capacity for {node_id}",
        )

        for task in computing_tasks[node_id]:
            task_id = task.getTaskId()
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError("AirFogSim task_id must be a non-empty string")
            assigned_node_id = task.getAssignedTo()
            if assigned_node_id != node_id:
                raise ValueError(
                    f"assigned node mismatch for {task_id}: {assigned_node_id!r} != {node_id!r}"
                )
            current_node_id = task.getCurrentNodeId()
            if current_node_id != node_id:
                raise ValueError(
                    f"current node mismatch for {task_id}: {current_node_id!r} != {node_id!r}"
                )
            total_cpu = _finite(task.getTaskCPU(), field=f"total CPU for {task_id}")
            computed_cpu = _finite(
                task.getComputedSize(),
                field=f"computed CPU for {task_id}",
            )
            if total_cpu < 0.0 or computed_cpu < 0.0:
                raise ValueError(f"CPU work must be nonnegative for {task_id}")
            if computed_cpu > total_cpu + max(1e-12, abs(total_cpu) * 1e-12):
                raise ValueError(f"computed CPU exceeds total CPU for {task_id}")
            remaining_work = max(total_cpu - computed_cpu, 0.0)
            demands.append(CpuTaskDemand(task_id, node_id, remaining_work))
            source_task_classes.add(_canonical_task_class(task))

    decision = allocate_work_conserving_cpu(demands, capacities, slot_seconds)
    return AirFogSimCpuDecision(
        allocations=decision.as_allocation_dict(),
        decision=decision,
        source_task_classes=tuple(sorted(source_task_classes)),
    )


def make_airfogsim_cpu_callback(
    env: Any,
    *,
    slot_seconds: float | None = None,
) -> Callable[[Mapping[str, Iterable[Any]]], dict[str, float]]:
    """Build the callback consumed by ``TaskManager.computeTasks``."""

    def callback(computing_tasks: Mapping[str, Iterable[Any]]) -> dict[str, float]:
        return allocate_airfogsim_cpu(
            env,
            computing_tasks,
            slot_seconds=slot_seconds,
        ).allocations

    callback.__name__ = "pi_jwm_cpu_inner_rule_v1_callback"
    callback.rule_version = CPU_INNER_RULE_VERSION  # type: ignore[attr-defined]
    return callback
