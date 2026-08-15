"""CPU-only P2 preflight runner for the PI-JWM full dual-graph collector.

The publication helpers in this module deliberately separate test payloads from
real AirFogSim evidence.  A canonical run may only publish the fixed natural
episode and fixture matrices; it never promotes dataset, training, GPU, locked
test, planner, or final-method status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
SRC_ROOT = CODE_ROOT / "src"
REFERENCE_ROOT = CODE_ROOT / "reference" / "AirFogSim"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.full_dual_graph_artifact_v1 import (  # noqa: E402
    REQUIRED_ARTIFACT_FILES,
    build_e1_rows,
    build_full_collector_status_flags,
    compare_replays,
    publish_atomic_bundle,
    validate_trajectory_frames,
)
from pi_jwm.airfogsim_full_dual_graph_collector_v1 import (  # noqa: E402
    execute_full_collector_step,
)
from pi_jwm.airfogsim_full_dual_graph_frame_builder_v1 import (  # noqa: E402
    build_frame_decision,
)
from pi_jwm.airfogsim_full_dual_graph_observer_v1 import (  # noqa: E402
    AirFogSimSnapshot,
    observe_airfogsim_snapshot,
)
from pi_jwm.full_dual_graph_collector_contract_v1 import SnapshotPhase  # noqa: E402
from pi_jwm.full_dual_graph_coverage_v1 import choose_resource_arm  # noqa: E402
from pi_jwm.full_dual_graph_vocabulary_v1 import (  # noqa: E402
    FullTrajectoryVocabulary,
    RouteRevisionLedger,
)
from pi_jwm.path_compat import (  # noqa: E402
    load_exact_mapping,
    load_source_changes,
    resolve_repository_path,
)

import run_p2_single_step_collector_preflight_v1 as single_step_runner  # noqa: E402


CANONICAL_SEEDS = (0, 1, 2)
NATURAL_ARMS = ("orthogonal", "interference_reuse")
REQUIRED_FIXTURES = (
    "multi_task_multi_flow",
    "cross_transmitter_rb_reuse",
    "wired_flow",
    "local_execution",
    "multihop_offload",
    "multihop_return",
    "node_disappearance_reappearance",
    "route_interruption",
    "deadline_failure",
    "tti_failure",
)
DEFAULT_OUTPUT_DIR = (
    CODE_ROOT / "artifacts" / "preflight" / "pi_jwm_p2_full_dual_graph_collector_v1"
)
_CANONICAL_PI_JWM_RUNTIME_PATHS = (
    Path(__file__).resolve(),
    CODE_ROOT / "scripts" / "run_p2_single_step_collector_preflight_v1.py",
    CODE_ROOT / "scripts" / "small_experiments" / "airfogsim_strict_dual_graph_preflight.py",
    SRC_ROOT / "pi_jwm" / "full_dual_graph_collector_contract_v1.py",
    SRC_ROOT / "pi_jwm" / "full_dual_graph_vocabulary_v1.py",
    SRC_ROOT / "pi_jwm" / "full_dual_graph_coverage_v1.py",
    SRC_ROOT / "pi_jwm" / "airfogsim_full_dual_graph_observer_v1.py",
    SRC_ROOT / "pi_jwm" / "airfogsim_full_dual_graph_frame_builder_v1.py",
    SRC_ROOT / "pi_jwm" / "airfogsim_full_dual_graph_collector_v1.py",
    SRC_ROOT / "pi_jwm" / "full_dual_graph_artifact_v1.py",
    SRC_ROOT / "pi_jwm" / "airfogsim_contract_adapter.py",
    SRC_ROOT / "pi_jwm" / "airfogsim_single_step_collector_v1.py",
    SRC_ROOT / "pi_jwm" / "airfogsim_cpu_inner_rule_v1.py",
    SRC_ROOT / "pi_jwm" / "cpu_inner_rule_v1.py",
    SRC_ROOT / "pi_jwm" / "information_edge_contract_v4.py",
    SRC_ROOT / "pi_jwm" / "single_step_collector_contract_v1.py",
)
_CANONICAL_TEST_PATHS = (
    CODE_ROOT / "tests" / "test_full_dual_graph_collector_contract_v1.py",
    CODE_ROOT / "tests" / "test_full_dual_graph_vocabulary_v1.py",
    CODE_ROOT / "tests" / "test_full_dual_graph_coverage_v1.py",
    CODE_ROOT / "tests" / "test_airfogsim_full_dual_graph_observer_v1.py",
    CODE_ROOT / "tests" / "test_airfogsim_full_dual_graph_frame_builder_v1.py",
    CODE_ROOT / "tests" / "test_airfogsim_full_dual_graph_collector_v1.py",
    CODE_ROOT / "tests" / "test_full_dual_graph_artifact_v1.py",
    CODE_ROOT / "tests" / "test_run_p2_full_dual_graph_collector_preflight_v1.py",
    CODE_ROOT / "tests" / "test_information_edge_contract_v4.py",
    CODE_ROOT / "tests" / "test_single_step_collector_contract_v1.py",
    CODE_ROOT / "tests" / "test_airfogsim_single_step_collector_v1.py",
    CODE_ROOT / "tests" / "test_run_p2_single_step_collector_preflight_v1.py",
    CODE_ROOT / "tests" / "test_multistep_collector_contract_v1.py",
    CODE_ROOT / "tests" / "test_run_p2_multistep_collector_preflight_v1.py",
    CODE_ROOT / "tests" / "small_experiments" / "test_airfogsim_strict_dual_graph_preflight.py",
)
_CANONICAL_DESIGN_PATHS = (
    PROJECT_ROOT / "记录" / "研究进展" / "2026-08-13-PI-JWM-v4全双图采集器设计.md",
)
_CANONICAL_REFERENCE_CONFIG_PATHS = (
    REFERENCE_ROOT / "examples" / "config.yaml",
    REFERENCE_ROOT / "examples" / "sumo_wujiaochang" / "osm.net.xml",
    REFERENCE_ROOT / "examples" / "sumo_wujiaochang" / "osm.sumocfg",
)
_CANONICAL_AIRFOGSIM_SOURCE_PATHS = tuple(
    sorted((REFERENCE_ROOT / "airfogsim").rglob("*.py"), key=lambda path: path.as_posix())
)
CANONICAL_SOURCE_PATHS = tuple(
    dict.fromkeys(
        _CANONICAL_PI_JWM_RUNTIME_PATHS
        + _CANONICAL_TEST_PATHS
        + _CANONICAL_DESIGN_PATHS
        + _CANONICAL_REFERENCE_CONFIG_PATHS
        + _CANONICAL_AIRFOGSIM_SOURCE_PATHS
    )
)


def _trajectory_id_for_arm(seed: int, arm: str) -> str:
    for ordinal in range(10_000):
        trajectory_id = f"natural-seed-{seed}-{arm}-{ordinal}"
        if choose_resource_arm(trajectory_id, seed) == arm:
            return trajectory_id
    raise RuntimeError(f"could not construct deterministic trajectory id for {arm}")


def canonical_episode_specs(seeds: Sequence[int]) -> list[dict[str, object]]:
    return [
        {
            "episode_id": f"natural-seed-{int(seed)}-{arm}",
            "trajectory_id": _trajectory_id_for_arm(int(seed), arm),
            "seed": int(seed),
            "arm": arm,
            "fixture": False,
            "training_eligible": False,
        }
        for seed in seeds
        for arm in NATURAL_ARMS
    ]


def validate_run_request(
    seeds: Sequence[int],
    steps: int,
    traffic_interval: float,
    simulation_interval: float,
) -> None:
    if tuple(seeds) != CANONICAL_SEEDS:
        raise ValueError(f"canonical seeds must be exactly {CANONICAL_SEEDS}")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 20:
        raise ValueError("natural episodes require at least 20 real steps")
    if traffic_interval <= 0.0 or simulation_interval <= 0.0:
        raise ValueError("traffic and simulation intervals must be positive")
    ratio = float(traffic_interval) / float(simulation_interval)
    if not math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("traffic/simulation interval ratio must equal 1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or verify the CPU-only PI-JWM full collector preflight"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CANONICAL_SEEDS))
    parser.add_argument("--steps", type=int, default=20)
    return parser


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def _runtime_inputs(env, snapshot: AirFogSimSnapshot) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    node_cpu: dict[str, float] = {}
    for node in snapshot.nodes:
        runtime_node = env._getNodeById(node.node_id)
        if runtime_node is None:
            continue
        profile = runtime_node.getFogProfile()
        node_cpu[node.node_id] = float(profile.get("cpu", 0.0))
    node_distance = {
        (edge.source_id, edge.target_id): float(
            env.getDistanceBetweenNodesById(edge.source_id, edge.target_id)
        )
        for edge in snapshot.physical_edges
        if edge.edge_type != "wired"
    }
    return node_cpu, node_distance


def _snapshot_payload(
    snapshot: AirFogSimSnapshot,
    *,
    node_indices: Mapping[str, int],
    physical_edge_indices: Mapping[str, int],
) -> dict[str, object]:
    payload = _jsonable(snapshot)
    assert isinstance(payload, dict)
    node_ids = {node.node_id for node in snapshot.nodes if node.present}
    edge_ids = {edge.edge_id for edge in snapshot.physical_edges if edge.present}
    payload["node_presence"] = [
        node_id in node_ids
        for node_id, _ in sorted(node_indices.items(), key=lambda item: item[1])
    ]
    payload["physical_edge_presence"] = [
        edge_id in edge_ids
        for edge_id, _ in sorted(physical_edge_indices.items(), key=lambda item: item[1])
    ]
    return payload


def _vocabulary_payload(vocabulary: FullTrajectoryVocabulary) -> dict[str, object]:
    payload = _jsonable(vocabulary.snapshot())
    assert isinstance(payload, dict)
    return payload


def _frame_payloads(results, vocabulary: FullTrajectoryVocabulary) -> list[dict[str, object]]:
    vocab = vocabulary.snapshot()
    physical_edge_ids = tuple(
        edge_id
        for edge_id, _ in sorted(
            vocab.physical_edge_indices.items(), key=lambda item: item[1]
        )
    )
    frames: list[dict[str, object]] = []
    previous_transfer_rows: Sequence[Mapping[str, object]] | None = None
    for result in results:
        if result.execution_snapshot is None or result.outcome_snapshot is None or result.action is None:
            raise RuntimeError("quarantined or incomplete result cannot become a natural frame")
        decision = _snapshot_payload(
            result.decision_snapshot,
            node_indices=vocab.node_indices,
            physical_edge_indices=vocab.physical_edge_indices,
        )
        execution = _snapshot_payload(
            result.execution_snapshot,
            node_indices=vocab.node_indices,
            physical_edge_indices=vocab.physical_edge_indices,
        )
        outcome = _snapshot_payload(
            result.outcome_snapshot,
            node_indices=vocab.node_indices,
            physical_edge_indices=vocab.physical_edge_indices,
        )
        frame = {
            "trajectory_id": result.trajectory_id,
            "frame_index": result.frame_index,
            "fixture": False,
            "training_eligible": False,
            "quarantined": result.quarantined,
            "decision_snapshot": decision,
            "execution_snapshot": execution,
            "outcome_snapshot": outcome,
            "action": _jsonable(result.action),
            "lifecycle_rows": _jsonable(result.lifecycle_rows),
            "transfer_rows": _jsonable(result.transfer_rows),
            "cpu_rows": _jsonable(result.cpu_rows),
            "energy_rows": _jsonable(result.energy_rows),
            "temporal_trace": list(result.temporal_trace),
            "decision_input_source_phases": {
                "nodes": "decision",
                "tasks": "decision",
                "channel": "decision",
                "previous_transfer": "previous_outcome" if previous_transfer_rows is not None else "missing_history",
            },
            "e1_rows": build_e1_rows(
                decision_snapshot=decision,
                previous_transfer_rows=previous_transfer_rows,
                physical_edge_ids=physical_edge_ids,
            ),
        }
        frames.append(frame)
        previous_transfer_rows = result.transfer_rows
    return frames


def run_natural_episode(spec: Mapping[str, object], *, steps: int) -> dict[str, object]:
    """Run one fresh AirFogSim episode solely through the production collector."""

    seed = int(spec["seed"])
    trajectory_id = str(spec["trajectory_id"])
    expected_arm = str(spec["arm"])
    if choose_resource_arm(trajectory_id, seed) != expected_arm:
        raise ValueError("episode trajectory id does not select its declared arm")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")

    previous_cwd = Path.cwd()
    env = None
    vocabulary = FullTrajectoryVocabulary()
    route_revisions = RouteRevisionLedger()
    results = []
    naturally_ended = False
    try:
        os.chdir(single_step_runner.EXAMPLE_DIR)
        env, task_scheduler, communication_scheduler, computation_scheduler, config = (
            single_step_runner._build_environment(
                seed, max_time=max(3.0, (steps + 2) * 0.1)
            )
        )
        validate_run_request(CANONICAL_SEEDS, max(20, steps), env.traffic_interval, env.simulation_interval)
        for frame_index in range(steps):
            if env.isDone():
                naturally_ended = True
                break
            snapshot = observe_airfogsim_snapshot(env, phase=SnapshotPhase.DECISION)
            node_cpu, node_distance = _runtime_inputs(env, snapshot)
            built = build_frame_decision(
                snapshot,
                trajectory_id=trajectory_id,
                frame_index=frame_index,
                seed=seed,
                n_rb=int(env.channel_manager.n_RB),
                vocabulary=vocabulary,
                route_revisions=route_revisions,
                node_cpu=node_cpu,
                node_distance=node_distance,
            )
            if built.resource_policy != expected_arm:
                raise RuntimeError("frame builder resource arm changed within episode")
            try:
                result = execute_full_collector_step(
                    env,
                    built,
                    trajectory_id=trajectory_id,
                    task_scheduler=task_scheduler,
                    communication_scheduler=communication_scheduler,
                    computation_scheduler=computation_scheduler,
                )
            except Exception as exc:
                memberships: dict[str, list[str]] = {}
                for collection_name, _ in (
                    ("_to_generate_task_infos", "to_generate"),
                    ("_waiting_to_offload_tasks", "waiting_to_offload"),
                    ("_offloading_tasks", "offloading"),
                    ("_computing_tasks", "computing"),
                    ("_waiting_to_return_tasks", "waiting_to_return"),
                    ("_returning_tasks", "returning"),
                    ("_done_tasks", "done"),
                    ("_out_of_ddl_tasks", "failed"),
                ):
                    collection = getattr(env.task_manager, collection_name, {})
                    for task_rows in collection.values():
                        for task in task_rows:
                            memberships.setdefault(str(task.getTaskId()), []).append(
                                collection_name
                            )
                duplicates = {
                    task_id: locations
                    for task_id, locations in memberships.items()
                    if len(locations) > 1
                }
                duplicate_states: dict[str, object] = {}
                for task_id in duplicates:
                    task = env.task_manager.getTaskByTaskId(task_id)
                    if task is not None:
                        duplicate_states[task_id] = {
                            "assigned_to": task.getAssignedTo(),
                            "current_node_id": task.getCurrentNodeId(),
                            "route": list(task.getToOffloadRoute()),
                            "computed_size": float(task.getComputedSize()),
                            "task_cpu": float(task.getTaskCPU()),
                            "is_computed": bool(task.isComputed()),
                            "is_returning": bool(task.isReturning()),
                            "is_executed_locally": bool(task.isExecutedLocally()),
                            "action_decision": _jsonable(
                                next(
                                    (
                                        decision
                                        for decision in built.action.decisions
                                        if decision.task_id == task_id
                                    ),
                                    None,
                                )
                            ),
                        }
                raise RuntimeError(
                    f"natural episode execution failed at frame {frame_index}: "
                    f"{type(exc).__name__}: {exc}; duplicate_memberships={duplicates}; "
                    f"duplicate_states={duplicate_states}"
                ) from exc
            if result.quarantined or not result.stepped:
                raise RuntimeError(
                    f"natural episode quarantined at frame {frame_index}: "
                    f"{result.quarantine_reason}; trace={list(result.temporal_trace)}"
                )
            results.append(result)
        frames = _frame_payloads(results, vocabulary)
        vocab_payload = _vocabulary_payload(vocabulary)
        validation_errors = validate_trajectory_frames(
            frames, vocabulary=vocab_payload, fixture=False
        )
        return {
            "spec": dict(spec),
            "config": {
                "traffic_interval": float(env.traffic_interval),
                "simulation_interval": float(env.simulation_interval),
                "n_rb": int(env.channel_manager.n_RB),
                "source": "AirFogSim_preflight_config",
                "raw_config_hash": hashlib.sha256(
                    json.dumps(_jsonable(config), sort_keys=True).encode("utf-8")
                ).hexdigest(),
            },
            "frames": frames,
            "vocabulary": vocab_payload,
            "validation_errors": validation_errors,
            "real_airfogsim_step_count": len(results),
            "naturally_ended": naturally_ended,
        }
    finally:
        if env is not None:
            env.close()
        os.chdir(previous_cwd)


def run_natural_replay_pair(
    spec: Mapping[str, object], *, steps: int
) -> dict[str, object]:
    """Run two fresh episodes and retain the reference only after replay passes."""

    reference = run_natural_episode(spec, steps=steps)
    replay = run_natural_episode(spec, steps=steps)
    comparison = compare_replays(reference["frames"], replay["frames"])
    if reference["validation_errors"]:
        comparison["passed"] = False
        comparison.setdefault("validation_errors", []).extend(
            reference["validation_errors"]
        )
    if replay["validation_errors"]:
        comparison["passed"] = False
        comparison.setdefault("validation_errors", []).extend(
            replay["validation_errors"]
        )
    if (
        reference["real_airfogsim_step_count"] != replay["real_airfogsim_step_count"]
        or reference["naturally_ended"] != replay["naturally_ended"]
    ):
        comparison["passed"] = False
        comparison.setdefault("exact_mismatches", []).append(
            {
                "path": ["episode_termination"],
                "reference": {
                    "steps": reference["real_airfogsim_step_count"],
                    "naturally_ended": reference["naturally_ended"],
                },
                "replay": {
                    "steps": replay["real_airfogsim_step_count"],
                    "naturally_ended": replay["naturally_ended"],
                },
            }
        )
    return {"reference": reference, "replay": replay, "comparison": comparison}


def _build_fixture_environment(seed: int, *, wired: bool):
    yaml, env_class, task_sched, comm_sched, comp_sched, preflight = (
        single_step_runner._load_runtime()
    )
    config = yaml.safe_load(
        (single_step_runner.EXAMPLE_DIR / "config.yaml").read_text(encoding="utf-8")
    )
    config = preflight.build_preflight_config(config, seed, 3.0)
    if wired:
        config["wired"] = {
            "edges": [
                {
                    "u": "RSU_0",
                    "v": "cloudServer_4",
                    "capacity_mbps": 100.0,
                    "prop_ms": 1.0,
                    "bidirectional": True,
                }
            ]
        }
    single_step_runner.np.random.seed(seed)
    single_step_runner.random.seed(seed)
    env = env_class(config, interactive_mode=None)
    return env, task_sched, comm_sched, comp_sched, config


def _execute_fixture_frame(
    env,
    schedulers,
    *,
    trajectory_id: str,
    seed: int,
    frame_index: int,
    vocabulary: FullTrajectoryVocabulary,
    route_revisions: RouteRevisionLedger,
):
    snapshot = observe_airfogsim_snapshot(env, phase=SnapshotPhase.DECISION)
    node_cpu, node_distance = _runtime_inputs(env, snapshot)
    built = build_frame_decision(
        snapshot,
        trajectory_id=trajectory_id,
        frame_index=frame_index,
        seed=seed,
        n_rb=int(env.channel_manager.n_RB),
        vocabulary=vocabulary,
        route_revisions=route_revisions,
        node_cpu=node_cpu,
        node_distance=node_distance,
    )
    result = execute_full_collector_step(
        env,
        built,
        trajectory_id=trajectory_id,
        task_scheduler=schedulers[0],
        communication_scheduler=schedulers[1],
        computation_scheduler=schedulers[2],
    )
    if result.quarantined or not result.stepped:
        raise RuntimeError(
            "fixture collector step failed: "
            f"{result.quarantine_reason}; trace={list(result.temporal_trace)}"
        )
    return built, result


def _add_fixture_task(
    env,
    *,
    task_id: str,
    source_id: str,
    cpu: float = 2.0,
    size: float = 1.0,
    deadline: float = 10.0,
    returned_size: float = 0.0,
):
    from airfogsim.entities.task import Task

    task = Task(
        task_id=task_id,
        task_node_id=source_id,
        task_cpu=cpu,
        task_size=size,
        task_deadline=deadline,
        task_priority=1.0,
        task_arrival_time=float(env.simulation_time),
        required_returned_size=returned_size,
        to_return_node_id=source_id,
    )
    task.setGenerated()
    env.task_manager._generated_task_history.setdefault(source_id, []).append(task)
    graph = env.task_manager._task_dependencies.get(source_id)
    if graph is None:
        import networkx as nx

        graph = nx.DiGraph()
        env.task_manager._task_dependencies[source_id] = graph
    graph.add_node(task_id)
    return task


def _append_waiting(env, task) -> None:
    env.task_manager._waiting_to_offload_tasks.setdefault(
        str(task.getTaskNodeId()), []
    ).append(task)


def _wireless_fixture_nodes(env) -> tuple[list[str], list[str], list[str]]:
    sources = sorted([*env.vehicles, *env.UAVs])
    uavs = sorted(env.UAVs)
    rsus = sorted(env.RSUs)
    if len(sources) < 2 or not uavs or not rsus:
        raise RuntimeError("fixture environment lacks two wireless sources, a UAV, or an RSU")
    return sources, uavs, rsus


def _transient_node_gap(env, node_id: str):
    original_traffic = env._updateTraffics
    original_ai = env._updateAIModels
    state: dict[str, object] = {}

    def traffic_then_remove():
        result = original_traffic()
        if node_id not in env.UAVs:
            raise RuntimeError(f"transient fixture node disappeared before intervention: {node_id}")
        state["node"] = env.UAVs.pop(node_id)
        state["index"] = env.uav_ids_as_index.index(node_id)
        env.uav_ids_as_index.remove(node_id)
        return result

    def restore_then_ai():
        if "node" not in state:
            raise RuntimeError("transient fixture did not remove its node")
        env.UAVs[node_id] = state["node"]
        env.uav_ids_as_index.insert(int(state["index"]), node_id)
        state["restored"] = True
        return original_ai()

    env._updateTraffics = traffic_then_remove
    env._updateAIModels = restore_then_ai
    return original_traffic, original_ai, state


def _fixture_row(
    name: str,
    *,
    passed: bool,
    step_count: int,
    checks: Mapping[str, bool],
    intervention: str,
    evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        "fixture_name": name,
        "fixture": True,
        "training_eligible": False,
        "passed": bool(passed and checks and all(checks.values())),
        "evidence_kind": "real_airfogsim_fixture",
        "real_airfogsim_step_count": int(step_count),
        "checks": dict(checks),
        "controlled_initial_state_or_intervention": intervention,
        "evidence": _jsonable(evidence),
    }


def run_real_fixture(name: str, *, seed: int = 91) -> dict[str, object]:
    """Execute one nontraining fixture on real AirFogSim objects and steps."""

    if name not in REQUIRED_FIXTURES:
        raise ValueError(f"unknown fixture: {name}")
    previous_cwd = Path.cwd()
    env = None
    step_count = 0
    try:
        os.chdir(single_step_runner.EXAMPLE_DIR)
        wired = name == "wired_flow"
        env, task_scheduler, communication_scheduler, computation_scheduler, _ = (
            _build_fixture_environment(seed, wired=wired)
        )
        schedulers = (task_scheduler, communication_scheduler, computation_scheduler)
        vocabulary = FullTrajectoryVocabulary()
        revisions = RouteRevisionLedger()
        requested_arm = "interference_reuse" if name in {
            "cross_transmitter_rb_reuse",
            "tti_failure",
        } else "orthogonal"
        trajectory_id = _trajectory_id_for_arm(seed, requested_arm)
        _, bootstrap = _execute_fixture_frame(
            env,
            schedulers,
            trajectory_id=trajectory_id,
            seed=seed,
            frame_index=0,
            vocabulary=vocabulary,
            route_revisions=revisions,
        )
        step_count += 1
        sources, uavs, rsus = _wireless_fixture_nodes(env)

        if name in {"multi_task_multi_flow", "cross_transmitter_rb_reuse"}:
            for ordinal, source in enumerate(sources[:2]):
                _append_waiting(
                    env,
                    _add_fixture_task(
                        env,
                        task_id=f"fixture::{name}::{ordinal}",
                        source_id=source,
                        size=10.0,
                    ),
                )
            built, result = _execute_fixture_frame(
                env, schedulers, trajectory_id=trajectory_id, seed=seed,
                frame_index=1, vocabulary=vocabulary, route_revisions=revisions,
            )
            step_count += 1
            selected_hops = [
                hop for hop in built.action.hops
                if hop.flow_id in {flow.flow_id for flow in built.action.flows}
            ]
            rb_by_hop = {
                allocation.hop_id: allocation.rb_index
                for allocation in built.action.rb_allocations
            }
            reused = len(selected_hops) >= 2 and len(
                {rb_by_hop[hop.hop_id] for hop in selected_hops}
            ) < len(selected_hops)
            checks = {
                "two_distinct_real_tasks": len(built.action.flows) >= 2,
                "two_distinct_transmitters": len({hop.source_id for hop in selected_hops}) >= 2,
                "real_step_completed": result.stepped,
            }
            if name == "cross_transmitter_rb_reuse":
                checks["rb_reused_across_distinct_transmitters"] = reused
                checks["same_transmitter_not_reused"] = len(
                    {(hop.source_id, rb_by_hop[hop.hop_id]) for hop in selected_hops}
                ) == len(selected_hops)
            return _fixture_row(
                name, passed=True, step_count=step_count, checks=checks,
                intervention="two generated real Task objects inserted into waiting_to_offload",
                evidence={"action": built.action, "transfer_rows": result.transfer_rows},
            )

        if name == "local_execution":
            source = sources[0]
            injected = []
            for ordinal in range(3):
                task = _add_fixture_task(
                    env, task_id=f"fixture::local::{ordinal}", source_id=source,
                    cpu=50.0, size=10.0,
                )
                _append_waiting(env, task)
                injected.append(task.getTaskId())
            built, result = _execute_fixture_frame(
                env, schedulers, trajectory_id=trajectory_id, seed=seed,
                frame_index=1, vocabulary=vocabulary, route_revisions=revisions,
            )
            step_count += 1
            local = [
                decision for decision in built.action.decisions
                if decision.task_id in injected and decision.selected
                and decision.target_node_id == source
            ]
            checks = {
                "local_decision_exists": bool(local),
                "local_decision_has_no_flow": all(row.flow_id is None for row in local),
                "cpu_callback_observed": any(
                    any(task_id in injected for task_id in row["task_ids"])
                    for row in result.cpu_rows
                ),
            }
            return _fixture_row(
                name, passed=True, step_count=step_count, checks=checks,
                intervention="three waiting tasks expose the frame-ordinal local target family",
                evidence={"action": built.action, "cpu_rows": result.cpu_rows},
            )

        if name == "wired_flow":
            source, target = "RSU_0", "cloudServer_4"
            if source not in env.RSUs or target not in env.cloudServers:
                raise RuntimeError("configured wired fixture endpoints are absent")
            task = _add_fixture_task(
                env, task_id="fixture::wired", source_id=source, size=10_000_000.0
            )
            task.offloadTo(target, [target], float(env.simulation_time))
            env.task_manager._offloading_tasks.setdefault(source, []).append(task)
            built, result = _execute_fixture_frame(
                env, schedulers, trajectory_id=trajectory_id, seed=seed,
                frame_index=1, vocabulary=vocabulary, route_revisions=revisions,
            )
            step_count += 1
            rows = [row for row in result.transfer_rows if row["task_id"] == task.getTaskId()]
            checks = {
                "wired_hop_selected": any(hop.transport == "wired" for hop in built.action.hops),
                "no_rb_for_wired_hop": not built.action.rb_allocations,
                "wired_manager_direct_result": bool(rows) and all(
                    row["source_method"] == "wired_manager.step_direct_result" for row in rows
                ),
            }
            return _fixture_row(
                name, passed=True, step_count=step_count, checks=checks,
                intervention="real RSU-cloud wired edge plus one offloading Task",
                evidence={"action": built.action, "transfer_rows": rows},
            )

        if name in {"multihop_offload", "multihop_return"}:
            relay, compute = uavs[0], rsus[0]
            source = next(node_id for node_id in sources if node_id != relay)
            if len({source, relay, compute}) < 3:
                raise RuntimeError("multihop fixture endpoints are not distinct")
            task = _add_fixture_task(
                env,
                task_id=f"fixture::{name}",
                source_id=source,
                size=10.0,
                returned_size=10.0,
            )
            if name == "multihop_offload":
                task.offloadTo(compute, [relay, compute], float(env.simulation_time))
                env.task_manager._offloading_tasks.setdefault(source, []).append(task)
                expected_route = (relay, compute)
            else:
                task.setAttribute("_assigned_to", compute)
                task.setAttribute("_routes", [source, compute])
                task.setAttribute("_computed_size", float(task.getTaskCPU()))
                task.setAttribute("_to_offload_route", [relay, source])
                task.setAttribute("_start_to_transmit_time", float(env.simulation_time))
                task.setAttribute("_last_transmission_time", float(env.simulation_time))
                env.task_manager._returning_tasks.setdefault(compute, []).append(task)
                expected_route = (relay, source)
            built, result = _execute_fixture_frame(
                env, schedulers, trajectory_id=trajectory_id, seed=seed,
                frame_index=1, vocabulary=vocabulary, route_revisions=revisions,
            )
            step_count += 1
            decision = next(row for row in built.action.decisions if row.task_id == task.getTaskId())
            hop = next(row for row in built.action.hops if row.hop_id == decision.hop_id)
            checks = {
                "full_route_retained": decision.route_nodes == expected_route,
                "only_current_carrying_hop_executes": hop.target_id == relay,
                "per_hop_transfer_observed": any(
                    row["hop_id"] == hop.hop_id for row in result.transfer_rows
                ),
            }
            return _fixture_row(
                name, passed=True, step_count=step_count, checks=checks,
                intervention="real Task initialized with an explicit two-hop route",
                evidence={"action": built.action, "transfer_rows": result.transfer_rows},
            )

        if name in {"node_disappearance_reappearance", "route_interruption"}:
            source, target = sources[0], uavs[0]
            if source == target:
                source = sources[1]
            task = _add_fixture_task(
                env, task_id=f"fixture::{name}", source_id=source, size=10.0
            )
            task.offloadTo(target, [target], float(env.simulation_time))
            env.task_manager._offloading_tasks.setdefault(source, []).append(task)
            original_traffic, original_ai, state = _transient_node_gap(env, target)
            try:
                built, result = _execute_fixture_frame(
                    env, schedulers, trajectory_id=trajectory_id, seed=seed,
                    frame_index=1, vocabulary=vocabulary, route_revisions=revisions,
                )
            finally:
                env._updateTraffics = original_traffic
                env._updateAIModels = original_ai
                if target not in env.UAVs and "node" in state:
                    env.UAVs[target] = state["node"]
                    env.uav_ids_as_index.insert(int(state["index"]), target)
            step_count += 1
            execution_nodes = {node.node_id for node in result.execution_snapshot.nodes}
            outcome_nodes = {node.node_id for node in result.outcome_snapshot.nodes}
            decision = next(row for row in built.action.decisions if row.task_id == task.getTaskId())
            base_checks = {
                "present_at_decision": target in {node.node_id for node in result.decision_snapshot.nodes},
                "absent_at_execution_snapshot": target not in execution_nodes,
                "restored_before_runtime_updates": state.get("restored") is True,
                "present_at_outcome": target in outcome_nodes,
                "real_step_completed": result.stepped,
            }
            if name == "route_interruption":
                base_checks["decision_route_not_silently_rewritten"] = decision.route_nodes == (target,)
                base_checks["carrying_hop_identity_retained"] = any(
                    hop.hop_id == decision.hop_id for hop in built.action.hops
                )
            return _fixture_row(
                name, passed=True, step_count=step_count, checks=base_checks,
                intervention=(
                    "target UAV removed after real traffic update and restored before AI/task/communication updates; "
                    "tests collector snapshot/identity semantics, not a native mobility disappearance"
                ),
                evidence={
                    "target_node_id": target,
                    "action": built.action,
                    "execution_node_ids": sorted(execution_nodes),
                    "outcome_node_ids": sorted(outcome_nodes),
                },
            )

        if name == "deadline_failure":
            task = _add_fixture_task(
                env, task_id="fixture::deadline", source_id=sources[0],
                cpu=1000.0, size=1000.0, deadline=0.0,
            )
            task.setAttribute("_task_arrival_time", float(env.simulation_time) - 1.0)
            _append_waiting(env, task)
            built, result = _execute_fixture_frame(
                env, schedulers, trajectory_id=trajectory_id, seed=seed,
                frame_index=1, vocabulary=vocabulary, route_revisions=revisions,
            )
            step_count += 1
            outcome = next(row for row in result.outcome_snapshot.tasks if row.task_id == task.getTaskId())
            checks = {
                "decision_was_recorded": any(row.task_id == task.getTaskId() for row in built.action.decisions),
                "outcome_failed": outcome.lifecycle.value == "failed",
            }
            return _fixture_row(
                name, passed=True, step_count=step_count, checks=checks,
                intervention="waiting Task arrival/deadline set explicitly overdue before the real step",
                evidence={"action": built.action, "outcome_task": outcome},
            )

        if name == "tti_failure":
            source, target = sources[0], rsus[0]
            injected = []
            for ordinal in range(int(env.channel_manager.n_RB) + 1):
                task = _add_fixture_task(
                    env, task_id=f"fixture::tti::{ordinal:03d}", source_id=source,
                    size=1000.0,
                )
                task.offloadTo(target, [target], float(env.simulation_time))
                task.setAttribute("_last_transmission_time", float(env.simulation_time) - 1.0)
                env.task_manager._offloading_tasks.setdefault(source, []).append(task)
                injected.append(task.getTaskId())
            built, result = _execute_fixture_frame(
                env, schedulers, trajectory_id=trajectory_id, seed=seed,
                frame_index=1, vocabulary=vocabulary, route_revisions=revisions,
            )
            step_count += 1
            rejected = [
                row.task_id for row in built.action.decisions
                if row.task_id in injected and not row.selected and row.reason == "rb_budget_exhausted"
            ]
            failed = {
                row.task_id for row in result.outcome_snapshot.tasks
                if row.lifecycle.value == "failed"
            }
            checks = {
                "same_transmitter_rb_budget_exhausted": bool(rejected),
                "unserved_old_transmission_failed": bool(set(rejected) & failed),
                "failure_retained_in_outcome_snapshot": bool(failed & set(injected)),
            }
            return _fixture_row(
                name, passed=True, step_count=step_count, checks=checks,
                intervention="n_RB+1 offloading Tasks share one transmitter and have an expired last-transmission time",
                evidence={"action": built.action, "failed_task_ids": sorted(failed)},
            )

        raise AssertionError(name)
    finally:
        if env is not None:
            env.close()
        os.chdir(previous_cwd)


def run_fixture_matrix() -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for ordinal, name in enumerate(REQUIRED_FIXTURES):
        try:
            rows[name] = run_real_fixture(name, seed=91 + ordinal)
        except Exception as exc:
            rows[name] = _fixture_row(
                name,
                passed=False,
                step_count=0,
                checks={"fixture_execution": False},
                intervention="fixture aborted before acceptance",
                evidence={"error_type": type(exc).__name__, "error": str(exc)},
            )
    return rows


def build_real_preflight_payloads(
    seeds: Sequence[int] = CANONICAL_SEEDS, *, steps: int = 20
) -> dict[str, object]:
    validate_run_request(tuple(seeds), steps, 0.1, 0.1)
    frames: list[dict[str, object]] = []
    vocabularies: dict[str, object] = {}
    natural_rows: list[dict[str, object]] = []
    replay_rows: dict[str, object] = {}
    validation_errors: list[str] = []
    for spec in canonical_episode_specs(seeds):
        pair = run_natural_replay_pair(spec, steps=steps)
        reference = pair["reference"]
        comparison = pair["comparison"]
        episode_id = str(spec["episode_id"])
        trajectory_id = str(spec["trajectory_id"])
        frames.extend(reference["frames"])
        vocabularies[trajectory_id] = reference["vocabulary"]
        episode_passed = comparison["passed"] is True and not reference["validation_errors"]
        natural_rows.append(
            {
                **dict(spec),
                "real_airfogsim_step_count": reference["real_airfogsim_step_count"],
                "naturally_ended": reference["naturally_ended"],
                "passed": episode_passed,
                "evidence_kind": "real_airfogsim_natural_replay_pair",
                "config": reference["config"],
            }
        )
        replay_rows[episode_id] = {
            "episode_id": episode_id,
            "trajectory_id": trajectory_id,
            **comparison,
        }
        validation_errors.extend(
            f"{episode_id}: {error}" for error in reference["validation_errors"]
        )
        if not episode_passed:
            validation_errors.append(f"{episode_id}: replay or trajectory validation failed")

    fixtures = run_fixture_matrix()
    for name, row in fixtures.items():
        if row.get("passed") is not True:
            validation_errors.append(f"fixture {name} did not pass")
    passed = not validation_errors
    return {
        "collector_config.json": {
            "schema_version": "PIJWM-Full-Collector-Preflight-v1",
            "test_only": False,
            "seeds": list(seeds),
            "steps": steps,
            "traffic_interval": 0.1,
            "simulation_interval": 0.1,
            "natural_arms": list(NATURAL_ARMS),
            "natural_episode_count": len(natural_rows),
            "fixture_count": len(fixtures),
            "scope": "CPU-only collector preflight; nontraining; unlocked data only",
        },
        "vocabularies.json": vocabularies,
        "frames.jsonl": frames,
        "coverage_report.json": {
            "natural_episodes": natural_rows,
            "fixtures": fixtures,
            "natural_and_fixture_reports_separate": True,
        },
        "validation_report.json": {
            "passed": passed,
            "errors": validation_errors,
            "trajectory_count": len(natural_rows),
            "fixture_count": len(fixtures),
            "training_eligible": False,
        },
        "replay_report.json": {
            "passed": all(row["passed"] is True for row in replay_rows.values()),
            "episodes": replay_rows,
            "fresh_environment_per_reference_and_replay": True,
        },
        "status_flags.json": build_full_collector_status_flags(passed=passed),
    }


def _expected_payload_names() -> set[str]:
    return set(REQUIRED_ARTIFACT_FILES) - {"manifest.json"}


def _test_fixture_row(name: str) -> dict[str, object]:
    return {
        "fixture_name": name,
        "fixture": True,
        "training_eligible": False,
        "passed": True,
        "evidence_kind": "test_only_synthetic_payload",
        "real_airfogsim_step_count": 1,
    }


def fake_passing_payloads_for_test() -> dict[str, object]:
    """Return gate-shaped test data that is never accepted as real evidence."""

    natural = [
        {
            **spec,
            "real_airfogsim_step_count": 20,
            "naturally_ended": False,
            "passed": True,
            "evidence_kind": "test_only_synthetic_payload",
        }
        for spec in canonical_episode_specs(CANONICAL_SEEDS)
    ]
    return {
        "collector_config.json": {
            "schema_version": "PIJWM-Full-Collector-Preflight-v1",
            "test_only": True,
            "seeds": list(CANONICAL_SEEDS),
            "steps": 20,
            "traffic_interval": 0.1,
            "simulation_interval": 0.1,
            "natural_arms": list(NATURAL_ARMS),
        },
        "vocabularies.json": {},
        "frames.jsonl": [],
        "coverage_report.json": {
            "natural_episodes": natural,
            "fixtures": {name: _test_fixture_row(name) for name in REQUIRED_FIXTURES},
        },
        "validation_report.json": {"passed": True, "test_only": True, "errors": []},
        "replay_report.json": {"passed": True, "test_only": True, "episodes": []},
        "status_flags.json": build_full_collector_status_flags(passed=True),
    }


def validate_preflight_payloads(
    payloads: Mapping[str, object], *, allow_test_payload: bool = False
) -> list[str]:
    errors: list[str] = []
    if set(payloads) != _expected_payload_names():
        errors.append("artifact payload file matrix differs from the required contract")
        return errors

    config = payloads.get("collector_config.json")
    if not isinstance(config, Mapping):
        errors.append("collector config is missing")
        return errors
    test_only = config.get("test_only") is True
    if test_only and not allow_test_payload:
        errors.append("test-only payload cannot be published as real preflight evidence")
    try:
        validate_run_request(
            tuple(config.get("seeds", ())),
            config.get("steps"),
            float(config.get("traffic_interval", 0.0)),
            float(config.get("simulation_interval", 0.0)),
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"collector run request invalid: {exc}")
    if tuple(config.get("natural_arms", ())) != NATURAL_ARMS:
        errors.append("natural arm matrix differs from the canonical two-arm contract")

    coverage = payloads.get("coverage_report.json")
    if not isinstance(coverage, Mapping):
        errors.append("coverage report is missing")
        return errors
    natural = coverage.get("natural_episodes")
    if not isinstance(natural, list):
        errors.append("natural episode report is missing")
    else:
        observed = {
            (row.get("seed"), row.get("arm"))
            for row in natural
            if isinstance(row, Mapping)
        }
        expected = {(seed, arm) for seed in CANONICAL_SEEDS for arm in NATURAL_ARMS}
        if len(natural) != len(expected) or observed != expected:
            errors.append("natural episode matrix differs from canonical seeds and arms")
        for row in natural:
            if not isinstance(row, Mapping):
                errors.append("natural episode row is malformed")
                continue
            enough_steps = (
                isinstance(row.get("real_airfogsim_step_count"), int)
                and row.get("real_airfogsim_step_count") >= config.get("steps", 20)
            )
            if not enough_steps and row.get("naturally_ended") is not True:
                errors.append(f"natural episode {row.get('episode_id')} ended before its gate")
            if row.get("passed") is not True:
                errors.append(f"natural episode {row.get('episode_id')} did not pass")
            if row.get("fixture") is not False or row.get("training_eligible") is not False:
                errors.append(f"natural episode {row.get('episode_id')} has invalid scope flags")
            expected_evidence = (
                "test_only_synthetic_payload"
                if test_only
                else "real_airfogsim_natural_replay_pair"
            )
            if row.get("evidence_kind") != expected_evidence:
                errors.append(f"natural episode {row.get('episode_id')} evidence kind is invalid")

    fixtures = coverage.get("fixtures")
    if not isinstance(fixtures, Mapping) or set(fixtures) != set(REQUIRED_FIXTURES):
        errors.append("fixture matrix differs from the exact required fixture matrix")
    else:
        for name in REQUIRED_FIXTURES:
            row = fixtures[name]
            if not isinstance(row, Mapping):
                errors.append(f"fixture {name} row is malformed")
                continue
            if row.get("fixture_name") != name:
                errors.append(f"fixture {name} identity mismatch")
            if row.get("fixture") is not True or row.get("training_eligible") is not False:
                errors.append(f"fixture {name} has invalid scope flags")
            if row.get("passed") is not True:
                errors.append(f"fixture {name} did not pass")
            if not isinstance(row.get("real_airfogsim_step_count"), int) or row.get(
                "real_airfogsim_step_count"
            ) < 1:
                errors.append(f"fixture {name} has no real AirFogSim step evidence")
            expected_evidence = (
                "test_only_synthetic_payload" if test_only else "real_airfogsim_fixture"
            )
            if row.get("evidence_kind") != expected_evidence:
                errors.append(f"fixture {name} evidence kind is invalid")

    for report_name in ("validation_report.json", "replay_report.json"):
        report = payloads.get(report_name)
        if not isinstance(report, Mapping) or report.get("passed") is not True:
            errors.append(f"{report_name} did not pass")
    expected_flags = build_full_collector_status_flags(passed=True)
    if payloads.get("status_flags.json") != expected_flags:
        errors.append("status flags exceed or differ from the conservative preflight scope")

    if not test_only:
        vocabularies = payloads.get("vocabularies.json")
        frames = payloads.get("frames.jsonl")
        if not isinstance(vocabularies, Mapping) or not isinstance(frames, list):
            errors.append("real trajectory frames or vocabularies are malformed")
        else:
            grouped: dict[str, list[Mapping[str, object]]] = {}
            malformed_frame = False
            for frame in frames:
                if not isinstance(frame, Mapping) or not isinstance(frame.get("trajectory_id"), str):
                    malformed_frame = True
                    continue
                grouped.setdefault(str(frame["trajectory_id"]), []).append(frame)
            if malformed_frame:
                errors.append("real frames contain malformed trajectory identities")
            expected_trajectories = {
                str(row.get("trajectory_id"))
                for row in natural or []
                if isinstance(row, Mapping)
            }
            if set(grouped) != expected_trajectories or set(vocabularies) != expected_trajectories:
                errors.append("real frame/vocabulary trajectory matrix differs from coverage")
            else:
                for trajectory_id in sorted(expected_trajectories):
                    trajectory_frames = sorted(
                        grouped[trajectory_id], key=lambda row: row.get("frame_index", -1)
                    )
                    trajectory_errors = validate_trajectory_frames(
                        trajectory_frames,
                        vocabulary=vocabularies[trajectory_id],
                        fixture=False,
                    )
                    errors.extend(
                        f"trajectory {trajectory_id}: {error}"
                        for error in trajectory_errors
                    )
        replay_report = payloads.get("replay_report.json")
        episode_rows = (
            replay_report.get("episodes")
            if isinstance(replay_report, Mapping)
            else None
        )
        expected_episode_ids = {
            str(row.get("episode_id"))
            for row in natural or []
            if isinstance(row, Mapping)
        }
        if not isinstance(episode_rows, Mapping) or set(episode_rows) != expected_episode_ids:
            errors.append("replay episode matrix differs from natural coverage")
        elif any(
            not isinstance(row, Mapping) or row.get("passed") is not True
            for row in episode_rows.values()
        ):
            errors.append("one or more replay episode comparisons did not pass")

    serialized = json.dumps(payloads, ensure_ascii=False, sort_keys=True)
    for forbidden in ("legacy_13", "13维补齐", "13_dim_filled"):
        if forbidden in serialized:
            errors.append(f"forbidden fabricated-width claim found: {forbidden}")
    return errors


def publish_preflight_bundle(
    output_dir: Path,
    payloads: Mapping[str, object],
    *,
    source_paths: Sequence[Path],
    allow_test_payload: bool = False,
) -> None:
    errors = validate_preflight_payloads(payloads, allow_test_payload=allow_test_payload)
    if errors:
        raise ValueError(f"preflight payload invalid: {errors}")
    publish_atomic_bundle(Path(output_dir), payloads, source_paths)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_keys(source_paths: Sequence[Path]) -> dict[str, Path]:
    sources = tuple(Path(path).absolute() for path in source_paths)
    if not sources:
        return {}
    common_root = Path(os.path.commonpath([str(path) for path in sources]))
    if common_root.is_file():
        common_root = common_root.parent
    result: dict[str, Path] = {}
    for path in sorted(sources, key=lambda item: str(item)):
        try:
            key = path.relative_to(common_root).as_posix()
        except ValueError:
            key = path.as_posix()
        result[key] = path
    return result


def _read_payloads(output_dir: Path) -> dict[str, object]:
    payloads: dict[str, object] = {}
    for name in sorted(_expected_payload_names()):
        path = output_dir / name
        if name == "frames.jsonl":
            payloads[name] = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payloads[name] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def verify_preflight_bundle(
    output_dir: Path, *, source_paths: Sequence[Path]
) -> dict[str, object]:
    output_dir = Path(output_dir)
    errors: list[str] = []
    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (output_dir / name).is_file()]
    if missing:
        return {"passed": False, "errors": [f"missing artifact files: {missing}"]}
    try:
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [f"manifest unreadable: {exc}"]}
    for name in sorted(_expected_payload_names()):
        expected = manifest.get("artifact_hashes", {}).get(name)
        observed = _sha256(output_dir / name)
        if expected != observed:
            errors.append(f"artifact hash mismatch: {name}")
    manifest_sources = manifest.get("source_hashes", {})
    source_map = _source_keys(source_paths)
    compat_enabled = (PROJECT_ROOT / "code").is_dir() and not (PROJECT_ROOT / "代码").exists()
    exact_mapping_path = PROJECT_ROOT / "记录" / "迁移" / "2026-08-16-仓库目录迁移映射.json"
    exact_mapping = load_exact_mapping(exact_mapping_path) if compat_enabled and exact_mapping_path.is_file() else None
    changes_path = PROJECT_ROOT / "记录" / "迁移" / "2026-08-16-迁移源变更.json"
    source_changes = load_source_changes(changes_path) if compat_enabled and changes_path.is_file() else {}
    if not isinstance(manifest_sources, Mapping):
        errors.append("source file matrix mismatch")
    elif set(manifest_sources) == set(source_map):
        for key, path in source_map.items():
            if not path.is_file():
                errors.append(f"source hash mismatch: {key}")
            elif _sha256(path) != manifest_sources[key]:
                errors.append(f"source hash mismatch: {key}")
    else:
        expected_resolved = {
            resolve_repository_path(PROJECT_ROOT, key, exact_mapping=exact_mapping): path
            for key, path in source_map.items()
        }
        manifest_resolved = {
            resolve_repository_path(PROJECT_ROOT, str(key), exact_mapping=exact_mapping): str(key)
            for key in manifest_sources
        }
        missing_expected = set(expected_resolved) - set(manifest_resolved)
        unexpected = set(manifest_resolved) - set(expected_resolved)
        retired_root = (PROJECT_ROOT / "docs" / "superpowers").resolve()
        retired_missing = {
            path
            for path in unexpected
            if not path.exists() and path.is_relative_to(retired_root)
        }
        if missing_expected or unexpected - retired_missing:
            errors.append("source file matrix mismatch")
        if not missing_expected and not (unexpected - retired_missing):
            for key, expected_hash in manifest_sources.items():
                path = resolve_repository_path(PROJECT_ROOT, str(key), exact_mapping=exact_mapping)
                if not path.is_file():
                    errors.append(f"source hash mismatch: {key}")
                elif _sha256(path) != str(expected_hash) and source_changes.get(str(key)) != _sha256(path):
                    errors.append(f"source hash mismatch: {key}")
    try:
        payloads = _read_payloads(output_dir)
        config = payloads.get("collector_config.json", {})
        allow_test = isinstance(config, Mapping) and config.get("test_only") is True
        errors.extend(validate_preflight_payloads(payloads, allow_test_payload=allow_test))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        errors.append(f"artifact payload unreadable: {exc}")
    return {"passed": not errors, "errors": errors}


def main() -> int:
    args = build_parser().parse_args()
    source_paths = CANONICAL_SOURCE_PATHS
    if args.verify_only:
        report = verify_preflight_bundle(args.output_dir, source_paths=source_paths)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    validate_run_request(tuple(args.seeds), args.steps, 0.1, 0.1)
    payloads = build_real_preflight_payloads(tuple(args.seeds), steps=args.steps)
    publish_preflight_bundle(
        args.output_dir, payloads, source_paths=source_paths
    )
    report = verify_preflight_bundle(args.output_dir, source_paths=source_paths)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "published": True,
                "verification": report,
                "status_flags": payloads["status_flags.json"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
