"""CPU-only, ledger-bound P2-B v2 full dual-graph collector preflight."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.action_attempt_ledger_v1 import (  # noqa: E402
    ActionAttemptLedger,
    AttemptIdentity,
    summarize_attempts,
)
from pi_jwm.airfogsim_full_dual_graph_collector_v2 import (  # noqa: E402
    CollectorAttemptRejected,
    execute_full_collector_step_v2,
)
from pi_jwm.airfogsim_full_dual_graph_frame_builder_v1 import (  # noqa: E402
    build_frame_decision,
)
from pi_jwm.airfogsim_full_dual_graph_observer_v1 import (  # noqa: E402
    observe_airfogsim_snapshot,
)
from pi_jwm.full_dual_graph_artifact_v1 import (  # noqa: E402
    build_e1_rows,
    build_full_collector_status_flags,
    compare_replays,
    validate_trajectory_frames,
)
from pi_jwm.full_dual_graph_artifact_v2 import (  # noqa: E402
    assert_publish_targets_absent,
    publish_failure_bundle,
    publish_success_bundle,
    verify_failure_bundle,
    verify_success_bundle,
)
from pi_jwm.full_dual_graph_collector_contract_v1 import (  # noqa: E402
    CollectorContractError,
    SnapshotPhase,
)
from pi_jwm.full_dual_graph_coverage_v1 import choose_resource_arm  # noqa: E402
from pi_jwm.full_dual_graph_vocabulary_v1 import (  # noqa: E402
    FullTrajectoryVocabulary,
    RouteRevisionLedger,
)

import run_p2_full_dual_graph_collector_preflight_v1 as v1  # noqa: E402


CANONICAL_SEEDS = (0, 1, 2)
NATURAL_ARMS = ("orthogonal", "interference_reuse")
REQUIRED_FIXTURES = v1.REQUIRED_FIXTURES
DEFAULT_OUTPUT_DIR = (
    CODE_ROOT
    / "artifacts"
    / "preflight"
    / "pi_jwm_p2_full_dual_graph_collector_v2_candidate_20260814"
)
CANONICAL_SOURCE_PATHS = tuple(
    dict.fromkeys(
        (
            SRC_ROOT / "pi_jwm" / "action_attempt_ledger_v1.py",
            SRC_ROOT / "pi_jwm" / "airfogsim_full_dual_graph_collector_v2.py",
            SRC_ROOT / "pi_jwm" / "full_dual_graph_artifact_v2.py",
            Path(__file__).resolve(),
            CODE_ROOT / "tests" / "test_action_attempt_ledger_v1.py",
            CODE_ROOT / "tests" / "test_airfogsim_full_dual_graph_collector_v2.py",
            CODE_ROOT / "tests" / "test_full_dual_graph_artifact_v2.py",
            CODE_ROOT / "tests" / "test_run_p2_full_dual_graph_collector_preflight_v2.py",
            PROJECT_ROOT
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-08-14-p2-action-attempt-ledger-v1-design.md",
            PROJECT_ROOT
            / "docs"
            / "superpowers"
            / "plans"
            / "2026-08-14-p2-action-attempt-ledger-v1.md",
            *v1.CANONICAL_SOURCE_PATHS,
        )
    )
)


class PreflightRunFailure(RuntimeError):
    """A rejected terminal attempt that must become a failure bundle."""

    def __init__(
        self,
        record: Mapping[str, object],
        cause: BaseException,
    ) -> None:
        self.record = dict(record)
        self.cause = cause
        super().__init__(f"{type(cause).__name__}: {cause}")


def _trajectory_id_for_arm(seed: int, arm: str) -> str:
    return v1._trajectory_id_for_arm(seed, arm)


def canonical_episode_specs(seeds: Sequence[int]) -> list[dict[str, object]]:
    return v1.canonical_episode_specs(seeds)


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
        description="Run or verify the CPU-only ledger-bound PI-JWM P2-B v2 preflight"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CANONICAL_SEEDS))
    parser.add_argument("--steps", type=int, default=20)
    return parser


def _scheduler_tuple(schedulers) -> tuple[object, object, object]:
    if isinstance(schedulers, tuple) and len(schedulers) == 3:
        return schedulers
    return schedulers, schedulers, schedulers


def fixture_attempt_identity(
    name: str,
    *,
    seed: int,
    run_role: str,
    frame_index: int,
) -> AttemptIdentity:
    if name not in REQUIRED_FIXTURES:
        raise ValueError(f"unknown fixture: {name}")
    if run_role not in {"bootstrap", "fixture"}:
        raise ValueError("fixture identity role must be bootstrap or fixture")
    requested_arm = (
        "interference_reuse"
        if name in {"cross_transmitter_rb_reuse", "tti_failure"}
        else "orthogonal"
    )
    return AttemptIdentity(
        run_role=run_role,
        episode_id=f"fixture::{name}::seed::{seed}",
        trajectory_id=_trajectory_id_for_arm(seed, requested_arm),
        frame_index=frame_index,
        candidate_ordinal=0,
    )


def execute_attempt_frame(
    env,
    schedulers,
    *,
    identity: AttemptIdentity,
    seed: int,
    ledger: ActionAttemptLedger,
    vocabulary: FullTrajectoryVocabulary | None = None,
    route_revisions: RouteRevisionLedger | None = None,
    observer=observe_airfogsim_snapshot,
    builder=build_frame_decision,
    runtime_inputs=v1._runtime_inputs,
):
    """Observe, begin, build, validate, and execute one candidate without retry."""

    decision_snapshot = observer(env, phase=SnapshotPhase.DECISION)
    attempt = ledger.begin(identity)
    vocabulary = vocabulary if vocabulary is not None else FullTrajectoryVocabulary()
    route_revisions = (
        route_revisions if route_revisions is not None else RouteRevisionLedger()
    )
    try:
        node_cpu, node_distance = runtime_inputs(env, decision_snapshot)
        built = builder(
            decision_snapshot,
            trajectory_id=identity.trajectory_id,
            frame_index=identity.frame_index,
            seed=seed,
            n_rb=int(env.channel_manager.n_RB),
            vocabulary=vocabulary,
            route_revisions=route_revisions,
            node_cpu=node_cpu,
            node_distance=node_distance,
        )
    except CollectorContractError as exc:
        row = attempt.reject(
            terminal_stage="contract_validation",
            rejection_code="contract_validation_error",
            rejection_detail=f"{type(exc).__name__}: {exc}",
            environment_mutation_status="none",
        )
        raise PreflightRunFailure(row, exc) from exc
    except Exception as exc:
        row = attempt.reject(
            terminal_stage="candidate_build",
            rejection_code="candidate_build_error",
            rejection_detail=f"{type(exc).__name__}: {exc}",
            environment_mutation_status="none",
        )
        raise PreflightRunFailure(row, exc) from exc

    attempt.candidate_built(built.action)
    attempt.contract_validated()
    task_scheduler, communication_scheduler, computation_scheduler = _scheduler_tuple(
        schedulers
    )
    try:
        result = execute_full_collector_step_v2(
            env,
            built,
            attempt=attempt,
            trajectory_id=identity.trajectory_id,
            task_scheduler=task_scheduler,
            communication_scheduler=communication_scheduler,
            computation_scheduler=computation_scheduler,
            observer=observer,
        )
    except CollectorAttemptRejected as exc:
        raise PreflightRunFailure(exc.record, exc) from exc
    return built, result


def run_natural_episode_v2(
    spec: Mapping[str, object],
    *,
    steps: int,
    run_role: str,
    ledger: ActionAttemptLedger,
) -> dict[str, object]:
    if run_role not in {"natural_reference", "natural_replay"}:
        raise ValueError("natural episode role must be reference or replay")
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
        os.chdir(v1.single_step_runner.EXAMPLE_DIR)
        env, task_scheduler, communication_scheduler, computation_scheduler, config = (
            v1.single_step_runner._build_environment(
                seed, max_time=max(3.0, (steps + 2) * 0.1)
            )
        )
        validate_run_request(
            CANONICAL_SEEDS,
            max(20, steps),
            env.traffic_interval,
            env.simulation_interval,
        )
        schedulers = (task_scheduler, communication_scheduler, computation_scheduler)
        for frame_index in range(steps):
            if env.isDone():
                naturally_ended = True
                break
            identity = AttemptIdentity(
                run_role=run_role,
                episode_id=str(spec["episode_id"]),
                trajectory_id=trajectory_id,
                frame_index=frame_index,
                candidate_ordinal=0,
            )
            built, result = execute_attempt_frame(
                env,
                schedulers,
                identity=identity,
                seed=seed,
                ledger=ledger,
                vocabulary=vocabulary,
                route_revisions=route_revisions,
            )
            if built.resource_policy != expected_arm:
                raise RuntimeError("frame builder resource arm changed within episode")
            results.append(result)
        frames = v1._frame_payloads(results, vocabulary)
        vocabulary_payload = v1._vocabulary_payload(vocabulary)
        validation_errors = validate_trajectory_frames(
            frames, vocabulary=vocabulary_payload, fixture=False
        )
        return {
            "spec": dict(spec),
            "config": {
                "traffic_interval": float(env.traffic_interval),
                "simulation_interval": float(env.simulation_interval),
                "n_rb": int(env.channel_manager.n_RB),
                "source": "AirFogSim_preflight_config",
                "raw_config_hash": v1.hashlib.sha256(
                    json.dumps(v1._jsonable(config), sort_keys=True).encode("utf-8")
                ).hexdigest(),
            },
            "frames": frames,
            "vocabulary": vocabulary_payload,
            "validation_errors": validation_errors,
            "real_airfogsim_step_count": len(results),
            "naturally_ended": naturally_ended,
        }
    finally:
        if env is not None:
            env.close()
        os.chdir(previous_cwd)


def run_natural_replay_pair_v2(
    spec: Mapping[str, object],
    *,
    steps: int,
    ledger: ActionAttemptLedger,
) -> dict[str, object]:
    reference = run_natural_episode_v2(
        spec, steps=steps, run_role="natural_reference", ledger=ledger
    )
    replay = run_natural_episode_v2(
        spec, steps=steps, run_role="natural_replay", ledger=ledger
    )
    comparison = compare_replays(reference["frames"], replay["frames"])
    if reference["validation_errors"]:
        comparison["passed"] = False
        comparison.setdefault("validation_errors", []).extend(
            reference["validation_errors"]
        )
    if replay["validation_errors"]:
        comparison["passed"] = False
        comparison.setdefault("validation_errors", []).extend(replay["validation_errors"])
    if (
        reference["real_airfogsim_step_count"]
        != replay["real_airfogsim_step_count"]
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


def _execute_fixture_frame_v2(
    env,
    schedulers,
    *,
    name: str,
    seed: int,
    run_role: str,
    frame_index: int,
    ledger: ActionAttemptLedger,
    vocabulary: FullTrajectoryVocabulary,
    route_revisions: RouteRevisionLedger,
):
    return execute_attempt_frame(
        env,
        schedulers,
        identity=fixture_attempt_identity(
            name,
            seed=seed,
            run_role=run_role,
            frame_index=frame_index,
        ),
        seed=seed,
        ledger=ledger,
        vocabulary=vocabulary,
        route_revisions=route_revisions,
    )


def run_real_fixture_v2(
    name: str,
    *,
    seed: int,
    ledger: ActionAttemptLedger,
) -> dict[str, object]:
    """Execute one controlled fixture through two ledger-observed real steps."""

    if name not in REQUIRED_FIXTURES:
        raise ValueError(f"unknown fixture: {name}")
    previous_cwd = Path.cwd()
    env = None
    step_count = 0
    try:
        os.chdir(v1.single_step_runner.EXAMPLE_DIR)
        wired = name == "wired_flow"
        env, task_scheduler, communication_scheduler, computation_scheduler, _ = (
            v1._build_fixture_environment(seed, wired=wired)
        )
        schedulers = (task_scheduler, communication_scheduler, computation_scheduler)
        vocabulary = FullTrajectoryVocabulary()
        revisions = RouteRevisionLedger()
        _, bootstrap = _execute_fixture_frame_v2(
            env,
            schedulers,
            name=name,
            seed=seed,
            run_role="bootstrap",
            frame_index=0,
            ledger=ledger,
            vocabulary=vocabulary,
            route_revisions=revisions,
        )
        step_count += 1
        sources, uavs, rsus = v1._wireless_fixture_nodes(env)

        if name in {"multi_task_multi_flow", "cross_transmitter_rb_reuse"}:
            for ordinal, source in enumerate(sources[:2]):
                v1._append_waiting(
                    env,
                    v1._add_fixture_task(
                        env,
                        task_id=f"fixture::{name}::{ordinal}",
                        source_id=source,
                        size=10.0,
                    ),
                )
            built, result = _execute_fixture_frame_v2(
                env,
                schedulers,
                name=name,
                seed=seed,
                run_role="fixture",
                frame_index=1,
                ledger=ledger,
                vocabulary=vocabulary,
                route_revisions=revisions,
            )
            step_count += 1
            selected_hops = [
                hop
                for hop in built.action.hops
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
                "two_distinct_transmitters": len(
                    {hop.source_id for hop in selected_hops}
                )
                >= 2,
                "real_step_completed": result.stepped,
            }
            if name == "cross_transmitter_rb_reuse":
                checks["rb_reused_across_distinct_transmitters"] = reused
                checks["same_transmitter_not_reused"] = len(
                    {
                        (hop.source_id, rb_by_hop[hop.hop_id])
                        for hop in selected_hops
                    }
                ) == len(selected_hops)
            return v1._fixture_row(
                name,
                passed=True,
                step_count=step_count,
                checks=checks,
                intervention="two generated real Task objects inserted into waiting_to_offload",
                evidence={"action": built.action, "transfer_rows": result.transfer_rows},
            )

        if name == "local_execution":
            source = sources[0]
            injected = []
            for ordinal in range(3):
                task = v1._add_fixture_task(
                    env,
                    task_id=f"fixture::local::{ordinal}",
                    source_id=source,
                    cpu=50.0,
                    size=10.0,
                )
                v1._append_waiting(env, task)
                injected.append(task.getTaskId())
            built, result = _execute_fixture_frame_v2(
                env,
                schedulers,
                name=name,
                seed=seed,
                run_role="fixture",
                frame_index=1,
                ledger=ledger,
                vocabulary=vocabulary,
                route_revisions=revisions,
            )
            step_count += 1
            local = [
                decision
                for decision in built.action.decisions
                if decision.task_id in injected
                and decision.selected
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
            return v1._fixture_row(
                name,
                passed=True,
                step_count=step_count,
                checks=checks,
                intervention="three waiting tasks expose the frame-ordinal local target family",
                evidence={"action": built.action, "cpu_rows": result.cpu_rows},
            )

        if name == "wired_flow":
            source, target = "RSU_0", "cloudServer_4"
            if source not in env.RSUs or target not in env.cloudServers:
                raise RuntimeError("configured wired fixture endpoints are absent")
            task = v1._add_fixture_task(
                env,
                task_id="fixture::wired",
                source_id=source,
                size=10_000_000.0,
            )
            task.offloadTo(target, [target], float(env.simulation_time))
            env.task_manager._offloading_tasks.setdefault(source, []).append(task)
            built, result = _execute_fixture_frame_v2(
                env,
                schedulers,
                name=name,
                seed=seed,
                run_role="fixture",
                frame_index=1,
                ledger=ledger,
                vocabulary=vocabulary,
                route_revisions=revisions,
            )
            step_count += 1
            rows = [
                row
                for row in result.transfer_rows
                if row["task_id"] == task.getTaskId()
            ]
            checks = {
                "wired_hop_selected": any(
                    hop.transport == "wired" for hop in built.action.hops
                ),
                "no_rb_for_wired_hop": not built.action.rb_allocations,
                "wired_manager_direct_result": bool(rows)
                and all(
                    row["source_method"] == "wired_manager.step_direct_result"
                    for row in rows
                ),
            }
            return v1._fixture_row(
                name,
                passed=True,
                step_count=step_count,
                checks=checks,
                intervention="real RSU-cloud wired edge plus one offloading Task",
                evidence={"action": built.action, "transfer_rows": rows},
            )

        if name in {"multihop_offload", "multihop_return"}:
            relay, compute = uavs[0], rsus[0]
            source = next(node_id for node_id in sources if node_id != relay)
            if len({source, relay, compute}) < 3:
                raise RuntimeError("multihop fixture endpoints are not distinct")
            task = v1._add_fixture_task(
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
            built, result = _execute_fixture_frame_v2(
                env,
                schedulers,
                name=name,
                seed=seed,
                run_role="fixture",
                frame_index=1,
                ledger=ledger,
                vocabulary=vocabulary,
                route_revisions=revisions,
            )
            step_count += 1
            decision = next(
                row
                for row in built.action.decisions
                if row.task_id == task.getTaskId()
            )
            hop = next(
                row for row in built.action.hops if row.hop_id == decision.hop_id
            )
            checks = {
                "full_route_retained": decision.route_nodes == expected_route,
                "only_current_carrying_hop_executes": hop.target_id == relay,
                "per_hop_transfer_observed": any(
                    row["hop_id"] == hop.hop_id for row in result.transfer_rows
                ),
            }
            return v1._fixture_row(
                name,
                passed=True,
                step_count=step_count,
                checks=checks,
                intervention="real Task initialized with an explicit two-hop route",
                evidence={"action": built.action, "transfer_rows": result.transfer_rows},
            )

        if name in {"node_disappearance_reappearance", "route_interruption"}:
            source, target = sources[0], uavs[0]
            if source == target:
                source = sources[1]
            task = v1._add_fixture_task(
                env,
                task_id=f"fixture::{name}",
                source_id=source,
                size=10.0,
            )
            task.offloadTo(target, [target], float(env.simulation_time))
            env.task_manager._offloading_tasks.setdefault(source, []).append(task)
            original_traffic, original_ai, state = v1._transient_node_gap(env, target)
            try:
                built, result = _execute_fixture_frame_v2(
                    env,
                    schedulers,
                    name=name,
                    seed=seed,
                    run_role="fixture",
                    frame_index=1,
                    ledger=ledger,
                    vocabulary=vocabulary,
                    route_revisions=revisions,
                )
            finally:
                env._updateTraffics = original_traffic
                env._updateAIModels = original_ai
                if target not in env.UAVs and "node" in state:
                    env.UAVs[target] = state["node"]
                    env.uav_ids_as_index.insert(int(state["index"]), target)
            step_count += 1
            execution_nodes = {
                node.node_id for node in result.execution_snapshot.nodes
            }
            outcome_nodes = {node.node_id for node in result.outcome_snapshot.nodes}
            decision = next(
                row
                for row in built.action.decisions
                if row.task_id == task.getTaskId()
            )
            checks = {
                "present_at_decision": target
                in {node.node_id for node in result.decision_snapshot.nodes},
                "absent_at_execution_snapshot": target not in execution_nodes,
                "restored_before_runtime_updates": state.get("restored") is True,
                "present_at_outcome": target in outcome_nodes,
                "real_step_completed": result.stepped,
            }
            if name == "route_interruption":
                checks["decision_route_not_silently_rewritten"] = (
                    decision.route_nodes == (target,)
                )
                checks["carrying_hop_identity_retained"] = any(
                    hop.hop_id == decision.hop_id for hop in built.action.hops
                )
            return v1._fixture_row(
                name,
                passed=True,
                step_count=step_count,
                checks=checks,
                intervention=(
                    "target UAV removed after real traffic update and restored before "
                    "AI/task/communication updates; tests collector snapshot/identity "
                    "semantics, not a native mobility disappearance"
                ),
                evidence={
                    "target_node_id": target,
                    "action": built.action,
                    "execution_node_ids": sorted(execution_nodes),
                    "outcome_node_ids": sorted(outcome_nodes),
                },
            )

        if name == "deadline_failure":
            task = v1._add_fixture_task(
                env,
                task_id="fixture::deadline",
                source_id=sources[0],
                cpu=1000.0,
                size=1000.0,
                deadline=0.0,
            )
            task.setAttribute("_task_arrival_time", float(env.simulation_time) - 1.0)
            v1._append_waiting(env, task)
            built, result = _execute_fixture_frame_v2(
                env,
                schedulers,
                name=name,
                seed=seed,
                run_role="fixture",
                frame_index=1,
                ledger=ledger,
                vocabulary=vocabulary,
                route_revisions=revisions,
            )
            step_count += 1
            outcome = next(
                row
                for row in result.outcome_snapshot.tasks
                if row.task_id == task.getTaskId()
            )
            checks = {
                "decision_was_recorded": any(
                    row.task_id == task.getTaskId()
                    for row in built.action.decisions
                ),
                "outcome_failed": outcome.lifecycle.value == "failed",
            }
            return v1._fixture_row(
                name,
                passed=True,
                step_count=step_count,
                checks=checks,
                intervention=(
                    "waiting Task arrival/deadline set explicitly overdue before the real step"
                ),
                evidence={"action": built.action, "outcome_task": outcome},
            )

        if name == "tti_failure":
            source, target = sources[0], rsus[0]
            injected = []
            for ordinal in range(int(env.channel_manager.n_RB) + 1):
                task = v1._add_fixture_task(
                    env,
                    task_id=f"fixture::tti::{ordinal:03d}",
                    source_id=source,
                    size=1000.0,
                )
                task.offloadTo(target, [target], float(env.simulation_time))
                task.setAttribute(
                    "_last_transmission_time", float(env.simulation_time) - 1.0
                )
                env.task_manager._offloading_tasks.setdefault(source, []).append(task)
                injected.append(task.getTaskId())
            built, result = _execute_fixture_frame_v2(
                env,
                schedulers,
                name=name,
                seed=seed,
                run_role="fixture",
                frame_index=1,
                ledger=ledger,
                vocabulary=vocabulary,
                route_revisions=revisions,
            )
            step_count += 1
            rejected = [
                row.task_id
                for row in built.action.decisions
                if row.task_id in injected
                and not row.selected
                and row.reason == "rb_budget_exhausted"
            ]
            failed = {
                row.task_id
                for row in result.outcome_snapshot.tasks
                if row.lifecycle.value == "failed"
            }
            checks = {
                "same_transmitter_rb_budget_exhausted": bool(rejected),
                "unserved_old_transmission_failed": bool(set(rejected) & failed),
                "failure_retained_in_outcome_snapshot": bool(failed & set(injected)),
            }
            return v1._fixture_row(
                name,
                passed=True,
                step_count=step_count,
                checks=checks,
                intervention=(
                    "n_RB+1 offloading Tasks share one transmitter and have an expired "
                    "last-transmission time"
                ),
                evidence={"action": built.action, "failed_task_ids": sorted(failed)},
            )

        raise AssertionError(name)
    finally:
        if env is not None:
            env.close()
        os.chdir(previous_cwd)


def run_fixture_matrix_v2(
    *, ledger: ActionAttemptLedger
) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for ordinal, name in enumerate(REQUIRED_FIXTURES):
        row = run_real_fixture_v2(name, seed=91 + ordinal, ledger=ledger)
        if row.get("passed") is not True:
            raise RuntimeError(f"fixture semantic checks failed: {name}")
        rows[name] = row
    return rows


def build_real_preflight_payloads_v2(
    seeds: Sequence[int] = CANONICAL_SEEDS,
    *,
    steps: int = 20,
    ledger: ActionAttemptLedger,
) -> dict[str, object]:
    validate_run_request(tuple(seeds), steps, 0.1, 0.1)
    frames: list[dict[str, object]] = []
    vocabularies: dict[str, object] = {}
    natural_rows: list[dict[str, object]] = []
    replay_rows: dict[str, object] = {}
    validation_errors: list[str] = []
    for spec in canonical_episode_specs(seeds):
        pair = run_natural_replay_pair_v2(spec, steps=steps, ledger=ledger)
        reference = pair["reference"]
        comparison = pair["comparison"]
        episode_id = str(spec["episode_id"])
        trajectory_id = str(spec["trajectory_id"])
        frames.extend(reference["frames"])
        vocabularies[trajectory_id] = reference["vocabulary"]
        episode_passed = (
            comparison["passed"] is True and not reference["validation_errors"]
        )
        natural_rows.append(
            {
                **dict(spec),
                "real_airfogsim_step_count": reference["real_airfogsim_step_count"],
                "naturally_ended": reference["naturally_ended"],
                "passed": episode_passed,
                "evidence_kind": "real_airfogsim_natural_replay_pair_with_attempt_ledger",
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
            validation_errors.append(
                f"{episode_id}: replay or trajectory validation failed"
            )

    fixtures = run_fixture_matrix_v2(ledger=ledger)
    for name, row in fixtures.items():
        if row.get("passed") is not True:
            validation_errors.append(f"fixture {name} did not pass")
    attempts = ledger.terminal_records()
    ledger_summary = summarize_attempts(attempts)
    passed = not validation_errors
    return {
        "collector_config.json": {
            "schema_version": "PIJWM-Full-Collector-Preflight-v2",
            "test_only": False,
            "seeds": list(seeds),
            "steps": steps,
            "traffic_interval": 0.1,
            "simulation_interval": 0.1,
            "natural_arms": list(NATURAL_ARMS),
            "natural_episode_count": len(natural_rows),
            "fixture_count": len(fixtures),
            "scope": "CPU-only collector preflight; nontraining; unlocked data only",
            "formal_data_approved": False,
            "training_eligible": False,
        },
        "vocabularies.json": vocabularies,
        "frames.jsonl": frames,
        "action_attempts.jsonl": attempts,
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
            "ledger_summary": ledger_summary,
            "formal_data_approved": False,
            "training_eligible": False,
        },
        "replay_report.json": {
            "passed": all(row["passed"] is True for row in replay_rows.values()),
            "episodes": replay_rows,
            "fresh_environment_per_reference_and_replay": True,
        },
        "status_flags.json": build_full_collector_status_flags(passed=passed),
    }


def _failure_report(
    failure: BaseException,
    record: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "failure_scope": "attempt" if record is not None else "run_level_observation_failure",
        "attempt_id": record.get("attempt_id") if record is not None else None,
        "run_role": record.get("run_role") if record is not None else None,
        "terminal_stage": record.get("terminal_stage") if record is not None else None,
        "rejection_code": record.get("rejection_code") if record is not None else None,
        "quarantined": record.get("quarantined") if record is not None else False,
        "error_type": type(failure).__name__,
        "error_detail": str(failure) or type(failure).__name__,
        "formal_data_approved": False,
        "training_eligible": False,
        "gpu_started": False,
        "locked_test_accessed": False,
    }


def run_and_publish(
    output_dir: Path,
    *,
    collector: Callable[[ActionAttemptLedger], Mapping[str, object]],
    source_paths: Sequence[Path] = CANONICAL_SOURCE_PATHS,
) -> dict[str, object]:
    output = Path(output_dir)
    assert_publish_targets_absent(output)
    ledger = ActionAttemptLedger()
    try:
        payloads = dict(collector(ledger))
        recorded = ledger.terminal_records()
        if recorded and payloads.get("action_attempts.jsonl") != recorded:
            raise ValueError("collector payload ledger differs from runtime ledger")
        publish_success_bundle(output, payloads, source_paths)
    except PreflightRunFailure as exc:
        failed = publish_failure_bundle(
            output,
            ledger.terminal_records(),
            _failure_report(exc.cause, exc.record),
            source_paths,
        )
        return {
            "published": "failure",
            "output_dir": str(failed.absolute()),
            "verification": verify_failure_bundle(failed, source_paths),
        }
    except Exception as exc:
        failed = publish_failure_bundle(
            output,
            ledger.terminal_records(),
            _failure_report(exc, None),
            source_paths,
        )
        return {
            "published": "failure",
            "output_dir": str(failed.absolute()),
            "verification": verify_failure_bundle(failed, source_paths),
        }
    return {
        "published": "success",
        "output_dir": str(output.absolute()),
        "verification": verify_success_bundle(output, source_paths),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        report = verify_success_bundle(args.output_dir, CANONICAL_SOURCE_PATHS)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    validate_run_request(tuple(args.seeds), args.steps, 0.1, 0.1)
    result = run_and_publish(
        args.output_dir,
        collector=lambda ledger: build_real_preflight_payloads_v2(
            tuple(args.seeds), steps=args.steps, ledger=ledger
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["published"] == "success" and result["verification"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
