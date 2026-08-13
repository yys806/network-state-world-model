from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
SRC_ROOT = CODE_ROOT / "src"
REFERENCE_ROOT = CODE_ROOT / "reference" / "AirFogSim"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
DEFAULT_OUTPUT_DIR = (
    CODE_ROOT / "artifacts" / "preflight" / "pi_jwm_p2_multistep_collector_v1"
)
for path in (SRC_ROOT, REFERENCE_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_p2_single_step_collector_preflight_v1 as single_step  # noqa: E402
from pi_jwm.multistep_collector_contract_v1 import (  # noqa: E402
    EdgeIdentity,
    LinkHistoryLedger,
    LinkOutcome,
    MULTISTEP_CONTRACT_VERSION,
    TrajectoryVocabulary,
)


REQUIRED_FILES = (
    "trajectory_frames.json",
    "vocabularies.json",
    "temporal_trace.json",
    "history_alignment_audit.json",
    "resource_bundle.json",
    "validation_report.json",
    "summary.json",
    "manifest.json",
)
EXPECTED_TRACE = (
    "action_validated",
    "decision_time_observation_captured",
    "cpu_callback_installed",
    "action_setters_called",
    "env_step_started",
    "env_step_finished",
)
FORBIDDEN_TRUE_FLAGS = (
    "v4_collector_implemented",
    "v4_dataset_complete",
    "model_training_started",
    "gpu_started",
    "locked_test_accessed",
    "candidate_rollout_planner_complete",
    "final_method_frozen",
    "training_eligible",
)


def _status_flags() -> dict[str, bool]:
    flags = single_step.build_single_step_status_flags()
    flags["multistep_real_airfogsim_executed"] = False
    return flags


def _outcome(active: float, rate: float, served: float) -> list[dict[str, Any]]:
    return [{
        "edge_id": "ie::uav0::rsu0",
        "edge_index": 0,
        "active_flow_count": active,
        "effective_rate_per_s": rate,
        "served_data": served,
    }]


def _history(values: list[float], valid: bool, reason: str) -> list[dict[str, Any]]:
    return [{
        "edge_id": "ie::uav0::rsu0",
        "edge_index": 0,
        "values": values,
        "valid": valid,
        "missing_reason": reason,
    }]


def fake_passing_payloads_for_test() -> dict[str, Any]:
    positive = _outcome(1.0, 4.0, 0.4)
    zero = _outcome(0.0, 0.0, 0.0)
    frames = [
        {
            "frame_index": 0,
            "edge_id": "ie::uav0::rsu0",
            "edge_index": 0,
            "temporal_trace": list(EXPECTED_TRACE),
            "pre_link_history": _history([0.0, 0.0, 0.0], False, "no_history"),
            "pre_link_history_source": [],
            "outcome_link": positive,
        },
        {
            "frame_index": 1,
            "edge_id": "ie::uav0::rsu0",
            "edge_index": 0,
            "temporal_trace": list(EXPECTED_TRACE),
            "pre_link_history": _history([1.0, 4.0, 0.4], True, "none"),
            "pre_link_history_source": copy.deepcopy(positive),
            "outcome_link": zero,
        },
        {
            "frame_index": 2,
            "edge_id": "ie::uav0::rsu0",
            "edge_index": 0,
            "temporal_trace": list(EXPECTED_TRACE),
            "pre_link_history": _history([0.0, 0.0, 0.0], True, "none"),
            "pre_link_history_source": copy.deepcopy(zero),
            "outcome_link": copy.deepcopy(zero),
        },
    ]
    for frame in frames:
        frame["decision_time_channel"] = {
            "capture_phase": "before_action_setters",
            "simulation_time": float(frame["frame_index"]),
            "source": "uav0",
            "target": "rsu0",
            "rb_indices": [0],
            "channel_attenuation_db": [10.0],
            "source_method": "channel_manager.getCSI",
        }
        frame["transfer_events"] = []
        frame["flow_id"] = "task0"
        frame["flow_index"] = 0
        frame["action"] = {"offloads": [], "assignment_coo": []}
    flags = _status_flags()
    return {
        "trajectory_frames.json": {"frames": frames, "test_fixture": True},
        "vocabularies.json": {
            "node_indices": {"rsu0": 0, "uav0": 1},
            "edge_indices": {"ie::uav0::rsu0": 0},
            "flow_indices": {"task0": 0},
            "frame_snapshots": [
                {
                    "frame_index": index,
                    "node_indices": {"rsu0": 0, "uav0": 1},
                    "edge_indices": {"ie::uav0::rsu0": 0},
                    "flow_indices": {"task0": 0},
                }
                for index in range(3)
            ],
        },
        "temporal_trace.json": {
            "frames": [
                {"frame_index": index, "phases": list(EXPECTED_TRACE)}
                for index in range(3)
            ]
        },
        "history_alignment_audit.json": {"passed": True, "frame_count": 3},
        "resource_bundle.json": {"cpu_rows": [], "energy_rows": [], "test_fixture": True},
        "validation_report.json": {"passed": True, "checks": {"test_fixture": True}},
        "summary.json": {
            "scope": "three_frame_nontraining_temporal_fixture",
            "required_files": list(REQUIRED_FILES),
            "status_flags": flags,
            "limitations": [
                "observed communication edge vocabulary only",
                "not a complete strict dual graph",
                "not training eligible",
            ],
        },
    }


def _outcome_values(row: dict[str, Any]) -> list[float]:
    return [
        float(row["active_flow_count"]),
        float(row["effective_rate_per_s"]),
        float(row["served_data"]),
    ]


def validate_multistep_payloads(payloads: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    frames = payloads.get("trajectory_frames.json", {}).get("frames", [])
    if len(frames) != 3 or [row.get("frame_index") for row in frames] != [0, 1, 2]:
        return ["multistep evidence requires contiguous frames 0,1,2"]
    vocabulary = payloads.get("vocabularies.json", {})
    edge_indices = vocabulary.get("edge_indices", {})
    if len(edge_indices) != 1:
        errors.append("multistep evidence requires one observed-edge identity")
    for snapshot in vocabulary.get("frame_snapshots", []):
        frame_index = snapshot.get("frame_index")
        for name in ("node_indices", "edge_indices", "flow_indices"):
            canonical = vocabulary.get(name, {})
            observed = snapshot.get(name, {})
            if any(canonical.get(identity) != index for identity, index in observed.items()):
                errors.append(
                    f"multistep evidence frame {frame_index}: {name} vocabulary reindexed"
                )
    for frame in frames:
        prefix = f"multistep evidence frame {frame.get('frame_index')}"
        if tuple(frame.get("temporal_trace", ())) != EXPECTED_TRACE:
            errors.append(f"{prefix}: temporal trace mismatch")
        decision = frame.get("decision_time_channel", {})
        if (
            decision.get("capture_phase") != "before_action_setters"
            or decision.get("source_method") != "channel_manager.getCSI"
        ):
            errors.append(f"{prefix}: invalid decision-time channel evidence")
        edge_id = frame.get("edge_id")
        edge_index = frame.get("edge_index")
        if edge_indices.get(edge_id) != edge_index:
            errors.append(f"{prefix}: edge index is not stable")
        histories = frame.get("pre_link_history", [])
        outcomes = frame.get("outcome_link", [])
        if len(histories) != 1 or len(outcomes) != 1:
            errors.append(f"{prefix}: expected one history and outcome row")
            continue
        if histories[0].get("edge_id") != edge_id or histories[0].get("edge_index") != edge_index:
            errors.append(f"{prefix}: history edge identity mismatch")
        if outcomes[0].get("edge_id") != edge_id or outcomes[0].get("edge_index") != edge_index:
            errors.append(f"{prefix}: outcome edge identity mismatch")
        action = frame.get("action", {})
        for record in action.get("assignment_coo", []):
            if (
                not isinstance(record, list)
                or len(record) != 4
                or record[1] != frame.get("flow_index")
                or record[2] != edge_index
            ):
                errors.append(f"{prefix}: action COO identity mismatch")
        try:
            outcome_values = _outcome_values(outcomes[0])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{prefix}: invalid outcome values")
            continue
        if any(value < 0.0 for value in outcome_values):
            errors.append(f"{prefix}: outcome values must be nonnegative")
        for event in frame.get("transfer_events", []):
            if "attenuation_db" in event:
                errors.append(f"{prefix}: ambiguous legacy attenuation remains")
            if (
                event.get("capture_phase") != "after_fast_fading_before_transfer"
                or event.get("temporal_role")
                != "outcome_only_not_same_frame_decision_input"
            ):
                errors.append(f"{prefix}: invalid outcome channel evidence")

    first_history = frames[0]["pre_link_history"][0]
    if (
        first_history.get("values") != [0.0, 0.0, 0.0]
        or first_history.get("valid") is not False
        or first_history.get("missing_reason") != "no_history"
        or frames[0].get("pre_link_history_source") != []
    ):
        errors.append("multistep evidence first-frame history is not NO_HISTORY")
    for index in (1, 2):
        previous = frames[index - 1]["outcome_link"]
        source = frames[index].get("pre_link_history_source")
        history = frames[index]["pre_link_history"][0]
        expected_values = _outcome_values(previous[0])
        if source != previous:
            errors.append(f"multistep evidence frame {index}: history source mismatch")
        if history.get("values") != expected_values:
            errors.append(f"multistep evidence frame {index}: history values mismatch")
        if history.get("valid") is not True or history.get("missing_reason") != "none":
            errors.append(f"multistep evidence frame {index}: observed history marked missing")
    trace_rows = payloads.get("temporal_trace.json", {}).get("frames", [])
    if [row.get("phases") for row in trace_rows] != [row["temporal_trace"] for row in frames]:
        errors.append("multistep evidence duplicated temporal trace mismatch")
    flags = payloads.get("summary.json", {}).get("status_flags", {})
    for name in FORBIDDEN_TRUE_FLAGS:
        if flags.get(name) is not False:
            errors.append(f"multistep evidence unsafe status flag: {name}")
    if payloads.get("trajectory_frames.json", {}).get("test_fixture") is not True:
        if flags.get("multistep_real_airfogsim_executed") is not True:
            errors.append("multistep evidence real execution flag is not true")
        if frames[0].get("action", {}).get("offloads") in (None, []):
            errors.append("multistep evidence frame 0 has no offload action")
        if not frames[0].get("action", {}).get("assignment_coo"):
            errors.append("multistep evidence frame 0 has no RB action")
        for index in (1, 2):
            action = frames[index].get("action", {})
            if action.get("offloads") != [] or action.get("assignment_coo") != []:
                errors.append(f"multistep evidence frame {index} must have an empty action")
    return errors


def _projected_history_rows(projected, edge_index: int) -> list[dict[str, Any]]:
    return [
        {
            "edge_id": row.edge_id,
            "edge_index": edge_index,
            "values": list(row.values),
            "valid": row.valid,
            "missing_reason": row.missing_reason.name.lower(),
        }
        for row in projected
    ]


def _outcome_rows(
    edge_id: str, edge_index: int, events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [{
        "edge_id": edge_id,
        "edge_index": edge_index,
        "active_flow_count": float(len(events)),
        "effective_rate_per_s": float(
            sum(sum(event["rate_per_s"]) for event in events)
        ),
        "served_data": float(sum(event["delivered_data"] for event in events)),
    }]


def _vocabulary_payload(snapshot) -> dict[str, Any]:
    return {
        "node_indices": dict(snapshot.node_indices),
        "edge_indices": dict(snapshot.edge_indices),
        "flow_indices": dict(snapshot.flow_indices),
        "node_presence": list(snapshot.node_presence),
        "edge_presence": list(snapshot.edge_presence),
        "flow_presence": list(snapshot.flow_presence),
    }


def build_real_payloads(seed: int = 0) -> dict[str, Any]:
    old_cwd = Path.cwd()
    env = None
    try:
        os.chdir(single_step.EXAMPLE_DIR)
        env, task_sched, comm_sched, comp_sched, config = single_step._build_environment(
            seed, 5.0
        )
        task, warmup_steps = single_step._warm_to_branch(env)
        source = str(task.getTaskNodeId())
        task_id = str(task.getTaskId())
        target = single_step._choose_remote_targets(env, source)[0]
        edge_id = f"ie::{source}::{target}"
        edge = EdgeIdentity(
            edge_id=edge_id,
            source_id=source,
            target_id=target,
            edge_class=f"wireless:{env._getNodeTypeById(source)}2{env._getNodeTypeById(target)}",
        )
        n_rb = int(env.channel_manager.n_RB)
        vocabulary = TrajectoryVocabulary()
        history = LinkHistoryLedger(edge_ids=(edge_id,))
        frames: list[dict[str, Any]] = []
        vocabulary_frames: list[dict[str, Any]] = []
        resource_frames: list[dict[str, Any]] = []
        previous_outcome_rows: list[dict[str, Any]] = []

        for frame_index in range(3):
            node_ids = single_step._all_node_ids(env)
            vocabulary_snapshot = vocabulary.observe(
                node_ids=node_ids, edges=(edge,), flow_ids=(task_id,)
            )
            edge_index = vocabulary_snapshot.edge_indices[edge_id]
            projected = history.project(edge_ids=(edge_id,))
            history_rows = _projected_history_rows(projected, edge_index)
            rb_assignments = (
                tuple(
                    single_step.RbAssignment(0, 0, edge_index, rb_index)
                    for rb_index in range(n_rb)
                )
                if frame_index == 0
                else ()
            )
            offloads = (
                (
                    single_step.OffloadAction(
                        source, task_id, target, (target,)
                    ),
                )
                if frame_index == 0
                else ()
            )
            action = single_step.CandidateAction(
                candidate_id=f"trajectory_frame_{frame_index}",
                offloads=offloads,
                rb_assignments=rb_assignments,
            )
            energy_before = single_step._energy_snapshot(env)
            event_start = len(env.pi_jwm_transfer_events)
            pre_time = float(env.simulation_time)
            result = single_step.execute_candidate(
                env,
                action,
                task_ids=(task_id,),
                node_ids=node_ids,
                edge_count=len(vocabulary_snapshot.edge_indices),
                flow_count=len(vocabulary_snapshot.flow_indices),
                n_rb=n_rb,
                task_scheduler=task_sched,
                communication_scheduler=comm_sched,
                computation_scheduler=comp_sched,
                pre_action_observer=lambda: single_step._capture_decision_time_channel(
                    env, source, target, list(range(n_rb))
                ),
            )
            events = copy.deepcopy(env.pi_jwm_transfer_events[event_start:])
            energy_after = single_step._energy_snapshot(env)
            candidate_resource = {
                "candidate_id": f"trajectory_frame_{frame_index}",
                "transfer_events": events,
                "cpu_rows": list(result.cpu_rows),
                "energy_before": energy_before,
                "energy_after": energy_after,
                "energy_costs": {
                    "fly_unit_cost": float(env.energy_manager._fly_unit_cost),
                    "hover_unit_cost": float(env.energy_manager._hover_unit_cost),
                    "sensing_unit_cost": float(env.energy_manager._sensing_unit_cost),
                    "send_unit_cost": float(env.energy_manager._send_unit_cost),
                    "receive_unit_cost": float(env.energy_manager._receive_unit_cost),
                },
            }
            energy_rows = single_step._build_energy_rows(candidate_resource)
            outcome_rows = _outcome_rows(edge_id, edge_index, events)
            outcome = outcome_rows[0]
            frame_checks = {
                "one_real_step": bool(
                    float(env.simulation_time) > pre_time and result.stepped
                ),
                "temporal_trace": tuple(result.temporal_trace) == EXPECTED_TRACE,
                "cpu_conservation": single_step._validate_cpu_candidate(candidate_resource),
                "energy_conservation": single_step._validate_energy_rows(energy_rows),
                "event_count_expected": bool(events) if frame_index == 0 else not events,
            }
            if not all(frame_checks.values()):
                raise RuntimeError(
                    f"real multistep frame validation failed at {frame_index}: {frame_checks}"
                )
            history.commit(
                frame_index=frame_index,
                outcomes={
                    edge_id: LinkOutcome(
                        outcome["active_flow_count"],
                        outcome["effective_rate_per_s"],
                        outcome["served_data"],
                    )
                },
                frame_validated=True,
            )
            action_payload = {
                "candidate_id": action.candidate_id,
                "offloads": [
                    {
                        "task_node_id": row.task_node_id,
                        "task_id": row.task_id,
                        "target_node_id": row.target_node_id,
                        "route_nodes": list(row.route_nodes),
                    }
                    for row in offloads
                ],
                "assignment_coo": [list(row.as_tuple()) for row in rb_assignments],
            }
            frame = {
                "frame_index": frame_index,
                "simulation_time_before": pre_time,
                "simulation_time_after": float(env.simulation_time),
                "edge_id": edge_id,
                "edge_index": edge_index,
                "flow_id": task_id,
                "flow_index": vocabulary_snapshot.flow_indices[task_id],
                "decision_time_channel": copy.deepcopy(result.pre_action_observation),
                "temporal_trace": list(result.temporal_trace),
                "action": action_payload,
                "pre_link_history": history_rows,
                "pre_link_history_source": copy.deepcopy(previous_outcome_rows),
                "outcome_link": outcome_rows,
                "transfer_events": events,
                "frame_checks": frame_checks,
            }
            frames.append(frame)
            vocabulary_frames.append({
                "frame_index": frame_index,
                **_vocabulary_payload(vocabulary_snapshot),
            })
            resource_frames.append({
                "frame_index": frame_index,
                "cpu_rows": list(result.cpu_rows),
                "energy_rows": energy_rows,
                "observed_order": list(env.pi_jwm_order[-5:]),
            })
            previous_outcome_rows = copy.deepcopy(outcome_rows)

        final_vocabulary = vocabulary.snapshot()
        flags = _status_flags()
        flags["multistep_real_airfogsim_executed"] = True
        payloads = {
            "trajectory_frames.json": {
                "scope": "three_frame_nontraining_temporal_fixture",
                "seed": seed,
                "warmup_steps": warmup_steps,
                "config_hash": single_step.canonical_hash(config),
                "frames": frames,
            },
            "vocabularies.json": {
                **_vocabulary_payload(final_vocabulary),
                "edge_bindings": [asdict(edge)],
                "frame_snapshots": vocabulary_frames,
            },
            "temporal_trace.json": {
                "frames": [
                    {"frame_index": row["frame_index"], "phases": row["temporal_trace"]}
                    for row in frames
                ]
            },
            "history_alignment_audit.json": {
                "passed": True,
                "frame_count": len(frames),
                "first_frame_no_history": True,
                "second_frame_from_first_outcome": True,
                "third_frame_valid_zero_from_second_outcome": True,
            },
            "resource_bundle.json": {
                "cpu_rule_version": "PIJWM-CPU-Inner-Rule-v1",
                "frames": resource_frames,
            },
            "validation_report.json": {
                "passed": True,
                "checks": {
                    "three_real_steps": len(frames) == 3,
                    "first_frame_positive_transfer": frames[0]["outcome_link"][0]["served_data"] > 0.0,
                    "later_frames_zero_transfer": all(
                        row["outcome_link"][0]["served_data"] == 0.0 for row in frames[1:]
                    ),
                    "stable_edge_index": len({row["edge_index"] for row in frames}) == 1,
                    "all_frame_checks": all(
                        all(row["frame_checks"].values()) for row in frames
                    ),
                },
            },
            "summary.json": {
                "scope": "three_frame_nontraining_temporal_fixture",
                "seed": seed,
                "real_airfogsim_step_count": len(frames),
                "required_files": list(REQUIRED_FILES),
                "status_flags": flags,
                "limitations": [
                    "one fixed three-frame seed-0 trajectory",
                    "observed communication edge vocabulary only",
                    "not a complete strict dual graph",
                    "not a v4 trajectory dataset",
                    "not training eligible",
                ],
            },
        }
        evidence_errors = validate_multistep_payloads(payloads)
        if evidence_errors:
            raise RuntimeError(f"real multistep evidence invalid: {evidence_errors}")
        return payloads
    finally:
        if env is not None:
            env.close()
        os.chdir(old_cwd)


def write_preflight_bundle(output_dir: Path, payloads: dict[str, Any]) -> dict[str, Any]:
    expected = set(REQUIRED_FILES) - {"manifest.json"}
    if set(payloads) != expected:
        raise ValueError(f"payload names differ: {sorted(set(payloads) ^ expected)}")
    if payloads["validation_report.json"].get("passed") is not True:
        raise ValueError("validation report did not pass; refusing publication")
    evidence_errors = validate_multistep_payloads(payloads)
    if evidence_errors:
        raise ValueError(f"multistep evidence invalid: {evidence_errors}")
    output_dir = Path(output_dir)
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty artifact directory: {output_dir}")
        output_dir.rmdir()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        for name, value in payloads.items():
            single_step._write_json(temporary / name, value)
        source_files = (
            PROJECT_ROOT / "docs" / "superpowers" / "specs" / "2026-08-13-p2-multistep-temporal-contract-design.md",
            PROJECT_ROOT / "docs" / "superpowers" / "plans" / "2026-08-13-p2-multistep-temporal-contract.md",
            CODE_ROOT / "scripts" / "run_p2_multistep_collector_preflight_v1.py",
            CODE_ROOT / "scripts" / "run_p2_single_step_collector_preflight_v1.py",
            CODE_ROOT / "tests" / "test_run_p2_multistep_collector_preflight_v1.py",
            CODE_ROOT / "tests" / "test_multistep_collector_contract_v1.py",
            CODE_ROOT / "src" / "pi_jwm" / "multistep_collector_contract_v1.py",
            CODE_ROOT / "src" / "pi_jwm" / "airfogsim_single_step_collector_v1.py",
            REFERENCE_ROOT / "airfogsim" / "airfogsim_env.py",
            REFERENCE_ROOT / "airfogsim" / "manager" / "channel_manager_cp.py",
        )
        manifest = {
            "schema_version": "PIJWM-P2-Multistep-Manifest-v1",
            "contract_version": MULTISTEP_CONTRACT_VERSION,
            "status_flags": copy.deepcopy(payloads["summary.json"]["status_flags"]),
            "artifact_hashes": {
                name: single_step.file_hash(temporary / name) for name in sorted(expected)
            },
            "source_hashes": {
                str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): single_step.file_hash(path)
                for path in source_files
            },
        }
        single_step._write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return payloads["summary.json"]


def verify_preflight_bundle(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    missing = [name for name in REQUIRED_FILES if not (output_dir / name).is_file()]
    if missing:
        return {"passed": False, "errors": [f"missing files: {missing}"]}
    payloads = {
        name: json.loads((output_dir / name).read_text(encoding="utf-8"))
        for name in REQUIRED_FILES if name != "manifest.json"
    }
    errors = validate_multistep_payloads(payloads)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.get("artifact_hashes", {}).items():
        if single_step.file_hash(output_dir / name) != expected:
            errors.append(f"artifact hash mismatch: {name}")
    for relative, expected in manifest.get("source_hashes", {}).items():
        path = PROJECT_ROOT / Path(relative)
        if not path.is_file() or single_step.file_hash(path) != expected:
            errors.append(f"source hash mismatch: {relative}")
    return {"passed": not errors, "errors": errors}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P2 three-frame temporal CPU preflight")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_only:
        report = verify_preflight_bundle(args.output_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    payloads = build_real_payloads(args.seed)
    summary = write_preflight_bundle(args.output_dir, payloads)
    verification = verify_preflight_bundle(args.output_dir)
    print(
        json.dumps(
            {"summary": summary, "verification": verification},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
