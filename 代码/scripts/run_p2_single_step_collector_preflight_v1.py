from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
SRC_ROOT = CODE_ROOT / "src"
REFERENCE_ROOT = CODE_ROOT / "reference" / "AirFogSim"
EXAMPLE_DIR = REFERENCE_ROOT / "examples"
SMALL_EXPERIMENTS = CODE_ROOT / "scripts" / "small_experiments"
DEFAULT_OUTPUT_DIR = CODE_ROOT / "artifacts" / "preflight" / "pi_jwm_p2_single_step_collector_v1"
for path in (SRC_ROOT, REFERENCE_ROOT, EXAMPLE_DIR, SMALL_EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.airfogsim_contract_adapter import (  # noqa: E402
    apply_transmission_totals,
    direct_transmission_totals,
)
from pi_jwm.airfogsim_single_step_collector_v1 import execute_candidate  # noqa: E402
from pi_jwm.information_edge_contract_v4 import (  # noqa: E402
    CONTRACT_VERSION,
    MissingReason,
    build_field_registry,
    validate_assignment_coo,
    validate_field_values,
    validate_link_outcome,
    validate_prev_field_timing,
    validate_rb_outcome,
)
from pi_jwm.single_step_collector_contract_v1 import (  # noqa: E402
    COLLECTOR_CONTRACT_VERSION,
    CandidateAction,
    OffloadAction,
    RbAssignment,
    build_single_step_status_flags,
)


REQUIRED_FILES = (
    "candidate_comparison.json",
    "action_ledger.json",
    "transfer_events.json",
    "single_step_graph.json",
    "resource_bundle.json",
    "field_mask_audit.json",
    "validation_report.json",
    "summary.json",
    "manifest.json",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def fake_passing_payloads_for_test() -> dict[str, Any]:
    flags = build_single_step_status_flags()
    candidate_id = "test_fixture"
    rb_indices = [0]
    decision_time_channel = {
        "capture_phase": "before_action_setters",
        "simulation_time": 0.0,
        "source": "uav0",
        "target": "rsu0",
        "rb_indices": rb_indices,
        "channel_attenuation_db": [10.0],
        "source_method": "channel_manager.getCSI",
    }
    temporal_trace = [
        "action_validated",
        "decision_time_observation_captured",
        "cpu_callback_installed",
        "action_setters_called",
        "env_step_started",
        "env_step_finished",
    ]
    return {
        "candidate_comparison.json": {"observable_difference": True, "test_fixture": True},
        "action_ledger.json": {"actions": []},
        "transfer_events.json": {
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "events": [
                        {
                            "source": "uav0",
                            "target": "rsu0",
                            "rb_indices": rb_indices,
                            "outcome_channel_attenuation_db": [11.0],
                            "capture_phase": "after_fast_fading_before_transfer",
                            "temporal_role": "outcome_only_not_same_frame_decision_input",
                        }
                    ],
                }
            ]
        },
        "single_step_graph.json": {"scope": "single_step_nontraining", "candidates": []},
        "resource_bundle.json": {"cpu_rows": [], "energy_rows": []},
        "field_mask_audit.json": {
            "contract_version": CONTRACT_VERSION,
            "candidate_audits": [
                {
                    "candidate_id": candidate_id,
                    "decision_time_channel": decision_time_channel,
                    "temporal_trace": temporal_trace,
                    "fields": [
                        {
                            "name": "pre_link.channel_attenuation_mean_db",
                            "value": 10.0,
                            "provenance": "derived_from_direct_decision_time_csi_before_setters",
                        },
                        {
                            "name": "pre_link.channel_attenuation_std_db",
                            "value": 0.0,
                            "provenance": "derived_from_direct_decision_time_csi_before_setters",
                        },
                        {
                            "name": "pre_rb_optional.channel_attenuation_db",
                            "value": [10.0],
                            "provenance": "direct_decision_time_csi_before_setters",
                        },
                    ],
                }
            ],
        },
        "validation_report.json": {"passed": True, "checks": {"test_fixture": True}},
        "summary.json": {
            "scope": "contract_fixture",
            "required_files": list(REQUIRED_FILES),
            "status_flags": flags,
        },
    }


def validate_temporal_payloads(payloads: dict[str, Any]) -> list[str]:
    """Recompute the action-pre versus outcome-side evidence boundary."""

    errors: list[str] = []
    audits = payloads.get("field_mask_audit.json", {}).get("candidate_audits", [])
    event_rows = payloads.get("transfer_events.json", {}).get("candidates", [])
    events_by_candidate = {
        row.get("candidate_id"): row.get("events", []) for row in event_rows
    }
    if not audits:
        return ["temporal evidence has no candidate audits"]
    expected_trace = (
        "action_validated",
        "decision_time_observation_captured",
        "cpu_callback_installed",
        "action_setters_called",
        "env_step_started",
        "env_step_finished",
    )
    for audit in audits:
        candidate_id = audit.get("candidate_id")
        prefix = f"temporal evidence candidate {candidate_id}"
        decision = audit.get("decision_time_channel", {})
        if decision.get("capture_phase") != "before_action_setters":
            errors.append(f"{prefix}: invalid decision capture phase")
        if decision.get("source_method") != "channel_manager.getCSI":
            errors.append(f"{prefix}: invalid decision source method")
        trace = tuple(audit.get("temporal_trace", ()))
        if trace != expected_trace:
            errors.append(f"{prefix}: invalid phase order")
        fields = {row.get("name"): row for row in audit.get("fields", [])}
        pre_rb = fields.get("pre_rb_optional.channel_attenuation_db", {})
        if pre_rb.get("provenance") != "direct_decision_time_csi_before_setters":
            errors.append(f"{prefix}: action-pre RB provenance is not decision-time CSI")
        decision_values = np.asarray(decision.get("channel_attenuation_db", []), dtype=float)
        pre_rb_values = np.asarray(pre_rb.get("value", []), dtype=float)
        if (
            pre_rb_values.shape != decision_values.shape
            or not np.isfinite(pre_rb_values).all()
            or not np.allclose(pre_rb_values, decision_values, rtol=1e-6, atol=1e-6)
        ):
            errors.append(f"{prefix}: action-pre RB values differ from decision-time CSI")
        if decision_values.size == 0 or not np.isfinite(decision_values).all():
            errors.append(f"{prefix}: decision-time CSI is empty or non-finite")
        else:
            expected = {
                "pre_link.channel_attenuation_mean_db": float(decision_values.mean()),
                "pre_link.channel_attenuation_std_db": float(decision_values.std()),
            }
            for name, value in expected.items():
                row = fields.get(name, {})
                if row.get("provenance") != "derived_from_direct_decision_time_csi_before_setters":
                    errors.append(f"{prefix}: invalid aggregate provenance for {name}")
                try:
                    matches = np.isclose(float(row.get("value")), value, rtol=1e-6, atol=1e-6)
                except (TypeError, ValueError):
                    matches = False
                if not matches:
                    errors.append(f"{prefix}: aggregate value mismatch for {name}")
        events = events_by_candidate.get(candidate_id, [])
        if not events:
            errors.append(f"{prefix}: no outcome event")
            continue
        for event in events:
            if "attenuation_db" in event:
                errors.append(f"{prefix}: ambiguous legacy attenuation field remains")
            if event.get("capture_phase") != "after_fast_fading_before_transfer":
                errors.append(f"{prefix}: invalid outcome capture phase")
            if event.get("temporal_role") != "outcome_only_not_same_frame_decision_input":
                errors.append(f"{prefix}: invalid outcome temporal role")
            for name in ("source", "target", "rb_indices"):
                if event.get(name) != decision.get(name):
                    errors.append(f"{prefix}: decision/outcome identity mismatch for {name}")
    return errors


def write_preflight_bundle(output_dir: Path, payloads: dict[str, Any]) -> dict[str, Any]:
    expected_payloads = set(REQUIRED_FILES) - {"manifest.json"}
    if set(payloads) != expected_payloads:
        raise ValueError(f"payload names differ: {sorted(set(payloads) ^ expected_payloads)}")
    if not bool(payloads["validation_report.json"].get("passed")):
        raise ValueError("validation report did not pass; refusing success manifest")
    temporal_errors = validate_temporal_payloads(payloads)
    if temporal_errors:
        raise ValueError(f"temporal evidence invalid: {temporal_errors}")
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
            _write_json(temporary / name, value)

        status_flags = copy.deepcopy(payloads["summary.json"]["status_flags"])
        source_files = (
            PROJECT_ROOT / "docs" / "superpowers" / "specs" / "2026-08-13-p2-single-step-collector-contract-design.md",
            PROJECT_ROOT / "docs" / "superpowers" / "specs" / "2026-08-13-p2-multistep-temporal-contract-design.md",
            CODE_ROOT / "scripts" / "run_p2_single_step_collector_preflight_v1.py",
            CODE_ROOT / "scripts" / "small_experiments" / "airfogsim_strict_dual_graph_preflight.py",
            CODE_ROOT / "tests" / "test_single_step_collector_contract_v1.py",
            CODE_ROOT / "tests" / "test_airfogsim_single_step_collector_v1.py",
            CODE_ROOT / "tests" / "test_airfogsim_contract_adapter.py",
            CODE_ROOT / "tests" / "test_airfogsim_cpu_inner_rule_v1.py",
            CODE_ROOT / "tests" / "test_cpu_inner_rule_v1.py",
            CODE_ROOT / "tests" / "test_information_edge_contract_v4.py",
            CODE_ROOT / "tests" / "test_run_p2_single_step_collector_preflight_v1.py",
            CODE_ROOT / "tests" / "small_experiments" / "test_airfogsim_strict_dual_graph_preflight.py",
            CODE_ROOT / "src" / "pi_jwm" / "single_step_collector_contract_v1.py",
            CODE_ROOT / "src" / "pi_jwm" / "airfogsim_single_step_collector_v1.py",
            CODE_ROOT / "src" / "pi_jwm" / "airfogsim_contract_adapter.py",
            CODE_ROOT / "src" / "pi_jwm" / "airfogsim_cpu_inner_rule_v1.py",
            CODE_ROOT / "src" / "pi_jwm" / "cpu_inner_rule_v1.py",
            CODE_ROOT / "src" / "pi_jwm" / "information_edge_contract_v4.py",
            REFERENCE_ROOT / "airfogsim" / "airfogsim_env.py",
            REFERENCE_ROOT / "airfogsim" / "manager" / "task_manager.py",
            REFERENCE_ROOT / "airfogsim" / "manager" / "channel_manager_cp.py",
        )
        missing_sources = [str(path) for path in source_files if not path.is_file()]
        if missing_sources:
            raise FileNotFoundError(f"manifest source files missing: {missing_sources}")
        manifest = {
            "schema_version": "PIJWM-P2-Single-Step-Manifest-v1",
            "collector_contract_version": COLLECTOR_CONTRACT_VERSION,
            "information_edge_contract_version": CONTRACT_VERSION,
            "status_flags": status_flags,
            "artifact_hashes": {
                name: file_hash(temporary / name) for name in sorted(expected_payloads)
            },
            "source_hashes": {
                str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): file_hash(path)
                for path in source_files
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return payloads["summary.json"]


def verify_preflight_bundle(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    missing = [name for name in REQUIRED_FILES if not (output_dir / name).is_file()]
    errors: list[str] = []
    if missing:
        errors.append(f"missing files: {missing}")
        return {"passed": False, "errors": errors}
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    payloads = {
        name: json.loads((output_dir / name).read_text(encoding="utf-8"))
        for name in REQUIRED_FILES
        if name != "manifest.json"
    }
    errors.extend(validate_temporal_payloads(payloads))
    for name, expected in manifest.get("artifact_hashes", {}).items():
        actual = file_hash(output_dir / name)
        if actual != expected:
            errors.append(f"artifact hash mismatch: {name}")
    for relative, expected in manifest.get("source_hashes", {}).items():
        actual_path = PROJECT_ROOT / Path(relative)
        if not actual_path.is_file() or file_hash(actual_path) != expected:
            errors.append(f"source hash mismatch: {relative}")
    flags = manifest.get("status_flags", {})
    for forbidden in (
        "v4_collector_implemented",
        "v4_dataset_complete",
        "model_training_started",
        "gpu_started",
        "locked_test_accessed",
        "candidate_rollout_planner_complete",
        "final_method_frozen",
        "training_eligible",
    ):
        if flags.get(forbidden) is not False:
            errors.append(f"unsafe status flag: {forbidden}")
    if flags.get("single_step_real_airfogsim_executed") is not True and not json.loads(
        (output_dir / "candidate_comparison.json").read_text(encoding="utf-8")
    ).get("test_fixture"):
        errors.append("real single-step execution flag is not true")
    return {"passed": not errors, "errors": errors}


def _plain(value: Any) -> float:
    if hasattr(value, "get"):
        value = value.get()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _task_snapshot(task: Any) -> dict[str, Any]:
    return {
        "task_id": str(task.getTaskId()),
        "task_node_id": str(task.getTaskNodeId()),
        "current_node_id": str(task.getCurrentNodeId()),
        "assigned_to": task.getAssignedTo(),
        "task_size": float(task.getTaskSize()),
        "transmitted_size": float(task.getTransmittedSize()),
        "task_cpu": float(task.getTaskCPU()),
        "computed_size": float(task.getComputedSize()),
        "deadline": float(task.getTaskDeadline()),
        "arrival_time": float(task.getTaskArrivalTime()),
    }


def _node_snapshot(env: Any) -> list[dict[str, Any]]:
    rows = []
    for collection_name, node_type in (
        ("vehicles", "V"), ("UAVs", "U"), ("RSUs", "I")
    ):
        for node_id, node in sorted(getattr(env, collection_name).items()):
            rows.append(
                {
                    "node_id": str(node_id),
                    "node_type": node_type,
                    "position": [float(value) for value in node.getPosition()],
                    "cpu": float(node.getFogProfile().get("cpu", 0.0)),
                }
            )
    return rows


def _waiting_tasks(env: Any) -> list[Any]:
    return sorted(
        (
            task
            for tasks in env.task_manager._waiting_to_offload_tasks.values()
            for task in tasks
            if env.task_manager.checkTaskDependency(task.getTaskNodeId(), task.getTaskId()) is True
        ),
        key=lambda task: (str(task.getTaskNodeId()), str(task.getTaskId())),
    )


def _pre_action_snapshot(env: Any, task: Any) -> dict[str, Any]:
    energy = {
        str(node_id): {key: _plain(value) for key, value in sorted(info.items())}
        for node_id, info in sorted(env.energy_manager._UAVs_energy_info.items())
    }
    return {
        "time": float(env.simulation_time),
        "selected_task": _task_snapshot(task),
        "nodes": _node_snapshot(env),
        "energy": energy,
        "n_rb": int(env.channel_manager.n_RB),
        "simulation_interval": float(env.simulation_interval),
        "python_random_state_hash": canonical_hash(repr(random.getstate())),
        "numpy_random_state_hash": canonical_hash(
            {
                "algorithm": np.random.get_state()[0],
                "keys": np.random.get_state()[1].tolist(),
                "position": int(np.random.get_state()[2]),
                "has_gauss": int(np.random.get_state()[3]),
                "cached_gaussian": float(np.random.get_state()[4]),
            }
        ),
    }


def _all_node_ids(env: Any) -> tuple[str, ...]:
    return tuple(sorted({*env.vehicles, *env.UAVs, *env.RSUs}))


def _choose_remote_targets(env: Any, source: str) -> tuple[str, str]:
    source_type = env._getNodeTypeById(source)
    candidates = [node_id for node_id in _all_node_ids(env) if node_id != source]
    wireless = [node_id for node_id in candidates if env._getNodeTypeById(node_id) in "VUI"]
    if source_type not in "VUI" or len(wireless) < 2:
        raise RuntimeError(f"fewer than two wireless remote candidates for source: {source}")
    ordered = sorted(
        wireless,
        key=lambda node_id: (env.getDistanceBetweenNodesById(source, node_id), node_id),
    )
    return ordered[0], ordered[1]


def _capture_rb_values(env: Any, profile: dict[str, Any]) -> dict[str, Any]:
    channel_type = str(profile["channel_type"])
    tx_idx = int(profile["tx_idx"])
    rx_idx = int(profile["rx_idx"])
    rb_indices = [int(value) for value in profile["RB_Nos"]]
    tx_type, rx_type = channel_type.split("2")
    csi = env.channel_manager.getCSI(tx_idx, rx_idx, tx_type, rx_type)
    rate = env.channel_manager.getRateByChannelType(tx_idx, rx_idx, channel_type, rb_indices)
    interference = getattr(env.channel_manager, f"{channel_type}_Interference")
    sinr = getattr(env.channel_manager, f"{channel_type}_SINR")
    outage = getattr(env.channel_manager, f"is_{channel_type}_outage")
    return {
        "outcome_channel_attenuation_db": [_plain(csi[rb]) for rb in rb_indices],
        "rate_per_s": [_plain(value) for value in rate],
        "interference_plus_noise_mw": [_plain(interference[tx_idx, rx_idx, rb]) for rb in rb_indices],
        "sinr_db": [_plain(sinr[tx_idx, rx_idx, rb]) for rb in rb_indices],
        "outage": [bool(_plain(outage[tx_idx, rx_idx, rb])) for rb in rb_indices],
    }


def _event_from_profile(env: Any, profile: dict[str, Any]) -> dict[str, Any]:
    task = profile["task"]
    route = list(task.getToOffloadRoute())
    source, target = str(task.getCurrentNodeId()), str(route[0])
    rb = _capture_rb_values(env, profile)
    planned = sum(rb["rate_per_s"]) * float(env.simulation_interval)
    remaining = max(float(task.getTaskSize()) - float(task.getTransmittedSize()), 0.0)
    return {
        "event_id": f"event::{task.getTaskId()}::{env.simulation_time:.6f}",
        "task_id": str(task.getTaskId()),
        "source": source,
        "target": target,
        "channel_type": str(profile["channel_type"]),
        "tx_idx": int(profile["tx_idx"]),
        "rx_idx": int(profile["rx_idx"]),
        "rb_indices": [int(value) for value in profile["RB_Nos"]],
        **rb,
        "planned_capacity": planned,
        "remaining_before": remaining,
        "delivered_data": min(planned, remaining),
        "time": float(env.simulation_time),
        "slot_seconds": float(env.simulation_interval),
        "noise_power_dbm": float(env.channel_manager.sig2_dB),
        "noise_power_mw": float(env.channel_manager.sig2),
        "rb_bandwidth_mhz": float(env.channel_manager.RB_bandwidth),
        "evidence": "direct_runtime_channel_event",
        "capture_phase": "after_fast_fading_before_transfer",
        "temporal_role": "outcome_only_not_same_frame_decision_input",
    }


def _capture_decision_time_channel(
    env: Any, source: str, target: str, rb_indices: list[int]
) -> dict[str, Any]:
    source_type = env._getNodeTypeById(source)
    target_type = env._getNodeTypeById(target)
    source_index = env._getNodeIdxById(source)
    target_index = env._getNodeIdxById(target)
    if source_type not in "VUI" or target_type not in "VUI":
        raise ValueError(f"decision-time CSI requires wireless endpoints: {source}, {target}")
    if source_index < 0 or target_index < 0:
        raise ValueError(f"decision-time CSI endpoint index missing: {source}, {target}")
    csi = env.channel_manager.getCSI(
        source_index, target_index, source_type, target_type
    )
    values = [_plain(csi[rb_index]) for rb_index in rb_indices]
    if not values or not np.isfinite(np.asarray(values, dtype=float)).all():
        raise ValueError("decision-time CSI is empty or non-finite")
    return {
        "capture_phase": "before_action_setters",
        "simulation_time": float(env.simulation_time),
        "source": source,
        "target": target,
        "rb_indices": list(rb_indices),
        "channel_attenuation_db": values,
        "source_method": "channel_manager.getCSI",
    }


def _load_runtime():
    import yaml
    from airfogsim import AirFogSimEnv
    from airfogsim.scheduler import CommunicationScheduler, ComputationScheduler, TaskScheduler
    import airfogsim_strict_dual_graph_preflight as preflight

    class ObservedAirFogSimEnv(AirFogSimEnv):
        def __init__(self, *args: Any, **kwargs: Any):
            self.pi_jwm_transfer_events: list[dict[str, Any]] = []
            self.pi_jwm_order: list[str] = []
            super().__init__(*args, **kwargs)

        def _updateWirelessCommunication(self):
            self.pi_jwm_order.append("wireless_communication")
            activated = self._allocate_communication_RBs(
                self.activated_offloading_tasks_with_RB_Nos
            )
            self._compute_communication_rate(activated)
            events = [_event_from_profile(self, profile) for profile in activated.values()]
            self._execute_communication(activated)
            sending, receiving = direct_transmission_totals(events)
            apply_transmission_totals(self.channel_manager, sending, receiving)
            self.pi_jwm_transfer_events.extend(events)

        def _updateWiredCommunication(self):
            self.pi_jwm_order.append("wired_communication")
            return super()._updateWiredCommunication()

        def _updateComputation(self):
            self.pi_jwm_order.append("computation")
            return super()._updateComputation()

        def _updateStorage(self):
            self.pi_jwm_order.append("storage")
            return super()._updateStorage()

        def _updateEnergy(self):
            self.pi_jwm_order.append("energy")
            return super()._updateEnergy()

    return yaml, ObservedAirFogSimEnv, TaskScheduler, CommunicationScheduler, ComputationScheduler, preflight


def _build_environment(seed: int, max_time: float):
    yaml, env_class, task_sched, comm_sched, comp_sched, preflight = _load_runtime()
    config = yaml.safe_load((EXAMPLE_DIR / "config.yaml").read_text(encoding="utf-8"))
    config = preflight.build_preflight_config(config, seed, max_time)
    np.random.seed(seed)
    random.seed(seed)
    env = env_class(config, interactive_mode=None)
    return env, task_sched, comm_sched, comp_sched, config


def _warm_to_branch(env: Any, max_steps: int = 30) -> tuple[Any, int]:
    for step in range(max_steps + 1):
        ready = _waiting_tasks(env)
        if ready:
            return ready[0], step
        env.alloc_cpu_callback = lambda _: {}
        env.step()
    raise RuntimeError(f"no ready offload task within {max_steps} warm-up steps")


def _energy_snapshot(env: Any) -> dict[str, dict[str, Any]]:
    return copy.deepcopy(env.energy_manager._UAVs_energy_info)


def _run_one_candidate(seed: int, candidate_id: str, target_rank: int) -> dict[str, Any]:
    old_cwd = Path.cwd()
    env = None
    try:
        os.chdir(EXAMPLE_DIR)
        env, task_sched, comm_sched, comp_sched, config = _build_environment(seed, 5.0)
        task, warmup_steps = _warm_to_branch(env)
        snapshot = _pre_action_snapshot(env, task)
        source = str(task.getTaskNodeId())
        task_id = str(task.getTaskId())
        targets = _choose_remote_targets(env, source)
        if target_rank not in (0, 1):
            raise ValueError(f"target_rank must be zero or one: {target_rank}")
        target = targets[target_rank]
        rb_assignments = tuple(
            RbAssignment(0, 0, 0, rb_index)
            for rb_index in range(int(env.channel_manager.n_RB))
        )
        edge_count = 1
        action = CandidateAction(
            candidate_id=candidate_id,
            offloads=(OffloadAction(source, task_id, target, (target,)),),
            rb_assignments=rb_assignments,
        )
        energy_before = _energy_snapshot(env)
        result = execute_candidate(
            env,
            action,
            task_ids=(task_id,),
            node_ids=_all_node_ids(env),
            edge_count=edge_count,
            flow_count=1,
            n_rb=int(env.channel_manager.n_RB),
            task_scheduler=task_sched,
            communication_scheduler=comm_sched,
            computation_scheduler=comp_sched,
            pre_action_observer=lambda: _capture_decision_time_channel(
                env,
                source,
                target,
                [row.rb_index for row in rb_assignments],
            ),
        )
        energy_after = _energy_snapshot(env)
        return {
            "candidate_id": candidate_id,
            "seed": seed,
            "config_hash": canonical_hash(config),
            "warmup_steps": warmup_steps,
            "pre_action_snapshot": snapshot,
            "pre_action_snapshot_hash": canonical_hash(snapshot),
            "action": {
                "candidate_id": candidate_id,
                "offloads": [
                    {
                        "task_node_id": source,
                        "task_id": task_id,
                        "target_node_id": target,
                        "route_nodes": [target],
                    }
                ],
                "assignment_coo": [list(row.as_tuple()) for row in rb_assignments],
            },
            "transfer_events": copy.deepcopy(env.pi_jwm_transfer_events),
            "decision_time_channel": copy.deepcopy(result.pre_action_observation),
            "temporal_trace": list(result.temporal_trace),
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
            "observed_order": list(env.pi_jwm_order[-5:]),
            "simulator_order_contract": list(result.simulator_order),
            "post_time": float(env.simulation_time),
        }
    finally:
        if env is not None:
            env.close()
        os.chdir(old_cwd)


def _field_audit(candidate: dict[str, Any]) -> dict[str, Any]:
    events = candidate["transfer_events"]
    if not events:
        return {
            "candidate_id": candidate["candidate_id"],
            "edge_count": 0,
            "fields": [],
            "validation": {"no_wireless_edge": True},
        }
    event = events[0]
    source = str(event["source"])
    target = str(event["target"])
    structure = {
        "edge_present": True,
        "edge_type": str(event["channel_type"]),
        "endpoint_ids": [source, target],
        "physical_edge_id": f"pe::{source}::{target}",
        "information_edge_id": f"ie::{source}::{target}",
        "cep_relation": "same_directed_endpoints",
    }
    if source == target or structure["edge_type"] not in {
        "V2V", "V2U", "V2I", "U2V", "U2U", "U2I", "I2V", "I2U", "I2I"
    }:
        raise ValueError(f"invalid E0 wireless structure: {structure}")
    rb_count = len(event["rb_indices"])
    decision_time_channel = candidate["decision_time_channel"]
    attenuation = np.asarray(
        decision_time_channel["channel_attenuation_db"], dtype=np.float32
    ).reshape(1, 1, rb_count)
    valid = np.ones_like(attenuation, dtype=bool)
    none = np.zeros_like(attenuation, dtype=np.int16)
    validate_field_values("pre_rb_optional.channel_attenuation_db", attenuation, valid, none)

    prior_shape = (1, 1)
    prior_valid = np.zeros(prior_shape, dtype=bool)
    prior_reason = np.full(prior_shape, MissingReason.NO_HISTORY.value, dtype=np.int16)
    validate_prev_field_timing(prior_valid, prior_reason)
    for name in (
        "pre_link.prev_active_flow_count",
        "pre_link.prev_effective_rate_per_s",
        "pre_link.prev_served_data",
    ):
        validate_field_values(name, np.zeros(prior_shape, dtype=np.float32), prior_valid, prior_reason)

    rate_by_rb = np.asarray(event["rate_per_s"], dtype=np.float64).reshape(1, 1, rb_count)
    rate = rate_by_rb.sum(axis=-1)
    served = np.asarray([[event["delivered_data"]]], dtype=np.float64)
    remaining = np.asarray([[event["remaining_before"]]], dtype=np.float64)
    validate_link_outcome(
        effective_rate_per_s=rate,
        served_data=served,
        slot_seconds=float(event["slot_seconds"]),
        remaining_before=remaining,
        assigned_rate_by_rb=rate_by_rb,
    )
    outage = np.asarray(event["outage"], dtype=bool).reshape(1, 1, rb_count)
    interference = np.asarray(event["interference_plus_noise_mw"], dtype=np.float64).reshape(1, 1, rb_count)
    validate_rb_outcome(
        rate_per_s=rate_by_rb,
        outage=outage,
        interference_plus_noise_mw=interference,
        noise_power_mw=float(event["noise_power_mw"]),
    )
    coo = np.asarray(candidate["action"]["assignment_coo"], dtype=np.int64).reshape((-1, 4))
    validate_assignment_coo(coo, (1, 1, 1, int(candidate["pre_action_snapshot"]["n_rb"])))
    return {
        "candidate_id": candidate["candidate_id"],
        "edge_count": 1,
        "fields": [
            {
                "name": "structure.edge_present",
                "value": True,
                "valid_mask": True,
                "missing_reason": MissingReason.NONE.name.lower(),
                "provenance": "direct_activated_profile",
            },
            {
                "name": "structure.edge_type",
                "value": structure["edge_type"],
                "valid_mask": True,
                "missing_reason": MissingReason.NONE.name.lower(),
                "provenance": "direct_activated_profile",
            },
            {
                "name": "structure.endpoint_index",
                "value": structure["endpoint_ids"],
                "valid_mask": True,
                "missing_reason": MissingReason.NONE.name.lower(),
                "provenance": "stable_node_ids_before_indexing",
            },
            {
                "name": "structure.cep_physical_edge_index",
                "value": structure["physical_edge_id"],
                "valid_mask": True,
                "missing_reason": MissingReason.NONE.name.lower(),
                "provenance": "same_directed_endpoints",
            },
            {
                "name": "pre_link.channel_attenuation_mean_db",
                "value": float(attenuation.mean()),
                "valid_mask": True,
                "missing_reason": MissingReason.NONE.name.lower(),
                "provenance": "derived_from_direct_decision_time_csi_before_setters",
            },
            {
                "name": "pre_link.channel_attenuation_std_db",
                "value": float(attenuation.std()),
                "valid_mask": True,
                "missing_reason": MissingReason.NONE.name.lower(),
                "provenance": "derived_from_direct_decision_time_csi_before_setters",
            },
            {
                "name": "pre_rb_optional.channel_attenuation_db",
                "value": attenuation.reshape(-1).tolist(),
                "valid_mask": [True] * rb_count,
                "missing_reason": [MissingReason.NONE.name.lower()] * rb_count,
                "provenance": "direct_decision_time_csi_before_setters",
            },
            *[
                {
                    "name": name,
                    "value": 0.0,
                    "valid_mask": False,
                    "missing_reason": MissingReason.NO_HISTORY.name.lower(),
                    "provenance": "first_frame_no_history",
                }
                for name in (
                    "pre_link.prev_active_flow_count",
                    "pre_link.prev_effective_rate_per_s",
                    "pre_link.prev_served_data",
                )
            ],
        ],
        "optional_current_rb_direct": True,
        "decision_time_channel": decision_time_channel,
        "temporal_trace": candidate["temporal_trace"],
        "structure": structure,
        "validation": {
            "e0_structure_and_cep": True,
            "masked_fields": True,
            "assignment_coo": True,
            "link_outcome": True,
            "rb_outcome": True,
        },
    }


def _validate_cpu_candidate(candidate: dict[str, Any]) -> bool:
    for callback_row in candidate["cpu_rows"]:
        if callback_row["rule_version"] != "PIJWM-CPU-Inner-Rule-v1":
            return False
        before = callback_row["computed_before"]
        after = callback_row["computed_after"]
        served = callback_row["served_work"]
        if set(before) != set(after) or set(before) != set(served):
            return False
        for task_id in before:
            if not np.isclose(
                after[task_id] - before[task_id],
                served[task_id],
                rtol=1e-9,
                atol=1e-10,
            ):
                return False
        for summary in callback_row["node_summaries"]:
            if summary["total_allocated_cpu"] > summary["capacity"] + 1e-10:
                return False
            expected = min(summary["capacity"], summary["total_demand_rate"])
            if not np.isclose(
                summary["total_allocated_cpu"], expected, rtol=1e-9, atol=1e-10
            ):
                return False
    return True


def _build_energy_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    sending, receiving = direct_transmission_totals(
        candidate["transfer_events"], amount_field="planned_capacity"
    )
    rows = []
    before = candidate["energy_before"]
    after = candidate["energy_after"]
    costs = candidate["energy_costs"]
    for uav_id in sorted(set(before) & set(after)):
        before_info, after_info = before[uav_id], after[uav_id]
        expected = (
            float(after_info.get("is_flying", False)) * costs["fly_unit_cost"]
            + float(after_info.get("is_hovering", False)) * costs["hover_unit_cost"]
            + float(after_info.get("using_sensor_num", 0)) * costs["sensing_unit_cost"]
            + float(after_info.get("sending_data_size", 0.0)) * costs["send_unit_cost"]
            + float(after_info.get("receiving_data_size", 0.0)) * costs["receive_unit_cost"]
        )
        observed = float(before_info["energy"]) - float(after_info["energy"])
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "uav_id": uav_id,
                "energy_before": float(before_info["energy"]),
                "energy_after": float(after_info["energy"]),
                "expected_consumption": expected,
                "observed_consumption": observed,
                "equation_residual": observed - expected,
                "event_sending_data_size": float(sending.get(uav_id, 0.0)),
                "event_receiving_data_size": float(receiving.get(uav_id, 0.0)),
                "simulator_sending_data_size": float(after_info.get("sending_data_size", 0.0)),
                "simulator_receiving_data_size": float(after_info.get("receiving_data_size", 0.0)),
            }
        )
    return rows


def _validate_energy_rows(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(
        abs(row["equation_residual"]) <= 1e-8
        and np.isclose(
            row["simulator_sending_data_size"],
            row["event_sending_data_size"],
            rtol=1e-9,
            atol=1e-10,
        )
        and np.isclose(
            row["simulator_receiving_data_size"],
            row["event_receiving_data_size"],
            rtol=1e-9,
            atol=1e-10,
        )
        for row in rows
    )


def build_real_payloads(seed: int = 0) -> dict[str, Any]:
    nearest = _run_one_candidate(seed, "nearest_remote", 0)
    alternate = _run_one_candidate(seed, "alternate_remote", 1)
    candidates = (nearest, alternate)
    same_snapshot = nearest["pre_action_snapshot_hash"] == alternate["pre_action_snapshot_hash"]
    same_config = nearest["config_hash"] == alternate["config_hash"]
    nearest_tasks = [task for row in nearest["cpu_rows"] for task in row["task_ids"]]
    alternate_tasks = [task for row in alternate["cpu_rows"] for task in row["task_ids"]]
    same_assignment = nearest["action"]["assignment_coo"] == alternate["action"]["assignment_coo"]
    target_differs = (
        nearest["action"]["offloads"][0]["target_node_id"]
        != alternate["action"]["offloads"][0]["target_node_id"]
    )
    cpu_targets = {
        row["candidate_id"]: row["cpu_rows"][0]["node_summaries"][0]["node_id"]
        if row["cpu_rows"] and row["cpu_rows"][0]["node_summaries"] else None
        for row in candidates
    }
    action_targets = {
        row["candidate_id"]: row["action"]["offloads"][0]["target_node_id"]
        for row in candidates
    }
    observable_difference = (
        nearest["transfer_events"] != alternate["transfer_events"]
        or nearest["cpu_rows"] != alternate["cpu_rows"]
    )
    expected_order = ["wireless_communication", "wired_communication", "computation", "storage", "energy"]
    order_valid = all(row["observed_order"] == expected_order for row in candidates)
    audits = [_field_audit(row) for row in candidates]
    energy_rows = [energy for row in candidates for energy in _build_energy_rows(row)]
    checks = {
        "same_seed_config": same_config,
        "same_pre_action_snapshot": same_snapshot,
        "same_rb_assignment_single_factor_target_change": same_assignment and target_differs,
        "one_real_step_per_candidate": all(row["post_time"] > row["pre_action_snapshot"]["time"] for row in candidates),
        "simulator_order_observed": order_valid,
        "candidate_difference_observable": observable_difference,
        "both_candidates_reach_cpu_callback": bool(nearest_tasks) and bool(alternate_tasks),
        "cpu_execution_node_matches_candidate_target": cpu_targets == action_targets,
        "cpu_rule_and_observed_delta_conservation": all(
            _validate_cpu_candidate(row) for row in candidates
        ),
        "both_candidates_have_positive_direct_transfer": all(
            row["transfer_events"]
            and sum(event["delivered_data"] for event in row["transfer_events"]) > 0.0
            for row in candidates
        ),
        "energy_equation_and_direct_event_inputs": _validate_energy_rows(energy_rows),
        "v4_field_audit_passed": all(all(audit["validation"].values()) for audit in audits),
    }
    if not all(checks.values()):
        raise RuntimeError(f"real P2 preflight validation failed: {checks}")

    flags = build_single_step_status_flags()
    flags["single_step_real_airfogsim_executed"] = True
    comparison = {
        "seed": seed,
        "same_pre_action_snapshot_hash": same_snapshot,
        "pre_action_snapshot_hash": nearest["pre_action_snapshot_hash"],
        "candidate_only_difference": "offload_target_node",
        "assignment_coo_identical": same_assignment,
        "nearest_compute_task_ids": nearest_tasks,
        "alternate_compute_task_ids": alternate_tasks,
        "nearest_transfer_event_count": len(nearest["transfer_events"]),
        "alternate_transfer_event_count": len(alternate["transfer_events"]),
        "observable_difference": observable_difference,
    }
    graph_candidates = []
    for row in candidates:
        action = row["action"]["offloads"][0]
        source, target = action["task_node_id"], action["target_node_id"]
        edges = [] if source == target else [{
            "physical_edge_id": f"pe::{source}::{target}",
            "information_edge_id": f"ie::{source}::{target}",
            "source": source,
            "target": target,
            "cep_relation": "same_directed_endpoints",
            "edge_class": "wireless",
        }]
        graph_candidates.append({
            "candidate_id": row["candidate_id"],
            "nodes": row["pre_action_snapshot"]["nodes"],
            "edges": edges,
            "task": row["pre_action_snapshot"]["selected_task"],
        })
    return {
        "candidate_comparison.json": comparison,
        "action_ledger.json": {"contract_version": COLLECTOR_CONTRACT_VERSION, "candidates": [row["action"] for row in candidates]},
        "transfer_events.json": {"candidates": [{"candidate_id": row["candidate_id"], "events": row["transfer_events"]} for row in candidates]},
        "single_step_graph.json": {"scope": "single_step_nontraining", "candidates": graph_candidates},
        "resource_bundle.json": {
            "cpu_rule_version": "PIJWM-CPU-Inner-Rule-v1",
            "energy_accounting": "direct_event_source_target_repair_before_native_energy_update",
            "candidates": [
                {
                    "candidate_id": row["candidate_id"],
                    "cpu_rows": row["cpu_rows"],
                    "energy_before": row["energy_before"],
                    "energy_after": row["energy_after"],
                    "observed_order": row["observed_order"],
                }
                for row in candidates
            ],
            "energy_rows": energy_rows,
        },
        "field_mask_audit.json": {
            "contract_version": CONTRACT_VERSION,
            "registry": build_field_registry(),
            "candidate_audits": audits,
            "training_eligible": False,
        },
        "validation_report.json": {"passed": True, "checks": checks},
        "summary.json": {
            "scope": "single_step_nontraining",
            "seed": seed,
            "candidate_ids": ["nearest_remote", "alternate_remote"],
            "required_files": list(REQUIRED_FILES),
            "status_flags": flags,
            "limitations": [
                "one real AirFogSim step per candidate only",
                "not a v4 trajectory dataset",
                "not a world-model candidate-rollout planner",
                "not training eligible",
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PI-JWM P2 real single-step CPU preflight")
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
    print(json.dumps({"summary": summary, "verification": verification}, ensure_ascii=False, indent=2))
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
