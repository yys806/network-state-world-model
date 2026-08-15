from __future__ import annotations

"""Paired-action causal sensitivity audit for PI-JWM small experiment 05."""

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_contract_adapter import (
    ADAPTER_VERSION,
    activated_transmission_events,
    apply_transmission_totals,
    capacity_safe_cpu_allocations,
    direct_transmission_totals,
)


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_pre_intervention_hash(run: dict[str, Any]) -> str:
    return canonical_json_hash(run.get("pre_intervention_state", {}))


def canonical_exogenous_hash(run: dict[str, Any], horizon: int | None = None) -> str:
    trajectory = list(run.get("exogenous_trajectory", []))
    if horizon is not None:
        trajectory = trajectory[: int(horizon)]
    return canonical_json_hash(trajectory)


def validate_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    intervention: dict[str, Any],
) -> dict[str, Any]:
    left_pre_hash = canonical_pre_intervention_hash(left)
    right_pre_hash = canonical_pre_intervention_hash(right)
    left_exogenous_hash = canonical_exogenous_hash(left)
    right_exogenous_hash = canonical_exogenous_hash(right)
    left_action = dict(left.get("action", {}))
    right_action = dict(right.get("action", {}))
    action_fields = sorted(set(left_action) | set(right_action))
    changed_action_fields = [
        field
        for field in action_fields
        if canonical_json_hash(left_action.get(field))
        != canonical_json_hash(right_action.get(field))
    ]
    expected_field = str(intervention.get("changed_action_field", ""))
    errors: list[str] = []
    if left_pre_hash != right_pre_hash:
        errors.append("pre_intervention_mismatch")
    if left_exogenous_hash != right_exogenous_hash:
        errors.append("exogenous_trajectory_mismatch")
    if changed_action_fields != [expected_field]:
        errors.append("unexpected_action_difference")
    if not left.get("action_feasible", False) or not right.get("action_feasible", False):
        errors.append("action_infeasible")
    if not left.get("action_applied", False) or not right.get("action_applied", False):
        errors.append("action_not_applied")
    if left.get("seed") != right.get("seed"):
        errors.append("seed_mismatch")
    if intervention.get("pair_kind") not in {"offload", "rb"}:
        errors.append("invalid_pair_kind")
    return {
        "pair_valid": not errors,
        "errors": sorted(set(errors)),
        "pair_kind": intervention.get("pair_kind"),
        "changed_action_fields": changed_action_fields,
        "expected_changed_action_field": expected_field,
        "left_pre_hash": left_pre_hash,
        "right_pre_hash": right_pre_hash,
        "left_exogenous_hash": left_exogenous_hash,
        "right_exogenous_hash": right_exogenous_hash,
        "left_action_hash": canonical_json_hash(left_action),
        "right_action_hash": canonical_json_hash(right_action),
    }


def _successor_at_horizon(run: dict[str, Any], horizon: int) -> dict[str, Any] | None:
    by_step = {
        int(row.get("offset_step", index + 1)): row
        for index, row in enumerate(run.get("successor_states", []))
    }
    return by_step.get(int(horizon))


def compute_horizon_effects(
    left: dict[str, Any],
    right: dict[str, Any],
    horizons: tuple[int, ...] = (1, 5, 20),
) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    numeric_fields = (
        "transmitted_data",
        "computed_data",
        "active_link_count",
        "rate_sum",
        "rb_use",
        "cpu_use",
        "delay",
    )
    categorical_fields = (
        "assigned_node_id",
        "current_node_id",
        "lifecycle",
        "completed",
    )
    for horizon in horizons:
        left_state = _successor_at_horizon(left, int(horizon))
        right_state = _successor_at_horizon(right, int(horizon))
        if left_state is None or right_state is None:
            effects.append(
                {
                    "horizon": int(horizon),
                    "state_available": False,
                    "any_successor_changed": False,
                }
            )
            continue
        row: dict[str, Any] = {
            "horizon": int(horizon),
            "state_available": True,
        }
        for field in categorical_fields:
            row[f"{field.removesuffix('_id')}_changed"] = (
                left_state.get(field) != right_state.get(field)
            )
        for field in numeric_fields:
            row[f"{field}_delta"] = float(left_state.get(field, 0.0)) - float(
                right_state.get(field, 0.0)
            )
        row["any_successor_changed"] = any(
            bool(row.get(f"{field.removesuffix('_id')}_changed"))
            for field in categorical_fields
        ) or any(abs(float(row[f"{field}_delta"])) > 1e-12 for field in numeric_fields)
        effects.append(row)
    return effects


def validate_action_sensitivity(
    pair_report: dict[str, Any],
    effects: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    if not pair_report.get("pair_valid", False):
        errors.append("invalid_pair")
    if not effects or not all(row.get("state_available", False) for row in effects):
        errors.append("missing_successor_state")
    changed_horizon_count = sum(
        bool(row.get("any_successor_changed", False)) for row in effects
    )
    if changed_horizon_count == 0:
        errors.append("no_successor_effect")
    return {
        "action_sensitivity_valid": not errors,
        "errors": errors,
        "changed_horizon_count": changed_horizon_count,
        "evaluated_horizon_count": len(effects),
    }


def build_exp05_corruption_report(
    left: dict[str, Any],
    right: dict[str, Any],
    intervention: dict[str, Any],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def evaluate(
        corruption_id: str,
        corrupt_left: dict[str, Any],
        corrupt_right: dict[str, Any],
        corrupt_intervention: dict[str, Any],
        expected_error: str,
        sensitivity_error: bool = False,
    ) -> None:
        pair_report = validate_pair(corrupt_left, corrupt_right, corrupt_intervention)
        effects = compute_horizon_effects(corrupt_left, corrupt_right)
        sensitivity = validate_action_sensitivity(pair_report, effects)
        errors = sensitivity["errors"] if sensitivity_error else pair_report["errors"]
        cases.append(
            {
                "corruption_id": corruption_id,
                "expected_error": expected_error,
                "detected": expected_error in errors,
            }
        )

    corrupt = copy.deepcopy(right)
    corrupt["pre_intervention_state"]["corrupt_marker"] = True
    evaluate("prestate_mismatch", left, corrupt, intervention, "pre_intervention_mismatch")

    corrupt = copy.deepcopy(right)
    extra_field = "rb_indices" if intervention["changed_action_field"] != "rb_indices" else "target_node_id"
    corrupt["action"][extra_field] = [999] if extra_field == "rb_indices" else "corrupt_target"
    evaluate("two_action_fields", left, corrupt, intervention, "unexpected_action_difference")

    corrupt = copy.deepcopy(right)
    corrupt["action_applied"] = False
    evaluate("action_not_applied", left, corrupt, intervention, "action_not_applied")

    corrupt = copy.deepcopy(right)
    corrupt.setdefault("exogenous_trajectory", []).append({"corrupt_marker": True})
    evaluate(
        "exogenous_divergence",
        left,
        corrupt,
        intervention,
        "exogenous_trajectory_mismatch",
    )

    corrupt = copy.deepcopy(right)
    corrupt["successor_states"] = copy.deepcopy(left.get("successor_states", []))
    evaluate(
        "copied_successor",
        left,
        corrupt,
        intervention,
        "no_successor_effect",
        sensitivity_error=True,
    )
    return {
        "all_corruptions_detected": all(row["detected"] for row in cases),
        "cases": cases,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_code_metadata() -> dict[str, Any]:
    code_root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "airfogsim_strict_dual_graph_preflight.py",
        code_root / "reference" / "AirFogSim" / "airfogsim" / "airfogsim_env.py",
        code_root / "reference" / "AirFogSim" / "airfogsim" / "airfogsim_algorithm.py",
        code_root / "reference" / "AirFogSim" / "airfogsim" / "manager" / "task_manager.py",
        code_root / "reference" / "AirFogSim" / "airfogsim" / "manager" / "channel_manager_cp.py",
        code_root / "reference" / "AirFogSim" / "airfogsim" / "entities" / "task.py",
        code_root / "src" / "pi_jwm" / "airfogsim_contract_adapter.py",
    ]
    hashes = {
        str(path.relative_to(code_root)).replace("\\", "/"): _sha256_file(path)
        for path in paths
        if path.exists()
    }
    return {"files": hashes, "aggregate_hash": canonical_json_hash(hashes)}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _report_markdown(validation: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        "# 小实验05：配对动作因果敏感性",
        "",
        "## 冻结结果",
        "",
        f"- `experiment_completed`: `{str(validation['experiment_completed']).lower()}`",
        f"- `action_sensitivity_ready`: `{str(validation['action_sensitivity_ready']).lower()}`",
        f"- seeds：`{summary['seeds']}`",
        f"- 卸载有效配对：`{validation['valid_effective_pairs_by_kind']['offload']}`",
        f"- RB有效配对：`{validation['valid_effective_pairs_by_kind']['rb']}`",
        f"- 破坏检测：`{str(validation['corruption_detection_passed']).lower()}`",
        "",
        "## 证据边界",
        "",
        "该实验只证明在相同seed、相同干预前状态与相同外生移动轨迹下，登记的合法卸载/RB动作会在AirFogSim中产生可辨识后继差异；不证明某个动作更优，也不证明PI-JWM已经学会该因果关系。",
    ]
    return "\n".join(lines) + "\n"


def run_exp05(
    output_dir: Path,
    seeds: list[int],
    max_time: float,
    pair_runner: Any,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    corruption_cases: list[dict[str, Any]] = []

    for seed in seeds:
        for pair_kind in ("offload", "rb"):
            expected_field = "target_node_id" if pair_kind == "offload" else "rb_indices"
            intervention = {
                "pair_kind": pair_kind,
                "changed_action_field": expected_field,
            }
            left = pair_runner(int(seed), float(max_time), pair_kind, "left")
            right = pair_runner(int(seed), float(max_time), pair_kind, "right")
            repeat_left = pair_runner(int(seed), float(max_time), pair_kind, "left")
            repeat_right = pair_runner(int(seed), float(max_time), pair_kind, "right")
            pair_id = f"pair::{pair_kind}::seed_{int(seed)}"
            pair_report = validate_pair(left, right, intervention)
            effects = compute_horizon_effects(left, right)
            sensitivity = validate_action_sensitivity(pair_report, effects)
            left_run_hash = canonical_json_hash(left)
            right_run_hash = canonical_json_hash(right)
            repeat_left_run_hash = canonical_json_hash(repeat_left)
            repeat_right_run_hash = canonical_json_hash(repeat_right)
            reproducible = (
                left_run_hash == repeat_left_run_hash
                and right_run_hash == repeat_right_run_hash
            )
            accepted = (
                pair_report["pair_valid"]
                and sensitivity["action_sensitivity_valid"]
                and reproducible
            )
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "seed": int(seed),
                    "pair_kind": pair_kind,
                    "changed_action_fields": pair_report["changed_action_fields"],
                    "pair_valid": pair_report["pair_valid"],
                    "pair_errors": pair_report["errors"],
                    "action_sensitivity_valid": sensitivity["action_sensitivity_valid"],
                    "sensitivity_errors": sensitivity["errors"],
                    "changed_horizon_count": sensitivity["changed_horizon_count"],
                    "reproducible": reproducible,
                    "accepted": accepted,
                    "left_pre_hash": pair_report["left_pre_hash"],
                    "right_pre_hash": pair_report["right_pre_hash"],
                    "left_exogenous_hash": pair_report["left_exogenous_hash"],
                    "right_exogenous_hash": pair_report["right_exogenous_hash"],
                    "left_action": left.get("action", {}),
                    "right_action": right.get("action", {}),
                    "left_run_hash": left_run_hash,
                    "right_run_hash": right_run_hash,
                    "repeat_left_run_hash": repeat_left_run_hash,
                    "repeat_right_run_hash": repeat_right_run_hash,
                }
            )
            for effect in effects:
                effect_rows.append({"pair_id": pair_id, "seed": int(seed), "pair_kind": pair_kind, **effect})
            trajectories.extend(
                [
                    {"pair_id": pair_id, "variant": "left", "run": left},
                    {"pair_id": pair_id, "variant": "right", "run": right},
                    {
                        "pair_id": pair_id,
                        "variant": "repeat_left",
                        "run": repeat_left,
                    },
                    {
                        "pair_id": pair_id,
                        "variant": "repeat_right",
                        "run": repeat_right,
                    },
                ]
            )
            corruption = build_exp05_corruption_report(left, right, intervention)
            corruption_cases.extend(
                {"pair_id": pair_id, **case} for case in corruption["cases"]
            )

    counts = {
        pair_kind: sum(row["accepted"] and row["pair_kind"] == pair_kind for row in pair_rows)
        for pair_kind in ("offload", "rb")
    }
    corruption_passed = bool(corruption_cases) and all(
        row["detected"] for row in corruption_cases
    )
    readiness = all(counts[kind] >= 2 for kind in counts) and corruption_passed
    validation = {
        "experiment_completed": True,
        "action_sensitivity_ready": readiness,
        "valid_effective_pairs_by_kind": counts,
        "corruption_detection_passed": corruption_passed,
        "total_pairs": len(pair_rows),
        "accepted_pairs": sum(row["accepted"] for row in pair_rows),
        "failed_pair_ids": [row["pair_id"] for row in pair_rows if not row["accepted"]],
    }
    bundle = {"pair_reports": pair_rows, "horizon_effects": effect_rows}
    summary = {
        "seeds": [int(seed) for seed in seeds],
        "max_time": float(max_time),
        "pair_count": len(pair_rows),
        "trajectory_count": len(trajectories),
        "effect_row_count": len(effect_rows),
    }
    config = {
        "schema_version": "AirFogSim-PIJWM-exp05-v1",
        "seeds": [int(seed) for seed in seeds],
        "max_time": float(max_time),
        "horizons": [1, 5, 20],
        "pair_kinds": ["offload", "rb"],
    }
    corruption_report = {
        "all_corruptions_detected": corruption_passed,
        "cases": corruption_cases,
    }

    _write_json(output_dir / "bundle.json", bundle)
    _write_json(output_dir / "trajectories.json", trajectories)
    _write_json(output_dir / "validation_report.json", validation)
    _write_json(output_dir / "corruption_report.json", corruption_report)
    _write_json(output_dir / "config_snapshot.json", config)
    _write_json(output_dir / "runtime_summary.json", summary)
    _write_csv(output_dir / "pair_reports.csv", pair_rows)
    _write_csv(output_dir / "horizon_effects.csv", effect_rows)
    (output_dir / "REPORT.md").write_text(
        _report_markdown(validation, summary), encoding="utf-8"
    )
    files = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": "AirFogSim-PIJWM-exp05-v1",
        "bundle_hash": canonical_json_hash(bundle),
        "config_hash": canonical_json_hash(config),
        "environment": {"python": sys.version, "platform": platform.platform()},
        "source_code": _source_code_metadata(),
        "files": {path.name: _sha256_file(path) for path in files},
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {**validation, "output_dir": str(output_dir)}


def _default_output_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "small_experiments"
        / "exp05_paired_action_sensitivity"
        / "paired_v1"
    )


def run_paired_seed(
    seed: int,
    max_time: float,
    pair_kind: str,
    variant: str,
) -> dict[str, Any]:
    if pair_kind not in {"offload", "rb"}:
        raise ValueError(f"Unknown pair kind: {pair_kind}")
    if variant not in {"left", "right"}:
        raise ValueError(f"Unknown pair variant: {variant}")

    code_root = Path(__file__).resolve().parents[2]
    reference_root = code_root / "reference" / "AirFogSim"
    example_dir = reference_root / "examples"
    script_dir = Path(__file__).resolve().parent
    for path in (script_dir, reference_root, example_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import numpy as np
    import yaml
    from airfogsim import AirFogSimEnv
    from airfogsim.scheduler import RewardScheduler, TaskScheduler
    from export_strict_actions_v0 import LoggingAlgorithmModule
    import airfogsim_strict_dual_graph_preflight as preflight

    def plain_float(value: Any) -> float:
        if hasattr(value, "get"):
            value = value.get()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    def physical_positions(env: Any) -> dict[str, list[float]]:
        rows: dict[str, list[float]] = {}
        for collection in (env.vehicles, env.UAVs, env.RSUs):
            for node_id, node in collection.items():
                position = node.getPosition()
                rows[str(node_id)] = [round(plain_float(value), 9) for value in position]
        return {key: rows[key] for key in sorted(rows)}

    def task_summary(task: Any) -> dict[str, Any]:
        return {
            "task_id": str(task.getTaskId()),
            "source_node_id": str(task.getTaskNodeId()),
            "assigned_node_id": task.getAssignedTo(),
            "current_node_id": str(task.getCurrentNodeId()),
            "lifecycle": str(task.task_lifecycle_state),
            "task_size": plain_float(task.getTaskSize()),
            "task_cpu": plain_float(task.getTaskCPU()),
            "transmitted_data": plain_float(task.getTransmittedSize()),
            "computed_data": plain_float(task.getComputedSize()),
        }

    class PairedObservedEnv(AirFogSimEnv):
        def __init__(self, *args: Any, **kwargs: Any):
            self.pi_watch_task_id: str | None = None
            self.pi_last_watch_link: dict[str, Any] = {
                "active_link_count": 0,
                "rate_sum": 0.0,
                "rb_use": 0,
                "rb_indices": [],
            }
            super().__init__(*args, **kwargs)

        def _updateWirelessCommunication(self) -> None:
            activated = self._allocate_communication_RBs(
                self.activated_offloading_tasks_with_RB_Nos
            )
            self._compute_communication_rate(activated)
            self.pi_last_watch_link = {
                "active_link_count": 0,
                "rate_sum": 0.0,
                "rb_use": 0,
                "rb_indices": [],
            }
            profile = activated.get(self.pi_watch_task_id)
            if profile is not None:
                rb_indices = [int(value) for value in profile.get("RB_Nos", [])]
                rates = self.channel_manager.getRateByChannelType(
                    profile["tx_idx"],
                    profile["rx_idx"],
                    profile["channel_type"],
                    rb_indices,
                )
                self.pi_last_watch_link = {
                    "active_link_count": 1,
                    "rate_sum": sum(plain_float(value) for value in rates),
                    "rb_use": len(rb_indices),
                    "rb_indices": rb_indices,
                }
            energy_events = activated_transmission_events(self, activated)
            self._execute_communication(activated)
            sending, receiving = direct_transmission_totals(energy_events)
            apply_transmission_totals(self.channel_manager, sending, receiving)

    class PairedPolicy(LoggingAlgorithmModule):
        def __init__(self) -> None:
            super().__init__(int(seed))
            self.current_step = 0
            self.intervention_step: int | None = None
            self.selected_task_id: str | None = None
            self.pre_intervention_state: dict[str, Any] = {}
            self.action: dict[str, Any] = {}
            self.action_history: list[dict[str, Any]] = []
            self.action_feasible = False
            self.action_applied = False
            self.selected_cpu_use = 0.0

        def _capture_prestate(self, env: Any, task: Any) -> None:
            tasks = [
                task_summary(item)
                for item in preflight.iter_airfogsim_tasks(env.task_manager)
            ]
            self.pre_intervention_state = {
                "seed": int(seed),
                "time": round(plain_float(env.simulation_time), 9),
                "step": int(self.current_step),
                "selected_task": task_summary(task),
                "tasks": sorted(tasks, key=lambda row: row["task_id"]),
                "node_positions": physical_positions(env),
                "action_history": copy.deepcopy(self.action_history),
            }
            self.intervention_step = int(self.current_step)
            self.selected_task_id = str(task.getTaskId())
            env.pi_watch_task_id = self.selected_task_id

        def scheduleReturning(self, env: Any) -> None:
            waiting = self.taskScheduler.getWaitingToReturnTaskInfos(env)
            for _, tasks in waiting.items():
                for task in tasks:
                    route = [str(task.getTaskNodeId())]
                    self.taskScheduler.setTaskReturnRoute(env, task.getTaskId(), route)

        def scheduleOffloading(self, env: Any) -> None:
            ready = self.taskScheduler.getAllToOffloadTaskInfos(
                env, check_dependency=True
            )
            ready = sorted(ready, key=lambda row: str(row["task_id"]))
            for row in ready:
                task_node_id = str(row["task_node_id"])
                task_id = str(row["task_id"])
                neighbors = self.entityScheduler.getNeighborNodeInfosById(
                    env,
                    task_node_id,
                    sorted_by="distance",
                    max_num=5,
                )
                neighbors = sorted(
                    neighbors,
                    key=lambda item: (
                        plain_float(item.get("distance", 0.0)),
                        str(item.get("id", "")),
                    ),
                )
                if not neighbors:
                    continue
                is_intervention = (
                    pair_kind == "offload"
                    and self.intervention_step is None
                    and len(neighbors) >= 2
                )
                target_index = 0
                if is_intervention and variant == "right":
                    target_index = 1
                target_id = str(neighbors[target_index]["id"])
                task = env.task_manager.getTaskByTaskId(task_id)
                if is_intervention and task is not None:
                    self._capture_prestate(env, task)
                    self.action_feasible = target_id != task_node_id
                    self.action = {
                        "task_id": task_id,
                        "target_node_id": target_id,
                        "rb_indices": [],
                    }
                applied = self.taskScheduler.setTaskOffloading(
                    env,
                    task_node_id,
                    task_id,
                    target_id,
                    route=[target_id],
                )
                if is_intervention:
                    self.action_applied = bool(applied) and (
                        task is not None and str(task.getAssignedTo()) == target_id
                    )
                if applied:
                    self.action_history.append(
                        {
                            "step": int(self.current_step),
                            "kind": "offload",
                            "task_id": task_id,
                            "target_node_id": target_id,
                        }
                    )

        def scheduleCommunication(self, env: Any) -> None:
            n_rb = int(self.commScheduler.getNumberOfRB(env))
            infos = sorted(
                self.taskScheduler.getAllOffloadingTaskInfos(env),
                key=lambda row: str(row["task_id"]),
            )[:n_rb]
            avg = max(1, n_rb // max(1, len(infos)))
            rb_cursor = 0
            for row in infos:
                task_id = str(row["task_id"])
                allocated = [(rb_cursor + index) % n_rb for index in range(avg)]
                rb_cursor = (rb_cursor + avg) % n_rb
                is_intervention = (
                    pair_kind == "rb"
                    and self.intervention_step is None
                    and n_rb >= 2
                )
                task = env.task_manager.getTaskByTaskId(task_id)
                if is_intervention and task is not None:
                    self._capture_prestate(env, task)
                    allocated = [0] if variant == "left" else [n_rb - 1]
                    self.action_feasible = all(0 <= value < n_rb for value in allocated)
                    self.action = {
                        "task_id": task_id,
                        "target_node_id": str(task.getAssignedTo()),
                        "rb_indices": list(allocated),
                    }
                self.commScheduler.setCommunicationWithRB(env, task_id, allocated)
                if is_intervention:
                    self.action_applied = (
                        env.activated_offloading_tasks_with_RB_Nos.get(task_id)
                        == allocated
                    )
                if pair_kind == "offload" and task_id == self.selected_task_id:
                    self.action["rb_indices"] = list(allocated)
                self.action_history.append(
                    {
                        "step": int(self.current_step),
                        "kind": "rb",
                        "task_id": task_id,
                        "rb_indices": list(allocated),
                    }
                )

        def scheduleComputing(self, env: Any) -> None:
            self.selected_cpu_use = 0.0

            def alloc_cpu_callback(
                computing_tasks: dict[str, list[Any]], **kwargs: Any
            ) -> dict[str, float]:
                allocations = capacity_safe_cpu_allocations(
                    env,
                    computing_tasks,
                    max_tasks_per_node=3,
                )
                self.selected_cpu_use = float(
                    allocations.get(str(self.selected_task_id), 0.0)
                )
                return allocations

            self.compScheduler.setComputingCallBack(env, alloc_cpu_callback)

    config_path = example_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = preflight.build_preflight_config(config, int(seed), float(max_time))
    config["pi_jwm_exp05"] = {
        "schema_version": "AirFogSim-PIJWM-exp05-v1",
        "contract_adapter_version": ADAPTER_VERSION,
        "cpu_semantics": "per_node_capacity_safe_equal_share_max_three",
        "channel_energy_semantics": "direct_event_source_target_accounting",
        "seed": int(seed),
        "max_time": float(max_time),
        "pair_kind": pair_kind,
        "variant": variant,
        "policy": "same_policy_single_registered_intervention",
    }
    np.random.seed(int(seed))
    random.seed(int(seed))
    old_cwd = Path.cwd()
    env = None
    algorithm: PairedPolicy | None = None
    successor_states: list[dict[str, Any]] = []
    exogenous_trajectory: list[dict[str, Any]] = []
    step = 0
    try:
        os.chdir(example_dir)
        env = PairedObservedEnv(config, interactive_mode=None)
        algorithm = PairedPolicy()
        algorithm.initialize(env)
        RewardScheduler.setModel(env, "REWARD", "1/task_delay")
        while not env.isDone():
            algorithm.current_step = step
            algorithm.scheduleStep(env)
            env.step()
            step += 1
            if algorithm.intervention_step is None:
                continue
            offset = step - algorithm.intervention_step
            if not 1 <= offset <= 20:
                continue
            exogenous_trajectory.append(
                {
                    "offset_step": int(offset),
                    "time": round(plain_float(env.simulation_time), 9),
                    "node_positions": physical_positions(env),
                }
            )
            task = env.task_manager.getTaskByTaskId(algorithm.selected_task_id)
            if task is None:
                successor_states.append(
                    {
                        "offset_step": int(offset),
                        "assigned_node_id": None,
                        "current_node_id": None,
                        "lifecycle": "missing",
                        "transmitted_data": 0.0,
                        "computed_data": 0.0,
                        "active_link_count": 0,
                        "rate_sum": 0.0,
                        "rb_use": 0,
                        "cpu_use": 0.0,
                        "completed": False,
                        "delay": 0.0,
                    }
                )
                continue
            link = env.pi_last_watch_link
            successor_states.append(
                {
                    "offset_step": int(offset),
                    "assigned_node_id": task.getAssignedTo(),
                    "current_node_id": str(task.getCurrentNodeId()),
                    "lifecycle": str(task.task_lifecycle_state),
                    "transmitted_data": plain_float(task.getTransmittedSize()),
                    "computed_data": plain_float(task.getComputedSize()),
                    "active_link_count": int(link.get("active_link_count", 0)),
                    "rate_sum": plain_float(link.get("rate_sum", 0.0)),
                    "rb_use": int(link.get("rb_use", 0)),
                    "cpu_use": plain_float(algorithm.selected_cpu_use),
                    "completed": bool(task.isFinished()),
                    "delay": plain_float(task.task_delay),
                }
            )
            if pair_kind == "rb" and offset == 1:
                algorithm.action_applied = algorithm.action_applied and (
                    list(link.get("rb_indices", []))
                    == list(algorithm.action.get("rb_indices", []))
                )
    finally:
        if env is not None:
            env.close()
        os.chdir(old_cwd)

    assert algorithm is not None
    return {
        "schema_version": "AirFogSim-PIJWM-exp05-run-v1",
        "seed": int(seed),
        "max_time": float(max_time),
        "pair_kind": pair_kind,
        "variant": variant,
        "pre_intervention_state": algorithm.pre_intervention_state,
        "exogenous_trajectory": exogenous_trajectory,
        "action": algorithm.action,
        "action_feasible": bool(algorithm.action_feasible),
        "action_applied": bool(algorithm.action_applied),
        "successor_states": successor_states,
        "runtime": {
            "steps": int(step),
            "intervention_step": algorithm.intervention_step,
            "intervention_time": algorithm.pre_intervention_state.get("time"),
            "selected_task_id": algorithm.selected_task_id,
            "policy": "same_policy_single_registered_intervention",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PI-JWM experiment 05 paired-action causal sensitivity."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-time", type=float, default=12.0)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()
    if "run_paired_seed" not in globals():
        parser.error("AirFogSim paired runner is not implemented")
    result = run_exp05(
        output_dir=args.output_dir,
        seeds=args.seeds,
        max_time=args.max_time,
        pair_runner=run_paired_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
