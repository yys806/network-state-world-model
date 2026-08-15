from __future__ import annotations

"""Build the auditable formal AirFogSim dataset used by PI-JWM."""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SMALL_EXPERIMENT_DIR = Path(__file__).resolve().parent / "small_experiments"
for path in (SRC_ROOT, SMALL_EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.airfogsim_dataset_v2 import (
    aggregate_metric_reports,
    build_multiseed_window_index,
    validate_multiseed_window_index,
)
from pi_jwm.airfogsim_dual_graph_v2 import (
    build_dual_graph_v2_bundle,
    validate_dual_graph_v2_bundle,
)
from pi_jwm.airfogsim_metrics_v2 import compute_airfogsim_metrics_v2
from pi_jwm.formal_airfogsim_dataset_v1 import (
    SCHEMA_VERSION as PROTOCOL_SCHEMA_VERSION,
    TrajectorySpec,
    build_formal_trajectory_specs,
    validate_formal_protocol,
)
from pi_jwm.formal_airfogsim_graph_v1 import validate_formal_graph_boundary


SCHEMA_VERSION = "PI-JWM-AirFogSim-formal-dataset-v1"
REQUIRED_PHYSICAL_DIRECTIONS = (
    "V2V",
    "V2U",
    "V2I",
    "U2V",
    "U2U",
    "U2I",
    "I2V",
    "I2U",
    "I2I",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _trajectory_directory(output_dir: Path, spec: TrajectorySpec) -> Path:
    root = output_dir / "locked_test" if spec.split == "locked_test" else output_dir
    return root / "trajectories" / spec.trajectory_id


def _verified_trajectory(directory: Path, spec: TrajectorySpec) -> bool:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("trajectory_id") != spec.trajectory_id:
            return False
        if manifest.get("split") != spec.split:
            return False
        for relative, record in manifest.get("files", {}).items():
            path = directory / relative
            if not path.is_file() or _sha256(path) != record.get("sha256"):
                return False
        return bool(manifest.get("files"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _metric_inputs(
    resource_bundle: Mapping[str, Any], graph_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "task_records": list(graph_bundle.get("task_nodes", [])),
        "transfer_events": list(graph_bundle.get("source_transfer_events", [])),
        "physical_node_snapshots": list(
            graph_bundle.get("source_physical_node_snapshots", [])
        ),
        "physical_edge_snapshots": list(
            graph_bundle.get("source_physical_edge_snapshots", [])
        ),
        "task_dag_edges": list(graph_bundle.get("task_dag_edges", [])),
        "dependency_data_flows": [],
        "task_ledger": list(resource_bundle.get("task_ledger", [])),
        "rb_ledger": list(resource_bundle.get("rb_ledger", [])),
        "cpu_ledger": list(resource_bundle.get("cpu_ledger", [])),
        "uav_energy_ledger": list(resource_bundle.get("uav_energy_ledger", [])),
        "dependency_ledger": [],
    }


def _build_graph(runtime: Mapping[str, Any], spec: TrajectorySpec) -> dict[str, Any]:
    source = dict(runtime["source_bundle"])
    graph = build_dual_graph_v2_bundle(
        trajectory_id=spec.trajectory_id,
        physical_nodes=source.get("physical_nodes", []),
        physical_edges=source.get("physical_edges", []),
        task_records=source.get("information_nodes", []),
        dag_edges=source.get("information_edges", []),
        transfer_events=source.get("transfer_events", []),
        task_snapshots=source.get("task_snapshots", []),
        offload_actions=source.get("offload_actions", []),
        return_actions=source.get("return_actions", []),
        rb_actions=source.get("rb_actions", []),
    )
    kept_node_ids = {str(row["id"]) for row in graph["physical_nodes"]}
    kept_edge_ids = {str(row["id"]) for row in graph["physical_edges"]}
    graph["source_physical_node_snapshots"] = [
        dict(row)
        for row in source.get("physical_node_snapshots", [])
        if str(row.get("id")) in kept_node_ids
    ]
    graph["source_physical_edge_snapshots"] = [
        dict(row)
        for row in source.get("physical_edge_snapshots", [])
        if str(row.get("id")) in kept_edge_ids
    ]
    graph["source_cpu_actions"] = [
        dict(row) for row in runtime["bundle"].get("cpu_ledger", [])
    ]
    validate_formal_graph_boundary(graph)
    return graph


def _time_values(graph: Mapping[str, Any]) -> list[float]:
    return sorted(
        {
            round(float(row["observed_time"]), 6)
            for row in graph.get("source_physical_node_snapshots", [])
            if row.get("observed_time") is not None
        }
    )


def _trajectory_checks(
    graph: Mapping[str, Any],
    resource: Mapping[str, Any],
    spec: TrajectorySpec,
) -> dict[str, bool]:
    time_keys = set(_time_values(graph))
    task_snapshots = list(graph.get("source_task_snapshots", []))
    snapshot_keys = [
        (str(row.get("id")), round(float(row["observed_time"]), 6))
        for row in task_snapshots
        if row.get("observed_time") is not None
    ]
    cpu_rows = list(resource.get("cpu_ledger", []))
    dag_edges = list(graph.get("task_dag_edges", []))
    return {
        "time_aligned_task_snapshots": bool(snapshot_keys)
        and len(snapshot_keys) == len(task_snapshots)
        and len(snapshot_keys) == len(set(snapshot_keys))
        and all(time in time_keys for _, time in snapshot_keys),
        "action_ledgers_present": all(
            key in graph
            for key in (
                "source_offload_actions",
                "source_return_actions",
                "source_rb_actions",
                "source_cpu_actions",
            )
        ),
        "cpu_policy_trace_valid": all(
            row.get("policy_id") == spec.cpu_policy
            and float(row.get("allocated_cpu", -1.0)) >= 0.0
            and 0.0 <= float(row.get("allocated_fraction", -1.0)) <= 1.0 + 1e-8
            for row in cpu_rows
        ),
        "dag_precedence_only": all(row.get("data_mb") is None for row in dag_edges)
        and not any(
            row.get("flow_type") == "dependency_data"
            for row in graph.get("information_edges", [])
        )
        and not resource.get("dependency_ledger", []),
    }


def _generate_trajectory(
    *,
    target: Path,
    spec: TrajectorySpec,
    runtime_runner: Callable[..., Mapping[str, Any]],
    resource_validator: Callable[[dict[str, Any]], Mapping[str, Any]],
    max_time: float,
) -> None:
    runtime = dict(runtime_runner(spec, max_time=float(max_time)))
    resource = dict(runtime["bundle"])
    graph = _build_graph(runtime, spec)
    graph_validation = validate_dual_graph_v2_bundle(graph)
    resource_validation = dict(resource_validator(resource))
    metric_report = compute_airfogsim_metrics_v2(_metric_inputs(resource, graph))
    checks = _trajectory_checks(graph, resource, spec)
    times = _time_values(graph)
    observed_directions = sorted(
        {
            str(row["kind"])
            for row in graph.get("source_physical_edge_snapshots", [])
            if row.get("kind") is not None
        }
    )
    trajectory_summary = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": spec.trajectory_id,
        "seed": spec.seed,
        "split": spec.split,
        "cpu_policy": spec.cpu_policy,
        "scenario": spec.scenario.to_dict(),
        "time_values": times,
        "observed_steps": len(times),
        "observed_physical_directions": observed_directions,
        "cpu_action_count": len(resource.get("cpu_ledger", [])),
        "checks": checks,
        "graph_counts": graph_validation.get("counts", {}),
    }

    partial = target.parent / f".partial__{target.name}"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    _write_json(partial / "config_snapshot.json", runtime.get("config", {}))
    _write_json(partial / "dual_graph_v2_bundle.json", graph)
    _write_json(partial / "resource_bundle.json", resource)
    _write_json(partial / "graph_validation.json", graph_validation)
    _write_json(partial / "resource_validation.json", resource_validation)
    _write_json(partial / "metric_results.json", metric_report)
    _write_json(partial / "runtime_summary.json", runtime.get("runtime_summary", {}))
    _write_json(partial / "trajectory_summary.json", trajectory_summary)
    files = {
        path.name: _file_record(path)
        for path in sorted(partial.iterdir())
        if path.is_file()
    }
    _write_json(
        partial / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "trajectory_id": spec.trajectory_id,
            "seed": spec.seed,
            "split": spec.split,
            "files": files,
        },
    )
    if target.exists():
        raise RuntimeError(
            f"trajectory directory exists but failed manifest verification: {target}"
        )
    partial.rename(target)


def _read_trajectory(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    def read(name: str) -> dict[str, Any]:
        return json.loads((directory / name).read_text(encoding="utf-8"))

    return (
        read("trajectory_summary.json"),
        read("graph_validation.json"),
        read("resource_validation.json"),
        read("metric_results.json"),
    )


def _report(summary: Mapping[str, Any], validation: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# PI-JWM 正式 AirFogSim 数据集 v1",
            "",
            f"- 正式数据构建就绪：{validation['formal_dataset_ready']}",
            "- 正式训练就绪：False（张量化与训练前冻结仍单独执行）",
            f"- 轨迹：{summary['trajectory_count']} 条，其中锁定测试 {summary['locked_test_trajectory_count']} 条",
            f"- 可用窗口：{summary['window_count']}；锁定测试窗口：{summary['locked_test_window_count']}",
            f"- 场景：{summary['scenario_count']}；CPU 策略：{', '.join(summary['cpu_policies'])}",
            "",
            "锁定测试轨迹和窗口位于 `locked_test/`，未进入指标汇总。",
            "DAG 仅表示 AirFogSim 原生先后依赖，不添加虚构的数据载荷。",
            "",
        ]
    )


def build_formal_dataset(
    *,
    output_dir: Path,
    specs: Sequence[TrajectorySpec],
    runtime_runner: Callable[..., Mapping[str, Any]],
    resource_validator: Callable[[dict[str, Any]], Mapping[str, Any]],
    max_time: float,
    history_steps: int,
    horizon_steps: int,
    required_physical_directions: Sequence[str] = REQUIRED_PHYSICAL_DIRECTIONS,
) -> dict[str, Any]:
    """Build or resume the complete formal-v1 trajectory collection."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_specs = sorted(specs, key=lambda row: row.seed)
    protocol_validation = validate_formal_protocol(ordered_specs)
    if not protocol_validation["protocol_valid"]:
        raise ValueError(
            f"invalid formal dataset protocol: {protocol_validation['failed_checks']}"
        )

    _write_json(
        output_dir / "protocol.json",
        {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "trajectory_specs": [row.to_dict() for row in ordered_specs],
            "validation": protocol_validation,
        },
    )

    generated = 0
    reused = 0
    for spec in ordered_specs:
        target = _trajectory_directory(output_dir, spec)
        target.parent.mkdir(parents=True, exist_ok=True)
        if _verified_trajectory(target, spec):
            reused += 1
            continue
        _generate_trajectory(
            target=target,
            spec=spec,
            runtime_runner=runtime_runner,
            resource_validator=resource_validator,
            max_time=max_time,
        )
        generated += 1

    times_by_seed: dict[int, list[float]] = {}
    split_by_seed: dict[int, str] = {}
    metrics_by_seed: dict[int, dict[str, Any]] = {}
    trajectory_rows: list[dict[str, Any]] = []
    graph_ready = True
    resource_ready = True
    all_checks: list[dict[str, bool]] = []
    observed_directions: set[str] = set()
    total_cpu_actions = 0
    for spec in ordered_specs:
        directory = _trajectory_directory(output_dir, spec)
        summary, graph_validation, resource_validation, metric_report = _read_trajectory(
            directory
        )
        times_by_seed[spec.seed] = list(summary["time_values"])
        split_by_seed[spec.seed] = spec.split
        if spec.split != "locked_test":
            metrics_by_seed[spec.seed] = metric_report
        graph_ready &= bool(graph_validation.get("dual_graph_v2_ready", False))
        resource_ready &= bool(resource_validation.get("conservation_ready", False))
        all_checks.append(dict(summary["checks"]))
        observed_directions.update(summary["observed_physical_directions"])
        total_cpu_actions += int(summary["cpu_action_count"])
        trajectory_rows.append(
            {
                "trajectory_id": spec.trajectory_id,
                "seed": spec.seed,
                "split": spec.split,
                "scenario_id": spec.scenario.scenario_id,
                "cpu_policy": spec.cpu_policy,
                "observed_steps": summary["observed_steps"],
                "cpu_action_count": summary["cpu_action_count"],
                **summary["graph_counts"],
            }
        )

    windows = build_multiseed_window_index(
        times_by_seed,
        split_by_seed,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
    )
    window_validation = validate_multiseed_window_index(
        windows,
        times_by_seed,
        split_by_seed,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
    )
    unlocked_windows = [row for row in windows if row["split"] != "locked_test"]
    locked_windows = [row for row in windows if row["split"] == "locked_test"]
    _write_csv(output_dir / "window_index.csv", unlocked_windows)
    _write_csv(output_dir / "locked_test" / "window_index.csv", locked_windows)
    _write_csv(output_dir / "trajectory_index.csv", trajectory_rows)

    unlocked_splits = {
        seed: split
        for seed, split in split_by_seed.items()
        if split != "locked_test"
    }
    metric_rows, metric_summary = aggregate_metric_reports(
        metrics_by_seed, unlocked_splits
    )
    _write_csv(output_dir / "metrics_by_trajectory.csv", metric_rows)
    _write_json(output_dir / "metric_summary.json", metric_summary)

    checks = {
        "protocol_valid": bool(protocol_validation["protocol_valid"]),
        "all_dual_graphs_ready": graph_ready,
        "all_resource_ledgers_ready": resource_ready,
        "window_index_valid": bool(window_validation["window_index_valid"]),
        "time_aligned_task_snapshots": all(
            row["time_aligned_task_snapshots"] for row in all_checks
        ),
        "action_ledgers_present": all(row["action_ledgers_present"] for row in all_checks),
        "cpu_policy_trace_valid": total_cpu_actions > 0
        and all(row["cpu_policy_trace_valid"] for row in all_checks),
        "dag_precedence_only": all(row["dag_precedence_only"] for row in all_checks),
        "required_physical_directions_covered": set(required_physical_directions).issubset(
            observed_directions
        ),
        "locked_test_excluded_from_metrics": all(
            row.get("split") != "locked_test" for row in metric_rows
        ),
    }
    formal_dataset_ready = all(checks.values())
    validation = {
        "schema_version": SCHEMA_VERSION,
        "formal_dataset_ready": formal_dataset_ready,
        "formal_training_ready": False,
        "checks": checks,
        "window_validation": window_validation,
        "formal_training_blockers": [
            "tensor_contract_not_frozen",
            "training_statistics_not_fitted",
            "locked_test_remains_sealed_until_model_freeze",
        ],
    }
    split_counts = Counter(row.split for row in ordered_specs)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generation_mode": "audited_formal_airfogsim_v1",
        "trajectory_count": len(ordered_specs),
        "unlocked_trajectory_count": len(ordered_specs) - split_counts["locked_test"],
        "locked_test_trajectory_count": split_counts["locked_test"],
        "scenario_count": len({row.scenario.scenario_id for row in ordered_specs}),
        "split_counts": dict(sorted(split_counts.items())),
        "metric_splits": sorted(set(unlocked_splits.values())),
        "cpu_policies": sorted({row.cpu_policy for row in ordered_specs}),
        "history_steps": int(history_steps),
        "horizon_steps": int(horizon_steps),
        "window_count": len(unlocked_windows),
        "locked_test_window_count": len(locked_windows),
        "generated_trajectory_count": generated,
        "reused_trajectory_count": reused,
        "observed_physical_directions": sorted(observed_directions),
        "formal_training_ready": False,
    }
    _write_json(output_dir / "dataset_summary.json", summary)
    _write_json(output_dir / "validation_report.json", validation)
    (output_dir / "REPORT.md").write_text(
        _report(summary, validation), encoding="utf-8"
    )
    top_level_files = [
        path
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    ]
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "formal_dataset_ready": formal_dataset_ready,
            "generation_completed": len(ordered_specs) == 60,
            "field_masks_valid": checks["time_aligned_task_snapshots"]
            and checks["action_ledgers_present"],
            "splits_frozen": checks["protocol_valid"]
            and checks["locked_test_excluded_from_metrics"],
            "source_manifest_present": all(
                _verified_trajectory(_trajectory_directory(output_dir, spec), spec)
                for spec in ordered_specs
            ),
            "files": {path.name: _file_record(path) for path in top_level_files},
        },
    )
    return {
        "formal_dataset_ready": formal_dataset_ready,
        "formal_training_ready": False,
        "output_dir": str(output_dir),
        "trajectory_count": len(ordered_specs),
        "generated_trajectory_count": generated,
        "reused_trajectory_count": reused,
        "window_count": len(unlocked_windows),
    }


def _runtime_components() -> tuple[Callable[..., Mapping[str, Any]], Callable[..., Mapping[str, Any]]]:
    from formal_airfogsim_runtime_v1 import run_formal_airfogsim_trajectory
    import task_resource_conservation_audit as conservation_module

    return run_formal_airfogsim_trajectory, conservation_module.validate_exp04_bundle


def _default_output_dir() -> Path:
    return CODE_ROOT / "artifacts" / "datasets" / "airfogsim_formal_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PI-JWM formal AirFogSim dataset v1.")
    parser.add_argument("--max-time", type=float, default=30.0)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()
    runtime_runner, resource_validator = _runtime_components()
    result = build_formal_dataset(
        output_dir=args.output_dir,
        specs=build_formal_trajectory_specs(),
        runtime_runner=runtime_runner,
        resource_validator=resource_validator,
        max_time=args.max_time,
        history_steps=args.history,
        horizon_steps=args.horizon,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["formal_dataset_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
