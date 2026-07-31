from __future__ import annotations

"""Build a seed-isolated AirFogSim dual-graph v2 development dataset."""

import argparse
import csv
import hashlib
import json
import sys
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
    SCHEMA_VERSION,
    aggregate_metric_reports,
    build_multiseed_window_index,
    validate_multiseed_window_index,
)
from pi_jwm.airfogsim_dual_graph_v2 import (
    build_dual_graph_v2_bundle,
    validate_dual_graph_v2_bundle,
)
from pi_jwm.airfogsim_metrics_v2 import compute_airfogsim_metrics_v2


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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _default_splits(seeds: Sequence[int]) -> dict[int, str]:
    labels = ("dev_train", "dev_validation", "dev_calibration")
    ordered = sorted({int(seed) for seed in seeds})
    return {
        seed: labels[index] if index < len(labels) else "dev_train"
        for index, seed in enumerate(ordered)
    }


def _metric_inputs(resource_bundle: Mapping[str, Any], graph_bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_records": list(graph_bundle.get("task_nodes", [])),
        "transfer_events": list(graph_bundle.get("source_transfer_events", [])),
        "physical_node_snapshots": list(graph_bundle.get("source_physical_node_snapshots", [])),
        "physical_edge_snapshots": list(graph_bundle.get("source_physical_edge_snapshots", [])),
        "task_dag_edges": list(graph_bundle.get("task_dag_edges", [])),
        "dependency_data_flows": [
            row
            for row in graph_bundle.get("information_edges", [])
            if row.get("flow_type") == "dependency_data"
        ],
        "task_ledger": list(resource_bundle.get("task_ledger", [])),
        "rb_ledger": list(resource_bundle.get("rb_ledger", [])),
        "cpu_ledger": list(resource_bundle.get("cpu_ledger", [])),
        "uav_energy_ledger": list(resource_bundle.get("uav_energy_ledger", [])),
        "dependency_ledger": [],
    }


def _report(summary: Mapping[str, Any], validation: Mapping[str, Any]) -> str:
    lines = [
        "# AirFogSim双图v2多seed开发数据集",
        "",
        f"- 开发数据集就绪：{validation['development_dataset_ready']}",
        f"- 正式训练就绪：{validation['formal_training_ready']}",
        f"- seeds：{', '.join(str(seed) for seed in summary['seeds'])}",
        f"- 轨迹数：{summary['seed_count']}；窗口数：{summary['window_count']}",
        f"- 窗口：历史{summary['history_steps']}步，未来{summary['horizon_steps']}步",
        "",
        "## 每seed结果",
        "",
        "| seed | split | 物理节点/边 | 信息代理/流 | 任务/DAG | 窗口 | 完成率 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["seed_summaries"]:
        lines.append(
            "| {seed} | {split} | {physical_nodes}/{physical_edges} | "
            "{information_nodes}/{information_edges} | {task_nodes}/{task_dag_edges} | "
            "{window_count} | {task_completion_rate} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 跨seed指标",
            "",
            "| 指标 | 均值 | 标准差 | 最小值 | 最大值 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, values in summary["key_metric_summary"].items():
        lines.append(
            f"| {name} | {values['mean']} | {values['sample_std']} | "
            f"{values['minimum']} | {values['maximum']} |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 本数据集是single-pass development smoke，不包含锁定测试集。",
            "- 图为可变规模结构；固定张量的节点/边/流padding和mask尚未冻结。",
            "- 原生DAG没有依赖载荷，不能把precedence边当作dependency_data传输。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_multiseed_dataset(
    *,
    output_dir: Path,
    seeds: Sequence[int],
    max_time: float,
    history_steps: int,
    horizon_steps: int,
    runtime_runner: Callable[[int, float], Mapping[str, Any]],
    resource_validator: Callable[[dict[str, Any]], Mapping[str, Any]],
    split_by_seed: Mapping[int, str] | None = None,
    required_physical_directions: Sequence[str] = REQUIRED_PHYSICAL_DIRECTIONS,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_seeds = sorted({int(seed) for seed in seeds})
    if not ordered_seeds:
        raise ValueError("at least one seed is required")
    splits = dict(split_by_seed or _default_splits(ordered_seeds))
    if set(splits) != set(ordered_seeds):
        raise ValueError("split_by_seed must assign every requested seed exactly once")

    times_by_seed: dict[int, list[float]] = {}
    metric_reports: dict[int, dict[str, Any]] = {}
    graph_validations: dict[int, dict[str, Any]] = {}
    resource_validations: dict[int, dict[str, Any]] = {}
    trajectory_checks: dict[int, dict[str, bool]] = {}
    seed_summaries: list[dict[str, Any]] = []

    for seed in ordered_seeds:
        runtime = dict(runtime_runner(seed, float(max_time)))
        source = dict(runtime["source_bundle"])
        resource = dict(runtime["bundle"])
        physical_nodes = list(source.get("physical_nodes", []))
        trajectory_id = (
            str(physical_nodes[0].get("trajectory_id"))
            if physical_nodes and physical_nodes[0].get("trajectory_id") is not None
            else f"airfogsim_v2_seed_{seed}"
        )
        graph = build_dual_graph_v2_bundle(
            trajectory_id=trajectory_id,
            physical_nodes=physical_nodes,
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
            row
            for row in source.get("physical_node_snapshots", [])
            if str(row.get("id")) in kept_node_ids
        ]
        graph["source_physical_edge_snapshots"] = [
            row
            for row in source.get("physical_edge_snapshots", [])
            if str(row.get("id")) in kept_edge_ids
        ]
        trajectory_checks[seed] = {
            "time_aligned_task_snapshots": bool(graph["source_task_snapshots"]),
            "action_ledgers_frozen": all(
                key in source
                for key in ("offload_actions", "return_actions", "rb_actions")
            ),
        }
        graph_validation = validate_dual_graph_v2_bundle(graph)
        resource_validation = dict(resource_validator(resource))
        metric_report = compute_airfogsim_metrics_v2(_metric_inputs(resource, graph))

        seed_dir = output_dir / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        _write_json(seed_dir / "config_snapshot.json", runtime.get("config", {}))
        _write_json(seed_dir / "dual_graph_v2_bundle.json", graph)
        _write_json(seed_dir / "resource_bundle.json", resource)
        _write_json(seed_dir / "graph_validation.json", graph_validation)
        _write_json(seed_dir / "resource_validation.json", resource_validation)
        _write_json(seed_dir / "metric_results.json", metric_report)
        _write_json(seed_dir / "runtime_summary.json", runtime.get("runtime_summary", {}))
        seed_files = sorted(path for path in seed_dir.iterdir() if path.name != "manifest.json")
        _write_json(
            seed_dir / "manifest.json",
            {
                "seed": seed,
                "split": splits[seed],
                "generation_mode": "single_pass_development_smoke",
                "files": {path.name: _sha256(path) for path in seed_files},
            },
        )

        time_values = sorted(
            {
                float(row["observed_time"])
                for row in graph["source_physical_node_snapshots"]
                if row.get("observed_time") is not None
            }
        )
        time_keys = {round(value, 9) for value in time_values}
        task_snapshot_keys = [
            (str(row.get("id")), round(float(row["observed_time"]), 9))
            for row in graph["source_task_snapshots"]
            if row.get("observed_time") is not None
        ]
        task_times_aligned = (
            bool(task_snapshot_keys)
            and len(task_snapshot_keys) == len(graph["source_task_snapshots"])
            and len(task_snapshot_keys) == len(set(task_snapshot_keys))
            and all(time_value in time_keys for _, time_value in task_snapshot_keys)
        )
        observed_directions = sorted(
            {
                str(row.get("kind"))
                for row in graph["source_physical_edge_snapshots"]
                if row.get("kind") is not None
            }
        )
        trajectory_checks[seed]["time_aligned_task_snapshots"] = task_times_aligned
        trajectory_checks[seed]["required_physical_directions_covered"] = set(
            required_physical_directions
        ).issubset(observed_directions)
        times_by_seed[seed] = time_values
        metric_reports[seed] = metric_report
        graph_validations[seed] = graph_validation
        resource_validations[seed] = resource_validation
        counts = graph_validation["counts"]
        metric_by_name = {row["name"]: row for row in metric_report["metrics"]}
        seed_summaries.append(
            {
                "seed": seed,
                "split": splits[seed],
                **counts,
                "observed_steps": len(time_values),
                "physical_edge_directions": observed_directions,
                "task_snapshot_count": len(graph["source_task_snapshots"]),
                "offload_action_count": len(graph["source_offload_actions"]),
                "return_action_count": len(graph["source_return_actions"]),
                "rb_action_count": len(graph["source_rb_actions"]),
                "task_completion_rate": metric_by_name["task_completion_rate"]["value"],
            }
        )

    windows = build_multiseed_window_index(
        times_by_seed,
        splits,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
    )
    window_validation = validate_multiseed_window_index(
        windows,
        times_by_seed,
        splits,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
    )
    window_counts = {seed: 0 for seed in ordered_seeds}
    for row in windows:
        window_counts[int(row["seed"])] += 1
    for row in seed_summaries:
        row["window_count"] = window_counts[int(row["seed"])]

    metric_rows, metric_summary = aggregate_metric_reports(metric_reports, splits)
    key_metric_names = (
        "task_completion_rate",
        "successful_task_delay_p95",
        "information_throughput",
        "rb_utilization",
        "cpu_utilization",
        "uav_energy_per_completed_task",
        "completion_fairness_jain",
    )
    key_metric_summary = {
        name: metric_summary[name]
        for name in key_metric_names
    }
    all_graphs_ready = all(row["dual_graph_v2_ready"] for row in graph_validations.values())
    all_resources_ready = all(row.get("conservation_ready", False) for row in resource_validations.values())
    time_aligned_task_snapshots = all(
        trajectory_checks[seed]["time_aligned_task_snapshots"]
        for seed in ordered_seeds
    )
    action_ledgers_frozen = all(
        trajectory_checks[seed]["action_ledgers_frozen"]
        for seed in ordered_seeds
    )
    required_physical_directions_covered = all(
        trajectory_checks[seed]["required_physical_directions_covered"]
        for seed in ordered_seeds
    )
    development_ready = (
        all_graphs_ready
        and all_resources_ready
        and window_validation["window_index_valid"]
        and time_aligned_task_snapshots
        and action_ledgers_frozen
        and required_physical_directions_covered
    )
    formal_blockers = [
        "only_development_seeds",
        "no_locked_test_split",
        "variable_graph_tensorization_not_frozen",
        "single_pass_multiseed_generation",
    ]
    validation = {
        "schema_version": SCHEMA_VERSION,
        "development_dataset_ready": development_ready,
        "formal_training_ready": False,
        "checks": {
            "all_dual_graphs_ready": all_graphs_ready,
            "all_resource_ledgers_ready": all_resources_ready,
            "window_index_valid": window_validation["window_index_valid"],
            "time_aligned_task_snapshots": time_aligned_task_snapshots,
            "action_ledgers_frozen": action_ledgers_frozen,
            "required_physical_directions_covered": required_physical_directions_covered,
            "seed_split_isolated": not any(
                row["name"] == "seed_split_isolation" and not row["passed"]
                for row in window_validation["checks"]
            ),
        },
        "window_validation": window_validation,
        "formal_training_blockers": formal_blockers,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generation_mode": "single_pass_development_smoke",
        "seeds": ordered_seeds,
        "seed_count": len(ordered_seeds),
        "split_by_seed": {str(seed): splits[seed] for seed in ordered_seeds},
        "history_steps": int(history_steps),
        "horizon_steps": int(horizon_steps),
        "window_count": len(windows),
        "seed_summaries": seed_summaries,
        "key_metric_summary": key_metric_summary,
        "formal_training_ready": False,
        "formal_training_blockers": formal_blockers,
    }
    _write_csv(output_dir / "window_index.csv", windows)
    _write_csv(output_dir / "metrics_by_seed.csv", metric_rows)
    _write_json(output_dir / "metric_summary.json", metric_summary)
    _write_json(output_dir / "dataset_summary.json", summary)
    _write_json(output_dir / "validation_report.json", validation)
    (output_dir / "REPORT.md").write_text(_report(summary, validation), encoding="utf-8")
    top_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "files": {
                str(path.relative_to(output_dir)).replace("\\", "/"): _sha256(path)
                for path in top_files
            },
        },
    )
    return {
        "development_dataset_ready": development_ready,
        "formal_training_ready": False,
        "output_dir": str(output_dir),
        "seed_count": len(ordered_seeds),
        "window_count": len(windows),
    }


def _runtime_components() -> tuple[Callable[[int, float], Mapping[str, Any]], Callable[[dict[str, Any]], Mapping[str, Any]]]:
    import task_resource_conservation_audit as exp04

    return exp04.run_airfogsim_conservation_seed, exp04.validate_exp04_bundle


def _default_output_dir() -> Path:
    return CODE_ROOT / "artifacts" / "datasets" / "airfogsim_multiseed_v2_dev"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build AirFogSim dual-graph v2 development data.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-time", type=float, default=12.0)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()
    runtime_runner, resource_validator = _runtime_components()
    result = build_multiseed_dataset(
        output_dir=args.output_dir,
        seeds=args.seeds,
        max_time=args.max_time,
        history_steps=args.history,
        horizon_steps=args.horizon,
        runtime_runner=runtime_runner,
        resource_validator=resource_validator,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["development_dataset_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
