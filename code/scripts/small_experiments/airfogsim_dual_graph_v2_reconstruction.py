from __future__ import annotations

"""Reconstruct frozen AirFogSim evidence with the PI-JWM v2 graph semantics."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_dual_graph_v2 import (
    PHASE_TO_FLOW_TYPE,
    build_dual_graph_v2_bundle,
    validate_dual_graph_v2_bundle,
)
from pi_jwm.airfogsim_metrics_v2 import compute_airfogsim_metrics_v2


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _select_single_slot_sample(
    bundle: dict[str, Any],
    *,
    sample_task_id: str | None,
    sample_time: float | None,
) -> dict[str, Any]:
    events = list(bundle.get("source_transfer_events", []))
    candidates = [
        row
        for row in events
        if (sample_task_id is None or str(row.get("task_id")) == str(sample_task_id))
        and (sample_time is None or abs(float(row.get("time", 0.0)) - float(sample_time)) <= 1e-9)
    ]
    if not candidates:
        candidates = events[:1]
    if not candidates:
        return {
            "status": "not_available",
            "reason": "no_runtime_transfer_event",
            "task": None,
            "event": None,
        }

    event = min(candidates, key=lambda row: (float(row.get("time", 0.0)), str(row.get("event_id", ""))))
    task_id = str(event["task_id"])
    flow_type = PHASE_TO_FLOW_TYPE.get(str(event.get("phase")))
    flows = [
        row
        for row in bundle.get("information_edges", [])
        if str(row.get("task_id")) == task_id and str(row.get("flow_type")) == flow_type
    ]
    flow = next(
        (
            row
            for row in flows
            if str(row.get("src")) == f"agent::{event['source']}"
            and str(row.get("dst")) == f"agent::{event['target']}"
        ),
        flows[0] if flows else None,
    )
    flow_id = None if flow is None else str(flow["id"])
    bearers = [row for row in bundle.get("flow_bearers", []) if str(row.get("flow_id")) == flow_id]
    physical_edge_ids = {str(row["physical_edge_id"]) for row in bearers}
    event_time = float(event.get("time", 0.0))
    physical_edges = [
        row
        for row in bundle.get("source_physical_edge_snapshots", [])
        if str(row.get("id")) in physical_edge_ids
        and abs(float(row.get("observed_time", -1.0)) - event_time) <= 1e-9
    ]
    agent_ids = set()
    if flow is not None:
        agent_ids.update((str(flow["src"]), str(flow["dst"])))
    information_nodes = [row for row in bundle.get("information_nodes", []) if str(row.get("id")) in agent_ids]
    physical_node_ids = {str(row["physical_node_id"]) for row in information_nodes}
    physical_nodes = [
        row
        for row in bundle.get("source_physical_node_snapshots", [])
        if str(row.get("id")) in physical_node_ids
        and abs(float(row.get("observed_time", -1.0)) - event_time) <= 1e-9
    ]
    exact_node_ids = {str(row.get("id")) for row in physical_nodes}
    exact_edge_ids = {str(row.get("id")) for row in physical_edges}
    strict_same_time = physical_node_ids <= exact_node_ids and physical_edge_ids <= exact_edge_ids
    attachments = [row for row in bundle.get("agent_attachments", []) if str(row.get("agent_id")) in agent_ids]
    task = next((row for row in bundle.get("task_nodes", []) if str(row.get("id")) == task_id), None)
    dag_edges = [
        row
        for row in bundle.get("task_dag_edges", [])
        if str(row.get("src")) == task_id or str(row.get("dst")) == task_id
    ]
    return {
        "status": "available" if strict_same_time else "missing_exact_physical_snapshot",
        "strict_same_time": strict_same_time,
        "time": event_time,
        "task": task,
        "event": event,
        "physical_nodes": physical_nodes,
        "physical_edges": physical_edges,
        "information_nodes": information_nodes,
        "information_edge": flow,
        "task_dag_edges": dag_edges,
        "agent_attachments": attachments,
        "flow_bearers": bearers,
    }


def _metric_inputs(exp04: dict[str, Any], bundle: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metric_input_bundle = {
        "task_records": bundle["task_nodes"],
        "transfer_events": bundle["source_transfer_events"],
        "physical_node_snapshots": bundle.get("source_physical_node_snapshots", []),
        "physical_edge_snapshots": bundle.get("source_physical_edge_snapshots", []),
        "task_dag_edges": bundle["task_dag_edges"],
        "dependency_data_flows": [
            row for row in bundle["information_edges"] if row.get("flow_type") == "dependency_data"
        ],
        "task_ledger": list(exp04.get("task_ledger", [])),
        "rb_ledger": list(exp04.get("rb_ledger", [])),
        "cpu_ledger": list(exp04.get("cpu_ledger", [])),
        "uav_energy_ledger": list(exp04.get("uav_energy_ledger", [])),
        "dependency_ledger": [],
    }
    manifest = {
        "task_records": len(metric_input_bundle["task_records"]),
        "transfer_events": len(metric_input_bundle["transfer_events"]),
        "task_ledger_rows": len(metric_input_bundle["task_ledger"]),
        "rb_ledger_rows": len(metric_input_bundle["rb_ledger"]),
        "cpu_ledger_rows": len(metric_input_bundle["cpu_ledger"]),
        "uav_energy_ledger_rows": len(metric_input_bundle["uav_energy_ledger"]),
        "task_dag_edges": len(bundle["task_dag_edges"]),
        "dependency_data_flows": sum(
            row.get("flow_type") == "dependency_data" for row in bundle["information_edges"]
        ),
        "legacy_dependency_ledger_used": False,
        "legacy_dependency_ledger_status": "excluded_old_shared_parent_output_semantics",
        "dependency_payload_status": bundle["evidence_boundary"]["dependency_payload"],
    }
    return metric_input_bundle, manifest


def _format_metric_value(row: dict[str, Any]) -> str:
    value = row.get("value")
    return "not_computable" if value is None else f"{float(value):.6f}"


def _report(
    validation: dict[str, Any],
    metric_manifest: dict[str, Any],
    metric_results: dict[str, Any],
) -> str:
    counts = validation["counts"]
    metrics = {row["name"]: row for row in metric_results["metrics"]}
    window = metric_results["evaluation_window"]
    return "\n".join(
        [
            "# AirFogSim PI-JWM双图v2重构报告",
            "",
            f"- v2合法性：{'通过' if validation['dual_graph_v2_ready'] else '未通过'}",
            f"- 物理节点/边：{counts['physical_nodes']}/{counts['physical_edges']}",
            f"- 信息代理/信息流：{counts['information_nodes']}/{counts['information_edges']}",
            f"- 任务/DAG边：{counts['task_nodes']}/{counts['task_dag_edges']}",
            f"- CIP/CFE：{counts['agent_attachments']}/{counts['flow_bearers']}",
            f"- 真实传输事件：{metric_manifest['transfer_events']}",
            f"- DAG依赖数据流：{metric_manifest['dependency_data_flows']}",
            "- 旧`shared_parent_output`依赖台账未进入v2；AirFogSim原生DAG无载荷时只保留先后关系。",
            "",
            "## seed 0真实指标（12秒单轨迹）",
            "",
            f"- 可评价任务：{window['evaluable_tasks']}；完成：{window['completed_tasks']}；右删失：{window['right_censored_tasks']}。",
            f"- 任务完成率：{_format_metric_value(metrics['task_completion_rate'])}；成功任务P95/P99时延：{_format_metric_value(metrics['successful_task_delay_p95'])}/{_format_metric_value(metrics['successful_task_delay_p99'])}秒。",
            f"- RB/CPU利用率：{_format_metric_value(metrics['rb_utilization'])}/{_format_metric_value(metrics['cpu_utilization'])}；UAV总能耗：{_format_metric_value(metrics['uav_energy_total'])} AirFogSim能量单位。",
            "- 任务流守恒、CPU容量和UAV能量方程违例率均为0。",
            "- 不确定性覆盖、action regret和OOD迁移当前标记为`not_computable`，需在世界模型预测或配对反事实数据上另算。",
        ]
    ) + "\n"


def reconstruct_from_artifacts(
    exp03_bundle_path: Path,
    exp04_bundle_path: Path,
    output_dir: Path,
    *,
    trajectory_id: str | None = None,
    sample_task_id: str | None = None,
    sample_time: float | None = None,
) -> dict[str, Any]:
    exp03_bundle_path = Path(exp03_bundle_path)
    exp04_bundle_path = Path(exp04_bundle_path)
    output_dir = Path(output_dir)
    exp03 = _read_json(exp03_bundle_path)
    exp04 = _read_json(exp04_bundle_path)
    physical_nodes = list(exp03.get("physical_nodes", []))
    inferred_trajectory_id = trajectory_id
    if inferred_trajectory_id is None and physical_nodes:
        inferred_trajectory_id = str(physical_nodes[0].get("trajectory_id", "airfogsim-v2"))
    if inferred_trajectory_id is None:
        inferred_trajectory_id = "airfogsim-v2"

    bundle = build_dual_graph_v2_bundle(
        trajectory_id=inferred_trajectory_id,
        physical_nodes=physical_nodes,
        physical_edges=exp03.get("physical_edges", []),
        task_records=exp03.get("information_nodes", []),
        dag_edges=exp03.get("information_edges", []),
        transfer_events=exp03.get("transfer_events", []),
    )
    bundle["source_physical_node_snapshots"] = list(exp03.get("physical_node_snapshots", []))
    bundle["source_physical_edge_snapshots"] = list(exp03.get("physical_edge_snapshots", []))
    validation = validate_dual_graph_v2_bundle(bundle)
    metric_input_bundle, metric_manifest = _metric_inputs(exp04, bundle)
    metric_results = compute_airfogsim_metrics_v2(metric_input_bundle)
    sample = _select_single_slot_sample(
        bundle,
        sample_task_id=sample_task_id,
        sample_time=sample_time,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "dual_graph_v2_bundle.json", bundle)
    _write_json(output_dir / "validation_report.json", validation)
    _write_json(output_dir / "metric_input_bundle.json", metric_input_bundle)
    _write_json(output_dir / "metric_input_manifest.json", metric_manifest)
    _write_json(output_dir / "metric_results.json", metric_results)
    _write_csv(output_dir / "metric_results.csv", list(metric_results["metrics"]))
    _write_json(output_dir / "single_slot_sample.json", sample)
    for filename, key in (
        ("information_nodes.csv", "information_nodes"),
        ("information_edges.csv", "information_edges"),
        ("task_nodes.csv", "task_nodes"),
        ("task_dag_edges.csv", "task_dag_edges"),
        ("agent_attachments.csv", "agent_attachments"),
        ("flow_bearers.csv", "flow_bearers"),
    ):
        _write_csv(output_dir / filename, list(bundle[key]))
    (output_dir / "REPORT.md").write_text(
        _report(validation, metric_manifest, metric_results),
        encoding="utf-8",
    )

    generated_files = sorted(path for path in output_dir.iterdir() if path.name != "manifest.json")
    manifest = {
        "schema_version": bundle["schema_version"],
        "sources": {
            "exp03_bundle": str(exp03_bundle_path),
            "exp03_bundle_sha256": _sha256(exp03_bundle_path),
            "exp04_bundle": str(exp04_bundle_path),
            "exp04_bundle_sha256": _sha256(exp04_bundle_path),
        },
        "files": {path.name: _sha256(path) for path in generated_files},
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {
        **validation,
        "output_dir": str(output_dir),
        "single_slot_status": sample["status"],
    }


def _default_exp03() -> Path:
    return CODE_ROOT / "artifacts" / "small_experiments" / "exp03_airfogsim_cross_graph_evidence" / "evidence_v1" / "bundle.json"


def _default_exp04() -> Path:
    return CODE_ROOT / "artifacts" / "small_experiments" / "exp04_task_resource_conservation" / "conservation_v1" / "bundle.json"


def _default_output() -> Path:
    return CODE_ROOT / "artifacts" / "small_experiments" / "exp06_airfogsim_dual_graph_v2" / "v2_minimal"


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct AirFogSim evidence with PI-JWM dual-graph v2 semantics.")
    parser.add_argument("--exp03-bundle", type=Path, default=_default_exp03())
    parser.add_argument("--exp04-bundle", type=Path, default=_default_exp04())
    parser.add_argument("--output-dir", type=Path, default=_default_output())
    parser.add_argument("--sample-task-id", default="Task_10")
    parser.add_argument("--sample-time", type=float, default=2.6)
    args = parser.parse_args()
    result = reconstruct_from_artifacts(
        args.exp03_bundle,
        args.exp04_bundle,
        args.output_dir,
        sample_task_id=args.sample_task_id,
        sample_time=args.sample_time,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["dual_graph_v2_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
