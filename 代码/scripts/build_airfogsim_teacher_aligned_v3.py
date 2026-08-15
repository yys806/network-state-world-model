from __future__ import annotations

"""Build the teacher-aligned PI-JWM AirFogSim graph/tensor dataset."""

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_teacher_graph_v3 import (
    PHYSICAL_EDGE_RULE,
    SCHEMA_VERSION as GRAPH_SCHEMA_VERSION,
    remap_teacher_aligned_graph,
    validate_teacher_aligned_graph,
)
from pi_jwm.airfogsim_teacher_tensor_v3 import (
    SCHEMA_VERSION as TENSOR_SCHEMA_VERSION,
    TeacherAlignedTensorContract,
    tensorize_teacher_aligned_graph,
)


DATASET_SCHEMA_VERSION = "PIJWM-DG-Contract-v3-dataset"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capacity(rows: list[dict[str, str]], name: str) -> int:
    if not rows or any(str(row.get(name, "")).strip() == "" for row in rows):
        raise ValueError(f"trajectory index is missing capacity field {name}")
    return max(int(row[name]) for row in rows)


def _contract_from_index(
    rows: list[dict[str, str]], history_steps: int, horizon_steps: int
) -> TeacherAlignedTensorContract:
    if not rows:
        raise ValueError("no unlocked trajectories are available")
    max_nodes = _capacity(rows, "physical_nodes")
    return TeacherAlignedTensorContract(
        max_nodes=max_nodes,
        max_physical_edges=max(
            int(row["physical_nodes"]) * (int(row["physical_nodes"]) - 1)
            for row in rows
        ),
        max_information_edges=_capacity(rows, "physical_edges"),
        max_flows=_capacity(rows, "information_edges"),
        max_tasks=_capacity(rows, "task_nodes"),
        max_dag_edges=_capacity(rows, "task_dag_edges"),
        history_steps=history_steps,
        horizon_steps=horizon_steps,
    )


def _compact_mapping(
    graph: Mapping[str, Any], source_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    source_file = source_manifest.get("files", {}).get(
        "dual_graph_v2_bundle.json", {}
    )
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "trajectory_id": graph["trajectory_id"],
        "physical_edge_rule": graph["physical_edge_rule"],
        "source_bundle": {
            "schema_version": source_manifest.get("schema_version"),
            "sha256": source_file.get("sha256"),
            "size_bytes": source_file.get("size_bytes"),
        },
        "physical_nodes": list(graph["physical_nodes"]),
        "physical_edges": list(graph["physical_edges"]),
        "information_nodes": list(graph["information_nodes"]),
        "information_edges": list(graph["information_edges"]),
        "data_flows": list(graph["data_flows"]),
        "cip_relations": list(graph["cip_relations"]),
        "cep_relations": list(graph["cep_relations"]),
        "cfl_relations": list(graph["cfl_relations"]),
        "task_dag_edges": list(graph["task_dag_edges"]),
        "source_audit": dict(graph["source_audit"]),
        "dynamic_state_materialization": "trajectory_tensors.npz",
    }


def _masked_statistics(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    feature_count = values.shape[-1]
    flat = values.astype(np.float64, copy=False).reshape(-1, feature_count)
    flat_mask = mask.astype(bool, copy=False).reshape(-1, feature_count)
    count = flat_mask.sum(axis=0).astype(np.int64)
    total = np.where(flat_mask, flat, 0.0).sum(axis=0)
    total_sq = np.where(flat_mask, np.square(flat), 0.0).sum(axis=0)
    safe_count = np.maximum(count, 1)
    mean = total / safe_count
    variance = np.maximum(total_sq / safe_count - np.square(mean), 1e-12)
    scale = np.sqrt(variance)
    mean[count == 0] = 0.0
    scale[count == 0] = 1.0
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "count": count.tolist(),
    }


def _fit_train_only_stats(output_dir: Path, train_rows: list[dict[str, str]]) -> dict[str, Any]:
    specs = {
        "physical_node_state": "physical_node_feature_mask",
        "physical_edge_state": "physical_edge_feature_mask",
        "information_node_state": "information_node_feature_mask",
        "information_edge_state": "information_edge_feature_mask",
        "data_flow_state": "data_flow_present",
        "task_state": "task_present",
    }
    accumulators: dict[str, dict[str, np.ndarray] | None] = {
        name: None for name in specs
    }
    for row in train_rows:
        seed_dir = output_dir / f"seed_{int(row['seed']):03d}"
        with np.load(seed_dir / "trajectory_tensors.npz", allow_pickle=False) as loaded:
            for name, mask_name in specs.items():
                value = loaded[name]
                mask = loaded[mask_name]
                if mask.ndim == value.ndim - 1:
                    mask = np.broadcast_to(mask[..., None], value.shape)
                feature_count = value.shape[-1]
                current = accumulators[name]
                if current is None:
                    current = {
                        "count": np.zeros(feature_count, dtype=np.int64),
                        "total": np.zeros(feature_count, dtype=np.float64),
                        "total_sq": np.zeros(feature_count, dtype=np.float64),
                    }
                    accumulators[name] = current
                for feature_index in range(feature_count):
                    selected = value[..., feature_index][
                        mask[..., feature_index].astype(bool, copy=False)
                    ].astype(np.float64, copy=False)
                    current["count"][feature_index] += selected.size
                    current["total"][feature_index] += selected.sum(dtype=np.float64)
                    current["total_sq"][feature_index] += np.dot(selected, selected)

    features: dict[str, dict[str, Any]] = {}
    for name, current in accumulators.items():
        if current is None:
            raise ValueError(f"no training tensors available for {name}")
        count = current["count"]
        safe_count = np.maximum(count, 1)
        mean = current["total"] / safe_count
        variance = np.maximum(
            current["total_sq"] / safe_count - np.square(mean), 1e-12
        )
        scale = np.sqrt(variance)
        mean[count == 0] = 0.0
        scale[count == 0] = 1.0
        features[name] = {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "count": count.tolist(),
        }
    return {
        "schema_version": "PIJWM-DG-Contract-v3-normalization",
        "source_split": "train",
        "trajectory_count": len(train_rows),
        "features": features,
    }


def _load_completed_seed_report(
    output_dir: Path, row: Mapping[str, str]
) -> dict[str, Any]:
    seed_dir = output_dir / f"seed_{int(row['seed']):03d}"
    manifest_path = seed_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"resume seed directory is incomplete: {seed_dir}")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != DATASET_SCHEMA_VERSION
        or manifest.get("trajectory_id") != row["trajectory_id"]
        or int(manifest.get("seed", -1)) != int(row["seed"])
        or manifest.get("split") != row["split"]
    ):
        raise ValueError(f"resume seed manifest identity mismatch: {manifest_path}")
    required_files = {
        "teacher_graph_mapping.json",
        "trajectory_tensors.npz",
        "tensor_report.json",
        "graph_validation.json",
    }
    files = manifest.get("files", {})
    if set(files) != required_files:
        raise ValueError(f"resume seed manifest file set mismatch: {manifest_path}")
    for name, expected in files.items():
        path = seed_dir / name
        if (
            not path.is_file()
            or path.stat().st_size != int(expected["size_bytes"])
            or _sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"resume seed artifact integrity failure: {path}")
    validation = _read_json(seed_dir / "graph_validation.json")
    report = _read_json(seed_dir / "tensor_report.json")
    if not validation.get("teacher_aligned_graph_valid"):
        raise ValueError(f"resume seed graph validation failed: {seed_dir}")
    if not report.get("validation", {}).get("teacher_aligned_tensor_valid"):
        raise ValueError(f"resume seed tensor validation failed: {seed_dir}")
    return report


def _protocol(contract: TeacherAlignedTensorContract) -> dict[str, Any]:
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "framework": "PI-JWM",
        "simulator_role": "AirFogSim is a reusable simulator/data source only",
        "physical_edge_rule": PHYSICAL_EDGE_RULE,
        "physical_graph": {
            "nodes": "real hardware entities",
            "edges": "directed same-slot spatial relations derived from positions",
            "forbidden_fields": [
                "CSI",
                "channel_gain",
                "interference",
                "SINR",
                "RB",
                "rate",
                "throughput",
            ],
        },
        "information_graph": {
            "nodes": "one composite communication/computation/service agent per device",
            "edges": "directed AirFogSim communication-interface links",
            "field_roles": contract.to_dict()["information_edge_feature_roles"],
        },
        "cross_graph_relations": {
            "CIP": "information agent to unique physical device",
            "CEP": "information edge to same-endpoint physical spatial edge",
        },
        "business_relation": {
            "CFL": "data flow to endpoint-compatible information communication edge"
        },
        "task_boundary": "tasks and DAG edges are auxiliary business structures, not information graph nodes or edges",
        "missing_value_rule": "numeric zero plus false feature mask; zero alone is never evidence",
        "deprecated": ["CFE", "wireless channel fields on physical edges"],
    }


def build_teacher_aligned_dataset(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    graph_loader: Callable[[Mapping[str, str]], dict[str, Any]] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    source_summary = _read_json(source_dir / "dataset_summary.json")
    if source_summary.get("schema_version") != "PI-JWM-AirFogSim-formal-dataset-v1":
        raise ValueError("source dataset must use the formal AirFogSim v1 schema")
    rows = _read_csv(source_dir / "trajectory_index.csv")
    unlocked = [row for row in rows if row["split"] != "locked_test"]
    locked = [row for row in rows if row["split"] == "locked_test"]
    if not unlocked or not locked:
        raise ValueError("source dataset must contain unlocked and locked-test trajectories")
    history_steps = int(source_summary["history_steps"])
    horizon_steps = int(source_summary["horizon_steps"])
    contract = _contract_from_index(unlocked, history_steps, horizon_steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_has_files = any(output_dir.iterdir())
    if output_has_files and not resume:
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    if graph_loader is None:
        graph_loader = lambda row: _read_json(
            source_dir
            / "trajectories"
            / row["trajectory_id"]
            / "dual_graph_v2_bundle.json"
        )
    protocol_payload = _protocol(contract)
    contract_payload = contract.to_dict()
    contract_payload.update(
        {
            "normalization_source_split": "train",
            "locked_test_capacity_status": "not_inspected_before_model_freeze",
        }
    )
    window_rows = _read_csv(source_dir / "window_index.csv")
    if output_has_files:
        if _read_json(output_dir / "protocol.json") != protocol_payload:
            raise ValueError("resume protocol does not match requested source dataset")
        if _read_json(output_dir / "tensor_contract.json") != contract_payload:
            raise ValueError("resume tensor contract does not match requested source dataset")
        if _read_csv(output_dir / "window_index.csv") != window_rows:
            raise ValueError("resume window index does not match requested source dataset")
    else:
        _write_json(output_dir / "protocol.json", protocol_payload)
        _write_json(output_dir / "tensor_contract.json", contract_payload)
        _write_csv(
            output_dir / "window_index.csv",
            window_rows,
            list(window_rows[0].keys()),
        )

    reports: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    time_counts: dict[int, int] = {}
    for number, row in enumerate(unlocked, start=1):
        seed_dir = output_dir / f"seed_{int(row['seed']):03d}"
        if resume and seed_dir.exists():
            print(
                f"[{number}/{len(unlocked)}] verify existing {row['trajectory_id']} "
                f"(seed={row['seed']}, split={row['split']})",
                flush=True,
            )
            report = _load_completed_seed_report(output_dir, row)
            reports.append(report)
            time_counts[int(row["seed"])] = int(report["time_count"])
            index_rows.append(
                {
                    **row,
                    "v3_status": "materialized",
                    "v3_seed_dir": seed_dir.name,
                    "airfogsim_rerun_required": str(
                        report["source_audit"]["airfogsim_rerun_required"]
                    ).lower(),
                }
            )
            continue
        print(
            f"[{number}/{len(unlocked)}] remap {row['trajectory_id']} (seed={row['seed']}, split={row['split']})",
            flush=True,
        )
        source_manifest = _read_json(
            source_dir / "trajectories" / row["trajectory_id"] / "manifest.json"
        )
        teacher_graph = remap_teacher_aligned_graph(graph_loader(row))
        graph_validation = validate_teacher_aligned_graph(teacher_graph)
        arrays, tensor_report = tensorize_teacher_aligned_graph(teacher_graph, contract)
        seed_dir.mkdir(parents=True, exist_ok=False)
        mapping_path = seed_dir / "teacher_graph_mapping.json"
        tensor_path = seed_dir / "trajectory_tensors.npz"
        report_path = seed_dir / "tensor_report.json"
        validation_path = seed_dir / "graph_validation.json"
        _write_json(mapping_path, _compact_mapping(teacher_graph, source_manifest))
        np.savez_compressed(tensor_path, **arrays)
        report = {
            **tensor_report,
            "trajectory_id": row["trajectory_id"],
            "seed": int(row["seed"]),
            "split": row["split"],
            "source_audit": teacher_graph["source_audit"],
            "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
            "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
        }
        _write_json(report_path, report)
        _write_json(validation_path, graph_validation)
        _write_json(
            seed_dir / "manifest.json",
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "trajectory_id": row["trajectory_id"],
                "seed": int(row["seed"]),
                "split": row["split"],
                "files": {
                    path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
                    for path in (mapping_path, tensor_path, report_path, validation_path)
                },
            },
        )
        reports.append(report)
        time_counts[int(row["seed"])] = int(report["time_count"])
        index_rows.append(
            {
                **row,
                "v3_status": "materialized",
                "v3_seed_dir": seed_dir.name,
                "airfogsim_rerun_required": str(
                    teacher_graph["source_audit"]["airfogsim_rerun_required"]
                ).lower(),
            }
        )

    locked_records = []
    for row in locked:
        source_manifest = _read_json(
            source_dir
            / "locked_test"
            / "trajectories"
            / row["trajectory_id"]
            / "manifest.json"
        )
        bundle = source_manifest.get("files", {}).get("dual_graph_v2_bundle.json", {})
        locked_records.append(
            {
                "trajectory_id": row["trajectory_id"],
                "seed": int(row["seed"]),
                "split": row["split"],
                "source_bundle_sha256": bundle.get("sha256"),
                "source_bundle_size_bytes": bundle.get("size_bytes"),
                "label_content_read": False,
                "tensorized": False,
            }
        )
        index_rows.append(
            {
                **row,
                "v3_status": "locked_integrity_only",
                "v3_seed_dir": "",
                "airfogsim_rerun_required": "false",
            }
        )
    locked_payload = {
        "schema_version": "PIJWM-DG-Contract-v3-locked-integrity",
        "trajectory_count": len(locked_records),
        "label_content_read": False,
        "tensorized": False,
        "trajectories": locked_records,
    }
    _write_json(output_dir / "locked_test_integrity.json", locked_payload)
    _write_csv(
        output_dir / "trajectory_index.csv",
        index_rows,
        list(index_rows[0].keys()),
    )
    stats = _fit_train_only_stats(
        output_dir, [row for row in unlocked if row["split"] == "train"]
    )
    _write_json(output_dir / "normalization_stats.json", stats)

    unlocked_seed_set = {int(row["seed"]) for row in unlocked}
    locked_seed_set = {int(row["seed"]) for row in locked}
    window_bounds_valid = all(
        int(row["seed"]) in unlocked_seed_set
        and 0 <= int(row["input_start_index"]) < int(row["input_end_index"])
        and int(row["input_end_index"]) == int(row["label_start_index"])
        and int(row["label_start_index"])
        < int(row["label_end_index"])
        <= time_counts[int(row["seed"])]
        for row in window_rows
    )
    checks = {
        "source_schema_valid": True,
        "all_unlocked_graphs_valid": all(
            report["validation"]["teacher_aligned_tensor_valid"]
            for report in reports
        ),
        "all_v3_required_fields_available": all(
            not report["source_audit"]["required_missing"] for report in reports
        ),
        "physical_and_information_features_separate": not (
            {"csi_mean", "allocated_rb_count", "active_task_count", "rate_sum"}
            & set(contract_payload["physical_edge_features"])
        ),
        "cip_cep_cfl_materialized": all(
            report["counts"]["information_nodes"] > 0
            and report["counts"]["information_edges"] > 0
            and report["counts"]["cfl_relations"]
            == report["counts"]["data_flows"]
            for report in reports
        ),
        "window_bounds_valid": window_bounds_valid,
        "training_stats_are_train_only": stats["source_split"] == "train",
        "split_isolation_valid": all(
            not (output_dir / f"seed_{seed:03d}").exists() for seed in locked_seed_set
        ),
        "locked_test_labels_not_read": not locked_payload["label_content_read"],
        "locked_test_not_tensorized": not locked_payload["tensorized"],
    }
    ready = all(checks.values())
    validation = {
        "schema_version": "PIJWM-DG-Contract-v3-dataset-validation",
        "teacher_aligned_graph_tensor_ready": ready,
        "airfogsim_rerun_required": not checks["all_v3_required_fields_available"],
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }
    _write_json(output_dir / "validation_report.json", validation)
    summary = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_dataset": str(source_dir.resolve()),
        "raw_data_reused": True,
        "airfogsim_rerun_required": validation["airfogsim_rerun_required"],
        "teacher_aligned_graph_tensor_ready": ready,
        "formal_training_ready": False,
        "formal_training_blockers": [
            "R2 evaluation protocol not frozen",
            "R3 CPU model smoke not completed",
            "locked test remains sealed",
        ],
        "unlocked_trajectory_count": len(unlocked),
        "locked_test_trajectory_count": len(locked),
        "window_count": len(window_rows),
        "history_steps": history_steps,
        "horizon_steps": horizon_steps,
        "tensor_contract": contract_payload,
    }
    _write_json(output_dir / "dataset_summary.json", summary)
    manifest_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path != output_dir / "manifest.json"
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "PIJWM-DG-Contract-v3-manifest",
            "teacher_aligned_graph_tensor_ready": ready,
            "files": {
                path.relative_to(output_dir).as_posix(): {
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in manifest_files
            },
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build PIJWM-DG-Contract-v3 from saved AirFogSim trajectories."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=CODE_ROOT / "artifacts" / "datasets" / "airfogsim_formal_v1",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="verify and reuse completed seed outputs in an interrupted build",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_ROOT
        / "artifacts"
        / "datasets"
        / "airfogsim_teacher_aligned_v3",
    )
    args = parser.parse_args()
    summary = build_teacher_aligned_dataset(
        source_dir=args.source_dir, output_dir=args.output_dir, resume=args.resume
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["teacher_aligned_graph_tensor_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
