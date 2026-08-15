"""Build the frozen R2 PI-JWM evaluation-protocol artifact bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .evaluation_protocol_v3 import (
    SCHEMA_VERSION as REGISTRY_SCHEMA_VERSION,
    build_fair_experiment_protocol,
    build_factual_metric_mapping,
    build_metric_registry,
    validate_evaluation_protocol,
)
from .teacher_evaluation_v3 import (
    METHODS,
    SELECTION_COMPONENTS,
    SCHEMA_VERSION as BASELINE_SCHEMA_VERSION,
    evaluate_teacher_trajectory,
    summarize_teacher_reports,
)
from .airfogsim_teacher_tensor_v3 import (
    INFORMATION_EDGE_FEATURES,
    INFORMATION_NODE_FEATURES,
    PHYSICAL_EDGE_FEATURES,
    PHYSICAL_NODE_FEATURES,
)
from .airfogsim_tensor_v2 import FLOW_FEATURES, TASK_FEATURES
from .formal_airfogsim_graph_v1 import FORMAL_DAG_STATE_FEATURES


SCHEMA_VERSION = "PIJWM-Evaluation-Bundle-v3"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_manifest_file(
    path: Path, manifest_path: Path, *, entry_name: str | None = None
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source manifest does not exist: {manifest_path}")
    manifest = _read_json(manifest_path)
    entry_key = entry_name or path.name
    entry = manifest.get("files", {}).get(entry_key)
    if not isinstance(entry, Mapping):
        raise ValueError(f"source manifest does not lock {entry_key}: {manifest_path}")
    actual = _sha256(path)
    if actual != entry.get("sha256"):
        raise ValueError(f"source hash mismatch for {path.name}")
    if int(entry.get("size_bytes", -1)) != path.stat().st_size:
        raise ValueError(f"source size mismatch for {path.name}")
    return {"sha256": actual, "size_bytes": path.stat().st_size, "verified": True}


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _train_dag_statistics(
    dataset_root: Path, materialized: Iterable[Mapping[str, str]]
) -> dict[str, Any]:
    feature_count = len(FORMAL_DAG_STATE_FEATURES)
    count = np.zeros(feature_count, dtype=np.int64)
    total = np.zeros(feature_count, dtype=np.float64)
    total_square = np.zeros(feature_count, dtype=np.float64)
    for row in materialized:
        if row.get("split") != "train":
            continue
        tensor_path = dataset_root / row["v3_seed_dir"] / "trajectory_tensors.npz"
        with np.load(tensor_path, allow_pickle=False) as archive:
            values = np.asarray(archive["task_dag_state"], dtype=np.float64)
            valid = np.asarray(archive["task_dag_state_present"], dtype=bool)
            if values.shape[-1] != feature_count:
                raise ValueError(
                    "task_dag_state feature dimension disagrees with the frozen R1 contract"
                )
            selected = values[valid]
        count += selected.shape[0]
        total += selected.sum(axis=0)
        total_square += np.square(selected).sum(axis=0)
    if np.any(count == 0):
        raise ValueError("train split contains no observed DAG state for normalization")
    mean = total / count
    variance = np.maximum(total_square / count - mean * mean, 0.0)
    return {
        "feature_names": list(FORMAL_DAG_STATE_FEATURES),
        "count": count.astype(int).tolist(),
        "mean": mean.tolist(),
        "scale": np.maximum(np.sqrt(variance), 1e-6).tolist(),
    }


def _selection_scales(normalization_stats: Mapping[str, Any]) -> dict[str, float]:
    features = normalization_stats["features"]

    def scale(group: str, name: str, names: tuple[str, ...]) -> float:
        return float(features[group]["scale"][names.index(name)])

    pn = tuple(PHYSICAL_NODE_FEATURES)
    pe = tuple(PHYSICAL_EDGE_FEATURES)
    info_node = tuple(INFORMATION_NODE_FEATURES)
    info_edge = tuple(INFORMATION_EDGE_FEATURES)
    flow = tuple(FLOW_FEATURES)
    task = tuple(TASK_FEATURES)
    position_scale = math.sqrt(
        sum(scale("physical_node_state", name, pn) ** 2 for name in ("x", "y", "z"))
    )
    queue_scale = float(
        np.mean(
            [
                scale("information_node_state", name, info_node)
                for name in (
                    "unassigned_queue_count",
                    "tx_queue_count",
                    "return_queue_count",
                )
            ]
        )
    )
    values = {
        "state.physical_node.position.rmse": position_scale,
        "state.physical_node.motion.rmse": scale("physical_node_state", "speed", pn),
        "state.physical_edge.distance.rmse": scale("physical_edge_state", "distance", pe),
        "state.physical_edge.relative_speed.rmse": scale("physical_edge_state", "relative_speed", pe),
        "state.information_node.queue.mae": queue_scale,
        "state.information_node.cpu_backlog.mae": scale("information_node_state", "cpu_backlog", info_node),
        "state.information_edge.rate.rmse": scale("information_edge_state", "outcome.rate_sum", info_edge),
        "state.flow.remaining_data.mae": scale("data_flow_state", "remaining_data", flow),
        "state.task.deadline_remaining.mae": scale("task_state", "deadline_remaining", task),
        "state.dag.unfinished_parent_count.mae": float(
            features["task_dag_state"]["scale"][
                FORMAL_DAG_STATE_FEATURES.index("unfinished_parent_count")
            ]
        ),
    }
    if set(values) != set(SELECTION_COMPONENTS) or any(
        not math.isfinite(value) or value <= 0.0 for value in values.values()
    ):
        raise ValueError("train-only selection scales are incomplete or invalid")
    return values


def _validate_dataset_contract(
    tensor_contract: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    """Bind numeric tensor columns to the frozen R1 dual-graph semantics."""

    if tensor_contract.get("schema_version") != "PIJWM-DG-Contract-v3-tensor":
        raise ValueError("tensor contract schema_version is not the frozen R1 version")
    expected_features = {
        "physical_node_features": list(PHYSICAL_NODE_FEATURES),
        "physical_edge_features": list(PHYSICAL_EDGE_FEATURES),
        "information_node_features": list(INFORMATION_NODE_FEATURES),
        "information_edge_features": list(INFORMATION_EDGE_FEATURES),
    }
    for field, expected in expected_features.items():
        actual = tensor_contract.get(field)
        if actual != expected:
            raise ValueError(
                f"tensor contract feature order mismatch for {field}: "
                f"expected {expected}, got {actual}"
            )

    if protocol.get("schema_version") != "PIJWM-DG-Contract-v3":
        raise ValueError("dataset protocol schema_version is not the frozen R1 version")
    if protocol.get("framework") != "PI-JWM":
        raise ValueError("dataset protocol must identify PI-JWM as the framework")
    if protocol.get("simulator_role") != "AirFogSim is a reusable simulator/data source only":
        raise ValueError("dataset protocol assigns an invalid role to AirFogSim")
    if protocol.get("physical_edge_rule") != "complete_directed_spatial_relation":
        raise ValueError("dataset protocol physical-edge rule changed")
    if protocol.get("missing_value_rule") != (
        "numeric zero plus false feature mask; zero alone is never evidence"
    ):
        raise ValueError("dataset protocol missing-value rule changed")
    deprecated = set(protocol.get("deprecated", []))
    if "wireless channel fields on physical edges" not in deprecated:
        raise ValueError("dataset protocol no longer excludes channel fields from physical edges")


def _is_finite_or_na(report: Mapping[str, Any]) -> bool:
    for metric in report.get("metrics", {}).values():
        status = metric.get("status")
        if status == "computed":
            value = metric.get("value")
            if value is None or not math.isfinite(float(value)):
                return False
        elif status != "not_computable":
            return False
    return True


def _factual_availability(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    names = sorted({str(row.get("metric_id", "")) for row in rows if row.get("metric_id")})
    trajectories = sorted(
        {(str(row.get("environment_seed", "")), str(row.get("split", ""))) for row in rows}
    )
    metrics = []
    for name in names:
        selected = [row for row in rows if row.get("metric_id") == name]
        status_counts: dict[str, int] = {}
        for row in selected:
            status = str(row.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
        units = sorted({str(row.get("unit", "")) for row in selected})
        sources = sorted({str(row.get("source", "")) for row in selected})
        metrics.append(
            {
                "metric_id": name,
                "source_metric_names": sorted(
                    {str(row.get("source_metric_name", "")) for row in selected}
                ),
                "row_count": len(selected),
                "status_counts": status_counts,
                "units": units,
                "sources": sources,
                "availability": (
                    "computed"
                    if status_counts.get("computed", 0) == len(selected)
                    else "mixed_or_not_computable"
                ),
            }
        )
    return {
        "schema_version": "PIJWM-Factual-System-Metric-Availability-v3",
        "source_role": "AirFogSim factual trajectory sidecar; not a learned prediction",
        "locked_test_rows_included": False,
        "metric_count": len(names),
        "trajectory_count": len(trajectories),
        "row_count": len(rows),
        "metrics": metrics,
    }


def _information_activity(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    index = INFORMATION_EDGE_FEATURES.index("outcome.active_task_count")
    state = np.asarray(arrays["information_edge_state"][:, :, index])
    valid = (
        np.asarray(arrays["information_edge_present"], dtype=bool)
        & np.asarray(arrays["information_edge_feature_mask"][:, :, index], dtype=bool)
    )
    denominator = int(valid.sum())
    if denominator == 0:
        return {
            "status": "not_computable",
            "value": None,
            "count": 0,
            "numerator": None,
            "denominator": 0.0,
            "reason": "no observed information-edge activity states",
        }
    numerator = int(np.logical_and(valid, state > 0).sum())
    return {
        "status": "computed",
        "value": numerator / denominator,
        "count": denominator,
        "numerator": float(numerator),
        "denominator": float(denominator),
        "reason": None,
    }


def _canonical_factual_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    materialized: Iterable[Mapping[str, str]],
    information_activity_by_seed: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    mapping_rows = build_factual_metric_mapping()
    mapping = {row["source_metric_name"]: row for row in mapping_rows}
    registry = {row["metric_id"]: row for row in build_metric_registry()}
    index = {int(row["seed"]): dict(row) for row in materialized}
    source_rows = [dict(row) for row in rows]
    if any(row.get("split") == "locked_test" for row in source_rows):
        raise ValueError("factual sidecar must not contain locked_test metric rows")
    seen = {(int(row["seed"]), str(row["name"])) for row in source_rows}
    expected = {(seed, name) for seed in index for name in mapping}
    if seen != expected or len(source_rows) != len(expected):
        missing = sorted(expected - seen)[:5]
        extra = sorted(seen - expected)[:5]
        raise ValueError(
            f"factual sidecar is not a complete one-row-per-trajectory metric matrix; "
            f"missing={missing}, extra={extra}, rows={len(source_rows)}, expected={len(expected)}"
        )
    canonical = []
    for row in source_rows:
        seed = int(row["seed"])
        source_name = str(row["name"])
        trajectory = index[seed]
        if row.get("split") != trajectory.get("split"):
            raise ValueError(f"factual sidecar split mismatch for seed {seed}")
        status = "computed" if row.get("status") == "available" else str(row.get("status"))
        if status not in {"computed", "not_computable", "not_applicable"}:
            raise ValueError(f"unsupported factual metric status {status!r}")
        values = {
            "status": status,
            "value": _float_or_none(row.get("value")),
            "count": int(float(row.get("sample_count") or 0)),
            "numerator": _float_or_none(row.get("numerator")),
            "denominator": _float_or_none(row.get("denominator")),
            "reason": row.get("reason") or None,
        }
        if status == "computed" and values["value"] is None:
            raise ValueError(f"computed factual metric has no finite value: seed={seed}, name={source_name}")
        source_fields = [str(row.get("source", ""))]
        if source_name == "physical_link_active_ratio":
            values = dict(information_activity_by_seed[seed])
            source_fields = [
                "information_edge_state.outcome.active_task_count",
                "information_edge_present",
                "information_edge_feature_mask",
            ]
        canonical.append(
            {
                "trajectory_id": trajectory["trajectory_id"],
                "environment_seed": seed,
                "training_seed": None,
                "split": trajectory["split"],
                "method_id": "airfogsim_factual",
                "source_role": "factual_system_outcome",
                "metric_id": mapping[source_name]["canonical_metric_id"],
                "source_metric_name": source_name,
                "mapping_kind": mapping[source_name]["mapping_kind"],
                "status": values["status"],
                "value": values["value"],
                "unit": registry[mapping[source_name]["canonical_metric_id"]]["unit"],
                "source_unit": row.get("unit") or None,
                "count": values["count"],
                "numerator": values["numerator"],
                "denominator": values["denominator"],
                "reason": values["reason"],
                "source_fields": source_fields,
                "checkpoint_id": None,
                "training_budget": None,
                "runtime_device": None,
            }
        )
    return sorted(canonical, key=lambda row: (row["environment_seed"], row["metric_id"]))


def _flatten_baselines(reports: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for report in reports:
        for metric_id, metric in report["metrics"].items():
            flat.append(
                {
                    "trajectory_id": report["trajectory_id"],
                    "environment_seed": report["seed"],
                    "training_seed": None,
                    "split": report["split"],
                    "method_id": report["method"],
                    "source_role": "prediction_baseline",
                    "metric_id": metric_id,
                    "source_metric_name": None,
                    "mapping_kind": "direct",
                    "status": metric["status"],
                    "value": metric["value"],
                    "unit": metric["unit"],
                    "source_unit": None,
                    "count": metric["count"],
                    "numerator": metric["numerator"],
                    "denominator": metric["denominator"],
                    "reason": metric["reason"],
                    "source_fields": metric["source_fields"],
                    "checkpoint_id": None,
                    "training_budget": None,
                    "runtime_device": None,
                }
            )
    return flat


def _write_evaluation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "trajectory_id", "environment_seed", "training_seed", "split",
        "method_id", "source_role", "metric_id", "source_metric_name",
        "mapping_kind", "status", "value", "unit", "count", "numerator",
        "denominator", "reason", "source_fields", "source_unit",
        "checkpoint_id", "training_budget", "runtime_device",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["source_fields"] = json.dumps(
                serializable.get("source_fields", []), ensure_ascii=False
            )
            writer.writerow(serializable)


def _report_template() -> dict[str, Any]:
    return {
        "schema_version": "PIJWM-Formal-Report-Template-v3",
        "required_sections": [
            "method_identity_and_trainable_modules",
            "data_split_and_training_budget",
            "state_and_sparse_event_prediction",
            "factual_system_outcomes",
            "uncertainty_and_safety",
            "runtime_and_parameter_count",
            "paired_seed_statistics_and_failures",
        ],
        "required_result_columns": [
            "trajectory_id", "environment_seed", "training_seed", "split",
            "method_id", "source_role", "metric_id", "source_metric_name",
            "mapping_kind", "status", "value", "unit", "source_unit",
            "count", "numerator", "denominator", "reason", "source_fields",
            "checkpoint_id", "training_budget", "runtime_device",
        ],
        "allowed_status": ["computed", "not_computable", "not_applicable"],
        "not_computable_policy": "retain the row with status and reason; never replace with zero",
        "primary_aggregation": "macro mean over complete environment trajectories",
        "paired_comparison": "same environment trajectories paired across methods; training seeds reported separately",
        "locked_test_policy": "unused until R9 after method and protocol freeze",
    }


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            ).stdout.strip()
        )
        return {"head": head, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None}


def _input_provenance(
    *,
    dataset_root: Path,
    factual_metrics_csv: Path,
    verified_inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    code_files = {
        "evaluation_bundle_v3.py": Path(__file__).resolve(),
        "evaluation_protocol_v3.py": Path(__file__).with_name("evaluation_protocol_v3.py").resolve(),
        "teacher_evaluation_v3.py": Path(__file__).with_name("teacher_evaluation_v3.py").resolve(),
        "airfogsim_teacher_tensor_v3.py": Path(__file__).with_name(
            "airfogsim_teacher_tensor_v3.py"
        ).resolve(),
        "airfogsim_tensor_v2.py": Path(__file__).with_name("airfogsim_tensor_v2.py").resolve(),
        "airfogsim_sparse_diagnostics_v2.py": Path(__file__).with_name(
            "airfogsim_sparse_diagnostics_v2.py"
        ).resolve(),
        "formal_airfogsim_graph_v1.py": Path(__file__).with_name(
            "formal_airfogsim_graph_v1.py"
        ).resolve(),
    }
    repo_root = Path(__file__).resolve().parents[3]
    return {
        "schema_version": "PIJWM-Evaluation-Input-Provenance-v3",
        "dataset_id": dataset_root.name,
        "factual_dataset_id": factual_metrics_csv.parent.name,
        "verified_inputs": {name: dict(value) for name, value in verified_inputs.items()},
        "code_files": {
            name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for name, path in code_files.items()
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "git": _git_provenance(repo_root),
    }


def _manifest(output_dir: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_protocol_ready": True,
        "files": files,
    }


def build_evaluation_bundle(
    dataset_root: str | Path,
    factual_metrics_csv: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build R2 outputs while reading only materialized non-locked v3 tensors."""

    dataset_root = Path(dataset_root).resolve()
    factual_metrics_csv = Path(factual_metrics_csv).resolve()
    output_dir = Path(output_dir).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    if not factual_metrics_csv.is_file():
        raise FileNotFoundError(f"factual metric CSV does not exist: {factual_metrics_csv}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is nonempty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest = dataset_root / "manifest.json"
    factual_manifest = factual_metrics_csv.parent / "manifest.json"
    input_paths = {
        "dataset.trajectory_index.csv": dataset_root / "trajectory_index.csv",
        "dataset.dataset_summary.json": dataset_root / "dataset_summary.json",
        "dataset.normalization_stats.json": dataset_root / "normalization_stats.json",
        "dataset.locked_test_integrity.json": dataset_root / "locked_test_integrity.json",
        "dataset.tensor_contract.json": dataset_root / "tensor_contract.json",
        "dataset.protocol.json": dataset_root / "protocol.json",
        "factual.metrics_by_trajectory.csv": factual_metrics_csv,
        "factual.validation_report.json": factual_metrics_csv.parent / "validation_report.json",
    }
    verified_inputs: dict[str, dict[str, Any]] = {}
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required evaluation input does not exist: {path}")
        verified_inputs[name] = _verify_manifest_file(
            path,
            factual_manifest if name.startswith("factual.") else dataset_manifest,
        )

    tensor_contract = _read_json(input_paths["dataset.tensor_contract.json"])
    dataset_protocol = _read_json(input_paths["dataset.protocol.json"])
    _validate_dataset_contract(tensor_contract, dataset_protocol)

    index_rows = _read_csv(input_paths["dataset.trajectory_index.csv"])
    if len({row.get("trajectory_id") for row in index_rows}) != len(index_rows):
        raise ValueError("trajectory_index contains duplicate trajectory_id values")
    if len({row.get("seed") for row in index_rows}) != len(index_rows):
        raise ValueError("trajectory_index contains duplicate environment seeds")
    materialized = [
        row
        for row in index_rows
        if row.get("v3_status") == "materialized" and row.get("split") != "locked_test"
    ]
    locked = [row for row in index_rows if row.get("split") == "locked_test"]
    if not materialized:
        raise ValueError("dataset contains no materialized non-locked trajectories")

    for row in materialized:
        relative_tensor = Path(row["v3_seed_dir"]) / "trajectory_tensors.npz"
        tensor_path = dataset_root / relative_tensor
        if not tensor_path.is_file():
            raise FileNotFoundError(f"missing trajectory tensor: {tensor_path}")
        provenance_name = "tensor." + relative_tensor.as_posix()
        verified_inputs[provenance_name] = _verify_manifest_file(
            tensor_path,
            dataset_manifest,
            entry_name=relative_tensor.as_posix(),
        )

    dataset_summary = _read_json(input_paths["dataset.dataset_summary.json"])
    if int(dataset_summary.get("unlocked_trajectory_count", -1)) != len(materialized):
        raise ValueError("dataset summary unlocked trajectory count disagrees with index")
    if int(dataset_summary.get("locked_test_trajectory_count", -1)) != len(locked):
        raise ValueError("dataset summary locked trajectory count disagrees with index")
    if dataset_summary.get("schema_version") == "PIJWM-DG-Contract-v3-dataset":
        split_counts: dict[str, int] = {}
        for row in index_rows:
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        if split_counts != {
            "train": 36,
            "validation": 12,
            "calibration": 6,
            "locked_test": 6,
        }:
            raise ValueError(f"formal dataset split counts changed: {split_counts}")

    locked_integrity = _read_json(input_paths["dataset.locked_test_integrity.json"])
    locked_index_identity = {
        (int(row["seed"]), row["split"], row["trajectory_id"]) for row in locked
    }
    locked_integrity_identity = {
        (int(row["seed"]), row["split"], row["trajectory_id"])
        for row in locked_integrity.get("trajectories", [])
    }
    sealed_integrity = (
        locked_integrity.get("label_content_read") is False
        and locked_integrity.get("tensorized") is False
        and int(locked_integrity.get("trajectory_count", -1)) == len(locked)
        and all(
            row.get("label_content_read") is False and row.get("tensorized") is False
            for row in locked_integrity.get("trajectories", [])
        )
        and locked_index_identity == locked_integrity_identity
    )
    locked_seed_dirs_absent = all(
        not (dataset_root / f"seed_{int(row['seed']):03d}").exists() for row in locked
    )
    factual_validation = _read_json(input_paths["factual.validation_report.json"])
    if factual_validation.get("checks", {}).get("locked_test_excluded_from_metrics") is not True:
        raise ValueError("factual source does not certify locked_test exclusion")

    normalization_stats = _read_json(input_paths["dataset.normalization_stats.json"])
    if normalization_stats.get("source_split") != "train":
        raise ValueError("normalization statistics must be train-only")
    normalization_stats = json.loads(json.dumps(normalization_stats))
    normalization_stats["features"]["task_dag_state"] = _train_dag_statistics(
        dataset_root, materialized
    )
    normalization_stats["schema_version"] = "PIJWM-Evaluation-Normalization-v3"
    normalization_stats["derived_fields"] = {
        "task_dag_state": "computed only from materialized train trajectories"
    }
    _write_json(output_dir / "evaluation_normalization_stats.json", normalization_stats)
    normalization_sha = _sha256(output_dir / "evaluation_normalization_stats.json")
    selection_scales = _selection_scales(normalization_stats)
    _write_json(
        output_dir / "checkpoint_selection_scales.json",
        {
            "schema_version": "PIJWM-Checkpoint-Selection-Scales-v3",
            "source_split": "train",
            "normalization_stats_sha256": normalization_sha,
            "scales": selection_scales,
        },
    )

    environment_splits: dict[str, list[str]] = {}
    for row in index_rows:
        environment_splits.setdefault(row["split"], []).append(row["trajectory_id"])
    registry = build_metric_registry()
    fair_protocol = build_fair_experiment_protocol(
        environment_splits=environment_splits,
        normalization_stats_sha256=normalization_sha,
    )
    registry_validation = validate_evaluation_protocol(registry, fair_protocol)
    _write_json(
        output_dir / "metric_registry.json",
        {"schema_version": REGISTRY_SCHEMA_VERSION, "metrics": registry},
    )
    _write_json(output_dir / "fair_experiment_protocol.json", fair_protocol)
    _write_json(
        output_dir / "factual_metric_mapping.json",
        {
            "schema_version": "PIJWM-Factual-Metric-Mapping-v3",
            "mappings": build_factual_metric_mapping(),
        },
    )
    _write_json(output_dir / "report_template.json", _report_template())

    reports: list[dict[str, Any]] = []
    information_activity_by_seed: dict[int, dict[str, Any]] = {}
    for row in materialized:
        tensor_path = dataset_root / row["v3_seed_dir"] / "trajectory_tensors.npz"
        if not tensor_path.is_file():
            raise FileNotFoundError(f"missing trajectory tensor: {tensor_path}")
        with np.load(tensor_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        try:
            information_activity_by_seed[int(row["seed"])] = _information_activity(arrays)
            for method in METHODS:
                report = evaluate_teacher_trajectory(
                    arrays,
                    method=method,
                    normalization_stats=normalization_stats,
                )
                report.update(
                    {
                        "trajectory_id": row["trajectory_id"],
                        "seed": int(row["seed"]),
                        "split": row["split"],
                    }
                )
                reports.append(report)
        finally:
            del arrays

    summaries = {}
    splits = sorted({row["split"] for row in materialized})
    for method in METHODS:
        summaries[method] = {
            "all_nonlocked": summarize_teacher_reports(
                [report for report in reports if report["method"] == method],
                selection_scales=selection_scales,
            ),
            "by_split": {
                split: summarize_teacher_reports(
                    [
                        report
                        for report in reports
                        if report["method"] == method and report["split"] == split
                    ],
                    selection_scales=selection_scales,
                )
                for split in splits
            },
        }
    baseline = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "dataset_id": dataset_root.name,
        "normalization_stats_sha256": normalization_sha,
        "locked_test_evaluated": False,
        "methods": list(METHODS),
        "trajectory_report_count": len(reports),
        "trajectory_reports": reports,
        "summaries": summaries,
    }
    _write_json(output_dir / "baseline_metrics.json", baseline)
    baseline_rows = _flatten_baselines(reports)
    _write_evaluation_csv(output_dir / "baseline_metrics.csv", baseline_rows)

    factual_rows = _canonical_factual_rows(
        _read_csv(factual_metrics_csv),
        materialized=materialized,
        information_activity_by_seed=information_activity_by_seed,
    )
    factual = _factual_availability(factual_rows)
    _write_json(output_dir / "factual_system_metric_availability.json", factual)
    evaluation_rows = baseline_rows + factual_rows
    _write_json(
        output_dir / "evaluation_rows.json",
        {"schema_version": "PIJWM-Canonical-Evaluation-Rows-v3", "rows": evaluation_rows},
    )
    _write_evaluation_csv(output_dir / "evaluation_rows.csv", evaluation_rows)

    provenance = _input_provenance(
        dataset_root=dataset_root,
        factual_metrics_csv=factual_metrics_csv,
        verified_inputs=verified_inputs,
    )
    _write_json(output_dir / "input_provenance.json", provenance)

    split_counts: dict[str, int] = {}
    for row in materialized:
        split = row["split"]
        split_counts[split] = split_counts.get(split, 0) + 1
    checks = {
        "registry_and_fairness_protocol_valid": bool(
            registry_validation["evaluation_protocol_ready"]
        ),
        "all_materialized_nonlocked_trajectories_evaluated_twice": (
            len(reports) == 2 * len(materialized)
        ),
        "locked_labels_remained_sealed": sealed_integrity,
        "locked_tensor_directories_absent": locked_seed_dirs_absent,
        "baseline_values_finite_or_explicit_na": all(
            _is_finite_or_na(report) for report in reports
        ),
        "factual_sidecar_excludes_locked_test": not factual["locked_test_rows_included"],
        "factual_metric_matrix_complete": (
            factual["metric_count"] == len(build_factual_metric_mapping()) == 22
            and factual["row_count"] == 22 * len(materialized)
        ),
        "information_link_activity_recomputed_from_information_graph": all(
            "physical" not in " ".join(row["source_fields"]).lower()
            for row in factual_rows
            if row["metric_id"] == "system.information_link_active_ratio"
        ),
        "checkpoint_continuous_term_executable": (
            summaries["zero_state"]["by_split"]["validation"]["metrics"]
            ["selection.required_continuous.normalized_error"]["status"]
            == "computed"
        ),
        "canonical_evaluation_rows_unique": len(evaluation_rows)
        == len(
            {
                (
                    row["trajectory_id"],
                    row["method_id"],
                    row["metric_id"],
                    row.get("source_metric_name"),
                )
                for row in evaluation_rows
            }
        ),
        "all_input_hashes_verified": all(
            row.get("verified") is True for row in verified_inputs.values()
        ),
        "report_template_frozen": True,
    }
    validation = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_protocol_ready": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "evaluated_trajectory_count": len(materialized),
        "locked_trajectory_count": len(locked),
        "baseline_report_count": len(reports),
        "split_counts": split_counts,
        "factual_metric_count": factual["metric_count"],
        "canonical_evaluation_row_count": len(evaluation_rows),
        "registry_metric_count": len(registry),
    }
    _write_json(output_dir / "validation_report.json", validation)
    if not validation["evaluation_protocol_ready"]:
        raise RuntimeError(f"evaluation bundle validation failed: {validation['failed_checks']}")
    _write_json(output_dir / "manifest.json", _manifest(output_dir))
    return validation


__all__ = ["SCHEMA_VERSION", "build_evaluation_bundle"]
