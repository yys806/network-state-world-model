from __future__ import annotations

"""Audit legacy non-locked evidence against the PI-JWM information-edge v4 contract."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.information_edge_contract_v4 import (  # noqa: E402
    CONTRACT_VERSION,
    MissingReason,
    build_field_registry,
    build_legacy_slot_mapping,
    validate_masked_field,
)


AUDIT_VERSION = "PIJWM-P1-MVS-v1"
ALLOWED_SPLITS = frozenset({"train", "validation", "calibration"})
LEGACY_TENSOR_SCHEMA_VERSION = "PIJWM-DG-Contract-v3-tensor"
LEGACY_DATASET_SCHEMA_VERSION = "PIJWM-DG-Contract-v3-dataset"
CONTRACT_MODULE_PATH = SRC_ROOT / "pi_jwm" / "information_edge_contract_v4.py"


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partition_trajectory_rows(
    rows: Iterable[Mapping[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    selected: list[dict[str, str]] = []
    locked_metadata: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        split = str(row.get("split", ""))
        if split in ALLOWED_SPLITS:
            selected.append(row)
        elif split == "locked_test":
            locked_metadata.append(
                {
                    "trajectory_id": str(row["trajectory_id"]),
                    "seed": int(row["seed"]),
                    "split": split,
                }
            )
        else:
            raise ValueError(f"unknown split: {split}")
    return selected, locked_metadata


def resolve_seed_dir(dataset_root: Path, row: Mapping[str, str]) -> Path:
    if str(row.get("split")) not in ALLOWED_SPLITS:
        raise ValueError("only non-locked splits may resolve seed directories")
    root = Path(dataset_root).resolve()
    seed_name = str(row.get("v3_seed_dir") or f"seed_{int(row['seed']):03d}")
    candidate = (root / seed_name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("seed path escapes dataset root") from error
    return candidate


def _read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_nonlocked_tensor(
    dataset_root: Path, row: Mapping[str, str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if str(row.get("split")) not in ALLOWED_SPLITS:
        raise ValueError("locked or unknown split tensor access is forbidden")
    seed_dir = resolve_seed_dir(dataset_root, row)
    manifest_path = seed_dir / "manifest.json"
    tensor_path = seed_dir / "trajectory_tensors.npz"
    manifest = _read_manifest(manifest_path)
    if str(manifest.get("schema_version")) != LEGACY_DATASET_SCHEMA_VERSION:
        raise ValueError(f"seed manifest schema mismatch: {manifest_path}")
    if (
        str(manifest.get("trajectory_id")) != str(row["trajectory_id"])
        or int(manifest.get("seed", -1)) != int(row["seed"])
        or str(manifest.get("split")) != str(row["split"])
    ):
        raise ValueError(f"seed manifest identity mismatch: {manifest_path}")
    expected = manifest.get("files", {}).get("trajectory_tensors.npz")
    if not isinstance(expected, dict):
        raise ValueError(f"seed manifest lacks trajectory tensor metadata: {manifest_path}")
    if tensor_path.stat().st_size != int(expected.get("size_bytes", -1)):
        raise ValueError(f"trajectory tensor size mismatch: {tensor_path}")
    if sha256_file(tensor_path) != str(expected.get("sha256", "")):
        raise ValueError(f"trajectory tensor hash mismatch: {tensor_path}")
    with np.load(tensor_path, allow_pickle=False) as loaded:
        values = np.asarray(loaded["information_edge_state"])
        masks = np.asarray(loaded["information_edge_feature_mask"])
        presence = np.asarray(loaded["information_edge_present"])
        kind_index = np.asarray(loaded["information_edge_kind_index"])
    if masks.dtype != np.bool_:
        raise ValueError("legacy information-edge mask must have bool dtype")
    if presence.dtype != np.bool_:
        raise ValueError("information-edge presence must have bool dtype")
    if kind_index.dtype.kind not in "iu":
        raise ValueError("information-edge kind index must have integer dtype")
    return values, masks, presence, kind_index, manifest


def summarize_legacy_coverage(
    *,
    trajectory_id: str,
    split: str,
    scenario_id: str,
    edge_type: str,
    values: np.ndarray,
    masks: np.ndarray,
    presence_mask: np.ndarray,
) -> list[dict[str, Any]]:
    values = np.asarray(values)
    masks = np.asarray(masks)
    if masks.dtype != np.bool_:
        raise ValueError("legacy information-edge mask must have bool dtype")
    if values.shape != masks.shape:
        raise ValueError("legacy value and mask shapes differ")
    if values.ndim < 1 or values.shape[-1] != 18:
        raise ValueError("legacy information-edge array must end with 18 features")
    presence_mask = np.asarray(presence_mask)
    if presence_mask.dtype != np.bool_:
        raise ValueError("information-edge presence mask must have bool dtype")
    if presence_mask.shape != values.shape[:-1]:
        raise ValueError("information-edge presence shape differs from values")
    mapping = build_legacy_slot_mapping()
    rows: list[dict[str, Any]] = []
    for index, mapping_row in enumerate(mapping):
        value = values[..., index][presence_mask]
        mask = masks[..., index][presence_mask]
        rows.append(
            {
                "trajectory_id": str(trajectory_id),
                "split": str(split),
                "scenario_id": str(scenario_id),
                "edge_type": str(edge_type),
                "legacy_index": index,
                "legacy_slot": mapping_row["legacy_slot"],
                "total_count": int(mask.size),
                "valid_count": int(mask.sum()),
                "valid_zero_count": int(np.count_nonzero(mask & (value == 0))),
                "invalid_nonzero_count": int(
                    np.count_nonzero((~mask) & (value != 0))
                ),
                "finite_valid_count": int(
                    np.count_nonzero(mask & np.isfinite(value))
                ),
                "evidence_scope": "legacy_observation_only",
                "v4_field_implemented": False,
            }
        )
    return rows


def _first_observed_candidates(
    row: Mapping[str, str],
    values: np.ndarray,
    masks: np.ndarray,
    presence: np.ndarray,
    kind_index: np.ndarray,
    edge_types: Sequence[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for legacy_index in range(18):
        locations = np.argwhere(masks[..., legacy_index] & presence)
        if locations.size == 0:
            continue
        location = locations[0]
        time_index = int(location[0])
        edge_index = int(location[1]) if len(location) > 1 else 0
        value = float(values[tuple(location) + (legacy_index,)])
        candidates.append(
            {
                "trajectory_id": str(row["trajectory_id"]),
                "split": str(row["split"]),
                "seed": int(row["seed"]),
                "time_index": time_index,
                "edge_index": edge_index,
                "legacy_index": legacy_index,
                "value": value,
                "valid": True,
                "scenario_id": str(row["scenario_id"]),
                "edge_type": str(edge_types[int(kind_index[edge_index])]),
            }
        )
    return candidates


def build_micro_sample(
    *, observed_candidates: Sequence[Mapping[str, Any]]
) -> dict[str, np.ndarray]:
    mapping = build_legacy_slot_mapping()
    rows: list[dict[str, Any]] = []
    for source in observed_candidates:
        legacy_index = int(source["legacy_index"])
        rows.append(
            {
                "sample_origin": "observed_nonlocked",
                "training_eligible": False,
                "trajectory_id": str(source["trajectory_id"]),
                "split": str(source["split"]),
                "seed": int(source["seed"]),
                "time_index": int(source["time_index"]),
                "edge_index": int(source["edge_index"]),
                "field_name": str(mapping[legacy_index]["v4_target"]),
                "candidate_v4_target": str(mapping[legacy_index]["v4_target"]),
                "v4_field_implemented": False,
                "value": float(source["value"]),
                "valid_mask": bool(source["valid"]),
                "missing_reason": MissingReason.NONE.value,
                "legacy_slot": str(mapping[legacy_index]["legacy_slot"]),
            }
        )
        rows[-1]["field_name"] = str(mapping[legacy_index]["legacy_slot"])
    fixture_cases = (
        ("valid_zero", 0.0, True, MissingReason.NONE),
        ("no_history", 0.0, False, MissingReason.NO_HISTORY),
        ("not_applicable", 0.0, False, MissingReason.NOT_APPLICABLE),
        ("not_collected", 0.0, False, MissingReason.NOT_COLLECTED),
        ("source_error", 0.0, False, MissingReason.SOURCE_ERROR),
    )
    for name, value, valid, reason in fixture_cases:
        rows.append(
            {
                "sample_origin": "contract_fixture",
                "training_eligible": False,
                "trajectory_id": "__contract_fixture__",
                "split": "fixture",
                "seed": -1,
                "time_index": -1,
                "edge_index": -1,
                "field_name": f"fixture.{name}",
                "candidate_v4_target": "",
                "v4_field_implemented": False,
                "value": value,
                "valid_mask": valid,
                "missing_reason": reason.value,
                "legacy_slot": "",
            }
        )
    sample = {
        "sample_origin": np.asarray([row["sample_origin"] for row in rows], dtype="U32"),
        "training_eligible": np.asarray([row["training_eligible"] for row in rows], dtype=bool),
        "trajectory_id": np.asarray([row["trajectory_id"] for row in rows], dtype="U96"),
        "split": np.asarray([row["split"] for row in rows], dtype="U16"),
        "seed": np.asarray([row["seed"] for row in rows], dtype=np.int64),
        "time_index": np.asarray([row["time_index"] for row in rows], dtype=np.int64),
        "edge_index": np.asarray([row["edge_index"] for row in rows], dtype=np.int64),
        "field_name": np.asarray([row["field_name"] for row in rows], dtype="U96"),
        "candidate_v4_target": np.asarray([row["candidate_v4_target"] for row in rows], dtype="U96"),
        "v4_field_implemented": np.asarray([row["v4_field_implemented"] for row in rows], dtype=bool),
        "value": np.asarray([row["value"] for row in rows], dtype=np.float32),
        "valid_mask": np.asarray([row["valid_mask"] for row in rows], dtype=bool),
        "missing_reason": np.asarray([row["missing_reason"] for row in rows], dtype=np.uint8),
        "legacy_slot": np.asarray([row["legacy_slot"] for row in rows], dtype="U64"),
    }
    validate_masked_field(sample["value"], sample["valid_mask"], sample["missing_reason"])
    return sample


def _aggregate_coverage(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    aggregate: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["split"]),
            str(row["scenario_id"]),
            str(row["edge_type"]),
            int(row["legacy_index"]),
            str(row["legacy_slot"]),
        )
        target = aggregate.setdefault(
            key,
            {
                "split": key[0],
                "scenario_id": key[1],
                "edge_type": key[2],
                "legacy_index": key[3],
                "legacy_slot": key[4],
                "trajectory_count": 0,
                "total_count": 0,
                "valid_count": 0,
                "valid_zero_count": 0,
                "invalid_nonzero_count": 0,
                "finite_valid_count": 0,
                "evidence_scope": "legacy_observation_only",
                "v4_field_implemented": False,
            },
        )
        target["trajectory_count"] += 1
        for name in (
            "total_count",
            "valid_count",
            "valid_zero_count",
            "invalid_nonzero_count",
            "finite_valid_count",
        ):
            target[name] += int(row[name])
    return [aggregate[key] for key in sorted(aggregate)]


def _validate_coverage_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    invalid_nonzero = sum(int(row["invalid_nonzero_count"]) for row in rows)
    if invalid_nonzero:
        raise ValueError(f"invalid nonzero legacy values detected: {invalid_nonzero}")
    invalid_finite = sum(
        int(row["valid_count"]) - int(row["finite_valid_count"]) for row in rows
    )
    if invalid_finite:
        raise ValueError(f"non-finite valid legacy values detected: {invalid_finite}")


def _existing_output_guard(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")


def _file_metadata(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _code_file_metadata() -> dict[str, dict[str, Any]]:
    paths = (Path(__file__).resolve(), CONTRACT_MODULE_PATH.resolve())
    return {str(path): _file_metadata(path) for path in paths}


def _locked_content_accessed(
    input_files: Mapping[str, Any], locked_seed_names: set[str]
) -> bool:
    return any(
        locked_seed_names.intersection(Path(path).parts)
        for path in input_files
    )


def _write_failure_bundle(
    *,
    failure_dir: Path,
    error: Exception,
    current_row: Mapping[str, Any] | None,
    input_files: Mapping[str, Any],
    locked_seed_names: set[str],
) -> None:
    _existing_output_guard(failure_dir)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{failure_dir.name}.staging-", dir=failure_dir.parent)
    )
    try:
        row = dict(current_row or {})
        rejected = [
            {
                "trajectory_id": str(row.get("trajectory_id", "")),
                "split": str(row.get("split", "")),
                "seed": str(row.get("seed", "")),
                "record_scope": "trajectory" if row else "audit",
                "reason": type(error).__name__,
                "detail": str(error),
            }
        ]
        write_csv(
            staging / "rejected_records.csv",
            rejected,
            ("trajectory_id", "split", "seed", "record_scope", "reason", "detail"),
        )
        locked_accessed = _locked_content_accessed(input_files, locked_seed_names)
        summary = {
            "contract_version": CONTRACT_VERSION,
            "audit_version": AUDIT_VERSION,
            "evidence_scope": "failed_p1_mvs_audit",
            "p1_mvs_complete": False,
            "locked_test_accessed": locked_accessed,
            "gpu_started": False,
            "rejected_record_count": 1,
            "error_type": type(error).__name__,
            "error_detail": str(error),
        }
        write_json(staging / "audit_summary.json", summary)
        managed_names = ("rejected_records.csv", "audit_summary.json")
        manifest = {
            "audit_version": AUDIT_VERSION,
            "contract_version": CONTRACT_VERSION,
            "audit_status": "failed",
            "locked_test_accessed": locked_accessed,
            "gpu_started": False,
            "input_files": dict(input_files),
            "code_files": _code_file_metadata(),
            "files": {
                name: _file_metadata(staging / name) for name in managed_names
            },
        }
        write_json(staging / "manifest.json", manifest)
        os.replace(staging, failure_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_audit(*, v3_dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    dataset_root = Path(v3_dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    _existing_output_guard(output_dir)
    failure_dir = output_dir.with_name(f"{output_dir.name}_failed")
    _existing_output_guard(failure_dir)
    index_path = dataset_root / "trajectory_index.csv"
    contract_path = dataset_root / "tensor_contract.json"

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    input_files: dict[str, dict[str, Any]] = {}
    locked_seed_names: set[str] = set()
    current_row: Mapping[str, Any] | None = None
    try:
        index_rows = read_csv(index_path)
        selected_rows, locked_metadata = partition_trajectory_rows(index_rows)
        locked_seed_names = {
            str(row.get("v3_seed_dir") or f"seed_{int(row['seed']):03d}")
            for row in index_rows
            if str(row.get("split")) == "locked_test"
        }
        if not selected_rows:
            raise ValueError("no non-locked trajectories are available")
        detailed_coverage: list[dict[str, Any]] = []
        observed_candidates: list[dict[str, Any]] = []
        input_files = {
            str(index_path): _file_metadata(index_path),
            str(contract_path): _file_metadata(contract_path),
        }
        contract = _read_manifest(contract_path)
        if str(contract.get("schema_version")) != LEGACY_TENSOR_SCHEMA_VERSION:
            raise ValueError("legacy tensor contract schema mismatch")
        expected_feature_order = [
            str(row["legacy_slot"]) for row in build_legacy_slot_mapping()
        ]
        if contract.get("information_edge_features") != expected_feature_order:
            raise ValueError("legacy information-edge feature order mismatch")
        edge_types = [str(value) for value in contract["information_edge_types"]]
        if not edge_types or len(edge_types) != len(set(edge_types)):
            raise ValueError("tensor contract edge types are empty or duplicated")
        for row in selected_rows:
            current_row = row
            values, masks, presence, kind_index, _ = load_nonlocked_tensor(
                dataset_root, row
            )
            if values.shape[:-1] != presence.shape:
                raise ValueError("information-edge presence shape differs from values")
            if np.any(masks & (~presence[..., None])):
                raise ValueError("legacy feature mask is true on an absent edge")
            if kind_index.shape != (values.shape[-2],):
                raise ValueError("information-edge kind index shape differs from edge axis")
            present_kind_indices = sorted(
                {int(value) for value in kind_index[np.any(presence, axis=0)]}
            )
            if any(value < 0 or value >= len(edge_types) for value in present_kind_indices):
                raise ValueError("information-edge kind index out of range")
            for edge_type_index in present_kind_indices:
                type_presence = presence & (
                    kind_index[None, :] == edge_type_index
                )
                trajectory_rows = summarize_legacy_coverage(
                    trajectory_id=str(row["trajectory_id"]),
                    split=str(row["split"]),
                    scenario_id=str(row["scenario_id"]),
                    edge_type=edge_types[edge_type_index],
                    values=values,
                    masks=masks,
                    presence_mask=type_presence,
                )
                _validate_coverage_rows(trajectory_rows)
                detailed_coverage.extend(trajectory_rows)
            observed_candidates.extend(
                _first_observed_candidates(
                    row,
                    values,
                    masks,
                    presence,
                    kind_index,
                    edge_types,
                )
            )
            seed_dir = resolve_seed_dir(dataset_root, row)
            for name in ("manifest.json", "trajectory_tensors.npz"):
                path = seed_dir / name
                input_files[str(path)] = _file_metadata(path)

        current_row = None
        locked_accessed = _locked_content_accessed(input_files, locked_seed_names)

        coverage_rows = _aggregate_coverage(detailed_coverage)
        _validate_coverage_rows(coverage_rows)
        valid_indices = sorted(
            {
                int(row["legacy_index"])
                for row in coverage_rows
                if int(row["valid_count"]) > 0
            }
        )
        acceptance = {
            "legacy_width_is_18": len(build_legacy_slot_mapping()) == 18,
            "legacy_valid_indices_match_audit": valid_indices == [0, 1, 8, 11, 12],
            "invalid_nonzero_count_is_zero": all(
                int(row["invalid_nonzero_count"]) == 0 for row in coverage_rows
            ),
            "valid_values_are_finite": all(
                int(row["valid_count"]) == int(row["finite_valid_count"])
                for row in coverage_rows
            ),
            "locked_test_content_not_read": not locked_accessed,
        }
        if not all(acceptance.values()):
            raise ValueError(f"P1 acceptance failed: {acceptance}")

        registry = {
            "contract_version": CONTRACT_VERSION,
            "fields": build_field_registry(),
            "missing_reasons": [
                {"name": reason.name.lower(), "value": reason.value}
                for reason in MissingReason
            ],
        }
        write_json(staging / "field_registry.json", registry)
        legacy_mapping = build_legacy_slot_mapping()
        write_csv(
            staging / "legacy_18_slot_mapping.csv",
            legacy_mapping,
            (
                "legacy_index",
                "legacy_slot",
                "decision",
                "v4_target",
                "source_status",
                "reason",
            ),
        )
        micro_sample = build_micro_sample(observed_candidates=observed_candidates)
        np.savez_compressed(staging / "micro_sample.npz", **micro_sample)
        write_csv(
            staging / "field_coverage.csv",
            coverage_rows,
            (
                "split",
                "scenario_id",
                "edge_type",
                "legacy_index",
                "legacy_slot",
                "trajectory_count",
                "total_count",
                "valid_count",
                "valid_zero_count",
                "invalid_nonzero_count",
                "finite_valid_count",
                "evidence_scope",
                "v4_field_implemented",
            ),
        )
        rejected_rows: list[dict[str, Any]] = []
        write_csv(
            staging / "rejected_records.csv",
            rejected_rows,
            ("trajectory_id", "split", "seed", "record_scope", "reason", "detail"),
        )
        split_counts = Counter(str(row["split"]) for row in selected_rows)
        scenario_ids = sorted({str(row["scenario_id"]) for row in selected_rows})
        observed_edge_types = sorted({str(row["edge_type"]) for row in coverage_rows})
        summary = {
            "contract_version": CONTRACT_VERSION,
            "audit_version": AUDIT_VERSION,
            "evidence_scope": "self_audited_p1_mvs",
            "p1_mvs_complete": True,
            "v4_collector_implemented": False,
            "v4_dataset_complete": False,
            "v4_model_trained": False,
            "locked_test_accessed": locked_accessed,
            "gpu_started": False,
            "cpu_core_action_decided": False,
            "nonlocked_trajectory_counts": dict(sorted(split_counts.items())),
            "scenario_count": len(scenario_ids),
            "scenario_ids": scenario_ids,
            "observed_edge_types": observed_edge_types,
            "locked_trajectory_metadata_count": len(locked_metadata),
            "legacy_valid_feature_indices": valid_indices,
            "legacy_valid_feature_count": len(valid_indices),
            "legacy_missing_feature_count": 18 - len(valid_indices),
            "training_eligible_micro_sample_count": int(
                micro_sample["training_eligible"].sum()
            ),
            "rejected_record_count": len(rejected_rows),
            "rejection_policy": "success bundle is published only when no record is rejected; failures are retained in a sibling _failed bundle",
            "acceptance": acceptance,
        }
        write_json(staging / "audit_summary.json", summary)

        managed_names = (
            "field_registry.json",
            "legacy_18_slot_mapping.csv",
            "micro_sample.npz",
            "field_coverage.csv",
            "rejected_records.csv",
            "audit_summary.json",
        )
        manifest = {
            "audit_version": AUDIT_VERSION,
            "contract_version": CONTRACT_VERSION,
            "locked_test_accessed": locked_accessed,
            "gpu_started": False,
            "input_files": input_files,
            "code_files": _code_file_metadata(),
            "locked_trajectory_metadata": locked_metadata,
            "files": {
                name: {
                    "sha256": sha256_file(staging / name),
                    "size_bytes": (staging / name).stat().st_size,
                }
                for name in managed_names
            },
        }
        manifest["protocol_sha256"] = manifest["files"]["field_registry.json"][
            "sha256"
        ]
        manifest["config_sha256"] = input_files[str(contract_path)]["sha256"]
        write_json(staging / "manifest.json", manifest)
        os.replace(staging, output_dir)
        return summary
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        _write_failure_bundle(
            failure_dir=failure_dir,
            error=error,
            current_row=current_row,
            input_files=input_files,
            locked_seed_names=locked_seed_names,
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v3-dataset-root",
        type=Path,
        default=CODE_ROOT
        / "artifacts"
        / "datasets"
        / "airfogsim_teacher_aligned_v3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_ROOT
        / "artifacts"
        / "audit"
        / "pi_jwm_p1_information_edge_contract_v4",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_audit(
        v3_dataset_root=args.v3_dataset_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
