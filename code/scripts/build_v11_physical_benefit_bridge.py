"""Build auditable PI-JWM v11 physical-benefit cache augmentation."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.v11_labeling import (
    load_candidate_interaction_cache,
    save_candidate_interaction_cache,
)
from pi_jwm.v11_physical_benefit import (
    PHYSICAL_TASK_PREDICTION_FEATURES,
    PHYSICAL_PREDICTION_FEATURES,
    align_decision_points,
    audit_physical_bridge_protocol,
    augment_candidate_batch_with_physical_benefit,
    build_physical_training_batch,
    fit_physical_benefit_bridge,
    predict_physical_benefit,
)
from pi_jwm.v11_selector import DEFAULT_SELECTOR_SEEDS, canonical_sha256, file_sha256


DEFAULT_OUTPUT_DIR = (
    CODE_ROOT / "artifacts/reports/pi_jwm_v11_physical_benefit_bridge_20260719/formal"
)
SPLITS = ("train", "calibration", "validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-train-csv", type=Path, required=True)
    parser.add_argument("--physical-calibration-csv", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--sample-index-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    records = [dict(row) for row in rows]
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in records:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _input_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "physical_train_csv": Path(args.physical_train_csv),
        "physical_calibration_csv": Path(args.physical_calibration_csv),
        "train_cache": Path(args.train_cache),
        "calibration_cache": Path(args.calibration_cache),
        "validation_cache": Path(args.validation_cache),
        "sample_index_csv": Path(args.sample_index_csv),
    }


def _require_inputs(paths: Mapping[str, Path]) -> None:
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"physical bridge inputs are missing: {missing}")


def _validate_physical_alignment(
    rows: Sequence[Mapping[str, Any]],
    sample_index_rows: Sequence[Mapping[str, Any]],
    allowed_seeds: Sequence[int],
) -> dict[str, Any]:
    allowed = set(int(seed) for seed in allowed_seeds)
    points: dict[tuple[int, float], dict[str, Any]] = {}
    for source in rows:
        seed = int(source["seed"])
        if seed not in allowed:
            raise ValueError(f"physical row seed {seed} is outside its formal split")
        key = (seed, round(float(source["decision_time"]), 8))
        sample_id = int(source["sample_id"])
        previous = points.get(key)
        if previous is not None and int(previous["sample_id"]) != sample_id:
            raise ValueError(f"physical decision point has conflicting sample IDs: {key}")
        points[key] = {
            "seed": seed,
            "decision_time": float(source["decision_time"]),
            "sample_id": sample_id,
        }
    aligned, rejected = align_decision_points(list(points.values()), sample_index_rows)
    mismatched = [
        row
        for row in aligned
        if int(points[(int(row["seed"]), round(float(row["decision_time"]), 8))]["sample_id"])
        != int(row["sample_id"])
    ]
    if rejected or mismatched:
        raise ValueError(
            f"physical rows fail exact sample-time alignment: rejected={len(rejected)} "
            f"mismatched={len(mismatched)}"
        )
    return {
        "num_rows": len(rows),
        "num_decision_points": len(points),
        "num_exactly_aligned_points": len(aligned),
        "passed": len(aligned) == len(points),
    }


def validate_augmented_feature_order(
    manifests: Mapping[str, Mapping[str, Any]],
    physical_scope: str = "full",
) -> tuple[str, ...]:
    missing = [split for split in SPLITS if split not in manifests]
    if missing:
        raise ValueError(f"missing augmented cache manifests: {missing}")
    orders = [tuple(str(name) for name in manifests[split].get("feature_names", ())) for split in SPLITS]
    if any(order != orders[0] for order in orders[1:]):
        raise ValueError("augmented cache feature order mismatch")
    expected = (
        PHYSICAL_PREDICTION_FEATURES
        if str(physical_scope) == "full"
        else PHYSICAL_TASK_PREDICTION_FEATURES
    )
    if str(physical_scope) not in {"full", "task_only"}:
        raise ValueError(f"unknown physical feature scope: {physical_scope}")
    if orders[0][-len(expected) :] != expected:
        raise ValueError("augmented cache feature order must end with physical predictions")
    return orders[0]


def audit_rejected_candidates(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Allow only standalone offload labels that have no selector candidate analogue."""

    expected = []
    unexpected = []
    for source in rows:
        row = dict(source)
        is_expected = (
            str(row.get("reason")) == "unsupported_action_family"
            and str(row.get("action_family")).strip().lower() == "offload_target"
        )
        (expected if is_expected else unexpected).append(row)
    return {
        "passed": not unexpected,
        "expected_exclusion_count": len(expected),
        "unexpected_rejection_count": len(unexpected),
        "expected_exclusion_families": sorted(
            {str(row.get("action_family")) for row in expected}
        ),
        "unexpected_rejections": unexpected,
    }


def _cache_metadata(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as arrays:
        fold = arrays["sample_fold_id"].copy() if "sample_fold_id" in arrays.files else np.asarray([])
        action_names = tuple(
            str(value) for value in arrays["interaction_action_feature_names"].tolist()
        )
        return (
            arrays["sample_ids"].copy(),
            arrays["sample_seed"].copy(),
            None if fold.size == 0 else fold,
            action_names,
        )


def _sha256_manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.name == "sha256_manifest.json":
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def build_reproduction_command(args: argparse.Namespace) -> str:
    command = ["python", str(Path(__file__).resolve())]
    for name, path in _input_paths(args).items():
        command.extend((f"--{name.replace('_', '-')}", str(path)))
    command.extend(("--output-dir", str(args.output_dir)))
    if bool(args.dry_run):
        command.append("--dry-run")
    return shlex.join(command)


def run(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _input_paths(args)
    _require_inputs(inputs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_hashes = {name: file_sha256(path) for name, path in inputs.items()}
    if bool(args.dry_run):
        summary = {
            "framework": "PI-JWM",
            "result_kind": "diagnostic_only",
            "dry_run": True,
            "input_sha256": input_hashes,
            "augmented_caches": {},
            "matched_test_accessed": False,
            "external_holdout_accessed": False,
            "actual_outcome_feature_count": 0,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "reproduction_command.txt").write_text(
            build_reproduction_command(args) + "\n", encoding="utf-8"
        )
        return summary

    sample_index_rows = _read_csv(inputs["sample_index_csv"])
    physical_rows = {
        "train": _read_csv(inputs["physical_train_csv"]),
        "calibration": _read_csv(inputs["physical_calibration_csv"]),
    }
    alignment = {
        split: _validate_physical_alignment(
            physical_rows[split], sample_index_rows, DEFAULT_SELECTOR_SEEDS[split]
        )
        for split in ("train", "calibration")
    }
    cache_paths = {split: inputs[f"{split}_cache"] for split in SPLITS}
    loaded = {
        split: load_candidate_interaction_cache(cache_paths[split]) for split in SPLITS
    }
    configuration_digests = {
        str(loaded[split][3].get("configuration_digest")) for split in SPLITS
    }
    if len(configuration_digests) != 1:
        raise ValueError("source schema-v6 cache configuration digest mismatch")
    for split in SPLITS:
        manifest = loaded[split][3]
        if int(manifest.get("schema_version", -1)) != 6 or str(manifest.get("split_name")) != split:
            raise ValueError(f"source cache contract mismatch for {split}")

    metadata = {split: _cache_metadata(cache_paths[split]) for split in SPLITS}
    training_batches = {}
    for split in ("train", "calibration"):
        batch = loaded[split][0]
        sample_ids, sample_seed, _, _ = metadata[split]
        training_batches[split] = build_physical_training_batch(
            physical_rows[split], sample_ids, sample_seed, batch
        )
    protocol_audit = audit_physical_bridge_protocol(
        training_batches["train"].feature_names,
        {
            "train": DEFAULT_SELECTOR_SEEDS["train"],
            "calibration": DEFAULT_SELECTOR_SEEDS["calibration"],
        },
        matched_test_accessed=False,
        external_holdout_accessed=False,
    )
    fitted, model_report = fit_physical_benefit_bridge(
        training_batches["train"], training_batches["calibration"]
    )
    rejected = [
        {"split": split, **row}
        for split in ("train", "calibration")
        for row in training_batches[split].rejected_rows
    ]
    rejection_audit = audit_rejected_candidates(rejected)
    bridge_mode = (
        "full"
        if bool(model_report["passed"])
        else "task_only"
        if bool(model_report.get("task_only_passed"))
        else "failed"
    )
    bridge_checks = {
        "model_gate": bridge_mode != "failed",
        "full_model_gate": bool(model_report["passed"]),
        "task_model_gate": bool(model_report.get("task_model_passed")),
        "energy_model_gate": bool(model_report.get("energy_model_passed")),
        "protocol_audit": bool(protocol_audit["passed"]),
        "alignment_audit": all(value["passed"] for value in alignment.values()),
        "expected_exclusions_only": bool(rejection_audit["passed"]),
    }
    required_checks = (
        "model_gate",
        "protocol_audit",
        "alignment_audit",
        "expected_exclusions_only",
    )
    bridge_gate_passed = all(bridge_checks[name] for name in required_checks)

    with (output_dir / "physical_benefit_bridge.pkl").open("wb") as handle:
        pickle.dump(fitted, handle)
    calibration_prediction = predict_physical_benefit(
        fitted, training_batches["calibration"].features
    )
    np.savez_compressed(
        output_dir / "bridge_predictions.npz",
        train_sample_ids=training_batches["train"].sample_ids,
        train_oof_task_mean=fitted.oof_task_mean,
        train_oof_energy_mean=fitted.oof_energy_mean,
        train_oof_fold_id=fitted.oof_fold_id,
        calibration_sample_ids=training_batches["calibration"].sample_ids,
        calibration_task_mean=calibration_prediction.task_mean,
        calibration_task_lcb=calibration_prediction.task_lcb,
        calibration_task_ucb=calibration_prediction.task_ucb,
        calibration_energy_mean=calibration_prediction.energy_mean,
        calibration_energy_lcb=calibration_prediction.energy_lcb,
        calibration_energy_ucb=calibration_prediction.energy_ucb,
    )
    _write_csv(
        output_dir / "aligned_physical_rows.csv",
        [
            {"split": split, **row}
            for split in ("train", "calibration")
            for row in physical_rows[split]
        ],
    )
    _write_csv(output_dir / "rejected_physical_rows.csv", rejected)

    augmented_manifests = {}
    augmented_records = {}
    for split in SPLITS:
        batch, outcome, interactions, source_manifest = loaded[split]
        sample_ids, sample_seed, sample_fold_id, action_names = metadata[split]
        augmented = augment_candidate_batch_with_physical_benefit(
            batch,
            fitted,
            outcome.default_index,
            include_energy=bridge_mode == "full",
        )
        output_path = output_dir / f"candidate_labels_{split}_physical.npz"
        manifest = save_candidate_interaction_cache(
            output_path,
            split_name=split,
            sample_ids=sample_ids,
            sample_seed=sample_seed,
            batch=augmented,
            outcome=outcome,
            interactions=interactions,
            action_feature_names=action_names,
            configuration_digest=source_manifest.get("configuration_digest"),
            protocol_metadata={
                **dict(source_manifest.get("protocol_metadata", {})),
                "physical_benefit_bridge": "diagnostic_only",
                "physical_feature_scope": bridge_mode,
                "physical_energy_result_kind": (
                    "observable_prediction" if bridge_mode == "full" else "audit_only"
                ),
                "source_cache_sha256": str(source_manifest.get("cache_sha256", "")),
            },
            sample_fold_id=sample_fold_id,
        )
        augmented_manifests[split] = manifest
        augmented_records[split] = {
            "path": output_path.name,
            "sha256": str(manifest["cache_sha256"]),
        }
    feature_order = validate_augmented_feature_order(
        augmented_manifests, physical_scope=bridge_mode
    )

    split_hashes = {
        split: canonical_sha256(
            {
                "seed_values": augmented_manifests[split].get("seed_values", []),
                "num_samples": augmented_manifests[split].get("num_samples", 0),
                "cache_sha256": augmented_manifests[split].get("cache_sha256", ""),
            }
        )
        for split in SPLITS
    }
    summary = {
        "framework": "PI-JWM",
        "result_kind": "diagnostic_only",
        "dry_run": False,
        "bridge_gate_passed": bridge_gate_passed,
        "bridge_mode": bridge_mode,
        "physical_feature_scope": bridge_mode,
        "physical_energy_result_kind": (
            "observable_prediction" if bridge_mode == "full" else "audit_only"
        ),
        "bridge_gate_checks": bridge_checks,
        "model_report": model_report,
        "protocol_audit": protocol_audit,
        "alignment_audit": alignment,
        "rejected_candidate_count": len(rejected),
        "rejection_audit": rejection_audit,
        "input_sha256": input_hashes,
        "split_hashes": split_hashes,
        "augmented_caches": augmented_records,
        "feature_names": list(feature_order),
        "matched_test_accessed": False,
        "external_holdout_accessed": False,
        "actual_outcome_feature_count": 0,
    }
    summary["bridge_manifest_digest"] = canonical_sha256(summary)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "reproduction_command.txt").write_text(
        build_reproduction_command(args) + "\n", encoding="utf-8"
    )
    (output_dir / "sha256_manifest.json").write_text(
        json.dumps(_sha256_manifest(output_dir), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
