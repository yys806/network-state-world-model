"""Build the smallest non-locked Step 3.2 batch validation bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))

from pi_jwm.step3_2_batch_preprocessing_v1 import (  # noqa: E402
    RawSource,
    apply_normalization,
    audit_batch_future_action_references,
    build_batch,
    fit_train_normalization_stats,
    write_batch_bundle,
)


DEFAULT_OUTPUT = ROOT / "code/artifacts/protocols/pi_jwm_step3_2_raw_to_dataset_batch_v1_20260919"
SOURCES = (
    RawSource(ROOT / "code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/real_communication_outcome_semantics.json", "dev_train"),
    RawSource(ROOT / "code/artifacts/protocols/pi_jwm_step3_2_raw_dev_seed1_20260919/real_communication_outcome_semantics.json", "dev_train"),
    RawSource(ROOT / "code/artifacts/protocols/pi_jwm_step3_2_raw_dev_seed2_20260919/real_communication_outcome_semantics.json", "dev_validation"),
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    bundle = build_batch(SOURCES)
    stats = fit_train_normalization_stats(bundle["samples"])
    normalized = apply_normalization(bundle["samples"], stats)
    future_reference_audit = audit_batch_future_action_references(SOURCES, bundle=bundle)
    bundle["future_reference_audit"] = future_reference_audit
    write_batch_bundle(bundle, stats, normalized, args.output_dir)
    rebuilt = build_batch(SOURCES)
    rebuilt["future_reference_audit"] = audit_batch_future_action_references(SOURCES, bundle=rebuilt)
    deterministic = _digest(bundle) == _digest(rebuilt)
    provenance = bundle["provenance"]
    train_ids = {row["trajectory_id"] for row in provenance if row["split"] == "dev_train"}
    validation_ids = {row["trajectory_id"] for row in provenance if row["split"] == "dev_validation"}
    split_isolation = bool(provenance) and len({row["trajectory_id"] for row in provenance}) == len(provenance) and not (train_ids & validation_ids)
    source_hashes = all(len(row.get("source_sha256", "")) == 64 for row in provenance)
    causal_windows = all(
        sample["metadata"]["history_frame_indices"][-1] == sample["metadata"]["anchor_decision_frame"]
        and sample["metadata"]["future_action_frame_indices"][0] == sample["metadata"]["anchor_decision_frame"]
        for sample in bundle["samples"]
    )
    train_only_fit = stats.get("source_split") == "dev_train" and all(
        feature.get("mask_policy") == "presence=true AND feature_mask=true AND value!=null; train split only"
        for feature in stats.get("features", {}).values()
    )
    report = {
        "schema_version": "PI-JWM-Step-3.2-Validation-Report-v1",
        "passed": deterministic,
        "checks": {
            "trajectory_level_split": split_isolation,
            "no_trajectory_cross_split": not bool(train_ids & validation_ids),
            "causal_windows_only": causal_windows,
            "train_only_normalization_fit": train_only_fit,
            "source_sha256_traceable": source_hashes,
            "seed_source_lineage_auditable": all("seed" in row and "lineage_key" in row for row in provenance),
            "deterministic_rebuild": deterministic,
            "batch_scope_non_locked": bundle["scope"],
        },
        "evidence_source": {
            "trajectory_level_split": "computed from final provenance trajectory_id sets",
            "causal_windows_only": "computed from final sample metadata frame indices",
            "train_only_normalization_fit": "fit_train_normalization_stats split filter and emitted mask policy",
        },
        "future_reference_audit": future_reference_audit,
        "observation_only": True,
        "formal_dataset": False,
        "training": False,
        "gpu": False,
        "locked_test_accessed": False,
        "trajectory_count": len(bundle["provenance"]),
        "window_count": len(bundle["samples"]),
        "split_counts": {
            split: sum(row["metadata"]["split"] == split for row in bundle["samples"])
            for split in ("dev_train", "dev_validation")
        },
    }
    (args.output_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "future_action_reference_audit.json").write_text(json.dumps(future_reference_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation_report_sha256"] = hashlib.sha256((args.output_dir / "validation_report.json").read_bytes()).hexdigest()
    manifest["future_action_reference_audit_sha256"] = hashlib.sha256((args.output_dir / "future_action_reference_audit.json").read_bytes()).hexdigest()
    manifest["source_policy"] = "reuse frozen Step 2.4 Raw collector; no schema or simulator changes"
    manifest["observation_only"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
