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
    evaluate_validation_checks,
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
    acceptance_checks = evaluate_validation_checks(bundle, stats, deterministic_rebuild=deterministic)
    report = {
        "schema_version": "PI-JWM-Step-3.2-Validation-Report-v1",
        "passed": all(acceptance_checks.values()),
        "checks": {**acceptance_checks, "batch_scope_non_locked": bundle["scope"]},
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
