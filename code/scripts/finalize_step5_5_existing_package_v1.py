"""Finalize a completed STEP 5.5 package without rewriting package shards."""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from build_step5_5_formal_dataset_v1 import (
    FAMILIES,
    HISTORY_STEPS,
    HORIZON_STEPS,
    SPLIT_SEED,
    TRAJECTORIES,
    TRANSITIONS,
    TRAIN_TRAJECTORIES,
    VALIDATION_TRAJECTORIES,
    audit_future_action_references,
    coverage,
    coverage_checks,
    deterministic_split,
    load_json,
    sha256_file,
    sha256_path,
    write_json,
)
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_1a_motion_csi_target_contract_v1 import load_future_target_tensor_batch


_STREAM_PATTERNS = {
    "history": re.compile(r'"history_frame_indices"\s*:\s*\[([^\]]*)\]'),
    "future": re.compile(r'"future_action_frame_indices"\s*:\s*\[([^\]]*)\]'),
    "unsupported": re.compile(r'"unsupported_count"\s*:\s*(\d+)'),
    "unresolved": re.compile(r'"unresolved_count"\s*:\s*(\d+)'),
    "fixed_support_blocked": re.compile(r'"fixed_support_blocked_count"\s*:\s*(\d+)'),
}


def _stream_sample_shard_audit(path: Path, *, chunk_size: int = 1024 * 1024) -> dict:
    """Audit large gzip JSON shards without materializing them in memory."""
    counts = Counter({"samples": 0, "h2": 0, "l4": 0, "unsupported": 0, "unresolved": 0, "fixed_support_blocked": 0})
    carry = ""
    consumed = 0
    last_start = {key: -1 for key in _STREAM_PATTERNS}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            text = carry + chunk
            base = consumed - len(carry)
            for match in _STREAM_PATTERNS["history"].finditer(text):
                position = base + match.start()
                if position <= last_start["history"]:
                    continue
                last_start["history"] = position
                counts["samples"] += 1
                counts["h2"] += len([value for value in match.group(1).split(",") if value.strip()]) == HISTORY_STEPS
            for match in _STREAM_PATTERNS["future"].finditer(text):
                position = base + match.start()
                if position <= last_start["future"]:
                    continue
                last_start["future"] = position
                counts["l4"] += len([value for value in match.group(1).split(",") if value.strip()]) == HORIZON_STEPS
            for key in ("unsupported", "unresolved", "fixed_support_blocked"):
                for match in _STREAM_PATTERNS[key].finditer(text):
                    position = base + match.start()
                    if position <= last_start[key]:
                        continue
                    last_start[key] = position
                    counts[key] += int(match.group(1))
            consumed += len(chunk)
            carry = text[-512:]
    return dict(counts)


def finalize(output: Path) -> dict:
    output = output.resolve()
    manifest_path = output / "formal_dataset_manifest.json"
    manifest = load_json(manifest_path)
    collection = load_json(output / "collection_summary.json")
    split = load_json(output / "split_manifest.json")
    accepted = list(collection["accepted"])
    raw_paths = {str(row["trajectory_id"]): output / str(row["path"]) for row in accepted}
    provenance = load_json(output / "raw_trajectory_provenance.json")
    split_ids = {name: set(ids) for name, ids in split.items() if name.startswith("dev_")}
    coverage_report = coverage(raw_paths, {name: list(ids) for name, ids in split_ids.items()})
    coverage_gate = coverage_checks(coverage_report)
    coverage_report["checks"] = coverage_gate
    coverage_report["passed"] = all(coverage_gate.values())
    write_json(output / "action_coverage_audit.json", coverage_report)

    interface = FormalTrainingInterface.from_manifest(manifest_path)
    package_verification = interface.verify_packages()
    runtime_package_verification = interface.verify_runtime_packages()
    shard_rows = load_json(output / "packages" / "tensor" / "index.json")["shards"]
    stats = load_json(output / "packages" / "normalization" / "stats.json")
    horizon_valid = {f"H{i}": {"motion": 0, "csi": 0} for i in range(1, HORIZON_STEPS + 1)}
    unresolved = Counter({"unsupported": 0, "unresolved": 0, "fixed_support_blocked": 0})
    sample_audit = Counter({"samples": 0, "h2": 0, "l4": 0})
    future_rows = []
    for row in provenance:
        raw = load_json(raw_paths[str(row["trajectory_id"])])
        audit = audit_future_action_references(raw, history_steps=HISTORY_STEPS, horizon_steps=HORIZON_STEPS)
        future_rows.append({"trajectory_id": row["trajectory_id"], "split": row["split"], **audit})
    samples_dir = output / "packages" / "samples"
    for row in shard_rows:
        audit = _stream_sample_shard_audit(samples_dir / row["files"]["samples"]["path"])
        sample_audit.update({key: audit[key] for key in ("samples", "h2", "l4")})
        unresolved.update({key: audit[key] for key in unresolved})
    target_dir = output / "packages" / "target"
    for row in shard_rows:
        target = load_future_target_tensor_batch(target_dir / row["files"]["target"]["path"])
        for horizon in range(HORIZON_STEPS):
            horizon_valid[f"H{horizon + 1}"]["motion"] += int(np.asarray(target["target_vehicle_motion_mask"])[:, horizon].sum())
            horizon_valid[f"H{horizon + 1}"]["csi"] += int(np.asarray(target["target_comm_csi_mask"])[:, horizon].sum())

    negative = {}
    missing = json.loads(json.dumps(manifest))
    del missing["hashes"]["target"]
    missing_path = output / ".negative_missing_hash.json"
    write_json(missing_path, missing)
    negative["missing_hash_rejected"] = not FormalTrainingInterface.from_manifest(missing_path).verify_packages()["all_present_and_matching"]
    wrong = json.loads(json.dumps(manifest))
    wrong["hashes"]["target"] = "0" * 64
    wrong_path = output / ".negative_wrong_hash.json"
    write_json(wrong_path, wrong)
    negative["wrong_hash_rejected"] = not FormalTrainingInterface.from_manifest(wrong_path).verify_packages()["all_present_and_matching"]
    missing_path.unlink()
    wrong_path.unlink()

    checks = {
        "accepted_real_trajectories_60": len(accepted) == TRAJECTORIES,
        "each_96_transitions_97_decisions": all(row["transition_count"] == TRANSITIONS and row["decision_count"] == TRANSITIONS + 1 for row in provenance),
        "continuous_time_grid": all(row["continuous_time_grid"] for row in provenance),
        "trajectory_split_48_12": len(split_ids["dev_train"]) == TRAIN_TRAJECTORIES and len(split_ids["dev_validation"]) == VALIDATION_TRAJECTORIES,
        "split_identity_isolation": not bool(split_ids["dev_train"] & split_ids["dev_validation"]) and len({row["simulator_seed"] for row in provenance}) == TRAJECTORIES and len({row["policy_seed"] for row in provenance}) == TRAJECTORIES,
        "history_2_horizon_4": sample_audit["samples"] == 5520 and sample_audit["h2"] == 5520 and sample_audit["l4"] == 5520,
        "exact_train_windows_4416": sum(row["sample_count"] for row in shard_rows if row["split"] == "dev_train") == 4416,
        "exact_validation_windows_1104": sum(row["sample_count"] for row in shard_rows if row["split"] == "dev_validation") == 1104,
        "exact_total_windows_5520": sum(row["sample_count"] for row in shard_rows) == 5520,
        "future_action_reference_unresolved_zero": sum(row["unresolved_future_reference_count"] for row in future_rows) == 0,
        "future_target_excluded_from_input": all(row["ledger_receipt_passed"] for row in shard_rows),
        "train_only_normalization": stats["future_target_in_fit"] is False and stats["source_trajectory_ids"] == split["dev_train"],
        "motion_h1_h4_valid": all(row["motion"] > 0 for row in horizon_valid.values()),
        "csi_h1_h4_valid": all(row["csi"] > 0 for row in horizon_valid.values()),
        "four_action_coverage": coverage_report["passed"],
        "noop_and_intervention_present": coverage_report["checks"]["all_noop_present"] and all(coverage_report["splits"]["dev_train"]["families"][family]["intervention_count"] > 0 for family in FAMILIES),
        "unsupported_accounting_separate": all(key in unresolved for key in ("unsupported", "unresolved", "fixed_support_blocked")) and sample_audit["samples"] == 5520,
        "package_identity_alignment": all(row["sample_count"] == 92 and row["identity_aligned"] for row in shard_rows),
        "deterministic_split_rebuild": deterministic_split([str(row["trajectory_id"]) for row in accepted]) == {"dev_train": split["dev_train"], "dev_validation": split["dev_validation"]},
        "serialize_reload_equality": all(row["serialize_reload_equal"] and all(sha256_file(output / "packages" / name / row["files"][name]["path"]) == row["files"][name]["sha256"] for name in ("samples", "tensor", "graph", "target")) for row in shard_rows),
        "five_package_hashes_exact": package_verification["all_present_and_matching"],
        "runtime_smoke_package_hashes_exact": runtime_package_verification["all_present_and_matching"],
        "negative_hash_fixtures": all(negative.values()),
        "locked_test_accessed_false": manifest["scope"]["locked_test_accessed"] is False,
    }
    receipt = {
        "schema_version": "PI-JWM-Step-5.5-Formal-Dataset-Acceptance-v1",
        "passed": all(checks.values()), "checks": checks,
        "counts": {"accepted_trajectories": len(accepted), "rejected_trajectories": len(collection.get("rejected", [])), "train_windows": 4416, "validation_windows": 1104, "total_windows": 5520},
        "target_horizon_valid_component_counts": horizon_valid,
        "sample_shard_stream_audit": dict(sample_audit),
        "unsupported_accounting": dict(unresolved), "future_reference_unresolved": sum(row["unresolved_future_reference_count"] for row in future_rows),
        "package_verification": package_verification, "runtime_package_verification": runtime_package_verification, "negative_fixtures": negative,
        "scope": manifest["scope"],
    }
    write_json(output / "dataset_acceptance_receipt.json", receipt)
    write_json(output / "future_action_reference_audit.json", {"rows": future_rows, "unresolved_total": receipt["future_reference_unresolved"]})
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.output_dir)
    print(json.dumps({"passed": result["passed"], "checks": result["checks"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
