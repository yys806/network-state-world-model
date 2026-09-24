"""Read-only audit of future Return birth in the accepted Formal v1 samples."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from pi_jwm.step5_5_fixed_support_audit_v1 import detect_future_return_birth


def audit(manifest_path: Path) -> dict:
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_root = root / manifest["packages"]["samples"]
    counts = Counter()
    by_split = {}
    examples = []
    unique_births = set()
    for path in sorted(sample_root.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            samples = json.load(handle)
        counts["trajectories"] += 1
        for sample in samples:
            split = sample["metadata"]["split"]
            by_split.setdefault(split, Counter())
            counts["windows"] += 1
            by_split[split]["windows"] += 1
            affected = False
            for horizon, frame in enumerate(sample["target"], start=1):
                side = detect_future_return_birth(sample, frame)
                for name in ("unsupported", "unresolved", "fixed_support_blocked"):
                    counts[name + "_components"] += side[name + "_count"]
                    by_split[split][name + "_components"] += side[name + "_count"]
                for row in side["unsupported"]:
                    unique_births.add((sample["metadata"]["sample_id"], row["flow_id"]))
                if side["unsupported_count"] or side["unresolved_count"]:
                    affected = True
                    if len(examples) < 12:
                        examples.append({"sample_id": sample["metadata"]["sample_id"], "horizon": horizon,
                                         "unsupported": side["unsupported_count"], "unresolved": side["unresolved_count"]})
            if affected:
                counts["affected_windows"] += 1
                by_split[split]["affected_windows"] += 1
    accepted = json.loads((root / "dataset_acceptance_receipt.json").read_text(encoding="utf-8"))
    result = {
        "schema_version": "PI-JWM-STEP-5.6A-Formal-Fixed-Support-Reaudit-v1",
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "counts": dict(counts), "unique_window_flow_births": len(unique_births),
        "by_split": {key: dict(value) for key, value in by_split.items()},
        "examples": examples,
        "previous_acceptance_accounting": accepted.get("unsupported_accounting"),
        "previous_acceptance_accounting_matches": accepted.get("unsupported_accounting", {}).get("unsupported") == counts["unsupported_components"],
        "dataset_packages_modified": False,
        "future_targets_model_input": False,
        "passed": counts["trajectories"] == 60 and counts["windows"] == 5520,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = audit(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": receipt["passed"], "counts": receipt["counts"], "previous_accounting_matches": receipt["previous_acceptance_accounting_matches"]}))
    if not receipt["passed"]:
        raise SystemExit(1)
