"""Capture or compare full Formal Dataset v1 package identities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))

from pi_jwm.step5_4_formal_training_readiness_v1 import sha256_file, sha256_path


def package_hashes(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes: dict[str, str] = {}
    for name in ("samples", "tensor", "graph", "target", "normalization"):
        value = manifest["packages"][name]
        path = Path(value)
        path = path if path.is_absolute() else (manifest_path.parent / path).resolve()
        hashes[name] = sha256_path(path)
    return hashes


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--dataset-receipt", type=Path)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    actual = package_hashes(args.manifest.resolve())
    dataset_root = args.manifest.resolve().parent
    source_identity = {
        "raw_trajectory_provenance_sha256": sha256_file(dataset_root / "raw_trajectory_provenance.json"),
        "split_manifest_sha256": sha256_file(dataset_root / "split_manifest.json"),
    }
    baseline_path = args.baseline.resolve()
    if args.capture:
        write_json(baseline_path, {"schema_version": "PI-JWM-Step-5.5-Deterministic-Build-Baseline-v1", "package_hashes": actual, **source_identity})
        print(json.dumps({"captured": True, "package_hashes": actual}, sort_keys=True))
        return 0
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected = baseline["package_hashes"]
    checks = {name: expected.get(name) == actual.get(name) for name in actual}
    receipt = {
        "schema_version": "PI-JWM-Step-5.5-Deterministic-Rebuild-Receipt-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "first_build_package_hashes": expected,
        "second_build_package_hashes": actual,
        "same_raw_sources": baseline.get("raw_trajectory_provenance_sha256") == source_identity["raw_trajectory_provenance_sha256"],
        "same_frozen_split": baseline.get("split_manifest_sha256") == source_identity["split_manifest_sha256"],
    }
    receipt["passed"] = bool(receipt["passed"] and receipt["same_raw_sources"] and receipt["same_frozen_split"])
    if args.receipt:
        write_json(args.receipt.resolve(), receipt)
    if args.dataset_receipt:
        dataset_receipt_path = args.dataset_receipt.resolve()
        dataset_receipt = json.loads(dataset_receipt_path.read_text(encoding="utf-8"))
        dataset_receipt.setdefault("checks", {})["deterministic_full_package_rebuild"] = receipt["passed"]
        dataset_receipt["deterministic_rebuild_receipt"] = receipt
        dataset_receipt["passed"] = all(dataset_receipt["checks"].values())
        write_json(dataset_receipt_path, dataset_receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
