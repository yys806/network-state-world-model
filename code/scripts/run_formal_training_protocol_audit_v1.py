from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_training_protocol_audit_v1 import (
    AGGREGATE_WINDOW_KEYS,
    PER_RB_WINDOW_KEYS,
    audit_training_protocol,
    build_training_protocol_freeze,
    collect_current_implementation_snapshot,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_audit(*, tensor_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    tensor_dir = Path(tensor_dir)
    output_dir = Path(output_dir)
    contract = _read_json(tensor_dir / "tensor_contract.json")
    validation = _read_json(tensor_dir / "validation_report.json")
    stats = _read_json(tensor_dir / "normalization_stats.json")
    window_rows = list(csv.DictReader((tensor_dir / "window_index.csv").open(encoding="utf-8-sig")))
    train_rows = [row for row in window_rows if row.get("split") == "train"]
    if not train_rows:
        raise ValueError("tensor directory has no train windows")
    seed = int(train_rows[0]["seed"])
    tensor_path = tensor_dir / f"seed_{seed:03d}" / "trajectory_tensors.npz"
    with np.load(tensor_path, allow_pickle=False) as arrays:
        tensor_keys = set(arrays.files)
    window_keys = set()
    if "link_activity" in tensor_keys:
        window_keys.add("link_activity_by_rb")
    if "link_activity_mask" in tensor_keys:
        window_keys.add("link_activity_mask_by_rb")
    if "link_rate_by_rb" in tensor_keys:
        window_keys.add("link_rate_by_rb")
    if "link_rate_by_rb_mask" in tensor_keys:
        window_keys.add("link_rate_by_rb_mask")
    if {"physical_edge_state", "physical_edge_feature_mask", "physical_edge_present"} <= tensor_keys:
        window_keys.update(AGGREGATE_WINDOW_KEYS)
    implementation = collect_current_implementation_snapshot()
    report = audit_training_protocol(
        contract_mode="aggregate_baseline",
        tensor_contract=contract,
        tensor_validation=validation,
        normalization_stats=stats,
        window_keys=window_keys,
        model_outputs=implementation["model_outputs"],
        loss_targets=implementation["loss_targets"],
        metric_sources=implementation["metric_sources"],
        locked_test_materialized=(tensor_dir / "locked_test").exists(),
    )
    report["source"] = {
        "tensor_dir": str(tensor_dir.resolve()),
        "train_window_count": len(train_rows),
        "tensor_seed_checked": seed,
        "tensor_key_count": len(tensor_keys),
    }
    _write_json(output_dir / "training_protocol_audit.json", report)
    _write_json(output_dir / "training_protocol_freeze.json", build_training_protocol_freeze(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit formal PI-JWM aggregate-baseline protocol without training.")
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(tensor_dir=args.tensor_dir, output_dir=args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["formal_training_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
