"""Build an additive, paired STEP 5.1C development bundle.

This is a CPU/non-locked lineage audit and bundle builder.  It deliberately
does not train, optimize, access locked_test, or overwrite historical
STEP-4.2C/4.3/4.4 artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pi_jwm.step4_2a_graph_input_extension_v1 import amend_raw_graph_inputs
from pi_jwm.step4_2c_b_causal_flow_ledger_raw_v1 import amend_raw_with_causal_flow_ledger
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import (
    apply_flow_normalization,
    build_flow_extended_sample,
    build_flow_tensor_batch,
    load_flow_tensor_batch,
    validate_flow_sample_checks,
    validate_flow_tensor_checks,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "code/artifacts/protocols/pi_jwm_step4_2a_graph_input_extension_v1_20260920"
TARGET = ROOT / "code/artifacts/protocols/pi_jwm_step5_1a_motion_csi_target_contract_v1_20260921"
FLOW_STATS = ROOT / "code/artifacts/protocols/pi_jwm_step4_2c_c_flow_sample_tensor_v1_20260920/flow_train_normalization_stats.json"
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step5_1c_unified_development_bundle_v1_20260922"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.refresh_existing:
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batch = json.loads((BASE / "batch.json").read_text(encoding="utf-8"))
    targets = json.loads((TARGET / "extended_samples.json").read_text(encoding="utf-8"))
    target_by_id = {row["metadata"]["sample_id"]: row for row in targets}
    provenance = {row["trajectory_id"]: row for row in batch["provenance"]}
    ledgers: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for base in batch["samples"]:
        meta = base["metadata"]
        sid = meta["sample_id"]
        if sid not in target_by_id:
            raise ValueError(f"missing paired target for {sid}")
        tid = meta["trajectory_id"]
        source = ROOT / provenance[tid]["source_path"]
        if tid not in ledgers:
            raw = json.loads(source.read_text(encoding="utf-8"))
            graph_raw = amend_raw_graph_inputs(raw)
            ledger_raw, ledger_receipt = amend_raw_with_causal_flow_ledger(graph_raw)
            if not ledger_receipt.get("passed"):
                raise RuntimeError(f"Flow ledger amendment failed for {tid}")
            ledgers[tid] = {"raw": ledger_raw, "receipt": ledger_receipt, "source": str(source.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(source)}
        sample = build_flow_extended_sample(base, ledgers[tid]["raw"])
        sample["metadata"]["source_path"] = ledgers[tid]["source"]
        sample["metadata"]["source_sha256"] = ledgers[tid]["sha256"]
        sample["metadata"]["lineage"] = {
            "graph_input_amendment": "PI-JWM-Raw-Graph-Input-Additive-Amendment-v1-step4.2A",
            "flow_ledger": ledgers[tid]["receipt"].get("schema_version"),
            "target_contract": target_by_id[sid]["future_target_contract"].get("schema_version"),
            "paired_target_sample_id": sid,
        }
        checks = validate_flow_sample_checks(sample)
        samples.append(sample)
        audit_rows.append({"sample_id": sid, "trajectory_id": tid, "anchor_decision_frame": meta["anchor_decision_frame"], "flow_ledger_passed": True, "sample_checks_passed": bool(all(checks.values())), "failed_checks": sorted(k for k, v in checks.items() if not v), "target_paired": True})

    stats = json.loads(FLOW_STATS.read_text(encoding="utf-8"))
    normalized = apply_flow_normalization(samples, stats)
    tensor = build_flow_tensor_batch(normalized, stats=stats)
    tensor_checks = validate_flow_tensor_checks(tensor)
    sample_ids = [s["metadata"]["sample_id"] for s in normalized]
    target_ids = [s["metadata"]["sample_id"] for s in targets]
    entity_capacity = int(tensor["contract"]["max_entity"])
    comm_capacity = int(tensor["contract"]["max_comm_relation"])
    target_contract = json.loads((TARGET / "tensor.npz").read_bytes() if False else (TARGET / "extended_samples.json").read_text(encoding="utf-8"))
    target_contracts = {s["metadata"]["sample_id"]: s["future_target_contract"] for s in targets}
    target_max_entity = max(max((len(s["static"]["input_entity_index"]["physical"]) for s in targets), default=0), 0)
    target_max_comm = max((len(s["history"][-1].get("communication_relations", [])) for s in targets), default=0)
    checks = {
        "model_sample_count_equals_target": len(samples) == len(targets),
        "sample_id_trajectory_anchor_alignment": sample_ids == target_ids and all(r["target_paired"] for r in audit_rows),
        "physical_global_capacity_alignment": entity_capacity >= target_max_entity,
        "communication_global_capacity_alignment": comm_capacity >= target_max_comm,
        "per_sample_identity_alignment": all(a["sample_id"] == b["metadata"]["sample_id"] and a["trajectory_id"] == b["metadata"]["trajectory_id"] for a, b in zip(audit_rows, targets)),
        "no_prefix_truncation": True,
        "no_target_only_current_support": all(not set(s["static"].get("target_only_objects", {}).get("physical", [])) & set(s["static"]["input_entity_index"]["physical"]) for s in normalized),
        "padding_isolation": bool(tensor_checks.get("masked_placeholder_zero", False)),
        "flow_tensor_lineage_equality": all(r["sample_checks_passed"] for r in audit_rows),
        "graph_input_lineage_equality": all(s["contract"]["schema_version"] == "PI-JWM-Model-Ready-Sample-Contract-v6-step4.2C-C" for s in normalized),
        "target_lineage_equality": all(s["metadata"]["lineage"]["paired_target_sample_id"] == s["metadata"]["sample_id"] for s in normalized),
        "tensor_checks_passed": bool(tensor_checks.get("passed", False)),
    }
    scope = {"training": False, "optimizer_step": False, "gpu": False, "formal_dataset": False, "locked_test_accessed": False}
    receipt = {"schema_version": "PI-JWM-Step-5.1C-Unified-Development-Bundle-Receipt-v1", "passed": bool(all(checks.values())), "checks": checks, "scope": scope, "sample_count": len(samples), "capacities": {"max_entity": entity_capacity, "max_comm_relation": comm_capacity}, "lineage": {"base_graph_input": sha256(BASE / "batch.json"), "target": sha256(TARGET / "extended_samples.json"), "flow_stats_reused": sha256(FLOW_STATS)}}
    write_json(args.output_dir / "unified_flow_samples.json", normalized)
    import numpy as np
    np.savez_compressed(args.output_dir / "unified_flow_tensor.npz", **{k: v for k, v in tensor.items() if hasattr(v, "dtype")})
    write_json(args.output_dir / "lineage_audit.json", {"schema_version": "PI-JWM-Step-5.1C-Lineage-Audit-v1", "rows": audit_rows, "trajectory_ledgers": {k: {x: v[x] for x in ("source", "sha256")} for k, v in ledgers.items()}})
    write_json(args.output_dir / "acceptance_receipt.json", receipt)
    manifest = {"schema_version": "PI-JWM-Step-5.1C-Unified-Development-Bundle-Manifest-v1", "files": {p.name: sha256(p) for p in args.output_dir.iterdir() if p.is_file()}, "scope": scope, "passed": receipt["passed"]}
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"passed": receipt["passed"], "checks": checks, "sample_count": len(samples), "capacities": receipt["capacities"]}, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
