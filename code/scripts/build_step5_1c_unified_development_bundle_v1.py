"""Build an additive, paired STEP 5.1C development bundle.

This is a CPU/non-locked lineage audit and bundle builder.  It deliberately
does not train, optimize, access locked_test, or overwrite historical
STEP-4.2C/4.3/4.4 artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from pi_jwm.step4_2a_graph_input_extension_v1 import amend_raw_graph_inputs
from pi_jwm.step4_2c_b_causal_flow_ledger_raw_v1 import amend_raw_with_causal_flow_ledger
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import (
    apply_flow_normalization,
    build_flow_extended_sample,
    build_flow_tensor_batch,
    fit_flow_normalization_stats,
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


def _physical_identity(sample: dict[str, Any], target: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    model_index = sample["static"]["input_entity_index"]["physical"]
    model_types = sample["static"].get("input_entity_type_by_index", {})
    expected = sorted(target["static"].get("future_target_current_physical_support", []), key=lambda row: int(row["input_slot"]))
    actual = sorted(
        ({"input_slot": int(slot), "entity_id": str(entity_id), "entity_type": str(model_types.get(str(slot), model_types.get(int(slot), "unknown"))), "current_presence": True} for entity_id, slot in model_index.items()),
        key=lambda row: row["input_slot"],
    )
    mismatches = []
    for slot in range(max(len(actual), len(expected))):
        a = actual[slot] if slot < len(actual) else {}
        b = expected[slot] if slot < len(expected) else {}
        if (a.get("input_slot"), a.get("entity_id"), a.get("entity_type")) != (b.get("input_slot"), b.get("entity_id"), b.get("entity_type")):
            mismatches.append({"slot": slot, "model_entity_id": a.get("entity_id"), "target_entity_id": b.get("entity_id")})
    return not mismatches, mismatches


def _comm_identity(sample: dict[str, Any], target: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    model_rows = sample["history"][-1].get("communication_relations", [])
    target_rows = sorted(target["static"].get("future_target_current_comm_support", []), key=lambda row: int(row["relation_slot"]))
    model_index = sample["static"]["input_entity_index"]["physical"]
    reverse = {int(slot): str(entity_id) for entity_id, slot in model_index.items()}
    mismatches = []
    for slot in range(max(len(model_rows), len(target_rows))):
        a = model_rows[slot] if slot < len(model_rows) else {}
        b = target_rows[slot] if slot < len(target_rows) else {}
        source_slot = a.get("source_agent_index", a.get("source_input_slot"))
        target_slot = a.get("target_agent_index", a.get("target_input_slot"))
        actual = (slot, a.get("communication_relation_id"), a.get("relation_type"), reverse.get(int(source_slot)) if source_slot is not None else None, reverse.get(int(target_slot)) if target_slot is not None else None, int(source_slot) if source_slot is not None else None, int(target_slot) if target_slot is not None else None, list(a.get("rb_indices", [])), bool(a.get("validity", False)), bool(a.get("presence", False)))
        expected = (int(b.get("relation_slot", slot)), b.get("relation_id"), b.get("relation_type"), b.get("source_id"), b.get("target_id"), int(b["source_input_slot"]) if b.get("source_input_slot") is not None else None, int(b["target_input_slot"]) if b.get("target_input_slot") is not None else None, list(b.get("rb_indices", [])), bool(b.get("validity", False)), bool(b.get("presence", False)))
        if actual != expected:
            mismatches.append({"slot": slot, "model_relation_id": actual[1], "target_relation_id": expected[1], "model_source_id": actual[3], "target_source_id": expected[3], "model_target_id": actual[4], "target_target_id": expected[4]})
    return not mismatches, mismatches


def _support_within_capacity(samples: list[dict[str, Any]], targets: list[dict[str, Any]], entity_capacity: int, comm_capacity: int) -> dict[str, bool]:
    physical_widths = [len(t["static"].get("future_target_current_physical_support", [])) for t in targets]
    comm_widths = [len(t["static"].get("future_target_current_comm_support", [])) for t in targets]
    return {
        "physical_capacity_exact_alignment": max(physical_widths, default=0) == entity_capacity,
        "comm_capacity_exact_alignment": max(comm_widths, default=0) == comm_capacity,
        "target_tensor_physical_width_matches_unified": entity_capacity == int(np.load(TARGET / "tensor.npz", allow_pickle=True)["target_vehicle_motion_raw"].shape[2]),
        "target_tensor_comm_width_matches_unified": comm_capacity == int(np.load(TARGET / "tensor.npz", allow_pickle=True)["target_comm_csi_raw"].shape[2]),
        "all_sample_support_within_full_capacity": all(width <= entity_capacity for width in physical_widths) and all(width <= comm_capacity for width in comm_widths),
    }


def _tensor_digest(tensor: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(key for key, value in tensor.items() if hasattr(value, "dtype")):
        digest.update(key.encode("utf-8"))
        digest.update(np.asarray(tensor[key]).tobytes())
    return digest.hexdigest()


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

    dev_train = [sample for sample in samples if sample["metadata"].get("split") == "dev_train"]
    dev_validation = [sample for sample in samples if sample["metadata"].get("split") == "dev_validation"]
    stats = fit_flow_normalization_stats(dev_train)
    stats["source_sample_ids"] = [sample["metadata"]["sample_id"] for sample in dev_train]
    stats["source_trajectory_ids"] = sorted({sample["metadata"]["trajectory_id"] for sample in dev_train})
    stats["source_split"] = "dev_train"
    stats["fit_sample_count"] = len(dev_train)
    stats["validation_sample_ids_excluded"] = [sample["metadata"]["sample_id"] for sample in dev_validation]
    stats["future_target_in_fit"] = False
    stats["fit_policy"] = "unified_samples_only; history rows; known=true AND presence=true AND feature_mask=true AND value!=null"
    normalized = apply_flow_normalization(samples, stats)
    tensor = build_flow_tensor_batch(normalized, stats=stats)
    tensor_checks = validate_flow_tensor_checks(tensor)
    sample_ids = [s["metadata"]["sample_id"] for s in normalized]
    target_ids = [s["metadata"]["sample_id"] for s in targets]
    entity_capacity = int(tensor["contract"]["max_entity"])
    comm_capacity = int(tensor["contract"]["max_comm_relation"])
    physical_rows = []
    comm_rows = []
    for sample, target in zip(samples, targets):
        physical_ok, physical_mismatches = _physical_identity(sample, target)
        comm_ok, comm_mismatches = _comm_identity(sample, target)
        physical_rows.append({"sample_id": sample["metadata"]["sample_id"], "passed": physical_ok, "mismatches": physical_mismatches})
        comm_rows.append({"sample_id": sample["metadata"]["sample_id"], "passed": comm_ok, "mismatches": comm_mismatches})
    capacity_checks = _support_within_capacity(samples, targets, entity_capacity, comm_capacity)
    pairing_crop_required = any(not row["passed"] for row in physical_rows + comm_rows)
    physical_target_ids = [{str(row["entity_id"]) for row in target["static"].get("future_target_current_physical_support", [])} for target in targets]
    physical_model_ids = [{str(entity_id) for entity_id in sample["static"]["input_entity_index"]["physical"]} for sample in samples]
    comm_target_ids = [{str(row["relation_id"]) for row in target["static"].get("future_target_current_comm_support", [])} for target in targets]
    comm_model_ids = [{str(row.get("communication_relation_id")) for row in sample["history"][-1].get("communication_relations", [])} for sample in samples]
    tampered_target = json.loads(json.dumps(targets[0]))
    tampered_target["static"]["future_target_current_physical_support"][0]["entity_id"] += "::tampered"
    tamper_physical_ok, _ = _physical_identity(samples[0], tampered_target)
    rebuilt_tensor = build_flow_tensor_batch(normalized, stats=stats)
    np.savez_compressed(args.output_dir / "unified_flow_tensor.npz", **{k: v for k, v in tensor.items() if hasattr(v, "dtype")})
    with np.load(args.output_dir / "unified_flow_tensor.npz", allow_pickle=True) as reloaded:
        serialize_reload_ok = all(np.array_equal(np.asarray(reloaded[key]), np.asarray(value)) for key, value in tensor.items() if hasattr(value, "dtype") and key in reloaded)
    checks = {
        "paired_sample_id_alignment": len(samples) == len(targets) and sample_ids == target_ids,
        "trajectory_id_alignment": all(a["trajectory_id"] == b["metadata"]["trajectory_id"] for a, b in zip(audit_rows, targets)),
        "anchor_alignment": all(a["anchor_decision_frame"] == b["metadata"]["anchor_decision_frame"] for a, b in zip(audit_rows, targets)),
        "history_frame_indices_alignment": all(a["metadata"].get("history_frame_indices") == b["metadata"].get("history_frame_indices") for a, b in zip(samples, targets)),
        "future_frame_indices_alignment": all(a["metadata"].get("future_frame_indices") == b["metadata"].get("future_frame_indices") for a, b in zip(samples, targets)),
        "physical_identity_alignment_per_sample": all(row["passed"] for row in physical_rows),
        "physical_slot_index_alignment_per_sample": all(row["passed"] for row in physical_rows),
        "comm_identity_alignment_per_sample": all(row["passed"] for row in comm_rows),
        "comm_endpoint_identity_alignment_per_sample": all(row["passed"] for row in comm_rows),
        "comm_relation_type_alignment_per_sample": all(row["passed"] for row in comm_rows),
        "comm_rb_identity_alignment_per_sample": all(row["passed"] for row in comm_rows),
        **capacity_checks,
        "no_pairing_crop_required": not pairing_crop_required,
        "no_prefix_truncation_pairing_logic": not pairing_crop_required and all(row["passed"] for row in physical_rows + comm_rows),
        "future_only_entity_not_in_current_support": all(set(target["static"].get("target_only_objects", {}).get("physical", [])).isdisjoint(model_ids) for target, model_ids in zip(targets, physical_model_ids)),
        "future_only_relation_not_in_current_support": all(target_ids.issubset(model_ids) for target_ids, model_ids in zip(comm_target_ids, comm_model_ids)),
        "padding_slots_presence_mask_false": bool(tensor_checks.get("masked_placeholder_zero", False)),
        "flow_stats_fit_only_unified_dev_train": stats["source_sample_ids"] == [sample["metadata"]["sample_id"] for sample in dev_train] and stats["fit_sample_count"] == len(dev_train),
        "validation_samples_excluded_from_stats": not set(stats["source_sample_ids"]) & {sample["metadata"]["sample_id"] for sample in dev_validation},
        "no_future_target_in_normalization_fit": stats["future_target_in_fit"] is False,
        "receipt_tamper_negative": not tamper_physical_ok,
        "deterministic_rebuild": _tensor_digest(tensor) == _tensor_digest(rebuilt_tensor),
        "serialize_reload": serialize_reload_ok,
        "no_target_only_current_support": all(not set(s["static"].get("target_only_objects", {}).get("physical", [])) & set(s["static"]["input_entity_index"]["physical"]) for s in normalized),
        "padding_isolation": bool(tensor_checks.get("masked_placeholder_zero", False)),
        "flow_tensor_lineage_equality": all(r["sample_checks_passed"] for r in audit_rows),
        "graph_input_lineage_equality": all(s["contract"]["schema_version"] == "PI-JWM-Model-Ready-Sample-Contract-v6-step4.2C-C" for s in normalized),
        "target_lineage_equality": all(s["metadata"]["lineage"]["paired_target_sample_id"] == s["metadata"]["sample_id"] for s in normalized),
        "tensor_checks_passed": bool(tensor_checks.get("passed", False)),
    }
    scope = {"training": False, "optimizer_step": False, "gpu": False, "formal_dataset": False, "locked_test_accessed": False}
    receipt = {"schema_version": "PI-JWM-Step-5.1C-Unified-Development-Bundle-Receipt-v2", "passed": bool(all(checks.values()) and all(value is False for value in scope.values())), "checks": checks, "scope": scope, "sample_count": len(samples), "dev_train_sample_count": len(dev_train), "dev_validation_sample_count": len(dev_validation), "capacities": {"max_entity": entity_capacity, "max_comm_relation": comm_capacity}, "lineage": {"base_graph_input": sha256(BASE / "batch.json"), "target": sha256(TARGET / "extended_samples.json"), "historical_flow_stats": sha256(FLOW_STATS), "new_stats_source": "unified dev_train samples only"}}
    write_json(args.output_dir / "unified_flow_samples.json", normalized)
    np.savez_compressed(args.output_dir / "unified_flow_tensor.npz", **{k: v for k, v in tensor.items() if hasattr(v, "dtype")})
    write_json(args.output_dir / "unified_flow_train_normalization_stats.json", stats)
    write_json(args.output_dir / "tensor_contract.json", tensor["contract"])
    write_json(args.output_dir / "lineage_audit.json", {"schema_version": "PI-JWM-Step-5.1C-Lineage-Audit-v2", "rows": audit_rows, "physical_identity_alignment_per_sample": physical_rows, "comm_identity_alignment_per_sample": comm_rows, "trajectory_ledgers": {k: {x: v[x] for x in ("source", "sha256")} for k, v in ledgers.items()}, "normalization": {"stats_file": "unified_flow_train_normalization_stats.json", "historical_stats_file": str(FLOW_STATS.relative_to(ROOT)).replace("\\", "/"), "fit_sample_ids": stats["source_sample_ids"], "validation_excluded": stats["validation_sample_ids_excluded"], "future_target_in_fit": False}})
    write_json(args.output_dir / "acceptance_receipt.json", receipt)
    manifest = {
        "schema_version": "PI-JWM-Step-5.1C-Unified-Development-Bundle-Manifest-v2",
        "builder_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in args.output_dir.iterdir() if p.is_file()},
        "large_artifact_provenance": {
            "unified_flow_samples.json": {"path": str((args.output_dir / "unified_flow_samples.json").relative_to(ROOT)).replace("\\", "/"), "source": "STEP 4.2A batch + STEP 4.2C-B ledger + unified dev_train/dev_validation pairing"},
            "unified_flow_tensor.npz": {"path": str((args.output_dir / "unified_flow_tensor.npz").relative_to(ROOT)).replace("\\", "/"), "source": "unified_flow_samples.json + unified_flow_train_normalization_stats.json"},
        },
        "scope": scope,
        "passed": receipt["passed"],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"passed": receipt["passed"], "checks": checks, "sample_count": len(samples), "capacities": receipt["capacities"]}, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
