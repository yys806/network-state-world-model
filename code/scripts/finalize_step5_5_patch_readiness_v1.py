"""Combine independent CPU/full-shard, structure and lifecycle receipts."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1"
PATCH = ROOT / "code/artifacts/audit/pi_jwm_step5_5_patch_20260923"
OLD_GPU = ROOT / "code/artifacts/protocols/pi_jwm_step5_4_gpu_training_readiness_v1_20260922/readiness_receipt.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    original = _load(DATASET / "dataset_acceptance_receipt.json")
    audit = _load(PATCH / "audit_receipt.json")
    cpu = _load(PATCH / "cpu_acceptance_receipt.json")
    traversal = _load(PATCH / "full_shard_traversal_receipt.json")
    gpu = _load(OLD_GPU)
    same_identity = len({audit["dataset_manifest_hash"], cpu["dataset_manifest_hash"], traversal["dataset_manifest_hash"]}) == 1
    checks = {
        "formal_dataset_artifact_ready": original["passed"] and original["counts"]["total_windows"] == 5520,
        "full_shard_index_and_payloads": traversal["passed"] and traversal["all_payload_hashes_verified"] and traversal["unique_windows"] == 5520,
        "h4_cpu_full_shard_batch": cpu["passed"] and all(cpu["checks"].values()),
        "training_stack_prior_and_patch": gpu["training_stack_readiness"] == "PASS" and cpu["passed"],
        "future_fixed_support_accounting": audit["passed"] and audit["future_fixed_support"]["unsupported"] > 0 and audit["future_fixed_support"]["fixed_support_blocked"] == audit["future_fixed_support"]["unsupported"] and audit["future_fixed_support"]["previous_field_only_receipt_counts_superseded"],
        "lifecycle_repair_audit": audit["lifecycle_repair"]["repair_semantics_verified"],
        "same_dataset_identity": same_identity,
        "gpu_prepared_prior_static_audit": gpu["gpu_codepath_readiness"] == "PREPARED" and not gpu["gpu"],
    }
    receipt = {
        "schema_version": "PI-JWM-Step-5.5-PATCH-Readiness-v1", "passed": all(checks.values()), "checks": checks,
        "dataset_manifest_hash": audit["dataset_manifest_hash"],
        "formal_dataset_artifact": "READY" if checks["formal_dataset_artifact_ready"] else "NOT_READY",
        "full_formal_dataset_loader": "VERIFIED" if checks["full_shard_index_and_payloads"] else "NOT_VERIFIED",
        "h4_full_data_consumption_path": "VERIFIED" if checks["h4_cpu_full_shard_batch"] else "NOT_VERIFIED",
        "future_fixed_support_accounting": "VERIFIED" if checks["future_fixed_support_accounting"] else "NOT_VERIFIED",
        "lifecycle_repair_audit": "PASS" if checks["lifecycle_repair_audit"] else "FAIL",
        "training_stack_readiness": "PASS" if checks["training_stack_prior_and_patch"] else "FAIL",
        "formal_dataset_readiness": "READY" if all(checks[key] for key in ("formal_dataset_artifact_ready", "full_shard_index_and_payloads", "future_fixed_support_accounting", "lifecycle_repair_audit", "same_dataset_identity")) else "NOT_READY",
        "gpu_codepath_readiness": "PREPARED" if checks["gpu_prepared_prior_static_audit"] else "NOT_PREPARED",
        "gpu_training_verified": False, "formal_training": False, "locked_test_accessed": False,
        "baseline": False, "planner": False, "performance_claim": False,
        "evidence": {"dataset": "dataset_acceptance_receipt.json", "audit": "audit_receipt.json", "cpu": "cpu_acceptance_receipt.json", "traversal": "full_shard_traversal_receipt.json", "gpu_static_prior": str(OLD_GPU.relative_to(ROOT)).replace("\\", "/")},
    }
    output = PATCH / "readiness_receipt.json"
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": receipt["passed"], "checks": checks, "formal_dataset_readiness": receipt["formal_dataset_readiness"]}, sort_keys=True))
    if not receipt["passed"]:
        raise SystemExit(1)
