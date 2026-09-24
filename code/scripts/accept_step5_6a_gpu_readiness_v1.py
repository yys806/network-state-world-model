"""Combine STEP 5.6A CUDA receipts without starting formal training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FormalTrajectorySampler


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def accept(manifest_path: Path, audit_dir: Path, config_path: Path) -> dict:
    interface = FormalTrainingInterface.from_manifest(manifest_path)
    smoke = _read(audit_dir / "gpu_smoke_receipt.json")
    validation = _read(audit_dir / "full_gpu_validation_receipt.json")
    reload_probe = _read(audit_dir / "checkpoint_forward_identity_receipt.json")
    remote_dataset = _read(audit_dir / "remote_dataset_identity_receipt.json")
    config = _read(config_path)
    patch_readiness = _read(audit_dir.parent / "pi_jwm_step5_5_patch_20260923" / "readiness_receipt.json")
    dataset_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    sampler = FormalTrajectorySampler(interface.samples, interface.train_indices, seed=5601)
    epoch = sampler.order_for_epoch(0)
    next_epoch = sampler.order_for_epoch(1)
    sampler_verified = (
        len(epoch) == len(set(epoch)) == 4416
        and set(epoch) == set(interface.train_indices)
        and set(epoch).isdisjoint(interface.validation_indices)
        and epoch != next_epoch
        and sampler.batch_for_global_step(0, 8) == list(epoch[:8])
        and sampler.batch_for_global_step(552, 8) == list(next_epoch[:8])
        and sampler.state_dict()["train_trajectories"] == 48
    )
    identity_match = (
        dataset_hash == interface.dataset_manifest_hash
        and all(row.get("dataset_manifest_sha256") == dataset_hash
                for row in (smoke, validation, reload_probe, remote_dataset))
        and remote_dataset.get("all_present_and_matching") is True
        and remote_dataset.get("package_hashes") == interface.manifest["hashes"]
        and patch_readiness.get("passed") is True
        and patch_readiness.get("dataset_manifest_hash") == dataset_hash
    )
    scope = {
        "formal_training": False,
        "locked_test_accessed": False,
        "baseline": False,
        "planner": False,
        "performance_claim": False,
    }
    checks = {
        "GPU_AVAILABLE": smoke.get("environment", {}).get("gpu") is not None,
        "CUDA_FORWARD": all(row.get("success") and row.get("horizon") == 4 for row in smoke.get("batch_probe", [])),
        "CUDA_BACKWARD": all(row.get("gradients_finite") for row in smoke.get("batch_probe", [])),
        "CUDA_OPTIMIZER_STEP": all(row.get("parameter_updated") for row in smoke.get("batch_probe", [])),
        "CUDA_CHECKPOINT_RELOAD": smoke.get("checkpoint_reload_identical") is True and reload_probe.get("passed") is True,
        "H4_GPU_RUNTIME": smoke.get("passed") is True,
        "FULL_1104_GPU_VALIDATION": (
            validation.get("passed") is True and validation.get("validation_windows") == 1104
            and validation.get("validation_trajectories") == 12 and validation.get("prior_only") is True
            and validation.get("parameter_unchanged") is True
            and math.isfinite(validation.get("L_Val", float("nan")))
        ),
        "SAMPLER_FORMAL_PATH": sampler_verified,
        "DATASET_IDENTITY": identity_match,
    }
    if not all(checks.values()):
        raise ValueError(f"STEP 5.6A acceptance failed: {[k for k, value in checks.items() if not value]}")
    if any(receipt.get("scope", {}).get("formal_training") is not False
           or receipt.get("scope", {}).get("locked_test_accessed") is not False
           for receipt in (smoke, validation, reload_probe)):
        raise ValueError("STEP 5.6A scope receipt is incompatible")
    if config.get("formal_training_config") != "AWAITING_RESEARCHER_DECISION":
        raise ValueError("formal training config decision state is unexpected")
    return {
        "schema_version": "PI-JWM-STEP-5.6A-GPU-Readiness-v1",
        "dataset_manifest_sha256": dataset_hash,
        "checks": {key: "MATCH" if key == "DATASET_IDENTITY" else "VERIFIED" if key == "SAMPLER_FORMAL_PATH" else True
                   for key in checks},
        "formal_training_config": config["formal_training_config"],
        "formal_training_readiness": "BLOCKED_BY_CONFIG_DECISION",
        "formal_dataset_artifact": patch_readiness["formal_dataset_artifact"],
        "formal_dataset_readiness": patch_readiness["formal_dataset_readiness"],
        "full_formal_dataset_loader": patch_readiness["full_formal_dataset_loader"],
        "training_stack_readiness": patch_readiness["training_stack_readiness"],
        "gpu_codepath_readiness": "VERIFIED",
        "gpu_training_verified": False,
        "scope": scope,
        "passed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = accept(args.manifest, args.audit_dir, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": result["checks"]}))
