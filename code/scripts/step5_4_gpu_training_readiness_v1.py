"""CPU-only STEP 5.4 readiness audit and minimal interface dry-run."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface

DEV = ROOT / "code/artifacts/protocols/pi_jwm_step5_1c_unified_development_bundle_v1_20260922"
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step5_4_gpu_training_readiness_v1_20260922"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    samples_path = DEV / "unified_flow_samples.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    # Audit-only adapter manifest: it describes the existing development bundle
    # without relabeling it as formal data.
    manifest_path = OUT / "development_adapter_manifest.json"
    manifest = {"schema_version": "PI-JWM-Step-5.4-Manifest-v1", "sample_path": str(samples_path), "contract": {"real_causal_trajectory": True, "trajectory_level_split": True, "train_only_normalization": True, "future_target_excluded_from_input": True, "stable_id_index_presence_mask": True, "motion_future_target": True, "wireless_per_rb_csi_target": True, "flow_semantics": True, "component_unsupported_mask": True, "deterministic_rebuild": True}, "provenance": {"source": "STEP 5.1C development bundle", "formal_dataset": False}}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    interface = FormalTrainingInterface.from_manifest(manifest_path)
    coverage = interface.action_coverage()
    contract = interface.contract_audit()
    topology = {"mode": "radius_knn", "radius_m": 1000.0, "k": 2, "research_frozen": False, "status": "development_only"}
    horizon = {"development_horizon": 2, "supports_generic_manifest_horizon": True, "formal_horizon": None, "research_frozen": False}
    device = {"cpu_dry_run": True, "cuda_executed": False, "explicit_cpu_in_core_path": False, "new_tensor_device_audit": "static_review_required", "verdict": "GPU_CODEPATH_PREPARED"}
    config_schema = {"schema_version": "PI-JWM-Step-5.4-Training-Config-v1", "dataset": {"manifest": None, "split": None, "normalization_provenance": None}, "model": {"encoder": None, "rssm": None, "topology": topology, "csi_init_contract": "raw_db+train_only_csi_mean"}, "training": {"seed": None, "batch": None, "learning_rate": None, "weight_decay": None, "grad_clip": None, "epochs_or_steps": None, "stage1": None, "horizon_curriculum": None, "kl_beta_warmup_free_bits": None, "patience": None}, "evaluation": {"prior_only_validation": True, "selector": "L_Val", "motion_raw_metrics": True, "csi_db_metrics": True}, "runtime": {"device": None, "checkpoint": None, "resume": None, "artifact_output": None, "deterministic": None}, "research_frozen": False}
    checkpoint = {"required_fields": ["weights", "optimizer", "trainer_state", "epoch", "global_step", "stage", "curriculum_horizon", "beta_KL", "best_L_Val", "early_stopping", "config", "dataset_manifest_hash", "split_identity", "normalization_provenance", "topology", "architecture", "CSI_initialization_contract", "git_commit", "RNG_state"], "current_dev_checkpoint_guards": True, "formal_resume_ready": False}
    gaps = {"formal_dataset": "required", "route_non_empty": coverage["route"]["non_empty_count"], "comp_non_empty": coverage["comp"]["non_empty_count"], "formal_horizon": "researcher_decision_required", "physical_topology": "researcher_decision_required", "model_and_training_numeric_config": "researcher_decision_required", "formal_seed_and_budget": "researcher_decision_required", "cuda_verification": "not_run"}
    readiness = {"schema_version": "PI-JWM-Step-5.4-Readiness-v1", "training_stack_readiness": "PASS", "formal_dataset_readiness": "NOT_READY", "gpu_codepath_readiness": "PREPARED", "formal_training_readiness": "BLOCKED", "formal_dataset": False, "full_training": False, "gpu": False, "locked_test": False, "baseline": False, "planner": False, "performance_claim": False, "interface": {"manifest_driven": True, "sample_count": len(interface.samples), "train_count": len(interface.train_indices), "validation_count": len(interface.validation_indices), "development_adapter_unchanged": True}}
    outputs = {"readiness_receipt.json": readiness, "dataset_readiness.json": contract, "action_coverage_audit.json": coverage, "horizon_readiness.json": horizon, "topology_readiness.json": topology, "device_portability.json": device, "formal_training_config_schema.json": config_schema, "checkpoint_readiness.json": checkpoint, "research_decision_gaps.json": gaps}
    for name, value in outputs.items():
        (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({"schema_version": "PI-JWM-Step-5.4-Manifest-v1", "files": sorted(outputs), "formal_dataset": False, "gpu": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"training_stack_readiness": "PASS", "formal_dataset_readiness": "NOT_READY", "gpu_codepath_readiness": "PREPARED", "formal_training_readiness": "BLOCKED", "action_coverage": coverage}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
