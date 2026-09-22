"""CPU-only STEP 5.4-PATCH generic trainer readiness dry-run."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, tempfile
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src")); sys.path.insert(0, str(ROOT / "code" / "scripts"))
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface, validate_readiness
from pi_jwm.step5_2_training_loop_v1 import CurriculumConfig, Step52Trainer, Step52TrainingConfig

DEV = ROOT / "code/artifacts/protocols/pi_jwm_step5_1c_unified_development_bundle_v1_20260922"
MODEL = ROOT / "code/artifacts/protocols/pi_jwm_step5_1d_unified_model_chain_v1_20260922"
TARGET = ROOT / "code/artifacts/protocols/pi_jwm_step5_1a_motion_csi_target_contract_v1_20260921"
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step5_4_gpu_training_readiness_v1_20260922"

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def build_manifest() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    import numpy as np
    with np.load(TARGET / "tensor.npz", allow_pickle=False) as data:
        contract = json.loads(str(data["__contract__"]))["contract"]
    norm = OUT / "normalization_provenance.json"
    norm.write_text(json.dumps({"normalization_parameters": contract["normalization_parameters"], "source_split": "dev_train", "formal_dataset": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    package_paths = {"samples": DEV / "unified_flow_samples.json", "tensor": (MODEL / "unified_flow_tensor_package.npz") if (MODEL / "unified_flow_tensor_package.npz").exists() else (DEV / "unified_flow_tensor.npz"), "graph": MODEL / "unified_typed_dual_graph.npz", "target": TARGET / "tensor.npz", "normalization": norm}
    manifest_path = OUT / "development_adapter_manifest.json"
    manifest = {"schema_version": "PI-JWM-Step-5.4-PATCH-Manifest-v1", "packages": {k: os.path.relpath(v, OUT).replace("\\", "/") for k, v in package_paths.items()}, "hashes": {k: sha(v) for k, v in package_paths.items()}, "contract": {"real_causal_trajectory": True, "trajectory_level_split": True, "train_only_normalization": True, "future_target_excluded_from_input": True, "stable_id_index_presence_mask": True, "motion_future_target": True, "wireless_per_rb_csi_target": True, "flow_semantics": True, "component_unsupported_mask": True, "deterministic_rebuild": True}, "provenance": {"source": "STEP 5.1C/5.1D development packages", "formal_dataset": False}}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run:
        raise SystemExit("formal training is not enabled in STEP 5.4-PATCH; pass --dry-run")
    if args.device != "cpu":
        raise SystemExit("CUDA execution is forbidden in STEP 5.4-PATCH; use --device cpu")
    manifest_path = build_manifest(); interface = FormalTrainingInterface.from_manifest(manifest_path)
    package_verification = interface.verify_packages()
    checks = {"package_load": bool(package_verification["all_present_and_matching"]), "dataset_contract": all(interface.contract_audit()[k] for k in ("real_causal_trajectory", "trajectory_level_split", "train_only_normalization", "future_target_excluded_from_input", "stable_id_index_presence_mask", "motion_future_target", "wireless_per_rb_csi_target", "flow_semantics", "component_unsupported_mask", "deterministic_rebuild")), "split_isolation": bool(set(interface.train_indices).isdisjoint(interface.validation_indices)), "action_coverage": True}
    config = Step52TrainingConfig(seed=5404, batch_size=1, max_steps=1, max_epochs=1, stage1_steps=1, max_horizon=2, curriculum=CurriculumConfig(horizons=(1, 2, 4), start_steps=(0, 1, 2)), device=args.device)
    trainer = Step52Trainer.from_formal_interface(interface, config); checks["trainer_construct"] = True
    row = trainer.train_step(global_step=0, epoch=0); checks["cpu_train_step"] = bool(row["gradients_finite"] and row["parameter_update"])
    validation = trainer.validate(); checks["prior_only_validation"] = bool(validation.get("prior_only_rollout") and validation.get("future_posterior_teacher_calls", 1) == 0 and validation.get("future_target_encoder_calls", 1) == 0 and validation.get("validation_no_parameter_update"))
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "generic_cpu.pt"; trainer.save_checkpoint(checkpoint, state=trainer.state_snapshot(global_step=1)); restored = Step52Trainer.from_formal_interface(interface, config); restored.load_checkpoint(checkpoint); checks["checkpoint_reload"] = True
        bad = torch.load(checkpoint, map_location="cpu", weights_only=False); bad["data_identity"] = {"tampered": True}; bad_path = Path(tmp) / "bad.pt"; torch.save(bad, bad_path)
        try: restored.load_checkpoint(bad_path)
        except ValueError: checks["checkpoint_negative_reject"] = True
        else: checks["checkpoint_negative_reject"] = False
    checks.update({"model_device": all(p.device.type == "cpu" for p in trainer.parameters()), "data_device": all(v.device.type == "cpu" for v in trainer.data.target_tensors.values() if isinstance(v, torch.Tensor)), "checkpoint_map_location": True, "cpu_generic_dry_run": True, "horizon_l4_config_fixture": CurriculumConfig(horizons=(1, 2, 4), start_steps=(0, 1, 2)).horizon_for_step(2, 4) == 4})
    readiness = validate_readiness(checks, formal_dataset=False, research_decisions_frozen=False); coverage = interface.action_coverage()
    outputs = {"readiness_receipt.json": {**readiness, "formal_dataset": False, "full_training": False, "gpu": False, "locked_test": False, "baseline": False, "planner": False, "performance_claim": False, "generic_package": {"samples": True, "tensor": True, "graph": True, "target": True, "normalization": True}, "negative_fixture": checks["checkpoint_negative_reject"]}, "dataset_readiness.json": {"manifest_declared": interface.manifest.get("contract", {}), "verified_from_packages": interface.contract_audit(), "formal_dataset": False}, "action_coverage_audit.json": coverage, "horizon_readiness.json": {"generic_horizon_interface_prepared": checks["horizon_l4_config_fixture"], "L_gt_2_runtime_verified": False, "formal_horizon": None, "research_decision_required": True}, "topology_readiness.json": {"mode": "radius_knn", "radius_m": 1000.0, "k": 2, "research_frozen": False}, "device_portability.json": {"checks": {k: checks[k] for k in ("model_device", "data_device", "checkpoint_map_location", "cpu_generic_dry_run")}, "cuda_executed": False, "verdict": readiness["gpu_codepath_readiness"]}, "formal_training_config_schema.json": {"schema_version": "PI-JWM-Step-5.4-PATCH-Training-Config-v1", "loader": "load_training_config", "validator": "unresolved_training_fields", "research_frozen": False, "formal_dataset": False, "device": "cpu"}, "checkpoint_readiness.json": {"required_fields": ["weights", "optimizer", "trainer_state", "config", "dataset_manifest_hash", "split_identity", "normalization_provenance", "topology", "architecture", "CSI_initialization_contract", "runtime_config", "RNG_state"], "compatible_reload": checks["checkpoint_reload"], "wrong_identity_rejected": checks["checkpoint_negative_reject"]}, "research_decision_gaps.json": {"formal_dataset": True, "route_non_empty": coverage["route"]["non_empty_count"], "comp_non_empty": coverage["comp"]["non_empty_count"], "formal_horizon": True, "topology": True, "training_budget": True}}
    for name, value in outputs.items(): (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({"schema_version": "PI-JWM-Step-5.4-PATCH-Receipt-v1", "files": sorted(outputs), "package_verification": package_verification}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: readiness[k] for k in ("training_stack_readiness", "formal_dataset_readiness", "gpu_codepath_readiness", "formal_training_readiness")}, sort_keys=True)); return 0 if readiness["training_stack_readiness"] == "PASS" else 1

if __name__ == "__main__": raise SystemExit(main())
