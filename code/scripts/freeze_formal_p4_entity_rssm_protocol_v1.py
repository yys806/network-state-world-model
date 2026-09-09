"""Freeze the post-CPU P4 entity-RSSM GPU protocol and evidence bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_protocol(
    *, tensor_root: str | Path, consistency_audit_dir: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    tensor_root = Path(tensor_root).resolve()
    consistency_audit_dir = Path(consistency_audit_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if any("locked_test" in str(path).lower() for path in (tensor_root, consistency_audit_dir, output_dir)):
        raise ValueError("locked_test path is forbidden")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    tensor_validation = _read_json(tensor_root / "validation_report.json")
    audit_path = consistency_audit_dir / "entity_aligned_rssm_consistency_audit.json"
    audit = _read_json(audit_path)
    if tensor_validation.get("formal_tensor_ready") is not True:
        raise ValueError("causal-motion tensor is not ready")
    if audit.get("status") != "ready_for_gpu_execution_sentinel":
        raise ValueError("CPU consistency audit is not ready")
    tensor_manifest_sha256 = _sha256(tensor_root / "manifest.json")
    if audit["evidence"]["tensor_manifest_sha256"] != tensor_manifest_sha256:
        raise ValueError("CPU audit and tensor manifest do not match")
    source_paths = [
        PROJECT_ROOT / "code/src/pi_jwm/formal_motion_state_v1.py",
        PROJECT_ROOT / "code/src/pi_jwm/formal_dual_graph_world_model_v1.py",
        PROJECT_ROOT / "code/src/pi_jwm/formal_entity_aligned_rssm_world_model_v1.py",
        PROJECT_ROOT / "code/src/pi_jwm/formal_world_model_loss_v1.py",
        PROJECT_ROOT / "code/src/pi_jwm/formal_p4_gate_v1.py",
        PROJECT_ROOT / "code/scripts/run_formal_dual_graph_gpu_train_v1.py",
        PROJECT_ROOT / "code/scripts/run_formal_p4_entity_rssm_gpu_batch_probe_v1.py",
        PROJECT_ROOT / "code/scripts/run_formal_p4_entity_rssm_gpu_v1.py",
        PROJECT_ROOT / "code/scripts/run_formal_entity_aligned_rssm_consistency_audit_v1.py",
    ]
    source_hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in source_paths
    }
    report = {
        "schema_version": "PI-JWM-P4-entity-aligned-RSSM-frozen-protocol-v1",
        "status": "ready_for_gpu_batch_probe",
        "method": "entity_aligned_dual_graph_rssm_v1",
        "model_version": "formal_entity_aligned_rssm_v1",
        "latent_dynamics": "entity_aligned_complete_rssm_prior_posterior_v1",
        "tensor": {
            "root": str(tensor_root),
            "manifest_sha256": tensor_manifest_sha256,
            "history_steps": 8,
            "horizon_steps": 20,
            "counts": {"train": 9828, "validation": 3276, "calibration": 1638},
            "node_motion_contract": "causal_backward_difference_v1",
        },
        "cpu_gate": {
            "audit_path": str(audit_path),
            "audit_sha256": _sha256(audit_path),
            "status": audit["status"],
        },
        "model": {
            "hidden_dim": 32,
            "stochastic_dim": 16,
            "entity_latent_layout": "node_physical_edge_flow_task_v1",
            "training_posterior_teacher": True,
            "deployment_prior_only": True,
            "deterministic_rule_layer": True,
            "zero_init_residual_heads": True,
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-5,
            "gradient_clip": 5.0,
            "base_epochs": 20,
            "rssm_max_epochs": 40,
            "total_max_epochs": 60,
            "rssm_min_epochs": 20,
            "rssm_patience": 10,
            "base_frozen_during_rssm": True,
            "kl_balance": 0.8,
            "kl_weight": 0.1,
            "overshooting_distance": 5,
            "overshooting_weight": 0.1,
            "teacher_reconstruction_weight": 0.5,
        },
        "checkpoint_selection": {
            "contract": "p4_gate_aware_v1",
            "order": [
                "failed_hard_gate_count",
                "maximum_normalized_gate_excess",
                "validation_state_nll",
            ],
            "save_every_base_epoch": True,
            "save_every_rssm_epoch": True,
        },
        "gpu_batch_probe": {
            "candidate_order": [8, 4, 2, 1],
            "maximum_memory_fraction": 0.85,
            "optimizer_step_performed": False,
        },
        "gpu_execution_sentinel": {
            "seed": 20260831,
            "counts": {"train": 256, "validation": 128, "calibration": 128},
            "total_epochs": 2,
            "performance_numbers_are_observational_only": True,
        },
        "formal_seed_order": [20260831, 20260830, 20260832],
        "followup_seed_condition": "seed 20260831 best checkpoint passes every single-seed numeric P4 gate",
        "source_hashes": source_hashes,
        "boundaries": {
            "locked_test_accessed": False,
            "locked_test_allowed": False,
            "formal_performance_claim_ready": False,
            "P4_nonlocked_accuracy_gate": "pending_gpu_execution_sentinel_and_formal_three_seed_runs",
            "P6_open": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "PI-JWM-P4-entity-aligned-RSSM-frozen-protocol-manifest-v1",
        "files": {
            "protocol.json": {
                "size_bytes": protocol_path.stat().st_size,
                "sha256": _sha256(protocol_path),
            }
        },
        "locked_test_accessed": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--consistency-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = freeze_protocol(
        tensor_root=args.tensor_root,
        consistency_audit_dir=args.consistency_audit_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
