"""Bounded CPU acceptance for the full-shard Trainer path; no formal training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from pi_jwm.step5_2_training_loop_v1 import CurriculumConfig, Step52TrainingConfig
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalTrainer


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1"
OUTPUT = ROOT / "code/artifacts/audit/pi_jwm_step5_5_patch_20260923/cpu_acceptance_receipt.json"


def run(manifest: Path) -> dict:
    interface = FormalTrainingInterface.from_manifest(manifest)
    config = Step52TrainingConfig(
        seed=5505, batch_size=2, max_horizon=4, max_steps=1, max_epochs=1,
        stage1_steps=0, curriculum=CurriculumConfig(horizons=(4,), start_steps=(0,)), device="cpu",
    )
    trainer = FullFormalTrainer.from_interface(interface, config)
    train_indices = trainer.data.train_indices
    validation_indices = trainer.data.validation_indices
    cross_train = [train_indices[0], train_indices[92]]
    cross_validation = [validation_indices[0], validation_indices[92]]
    trainer.eval()
    import torch
    with torch.no_grad():
        train_forward = trainer._run_batch(cross_train, 4, stage="prior_dominant_recursive", beta_kl=0.0, training=False)
    train_step = trainer.train_step(global_step=0, epoch=0)
    validation = trainer.validate_indices(cross_validation)
    original_validation_indices = trainer.data.validation_indices
    try:
        trainer.data.validation_indices = cross_validation
        aggregate_validation = trainer.validate()
    finally:
        trainer.data.validation_indices = original_validation_indices
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "step5_5_patch_cpu.pt"
        state = trainer.state_snapshot(global_step=1, epoch=0, curriculum_horizon=4)
        trainer.save_checkpoint(checkpoint, state=state)
        loaded = trainer.load_checkpoint(checkpoint)
        original_identity = trainer.data.identity
        trainer.data.identity = {**original_identity, "dataset_id": "wrong-dataset"}
        try:
            try:
                trainer.load_checkpoint(checkpoint)
            except ValueError as error:
                wrong_identity_rejected = "data identity" in str(error)
            else:
                wrong_identity_rejected = False
        finally:
            trainer.data.identity = original_identity
    checks = {
        "full_index_4416_1104": len(train_indices) == 4416 and len(validation_indices) == 1104,
        "cross_train_trajectory_batch": len({trainer.shards.index[i]["metadata"]["trajectory_id"] for i in cross_train}) == 2,
        "cross_validation_trajectory_batch": len({trainer.shards.index[i]["metadata"]["trajectory_id"] for i in cross_validation}) == 2,
        "h4_cross_train_forward": len(train_forward["horizon_rows"]) == 4 and not train_forward["posterior_teacher_used"],
        "h4_optimizer_step": bool(train_step["loss_finite"] and train_step["parameter_update"]["any_changed"] and train_step["rollout_horizon"] == 4),
        "h4_prior_only_validation": bool(validation["prior_only"] and len(validation["horizon_rows"]) == 4),
        "batch_validation_aggregation": bool(aggregate_validation["l_val_finite"] and aggregate_validation["validation_no_parameter_update"] and aggregate_validation["future_posterior_teacher_calls"] == 0),
        "checkpoint_reload": loaded == state,
        "wrong_dataset_identity_rejected": wrong_identity_rejected,
        "bounded_shard_memory": trainer.shards.peak_loaded_shards <= 2,
        "train_only_encoder_stats": trainer.encoder.normalization_stats["source_split"] == "dev_train" and trainer.encoder.normalization_stats["fit_trajectory_count"] == 48,
    }
    return {
        "schema_version": "PI-JWM-Step-5.5-PATCH-CPU-Acceptance-v1", "passed": all(checks.values()), "checks": checks,
        "dataset_manifest_hash": interface.dataset_manifest_hash,
        "sample_counts": {"train": len(train_indices), "validation": len(validation_indices)},
        "cross_train_sample_ids": [trainer.shards.index[i]["metadata"]["sample_id"] for i in cross_train],
        "cross_validation_sample_ids": validation["sample_ids"],
        "max_simultaneously_loaded_trajectory_shards": trainer.shards.peak_loaded_shards,
        "h4_train_step": {"loss_finite": train_step["loss_finite"], "parameter_update": train_step["parameter_update"], "rollout_horizon": train_step["rollout_horizon"]},
        "h4_validation": {"prior_only": validation["prior_only"], "horizon_count": len(validation["horizon_rows"])},
        "scope": {"cpu": True, "gpu": False, "formal_training": False, "locked_test_accessed": False, "baseline": False, "planner": False, "performance_claim": False},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DATASET / "formal_dataset_manifest.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    receipt = run(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": receipt["passed"], "checks": receipt["checks"]}, sort_keys=True))
