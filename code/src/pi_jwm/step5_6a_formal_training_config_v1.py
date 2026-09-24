"""Researcher-approved Formal Dataset v1 training numbers; no runner is started here."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from pi_jwm.step5_2_training_loop_v1 import CurriculumConfig, KLSchedule, Step52TrainingConfig
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface


DATASET_MANIFEST_SHA256 = "6392a08b31340812463be9e3f5f78891d39933c54d50c71cca1984b3bf9448bc"
DATASET_ID = "pi_jwm_formal_dataset_v1_h2_l4_20260923"


@dataclass(frozen=True)
class FormalTrainingConfigV1:
    dataset_id: str
    dataset_manifest_sha256: str
    training: Step52TrainingConfig
    train_windows: int = 4416
    validation_windows: int = 1104
    steps_per_epoch: int = 552
    validation_interval_steps: int = 1104
    checkpoint_save_interval_steps: int = 552
    learning_rate_schedule: str = "constant"
    checkpoint_selector: str = "argmin L_Val"
    best_checkpoint_policy: str = "save_on_strict_L_Val_improvement"
    precision: str = "fp32"

    def epoch_for_step(self, global_step: int) -> int:
        if not 0 <= global_step < self.training.max_steps:
            raise ValueError("global_step is outside the formal training budget")
        return global_step // self.steps_per_epoch

    def stage_for_step(self, global_step: int) -> str:
        self.epoch_for_step(global_step)
        return "posterior_assisted_warmup" if global_step < self.training.stage1_steps else "prior_dominant_recursive"

    def horizon_for_step(self, global_step: int) -> int:
        self.epoch_for_step(global_step)
        return self.training.curriculum.horizon_for_step(global_step, self.training.max_horizon)

    def validation_due(self, completed_steps: int) -> bool:
        return 0 < completed_steps <= self.training.max_steps and completed_steps % self.validation_interval_steps == 0

    def latest_checkpoint_due(self, completed_steps: int) -> bool:
        return 0 < completed_steps <= self.training.max_steps and completed_steps % self.checkpoint_save_interval_steps == 0

    @staticmethod
    def effective_adamw_defaults() -> dict[str, Any]:
        # Query PyTorch on a CPU dummy parameter; no optimizer step is run.
        parameter = torch.nn.Parameter(torch.zeros(1))
        group = torch.optim.AdamW([parameter]).param_groups[0]
        return {"betas": list(group["betas"]), "eps": float(group["eps"])}

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "PI-JWM-Formal-Training-Config-v1",
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "history_steps": 2,
            "horizon_steps": 4,
            "train_windows": self.train_windows,
            "validation_windows": self.validation_windows,
            "steps_per_epoch": self.steps_per_epoch,
            "training_config": asdict(self.training),
            "optimizer": {"name": "AdamW", "learning_rate": self.training.learning_rate,
                          "learning_rate_schedule": self.learning_rate_schedule,
                          "weight_decay": self.training.weight_decay,
                          "effective_defaults": self.effective_adamw_defaults(),
                          "gradient_clip_norm": self.training.gradient_clip_norm},
            "stage": {"stage1_steps": self.training.stage1_steps,
                      "transition_global_step": self.training.stage1_steps,
                      "stage1": "posterior_assisted_warmup",
                      "stage2": "prior_dominant_recursive"},
            "curriculum": asdict(self.training.curriculum),
            "kl": {**asdict(self.training.kl_schedule), "free_bits_unit": "per_latent_dimension",
                   "overshooting": False},
            "validation": {"mode": "prior_only", "windows": self.validation_windows,
                           "horizons": [1, 2, 3, 4],
                           "interval_completed_steps": self.validation_interval_steps},
            "checkpoint": {"latest_interval_completed_steps": self.checkpoint_save_interval_steps,
                           "best_policy": self.best_checkpoint_policy,
                           "selector": self.checkpoint_selector,
                           "patience_full_validations": self.training.checkpoint_patience},
            "resume": {"restore": ["model", "optimizer", "rng", "global_step", "epoch",
                                   "best_l_val", "early_stopping_counter", "curriculum_horizon"],
                       "sampler_order": "derive_from_formal_seed_and_global_step",
                       "step_index_policy": "global_step_is_zero_based; intervals_use_completed_steps"},
            "sampler": {"seed": self.training.seed,
                        "policy": "trajectory_shuffle_then_within_trajectory_shuffle",
                        "train_windows_exactly_once_per_epoch": True,
                        "validation_never_mixed": True},
            "precision": self.precision,
            "mixed_precision": False,
            "researcher_decision": True,
            "formal_training_started": False,
            "locked_test_accessed": False,
        }


def formal_training_config_v1(manifest_path: str | Path) -> FormalTrainingConfigV1:
    """Build the formal config from the accepted manifest without making a Trainer."""
    interface = FormalTrainingInterface.from_manifest(manifest_path)
    contract = interface.manifest.get("contract", {})
    split = interface.manifest.get("split", {})
    if (interface.dataset_manifest_hash != DATASET_MANIFEST_SHA256
            or interface.manifest.get("dataset_id") != DATASET_ID
            or contract.get("history_steps") != 2 or contract.get("horizon_steps") != 4
            or split.get("train_trajectories") != 48 or split.get("validation_trajectories") != 12
            or len(interface.train_indices) != 4416 or len(interface.validation_indices) != 1104):
        raise ValueError("Formal Dataset v1 identity or frozen 4416/1104 split mismatch")
    training = Step52TrainingConfig(
        seed=5601, batch_size=8, learning_rate=3e-4, weight_decay=0.0,
        gradient_clip_norm=1.0, max_epochs=10, max_steps=5520,
        stage1_steps=552, checkpoint_patience=3, max_horizon=4,
        curriculum=CurriculumConfig(horizons=(1, 2, 4), start_steps=(0, 1104, 2208)),
        kl_schedule=KLSchedule(target_beta=1.0, warmup_steps=1104, free_bits=0.1),
        device="cuda",
    )
    config = FormalTrainingConfigV1(DATASET_ID, interface.dataset_manifest_hash, training)
    if config.steps_per_epoch * training.batch_size != config.train_windows or config.steps_per_epoch * training.max_epochs != training.max_steps:
        raise ValueError("formal batch/epoch budget is inconsistent")
    if config.effective_adamw_defaults() != {"betas": [0.9, 0.999], "eps": 1e-8}:
        raise ValueError("effective PyTorch AdamW defaults differ from researcher decision")
    return config


__all__ = ["FormalTrainingConfigV1", "formal_training_config_v1", "DATASET_MANIFEST_SHA256", "DATASET_ID"]
