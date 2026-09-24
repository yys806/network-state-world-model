"""Focused tests for the STEP 5.2 CPU training-loop contract."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"
SCRIPTS = ROOT / "code" / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.step5_2_training_loop_v1 import (  # noqa: E402
    CurriculumConfig,
    KLSchedule,
    REQUIRED_STEP52_CHECKS,
    Step52TrainingConfig,
    Step52Trainer,
    aggregate_validation_horizon_rows,
    count_available_validation_windows,
    validate_step52_receipt,
)


class Step52TrainingLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Step52TrainingConfig(
            seed=5201,
            batch_size=1,
            stage1_steps=1,
            max_steps=2,
            max_epochs=1,
            checkpoint_patience=2,
        )
        cls.trainer = Step52Trainer.from_unified_development_bundle(cls.config)

    def test_schedule_is_configured_and_current_horizon_naturally_caps_at_l(self):
        curriculum = CurriculumConfig(horizons=(1, 2, 4), start_steps=(0, 1, 2))
        self.assertEqual(curriculum.horizon_for_step(0, 2), 1)
        self.assertEqual(curriculum.horizon_for_step(1, 2), 2)
        self.assertEqual(curriculum.horizon_for_step(2, 2), 2)
        self.assertEqual(curriculum.horizon_for_step(100, 2), 2)

    def test_kl_schedule_starts_at_zero_and_reaches_target(self):
        schedule = KLSchedule(target_beta=0.8, warmup_steps=4, free_bits=0.1)
        self.assertEqual(schedule.beta_at(0), 0.0)
        self.assertAlmostEqual(schedule.beta_at(2), 0.4)
        self.assertAlmostEqual(schedule.beta_at(4), 0.8)
        self.assertAlmostEqual(schedule.beta_at(999), 0.8)

    def test_optimizer_contains_trainable_modules_and_no_rule_parameters(self):
        audit = self.trainer.optimizer_parameter_audit()
        self.assertTrue(audit["all_required_groups_present"])
        self.assertEqual(audit["known_rule_parameter_count"], 0)
        self.assertTrue(audit["known_rule_parameters_excluded"])
        for name in (
            "encoder",
            "rssm_dynamics",
            "phy_prior",
            "comm_prior",
            "current_observation_posterior",
            "phy_future_posterior",
            "comm_future_posterior",
            "motion_target_encoder",
            "csi_target_encoder",
            "vehicle_motion_decoder",
            "csi_decoder",
        ):
            self.assertTrue(audit["groups"][name]["in_optimizer"], name)

    def test_stage1_teacher_and_stage2_prior_recursive_paths(self):
        stage1 = self.trainer.train_step(global_step=0, epoch=0)
        self.assertEqual(stage1["stage"], "posterior_assisted_warmup")
        self.assertTrue(stage1["posterior_teacher_used"])
        self.assertTrue(stage1["loss_finite"])
        self.assertEqual(stage1["rollout_horizon"], 1)

        stage2 = self.trainer.train_step(global_step=1, epoch=0)
        self.assertEqual(stage2["stage"], "prior_dominant_recursive")
        self.assertTrue(stage2["posterior_teacher_used"])  # KL teacher only.
        self.assertTrue(stage2["prior_only_rollout"])
        self.assertFalse(stage2["posterior_used_as_rollout_state"])
        self.assertTrue(stage2["recursive_state_feedback"])
        self.assertEqual(stage2["rollout_horizon"], 2)
        self.assertTrue(stage2["loss_finite"])
        self.assertTrue(stage2["parameter_update"]["any_changed"])

    def test_future_target_does_not_change_prior_and_changes_posterior(self):
        audit = self.trainer.prior_target_isolation_probe()
        self.assertTrue(audit["prior_mean_unchanged"])
        self.assertTrue(audit["prior_log_std_unchanged"])
        self.assertTrue(audit["posterior_mean_changed"])

    def test_current_observation_posterior_ignores_future_target(self):
        audit = self.trainer.current_latent_target_isolation_probe()
        self.assertTrue(audit["current_posterior_uses_current_observation"])
        self.assertTrue(audit["future_target_mutation_does_not_change_current_posterior"])

    def test_validation_is_prior_only_and_does_not_update_parameters(self):
        before = self.trainer.parameter_digest()
        result = self.trainer.validate()
        after = self.trainer.parameter_digest()
        self.assertEqual(before, after)
        self.assertTrue(result["model_eval"])
        self.assertTrue(result["no_grad"])
        self.assertTrue(result["prior_only_rollout"])
        self.assertEqual(result["posterior_rollout_calls"], 0)
        self.assertEqual(result["future_posterior_teacher_calls"], 0)
        self.assertEqual(result["future_target_encoder_calls"], 0)
        self.assertGreater(result["current_observation_posterior_calls"], 0)
        for row in result["per_horizon"]:
            self.assertIn("motion_numerator", row)
            self.assertIn("csi_numerator", row)
            self.assertIn("L_Mot", row)
            self.assertIn("L_CSI", row)
        self.assertTrue(result["selector_uses_l_val_only"])
        self.assertTrue(result["l_val_finite"])

    def test_validation_aggregation_uses_global_unequal_mask_counts(self):
        rows = aggregate_validation_horizon_rows({1: [
            {"motion_numerator": 2.0, "motion_count": 1, "csi_numerator": 10.0, "csi_count": 2, "available": False},
            {"motion_numerator": 6.0, "motion_count": 3, "csi_numerator": 0.0, "csi_count": 1, "available": False},
        ]})
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["L_Mot"], 2.0)
        self.assertAlmostEqual(rows[0]["L_CSI"], 10.0 / 3.0)
        self.assertAlmostEqual(rows[0]["L_Pred"], 0.5 * (2.0 + 10.0 / 3.0))

    def test_available_count_is_per_window_and_survives_reaggregation(self):
        motion = torch.zeros((8, 2, 2), dtype=torch.bool)
        csi = torch.zeros((8, 3, 2), dtype=torch.bool)
        motion[[0, 1, 2, 3, 4, 5], 0, 0] = True
        csi[[0, 1, 2, 3, 4, 6], 0, 0] = True
        self.assertEqual(count_available_validation_windows(motion, csi), 5)
        self.assertEqual(sum(count_available_validation_windows(motion[i:i + 1], csi[i:i + 1]) for i in range(8)), 5)
        base = {"motion_numerator": 10.0, "motion_count": 5, "csi_numerator": 12.0, "csi_count": 6}
        first = aggregate_validation_horizon_rows({1: [{**base, "available_sample_count": 5}]})
        second = aggregate_validation_horizon_rows({1: first})
        self.assertEqual(first[0]["available_sample_count"], 5)
        self.assertEqual(second, first)
        self.assertEqual(first[0]["L_Pred"], 2.0)

    def test_checkpoint_selector_and_early_stopping_use_only_l_val(self):
        state = self.trainer.state_snapshot(global_step=2, epoch=0, curriculum_horizon=2, best_l_val=1.0, early_stopping_counter=1)
        improved = self.trainer.update_validation_state(state, {"L_Val": 0.5, "L_KL": 99.0})
        self.assertEqual(improved["selector_metric"], "L_Val")
        self.assertEqual(improved["best_l_val"], 0.5)
        self.assertEqual(improved["early_stopping_counter"], 0)
        stale = self.trainer.update_validation_state(improved, {"L_Val": 0.6, "L_KL": 0.0})
        self.assertEqual(stale["best_l_val"], 0.5)
        self.assertEqual(stale["early_stopping_counter"], 1)

    def test_checkpoint_reload_and_resume_preserve_forward_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step52.pt"
            state = self.trainer.state_snapshot(global_step=7, epoch=2, curriculum_horizon=2)
            self.trainer.save_checkpoint(path, state=state)
            reloaded = Step52Trainer.from_unified_development_bundle(self.config)
            restored = reloaded.load_checkpoint(path)
            self.assertEqual(restored["global_step"], 7)
            self.assertEqual(restored["epoch"], 2)
            self.assertEqual(restored["curriculum_horizon"], 2)
            self.assertEqual(restored["best_l_val"], state["best_l_val"])
            self.assertEqual(restored["early_stopping_counter"], state["early_stopping_counter"])
            left = self.trainer.deterministic_forward_digest()
            right = reloaded.deterministic_forward_digest()
            self.assertEqual(left, right)

    def test_checkpoint_rejects_wrong_data_and_normalization_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step52.pt"
            self.trainer.save_checkpoint(path, state=self.trainer.state_snapshot(global_step=1))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["data_identity"] = dict(payload["data_identity"])
            payload["data_identity"]["target_sha256"] = "wrong"
            wrong_data = Path(tmp) / "wrong_data.pt"
            torch.save(payload, wrong_data)
            with self.assertRaisesRegex(ValueError, "data identity"):
                Step52Trainer.from_unified_development_bundle(self.config).load_checkpoint(wrong_data)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["normalization_provenance"] = {"target": {"csi_mean": -1.0}, "encoder_source_split": "dev_train"}
            wrong_norm = Path(tmp) / "wrong_norm.pt"
            torch.save(payload, wrong_norm)
            with self.assertRaisesRegex(ValueError, "normalization provenance"):
                Step52Trainer.from_unified_development_bundle(self.config).load_checkpoint(wrong_norm)

    def test_receipt_validator_rejects_required_and_forbidden_tampering(self):
        receipt = {
            "required_checks": {name: True for name in REQUIRED_STEP52_CHECKS},
            "forbidden_scope": {"training": False, "gpu": False},
            "passed": True,
        }
        self.assertTrue(validate_step52_receipt(receipt)["passed"])
        required_tamper = copy.deepcopy(receipt)
        required_tamper["required_checks"][next(iter(REQUIRED_STEP52_CHECKS))] = False
        self.assertFalse(validate_step52_receipt(required_tamper)["passed"])
        forbidden_tamper = copy.deepcopy(receipt)
        forbidden_tamper["forbidden_scope"]["gpu"] = True
        self.assertFalse(validate_step52_receipt(forbidden_tamper)["passed"])


if __name__ == "__main__":
    unittest.main()
