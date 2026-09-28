"""Focused CPU negative gates for the final formal training receipt."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import torch

from accept_step5_6c_formal_training_v1 import (
    ROOT, RUN_ID, checkpoint_identity, model_states_equal,
    validate_train_rows, validate_validation_rows,
)
from pi_jwm.step5_6a_formal_training_config_v1 import formal_training_config_v1


RUN_DIR = ROOT / "code/artifacts/formal_training" / RUN_ID
DATASET = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/formal_dataset_manifest.json"
CONFIG = ROOT / "code/artifacts/manifests/pi_jwm_step5_6a_formal_config_v1_20260924/formal_training_config_v1.json"


class FinalAcceptanceGatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = [json.loads(line) for line in (RUN_DIR / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        cls.validation = [json.loads(line) for line in (RUN_DIR / "validation_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        cls.protocol = formal_training_config_v1(DATASET)

    def test_exact_training_steps_and_wrong_stage_rejected(self) -> None:
        validate_train_rows(self.train, RUN_ID, self.protocol)
        with self.assertRaisesRegex(ValueError, "5520"):
            validate_train_rows(self.train[:-1], RUN_ID, self.protocol)
        rows = self.train.copy()
        rows[2208] = {**rows[2208], "rollout_horizon": 2}
        with self.assertRaisesRegex(ValueError, "stage/curriculum"):
            validate_train_rows(rows, RUN_ID, self.protocol)
        rows = self.train.copy()
        rows[-1] = {**rows[-1], "L_Total": float("nan")}
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_train_rows(rows, RUN_ID, self.protocol)

    def test_validation_coverage_selector_and_leakage_rejected(self) -> None:
        observations = validate_validation_rows(self.validation, RUN_ID)
        self.assertEqual([row["completed_steps"] for row in observations], [1104, 2208, 3312, 4416, 5520])
        self.assertEqual(min(observations, key=lambda row: row["L_Val"])["completed_steps"], 5520)
        rows = copy.deepcopy(self.validation)
        rows[-1]["validation_windows"] = 1103
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_validation_rows(rows, RUN_ID)
        rows = copy.deepcopy(self.validation)
        rows[-1]["future_target_encoder_calls"] = 1
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_validation_rows(rows, RUN_ID)
        rows = copy.deepcopy(self.validation)
        rows[-1]["L_Val"] += 0.01
        with self.assertRaisesRegex(ValueError, "L_Val"):
            validate_validation_rows(rows, RUN_ID)

    def test_checkpoint_identity_and_tensor_tamper_rejected(self) -> None:
        run = json.loads((RUN_DIR / "run_manifest.json").read_text(encoding="utf-8"))
        frozen = json.loads(CONFIG.read_text(encoding="utf-8"))
        best = torch.load(RUN_DIR / "checkpoints/best.pt", map_location="cpu", weights_only=False)
        latest = torch.load(RUN_DIR / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
        sample_index = best["data_identity"]["sample_index_sha256"]
        final_l_val = self.validation[-1]["L_Val"]
        checkpoint_identity(best, run, frozen, sample_index, 5520, final_l_val)
        self.assertTrue(model_states_equal(best["model_state"], latest["model_state"])[0])
        wrong = {**best, "state": {**best["state"], "formal_config_sha256": "wrong"}}
        with self.assertRaisesRegex(ValueError, "source/config"):
            checkpoint_identity(wrong, run, frozen, sample_index, 5520, final_l_val)
        wrong = {**best, "data_identity": {**best["data_identity"], "dataset_manifest_hash": "wrong"}}
        with self.assertRaisesRegex(ValueError, "Dataset/package"):
            checkpoint_identity(wrong, run, frozen, sample_index, 5520, final_l_val)
        changed = {module: dict(values) for module, values in latest["model_state"].items()}
        module = next(iter(changed))
        name = next(iter(changed[module]))
        changed[module][name] = changed[module][name].clone()
        changed[module][name].view(-1)[0] += 1
        self.assertFalse(model_states_equal(best["model_state"], changed)[0])


if __name__ == "__main__":
    unittest.main()
