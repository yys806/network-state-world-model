from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "code" / "src", ROOT / "code" / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.step5_2_training_loop_v1 import Step52Trainer, Step52TrainingConfig  # noqa: E402


class Step53EFormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Step52TrainingConfig(seed=5303, max_steps=2, stage1_steps=1)
        cls.trainer = Step52Trainer.from_unified_development_bundle(cls.config)

    def test_raw_db_train_mean_bias_contract(self):
        trainer = Step52Trainer.from_unified_development_bundle(self.config)
        bias = trainer.model.csi_decoder[-1].bias.detach()
        mean = float(trainer.data.target_normalization["csi_mean"])
        self.assertTrue(torch.allclose(bias, torch.full_like(bias, mean)))
        self.assertEqual(trainer.initialization_contract["normalization_provenance"]["source_split"], "dev_train")
        self.assertEqual(trainer.model.contract["csi_decoder_output_space"], "raw_db")
        self.assertTrue(torch.allclose(bias, torch.full_like(bias, mean), atol=1e-5, rtol=1e-6))

    def test_checkpoint_preserves_bias_and_rejects_contract_mismatch(self):
        trainer = self.trainer
        trainer.train_step(global_step=0, epoch=0)
        before = trainer.model.csi_decoder[-1].bias.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            trainer.save_checkpoint(path, state=trainer.state_snapshot(global_step=1))
            restored = Step52Trainer.from_unified_development_bundle(self.config)
            restored.load_checkpoint(path)
            self.assertTrue(torch.equal(before, restored.model.csi_decoder[-1].bias.detach()))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["initialization_contract"] = dict(payload["initialization_contract"])
            payload["initialization_contract"]["resolved_bias_value"] = -1.0
            bad = Path(tmp) / "bad.pt"
            torch.save(payload, bad)
            with self.assertRaisesRegex(ValueError, "initialization contract"):
                restored.load_checkpoint(bad)

    def test_config_rejects_non_v1_initialization(self):
        with self.assertRaises(ValueError):
            Step52TrainingConfig(csi_decoder_bias_init_policy="normalized_output")


if __name__ == "__main__":
    unittest.main()
