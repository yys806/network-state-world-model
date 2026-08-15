from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from test_r3_world_model import make_batch


def bindings() -> dict[str, str]:
    return {
        "tensor_contract_sha256": "a" * 64,
        "dataset_protocol_sha256": "b" * 64,
        "normalization_sha256": "c" * 64,
        "metric_registry_sha256": "d" * 64,
        "source_code_sha256": "e" * 64,
    }


class R3CheckpointTests(unittest.TestCase):
    def test_checkpoint_roundtrip_preserves_predictions(self):
        from pi_jwm.r3_checkpoint import load_r3_checkpoint, save_r3_checkpoint
        from pi_jwm.r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel

        torch.manual_seed(31)
        model = R3ReferenceWorldModel(R3ReferenceConfig(hidden_dim=8, history_steps=2))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = make_batch(horizon=2, action_value=0.5)
        before = model(batch, rollout_steps=2).predicted_belief.joint_latent.detach()

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.pt"
            save_r3_checkpoint(path, model, optimizer, bindings(), seed=20260804)
            restored = load_r3_checkpoint(path, expected_bindings=bindings())

        after = restored.model(batch, rollout_steps=2).predicted_belief.joint_latent
        torch.testing.assert_close(before, after)
        self.assertEqual(20260804, restored.seed)
        self.assertIsNotNone(restored.optimizer_state)

    def test_checkpoint_rejects_contract_hash_change(self):
        from pi_jwm.r3_checkpoint import load_r3_checkpoint, save_r3_checkpoint
        from pi_jwm.r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel

        model = R3ReferenceWorldModel(R3ReferenceConfig(hidden_dim=4, history_steps=2))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.pt"
            save_r3_checkpoint(path, model, None, bindings(), seed=1)
            changed = {**bindings(), "tensor_contract_sha256": "f" * 64}
            with self.assertRaisesRegex(ValueError, "tensor_contract_sha256"):
                load_r3_checkpoint(path, expected_bindings=changed)

    def test_checkpoint_rejects_unknown_or_tampered_component(self):
        from pi_jwm.r3_checkpoint import load_r3_checkpoint, save_r3_checkpoint
        from pi_jwm.r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel

        model = R3ReferenceWorldModel(R3ReferenceConfig(hidden_dim=4, history_steps=2))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.pt"
            save_r3_checkpoint(path, model, None, bindings(), seed=1)
            envelope = torch.load(path, map_location="cpu", weights_only=True)
            envelope["components"]["dynamics"] = "old_v2_1"
            torch.save(envelope, path)
            with self.assertRaisesRegex(ValueError, "components"):
                load_r3_checkpoint(path, expected_bindings=bindings())


if __name__ == "__main__":
    unittest.main()
