from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from test_r3_world_model import make_batch


def r5_bindings() -> dict[str, str]:
    return {
        "tensor_contract_sha256": "a" * 64,
        "dataset_protocol_sha256": "b" * 64,
        "normalization_sha256": "c" * 64,
        "metric_registry_sha256": "d" * 64,
        "source_code_sha256": "e" * 64,
        "r4_screening_manifest_sha256": "f" * 64,
        "r5_protocol_sha256": "1" * 64,
    }


class R5CheckpointTests(unittest.TestCase):
    def _protocol(self):
        from pi_jwm.r5_protocol import R5FormalProtocol

        return R5FormalProtocol(
            training_seeds=(20260803, 20260804, 20260805),
            max_epochs=100,
            patience=10,
            effective_batch_size=32,
            minimum_improvement=1.0e-4,
        )

    def test_roundtrip_preserves_combination_protocol_and_prediction(self):
        from pi_jwm.r5_checkpoint import load_r5_checkpoint, save_r5_checkpoint
        from pi_jwm.r5_world_model import build_r5_world_model

        torch.manual_seed(113)
        model = build_r5_world_model("C", hidden_dim=8, history_steps=2).eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
        batch = make_batch(horizon=2, action_value=0.25)
        before = model(batch, rollout_steps=2).predicted_belief.joint_latent.detach()

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "r5-c.pt"
            save_r5_checkpoint(
                path,
                model,
                optimizer,
                r5_bindings(),
                self._protocol(),
                learning_rate=1.0e-4,
                seed=20260803,
            )
            restored = load_r5_checkpoint(
                path,
                expected_bindings=r5_bindings(),
                expected_protocol=self._protocol(),
            )

        restored.model.eval()
        after = restored.model(batch, rollout_steps=2).predicted_belief.joint_latent
        torch.testing.assert_close(before, after)
        self.assertEqual("C", restored.model.combination_id)
        self.assertEqual(self._protocol(), restored.protocol)
        self.assertEqual(1.0e-4, restored.learning_rate)
        self.assertEqual(20260803, restored.seed)
        self.assertIsNotNone(restored.optimizer_state)

    def test_rejects_changed_upstream_binding(self):
        from pi_jwm.r5_checkpoint import load_r5_checkpoint, save_r5_checkpoint
        from pi_jwm.r5_world_model import build_r5_world_model

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "r5-b.pt"
            save_r5_checkpoint(
                path,
                build_r5_world_model("B", hidden_dim=8, history_steps=2),
                None,
                r5_bindings(),
                self._protocol(),
                learning_rate=1.0e-4,
                seed=20260804,
            )
            changed = {**r5_bindings(), "r5_protocol_sha256": "2" * 64}
            with self.assertRaisesRegex(ValueError, "r5_protocol_sha256"):
                load_r5_checkpoint(
                    path,
                    expected_bindings=changed,
                    expected_protocol=self._protocol(),
                )

    def test_rejects_tampered_combination(self):
        from pi_jwm.r5_checkpoint import load_r5_checkpoint, save_r5_checkpoint
        from pi_jwm.r5_world_model import build_r5_world_model

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "r5-d.pt"
            save_r5_checkpoint(
                path,
                build_r5_world_model("D", hidden_dim=8, history_steps=2),
                None,
                r5_bindings(),
                self._protocol(),
                learning_rate=1.0e-4,
                seed=20260805,
            )
            envelope = torch.load(path, map_location="cpu", weights_only=True)
            envelope["combination_id"] = "E"
            torch.save(envelope, path)
            with self.assertRaisesRegex(ValueError, "combination"):
                load_r5_checkpoint(
                    path,
                    expected_bindings=r5_bindings(),
                    expected_protocol=self._protocol(),
                )

    def test_rejects_seed_outside_frozen_protocol(self):
        from pi_jwm.r5_checkpoint import save_r5_checkpoint
        from pi_jwm.r5_world_model import build_r5_world_model

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "seed"):
                save_r5_checkpoint(
                    Path(temporary) / "r5-a.pt",
                    build_r5_world_model("A", hidden_dim=8, history_steps=2),
                    None,
                    r5_bindings(),
                    self._protocol(),
                    learning_rate=1.0e-4,
                    seed=7,
                )


if __name__ == "__main__":
    unittest.main()
