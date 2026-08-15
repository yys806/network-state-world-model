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

from test_r3_checkpoint import bindings
from test_r3_world_model import make_batch


class R4CheckpointTests(unittest.TestCase):
    def _budget(self):
        from pi_jwm.r4_checkpoint import R4TrainingBudget

        return R4TrainingBudget(
            epochs=30,
            patience=5,
            learning_rate=1e-3,
            training_seed=20260804,
        )

    def test_checkpoint_roundtrip_preserves_config_budget_and_prediction(self):
        from pi_jwm.r4_checkpoint import load_r4_checkpoint, save_r4_checkpoint
        from pi_jwm.r4_module_registry import make_single_module_config
        from pi_jwm.r4_world_model import build_r4_world_model

        torch.manual_seed(53)
        config = make_single_module_config(
            "coupling",
            "no_cross_graph_coupling_v1",
            hidden_dim=8,
            history_steps=2,
        )
        model = build_r4_world_model(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = make_batch(horizon=2, action_value=0.25)
        before = model(batch, rollout_steps=2).predicted_belief.joint_latent.detach()

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.pt"
            save_r4_checkpoint(
                path,
                model,
                optimizer,
                bindings(),
                self._budget(),
                seed=71,
            )
            restored = load_r4_checkpoint(path, expected_bindings=bindings())

        after = restored.model(batch, rollout_steps=2).predicted_belief.joint_latent
        torch.testing.assert_close(before, after)
        self.assertEqual(config, restored.model.config)
        self.assertEqual(self._budget(), restored.budget)
        self.assertEqual(71, restored.seed)
        self.assertIsNotNone(restored.optimizer_state)

    def test_checkpoint_rejects_tampered_component_configuration(self):
        from pi_jwm.r4_checkpoint import load_r4_checkpoint, save_r4_checkpoint
        from pi_jwm.r4_module_registry import reference_r4_config
        from pi_jwm.r4_world_model import build_r4_world_model

        model = build_r4_world_model(reference_r4_config(hidden_dim=4, history_steps=2))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.pt"
            save_r4_checkpoint(
                path,
                model,
                None,
                bindings(),
                self._budget(),
                seed=1,
            )
            envelope = torch.load(path, map_location="cpu", weights_only=True)
            envelope["components"]["dynamics"] = "graph_rssm_v1"
            torch.save(envelope, path)
            with self.assertRaisesRegex(ValueError, "components"):
                load_r4_checkpoint(path, expected_bindings=bindings())

    def test_checkpoint_rejects_contract_hash_change(self):
        from pi_jwm.r4_checkpoint import load_r4_checkpoint, save_r4_checkpoint
        from pi_jwm.r4_module_registry import reference_r4_config
        from pi_jwm.r4_world_model import build_r4_world_model

        model = build_r4_world_model(reference_r4_config(hidden_dim=4, history_steps=2))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.pt"
            save_r4_checkpoint(
                path,
                model,
                None,
                bindings(),
                self._budget(),
                seed=1,
            )
            changed = {**bindings(), "metric_registry_sha256": "f" * 64}
            with self.assertRaisesRegex(ValueError, "metric_registry_sha256"):
                load_r4_checkpoint(path, expected_bindings=changed)


if __name__ == "__main__":
    unittest.main()
