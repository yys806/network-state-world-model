from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r5_confirmation_checkpoint import (  # noqa: E402
    CONFIRMATION_REQUIRED_BINDINGS,
    load_confirmation_checkpoint,
    save_confirmation_checkpoint,
)
from pi_jwm.r5_module_confirmation import build_confirmation_model  # noqa: E402
from pi_jwm.r5_protocol import R5FormalProtocol  # noqa: E402


class R5ConfirmationCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = R5FormalProtocol(
            training_seeds=(20260803, 20260804, 20260805),
            max_epochs=100,
            patience=10,
            effective_batch_size=32,
            minimum_improvement=1.0e-4,
        )
        self.bindings = {key: "a" * 64 for key in CONFIRMATION_REQUIRED_BINDINGS}

    def test_round_trip_reconstructs_each_new_candidate(self) -> None:
        for combination_id in ("F", "G", "H", "J"):
            with self.subTest(combination_id=combination_id), tempfile.TemporaryDirectory() as tmp:
                model = build_confirmation_model(combination_id)
                optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
                path = Path(tmp) / "checkpoint.pt"
                save_confirmation_checkpoint(
                    path,
                    model,
                    optimizer,
                    self.bindings,
                    self.protocol,
                    learning_rate=1.0e-4,
                    seed=20260803,
                )
                loaded = load_confirmation_checkpoint(
                    path,
                    expected_bindings=self.bindings,
                    expected_protocol=self.protocol,
                )
                self.assertEqual(loaded.model.combination_id, combination_id)
                self.assertEqual(loaded.model.component_registry(), model.component_registry())
                self.assertIsNotNone(loaded.optimizer_state)

    def test_binding_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            model = build_confirmation_model("J")
            save_confirmation_checkpoint(
                path,
                model,
                None,
                self.bindings,
                self.protocol,
                learning_rate=1.0e-4,
                seed=20260803,
            )
            changed = dict(self.bindings)
            changed["confirmation_matrix_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "confirmation_matrix_sha256"):
                load_confirmation_checkpoint(
                    path,
                    expected_bindings=changed,
                    expected_protocol=self.protocol,
                )


if __name__ == "__main__":
    unittest.main()
