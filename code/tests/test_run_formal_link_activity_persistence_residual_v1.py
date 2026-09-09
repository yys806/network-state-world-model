from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_run_formal_dual_graph_gpu_train_v1 import _write_three_split_formal_fixture


class RunFormalLinkActivityPersistenceResidualV1Tests(unittest.TestCase):
    def test_method_registry_exposes_only_the_confirmed_persistence_residual_method(self):
        from run_formal_dual_graph_gpu_train_v1 import MODEL_SPECS

        self.assertIn("link_activity_persistence_residual_v1", MODEL_SPECS)
        self.assertEqual(
            "persistence_residual_v1",
            MODEL_SPECS["link_activity_persistence_residual_v1"]["link_activity_method"],
        )
        self.assertTrue(
            MODEL_SPECS["link_activity_persistence_residual_v1"]["residual"],
            "the confirmed candidate must preserve the existing residual state heads",
        )

    def test_runner_binds_train_only_prior_and_pos_weight_to_model_config(self):
        from run_formal_dual_graph_gpu_train_v1 import _build_learned_model

        _, config = _build_learned_model(
            "link_activity_persistence_residual_v1",
            hidden_dim=4,
            history_steps=3,
            horizon_steps=2,
            use_system_energy_head=False,
            link_pos_weight=50.0,
            link_activity_missing_prior=0.25,
        )
        self.assertEqual("persistence_residual_v1", config.link_activity_method)
        self.assertTrue(config.residual_state_prediction)
        self.assertEqual(50.0, config.link_activity_pos_weight)
        self.assertEqual(0.25, config.link_activity_missing_history_prior)

    def test_validator_accepts_same_semantics_and_rejects_missing_identity(self):
        from run_formal_dual_graph_gpu_train_v1 import validate_checkpoint_method_semantics

        validate_checkpoint_method_semantics(
            "link_activity_persistence_residual_v1",
            {
                "link_activity_method": "persistence_residual_v1",
                "mode": "coupled_dual_gnn",
                "residual_state_prediction": True,
            },
        )
        with self.assertRaisesRegex(ValueError, "link_activity_method"):
            validate_checkpoint_method_semantics(
                "link_activity_persistence_residual_v1", {}
            )

    def test_validator_rejects_wrong_base_mode_and_state_head_semantics(self):
        from run_formal_dual_graph_gpu_train_v1 import validate_checkpoint_method_semantics

        valid = {
            "link_activity_method": "persistence_residual_v1",
            "mode": "coupled_dual_gnn",
            "residual_state_prediction": True,
        }
        for field, wrong_value in (
            ("mode", "independent_dual_gnn"),
            ("residual_state_prediction", False),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                changed = dict(valid)
                changed[field] = wrong_value
                validate_checkpoint_method_semantics(
                    "link_activity_persistence_residual_v1", changed
                )

    def test_method_registry_contains_learned_persistence_residual(self):
        from pi_jwm.formal_world_model_baselines_v1 import method_registry

        registry = method_registry()
        self.assertEqual(
            "persistence_residual_v1",
            registry["link_activity_persistence_residual_v1"]["link_activity_method"],
        )
        self.assertTrue(
            registry["link_activity_persistence_residual_v1"]["residual_state_prediction"]
        )
        self.assertEqual(
            "coupled_dual_gnn",
            registry["link_activity_persistence_residual_v1"]["base_mode"],
        )

    def test_train_only_prior_helper_uses_link_activity_counts(self):
        from run_formal_dual_graph_gpu_train_v1 import _link_activity_train_prior

        report = {
            "source_split": "train",
            "counts": {"link_activity": {"positive": 3, "negative": 9}},
        }
        self.assertAlmostEqual(0.25, _link_activity_train_prior(report))

    def test_old_checkpoint_semantics_are_rejected_for_new_method(self):
        from run_formal_dual_graph_gpu_train_v1 import validate_checkpoint_method_semantics

        with self.assertRaisesRegex(ValueError, "link_activity_method"):
            validate_checkpoint_method_semantics(
                "link_activity_persistence_residual_v1",
                {"link_activity_method": "absolute_v1"},
            )

    def test_training_records_persistence_semantics_and_train_only_prior(self):
        from run_formal_dual_graph_gpu_train_v1 import run_formal_training

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            output_dir = root / "output"
            tensor_root.mkdir()
            _write_three_split_formal_fixture(tensor_root)
            train_tensor_path = tensor_root / "seed_000" / "trajectory_tensors.npz"
            with np.load(train_tensor_path, allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            arrays["physical_edge_state"][2, 0, 3] = 0.0
            np.savez_compressed(train_tensor_path, **arrays)

            run_formal_training(
                tensor_root=tensor_root,
                output_dir=output_dir,
                device="cpu",
                learned_methods=("link_activity_persistence_residual_v1",),
                seed=53,
                train_limit=1,
                evaluation_limit=1,
                hidden_dim=4,
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            )

            config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(
                "persistence_residual_v1",
                config["link_activity_methods"]["link_activity_persistence_residual_v1"],
            )
            self.assertEqual("train", config["link_activity_train_prior_source"]["source_split"])
            self.assertGreater(config["link_activity_train_prior"], 0.0)
            self.assertLess(config["link_activity_train_prior"], 1.0)
            summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(config["link_activity_methods"], summary["link_activity_methods"])
            self.assertEqual(
                config["link_activity_train_prior"], summary["link_activity_train_prior"]
            )
            self.assertEqual(
                config["link_activity_train_prior_source"],
                summary["link_activity_train_prior_source"],
            )


if __name__ == "__main__":
    unittest.main()
