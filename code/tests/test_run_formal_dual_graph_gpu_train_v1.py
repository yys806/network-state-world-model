from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_formal_airfogsim_window_v1 import _write_formal_fixture
from test_formal_system_window_v1 import _write_system_fixture


def _write_three_split_formal_fixture(
    root: Path,
    *,
    calibration_link_activity_observed: bool = True,
    calibration_link_labels: str = "mixed",
) -> None:
    _write_formal_fixture(root)
    shutil.copytree(root / "seed_001", root / "seed_002")
    index_path = root / "window_index.csv"
    with index_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    calibration_row = dict(rows[1])
    calibration_row.update(
        sample_id="seed002::window000000",
        seed="2",
        split="calibration",
    )
    rows.append(calibration_row)
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    calibration_path = root / "seed_002" / "trajectory_tensors.npz"
    with np.load(calibration_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    if calibration_link_labels == "mixed":
        arrays["physical_edge_state"][2, 0, 3] = 0.0
    elif calibration_link_labels == "all_zero":
        arrays["physical_edge_state"][2:, 0, 3] = 0.0
    elif calibration_link_labels == "all_one":
        arrays["physical_edge_state"][2:, 0, 3] = 1.0
    else:
        raise ValueError(f"unsupported calibration_link_labels: {calibration_link_labels}")
    np.savez_compressed(calibration_path, **arrays)
    if not calibration_link_activity_observed:
        with np.load(calibration_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        edge_feature_mask = np.ones_like(arrays["physical_edge_state"], dtype=bool)
        edge_feature_mask[..., 3] = False
        arrays["physical_edge_feature_mask"] = edge_feature_mask
        np.savez_compressed(calibration_path, **arrays)


class RunFormalDualGraphGpuTrainV1Tests(unittest.TestCase):
    def test_entity_aligned_rssm_identity_and_motion_contract_are_strict(self):
        from run_formal_dual_graph_gpu_train_v1 import (
            _build_learned_model,
            _reload_learned_model,
            validate_checkpoint_method_semantics,
        )

        method = "entity_aligned_dual_graph_rssm_v1"
        model, config = _build_learned_model(
            method,
            hidden_dim=8,
            history_steps=3,
            horizon_steps=2,
            use_system_energy_head=False,
            slot_seconds=0.1,
            node_position_scale=(2.0, 3.0, 4.0),
            node_motion_mean=(1.0,) * 6,
            node_motion_scale=(2.0,) * 6,
        )
        self.assertEqual(
            "entity_aligned_complete_rssm_prior_posterior_v1",
            model.latent_dynamics,
        )
        validate_checkpoint_method_semantics(method, config.__dict__)
        reloaded = _reload_learned_model(method, config.__dict__)
        reloaded.load_state_dict(model.state_dict(), strict=True)
        for missing in ("entity_latent_layout", "node_motion_contract"):
            broken = dict(config.__dict__)
            broken.pop(missing)
            with self.assertRaisesRegex(ValueError, missing):
                validate_checkpoint_method_semantics(method, broken)

    def test_entity_aligned_training_stages_are_disjoint(self):
        from run_formal_dual_graph_gpu_train_v1 import (
            _build_learned_model,
            _set_entity_training_stage,
        )

        model, _ = _build_learned_model(
            "entity_aligned_dual_graph_rssm_v1",
            hidden_dim=8,
            history_steps=3,
            horizon_steps=2,
            use_system_energy_head=False,
        )
        _set_entity_training_stage(model, "base")
        self.assertTrue(all(parameter.requires_grad for parameter in model.base.parameters()))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if not name.startswith("base.")
            )
        )
        _set_entity_training_stage(model, "rssm")
        self.assertTrue(all(not parameter.requires_grad for parameter in model.base.parameters()))
        self.assertTrue(
            all(
                parameter.requires_grad
                for name, parameter in model.named_parameters()
                if not name.startswith("base.")
            )
        )

    def test_node_x_safe_rssm_method_binds_the_single_loss_contract(self):
        from run_formal_dual_graph_gpu_train_v1 import (
            _build_learned_model,
            validate_checkpoint_method_semantics,
        )

        method = "complete_rssm_node_x_safe_dual_graph_v1"
        _, config = _build_learned_model(
            method,
            hidden_dim=4,
            history_steps=3,
            horizon_steps=2,
            use_system_energy_head=False,
        )
        self.assertEqual(
            "node_x_residual_non_degradation_v1",
            config.node_x_residual_loss_contract,
        )
        validate_checkpoint_method_semantics(method, config.__dict__)
        broken = dict(config.__dict__)
        broken["node_x_residual_loss_contract"] = "none"
        with self.assertRaisesRegex(ValueError, "node-x residual loss"):
            validate_checkpoint_method_semantics(method, broken)

    def test_complete_rssm_method_identity_is_strictly_reloaded(self):
        from run_formal_dual_graph_gpu_train_v1 import (
            _build_learned_model,
            _reload_learned_model,
            validate_checkpoint_method_semantics,
        )

        model, config = _build_learned_model(
            "complete_rssm_dual_graph_v1",
            hidden_dim=4,
            history_steps=3,
            horizon_steps=2,
            use_system_energy_head=False,
        )
        self.assertEqual("complete_rssm_prior_posterior_v1", model.latent_dynamics)
        reloaded = _reload_learned_model(
            "complete_rssm_dual_graph_v1", config.__dict__
        )
        reloaded.load_state_dict(model.state_dict(), strict=True)
        broken = dict(config.__dict__)
        broken["latent_dynamics"] = "deterministic"
        with self.assertRaisesRegex(ValueError, "latent_dynamics"):
            validate_checkpoint_method_semantics(
                "complete_rssm_dual_graph_v1", broken
            )
        missing_initialization = dict(config.__dict__)
        missing_initialization.pop("rssm_residual_head_initialization")
        with self.assertRaisesRegex(ValueError, "initialization"):
            validate_checkpoint_method_semantics(
                "complete_rssm_dual_graph_v1", missing_initialization
            )

    def test_complete_rssm_runs_through_formal_training_and_records_semantics(self):
        from run_formal_dual_graph_gpu_train_v1 import run_formal_training

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            output_dir = root / "output"
            tensor_root.mkdir()
            _write_three_split_formal_fixture(tensor_root)

            result = run_formal_training(
                tensor_root=tensor_root,
                output_dir=output_dir,
                device="cpu",
                learned_methods=("complete_rssm_dual_graph_v1",),
                seed=53,
                train_limit=1,
                evaluation_limit=1,
                hidden_dim=4,
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            )

            self.assertTrue(result["training_run_complete"])
            self.assertFalse(result["locked_test_accessed"])
            checkpoint = torch.load(
                output_dir / "checkpoints" / "complete_rssm_dual_graph_v1__best.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual("formal_complete_rssm_v1_1", checkpoint["model_version"])
            self.assertEqual(
                "complete_rssm_prior_posterior_v1", checkpoint["latent_dynamics"]
            )
            self.assertTrue(checkpoint["model_config"]["training_posterior_teacher"])
            self.assertTrue(checkpoint["model_config"]["deployment_prior_only"])
            config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            self.assertGreater(config["loss_weights"]["rssm_kl"], 0.0)
            self.assertGreater(config["loss_weights"]["rssm_teacher_reconstruction"], 0.0)
            self.assertGreater(config["loss_weights"]["rssm_overshooting"], 0.0)

    def test_gpu_protocol_accepts_only_nonlocked_splits_and_cuda(self):
        from run_formal_dual_graph_gpu_train_v1 import validate_gpu_protocol

        validate_gpu_protocol(("train", "validation", "calibration"), "cuda")
        with self.assertRaisesRegex(ValueError, "locked_test"):
            validate_gpu_protocol(("train", "locked_test"), "cuda")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            validate_gpu_protocol(("train", "validation"), "cpu")

    def test_move_nested_to_device_preserves_structure_and_metadata(self):
        from run_formal_dual_graph_gpu_train_v1 import move_nested_to_device

        value = {
            "tensor": torch.tensor([1.0]),
            "nested": [torch.tensor([2]), (torch.tensor([3]), "sample")],
            "split": "train",
        }
        moved = move_nested_to_device(value, torch.device("cpu"))

        self.assertEqual("cpu", moved["tensor"].device.type)
        self.assertEqual("cpu", moved["nested"][0].device.type)
        self.assertEqual("cpu", moved["nested"][1][0].device.type)
        self.assertEqual("sample", moved["nested"][1][1])
        self.assertEqual("train", moved["split"])

    def test_device_agnostic_core_writes_reloadable_auditable_artifacts(self):
        from run_formal_dual_graph_gpu_train_v1 import run_formal_training

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            output_dir = root / "output"
            tensor_root.mkdir()
            _write_three_split_formal_fixture(tensor_root)

            result = run_formal_training(
                tensor_root=tensor_root,
                output_dir=output_dir,
                device="cpu",
                learned_methods=("pooled_gru",),
                seed=37,
                train_limit=1,
                evaluation_limit=1,
                hidden_dim=4,
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            )

            self.assertTrue(result["training_run_complete"])
            self.assertFalse(result["locked_test_accessed"])
            self.assertEqual(["zero_activity", "last_persistence", "pooled_gru"], result["completed_methods"])
            config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(["train", "validation", "calibration"], config["splits"])
            self.assertEqual("calibration", config["threshold_selection_split"])
            self.assertEqual("inverse_temperature_v1", config["link_probability_calibration_method"])
            self.assertFalse(config["locked_test_accessed"])
            self.assertEqual("cpu", config["device"])
            self.assertTrue(output_dir.is_dir())
            self.assertEqual([], list(root.glob(f".{output_dir.name}.staging-*")))

            checkpoint_path = output_dir / "checkpoints" / "pooled_gru__best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual("pooled_gru", checkpoint["method"])
            self.assertEqual(1, checkpoint["best_epoch"])
            self.assertIn("model_state_dict", checkpoint)

            for relative in (
                "sample_ids.json",
                "class_weights.json",
                "training_history.json",
                "comparison.csv",
                "runtime.json",
                "run_summary.json",
                "manifest.json",
                "metrics/pooled_gru__link_probability_calibration.json",
            ):
                self.assertTrue((output_dir / relative).is_file(), relative)
            for split in ("validation", "calibration"):
                self.assertTrue((output_dir / "metrics" / f"pooled_gru__{split}.json").is_file())

            sidecar = json.loads(
                (output_dir / "metrics" / "pooled_gru__link_probability_calibration.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("inverse_temperature_v1", sidecar["calibration_method"])
            self.assertEqual("link_activity_logits", sidecar["raw_logit_source"])
            class_weights = json.loads((output_dir / "class_weights.json").read_text(encoding="utf-8"))
            self.assertEqual(class_weights["pos_weight"]["link_activity"], sidecar["pos_weight"])
            self.assertEqual("link_activity", sidecar["pos_weight_source_key"])
            self.assertEqual("calibration", sidecar["temperature_fit_split"])
            self.assertEqual("unweighted_bernoulli_nll", sidecar["temperature_fit_objective"])
            self.assertEqual("calibration", sidecar["threshold_selection_split"])
            self.assertFalse(sidecar["validation_used_for_fit_or_selection"])
            self.assertEqual([0.1, 0.3, 0.5, 0.7, 0.9], sidecar["legacy_raw_thresholds"])
            self.assertEqual(5, len(sidecar["legacy_raw_candidates"]))
            self.assertTrue(sidecar["decision_equivalence_to_legacy"])
            self.assertFalse(sidecar["locked_test_accessed"])
            self.assertFalse(sidecar["gpu_execution"])
            self.assertFalse(sidecar["formal_performance_claim_ready"])
            self.assertEqual(
                str((output_dir / "class_weights.json").resolve()),
                sidecar["pos_weight_source_path"],
            )
            self.assertEqual(
                hashlib.sha256((output_dir / "class_weights.json").read_bytes()).hexdigest(),
                sidecar["pos_weight_source_sha256"],
            )
            self.assertEqual(
                sidecar["selected_probability_threshold"],
                sidecar["mapped_probability_thresholds"][
                    str(sidecar["selected_legacy_raw_threshold"])
                ],
            )

            learned_validation = json.loads(
                (output_dir / "metrics" / "pooled_gru__validation.json").read_text(encoding="utf-8")
            )
            learned_calibration = json.loads(
                (output_dir / "metrics" / "pooled_gru__calibration.json").read_text(encoding="utf-8")
            )
            for report in (learned_validation, learned_calibration):
                self.assertEqual("event_probability", report["threshold_coordinate"]["link_activity"])
                self.assertEqual(
                    sidecar["selected_legacy_raw_threshold"],
                    report["legacy_raw_thresholds"]["link_activity"],
                )
                self.assertEqual(
                    sidecar["selected_probability_threshold"],
                    report["thresholds"]["link_activity"],
                )
                for metric in ("nll", "brier", "ece"):
                    self.assertEqual(
                        "computed",
                        report["horizons"]["overall"]["metrics"][f"event.link_activity.{metric}"]["status"],
                    )

            baseline_validation = json.loads(
                (output_dir / "metrics" / "zero_activity__validation.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(baseline_validation["link_probability_calibration"])
            for method in ("zero_activity", "last_persistence"):
                self.assertFalse(
                    (output_dir / "metrics" / f"{method}__link_probability_calibration.json").exists()
                )
                threshold_report = json.loads(
                    (output_dir / "metrics" / f"{method}__threshold_selection.json").read_text(
                        encoding="utf-8"
                    )
                )
                link_selection = threshold_report["events"]["link_activity"]
                self.assertEqual("legacy_sigmoid_score", link_selection["selected"]["coordinate"])
                self.assertNotIn("probability_threshold", link_selection["selected"])
                for candidate in link_selection["candidates"]:
                    self.assertEqual("legacy_sigmoid_score", candidate["coordinate"])
                    self.assertNotIn("probability_threshold", candidate)
            with (output_dir / "comparison.csv").open(encoding="utf-8", newline="") as handle:
                comparison_rows = {row["method"]: row for row in csv.DictReader(handle)}
            learned_row = comparison_rows["pooled_gru"]
            self.assertEqual(
                str(sidecar["selected_legacy_raw_threshold"]), learned_row["threshold"]
            )
            self.assertEqual(
                str(sidecar["selected_legacy_raw_threshold"]),
                learned_row["link_legacy_raw_threshold"],
            )
            self.assertEqual(
                str(sidecar["selected_probability_threshold"]),
                learned_row["link_probability_threshold"],
            )
            self.assertEqual(str(sidecar["temperature"]), learned_row["link_temperature"])
            self.assertEqual("inverse_temperature_v1", learned_row["link_calibration_method"])
            for method in ("zero_activity", "last_persistence", "pooled_gru"):
                self.assertEqual("legacy_raw_threshold", comparison_rows[method]["threshold_coordinate"])
            for method in ("zero_activity", "last_persistence"):
                self.assertEqual("", comparison_rows[method]["link_calibration_method"])
                self.assertEqual("", comparison_rows[method]["link_temperature"])
                self.assertEqual("", comparison_rows[method]["link_probability_threshold"])

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("PI-JWM-formal-training-manifest-v1", manifest["schema_version"])
            self.assertNotIn(".staging-", json.dumps(manifest, sort_keys=True))
            self.assertIn(
                "metrics/pooled_gru__link_probability_calibration.json", manifest["files"]
            )
            for relative, metadata in manifest["files"].items():
                self.assertNotIn(".staging-", relative)
                payload = (output_dir / relative).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), metadata["sha256"])
                self.assertEqual(len(payload), metadata["bytes"])

    def test_learned_link_calibration_rejects_empty_valid_calibration_samples(self):
        from run_formal_dual_graph_gpu_train_v1 import run_formal_training

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            output_dir = root / "output"
            tensor_root.mkdir()
            _write_three_split_formal_fixture(
                tensor_root,
                calibration_link_activity_observed=False,
            )

            with self.assertRaisesRegex(ValueError, "no valid calibration link samples"):
                run_formal_training(
                    tensor_root=tensor_root,
                    output_dir=output_dir,
                    device="cpu",
                    learned_methods=("pooled_gru",),
                    seed=37,
                    train_limit=1,
                    evaluation_limit=1,
                    hidden_dim=4,
                    epochs=1,
                    batch_size=1,
                    learning_rate=1e-3,
                )
            self.assertFalse((output_dir / "run_summary.json").exists())
            self.assertFalse(
                (output_dir / "metrics" / "pooled_gru__link_probability_calibration.json").exists()
            )
            self.assertEqual([], list(root.glob(f".{output_dir.name}.staging-*")))

    def test_learned_link_calibration_rejects_all_zero_and_all_one_labels(self):
        from run_formal_dual_graph_gpu_train_v1 import run_formal_training

        for label_case in ("all_zero", "all_one"):
            with self.subTest(label_case=label_case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                tensor_root = root / "tensor"
                output_dir = root / "output"
                tensor_root.mkdir()
                _write_three_split_formal_fixture(
                    tensor_root,
                    calibration_link_labels=label_case,
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "must contain both positive and negative",
                ):
                    run_formal_training(
                        tensor_root=tensor_root,
                        output_dir=output_dir,
                        device="cpu",
                        learned_methods=("pooled_gru",),
                        seed=37,
                        train_limit=1,
                        evaluation_limit=1,
                        hidden_dim=4,
                        epochs=1,
                        batch_size=1,
                        learning_rate=1e-3,
                    )
                self.assertFalse(output_dir.exists())
                self.assertEqual([], list(root.glob(f".{output_dir.name}.staging-*")))

    def test_existing_output_dir_is_rejected_without_modifying_it(self):
        from run_formal_dual_graph_gpu_train_v1 import run_formal_training

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            output_dir = root / "output"
            tensor_root.mkdir()
            _write_three_split_formal_fixture(tensor_root)
            output_dir.mkdir()
            marker = output_dir / "existing.txt"
            marker.write_text("preserve me", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "output directory already exists"):
                run_formal_training(
                    tensor_root=tensor_root,
                    output_dir=output_dir,
                    device="cpu",
                    learned_methods=("pooled_gru",),
                    seed=37,
                    train_limit=1,
                    evaluation_limit=1,
                    hidden_dim=4,
                    epochs=1,
                    batch_size=1,
                    learning_rate=1e-3,
                )
            self.assertEqual("preserve me", marker.read_text(encoding="utf-8"))
            self.assertEqual([], list(root.glob(f".{output_dir.name}.staging-*")))

    def test_data_seed_separates_fixed_window_selection_from_model_seed(self):
        from run_formal_dual_graph_gpu_train_v1 import run_formal_training

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            tensor_root.mkdir()
            _write_three_split_formal_fixture(tensor_root)
            outputs = []
            for model_seed in (37, 38):
                output_dir = root / f"output_{model_seed}"
                run_formal_training(
                    tensor_root=tensor_root,
                    output_dir=output_dir,
                    device="cpu",
                    learned_methods=("pooled_gru",),
                    seed=model_seed,
                    data_seed=101,
                    train_limit=1,
                    evaluation_limit=1,
                    hidden_dim=4,
                    epochs=1,
                    batch_size=1,
                    learning_rate=1e-3,
                )
                outputs.append(output_dir)
            sample_ids = [
                json.loads((output / "sample_ids.json").read_text(encoding="utf-8"))
                for output in outputs
            ]
            self.assertEqual(sample_ids[0], sample_ids[1])
            config = json.loads((outputs[0] / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(37, config["seed"])
            self.assertEqual(101, config["data_seed"])

    def test_state_mae_weight_override_is_recorded_and_applied(self):
        from run_formal_dual_graph_gpu_train_v1 import run_formal_training

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            output_dir = root / "output"
            tensor_root.mkdir()
            _write_three_split_formal_fixture(tensor_root)

            run_formal_training(
                tensor_root=tensor_root,
                output_dir=output_dir,
                device="cpu",
                learned_methods=("pooled_gru",),
                seed=47,
                train_limit=1,
                evaluation_limit=1,
                hidden_dim=4,
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
                state_mae_weight=0.5,
            )

            config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(0.5, config["loss_weights"]["state_mae"])

    def test_residual_method_maps_to_base_mode_and_writes_system_comparison_columns(self):
        from run_formal_dual_graph_gpu_train_v1 import run_formal_training

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            output_dir = root / "output"
            tensor_root.mkdir()
            _write_three_split_formal_fixture(tensor_root)

            result = run_formal_training(
                tensor_root=tensor_root,
                output_dir=output_dir,
                device="cpu",
                learned_methods=("independent_dual_gnn_residual",),
                seed=41,
                train_limit=1,
                evaluation_limit=1,
                hidden_dim=4,
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            )

            self.assertIn("independent_dual_gnn_residual", result["completed_methods"])
            checkpoint = torch.load(
                output_dir / "checkpoints" / "independent_dual_gnn_residual__best.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual("independent_dual_gnn", checkpoint["model_config"]["mode"])
            self.assertTrue(checkpoint["model_config"]["residual_state_prediction"])
            registry = json.loads((output_dir / "method_registry.json").read_text(encoding="utf-8"))
            self.assertTrue(registry["independent_dual_gnn_residual"]["residual_state_prediction"])
            with (output_dir / "comparison.csv").open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            for name in (
                "validation_throughput_mae",
                "validation_completion_rate_error",
                "validation_rb_occupancy_mae",
                "validation_task_delay_mae",
                "validation_task_deadline_mae",
                "validation_lifecycle_macro_f1",
                "validation_dag_unfinished_parent_mae",
                "calibration_throughput_mae",
                "calibration_task_delay_mae",
            ):
                self.assertIn(name, row)

    def test_system_sidecar_training_enables_energy_head_and_records_contract(self):
        from run_formal_dual_graph_gpu_train_v1 import run_formal_training

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            system_root = root / "system"
            output_dir = root / "output"
            tensor_root.mkdir()
            system_root.mkdir()
            _write_three_split_formal_fixture(tensor_root)
            _write_system_fixture(system_root)
            shutil.copytree(system_root / "seed_001", system_root / "seed_002")
            system_report_path = system_root / "seed_002" / "system_target_report.json"
            system_report = json.loads(system_report_path.read_text(encoding="utf-8"))
            system_report.update(seed=2, split="calibration")
            system_report_path.write_text(json.dumps(system_report), encoding="utf-8")

            result = run_formal_training(
                tensor_root=tensor_root,
                system_root=system_root,
                use_system_energy_head=True,
                output_dir=output_dir,
                device="cpu",
                learned_methods=("pooled_gru",),
                seed=43,
                train_limit=1,
                evaluation_limit=1,
                hidden_dim=4,
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
            )

            self.assertTrue(result["training_run_complete"])
            config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            self.assertTrue(config["use_system_energy_head"])
            self.assertEqual(str(system_root.resolve()), config["system_root"])
            checkpoint = torch.load(
                output_dir / "checkpoints" / "pooled_gru__best.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertTrue(checkpoint["model_config"]["use_system_energy_head"])
            self.assertTrue(
                any(name.startswith("uav_energy_head") for name in checkpoint["model_state_dict"])
            )


if __name__ == "__main__":
    unittest.main()
