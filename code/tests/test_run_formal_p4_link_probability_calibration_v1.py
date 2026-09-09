from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np
from torch.utils.data import DataLoader


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
for path in (SCRIPTS_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.formal_dual_graph_world_model_v1 import (
    FormalDualGraphWorldModel,
    FormalWorldModelConfig,
)
from pi_jwm.formal_airfogsim_window_v1 import FormalAirFogSimWindowDataset, FormalWindowConfig
from test_formal_airfogsim_window_v1 import _write_formal_fixture


def _provenance(**overrides):
    value = {
        "source_split": "train-only",
        "checkpoint_sha256": "checkpoint-sha",
        "tensor_manifest_sha256": "tensor-sha",
        "sample_ids": {
            "calibration": "calibration-sample-ids-sha",
            "validation": "validation-sample-ids-sha",
        },
        "pos_weight_source_path": "class_weights.json",
        "pos_weight_source_sha256": "class-weight-sha",
        "mask_counts": {"calibration": 40, "validation": 40},
        "locked_test_accessed": False,
        "gpu_execution": False,
        "formal_performance_claim_ready": False,
    }
    value.update(overrides)
    return value


def _inputs():
    # The 0.9 raw candidate is the unique calibration F1 winner.  Validation
    # mirrors the frozen distribution but is never used for fit or selection.
    logits = torch.full((2, 20, 1), 1.0, dtype=torch.float64)
    labels = torch.zeros((2, 20, 1), dtype=torch.int64)
    logits[:, :, 0] = torch.tensor(
        [[1.0] * 19 + [3.0], [1.0] * 19 + [2.5]], dtype=torch.float64
    )
    labels[:, :, 0] = torch.tensor(
        [[0] * 19 + [1], [0] * 19 + [1]], dtype=torch.int64
    )
    validation_logits = logits.clone()
    validation_labels = torch.zeros_like(labels)
    return logits, labels, validation_logits, validation_labels


def _validation_brier_regression_inputs():
    calibration_logits, calibration_labels, validation_logits, validation_labels = _inputs()
    validation_logits[0, 19, 0] = 3.0
    validation_labels[0, 19, 0] = 1
    validation_labels[1, 19, 0] = 1
    return calibration_logits, calibration_labels, validation_logits, validation_labels


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest(root: Path, names: list[str]) -> dict:
    return {
        "schema_version": "PI-JWM-AirFogSim-formal-tensor-manifest-v1",
        "files": {
            name: {"size_bytes": (root / name).stat().st_size, "sha256": _sha256(root / name)}
            for name in names
        },
    }


def _rule_stats() -> dict:
    return {
        "source_split": "train",
        "features": {
            "node_state": {"mean": [0.0] * 7, "scale": [1.0] * 7},
            "physical_edge_state": {"mean": [0.0] * 5, "scale": [1.0] * 5},
            "flow_state": {"mean": [0.0] * 5, "scale": [1.0] * 5},
            "task_state": {"mean": [0.0] * 8, "scale": [1.0] * 8},
        },
    }


def _write_runner_fixture(root: Path, *, seed: int = 20260831) -> tuple[Path, Path, Path]:
    tensor_root, run_root, output_dir = root / "tensor", root / "run", root / "output"
    tensor_root.mkdir()
    contract = {"history_steps": 8, "horizon_steps": 20, "n_rb": 50}
    _write_json(tensor_root / "tensor_contract.json", contract)
    _write_json(tensor_root / "normalization_stats.json", _rule_stats())
    (tensor_root / "window_index.csv").write_text(
        "sample_id,seed,split,input_start_index,input_end_index,label_start_index,label_end_index\n"
        "train-0,0,train,0,8,8,28\n"
        "calibration-0,1,calibration,0,8,8,28\n"
        "validation-0,2,validation,0,8,8,28\n",
        encoding="utf-8",
    )
    for tensor_seed in (1, 2):
        tensor_path = tensor_root / f"seed_{tensor_seed:03d}" / "trajectory_tensors.npz"
        tensor_path.parent.mkdir()
        tensor_path.write_bytes(f"fixture-seed-{tensor_seed}".encode("ascii"))
    _write_json(
        tensor_root / "manifest.json",
        _manifest(tensor_root, [
            "tensor_contract.json", "normalization_stats.json", "window_index.csv",
            "seed_001/trajectory_tensors.npz", "seed_002/trajectory_tensors.npz",
        ]),
    )
    run_root.mkdir()
    model_config = FormalWorldModelConfig(
        mode="coupled_dual_gnn",
        hidden_dim=4,
        history_steps=8,
        horizon_steps=20,
        residual_state_prediction=True,
        deterministic_rule_layer=True,
        rule_layer_stats=_rule_stats(),
        n_rb=50,
    )
    model = FormalDualGraphWorldModel(model_config)
    checkpoint_path = run_root / "checkpoints" / "coupled_dual_gnn_residual__best.pt"
    checkpoint_path.parent.mkdir()
    torch.save(
        {"method": "coupled_dual_gnn_residual", "model_config": asdict(model_config), "model_state_dict": model.state_dict()},
        checkpoint_path,
    )
    _write_json(
        run_root / "config.json",
        {
            "seed": seed,
            "locked_test_accessed": False,
            "dataset_manifest_sha256": _sha256(tensor_root / "manifest.json"),
            "history_steps": 8,
            "horizon_steps": 20,
            "deterministic_rule_layer": True,
            "learned_methods": ["coupled_dual_gnn_residual"],
        },
    )
    _write_json(
        run_root / "sample_ids.json",
        {"train": ["train-0"], "calibration": ["calibration-0"], "validation": ["validation-0"]},
    )
    _write_json(run_root / "class_weights.json", {"source_split": "train", "pos_weight": {"link_activity": 50.0}})
    _write_json(
        run_root / "manifest.json",
        _manifest(run_root, ["config.json", "sample_ids.json", "class_weights.json", "checkpoints/coupled_dual_gnn_residual__best.pt"]),
    )
    return tensor_root, run_root, output_dir


def _write_persistence_runner_fixture(root: Path) -> tuple[Path, Path, Path]:
    tensor_root, run_root, output_dir = _write_runner_fixture(root)
    old_checkpoint = run_root / "checkpoints" / "coupled_dual_gnn_residual__best.pt"
    checkpoint = torch.load(old_checkpoint, map_location="cpu", weights_only=True)
    checkpoint["method"] = "link_activity_persistence_residual_v1"
    checkpoint["model_config"].update(
        {
            "link_activity_method": "persistence_residual_v1",
            "link_activity_pos_weight": 50.0,
            "link_activity_missing_history_prior": 0.25,
        }
    )
    new_checkpoint = run_root / "checkpoints" / "link_activity_persistence_residual_v1__best.pt"
    torch.save(checkpoint, new_checkpoint)
    old_checkpoint.unlink()
    config = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    config["learned_methods"] = ["link_activity_persistence_residual_v1"]
    config["link_activity_methods"] = {
        "link_activity_persistence_residual_v1": "persistence_residual_v1"
    }
    _write_json(run_root / "config.json", config)
    _write_json(
        run_root / "metrics" / "link_activity_persistence_residual_v1__threshold_selection.json",
        {
            "selection_split": "calibration",
            "events": {
                "link_activity": {
                    "candidates": [
                        {"raw_threshold": value, "coordinate": "raw_weighted_score"}
                        for value in (0.1, 0.3, 0.5, 0.7, 0.9)
                    ],
                    "selected": {"raw_threshold": 0.5, "coordinate": "raw_weighted_score"},
                }
            },
        },
    )
    _write_json(
        run_root / "manifest.json",
        _manifest(
            run_root,
            [
                "config.json",
                "sample_ids.json",
                "class_weights.json",
                "checkpoints/link_activity_persistence_residual_v1__best.pt",
                "metrics/link_activity_persistence_residual_v1__threshold_selection.json",
            ],
        ),
    )
    return tensor_root, run_root, output_dir


class LinkProbabilityCalibrationAuditTests(unittest.TestCase):
    def test_resolves_new_method_threshold_from_calibration_selection(self):
        from run_formal_p4_link_probability_calibration_v1 import _resolve_frozen_raw_threshold

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "threshold_selection.json"
            _write_json(
                path,
                {
                    "selection_split": "calibration",
                    "events": {
                        "link_activity": {
                            "candidates": [
                                {"raw_threshold": value, "coordinate": "raw_weighted_score"}
                                for value in (0.1, 0.3, 0.5, 0.7, 0.9)
                            ],
                            "selected": {
                                "raw_threshold": 0.5,
                                "coordinate": "raw_weighted_score",
                            },
                        }
                    },
                },
            )
            self.assertEqual(
                _resolve_frozen_raw_threshold(
                    "link_activity_persistence_residual_v1", path
                ),
                0.5,
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["events"]["link_activity"]["candidates"] = payload["events"]["link_activity"]["candidates"][:-1]
            _write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "candidate set is incomplete"):
                _resolve_frozen_raw_threshold(
                    "link_activity_persistence_residual_v1", path
                )

    def test_new_method_enters_runner_with_its_selected_threshold(self):
        from run_formal_p4_link_probability_calibration_v1 import (
            run_formal_p4_link_probability_calibration,
        )

        calibration_logits, calibration_labels, validation_logits, validation_labels = _inputs()
        collected = [
            (
                [calibration_logits[:, k].reshape(-1) for k in range(20)],
                [calibration_labels[:, k].reshape(-1) for k in range(20)],
                40,
            ),
            (
                [validation_logits[:, k].reshape(-1) for k in range(20)],
                [validation_labels[:, k].reshape(-1) for k in range(20)],
                40,
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root, run_root, output_dir = _write_persistence_runner_fixture(root)
            with patch(
                "run_formal_p4_link_probability_calibration_v1._collect_split",
                side_effect=collected,
            ), patch(
                "run_formal_p4_link_probability_calibration_v1.build_link_probability_calibration_audit",
                return_value={"schema_version": "test"},
            ) as build_audit:
                run_formal_p4_link_probability_calibration(
                    run_root=run_root,
                    tensor_root=tensor_root,
                    method="link_activity_persistence_residual_v1",
                    output_dir=output_dir,
                )
            self.assertEqual(build_audit.call_args.kwargs["legacy_selected_raw_threshold"], 0.5)

    def test_builds_frozen_schema_and_keeps_validation_out_of_fit_and_selection(self):
        from run_formal_p4_link_probability_calibration_v1 import (
            build_link_probability_calibration_audit,
        )

        audit = build_link_probability_calibration_audit(*_inputs(), provenance=_provenance())

        self.assertEqual(audit["schema_version"], "PI-JWM-link-probability-calibration-v1")
        self.assertEqual(audit["calibration_method"], "inverse_temperature_v1")
        self.assertIsInstance(audit["log_temperature"], float)
        self.assertEqual(audit["temperature_fit_split"], "calibration")
        self.assertEqual(audit["threshold_selection_split"], "calibration")
        self.assertFalse(audit["validation_used_for_fit_or_selection"])
        self.assertEqual(audit["selected_legacy_raw_threshold"], 0.9)
        self.assertEqual(audit["legacy_raw_thresholds"], [0.1, 0.3, 0.5, 0.7, 0.9])
        self.assertFalse(audit["formal_performance_claim_ready"])
        self.assertFalse(audit["gpu_execution"])
        self.assertFalse(audit["locked_test_accessed"])
        self.assertEqual(audit["probability_metrics"]["calibration"]["overall"]["identity"]["count"], 40)
        self.assertEqual(audit["probability_metrics"]["validation"]["k=20"]["fitted"]["count"], 2)

    def test_preserves_all_legacy_decisions_and_selected_classification_and_ranking(self):
        from run_formal_p4_link_probability_calibration_v1 import (
            build_link_probability_calibration_audit,
        )

        audit = build_link_probability_calibration_audit(*_inputs(), provenance=_provenance())

        self.assertTrue(audit["decision_equivalence_to_legacy"])
        self.assertEqual(
            [item["raw_threshold"] for item in audit["legacy_raw_candidates"]],
            [0.1, 0.3, 0.5, 0.7, 0.9],
        )
        self.assertTrue(audit["classification_decision_reuses_legacy_raw"])
        self.assertTrue(audit["mapped_thresholds_strictly_monotonic"])
        self.assertTrue(all(item["direct_probability_threshold_decision_equivalent"] for item in audit["legacy_raw_candidates"]))
        self.assertTrue(all(item["formal_classification_decision_equivalent_to_legacy"] for item in audit["legacy_raw_candidates"]))
        for split in ("calibration", "validation"):
            self.assertEqual(
                audit["direct_probability_threshold_comparison_equivalence_by_split"][split],
                {"0.1": True, "0.3": True, "0.5": True, "0.7": True, "0.9": True},
            )
        for split in ("calibration", "validation"):
            comparison = audit["classification"][split]
            self.assertEqual(comparison["legacy_raw"]["counts"], comparison["fitted"]["counts"])
            self.assertEqual(comparison["legacy_raw"]["auprc"], comparison["fitted"]["auprc"])
            for name in ("precision", "recall", "f1"):
                self.assertEqual(comparison["legacy_raw"][name], comparison["fitted"][name])

    def test_identity_and_fitted_probability_metrics_are_separate_for_all_required_horizons(self):
        from run_formal_p4_link_probability_calibration_v1 import (
            build_link_probability_calibration_audit,
        )

        audit = build_link_probability_calibration_audit(*_inputs(), provenance=_provenance())

        for split in ("calibration", "validation"):
            for bucket in ("overall", "k=1", "k=5", "k=10", "k=20"):
                metrics = audit["probability_metrics"][split][bucket]
                for mode in ("identity", "fitted"):
                    self.assertEqual(set(("nll", "brier", "ece")), set(metrics[mode]) & {"nll", "brier", "ece"})
                    self.assertGreater(metrics[mode]["count"], 0)
        self.assertLessEqual(
            audit["probability_metrics"]["calibration"]["overall"]["fitted"]["nll"],
            audit["probability_metrics"]["calibration"]["overall"]["identity"]["nll"] + 1e-12,
        )

    def test_rejects_validation_probability_metric_regression_and_nonzero_mask_drift(self):
        from run_formal_p4_link_probability_calibration_v1 import (
            build_link_probability_calibration_audit,
        )

        with self.assertRaisesRegex(ValueError, "validation.*brier"):
            build_link_probability_calibration_audit(
                *_validation_brier_regression_inputs(), provenance=_provenance()
            )
        with self.assertRaisesRegex(ValueError, "mask_counts"):
            build_link_probability_calibration_audit(
                *_inputs(), provenance=_provenance(mask_counts={"calibration": 999, "validation": 888})
            )

    def test_float32_legacy_boundary_decision_matches_formal_raw_sigmoid(self):
        from run_formal_p4_link_probability_calibration_v1 import (
            build_link_probability_calibration_audit,
        )
        from pi_jwm.formal_binary_calibration_v1 import InverseTemperatureCalibration

        logits, labels, validation_logits, validation_labels = _inputs()
        boundary = torch.logit(torch.tensor(0.9, dtype=torch.float32))
        logits = logits.float()
        logits[0, 0, 0] = boundary
        audit = build_link_probability_calibration_audit(
            logits, labels, validation_logits, validation_labels, provenance=_provenance()
        )
        expected = torch.sigmoid(logits.reshape(-1)) >= 0.9
        self.assertTrue(expected[0].item())
        self.assertEqual(
            audit["classification"]["calibration"]["legacy_raw"]["counts"]["fp"],
            int((expected & ~labels.reshape(-1).bool()).sum().item()),
        )
        raw_scores = torch.sigmoid(logits.reshape(-1))
        truth = labels.reshape(-1).bool()
        calibration = InverseTemperatureCalibration(audit["pos_weight"], audit["log_temperature"])
        for candidate in audit["legacy_raw_candidates"]:
            decision = raw_scores >= candidate["raw_threshold"]
            self.assertEqual(candidate["tp"], int((decision & truth).sum().item()))
            self.assertEqual(candidate["fp"], int((decision & ~truth).sum().item()))
            self.assertEqual(candidate["fn"], int((~decision & truth).sum().item()))
            self.assertEqual(
                candidate["probability_threshold"],
                calibration.map_raw_threshold(candidate["raw_threshold"]),
            )
        self.assertEqual(
            audit["mapped_probability_thresholds"]["0.9"], calibration.map_raw_threshold(0.9)
        )
        self.assertEqual(audit["selected_probability_threshold"], calibration.map_raw_threshold(0.9))
        self.assertFalse(
            audit["direct_probability_threshold_comparison_equivalence_by_split"]["calibration"]["0.9"]
        )
        selected = next(row for row in audit["legacy_raw_candidates"] if row["raw_threshold"] == 0.9)
        self.assertFalse(selected["direct_probability_threshold_decision_equivalent"])
        self.assertTrue(selected["formal_classification_decision_equivalent_to_legacy"])
        self.assertTrue(audit["classification_decision_reuses_legacy_raw"])

    def test_real_runner_publishes_manifest_and_rejects_frozen_identity_drift(self):
        from run_formal_p4_link_probability_calibration_v1 import (
            run_formal_p4_link_probability_calibration,
        )

        calibration_logits, calibration_labels, validation_logits, validation_labels = _inputs()
        collected = [
            ([calibration_logits[:, k].reshape(-1) for k in range(20)], [calibration_labels[:, k].reshape(-1) for k in range(20)], 40),
            ([validation_logits[:, k].reshape(-1) for k in range(20)], [validation_labels[:, k].reshape(-1) for k in range(20)], 40),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root, run_root, output_dir = _write_runner_fixture(root)
            with patch("run_formal_p4_link_probability_calibration_v1._collect_split", side_effect=collected):
                audit = run_formal_p4_link_probability_calibration(
                    run_root=run_root, tensor_root=tensor_root,
                    method="coupled_dual_gnn_residual", output_dir=output_dir,
                )
            self.assertEqual(audit["sentinel_performance_gate"], "no_go")
            self.assertEqual(audit["p4_status"], "blocked")
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            entry = manifest["files"]["link_probability_calibration_audit.json"]
            artifact = output_dir / "link_probability_calibration_audit.json"
            self.assertEqual(entry["size_bytes"], artifact.stat().st_size)
            self.assertEqual(entry["sha256"], _sha256(artifact))
            with self.assertRaises(FileExistsError):
                run_formal_p4_link_probability_calibration(
                    run_root=run_root, tensor_root=tensor_root,
                    method="coupled_dual_gnn_residual", output_dir=output_dir,
                )
            self.assertFalse(list(root.glob(".output.staging-*")))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root, run_root, output_dir = _write_runner_fixture(root, seed=7)
            with self.assertRaisesRegex(ValueError, "seed"):
                run_formal_p4_link_probability_calibration(
                    run_root=run_root, tensor_root=tensor_root,
                    method="coupled_dual_gnn_residual", output_dir=output_dir,
                )
            self.assertFalse(output_dir.exists())

        for mutation in ("schema", "size_bytes", "sha256"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                tensor_root, run_root, output_dir = _write_runner_fixture(root)
                manifest_path = tensor_root / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if mutation == "schema":
                    manifest["schema_version"] = "wrong-schema"
                elif mutation == "size_bytes":
                    manifest["files"]["normalization_stats.json"]["size_bytes"] += 1
                else:
                    manifest["files"]["normalization_stats.json"]["sha256"] = "0" * 64
                _write_json(manifest_path, manifest)
                config_path = run_root / "config.json"
                config = json.loads(config_path.read_text(encoding="utf-8"))
                config["dataset_manifest_sha256"] = _sha256(manifest_path)
                _write_json(config_path, config)
                _write_json(
                    run_root / "manifest.json",
                    _manifest(run_root, ["config.json", "sample_ids.json", "class_weights.json", "checkpoints/coupled_dual_gnn_residual__best.pt"]),
                )
                with self.assertRaises(ValueError):
                    run_formal_p4_link_probability_calibration(
                        run_root=run_root, tensor_root=tensor_root,
                        method="coupled_dual_gnn_residual", output_dir=output_dir,
                    )
                self.assertFalse(output_dir.exists())
            self.assertFalse(list(root.glob(".output.staging-*")))

    def test_runner_rejects_missing_stats_sample_ids_and_manifest_bound_input_drift(self):
        from run_formal_p4_link_probability_calibration_v1 import (
            run_formal_p4_link_probability_calibration,
        )

        cases = (
            ("class_weight_hash", "class_weights.json"),
            ("checkpoint_hash", "checkpoints/coupled_dual_gnn_residual__best.pt"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root, run_root, output_dir = _write_runner_fixture(root)
            (tensor_root / "normalization_stats.json").unlink()
            with self.assertRaisesRegex(ValueError, "frozen run or tensor inputs are incomplete"):
                run_formal_p4_link_probability_calibration(
                    run_root=run_root, tensor_root=tensor_root,
                    method="coupled_dual_gnn_residual", output_dir=output_dir,
                )
            self.assertFalse(output_dir.exists())

        for name, relative in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                tensor_root, run_root, output_dir = _write_runner_fixture(root)
                path = run_root / relative
                if name == "class_weight_hash":
                    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
                else:
                    with path.open("ab") as handle:
                        handle.write(b"drift")
                with self.assertRaisesRegex(ValueError, "frozen run manifest identity drift"):
                    run_formal_p4_link_probability_calibration(
                        run_root=run_root, tensor_root=tensor_root,
                        method="coupled_dual_gnn_residual", output_dir=output_dir,
                    )
                self.assertFalse(output_dir.exists())
                self.assertFalse(list(root.glob(".output.staging-*")))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root, run_root, output_dir = _write_runner_fixture(root)
            stats = tensor_root / "normalization_stats.json"
            stats.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen tensor input identity drift"):
                run_formal_p4_link_probability_calibration(
                    run_root=run_root, tensor_root=tensor_root,
                    method="coupled_dual_gnn_residual", output_dir=output_dir,
                )
            self.assertFalse(output_dir.exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root, run_root, output_dir = _write_runner_fixture(root)
            sample_ids_path = run_root / "sample_ids.json"
            sample_ids = json.loads(sample_ids_path.read_text(encoding="utf-8"))
            sample_ids["calibration"] = ["not-present"]
            _write_json(sample_ids_path, sample_ids)
            _write_json(
                run_root / "manifest.json",
                _manifest(run_root, ["config.json", "sample_ids.json", "class_weights.json", "checkpoints/coupled_dual_gnn_residual__best.pt"]),
            )
            with self.assertRaisesRegex(ValueError, "selected sample IDs are missing"):
                run_formal_p4_link_probability_calibration(
                    run_root=run_root, tensor_root=tensor_root,
                    method="coupled_dual_gnn_residual", output_dir=output_dir,
                )
            self.assertFalse(output_dir.exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root, run_root, output_dir = _write_runner_fixture(root)
            checkpoint_path = run_root / "checkpoints" / "coupled_dual_gnn_residual__best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            checkpoint["model_config"]["mode"] = "pooled_gru"
            torch.save(checkpoint, checkpoint_path)
            _write_json(
                run_root / "manifest.json",
                _manifest(run_root, ["config.json", "sample_ids.json", "class_weights.json", "checkpoints/coupled_dual_gnn_residual__best.pt"]),
            )
            with self.assertRaisesRegex(ValueError, "model_config semantic identity drift"):
                run_formal_p4_link_probability_calibration(
                    run_root=run_root, tensor_root=tensor_root,
                    method="coupled_dual_gnn_residual", output_dir=output_dir,
                )
            self.assertFalse(output_dir.exists())

    def test_runner_rejects_canonical_manifest_rebinding_and_strict_state_dict_drift(self):
        from run_formal_p4_link_probability_calibration_v1 import (
            run_formal_p4_link_probability_calibration,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root, run_root, output_dir = _write_runner_fixture(root)
            expected_manifest_sha = _sha256(tensor_root / "manifest.json")
            window = tensor_root / "window_index.csv"
            window.write_text(window.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            _write_json(
                tensor_root / "manifest.json",
                _manifest(tensor_root, ["tensor_contract.json", "normalization_stats.json", "window_index.csv"]),
            )
            config = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
            config["dataset_manifest_sha256"] = _sha256(tensor_root / "manifest.json")
            _write_json(run_root / "config.json", config)
            _write_json(
                run_root / "manifest.json",
                _manifest(run_root, ["config.json", "sample_ids.json", "class_weights.json", "checkpoints/coupled_dual_gnn_residual__best.pt"]),
            )
            with self.assertRaisesRegex(ValueError, "canonical tensor manifest"):
                run_formal_p4_link_probability_calibration(
                    run_root=run_root, tensor_root=tensor_root, method="coupled_dual_gnn_residual",
                    output_dir=output_dir, _expected_tensor_manifest_sha256=expected_manifest_sha,
                )
            self.assertFalse(output_dir.exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root, run_root, output_dir = _write_runner_fixture(root)
            expected_manifest_sha = _sha256(tensor_root / "manifest.json")
            tensor_path = tensor_root / "seed_001" / "trajectory_tensors.npz"
            tensor_path.write_bytes(tensor_path.read_bytes() + b"-drift")
            _write_json(
                tensor_root / "manifest.json",
                _manifest(tensor_root, [
                    "tensor_contract.json", "normalization_stats.json", "window_index.csv",
                    "seed_001/trajectory_tensors.npz", "seed_002/trajectory_tensors.npz",
                ]),
            )
            config = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
            config["dataset_manifest_sha256"] = _sha256(tensor_root / "manifest.json")
            _write_json(run_root / "config.json", config)
            _write_json(
                run_root / "manifest.json",
                _manifest(run_root, ["config.json", "sample_ids.json", "class_weights.json", "checkpoints/coupled_dual_gnn_residual__best.pt"]),
            )
            with self.assertRaisesRegex(ValueError, "canonical tensor manifest"):
                run_formal_p4_link_probability_calibration(
                    run_root=run_root, tensor_root=tensor_root, method="coupled_dual_gnn_residual",
                    output_dir=output_dir, _expected_tensor_manifest_sha256=expected_manifest_sha,
                )
            self.assertFalse(output_dir.exists())

        for mutation in ("missing", "unexpected"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                tensor_root, run_root, output_dir = _write_runner_fixture(root)
                checkpoint_path = run_root / "checkpoints" / "coupled_dual_gnn_residual__best.pt"
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if mutation == "missing":
                    checkpoint["model_state_dict"].pop(next(iter(checkpoint["model_state_dict"])))
                else:
                    checkpoint["model_state_dict"]["unexpected.weight"] = torch.zeros(1)
                torch.save(checkpoint, checkpoint_path)
                _write_json(
                    run_root / "manifest.json",
                    _manifest(run_root, ["config.json", "sample_ids.json", "class_weights.json", "checkpoints/coupled_dual_gnn_residual__best.pt"]),
                )
                with self.assertRaises(RuntimeError):
                    run_formal_p4_link_probability_calibration(
                        run_root=run_root, tensor_root=tensor_root, method="coupled_dual_gnn_residual",
                        output_dir=output_dir,
                        _expected_tensor_manifest_sha256=_sha256(tensor_root / "manifest.json"),
                    )
                self.assertFalse(output_dir.exists())
                self.assertFalse(list(root.glob(".output.staging-*")))

    def test_collect_split_uses_real_dataset_valid_edge_and_aggregate_activity_mask(self):
        from run_formal_p4_link_probability_calibration_v1 import _collect_split

        with tempfile.TemporaryDirectory() as temporary:
            tensor_root = Path(temporary)
            _write_formal_fixture(tensor_root)
            tensor_path = tensor_root / "seed_001" / "trajectory_tensors.npz"
            with np.load(tensor_path, allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            feature_mask = np.ones_like(arrays["physical_edge_state"], dtype=bool)
            feature_mask[2, 0, 3] = False
            arrays["physical_edge_feature_mask"] = feature_mask
            np.savez_compressed(tensor_path, **arrays)
            dataset = FormalAirFogSimWindowDataset(
                tensor_root, split="validation", config=FormalWindowConfig(history_steps=2, horizon_steps=2)
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            batch = next(iter(loader))
            labels = batch["target"].get("aggregate_link_activity", batch["target"]["link_activity"])
            expected_valid = (
                torch.all(batch["static"]["physical_edge_endpoint_index"] >= 0, dim=-1)[:, None, :].expand_as(labels)
                & batch["target"]["aggregate_link_activity_mask"].bool()
            )
            self.assertFalse(expected_valid[0, 0, 0].item())
            self.assertFalse(expected_valid[0, :, 1].any().item())
            model = FormalDualGraphWorldModel(FormalWorldModelConfig(
                mode="coupled_dual_gnn", hidden_dim=4, history_steps=2, horizon_steps=2,
            ))
            logits, labels_out, mask_count = _collect_split(model, loader)
            self.assertEqual(mask_count, int(expected_valid.sum().item()))
            self.assertEqual(sum(value.numel() for value in logits), mask_count)
            self.assertEqual(sum(value.numel() for value in labels_out), mask_count)

    def test_owned_staging_race_does_not_overwrite_target(self):
        from run_formal_p4_link_probability_calibration_v1 import _publish_calibration_audit_atomically

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "output"
            def create_competing_target():
                output_dir.mkdir()
                (output_dir / "owner.txt").write_text("other", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _publish_calibration_audit_atomically(
                    output_dir, {"schema_version": "test"}, before_rename=create_competing_target
                )
            self.assertEqual((output_dir / "owner.txt").read_text(encoding="utf-8"), "other")
            self.assertFalse(list(root.glob(".output.staging-*")))

    def test_rejects_provenance_drift_empty_masks_nonfinite_logits_and_locked_test(self):
        from run_formal_p4_link_probability_calibration_v1 import (
            build_link_probability_calibration_audit,
        )

        calibration_logits, calibration_labels, validation_logits, validation_labels = _inputs()
        invalid_cases = (
            (_provenance(source_split="validation"), calibration_logits),
            (_provenance(mask_counts={"calibration": 0, "validation": 40}), calibration_logits),
            (_provenance(locked_test_accessed=True), calibration_logits),
            (_provenance(), torch.full_like(calibration_logits, float("nan"))),
            (_provenance(), torch.empty((0,), dtype=torch.float64)),
        )
        for provenance, logits in invalid_cases:
            with self.subTest(provenance=provenance, shape=tuple(logits.shape)):
                with self.assertRaises(ValueError):
                    build_link_probability_calibration_audit(
                        logits,
                        calibration_labels if logits.numel() else torch.empty((0,), dtype=torch.int64),
                        validation_logits,
                        validation_labels,
                        provenance=provenance,
                    )


if __name__ == "__main__":
    unittest.main()
