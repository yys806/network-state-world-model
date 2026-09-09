from __future__ import annotations

import math
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


def _provenance(*, mask_counts: dict[str, int] | None = None, **overrides: object) -> dict[str, object]:
    counts = mask_counts or {**{f"h{index}": 4 for index in range(1, 21)}, "total": 80}
    result: dict[str, object] = {
        "checkpoint_sha256": "a" * 64,
        "tensor_manifest_sha256": "b" * 64,
        "sample_ids_sha256": "c" * 64,
        "class_weights_path": "class_weights.json",
        "class_weights_sha256": "d" * 64,
        "class_weights_source_split": "train",
        "link_activity_pos_weight": 50.0,
        "mask_counts": counts,
        "checkpoint_reload": {"strict": True, "missing_keys": 0, "unexpected_keys": 0},
        "locked_test_accessed": False,
        "gpu_execution": False,
        "formal_performance_claim_ready": False,
    }
    result.update(overrides)
    return result


def _inputs() -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    # h1 has one continued positive, one newly active positive, one unobserved positive,
    # and one negative.  The remaining horizons preserve the same aligned shape.
    logits = [torch.tensor([3.0, -3.0, 3.0, -3.0], dtype=torch.float64) for _ in range(20)]
    labels = [torch.tensor([1, 1, 1, 0], dtype=torch.int64) for _ in range(20)]
    previous = [torch.tensor([1, 0, -1, 0], dtype=torch.int64) for _ in range(20)]
    persistence = [torch.tensor([1, 0, 1, 0], dtype=torch.int64) for _ in range(20)]
    return logits, labels, previous, persistence


class LinkRecallDiagnosisV1Tests(unittest.TestCase):
    def test_groups_recall_and_probability_quantiles_follow_frozen_raw_threshold(self):
        from run_formal_p4_link_recall_diagnosis_v1 import build_link_recall_diagnosis

        report = build_link_recall_diagnosis(*_inputs(), provenance=_provenance())

        self.assertEqual(report["analysis_split"], "validation")
        self.assertEqual(report["raw_threshold"], 0.9)
        self.assertEqual(report["pos_weight"], 50.0)
        self.assertFalse(report["temperature_fitted"])
        self.assertEqual(report["by_horizon"].keys(), {f"h{index}" for index in range(1, 21)})
        h1 = report["by_horizon"]["h1"]
        self.assertEqual(h1["continued_active"]["positive_count"], 1)
        self.assertEqual(h1["newly_active"]["candidate"]["fn"], 1)
        self.assertEqual(h1["newly_active"]["persistence"]["fn"], 1)
        self.assertEqual(h1["continued_active"]["persistence_tp_candidate_fn"], 0)
        self.assertEqual(h1["all_positive"]["candidate"]["tp"], 2)
        self.assertEqual(h1["all_positive"]["candidate"]["fn"], 1)
        self.assertEqual(h1["all_positive"]["persistence"]["tp"], 2)
        self.assertEqual(h1["previous_unobserved_positive_count"], 1)
        stats = h1["all_positive"]["candidate"]["statistics"]
        self.assertEqual(stats["tp"]["raw_logit"]["q0"], 3.0)
        self.assertAlmostEqual(stats["fn"]["raw_weighted_score"]["q50"], 1.0 / (1.0 + math.exp(3.0)))
        self.assertLess(stats["tp"]["weight_corrected_probability"]["q50"], 0.5)

    def test_float32_raw_decision_uses_original_dtype_at_nextafter_boundary(self):
        from run_formal_p4_link_recall_diagnosis_v1 import build_link_recall_diagnosis

        boundary = torch.tensor(0.9, dtype=torch.float32)
        logit = torch.logit(boundary)
        values = torch.stack((torch.nextafter(logit, torch.tensor(float("-inf"), dtype=torch.float32)), logit, torch.nextafter(logit, torch.tensor(float("inf"), dtype=torch.float32))))
        report = build_link_recall_diagnosis([values] * 20, [torch.ones(3, dtype=torch.int64)] * 20, [torch.ones(3, dtype=torch.int64)] * 20, [torch.ones(3, dtype=torch.int64)] * 20, _provenance(mask_counts={**{f"h{index}": 3 for index in range(1, 21)}, "total": 60}))
        expected = int((torch.sigmoid(values) >= 0.9).sum().item())
        self.assertEqual(report["by_horizon"]["h1"]["all_positive"]["candidate"]["tp"], expected)

    def test_all_four_groups_have_complete_metrics_and_empty_tp_fn_are_explicit(self):
        from run_formal_p4_link_recall_diagnosis_v1 import build_link_recall_diagnosis

        report = build_link_recall_diagnosis(*_inputs(), provenance=_provenance())
        for name in ("all_positive", "continued_active", "newly_active", "previous_unobserved"):
            bucket = report["by_horizon"]["h1"][name]
            self.assertEqual(bucket["status"], "computed")
            self.assertIn("candidate", bucket)
            self.assertIn("persistence", bucket)
        for outcome in ("tp", "fn"):
            for score in ("raw_logit", "raw_weighted_score", "weight_corrected_probability"):
                self.assertEqual(set(report["by_horizon"]["h1"]["all_positive"]["candidate"]["statistics"][outcome][score]), {"q0", "q10", "q50", "q90", "q100"})
        logits, labels, previous, persistence = _inputs()
        no_fn = build_link_recall_diagnosis(
            [torch.full((4,), 4.0)] * 20, labels, previous, persistence, _provenance()
        )
        self.assertEqual(no_fn["by_horizon"]["h1"]["all_positive"]["candidate"]["statistics"]["fn"], {"status": "not_computable", "reason": "empty_subset"})

    def test_empty_subset_is_explicitly_not_computable(self):
        from run_formal_p4_link_recall_diagnosis_v1 import build_link_recall_diagnosis

        logits, labels, previous, persistence = _inputs()
        previous = [torch.full((4,), -1, dtype=torch.int64) for _ in range(20)]
        report = build_link_recall_diagnosis(logits, labels, previous, persistence, _provenance())

        empty = report["by_horizon"]["h1"]["continued_active"]
        self.assertEqual(empty, {"status": "not_computable", "reason": "empty_subset"})
        self.assertEqual(report["by_horizon"]["h1"]["all_positive"]["positive_count"], 3)

    def test_rejects_input_and_provenance_boundary_drift(self):
        from run_formal_p4_link_recall_diagnosis_v1 import build_link_recall_diagnosis

        logits, labels, previous, persistence = _inputs()
        cases = (
            (logits[:-1], labels, previous, persistence, _provenance()),
            ([torch.tensor([float("nan")])] + logits[1:], labels, previous, persistence, _provenance()),
            (logits, labels, previous, persistence, _provenance(locked_test_accessed=True)),
            (logits, labels, previous, persistence, _provenance(gpu_execution=True)),
            (logits, labels, [torch.full((4,), 2)] * 20, persistence, _provenance()),
            (logits, [torch.tensor([1, 2, 1, 0])] * 20, previous, persistence, _provenance()),
            (logits, labels, previous, [torch.tensor([1, 2, 1, 0])] * 20, _provenance()),
            (logits, labels, previous, persistence, _provenance(checkpoint_sha256="bad")),
            (logits, labels, previous, persistence, _provenance(formal_performance_claim_ready=True)),
            (logits, labels, previous, persistence, _provenance(class_weights_source_split="validation")),
            (logits, labels, previous, persistence, _provenance(class_weights_sha256="not-a-sha")),
            (logits, labels, previous, persistence, _provenance(link_activity_pos_weight=49.0)),
            (logits, labels, previous, persistence[:-1], _provenance()),
            (logits, labels, previous, persistence, _provenance(mask_counts={**{f"h{index}": 4 for index in range(1, 21)}, "total": 79})),
        )
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    build_link_recall_diagnosis(*case)
        for reload in (
            {"strict": False, "missing_keys": 0, "unexpected_keys": 0},
            {"strict": True, "missing_keys": 1, "unexpected_keys": 0},
            {"strict": True, "missing_keys": 0, "unexpected_keys": 1},
        ):
            with self.subTest(reload=reload):
                with self.assertRaises(ValueError):
                    build_link_recall_diagnosis(logits, labels, previous, persistence, _provenance(checkpoint_reload=reload))

    def test_publish_is_atomic_and_refuses_overwrite(self):
        from run_formal_p4_link_recall_diagnosis_v1 import _publish_link_recall_diagnosis_atomically, _sha256

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            _publish_link_recall_diagnosis_atomically(output, {"schema_version": "test"})
            self.assertTrue((output / "link_recall_diagnosis.json").is_file())
            self.assertTrue((output / "manifest.json").is_file())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            entry = manifest["files"]["link_recall_diagnosis.json"]
            self.assertEqual(manifest["schema_version"], "PI-JWM-link-recall-diagnosis-manifest-v1")
            self.assertEqual(entry["size_bytes"], (output / "link_recall_diagnosis.json").stat().st_size)
            self.assertEqual(entry["sha256"], _sha256(output / "link_recall_diagnosis.json"))
            with self.assertRaises(FileExistsError):
                _publish_link_recall_diagnosis_atomically(output, {"schema_version": "test"})
            self.assertFalse(list(output.parent.glob(".output.staging-*")))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            with self.assertRaisesRegex(RuntimeError, "forced"):
                _publish_link_recall_diagnosis_atomically(output, {"schema_version": "test"}, before_rename=lambda: (_ for _ in ()).throw(RuntimeError("forced")))
            self.assertFalse(output.exists())
            self.assertFalse(list(output.parent.glob(".output.staging-*")))

    def test_collect_validation_uses_history_for_h1_and_prior_target_for_later_horizons(self):
        from run_formal_p4_link_recall_diagnosis_v1 import _previous_activity_and_mask

        history_activity = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.int64)
        history_mask = torch.tensor([[[True, True], [True, False]]])
        target_activity = torch.tensor([[[0, 1], [1, 0]] + [[0, 1]] * 18], dtype=torch.int64)
        target_mask = torch.tensor([[[True, True], [True, True]] + [[False, True]] * 18])
        activity, valid = _previous_activity_and_mask(history_activity, history_mask, target_activity, target_mask)

        self.assertTrue(torch.equal(activity[:, 0], history_activity[:, -1]))
        self.assertTrue(torch.equal(valid[:, 0], history_mask[:, -1]))
        self.assertTrue(torch.equal(activity[:, 1:], target_activity[:, :-1]))
        self.assertTrue(torch.equal(valid[:, 1:], target_mask[:, :-1]))

    def test_collect_validation_marks_invalid_previous_and_repeats_history_persistence(self):
        from run_formal_p4_link_recall_diagnosis_v1 import _collect_validation

        class Model(torch.nn.Module):
            def forward(self, batch):
                return {"link_activity_logits": torch.full((1, 20, 2), 3.0)}
        batch = {
            "history": {"aggregate_link_activity": torch.tensor([[[0, 1], [1, 0]]]), "aggregate_link_activity_mask": torch.tensor([[[True, True], [True, False]]])},
            "target": {"aggregate_link_activity": torch.tensor([[[1, 1]] * 20]), "aggregate_link_activity_mask": torch.tensor([[[True, True]] * 20])},
            "static": {"physical_edge_endpoint_index": torch.tensor([[[0, 1], [1, 0]]])},
        }
        logits, labels, previous, persistence, counts, fingerprints = _collect_validation(Model(), [batch])
        self.assertEqual(counts["h1"], 2)
        self.assertEqual(previous[0].tolist(), [1, -1])
        self.assertEqual(previous[1].tolist(), [1, 1])
        self.assertEqual(persistence[0].tolist(), [1, 0])
        self.assertEqual(labels[0].tolist(), [1, 1])
        self.assertEqual(set(fingerprints), {f"h{index}" for index in range(1, 21)})

    def test_runner_checks_validation_trajectory_and_stops_on_frozen_count_drift(self):
        from run_formal_p4_link_recall_diagnosis_v1 import _manifest, _sha256, run_formal_p4_link_recall_diagnosis

        class EmptyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
        with tempfile.TemporaryDirectory() as temporary:
            root, run_root, tensor_root = Path(temporary), Path(temporary) / "run", Path(temporary) / "tensor"
            (run_root / "checkpoints").mkdir(parents=True)
            (tensor_root / "seed_001").mkdir(parents=True)
            config = {"seed": 20260831, "locked_test_accessed": False, "deterministic_rule_layer": True, "learned_methods": ["coupled_dual_gnn_residual"]}
            for path, value in ((run_root / "config.json", config), (run_root / "sample_ids.json", {"validation": ["v1"]}), (run_root / "class_weights.json", {"source_split": "train", "pos_weight": {"link_activity": 50.0}}), (tensor_root / "tensor_contract.json", {"history_steps": 2, "horizon_steps": 20}), (tensor_root / "normalization_stats.json", {}), (tensor_root / "window_index.csv", "sample_id,split,seed\nv1,validation,1\n")):
                path.write_text(json.dumps(value) if isinstance(value, dict) else value, encoding="utf-8")
            (tensor_root / "seed_001" / "trajectory_tensors.npz").write_bytes(b"fixture")
            checkpoint = {"method": "coupled_dual_gnn_residual", "model_config": {"mode": "coupled_dual_gnn", "residual_state_prediction": True, "deterministic_rule_layer": True, "history_steps": 2, "horizon_steps": 20}, "model_state_dict": {"weight": torch.zeros(1)}}
            torch.save(checkpoint, run_root / "checkpoints" / "coupled_dual_gnn_residual__best.pt")
            tensor_manifest = _manifest(tensor_root)
            tensor_manifest["schema_version"] = "PI-JWM-AirFogSim-formal-tensor-manifest-v1"
            tensor_manifest["files"] = {name: {"size_bytes": entry["bytes"], "sha256": entry["sha256"]} for name, entry in tensor_manifest["files"].items()}
            (tensor_root / "manifest.json").write_text(json.dumps(tensor_manifest), encoding="utf-8")
            config["dataset_manifest_sha256"] = _sha256(tensor_root / "manifest.json")
            (run_root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (run_root / "manifest.json").write_text(json.dumps(_manifest(run_root)), encoding="utf-8")
            class FakeDataset:
                rows = [{"sample_id": "v1", "split": "validation", "seed": 1}]
                def __len__(self): return 1
                def __getitem__(self, index): return {}
            inputs = _inputs()
            with patch("run_formal_p4_link_recall_diagnosis_v1.FormalAirFogSimWindowDataset", return_value=FakeDataset()), patch("run_formal_p4_link_recall_diagnosis_v1._reload_learned_model", return_value=EmptyModel()):
                with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
                    run_formal_p4_link_recall_diagnosis(run_root=run_root, tensor_root=tensor_root, method="coupled_dual_gnn_residual", output_dir=root / "default-checkpoint", _expected_tensor_manifest_sha256=_sha256(tensor_root / "manifest.json"))
                with self.assertRaisesRegex(ValueError, "sample_ids SHA-256"):
                    run_formal_p4_link_recall_diagnosis(run_root=run_root, tensor_root=tensor_root, method="coupled_dual_gnn_residual", output_dir=root / "default-sample-ids", _expected_tensor_manifest_sha256=_sha256(tensor_root / "manifest.json"), _expected_checkpoint_sha256=_sha256(run_root / "checkpoints" / "coupled_dual_gnn_residual__best.pt"))
            with patch("run_formal_p4_link_recall_diagnosis_v1.FormalAirFogSimWindowDataset", return_value=FakeDataset()), patch("run_formal_p4_link_recall_diagnosis_v1._reload_learned_model", return_value=EmptyModel()), patch("run_formal_p4_link_recall_diagnosis_v1._collect_validation", return_value=(*inputs, _provenance()["mask_counts"], {f"h{index}": "a" * 64 for index in range(1, 21)})):
                with self.assertRaisesRegex(ValueError, "1902/557/5855"):
                    run_formal_p4_link_recall_diagnosis(run_root=run_root, tensor_root=tensor_root, method="coupled_dual_gnn_residual", output_dir=root / "output", _expected_tensor_manifest_sha256=_sha256(tensor_root / "manifest.json"), _expected_checkpoint_sha256=_sha256(run_root / "checkpoints" / "coupled_dual_gnn_residual__best.pt"), _expected_sample_ids_sha256=_sha256(run_root / "sample_ids.json"))
            for mutation in ("missing", "unexpected"):
                mutated = torch.load(run_root / "checkpoints" / "coupled_dual_gnn_residual__best.pt", map_location="cpu", weights_only=True)
                if mutation == "missing":
                    mutated["model_state_dict"].pop("weight")
                else:
                    mutated["model_state_dict"]["unexpected.weight"] = torch.zeros(1)
                torch.save(mutated, run_root / "checkpoints" / "coupled_dual_gnn_residual__best.pt")
                (run_root / "manifest.json").write_text(json.dumps(_manifest(run_root)), encoding="utf-8")
                with patch("run_formal_p4_link_recall_diagnosis_v1.FormalAirFogSimWindowDataset", return_value=FakeDataset()), patch("run_formal_p4_link_recall_diagnosis_v1._reload_learned_model", return_value=EmptyModel()):
                    with self.assertRaises(RuntimeError):
                        run_formal_p4_link_recall_diagnosis(run_root=run_root, tensor_root=tensor_root, method="coupled_dual_gnn_residual", output_dir=root / f"output-{mutation}", _expected_tensor_manifest_sha256=_sha256(tensor_root / "manifest.json"), _expected_checkpoint_sha256=_sha256(run_root / "checkpoints" / "coupled_dual_gnn_residual__best.pt"), _expected_sample_ids_sha256=_sha256(run_root / "sample_ids.json"))
                torch.save(checkpoint, run_root / "checkpoints" / "coupled_dual_gnn_residual__best.pt")
            (run_root / "manifest.json").write_text(json.dumps(_manifest(run_root)), encoding="utf-8")
            trajectory = tensor_root / "seed_001" / "trajectory_tensors.npz"
            trajectory.write_bytes(trajectory.read_bytes() + b"drift")
            with patch("run_formal_p4_link_recall_diagnosis_v1.FormalAirFogSimWindowDataset", return_value=FakeDataset()), patch("run_formal_p4_link_recall_diagnosis_v1._reload_learned_model", return_value=EmptyModel()):
                with self.assertRaisesRegex(ValueError, "frozen tensor input identity drift"):
                    run_formal_p4_link_recall_diagnosis(run_root=run_root, tensor_root=tensor_root, method="coupled_dual_gnn_residual", output_dir=root / "output-tensor", _expected_tensor_manifest_sha256=_sha256(tensor_root / "manifest.json"), _expected_checkpoint_sha256=_sha256(run_root / "checkpoints" / "coupled_dual_gnn_residual__best.pt"), _expected_sample_ids_sha256=_sha256(run_root / "sample_ids.json"))


if __name__ == "__main__":
    unittest.main()
