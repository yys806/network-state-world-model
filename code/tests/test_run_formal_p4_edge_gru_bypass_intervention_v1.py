from __future__ import annotations

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


def _provenance(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "checkpoint_sha256": "a" * 64,
        "tensor_manifest_sha256": "b" * 64,
        "sample_ids_sha256": "c" * 64,
        "class_weights_path": "class_weights.json",
        "class_weights_sha256": "d" * 64,
        "class_weights_source_split": "train",
        "link_activity_pos_weight": 50.0,
        "mask_counts": {**{f"h{index}": 1 for index in range(1, 21)}, "total": 20},
        "checkpoint_reload": {"strict": True, "missing_keys": 0, "unexpected_keys": 0},
        "locked_test_accessed": False,
        "gpu_execution": False,
    "formal_performance_claim_ready": False,
        "tensor_manifest_path": "tensor/manifest.json",
        "sample_ids_path": "run/sample_ids.json",
        "intervention": "edge_transition_returns_previous_hidden",
        "edge_transition_module": "FormalDualGraphWorldModel.edge_transition",
    }
    result.update(overrides)
    return result


def _sentinel_inputs() -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict[str, int]]:
    """Small-enough synthetic split whose baseline aggregate is the frozen sentinel."""
    baseline, intervention, labels, previous, persistence = [], [], [], [], []
    remaining_critical = 2791
    for horizon in range(1, 21):
        positives = 388 if horizon <= 17 else 387
        true_positives = 96 if horizon <= 2 else 95
        negatives = 28 if horizon <= 17 else 27
        baseline.append(torch.tensor([3.0] * true_positives + [-3.0] * (positives - true_positives) + [3.0] * negatives, dtype=torch.float32))
        intervention.append(torch.tensor([3.0] * positives + [-3.0] * negatives, dtype=torch.float32))
        labels.append(torch.tensor([1] * positives + [0] * negatives, dtype=torch.int64))
        false_negatives = positives - true_positives
        critical = min(remaining_critical, false_negatives)
        remaining_critical -= critical
        other = false_negatives - critical
        # The non-critical false negatives deliberately exercise newly-active and
        # previously-unobserved groups while preserving the frozen critical-set size.
        previous.append(torch.tensor([1] * (true_positives + critical) + [0] * (other // 2) + [-1] * (other - other // 2) + [0] * negatives, dtype=torch.int64))
        persistence.append(torch.tensor([1] * (true_positives + critical) + [0] * other + [0] * negatives, dtype=torch.int64))
    counts = {f"h{index + 1}": value.numel() for index, value in enumerate(baseline)}
    counts["total"] = sum(counts.values())
    return baseline, intervention, labels, previous, persistence, counts


class EdgeGruBypassInterventionV1Tests(unittest.TestCase):
    def test_public_interfaces_are_importable(self):
        from run_formal_p4_edge_gru_bypass_intervention_v1 import (
            build_edge_gru_bypass_intervention_audit,
            edge_gru_bypass,
            run_formal_p4_edge_gru_bypass_intervention,
        )

        self.assertTrue(callable(build_edge_gru_bypass_intervention_audit))
        self.assertTrue(callable(edge_gru_bypass))
        self.assertTrue(callable(run_formal_p4_edge_gru_bypass_intervention))

    def test_bypass_returns_pre_call_hidden_only_for_edge_transition_and_removes_hook_on_error(self):
        from run_formal_p4_edge_gru_bypass_intervention_v1 import edge_gru_bypass

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.edge_transition = torch.nn.GRUCell(2, 2)
                self.other_transition = torch.nn.GRUCell(2, 2)

        model = Model()
        message = torch.tensor([[0.3, -0.4]])
        hidden = torch.tensor([[0.7, 0.1]])
        other_before = model.other_transition(message, hidden)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with edge_gru_bypass(model) as audit:
                self.assertTrue(torch.equal(model.edge_transition(message, hidden), hidden))
                self.assertFalse(torch.equal(model.other_transition(message, hidden), hidden))
                self.assertEqual(audit["actual_hook_calls"], 1)
                raise RuntimeError("intentional")
        self.assertEqual(len(model.edge_transition._forward_hooks), 0)
        # A second direct call proves the GRU's normal forward is restored after exceptional exit.
        self.assertFalse(torch.equal(model.edge_transition(message, hidden), hidden))
        self.assertTrue(torch.equal(model.other_transition(message, hidden), other_before))

    def test_bypass_requires_a_real_gru_cell(self):
        from run_formal_p4_edge_gru_bypass_intervention_v1 import edge_gru_bypass

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.edge_transition = torch.nn.Linear(2, 2)

        with self.assertRaisesRegex(ValueError, "GRUCell"):
            with edge_gru_bypass(Model()):
                pass

    def test_build_preserves_float32_threshold_and_full_recall_groups(self):
        from run_formal_p4_edge_gru_bypass_intervention_v1 import _raw_decision, build_edge_gru_bypass_intervention_audit

        boundary = torch.logit(torch.tensor(0.9, dtype=torch.float32))
        values = torch.stack((torch.nextafter(boundary, torch.tensor(float("-inf"), dtype=torch.float32)), boundary, torch.nextafter(boundary, torch.tensor(float("inf"), dtype=torch.float32))))
        expected = int((torch.sigmoid(values) >= 0.9).sum().item())
        self.assertEqual(int(_raw_decision(values).sum().item()), expected)
        baseline, intervention, labels, previous, persistence, counts = _sentinel_inputs()
        provenance = _provenance(mask_counts=counts)
        fingerprints = {f"h{index}": "f" * 64 for index in range(1, 21)}
        report = build_edge_gru_bypass_intervention_audit(baseline, intervention, labels, previous, persistence, provenance)

        self.assertEqual(report["baseline"]["by_horizon"].keys(), {f"h{index}" for index in range(1, 21)})
        self.assertEqual(report["intervention"]["by_horizon"]["h1"]["continued_active"]["positive_count"], 388)
        self.assertGreater(report["baseline"]["by_horizon"]["h20"]["newly_active"]["positive_count"], 0)
        self.assertGreater(report["baseline"]["by_horizon"]["h20"]["previous_unobserved"]["positive_count"], 0)
        self.assertEqual(report["original_critical_set"]["status"], "computed")
        self.assertEqual(report["original_critical_set"]["baseline_candidate_negative_count"], 2791)
        self.assertEqual(report["original_critical_set"]["intervention_recovered_count"], 2791)
        self.assertEqual(report["baseline"]["by_horizon"]["h1"]["all_positive"]["candidate"]["tp"], 96)

    def test_build_emits_the_exact_machine_status_and_frozen_original_critical_set(self):
        from run_formal_p4_edge_gru_bypass_intervention_v1 import build_edge_gru_bypass_intervention_audit

        baseline, intervention, labels, previous, persistence, counts = _sentinel_inputs()
        report = build_edge_gru_bypass_intervention_audit(baseline, intervention, labels, previous, persistence, _provenance(mask_counts=counts))
        self.assertEqual(report["original_critical_set"]["baseline_candidate_negative_count"], 2791)
        self.assertEqual(report["edge_gru_transition_intervention"], "sufficient_to_explain_dominant_low_recall")
        for branch in ("baseline", "intervention"):
            for prediction in ("candidate", "persistence"):
                self.assertIn("precision", report[branch]["overall"][prediction])
                self.assertIn("recall", report[branch]["overall"][prediction])

    def test_build_rejects_baseline_sentinel_or_input_drift(self):
        from run_formal_p4_edge_gru_bypass_intervention_v1 import build_edge_gru_bypass_intervention_audit

        values = [torch.ones(1)] * 20
        labels = [torch.ones(1, dtype=torch.int64)] * 20
        previous = [torch.ones(1, dtype=torch.int64)] * 20
        persistence = [torch.ones(1, dtype=torch.int64)] * 20
        with self.assertRaisesRegex(ValueError, "1902/557/5855"):
            build_edge_gru_bypass_intervention_audit(values, values, labels, previous, persistence, _provenance())
        with self.assertRaises(ValueError):
            build_edge_gru_bypass_intervention_audit(values, values[:-1], labels, previous, persistence, _provenance())
        with self.assertRaises(ValueError):
            build_edge_gru_bypass_intervention_audit(values, values, labels, [torch.zeros(1, dtype=torch.int64)] * 19, persistence, _provenance())
        with self.assertRaises(ValueError):
            build_edge_gru_bypass_intervention_audit(values, values, labels, previous, persistence, _provenance(gpu_execution=True))
        with self.assertRaises(ValueError):
            build_edge_gru_bypass_intervention_audit(values, values, labels, previous, persistence, _provenance(locked_test_accessed=True))

    def test_six_sufficiency_gates_are_exposed_with_independent_failures(self):
        from run_formal_p4_edge_gru_bypass_intervention_v1 import _evaluate_sufficiency

        passing = {
            "recovered_continued": 3000,
            "intervention_continued_fn": 3000,
            "h1_continued_recall": 0.21,
            "h20_continued_recall": 0.21,
            "intervention_overall_f1": 0.48,
            "intervention_overall_fp": 1114,
        }
        self.assertEqual(_evaluate_sufficiency(passing)["status"], "sufficient_to_explain")
        for field, bad in (
            ("recovered_continued", 1395),
            ("intervention_continued_fn", 3337),
            ("h1_continued_recall", 0.199),
            ("h20_continued_recall", 0.199),
            ("intervention_overall_f1", 0.4723570868),
            ("intervention_overall_fp", 1115),
        ):
            with self.subTest(field=field):
                candidate = dict(passing)
                candidate[field] = bad
                decision = _evaluate_sufficiency(candidate)
                self.assertEqual(decision["status"], "not_sufficient")
                self.assertEqual(decision["failed_gates"], [field])

    def test_publish_is_atomic_and_cleans_staging(self):
        from run_formal_p4_edge_gru_bypass_intervention_v1 import _publish_edge_gru_bypass_intervention_atomically, _sha256

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            _publish_edge_gru_bypass_intervention_atomically(output, {"schema_version": "test"})
            payload = output / "edge_gru_bypass_intervention.json"
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["files"][payload.name]["size_bytes"], payload.stat().st_size)
            self.assertEqual(manifest["files"][payload.name]["sha256"], _sha256(payload))
            with self.assertRaises(FileExistsError):
                _publish_edge_gru_bypass_intervention_atomically(output, {})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            with self.assertRaisesRegex(RuntimeError, "forced"):
                _publish_edge_gru_bypass_intervention_atomically(output, {}, before_rename=lambda: (_ for _ in ()).throw(RuntimeError("forced")))
            self.assertFalse(output.exists())
            self.assertFalse(list(output.parent.glob(".output.staging-*")))

    def test_runner_calls_only_edge_gru_twenty_times_per_batch_and_publishes(self):
        from run_formal_p4_edge_gru_bypass_intervention_v1 import _sha256, run_formal_p4_edge_gru_bypass_intervention

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.edge_transition = torch.nn.GRUCell(2, 2)
                self.other_transition = torch.nn.GRUCell(2, 2)

        baseline, intervention, labels, previous, persistence, counts = _sentinel_inputs()
        model = Model()
        loader = [object(), object()]
        provenance = _provenance(mask_counts=counts)
        fingerprints = {f"h{index}": "f" * 64 for index in range(1, 21)}
        calls = {"count": 0}

        def collect(active_model, active_loader):
            for _batch in active_loader:
                hidden = torch.zeros(1, 2)
                for _ in range(20):
                    hidden = active_model.edge_transition(torch.ones(1, 2), hidden)
                    active_model.other_transition(torch.ones(1, 2), hidden)
                    calls["count"] += 1
            return (intervention if active_model.edge_transition._forward_hooks else baseline), labels, previous, persistence, counts, fingerprints

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnosis = root / "baseline.json"
            diagnosis.write_text(json.dumps({"overall": {"candidate": {"tp": 1902, "fp": 557, "fn": 5855}}, "provenance": {"checkpoint_sha256": "a" * 64, "sample_ids_sha256": "c" * 64, "tensor_manifest_sha256": "b" * 64}}), encoding="utf-8")
            with patch("run_formal_p4_edge_gru_bypass_intervention_v1._prepare_frozen_link_validation", return_value=(model, loader, provenance)), patch("run_formal_p4_edge_gru_bypass_intervention_v1._collect_validation", side_effect=collect):
                report = run_formal_p4_edge_gru_bypass_intervention(run_root=root / "run", tensor_root=root / "tensor", method="coupled_dual_gnn_residual", baseline_diagnosis_path=diagnosis, output_dir=root / "output", _expected_tensor_manifest_sha256="b" * 64, _expected_baseline_diagnosis_sha256=_sha256(diagnosis), _expected_checkpoint_sha256="a" * 64, _expected_sample_ids_sha256="c" * 64)
            # Both passes execute 20 horizons per batch; only the intervention pass is hooked.
            self.assertEqual(calls["count"], 80)
            self.assertEqual(report["provenance"]["actual_hook_calls"], 40)
            self.assertEqual(report["provenance"]["expected_hook_calls"], 40)
            self.assertTrue(report["provenance"]["hook_removed"])
            self.assertEqual(len(model.edge_transition._forward_hooks), 0)
            self.assertTrue((root / "output" / "manifest.json").is_file())

    def test_runner_rejects_baseline_hash_and_cross_run_input_drift(self):
        from run_formal_p4_edge_gru_bypass_intervention_v1 import _sha256, _verify_aligned_collections, run_formal_p4_edge_gru_bypass_intervention

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.edge_transition = torch.nn.GRUCell(1, 1)

        baseline, intervention, labels, previous, persistence, counts = _sentinel_inputs()
        model = Model()
        provenance = _provenance(mask_counts=counts)
        fingerprints = {f"h{index}": "f" * 64 for index in range(1, 21)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnosis = root / "baseline.json"
            diagnosis.write_text(json.dumps({"overall": {"candidate": {"tp": 1902, "fp": 557, "fn": 5855}}, "provenance": {"checkpoint_sha256": "a" * 64, "sample_ids_sha256": "c" * 64, "tensor_manifest_sha256": "b" * 64}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                run_formal_p4_edge_gru_bypass_intervention(run_root=root, tensor_root=root, method="coupled_dual_gnn_residual", baseline_diagnosis_path=diagnosis, output_dir=root / "bad")
            invalid = root / "invalid.json"
            invalid.write_text(json.dumps({"overall": {"candidate": {"tp": 1, "fp": 2, "fn": 3}}, "provenance": {"checkpoint_sha256": "a" * 64, "sample_ids_sha256": "c" * 64, "tensor_manifest_sha256": "b" * 64}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate counts"):
                run_formal_p4_edge_gru_bypass_intervention(run_root=root, tensor_root=root, method="coupled_dual_gnn_residual", baseline_diagnosis_path=invalid, output_dir=root / "invalid", _expected_baseline_diagnosis_sha256=_sha256(invalid))
            with patch("run_formal_p4_edge_gru_bypass_intervention_v1._prepare_frozen_link_validation", return_value=(model, [object()], provenance)), patch("run_formal_p4_edge_gru_bypass_intervention_v1._collect_validation", side_effect=[(baseline, labels, previous, persistence, counts, fingerprints), (intervention, labels, previous, persistence, counts, fingerprints)]):
                with self.assertRaisesRegex(ValueError, "hook call count drifted"):
                    run_formal_p4_edge_gru_bypass_intervention(run_root=root, tensor_root=root, method="coupled_dual_gnn_residual", baseline_diagnosis_path=diagnosis, output_dir=root / "hook-count", _expected_tensor_manifest_sha256="b" * 64, _expected_baseline_diagnosis_sha256=_sha256(diagnosis), _expected_checkpoint_sha256="a" * 64, _expected_sample_ids_sha256="c" * 64)
            changed_labels = [row.clone() for row in labels]
            changed_labels[0][0] = 0
            with patch("run_formal_p4_edge_gru_bypass_intervention_v1._prepare_frozen_link_validation", return_value=(model, [object()], provenance)), patch("run_formal_p4_edge_gru_bypass_intervention_v1._collect_validation", side_effect=[(baseline, labels, previous, persistence, counts, fingerprints), (intervention, changed_labels, previous, persistence, counts, fingerprints)]):
                with self.assertRaisesRegex(ValueError, "labels drifted"):
                    run_formal_p4_edge_gru_bypass_intervention(run_root=root, tensor_root=root, method="coupled_dual_gnn_residual", baseline_diagnosis_path=diagnosis, output_dir=root / "drift", _expected_tensor_manifest_sha256="b" * 64, _expected_baseline_diagnosis_sha256=_sha256(diagnosis), _expected_checkpoint_sha256="a" * 64, _expected_sample_ids_sha256="c" * 64)
            for name, position, mutated in (("previous_link_activity", 2, [torch.zeros_like(row) for row in previous]), ("persistence_predictions", 3, [torch.zeros_like(row) for row in persistence]), ("mask counts", 4, {**counts, "total": counts["total"] + 1}), ("mask fingerprints", 5, {**fingerprints, "h1": "0" * 64})):
                candidate = [intervention, labels, previous, persistence, counts, fingerprints]
                candidate[position] = mutated
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, name):
                        _verify_aligned_collections((baseline, labels, previous, persistence, counts, fingerprints), tuple(candidate))

    def test_preparation_preserves_strict_checkpoint_tensor_and_non_locked_guards(self):
        from run_formal_p4_link_recall_diagnosis_v1 import _manifest, _prepare_frozen_link_validation, _sha256

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
            checkpoint_path = run_root / "checkpoints" / "coupled_dual_gnn_residual__best.pt"
            torch.save(checkpoint, checkpoint_path)
            tensor_manifest = _manifest(tensor_root)
            tensor_manifest["schema_version"] = "PI-JWM-AirFogSim-formal-tensor-manifest-v1"
            tensor_manifest["files"] = {name: {"size_bytes": item["bytes"], "sha256": item["sha256"]} for name, item in tensor_manifest["files"].items()}
            (tensor_root / "manifest.json").write_text(json.dumps(tensor_manifest), encoding="utf-8")
            config["dataset_manifest_sha256"] = _sha256(tensor_root / "manifest.json")
            (run_root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (run_root / "manifest.json").write_text(json.dumps(_manifest(run_root)), encoding="utf-8")
            class Dataset:
                rows = [{"sample_id": "v1", "split": "validation", "seed": 1}]
                def __len__(self): return 1
                def __getitem__(self, index): return {}
            with patch("run_formal_p4_link_recall_diagnosis_v1.FormalAirFogSimWindowDataset", return_value=Dataset()), patch("run_formal_p4_link_recall_diagnosis_v1._reload_learned_model", return_value=EmptyModel()):
                model, loader, provenance = _prepare_frozen_link_validation(run_root, tensor_root, "coupled_dual_gnn_residual", _sha256(tensor_root / "manifest.json"), _sha256(checkpoint_path), _sha256(run_root / "sample_ids.json"))
            self.assertEqual(len(loader), 1)
            self.assertEqual(provenance["checkpoint_reload"], {"strict": True, "missing_keys": 0, "unexpected_keys": 0})
            self.assertEqual(provenance["selected_validation_tensor_inputs"][0]["size_bytes"], len(b"fixture"))
            self.assertEqual(provenance["selected_validation_tensor_inputs"][0]["sha256"], _sha256(tensor_root / "seed_001" / "trajectory_tensors.npz"))
            self.assertIsInstance(model, EmptyModel)
            with patch("run_formal_p4_link_recall_diagnosis_v1.FormalAirFogSimWindowDataset", return_value=Dataset()), patch("run_formal_p4_link_recall_diagnosis_v1._reload_learned_model", return_value=EmptyModel()):
                with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
                    _prepare_frozen_link_validation(run_root, tensor_root, "coupled_dual_gnn_residual", _sha256(tensor_root / "manifest.json"))
            with patch("run_formal_p4_link_recall_diagnosis_v1.FormalAirFogSimWindowDataset", return_value=Dataset()), patch("run_formal_p4_link_recall_diagnosis_v1._reload_learned_model", return_value=EmptyModel()):
                with self.assertRaisesRegex(ValueError, "sample_ids SHA-256"):
                    _prepare_frozen_link_validation(run_root, tensor_root, "coupled_dual_gnn_residual", _sha256(tensor_root / "manifest.json"), _sha256(checkpoint_path))
            checkpoint["model_state_dict"]["unexpected"] = torch.zeros(1)
            torch.save(checkpoint, checkpoint_path)
            (run_root / "manifest.json").write_text(json.dumps(_manifest(run_root)), encoding="utf-8")
            with patch("run_formal_p4_link_recall_diagnosis_v1.FormalAirFogSimWindowDataset", return_value=Dataset()), patch("run_formal_p4_link_recall_diagnosis_v1._reload_learned_model", return_value=EmptyModel()):
                with self.assertRaises(RuntimeError):
                    _prepare_frozen_link_validation(run_root, tensor_root, "coupled_dual_gnn_residual", _sha256(tensor_root / "manifest.json"), _sha256(checkpoint_path), _sha256(run_root / "sample_ids.json"))
            checkpoint["model_state_dict"] = {}
            torch.save(checkpoint, checkpoint_path)
            (run_root / "manifest.json").write_text(json.dumps(_manifest(run_root)), encoding="utf-8")
            with patch("run_formal_p4_link_recall_diagnosis_v1.FormalAirFogSimWindowDataset", return_value=Dataset()), patch("run_formal_p4_link_recall_diagnosis_v1._reload_learned_model", return_value=EmptyModel()):
                with self.assertRaises(RuntimeError):
                    _prepare_frozen_link_validation(run_root, tensor_root, "coupled_dual_gnn_residual", _sha256(tensor_root / "manifest.json"), _sha256(checkpoint_path), _sha256(run_root / "sample_ids.json"))

    def test_cli_uses_explicit_baseline_diagnosis_flag(self):
        from run_formal_p4_edge_gru_bypass_intervention_v1 import _parse_args, main

        with patch.object(sys, "argv", ["runner", "--run-root", "run", "--tensor-root", "tensor", "--method", "coupled_dual_gnn_residual", "--baseline-diagnosis", "baseline.json", "--output-dir", "output"]):
            args = _parse_args()
        self.assertEqual(args.baseline_diagnosis, Path("baseline.json"))
        with patch.object(sys, "argv", ["runner", "--run-root", "run", "--tensor-root", "tensor", "--method", "coupled_dual_gnn_residual", "--baseline-diagnosis", "baseline.json", "--output-dir", "output"]), patch("run_formal_p4_edge_gru_bypass_intervention_v1.run_formal_p4_edge_gru_bypass_intervention", return_value={"edge_gru_transition_intervention": "not_sufficient_to_explain_dominant_low_recall"}), patch("builtins.print") as printed:
            main()
        printed.assert_called_once_with("not_sufficient_to_explain_dominant_low_recall")

    def test_collector_fingerprint_distinguishes_same_count_different_positions(self):
        from run_formal_p4_link_recall_diagnosis_v1 import _collect_validation

        class FakeModel(torch.nn.Module):
            def eval(self):
                return self

        def make_batch(mask):
            labels = torch.ones(1, 20, 2, dtype=torch.int64)
            history = {
                "aggregate_link_activity": torch.ones(1, 2, 2, dtype=torch.int64),
                "aggregate_link_activity_mask": torch.ones(1, 2, 2, dtype=torch.bool),
            }
            static = {"physical_edge_endpoint_index": torch.zeros(1, 2, 2, dtype=torch.int64)}
            return {
                "history": history,
                "target": {"aggregate_link_activity": labels, "aggregate_link_activity_mask": mask},
                "static": static,
            }

        def prediction(_method, _model, batch, _config):
            return {"link_activity_logits": torch.zeros_like(batch["target"]["aggregate_link_activity"], dtype=torch.float32)}

        mask_a = torch.tensor([[[True, False]] * 20])
        mask_b = torch.tensor([[[False, True]] * 20])
        with patch("run_formal_p4_link_recall_diagnosis_v1._prediction", side_effect=prediction):
            _, _, _, _, counts_a, fingerprints_a = _collect_validation(FakeModel(), [make_batch(mask_a)])
            _, _, _, _, counts_b, fingerprints_b = _collect_validation(FakeModel(), [make_batch(mask_b)])
        self.assertEqual(counts_a, counts_b)
        self.assertNotEqual(fingerprints_a, fingerprints_b)


if __name__ == "__main__":
    unittest.main()
