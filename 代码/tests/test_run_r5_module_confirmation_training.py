from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
for root in (CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


class R5ModuleConfirmationTrainingTest(unittest.TestCase):
    def test_new_run_specs_exclude_reused_b_and_cover_four_by_three(self) -> None:
        from run_r5_module_confirmation_training import build_new_run_specs

        specs = build_new_run_specs()
        self.assertEqual(12, len(specs))
        self.assertEqual(
            [(candidate, seed) for candidate in "FGHJ" for seed in (20260803, 20260804, 20260805)],
            [(spec.combination_id, spec.training_seed) for spec in specs],
        )

    def test_downloaded_bundle_verifier_detects_tampering(self) -> None:
        from run_r5_module_confirmation_training import verify_manifest_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "result.json"
            payload.write_text('{"ok": true}\n', encoding="utf-8")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            manifest = {
                "r5_module_confirmation_complete": True,
                "files": {
                    "result.json": {"sha256": digest, "size_bytes": payload.stat().st_size}
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = verify_manifest_files(root)
            self.assertEqual(1, report["verified_file_count"])
            payload.write_bytes(b"x" * payload.stat().st_size)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_manifest_files(root)

    def test_full_bundle_verifier_rejects_manifest_only_bundle(self) -> None:
        from run_r5_module_confirmation_training import verify_downloaded_bundle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "result.json"
            payload.write_text('{"ok": true}\n', encoding="utf-8")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            manifest = {
                "r5_module_confirmation_complete": True,
                "files": {
                    "result.json": {"sha256": digest, "size_bytes": payload.stat().st_size}
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "training summary"):
                verify_downloaded_bundle(root)

    def test_non_cuda_gate_does_not_create_output(self) -> None:
        from run_r5_module_confirmation_training import require_confirmation_cuda

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                require_confirmation_cuda("cpu", output)
            self.assertFalse(output.exists())

    def test_remote_launcher_requires_gpu_smoke_before_formal_matrix(self) -> None:
        launcher = (CODE_ROOT / "scripts" / "run_r5_module_confirmation_remote.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--smoke-epochs 1", launcher)
        self.assertIn("verify_r5_module_confirmation_bundle.py --smoke", launcher)
        self.assertIn("smoke_failed", launcher)

    def test_smoke_bundle_verifier_checks_confirmation_checkpoint(self) -> None:
        from pi_jwm.r5_confirmation_checkpoint import save_confirmation_checkpoint
        from pi_jwm.r5_module_confirmation import build_confirmation_model
        from pi_jwm.r5_protocol import R5FormalProtocol
        from run_r5_module_confirmation_training import verify_smoke_bundle

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = R5FormalProtocol(
                training_seeds=(20260803, 20260804, 20260805),
                max_epochs=100,
                patience=10,
                effective_batch_size=32,
                minimum_improvement=1.0e-4,
            )
            bindings = {
                key: "a" * 64
                for key in (
                    "dataset_protocol_sha256",
                    "tensor_contract_sha256",
                    "normalization_sha256",
                    "metric_registry_sha256",
                    "source_code_sha256",
                    "r4_screening_manifest_sha256",
                    "r5_protocol_sha256",
                    "existing_r5_manifest_sha256",
                    "confirmation_matrix_sha256",
                )
            }
            checkpoint = root / "combinations" / "F" / "seed_20260803" / "best_checkpoint.pt"
            model = build_confirmation_model("F", hidden_dim=4, history_steps=8)
            save_confirmation_checkpoint(
                checkpoint,
                model,
                None,
                bindings,
                protocol,
                learning_rate=1.0e-4,
                seed=20260803,
            )
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            report = {
                "schema_version": "PIJWM-R5-Module-Confirmation-GPU-Training-v1",
                "run_id": "F__seed_20260803",
                "combination_id": "F",
                "training_seed": 20260803,
                "status": "completed",
                "checkpoint_sha256": digest,
                "checkpoint": "combinations/F/seed_20260803/best_checkpoint.pt",
                "locked_test_accessed": False,
            }
            (root / "run_report.json").write_text(json.dumps(report), encoding="utf-8")
            (root / "trained_run_reports.json").write_text(json.dumps([report]), encoding="utf-8")
            reused = [
                {
                    "combination_id": "B",
                    "training_seed": seed,
                    "status": "completed",
                    "evidence_origin": "reused_verified_r5_B",
                    "source_manifest_sha256": bindings["existing_r5_manifest_sha256"],
                    "locked_test_accessed": False,
                }
                for seed in (20260803, 20260804, 20260805)
            ]
            (root / "reused_run_reports.json").write_text(json.dumps(reused), encoding="utf-8")
            (root / "failed_runs.json").write_text("[]", encoding="utf-8")
            (root / "training_protocol.json").write_text(
                json.dumps({**protocol.to_dict(), "new_combinations": ["F", "G", "H", "J"], "reused_combination": "B", "smoke_only": True}),
                encoding="utf-8",
            )
            (root / "input_provenance.json").write_text(
                json.dumps({"bindings": bindings, "locked_test_accessed": False}),
                encoding="utf-8",
            )
            (root / "training_summary.json").write_text(
                json.dumps(
                    {
                        "r5_module_confirmation_complete": False,
                        "smoke_only": True,
                        "new_expected_run_count": 1,
                        "new_completed_run_count": 1,
                        "reused_run_count": 3,
                        "total_evidence_run_count": 4,
                        "failed_run_count": 0,
                        "locked_test_accessed": False,
                        "selection_status": "incomplete",
                    }
                ),
                encoding="utf-8",
            )
            files = {}
            for path in root.rglob("*"):
                if path.is_file() and path.name != "manifest.json":
                    files[path.relative_to(root).as_posix()] = {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size_bytes": path.stat().st_size,
                    }
            (root / "manifest.json").write_text(
                json.dumps({"r5_module_confirmation_complete": False, "files": files}),
                encoding="utf-8",
            )
            result = verify_smoke_bundle(root)
            self.assertTrue(result["smoke_verified"])
            self.assertEqual(1, result["verified_checkpoint_count"])

    def test_full_bundle_verifier_checks_all_twelve_new_checkpoints(self) -> None:
        from pi_jwm.r5_confirmation_checkpoint import save_confirmation_checkpoint
        from pi_jwm.r5_module_confirmation import build_confirmation_model
        from pi_jwm.r5_protocol import REQUIRED_PUBLIC_METRIC_GATES, R5FormalProtocol
        from run_r5_module_confirmation_training import verify_downloaded_bundle

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = R5FormalProtocol(
                training_seeds=(20260803, 20260804, 20260805),
                max_epochs=100,
                patience=10,
                effective_batch_size=32,
                minimum_improvement=1.0e-4,
            )
            bindings = {
                key: "b" * 64
                for key in (
                    "dataset_protocol_sha256",
                    "tensor_contract_sha256",
                    "normalization_sha256",
                    "metric_registry_sha256",
                    "source_code_sha256",
                    "r4_screening_manifest_sha256",
                    "r5_protocol_sha256",
                    "existing_r5_manifest_sha256",
                    "confirmation_matrix_sha256",
                )
            }
            reports = []
            for combination in ("F", "G", "H", "J"):
                for seed in (20260803, 20260804, 20260805):
                    model = build_confirmation_model(combination, hidden_dim=4, history_steps=8)
                    checkpoint = root / "combinations" / combination / f"seed_{seed}" / "best_checkpoint.pt"
                    save_confirmation_checkpoint(
                        checkpoint,
                        model,
                        None,
                        bindings,
                        protocol,
                        learning_rate=1.0e-4,
                        seed=seed,
                    )
                    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    report = {
                        "schema_version": "PIJWM-R5-Module-Confirmation-GPU-Training-v1",
                        "run_id": f"{combination}__seed_{seed}",
                        "combination_id": combination,
                        "training_seed": seed,
                        "status": "completed",
                        "checkpoint": checkpoint.relative_to(root).as_posix(),
                        "checkpoint_sha256": digest,
                        "locked_test_accessed": False,
                        "final_validation": {
                            "metrics": {
                                metric_id: {"status": "computed", "value": 0.0}
                                for metric_id in REQUIRED_PUBLIC_METRIC_GATES
                            }
                        },
                    }
                    run_report_path = checkpoint.parent / "run_report.json"
                    run_report_path.write_text(json.dumps(report), encoding="utf-8")
                    reports.append(report)
            reused = [
                {
                    "combination_id": "B",
                    "training_seed": seed,
                    "status": "completed",
                    "evidence_origin": "reused_verified_r5_B",
                    "source_manifest_sha256": bindings["existing_r5_manifest_sha256"],
                    "locked_test_accessed": False,
                }
                for seed in (20260803, 20260804, 20260805)
            ]
            (root / "trained_run_reports.json").write_text(json.dumps(reports), encoding="utf-8")
            (root / "reused_run_reports.json").write_text(json.dumps(reused), encoding="utf-8")
            (root / "failed_runs.json").write_text("[]", encoding="utf-8")
            (root / "training_protocol.json").write_text(
                json.dumps({**protocol.to_dict(), "new_combinations": ["F", "G", "H", "J"], "reused_combination": "B", "smoke_only": False}),
                encoding="utf-8",
            )
            (root / "input_provenance.json").write_text(
                json.dumps({"bindings": bindings, "locked_test_accessed": False}),
                encoding="utf-8",
            )
            (root / "training_summary.json").write_text(
                json.dumps(
                    {
                        "r5_module_confirmation_complete": True,
                        "smoke_only": False,
                        "new_expected_run_count": 12,
                        "new_completed_run_count": 12,
                        "reused_run_count": 3,
                        "total_evidence_run_count": 15,
                        "failed_run_count": 0,
                        "locked_test_accessed": False,
                        "selection_status": "descriptive_only",
                    }
                ),
                encoding="utf-8",
            )
            (root / "combination_summary.json").write_text(
                json.dumps(
                    {
                        "selection_status": "descriptive_only",
                        "combinations": {key: {} for key in ("B", "F", "G", "H", "J")},
                    }
                ),
                encoding="utf-8",
            )
            files = {}
            for path in root.rglob("*"):
                if path.is_file() and path.name != "manifest.json":
                    files[path.relative_to(root).as_posix()] = {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size_bytes": path.stat().st_size,
                    }
            (root / "manifest.json").write_text(
                json.dumps({"r5_module_confirmation_complete": True, "locked_test_accessed": False, "files": files}),
                encoding="utf-8",
            )
            result = verify_downloaded_bundle(root)
            self.assertTrue(result["verified"])
            self.assertEqual(12, result["verified_checkpoint_count"])

    def test_runner_passes_r4_config_directly_to_training_step(self) -> None:
        import run_r5_module_confirmation_training as runner

        evaluation = CODE_ROOT / "artifacts" / "evaluation" / "pi_jwm_eval_protocol_v3"
        dataset = CODE_ROOT / "artifacts" / "datasets" / "airfogsim_teacher_aligned_v3"
        r4_root = CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r4_gpu_screening_v1"
        existing = CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r5_gpu_training_v1"
        captured = {}

        def fake_train(spec, **kwargs):
            captured["config"] = kwargs["config"]
            return {"run_id": spec.run_id, "combination_id": spec.combination_id, "training_seed": spec.training_seed}

        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(runner, "require_confirmation_cuda", return_value=torch.device("cpu")), \
                mock.patch.object(runner.torch.cuda, "current_device", return_value=0), \
                mock.patch.object(runner.torch.cuda, "set_device"), \
                mock.patch.object(runner, "_train_one_run", side_effect=fake_train):
                runner.run_confirmation_training(
                    dataset,
                    evaluation,
                    r4_root,
                    existing,
                    Path(temporary) / "confirmation",
                    combination_ids=("F",),
                    training_seeds=(20260803,),
                    smoke_epochs=1,
                )
        self.assertEqual("no_cross_graph_coupling_v1", captured["config"].coupling)

    def test_local_b_evidence_is_reusable_under_the_frozen_protocol(self) -> None:
        from pi_jwm.r4_gpu_screening import build_training_window_schedule, build_validation_windows
        from pi_jwm.r5_protocol import load_r5_protocol
        from run_r5_module_confirmation_training import (
            validate_reused_b_results,
            validate_reused_window_schedules,
        )

        evaluation = CODE_ROOT / "artifacts" / "evaluation" / "pi_jwm_eval_protocol_v3"
        existing = CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r5_gpu_training_v1"
        reports = validate_reused_b_results(existing, load_r5_protocol(evaluation))
        self.assertEqual([20260803, 20260804, 20260805], [row["training_seed"] for row in reports])
        self.assertTrue(all(row["evidence_origin"] == "reused_verified_r5_B" for row in reports))

        bindings = json.loads((existing / "input_provenance.json").read_text(encoding="utf-8"))["bindings"]
        mismatched = dict(bindings)
        mismatched["normalization_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "normalization_sha256"):
            validate_reused_b_results(
                existing,
                load_r5_protocol(evaluation),
                expected_bindings=mismatched,
            )

        dataset = CODE_ROOT / "artifacts" / "datasets" / "airfogsim_teacher_aligned_v3"
        seeds = (20260803, 20260804, 20260805)
        schedules = {
            seed: build_training_window_schedule(
                dataset,
                epochs=100,
                windows_per_epoch=32,
                seed=seed,
            )
            for seed in seeds
        }
        validation = {
            seed: build_validation_windows(
                dataset,
                split="validation",
                horizons=(1, 5, 20),
                seed=seed,
            )
            for seed in seeds
        }
        calibration = {
            seed: build_validation_windows(
                dataset,
                split="calibration",
                horizons=(1, 5, 20),
                seed=seed,
            )
            for seed in seeds
        }
        validate_reused_window_schedules(existing, schedules, validation, calibration)

        schedules[20260803][0] = tuple(reversed(schedules[20260803][0]))
        with self.assertRaisesRegex(ValueError, "training schedule mismatch"):
            validate_reused_window_schedules(existing, schedules, validation, calibration)


if __name__ == "__main__":
    unittest.main()
