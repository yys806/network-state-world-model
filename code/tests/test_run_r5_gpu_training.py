from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
for root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


class RunR5GpuTrainingTests(unittest.TestCase):
    def setUp(self):
        from pi_jwm.r5_protocol import R5FormalProtocol

        self.protocol = R5FormalProtocol(
            training_seeds=(20260803, 20260804, 20260805),
            max_epochs=100,
            patience=10,
            effective_batch_size=32,
            minimum_improvement=1.0e-4,
        )

    def test_run_specs_are_exact_five_by_three_matrix(self):
        from run_r5_gpu_training import build_run_specs

        specs = build_run_specs(self.protocol)
        self.assertEqual(15, len(specs))
        self.assertEqual(
            [(combination, seed) for combination in "ABCDE" for seed in self.protocol.training_seeds],
            [(spec.combination_id, spec.training_seed) for spec in specs],
        )
        with self.assertRaisesRegex(ValueError, "combination"):
            build_run_specs(self.protocol, combination_ids=("A", "F"))
        with self.assertRaisesRegex(ValueError, "seed"):
            build_run_specs(self.protocol, training_seeds=(7,))

    def test_non_cuda_and_locked_test_fail_before_output_creation(self):
        from run_r5_gpu_training import require_cuda_device, validate_training_splits

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "formal"
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                require_cuda_device("cpu", output)
            self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "locked_test"):
            validate_training_splits(("train", "validation", "locked_test"))

    def test_resumable_state_requires_matching_checkpoint_and_input_fingerprint(self):
        from run_r5_gpu_training import load_resumable_run_state

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint = run_dir / "last_checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            state = {
                "schema_version": "PIJWM-R5-GPU-Run-State-v1",
                "combination_id": "B",
                "training_seed": 20260803,
                "last_epoch": 4,
                "best_epoch": 3,
                "best_validation_protocol_score": 1.25,
                "stale_epochs": 1,
                "last_checkpoint_sha256": digest,
                "input_fingerprint": "a" * 64,
            }
            (run_dir / "run_state.json").write_text(json.dumps(state), "utf-8")
            restored = load_resumable_run_state(
                run_dir,
                combination_id="B",
                training_seed=20260803,
                input_fingerprint="a" * 64,
            )
            self.assertEqual(4, restored["last_epoch"])

            changed = {**state, "input_fingerprint": "b" * 64}
            (run_dir / "run_state.json").write_text(json.dumps(changed), "utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_resumable_run_state(
                    run_dir,
                    combination_id="B",
                    training_seed=20260803,
                    input_fingerprint="a" * 64,
                )

    def test_multi_seed_summary_reports_metrics_without_automatic_winner(self):
        from run_r5_gpu_training import summarize_completed_runs

        reports = []
        for combination_index, combination in enumerate("ABCDE"):
            for seed_index, seed in enumerate(self.protocol.training_seeds):
                reports.append(
                    {
                        "combination_id": combination,
                        "training_seed": seed,
                        "best_validation_protocol_score": float(combination_index + seed_index),
                        "runtime_seconds": 10.0 + seed_index,
                        "peak_cuda_memory_bytes": 100 + combination_index,
                        "final_validation": {
                            "metrics": {
                                "event.information_link_activity.auprc": {"status": "computed", "value": 0.1 + seed_index},
                                "link.active_only_rate.mae": {"status": "computed", "value": 1.0 + seed_index},
                                "task.lifecycle.macro_f1": {"status": "computed", "value": 0.2 + seed_index},
                                "selection.required_continuous.normalized_error": {"status": "computed", "value": 2.0 + seed_index},
                            }
                        },
                    }
                )
        summary = summarize_completed_runs(reports, self.protocol)
        self.assertEqual(set("ABCDE"), set(summary["combinations"]))
        self.assertEqual(3, summary["combinations"]["A"]["training_seed_count"])
        self.assertNotIn("winner", summary)
        self.assertEqual("descriptive_only", summary["selection_status"])

    def test_cli_exposes_explicit_smoke_epoch_limit(self):
        from run_r5_gpu_training import _parser

        args = _parser().parse_args(
            [
                "--dataset-root", "dataset",
                "--evaluation-root", "evaluation",
                "--r4-screening-root", "r4",
                "--output-dir", "output",
                "--combination", "A",
                "--seed", "20260803",
                "--smoke-epochs", "1",
            ]
        )
        self.assertEqual(1, args.smoke_epochs)
        self.assertEqual(["A"], args.combinations)
        self.assertEqual([20260803], args.seeds)


if __name__ == "__main__":
    unittest.main()
