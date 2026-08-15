from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_gpu_training_protocol import (  # noqa: E402
    CheckpointMetric,
    ValidationCheckpointGate,
    build_default_gpu_training_protocol,
    validate_formal_run_records,
)


class R6GPUTrainingProtocolTest(unittest.TestCase):
    def test_default_matrix_has_exactly_eighteen_equal_budget_runs(self) -> None:
        protocol = build_default_gpu_training_protocol()
        runs = protocol.formal_runs()
        self.assertEqual(18, len(runs))
        self.assertEqual({"actor_critic", "ppo_clipped"}, {run.method_id for run in runs})
        self.assertEqual(
            {"explicit_only", "latent_only", "explicit_latent"},
            {run.state_mode for run in runs},
        )
        self.assertEqual({20260803, 20260804, 20260805}, {run.seed for run in runs})
        budgets = {(run.max_environment_steps, run.rollout_length, run.minibatch_size) for run in runs}
        self.assertEqual({(100000, 128, 32)}, budgets)
        self.assertEqual(18, len({run.run_id for run in runs}))
        self.assertFalse(protocol.locked_test_accessed)
        self.assertEqual("validation", protocol.checkpoint_split)
        self.assertEqual("calibration", protocol.threshold_split)
        self.assertEqual("online_airfogsim_strict_dual_graph", protocol.state_source)
        self.assertEqual(8, protocol.online_history_steps)
        self.assertEqual(64, protocol.validation_step_limit)
        self.assertTrue(protocol.atomic_resume_checkpoint)

    def test_run_validator_keeps_failures_and_rejects_smoke_or_missing_runs(self) -> None:
        protocol = build_default_gpu_training_protocol()
        records = [
            {"run_id": run.run_id, "status": "failed" if index == 0 else "complete", "formal": True}
            for index, run in enumerate(protocol.formal_runs())
        ]
        summary = validate_formal_run_records(protocol, records)
        self.assertEqual(1, summary.failed_count)
        self.assertEqual(17, summary.complete_count)
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_formal_run_records(protocol, records[:-1])
        bad = list(records)
        bad[0] = {**bad[0], "formal": False}
        with self.assertRaisesRegex(ValueError, "smoke"):
            validate_formal_run_records(protocol, bad)

    def test_checkpoint_gate_uses_validation_return_then_frozen_tiebreak(self) -> None:
        gate = ValidationCheckpointGate(patience=2)
        first = gate.update(CheckpointMetric(10000, 4.0, 0.8, 1.2, 0))
        self.assertTrue(first.improved)
        worse_delay = gate.update(CheckpointMetric(20000, 4.0, 0.8, 1.3, 0))
        self.assertFalse(worse_delay.improved)
        hard_invalid = gate.update(CheckpointMetric(30000, 9.0, 1.0, 0.1, 1))
        self.assertFalse(hard_invalid.eligible)
        self.assertTrue(hard_invalid.should_stop)


if __name__ == "__main__":
    unittest.main()
