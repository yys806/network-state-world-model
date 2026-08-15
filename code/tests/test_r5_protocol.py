from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class R5ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.evaluation_root = (
            CODE_ROOT / "artifacts" / "evaluation" / "pi_jwm_eval_protocol_v3"
        )

    def test_protocol_loads_frozen_three_seed_formal_budget(self):
        from pi_jwm.r5_protocol import load_r5_protocol

        protocol = load_r5_protocol(self.evaluation_root)
        self.assertEqual((20260803, 20260804, 20260805), protocol.training_seeds)
        self.assertEqual(100, protocol.max_epochs)
        self.assertEqual(10, protocol.patience)
        self.assertEqual(32, protocol.effective_batch_size)
        self.assertEqual(1.0e-4, protocol.minimum_improvement)
        self.assertEqual("validation", protocol.checkpoint_split)
        self.assertEqual("calibration", protocol.calibration_split)

    def test_exact_a_to_e_combinations_are_machine_readable(self):
        from pi_jwm.r5_protocol import r5_combination_matrix

        matrix = r5_combination_matrix(hidden_dim=8, history_steps=2)
        self.assertEqual(["A", "B", "C", "D", "E"], list(matrix))
        reference = matrix["A"].config
        self.assertEqual("deterministic_graph_gru_v1", reference.dynamics)
        self.assertEqual("graph_rssm_v1", matrix["B"].config.dynamics)
        self.assertEqual("heteroscedastic_typed_v1", matrix["C"].config.head)
        self.assertEqual("explicit_dag_message_passing_v1", matrix["D"].config.dag)
        self.assertEqual("soft_predicted_presence_v1", matrix["E"].config.presence)
        for combination in matrix.values():
            json.dumps(combination.to_dict(), ensure_ascii=False)
            self.assertTrue(combination.question)

    def test_unknown_combination_and_locked_test_are_rejected(self):
        from pi_jwm.r5_protocol import get_r5_combination, validate_r5_splits

        with self.assertRaisesRegex(ValueError, "unknown R5 combination"):
            get_r5_combination("F")
        with self.assertRaisesRegex(ValueError, "locked_test"):
            validate_r5_splits(("train", "locked_test"))

    def test_independent_metric_gates_prevent_single_score_freeze(self):
        from pi_jwm.r5_protocol import REQUIRED_PUBLIC_METRIC_GATES

        self.assertEqual(
            {
                "event.information_link_activity.auprc",
                "link.active_only_rate.mae",
                "task.lifecycle.macro_f1",
                "selection.required_continuous.normalized_error",
            },
            set(REQUIRED_PUBLIC_METRIC_GATES),
        )


if __name__ == "__main__":
    unittest.main()
