from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pi_jwm.evaluation.result_protocol import (  # noqa: E402
    build_result_protocol,
    classify_bridge_result,
    validate_result_protocol,
)


class ResultProtocolTest(unittest.TestCase):
    def test_deployable_result_allows_test_evaluation_but_not_test_fitting(self) -> None:
        protocol = build_result_protocol(
            result_kind="deployable",
            fit_splits=("train", "val"),
            selection_split="val",
            evaluation_split="test",
        )

        self.assertEqual(protocol["result_kind"], "deployable")
        self.assertEqual(protocol["evaluation_split"], "test")
        with self.assertRaisesRegex(ValueError, "test split.*fitting"):
            build_result_protocol(
                result_kind="deployable",
                fit_splits=("train", "test"),
                selection_split="val",
                evaluation_split="test",
            )

    def test_deployable_result_cannot_select_on_test(self) -> None:
        with self.assertRaisesRegex(ValueError, "select.*test split"):
            build_result_protocol(
                result_kind="deployable",
                fit_splits=("train",),
                selection_split="test",
                evaluation_split="test",
            )

    def test_bridge_classification_marks_label_aware_modes_as_references(self) -> None:
        self.assertEqual(
            classify_bridge_result("policy", "threshold", "predicted_all"),
            "deployable",
        )
        for action_generator, action_decoder, mode in (
            ("true_future", "threshold", "predicted_all"),
            ("policy", "oracle_topk", "predicted_all"),
            ("policy", "threshold", "true_first_pred_rest"),
            ("policy_activity_true_value", "threshold", "predicted_all"),
        ):
            self.assertEqual(
                classify_bridge_result(action_generator, action_decoder, mode),
                "true_future_reference",
            )

    def test_protocol_rejects_unknown_kinds_and_splits(self) -> None:
        with self.assertRaisesRegex(ValueError, "result_kind"):
            validate_result_protocol(
                {
                    "result_kind": "final_magic",
                    "fit_splits": ["train"],
                    "selection_split": "val",
                    "evaluation_split": "test",
                }
            )
        with self.assertRaisesRegex(ValueError, "split"):
            build_result_protocol(
                result_kind="sample_oracle",
                fit_splits=("future",),
                selection_split="none",
                evaluation_split="test",
            )


if __name__ == "__main__":
    unittest.main()
