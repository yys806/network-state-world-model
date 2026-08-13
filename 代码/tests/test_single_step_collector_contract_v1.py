from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.single_step_collector_contract_v1 import (  # noqa: E402
    CandidateAction,
    OffloadAction,
    RbAssignment,
    build_single_step_status_flags,
    validate_candidate_action,
)


class SingleStepCollectorContractV1Tests(unittest.TestCase):
    def test_candidate_action_rejects_duplicate_rb_records(self):
        action = CandidateAction(
            candidate_id="local",
            offloads=(OffloadAction("veh0", "task0", "veh0", ("veh0",)),),
            rb_assignments=(RbAssignment(0, 0, 0, 0), RbAssignment(0, 0, 0, 0)),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_candidate_action(
                action, task_ids=("task0",), edge_count=1, flow_count=1, n_rb=1
            )

    def test_candidate_action_rejects_unknown_task_and_invalid_route(self):
        action = CandidateAction(
            candidate_id="remote",
            offloads=(OffloadAction("veh0", "missing", "rsu0", ("veh0", "rsu0")),),
            rb_assignments=(),
        )
        with self.assertRaisesRegex(ValueError, "task_id"):
            validate_candidate_action(
                action, task_ids=("task0",), edge_count=1, flow_count=1, n_rb=1
            )

    def test_bundle_flags_keep_single_step_separate_from_training(self):
        flags = build_single_step_status_flags()
        self.assertFalse(flags["single_step_real_airfogsim_executed"])
        self.assertFalse(flags["v4_collector_implemented"])
        self.assertFalse(flags["training_eligible"])

    def test_valid_action_is_normalized_to_plain_records(self):
        action = CandidateAction(
            candidate_id="local",
            offloads=(OffloadAction("veh0", "task0", "veh0", ("veh0",)),),
            rb_assignments=(RbAssignment(0, 0, 0, 0),),
        )
        normalized = validate_candidate_action(
            action, task_ids=("task0",), edge_count=1, flow_count=1, n_rb=1
        )
        self.assertEqual("local", normalized.candidate_id)
        self.assertEqual((0, 0, 0, 0), normalized.rb_assignments[0].as_tuple())


if __name__ == "__main__":
    unittest.main()
