from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.action_attempt_ledger_v1 import (  # noqa: E402
    ActionAttemptLedger,
    AttemptIdentity,
)
from pi_jwm.formal_airfogsim_data_audit_v1 import (  # noqa: E402
    audit_formal_dataset,
    validate_action_attempt_rows,
)


def _attempt_rows(count: int = 300) -> list[dict[str, object]]:
    ledger = ActionAttemptLedger()
    rows = []
    for frame_index in range(count):
        identity = AttemptIdentity(
            run_role="natural_reference",
            episode_id="audit-episode",
            trajectory_id="audit-trajectory",
            frame_index=frame_index,
            candidate_ordinal=0,
        )
        attempt = ledger.begin(identity)
        attempt.candidate_built({"frame": frame_index})
        attempt.contract_validated()
        attempt.pre_setter_revalidated()
        attempt.setters_applied()
        attempt.env_step_started()
        attempt.env_step_completed()
        attempt.outcome_captured()
        rows.append(attempt.accept())
    return rows


class FormalAirFogSimDataAuditTests(unittest.TestCase):
    def test_valid_action_attempt_rows_are_contiguous_and_ledger_valid(self):
        rows = _attempt_rows()
        self.assertEqual(
            validate_action_attempt_rows(
                rows,
                trajectory_id="audit-trajectory",
            ),
            (),
        )

    def test_tampered_action_frame_is_rejected(self):
        rows = _attempt_rows()
        tampered = copy.deepcopy(rows)
        tampered[42]["frame_index"] = 43
        errors = validate_action_attempt_rows(
            tampered,
            trajectory_id="audit-trajectory",
        )
        self.assertTrue(errors)
        self.assertIn("contiguous 0..N-1", " ".join(errors))

    def test_missing_dataset_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = audit_formal_dataset(Path(temporary) / "missing")
        self.assertFalse(report["audit_ready"])
        self.assertFalse(report["formal_data_approved"])
        self.assertFalse(report["training_eligible"])
        self.assertIn("dataset_directory_present", report["failed_checks"])


if __name__ == "__main__":
    unittest.main()
