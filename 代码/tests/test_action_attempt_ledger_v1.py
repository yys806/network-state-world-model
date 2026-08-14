from __future__ import annotations

import copy
import dataclasses
import sys
import unittest
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.action_attempt_ledger_v1 import (  # noqa: E402
    ActionAttemptLedger,
    AttemptIdentity,
    LedgerContractError,
    candidate_digest,
    summarize_attempts,
    validate_attempt_records,
)


def identity(
    *,
    run_role: str = "natural_reference",
    frame_index: int = 0,
    candidate_ordinal: int = 0,
) -> AttemptIdentity:
    return AttemptIdentity(
        run_role=run_role,
        episode_id="natural-seed-0-orthogonal",
        trajectory_id="natural-seed-0-orthogonal-trajectory",
        frame_index=frame_index,
        candidate_ordinal=candidate_ordinal,
    )


def accepted_record(
    ledger: ActionAttemptLedger,
    attempt_identity: AttemptIdentity,
    *,
    candidate: object | None = None,
) -> dict[str, object]:
    attempt = ledger.begin(attempt_identity)
    attempt.candidate_built({"action": 1} if candidate is None else candidate)
    attempt.contract_validated()
    attempt.pre_setter_revalidated()
    attempt.setters_applied()
    attempt.env_step_started()
    attempt.env_step_completed()
    attempt.outcome_captured()
    return attempt.accept()


class AttemptIdentityAndDigestTests(unittest.TestCase):
    def test_attempt_id_is_stable_and_run_role_separated(self):
        ledger = ActionAttemptLedger()
        reference = identity()
        replay = dataclasses.replace(reference, run_role="natural_replay")

        self.assertEqual(ledger.begin_id(reference), ledger.begin_id(reference))
        self.assertNotEqual(ledger.begin_id(reference), ledger.begin_id(replay))
        self.assertTrue(ledger.begin_id(reference).startswith("attempt::"))

    def test_identity_rejects_empty_boolean_negative_and_unknown_values(self):
        invalid = (
            {"run_role": "unknown"},
            {"episode_id": ""},
            {"trajectory_id": "  "},
            {"frame_index": True},
            {"frame_index": -1},
            {"candidate_ordinal": False},
            {"candidate_ordinal": -1},
            {"candidate_ordinal": 1},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(LedgerContractError):
                dataclasses.replace(identity(), **changes)

        fixture = identity(run_role="fixture", candidate_ordinal=1)
        self.assertEqual(fixture.candidate_ordinal, 1)

    def test_candidate_digest_is_canonical_for_dataclass_enum_and_mapping_order(self):
        class Mode(str, Enum):
            ACTIVE = "active"

        @dataclass(frozen=True)
        class Candidate:
            amount: int
            mode: Mode

        left = {"z": [3, 2], "a": Candidate(amount=1, mode=Mode.ACTIVE)}
        right = {"a": Candidate(mode=Mode.ACTIVE, amount=1), "z": [3, 2]}
        changed = {"a": Candidate(mode=Mode.ACTIVE, amount=2), "z": [3, 2]}

        self.assertEqual(candidate_digest(left), candidate_digest(right))
        self.assertNotEqual(candidate_digest(left), candidate_digest(changed))


class AttemptStateMachineTests(unittest.TestCase):
    def test_accepted_attempt_records_complete_success_path(self):
        ledger = ActionAttemptLedger()
        row = accepted_record(ledger, identity())

        self.assertEqual(
            row["stage_trace"],
            [
                "begun",
                "candidate_built",
                "contract_validated",
                "pre_setter_revalidated",
                "setters_applied",
                "env_step_started",
                "env_step_completed",
                "outcome_captured",
            ],
        )
        self.assertEqual(row["disposition"], "accepted")
        self.assertEqual(row["terminal_stage"], "outcome_captured")
        self.assertTrue(row["env_step_called"])
        self.assertTrue(row["env_step_completed"])
        self.assertEqual(row["environment_mutation_status"], "confirmed")
        self.assertFalse(row["quarantined"])
        self.assertFalse(row["training_eligible"])
        self.assertIsNone(row["rejection_code"])
        self.assertEqual(validate_attempt_records([row]), ())

    def test_duplicate_id_illegal_transition_and_duplicate_terminal_are_rejected(self):
        ledger = ActionAttemptLedger()
        attempt = ledger.begin(identity())
        with self.assertRaisesRegex(LedgerContractError, "duplicate attempt_id"):
            ledger.begin(identity())
        with self.assertRaisesRegex(LedgerContractError, "transition"):
            attempt.contract_validated()

        attempt.candidate_built({"action": 1})
        attempt.contract_validated()
        attempt.pre_setter_revalidated()
        attempt.setters_applied()
        attempt.env_step_started()
        attempt.env_step_completed()
        attempt.outcome_captured()
        attempt.accept()
        with self.assertRaisesRegex(LedgerContractError, "terminal"):
            attempt.accept()

    def test_setter_calls_are_ordered_details_not_caller_supplied_counts(self):
        ledger = ActionAttemptLedger()
        attempt = ledger.begin(identity(run_role="fixture"))
        attempt.candidate_built({"action": 1})
        attempt.contract_validated()
        attempt.pre_setter_revalidated()
        first = attempt.setter_started(setter_kind="cpu_callback", task_id=None)
        attempt.setter_finished(first, call_completed=True, succeeded=True)
        second = attempt.setter_started(setter_kind="offload", task_id="task-1")
        attempt.setter_finished(second, call_completed=False, succeeded=False,
                                error_type="RuntimeError", error_detail="setter failed")
        row = attempt.reject(
            terminal_stage="setter_application",
            rejection_code="setter_application_error",
            rejection_detail="RuntimeError: setter failed",
            environment_mutation_status="confirmed",
        )

        self.assertEqual([call["ordinal"] for call in row["setter_calls"]], [0, 1])
        self.assertTrue(row["setter_calls"][0]["succeeded"])
        self.assertFalse(row["setter_calls"][1]["call_completed"])
        self.assertTrue(row["quarantined"])
        self.assertEqual(validate_attempt_records([row]), ())

    def test_first_runtime_call_failure_is_unknown_and_quarantined(self):
        ledger = ActionAttemptLedger()
        attempt = ledger.begin(identity(run_role="fixture"))
        attempt.candidate_built({"action": 1})
        attempt.contract_validated()
        attempt.pre_setter_revalidated()
        ordinal = attempt.setter_started(setter_kind="rb", task_id="task-1")
        attempt.setter_finished(
            ordinal,
            call_completed=False,
            succeeded=False,
            error_type="RuntimeError",
            error_detail="unknown mutation",
        )
        row = attempt.reject(
            terminal_stage="setter_application",
            rejection_code="setter_application_error",
            rejection_detail="RuntimeError: unknown mutation",
            environment_mutation_status="unknown_after_runtime_call",
        )

        self.assertTrue(row["quarantined"])
        self.assertEqual(row["environment_mutation_status"], "unknown_after_runtime_call")
        self.assertEqual(validate_attempt_records([row]), ())

    def test_pre_runtime_rejection_has_null_digest_when_build_never_completed(self):
        ledger = ActionAttemptLedger()
        attempt = ledger.begin(identity())
        row = attempt.reject(
            terminal_stage="candidate_build",
            rejection_code="candidate_build_error",
            rejection_detail="ValueError: no candidate",
            environment_mutation_status="none",
        )

        self.assertIsNone(row["candidate_digest"])
        self.assertEqual(row["setter_calls"], [])
        self.assertFalse(row["env_step_called"])
        self.assertFalse(row["quarantined"])
        self.assertEqual(validate_attempt_records([row]), ())

    def test_rejection_requires_reason_and_consistent_mutation_quarantine(self):
        ledger = ActionAttemptLedger()
        attempt = ledger.begin(identity())
        with self.assertRaises(LedgerContractError):
            attempt.reject(
                terminal_stage="candidate_build",
                rejection_code="",
                rejection_detail="missing code",
                environment_mutation_status="none",
            )
        with self.assertRaises(LedgerContractError):
            attempt.reject(
                terminal_stage="candidate_build",
                rejection_code="candidate_build_error",
                rejection_detail="bad mutation claim",
                environment_mutation_status="confirmed",
            )


class TerminalRecordValidationTests(unittest.TestCase):
    def test_validator_detects_tampered_success_and_mutation_fields(self):
        row = accepted_record(ActionAttemptLedger(), identity())
        cases = []
        missing_step = copy.deepcopy(row)
        missing_step["env_step_completed"] = False
        cases.append(missing_step)
        missing_digest = copy.deepcopy(row)
        missing_digest["candidate_digest"] = None
        cases.append(missing_digest)
        quarantined = copy.deepcopy(row)
        quarantined["quarantined"] = True
        cases.append(quarantined)
        skipped = copy.deepcopy(row)
        skipped["stage_trace"].remove("contract_validated")
        cases.append(skipped)

        for tampered in cases:
            with self.subTest(tampered=tampered):
                self.assertTrue(validate_attempt_records([tampered]))

    def test_validator_detects_duplicate_ids_and_binary_conservation(self):
        row = accepted_record(ActionAttemptLedger(), identity())
        errors = validate_attempt_records([row, copy.deepcopy(row)])
        self.assertTrue(any("duplicate attempt_id" in error for error in errors))

        summary = summarize_attempts([row])
        natural = summary["by_run_role"]["natural_reference"]
        self.assertEqual(natural["attempt_count"], 1)
        self.assertEqual(natural["accepted_count"], 1)
        self.assertEqual(natural["rejected_count"], 0)
        self.assertEqual(natural["rejection_rate"], 0.0)
        self.assertTrue(summary["binary_conservation_passed"])

    def test_summary_separates_all_roles_and_never_mixes_main_rate(self):
        ledger = ActionAttemptLedger()
        rows = [accepted_record(ledger, identity())]
        for role, frame in (("natural_replay", 0), ("fixture", 1), ("bootstrap", 2)):
            attempt = ledger.begin(identity(run_role=role, frame_index=frame))
            rows.append(
                attempt.reject(
                    terminal_stage="candidate_build",
                    rejection_code="controlled_failure",
                    rejection_detail=f"{role} failure",
                    environment_mutation_status="none",
                )
            )

        summary = summarize_attempts(rows)
        self.assertEqual(summary["natural_reference_rejection_rate"], 0.0)
        self.assertEqual(summary["by_run_role"]["natural_reference"]["attempt_count"], 1)
        for role in ("natural_replay", "fixture", "bootstrap"):
            self.assertEqual(summary["by_run_role"][role]["rejection_rate"], 1.0)
        self.assertTrue(summary["binary_conservation_passed"])


if __name__ == "__main__":
    unittest.main()
