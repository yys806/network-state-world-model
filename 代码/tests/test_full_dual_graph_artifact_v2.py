from __future__ import annotations

import copy
import json
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
    summarize_attempts,
)
from pi_jwm.full_dual_graph_artifact_v1 import (  # noqa: E402
    build_full_collector_status_flags,
)
from pi_jwm.full_dual_graph_artifact_v2 import (  # noqa: E402
    ARTIFACT_CONTRACT_VERSION,
    FAILURE_REQUIRED_FILES,
    SUCCESS_REQUIRED_FILES,
    assert_publish_targets_absent,
    publish_failure_bundle,
    publish_success_bundle,
    validate_bundle_alignment,
    validate_success_payloads,
    verify_failure_bundle,
    verify_success_bundle,
)

from test_full_dual_graph_artifact_v1 import VOCABULARY, frames  # noqa: E402


def accepted_attempt(
    ledger: ActionAttemptLedger,
    *,
    run_role: str,
    frame_index: int,
    action: object,
) -> dict[str, object]:
    attempt = ledger.begin(
        AttemptIdentity(
            run_role=run_role,
            episode_id="episode-0",
            trajectory_id="traj0",
            frame_index=frame_index,
            candidate_ordinal=0,
        )
    )
    attempt.candidate_built(action)
    attempt.contract_validated()
    attempt.pre_setter_revalidated()
    attempt.setters_applied()
    attempt.env_step_started()
    attempt.env_step_completed()
    attempt.outcome_captured()
    return attempt.accept()


def rejected_attempt(*, run_role: str = "fixture") -> dict[str, object]:
    ledger = ActionAttemptLedger()
    attempt = ledger.begin(
        AttemptIdentity(
            run_role=run_role,
            episode_id="failed-episode",
            trajectory_id="failed-trajectory",
            frame_index=0,
            candidate_ordinal=0,
        )
    )
    return attempt.reject(
        terminal_stage="candidate_build",
        rejection_code="candidate_build_error",
        rejection_detail="ValueError: controlled failure",
        environment_mutation_status="none",
    )


def passing_payloads() -> dict[str, object]:
    frame_rows = frames()
    for frame in frame_rows:
        frame["trajectory_id"] = "traj0"
        frame["training_eligible"] = False
    ledger = ActionAttemptLedger()
    attempts = []
    for run_role in ("natural_reference", "natural_replay"):
        for frame in frame_rows:
            attempts.append(
                accepted_attempt(
                    ledger,
                    run_role=run_role,
                    frame_index=int(frame["frame_index"]),
                    action=frame["action"],
                )
            )
    summary = summarize_attempts(attempts)
    return {
        "collector_config.json": {
            "schema_version": "PIJWM-Full-Collector-Preflight-v2",
            "test_only": True,
            "seeds": [0],
            "steps": 2,
            "natural_arms": ["orthogonal"],
            "training_eligible": False,
            "formal_data_approved": False,
        },
        "vocabularies.json": {"traj0": VOCABULARY},
        "frames.jsonl": frame_rows,
        "action_attempts.jsonl": attempts,
        "coverage_report.json": {
            "natural_episodes": [{"episode_id": "episode-0", "trajectory_id": "traj0"}],
            "fixtures": {},
            "natural_and_fixture_reports_separate": True,
        },
        "validation_report.json": {
            "passed": True,
            "errors": [],
            "ledger_summary": summary,
            "training_eligible": False,
            "formal_data_approved": False,
        },
        "replay_report.json": {
            "passed": True,
            "episodes": {"episode-0": {"passed": True}},
            "fresh_environment_per_reference_and_replay": True,
        },
        "status_flags.json": build_full_collector_status_flags(passed=True),
    }


class V2AlignmentTests(unittest.TestCase):
    def test_exact_file_matrices_are_version_isolated(self):
        self.assertEqual(ARTIFACT_CONTRACT_VERSION, "PIJWM-Full-Dual-Graph-Artifact-v2")
        self.assertEqual(len(SUCCESS_REQUIRED_FILES), 9)
        self.assertEqual(len(FAILURE_REQUIRED_FILES), 3)
        self.assertIn("action_attempts.jsonl", SUCCESS_REQUIRED_FILES)

    def test_valid_alignment_recomputes_digest_and_ignores_summary_numbers(self):
        payloads = passing_payloads()
        payloads["validation_report.json"]["ledger_summary"] = {
            "natural_reference_rejection_rate": 0.75
        }

        self.assertEqual(validate_success_payloads(payloads), ())
        self.assertEqual(
            validate_bundle_alignment(
                payloads["frames.jsonl"], payloads["action_attempts.jsonl"]
            ),
            (),
        )

    def test_mapping_digest_role_retry_and_rejection_tampering_are_rejected(self):
        base = passing_payloads()
        cases = []

        wrong_frame = copy.deepcopy(base)
        wrong_frame["frames.jsonl"][0]["trajectory_id"] = "wrong"
        cases.append(wrong_frame)

        wrong_digest = copy.deepcopy(base)
        wrong_digest["action_attempts.jsonl"][0]["candidate_digest"] = "0" * 64
        cases.append(wrong_digest)

        hidden_retry = copy.deepcopy(base)
        hidden_retry["action_attempts.jsonl"][2]["candidate_ordinal"] = 1
        cases.append(hidden_retry)

        role_mix = copy.deepcopy(base)
        role_mix["action_attempts.jsonl"][2]["run_role"] = "fixture"
        cases.append(role_mix)

        rejected = copy.deepcopy(base)
        rejected["action_attempts.jsonl"].append(rejected_attempt())
        cases.append(rejected)

        for payloads in cases:
            with self.subTest(case=len(cases)):
                self.assertTrue(validate_success_payloads(payloads))


class V2PublicationTests(unittest.TestCase):
    def test_success_publication_is_atomic_verifiable_and_immutable(self):
        source = SRC_ROOT / "pi_jwm" / "full_dual_graph_artifact_v1.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "bundle"
            publish_success_bundle(output, passing_payloads(), [source])

            self.assertEqual(set(SUCCESS_REQUIRED_FILES), {path.name for path in output.iterdir()})
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], ARTIFACT_CONTRACT_VERSION)
            self.assertTrue(all(not Path(key).is_absolute() for key in manifest["source_hashes"]))
            self.assertTrue(all(not key.startswith(".worktrees/") for key in manifest["source_hashes"]))
            self.assertEqual(verify_success_bundle(output, [source]), {"passed": True, "errors": []})
            with self.assertRaises(FileExistsError):
                assert_publish_targets_absent(output)
            with self.assertRaises(FileExistsError):
                publish_success_bundle(output, passing_payloads(), [source])

    def test_success_verifier_detects_ledger_and_source_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            source.write_text("value = 1\n", encoding="utf-8")
            output = root / "bundle"
            publish_success_bundle(output, passing_payloads(), [source])
            with (output / "action_attempts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            self.assertFalse(verify_success_bundle(output, [source])["passed"])

            source.write_text("value = 2\n", encoding="utf-8")
            errors = verify_success_bundle(output, [source])["errors"]
            self.assertTrue(any("source hash mismatch" in error for error in errors))

    def test_failure_publication_has_three_files_and_blocks_both_destinations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            source.write_text("value = 1\n", encoding="utf-8")
            output = root / "candidate"
            attempt = rejected_attempt()
            failure_report = {
                "failure_scope": "attempt",
                "attempt_id": attempt["attempt_id"],
                "run_role": attempt["run_role"],
                "terminal_stage": attempt["terminal_stage"],
                "rejection_code": attempt["rejection_code"],
                "quarantined": attempt["quarantined"],
                "error_type": "ValueError",
                "error_detail": "controlled failure",
                "formal_data_approved": False,
                "training_eligible": False,
                "gpu_started": False,
                "locked_test_accessed": False,
            }

            failed = publish_failure_bundle(output, [attempt], failure_report, [source])
            self.assertEqual(failed, root / "candidate_failed")
            self.assertFalse(output.exists())
            self.assertEqual(set(FAILURE_REQUIRED_FILES), {path.name for path in failed.iterdir()})
            self.assertEqual(verify_failure_bundle(failed, [source]), {"passed": True, "errors": []})
            with self.assertRaises(FileExistsError):
                assert_publish_targets_absent(output)

    def test_missing_source_leaves_no_bundle_or_partial_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "bundle"
            with self.assertRaises(FileNotFoundError):
                publish_success_bundle(output, passing_payloads(), [root / "missing.py"])
            self.assertFalse(output.exists())
            self.assertFalse((root / "bundle_failed").exists())
            self.assertEqual(list(root.glob(".*.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
