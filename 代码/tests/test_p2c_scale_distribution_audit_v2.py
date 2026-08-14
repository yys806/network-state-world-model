from __future__ import annotations

import copy
import hashlib
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
)
from pi_jwm.full_dual_graph_artifact_v2 import publish_success_bundle  # noqa: E402
from pi_jwm.p2c_scale_distribution_audit_v2 import (  # noqa: E402
    AUDIT_SCHEMA_VERSION,
    AuditContractError,
    audit_bundle,
)


P2B_V1_BUNDLE = (
    CODE_ROOT
    / "artifacts"
    / "preflight"
    / "pi_jwm_p2_full_dual_graph_collector_v1"
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def accepted_attempt(ledger, *, role, episode_id, trajectory_id, frame):
    attempt = ledger.begin(
        AttemptIdentity(
            run_role=role,
            episode_id=episode_id,
            trajectory_id=trajectory_id,
            frame_index=int(frame["frame_index"]),
            candidate_ordinal=0,
        )
    )
    attempt.candidate_built(frame["action"])
    attempt.contract_validated()
    attempt.pre_setter_revalidated()
    attempt.setters_applied()
    attempt.env_step_started()
    attempt.env_step_completed()
    attempt.outcome_captured()
    return attempt.accept()


def build_real_shape_v2_payloads():
    payloads = {
        name: (
            load_jsonl(P2B_V1_BUNDLE / name)
            if name == "frames.jsonl"
            else load_json(P2B_V1_BUNDLE / name)
        )
        for name in (
            "collector_config.json",
            "vocabularies.json",
            "frames.jsonl",
            "coverage_report.json",
            "validation_report.json",
            "replay_report.json",
            "status_flags.json",
        )
    }
    payloads["collector_config.json"]["schema_version"] = (
        "PIJWM-Full-Collector-Preflight-v2"
    )
    payloads["collector_config.json"]["formal_data_approved"] = False
    payloads["collector_config.json"]["training_eligible"] = False
    payloads["validation_report.json"]["formal_data_approved"] = False
    payloads["validation_report.json"]["training_eligible"] = False
    payloads["validation_report.json"]["action_rejection_count"] = 999

    episode_by_trajectory = {
        row["trajectory_id"]: row["episode_id"]
        for row in payloads["coverage_report.json"]["natural_episodes"]
    }
    ledger = ActionAttemptLedger()
    attempts = []
    for role in ("natural_reference", "natural_replay"):
        for frame in payloads["frames.jsonl"]:
            attempts.append(
                accepted_attempt(
                    ledger,
                    role=role,
                    episode_id=episode_by_trajectory[frame["trajectory_id"]],
                    trajectory_id=frame["trajectory_id"],
                    frame=frame,
                )
            )
    payloads["action_attempts.jsonl"] = attempts
    return payloads


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite_ledger_and_rehash(bundle: Path, rows) -> None:
    with (bundle / "action_attempts.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = bundle / "manifest.json"
    manifest = load_json(manifest_path)
    manifest["artifact_hashes"]["action_attempts.jsonl"] = sha256(
        bundle / "action_attempts.jsonl"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


class P2CScaleDistributionAuditV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.py"
        self.source.write_text("value = 1\n", encoding="utf-8")
        self.bundle = self.root / "p2b-v2"
        publish_success_bundle(
            self.bundle,
            build_real_shape_v2_payloads(),
            [self.source],
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_valid_ledger_ignores_forged_summary_and_keeps_three_blockers(self):
        report = audit_bundle(self.bundle, project_root=self.root)

        self.assertEqual(report["schema_version"], AUDIT_SCHEMA_VERSION)
        rejection = report["rejection_quarantine"]
        self.assertEqual(rejection["action_attempt_count"], 120)
        self.assertEqual(rejection["action_accepted_count"], 120)
        self.assertEqual(rejection["action_rejection_count"], 0)
        self.assertEqual(rejection["action_rejection_rate"], 0.0)
        self.assertEqual(rejection["reported_summary_rejection_count_ignored"], 999)
        self.assertNotIn(
            "action_rejection_rate_not_observed", report["blocking_reasons"]
        )
        self.assertEqual(
            set(report["blocking_reasons"]),
            {
                "scenario_matrix_not_frozen",
                "formal_scale_not_frozen",
                "formal_split_not_frozen",
            },
        )
        self.assertEqual(report["audit_status"], "blocked")
        self.assertFalse(report["candidate_formal_data_config"]["formal_data_approved"])

    def test_v1_bundle_without_ledger_cannot_claim_observed_rejection_rate(self):
        with self.assertRaisesRegex(AuditContractError, "action_attempts.jsonl"):
            audit_bundle(P2B_V1_BUNDLE, project_root=CODE_ROOT.parent)

    def test_duplicate_deleted_digest_role_and_mutation_tampering_are_rejected(self):
        original = load_jsonl(self.bundle / "action_attempts.jsonl")
        mutations = []

        duplicate = copy.deepcopy(original)
        duplicate.append(copy.deepcopy(duplicate[0]))
        mutations.append(duplicate)

        deleted = copy.deepcopy(original)
        deleted.pop(0)
        mutations.append(deleted)

        digest = copy.deepcopy(original)
        digest[0]["candidate_digest"] = "0" * 64
        mutations.append(digest)

        role = copy.deepcopy(original)
        role[120]["run_role"] = "fixture"
        mutations.append(role)

        mutation = copy.deepcopy(original)
        mutation[0]["environment_mutation_status"] = "none"
        mutations.append(mutation)

        for ordinal, rows in enumerate(mutations):
            case_bundle = self.root / f"case-{ordinal}"
            case_bundle.mkdir()
            for path in self.bundle.iterdir():
                (case_bundle / path.name).write_bytes(path.read_bytes())
            rewrite_ledger_and_rehash(case_bundle, rows)
            with self.subTest(ordinal=ordinal), self.assertRaises(AuditContractError):
                audit_bundle(case_bundle, project_root=self.root)


if __name__ == "__main__":
    unittest.main()
