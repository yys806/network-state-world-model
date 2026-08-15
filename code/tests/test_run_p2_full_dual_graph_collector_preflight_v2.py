from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

spec = importlib.util.spec_from_file_location(
    "p2_full_runner_v2",
    SCRIPTS_ROOT / "run_p2_full_dual_graph_collector_preflight_v2.py",
)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)

from pi_jwm.action_attempt_ledger_v1 import (  # noqa: E402
    ActionAttemptLedger,
    AttemptIdentity,
    validate_attempt_records,
)
from pi_jwm.full_dual_graph_collector_contract_v1 import (  # noqa: E402
    CollectorContractError,
    SnapshotPhase,
    TaskLifecycle,
)

from test_airfogsim_full_dual_graph_collector_v1 import (  # noqa: E402
    FakeEnv,
    FakeTask,
    SpySchedulers,
    phase_observer,
    physical_snapshot,
)
from test_full_dual_graph_artifact_v2 import passing_payloads  # noqa: E402


class RunnerContractTests(unittest.TestCase):
    def test_cli_and_canonical_request_have_no_expansive_options(self):
        parser = runner.build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertEqual(
            option_strings,
            {"-h", "--help", "--output-dir", "--verify-only", "--seeds", "--steps"},
        )
        forbidden = ("gpu", "train", "locked", "dataset", "formal")
        self.assertFalse(any(token in option.lower() for option in option_strings for token in forbidden))
        self.assertEqual(runner.CANONICAL_SEEDS, (0, 1, 2))
        self.assertEqual(runner.NATURAL_ARMS, ("orthogonal", "interference_reuse"))
        runner.validate_run_request((0, 1, 2), 20, 0.1, 0.1)
        with self.assertRaises(ValueError):
            runner.validate_run_request((0, 1), 20, 0.1, 0.1)
        with self.assertRaises(ValueError):
            runner.validate_run_request((0, 1, 2), 19, 0.1, 0.1)

    def test_canonical_episode_matrix_is_three_seeds_by_two_arms(self):
        specs = runner.canonical_episode_specs(runner.CANONICAL_SEEDS)
        self.assertEqual(len(specs), 6)
        self.assertEqual(
            {(row["seed"], row["arm"]) for row in specs},
            {(seed, arm) for seed in runner.CANONICAL_SEEDS for arm in runner.NATURAL_ARMS},
        )
        self.assertTrue(all(row["training_eligible"] is False for row in specs))
        self.assertTrue(all(row["fixture"] is False for row in specs))

    def test_fixture_identity_separates_bootstrap_and_controlled_attempt(self):
        bootstrap = runner.fixture_attempt_identity(
            "wired_flow", seed=93, run_role="bootstrap", frame_index=0
        )
        fixture = runner.fixture_attempt_identity(
            "wired_flow", seed=93, run_role="fixture", frame_index=1
        )
        self.assertNotEqual(bootstrap.run_role, fixture.run_role)
        self.assertNotEqual(
            ActionAttemptLedger().begin_id(bootstrap),
            ActionAttemptLedger().begin_id(fixture),
        )
        self.assertEqual(bootstrap.candidate_ordinal, 0)
        self.assertEqual(fixture.candidate_ordinal, 0)


class AttemptFrameTests(unittest.TestCase):
    def setUp(self):
        self.task = FakeTask("task0", "v0", TaskLifecycle.WAITING_TO_OFFLOAD)
        self.snapshot = physical_snapshot([self.task])
        self.env = FakeEnv([self.task])
        self.schedulers = SpySchedulers()
        self.identity = AttemptIdentity(
            run_role="natural_reference",
            episode_id="episode-0",
            trajectory_id="traj0",
            frame_index=0,
            candidate_ordinal=0,
        )
        self.runtime_inputs = lambda env, snapshot: (
            {"v0": 2.0, "v1": 2.0, "u0": 5.0},
            {
                ("v0", "v1"): 1.0,
                ("v0", "u0"): 5.0,
                ("v1", "u0"): 4.0,
                ("u0", "v0"): 5.0,
            },
        )

    def test_observation_failure_is_run_level_and_does_not_enter_denominator(self):
        ledger = ActionAttemptLedger()

        def fail_observer(env, *, phase):
            del env, phase
            raise RuntimeError("decision snapshot unavailable")

        with self.assertRaisesRegex(RuntimeError, "decision snapshot unavailable"):
            runner.execute_attempt_frame(
                self.env,
                self.schedulers,
                identity=self.identity,
                seed=0,
                ledger=ledger,
                observer=fail_observer,
            )
        self.assertEqual(ledger.terminal_records(), [])
        self.assertEqual(self.schedulers.calls, [])
        self.assertEqual(self.env.step_calls, 0)

    def test_contract_and_candidate_build_failures_terminalize_without_digest(self):
        cases = (
            (
                CollectorContractError("rb_out_of_range", "bad candidate"),
                "contract_validation",
                "contract_validation_error",
            ),
            (ValueError("builder crashed"), "candidate_build", "candidate_build_error"),
        )
        for error, terminal_stage, rejection_code in cases:
            ledger = ActionAttemptLedger()

            def fail_builder(*args, **kwargs):
                del args, kwargs
                raise error

            with self.subTest(error=error), self.assertRaises(runner.PreflightRunFailure) as caught:
                runner.execute_attempt_frame(
                    self.env,
                    self.schedulers,
                    identity=self.identity,
                    seed=0,
                    ledger=ledger,
                    observer=phase_observer(self.snapshot, self.env),
                    builder=fail_builder,
                    runtime_inputs=self.runtime_inputs,
                )
            row = caught.exception.record
            self.assertEqual(row["terminal_stage"], terminal_stage)
            self.assertEqual(row["rejection_code"], rejection_code)
            self.assertIsNone(row["candidate_digest"])
            self.assertEqual(row["environment_mutation_status"], "none")
            self.assertEqual(validate_attempt_records(ledger.terminal_records()), ())

    def test_success_uses_real_builder_and_v2_executor_once(self):
        ledger = ActionAttemptLedger()
        built, result = runner.execute_attempt_frame(
            self.env,
            self.schedulers,
            identity=self.identity,
            seed=0,
            ledger=ledger,
            observer=phase_observer(self.snapshot, self.env),
            runtime_inputs=self.runtime_inputs,
        )

        self.assertEqual(built.action.frame_index, 0)
        self.assertTrue(result.stepped)
        self.assertEqual(self.env.step_calls, 1)
        rows = ledger.terminal_records()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["disposition"], "accepted")
        self.assertEqual(rows[0]["run_role"], "natural_reference")
        self.assertEqual(validate_attempt_records(rows), ())


class TopLevelPublicationTests(unittest.TestCase):
    def test_existing_failed_target_blocks_before_collector_is_called(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "candidate"
            (root / "candidate_failed").mkdir()
            calls = []

            def collector(ledger):
                calls.append(ledger)
                return passing_payloads()

            with self.assertRaises(FileExistsError):
                runner.run_and_publish(output, collector=collector, source_paths=[Path(__file__)])
            self.assertEqual(calls, [])

    def test_rejection_publishes_failure_once_and_never_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "candidate"
            source = root / "source.py"
            source.write_text("value = 1\n", encoding="utf-8")

            def collector(ledger):
                attempt = ledger.begin(
                    AttemptIdentity(
                        run_role="natural_reference",
                        episode_id="episode-0",
                        trajectory_id="traj0",
                        frame_index=0,
                        candidate_ordinal=0,
                    )
                )
                row = attempt.reject(
                    terminal_stage="candidate_build",
                    rejection_code="candidate_build_error",
                    rejection_detail="ValueError: controlled rejection",
                    environment_mutation_status="none",
                )
                raise runner.PreflightRunFailure(row, ValueError("controlled rejection"))

            result = runner.run_and_publish(output, collector=collector, source_paths=[source])
            self.assertFalse(output.exists())
            self.assertTrue((root / "candidate_failed").is_dir())
            self.assertEqual(result["published"], "failure")
            self.assertTrue(result["verification"]["passed"])
            rows = [
                json.loads(line)
                for line in (root / "candidate_failed" / "action_attempts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["disposition"], "rejected")

    def test_success_publishes_nine_files_and_recomputes_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "candidate"
            source = root / "source.py"
            source.write_text("value = 1\n", encoding="utf-8")

            result = runner.run_and_publish(
                output,
                collector=lambda ledger: passing_payloads(),
                source_paths=[source],
            )

            self.assertEqual(result["published"], "success")
            self.assertTrue(result["verification"]["passed"])
            self.assertEqual(len(list(output.iterdir())), 9)


class RealAirFogSimRoleTests(unittest.TestCase):
    def test_one_frame_reference_and_replay_have_distinct_ids_matching_digests(self):
        spec_row = runner.canonical_episode_specs(runner.CANONICAL_SEEDS)[0]
        ledger = ActionAttemptLedger()

        pair = runner.run_natural_replay_pair_v2(spec_row, steps=1, ledger=ledger)

        self.assertTrue(pair["comparison"]["passed"])
        rows = ledger.terminal_records()
        self.assertEqual(len(rows), 2)
        by_role = {row["run_role"]: row for row in rows}
        self.assertEqual(set(by_role), {"natural_reference", "natural_replay"})
        self.assertNotEqual(
            by_role["natural_reference"]["attempt_id"],
            by_role["natural_replay"]["attempt_id"],
        )
        self.assertEqual(
            by_role["natural_reference"]["candidate_digest"],
            by_role["natural_replay"]["candidate_digest"],
        )
        self.assertEqual(validate_attempt_records(rows), ())

    def test_real_fixture_matrix_records_exact_bootstrap_and_fixture_roles(self):
        ledger = ActionAttemptLedger()

        fixtures = runner.run_fixture_matrix_v2(ledger=ledger)

        self.assertEqual(set(fixtures), set(runner.REQUIRED_FIXTURES))
        self.assertTrue(all(row["passed"] is True for row in fixtures.values()))
        rows = ledger.terminal_records()
        self.assertEqual(len(rows), 2 * len(runner.REQUIRED_FIXTURES))
        self.assertEqual(
            sum(row["run_role"] == "bootstrap" for row in rows),
            len(runner.REQUIRED_FIXTURES),
        )
        self.assertEqual(
            sum(row["run_role"] == "fixture" for row in rows),
            len(runner.REQUIRED_FIXTURES),
        )
        self.assertTrue(all(row["disposition"] == "accepted" for row in rows))
        self.assertEqual(validate_attempt_records(rows), ())


if __name__ == "__main__":
    unittest.main()
