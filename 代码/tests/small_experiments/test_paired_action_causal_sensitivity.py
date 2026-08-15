from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "small_experiments"
    / "paired_action_causal_sensitivity.py"
)


def load_subject():
    if not SCRIPT_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("exp05_subject", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def valid_pair_runs():
    base = {
        "seed": 0,
        "pre_intervention_state": {
            "time": 1.0,
            "task_id": "m0",
            "task_state": "waiting",
            "node_positions": {"p0": [0.0, 0.0, 0.0]},
        },
        "exogenous_trajectory": [
            {"step": 1, "positions": {"p0": [1.0, 0.0, 0.0]}},
            {"step": 2, "positions": {"p0": [2.0, 0.0, 0.0]}},
        ],
        "action": {
            "task_id": "m0",
            "target_node_id": "p1",
            "rb_indices": [0, 1],
        },
        "action_feasible": True,
        "action_applied": True,
        "successor_states": [],
    }
    left = copy.deepcopy(base)
    right = copy.deepcopy(base)
    left["variant"] = "left"
    right["variant"] = "right"
    right["action"]["target_node_id"] = "p2"
    return left, right


def pair_with_successor_effect():
    left, right = valid_pair_runs()
    states = []
    for step in range(1, 21):
        states.append(
            {
                "offset_step": step,
                "assigned_node_id": "p1",
                "current_node_id": "p0" if step < 2 else "p1",
                "lifecycle": "offloading" if step < 2 else "computing",
                "transmitted_data": float(step),
                "computed_data": float(max(step - 1, 0)),
                "active_link_count": 1 if step == 1 else 0,
                "rate_sum": 2.0 if step == 1 else 0.0,
                "rb_use": 2 if step == 1 else 0,
                "cpu_use": 1.0 if step >= 2 else 0.0,
                "completed": step >= 20,
                "delay": float(step) * 0.1,
            }
        )
    left["successor_states"] = copy.deepcopy(states)
    right["successor_states"] = copy.deepcopy(states)
    for row in right["successor_states"]:
        row["assigned_node_id"] = "p2"
        row["current_node_id"] = "p2" if row["offset_step"] >= 3 else "p0"
        row["transmitted_data"] += 0.5
        row["rate_sum"] += 1.0 if row["offset_step"] == 1 else 0.0
        row["delay"] += 0.05
    return left, right


class PairProtocolTests(unittest.TestCase):
    def setUp(self):
        self.subject = load_subject()

    def test_script_and_pair_validator_exist(self):
        self.assertTrue(SCRIPT_PATH.exists(), msg="exp05 script has not been implemented")
        validator = getattr(self.subject, "validate_pair", None) if self.subject else None
        self.assertTrue(callable(validator), msg="pair validator is missing")

    def test_valid_pair_has_equal_prestate_and_one_registered_action_change(self):
        validator = getattr(self.subject, "validate_pair", None)
        if not callable(validator):
            self.fail("pair validator is missing")
        left, right = valid_pair_runs()

        report = validator(
            left,
            right,
            {"pair_kind": "offload", "changed_action_field": "target_node_id"},
        )

        self.assertTrue(report["pair_valid"])
        self.assertEqual(["target_node_id"], report["changed_action_fields"])
        self.assertEqual(report["left_pre_hash"], report["right_pre_hash"])
        self.assertEqual(report["left_exogenous_hash"], report["right_exogenous_hash"])

    def test_prestate_mismatch_is_rejected(self):
        validator = getattr(self.subject, "validate_pair", None)
        left, right = valid_pair_runs()
        right["pre_intervention_state"]["time"] = 1.1

        report = validator(left, right, {"pair_kind": "offload", "changed_action_field": "target_node_id"})

        self.assertFalse(report["pair_valid"])
        self.assertIn("pre_intervention_mismatch", report["errors"])

    def test_two_action_components_changed_are_rejected(self):
        validator = getattr(self.subject, "validate_pair", None)
        left, right = valid_pair_runs()
        right["action"]["rb_indices"] = [2, 3]

        report = validator(left, right, {"pair_kind": "offload", "changed_action_field": "target_node_id"})

        self.assertFalse(report["pair_valid"])
        self.assertIn("unexpected_action_difference", report["errors"])

    def test_unapplied_or_infeasible_action_is_rejected(self):
        validator = getattr(self.subject, "validate_pair", None)
        left, right = valid_pair_runs()
        right["action_applied"] = False

        report = validator(left, right, {"pair_kind": "offload", "changed_action_field": "target_node_id"})

        self.assertFalse(report["pair_valid"])
        self.assertIn("action_not_applied", report["errors"])

    def test_exogenous_divergence_is_rejected(self):
        validator = getattr(self.subject, "validate_pair", None)
        left, right = valid_pair_runs()
        right["exogenous_trajectory"][1]["positions"]["p0"][0] = 2.1

        report = validator(left, right, {"pair_kind": "offload", "changed_action_field": "target_node_id"})

        self.assertFalse(report["pair_valid"])
        self.assertIn("exogenous_trajectory_mismatch", report["errors"])


class HorizonEffectTests(unittest.TestCase):
    def setUp(self):
        self.subject = load_subject()

    def test_k_1_5_20_effects_cover_task_link_and_resource_state(self):
        calculator = getattr(self.subject, "compute_horizon_effects", None)
        self.assertTrue(callable(calculator), msg="horizon effect calculator is missing")
        if not callable(calculator):
            return
        left, right = pair_with_successor_effect()

        effects = calculator(left, right, horizons=(1, 5, 20))

        self.assertEqual([1, 5, 20], [row["horizon"] for row in effects])
        self.assertTrue(effects[0]["assigned_node_changed"])
        self.assertAlmostEqual(-0.5, effects[1]["transmitted_data_delta"])
        self.assertTrue(all(row["any_successor_changed"] for row in effects))

    def test_valid_pair_with_changed_successor_passes_sensitivity_gate(self):
        calculator = getattr(self.subject, "compute_horizon_effects", None)
        validator = getattr(self.subject, "validate_action_sensitivity", None)
        self.assertTrue(callable(validator), msg="action sensitivity validator is missing")
        if not callable(calculator) or not callable(validator):
            return
        left, right = pair_with_successor_effect()
        pair_report = self.subject.validate_pair(
            left,
            right,
            {"pair_kind": "offload", "changed_action_field": "target_node_id"},
        )

        report = validator(pair_report, calculator(left, right))

        self.assertTrue(report["action_sensitivity_valid"])
        self.assertGreater(report["changed_horizon_count"], 0)

    def test_copied_successor_trajectory_is_rejected_even_when_actions_differ(self):
        calculator = getattr(self.subject, "compute_horizon_effects", None)
        validator = getattr(self.subject, "validate_action_sensitivity", None)
        left, right = pair_with_successor_effect()
        right["successor_states"] = copy.deepcopy(left["successor_states"])
        pair_report = self.subject.validate_pair(
            left,
            right,
            {"pair_kind": "offload", "changed_action_field": "target_node_id"},
        )

        report = validator(pair_report, calculator(left, right))

        self.assertFalse(report["action_sensitivity_valid"])
        self.assertIn("no_successor_effect", report["errors"])


class Exp05ExportAndCorruptionTests(unittest.TestCase):
    def setUp(self):
        self.subject = load_subject()

    def test_cli_exposes_seeds_time_and_output_arguments(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(0, result.returncode, msg=result.stderr)
        self.assertIn("--seeds", result.stdout)
        self.assertIn("--max-time", result.stdout)
        self.assertIn("--output-dir", result.stdout)

    def test_airfogsim_pair_runner_rejects_unknown_protocol_before_starting_simulator(self):
        runner = getattr(self.subject, "run_paired_seed", None)
        self.assertTrue(callable(runner), msg="AirFogSim paired replay runner is missing")
        if not callable(runner):
            return
        with self.assertRaises(ValueError):
            runner(0, 1.0, "unknown", "left")

    def test_destructive_pair_mutations_are_all_detected(self):
        builder = getattr(self.subject, "build_exp05_corruption_report", None)
        self.assertTrue(callable(builder), msg="exp05 corruption builder is missing")
        if not callable(builder):
            return
        left, right = pair_with_successor_effect()

        report = builder(
            left,
            right,
            {"pair_kind": "offload", "changed_action_field": "target_node_id"},
        )

        self.assertTrue(report["all_corruptions_detected"])
        self.assertEqual(5, len(report["cases"]))

    def test_fake_multiseed_pairs_are_frozen_with_effect_tables(self):
        runner = getattr(self.subject, "run_exp05", None)
        self.assertTrue(callable(runner), msg="exp05 export runner is missing")
        if not callable(runner):
            return

        def fake_pair_runner(seed, max_time, pair_kind, variant):
            left, right = pair_with_successor_effect()
            if pair_kind == "rb":
                left["action"]["target_node_id"] = "p1"
                right["action"]["target_node_id"] = "p1"
                left["action"]["rb_indices"] = [0, 1]
                right["action"]["rb_indices"] = [2, 3]
            selected = left if variant == "left" else right
            selected["seed"] = seed
            selected["max_time"] = max_time
            selected["pair_kind"] = pair_kind
            selected["variant"] = variant
            return selected

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result = runner(
                output_dir=output_dir,
                seeds=[0, 1],
                max_time=1.0,
                pair_runner=fake_pair_runner,
            )
            expected = {
                "bundle.json",
                "trajectories.json",
                "pair_reports.csv",
                "horizon_effects.csv",
                "validation_report.json",
                "corruption_report.json",
                "config_snapshot.json",
                "runtime_summary.json",
                "REPORT.md",
                "manifest.json",
            }
            self.assertTrue(result["action_sensitivity_ready"])
            self.assertTrue(result["experiment_completed"])
            self.assertEqual(2, result["valid_effective_pairs_by_kind"]["offload"])
            self.assertEqual(2, result["valid_effective_pairs_by_kind"]["rb"])
            self.assertEqual(expected, {path.name for path in output_dir.iterdir()})
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("AirFogSim-PIJWM-exp05-v1", manifest["schema_version"])
            self.assertIn("horizon_effects.csv", manifest["files"])
            self.assertIn(
                "src/pi_jwm/airfogsim_contract_adapter.py",
                manifest["source_code"]["files"],
            )
            summary = json.loads((output_dir / "runtime_summary.json").read_text(encoding="utf-8"))
            trajectories = json.loads((output_dir / "trajectories.json").read_text(encoding="utf-8"))
            self.assertEqual(16, summary["trajectory_count"])
            self.assertEqual(16, len(trajectories))
            self.assertIn("repeat_left_run_hash", json.loads((output_dir / "bundle.json").read_text(encoding="utf-8"))["pair_reports"][0])


if __name__ == "__main__":
    unittest.main()
