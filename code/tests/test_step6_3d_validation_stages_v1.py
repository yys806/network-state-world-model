"""Synthetic in-memory orchestration oracles; never formal Validation outcomes."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/scripts"), str(ROOT / "code/src")]
import run_step6_3d_formal_cpu_matrix_v1 as matrix

ANCHORS = tuple(f"TRAIN_ONLY_SYNTHETIC_{i}" for i in range(64))
CONFIGS = {"HRS": (1, None), "S-CEM": (4, .1), "MH-CEM": (4, .1)}


def synthetic_case(anchor, method, seed, budget):
    number = ANCHORS.index(anchor) * 5 + matrix.VALIDATION_SEEDS.index(seed)
    low, high = [0, 0., 0., 0., 0.], [1, 0., 0., 0., 0.]
    pairs = ((low, None), (None, low), (low, high), (high, low), (low, low), (None, None))
    s, hrs = pairs[number % 6]
    objective = {"S-CEM": s, "HRS": hrs, "MH-CEM": low if number % 2 else high}[method]
    return {"synthetic_fixture": True, "sample_id": anchor, "method": method, "seed": seed,
        "budget": budget, "iterations": CONFIGS[method][0], "elite_ratio": CONFIGS[method][1],
        "outcome": {"best_objective": objective, "h4_scoreable_count": int(objective is not None),
            "h4_unscoreable_count": 1, "complete_sequence_count": 3,
            "budget_receipt": {"B_WM": budget, "N_unique_transition_evals": budget,
                "N_complete_sequences": 3, "N_dead_end_branches": 3, "N_cache_hits": 4,
                "N_proposed_steps": 1026, "N_admitted_steps": 1025, "N_rejected_steps": 1,
                "wall_clock_seconds_diagnostic_only": 1.}},
        "score_residuals": {"H4_SUPPORT_BOUNDARY:UNSUPPORTED_FUTURE_RETURN_BIRTH:task:H1": 2,
                            "ScorerStateInconsistency:synthetic": 1},
        "support_horizon_counts": {"SCORER_EXCEPTION": 1}}


def cases(budget=1024):
    return {(budget, a, s, m): synthetic_case(a, m, s, budget)
        for a in ANCHORS for s in matrix.VALIDATION_SEEDS for m in CONFIGS}


class ValidationStageTests(unittest.TestCase):
    def test_cli_requires_an_explicit_validation_stage(self):
        for args in (("--phase", "validation"), ("--phase", "train", "--validation-stage", "primary")):
            with patch.object(sys, "argv", ["matrix", *args]), self.assertRaises(SystemExit) as error:
                matrix.main()
            self.assertEqual(error.exception.code, 2)

    def test_primary_and_diagnostic_have_disjoint_budgets(self):
        self.assertEqual(matrix.validation_stage_budgets("primary"), (1024,))
        self.assertEqual(matrix.validation_stage_budgets("diagnostic"), (256, 512))
        with self.assertRaises(ValueError):
            matrix.validation_stage_budgets("all")

    def test_six_categories_and_residual_diagnostics(self):
        summary = matrix.summarize_validation_budget(cases(), ANCHORS, matrix.VALIDATION_SEEDS, 1024)
        paired = summary["paired_outcomes"]["S-CEM_vs_HRS"]
        self.assertEqual(list(paired["six_category_counts"].values()), [54, 54, 53, 53, 53, 53])
        self.assertEqual(paired["total_paired_cases"], 320)
        self.assertEqual((paired["win"], paired["tie"], paired["loss"]), (107, 106, 107))
        self.assertEqual(len(paired["per_anchor_seed_outcomes"]), 64)
        self.assertTrue(all(len(row) == 5 for row in paired["per_anchor_seed_outcomes"].values()))
        d = summary["method_diagnostics"]["S-CEM"]
        self.assertEqual(d["return_birth_support_boundary_count"], 640)
        self.assertEqual(d["scorer_exception_count"], 320)
        self.assertEqual(d["grammar_dead_end_count"], 960)
        self.assertEqual(d["N_unique_transition_evals"], 327680)

    def test_missing_pairs_and_wrong_anchor_count_rejected(self):
        incomplete = cases()
        incomplete.pop(next(iter(incomplete)))
        with self.assertRaises(ValueError):
            matrix.summarize_validation_budget(incomplete, ANCHORS, matrix.VALIDATION_SEEDS, 1024)
        with self.assertRaises(ValueError):
            matrix.summarize_validation_budget(cases(), ANCHORS[:-1], matrix.VALIDATION_SEEDS, 1024)
        inconsistent = cases()
        next(iter(inconsistent.values()))["support_horizon_counts"]["SCORER_EXCEPTION"] = 2
        with self.assertRaisesRegex(ValueError, "scorer exception diagnostics disagree"):
            matrix.summarize_validation_budget(inconsistent, ANCHORS, matrix.VALIDATION_SEEDS, 1024)

    def test_stage_a_stops_and_stage_b_preserves_selection(self):
        execution = {"source_sha256": {"runner": "synthetic"}, "locked_test": False}
        calls = []
        def solve(phase, anchor, method, seed, budget, k, rho, source, identity):
            calls.append(budget)
            return synthetic_case(anchor, method, seed, budget)
        with tempfile.TemporaryDirectory() as temp, patch.object(matrix, "OUT", Path(temp)), \
             patch.object(matrix, "solve_or_resume", solve):
            matrix.execute_validation_stage(execution, "primary", ANCHORS, CONFIGS,
                execution["source_sha256"], {"synthetic_fixture": True})
            self.assertEqual(len(calls), 960)
            self.assertEqual(set(calls), {1024})
            selection = Path(temp) / "08_selected_method.json"
            before = selection.read_bytes()
            calls.clear()
            matrix.execute_validation_stage(execution, "diagnostic", ANCHORS, CONFIGS,
                execution["source_sha256"], {"synthetic_fixture": True})
            self.assertEqual(len(calls), 1920)
            self.assertEqual(set(calls), {256, 512})
            self.assertEqual(selection.read_bytes(), before)
            self.assertTrue((Path(temp) / "07_validation_stage_a_primary_comparison_receipt.json").exists())
            chosen = matrix.read(selection)
            primary = matrix.read(Path(temp) / "07_validation_stage_a_primary_comparison_receipt.json")
            ci = {(a,b):primary["comparison"]["1024"]["paired_outcomes"][f"{a}_vs_{b}"]["cluster_bootstrap"]
                for a,b in matrix.PAIRS}
            self.assertEqual(chosen["selected_method"], matrix.select_method(ci))
            matrix.write_atomic(selection, {**chosen, "primary_comparison_sha256": "stale"})
            calls.clear()
            with self.assertRaisesRegex(ValueError, "Stage A method selection artifacts disagree"):
                matrix.execute_validation_stage(execution, "diagnostic", ANCHORS, CONFIGS,
                    execution["source_sha256"], {"synthetic_fixture": True})
            self.assertEqual(calls, [])

    def test_stage_b_requires_primary_review_artifacts(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(matrix, "OUT", Path(temp)), \
             patch.object(matrix, "solve_or_resume") as solve:
            with self.assertRaises(ValueError):
                matrix.execute_validation_stage({"locked_test": False}, "diagnostic", ANCHORS, CONFIGS, {}, {})
            solve.assert_not_called()

    def test_resume_identity_fields_and_skip_correct_case(self):
        execution = matrix.execution_identity(device="cuda", gpu_model="NVIDIA GeForce RTX 3080 Ti",
            batch_size=16, checkpoint_sha256="checkpoint", source_sha256={"runner": "new"})
        fields = dict(sample_id="TRAIN_ONLY_SYNTHETIC", method="HRS", seed=6301,
            budget=1024, iterations=1, elite_ratio=None)
        correct = {**fields, **execution}
        with tempfile.TemporaryDirectory() as temp, patch.object(matrix, "RESULTS", Path(temp)), \
             patch.object(matrix, "run") as run:
            path = matrix.result_path("synthetic", fields["sample_id"], "HRS", 6301, 1024, 1, None)
            matrix.write_atomic(path, correct)
            self.assertEqual(matrix.solve_or_resume("synthetic", fields["sample_id"], "HRS", 6301, 1024,
                1, None, execution["source_sha256"], execution), correct)
            run.assert_not_called()
            for key, bad in {"source_sha256": {}, "execution_config_id": "bad", "execution_device": "cpu",
                    "gpu_model": "4090", "batch_size": 32, "checkpoint_sha256": "wrong"}.items():
                matrix.write_atomic(path, {**correct, key: bad})
                with self.assertRaises(ValueError):
                    matrix.solve_or_resume("synthetic", fields["sample_id"], "HRS", 6301, 1024,
                        1, None, execution["source_sha256"], execution)
            run.assert_not_called()

    def test_new_validation_config_rejects_historical_identity(self):
        config_path = ROOT / "code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/03_validation_execution_config_3080ti.json"
        config = matrix.read(config_path)
        execution = {key: config[key] for key in matrix.EXECUTION_KEYS}
        matrix.validate_validation_provenance(config, execution)
        historical = matrix.read(matrix.QUALIFICATION)
        with self.assertRaises(ValueError):
            matrix.validate_validation_provenance(historical, execution)
        parents = {key: value.copy() for key, value in config["parents"].items()}
        parents["train_frozen_configs"]["sha256"] = "bad"
        with self.assertRaises(ValueError):
            matrix.validate_validation_provenance({**config, "parents": parents}, execution)


if __name__ == "__main__":
    unittest.main()
