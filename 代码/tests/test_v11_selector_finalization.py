import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class SelectorSplitProtocolTest(unittest.TestCase):
    def test_default_seed_split_is_disjoint_and_complete_for_original_dataset(self):
        from pi_jwm.v11_selector import build_selector_split

        sample_seed = np.repeat(np.arange(60, dtype=np.int64), 3)
        split = build_selector_split(sample_seed)

        observed = {}
        for name, indices in split.items():
            observed[name] = set(sample_seed[np.asarray(indices, dtype=np.int64)].tolist())
        self.assertEqual(observed["train"], set(range(0, 16)) | set(range(20, 44)))
        self.assertEqual(observed["calibration"], set(range(44, 50)))
        self.assertEqual(observed["validation"], set(range(50, 60)))
        self.assertEqual(observed["background"], {16, 17})
        self.assertEqual(observed["matched_test"], {18, 19})
        names = list(observed)
        for left_idx, left in enumerate(names):
            for right in names[left_idx + 1 :]:
                self.assertFalse(observed[left] & observed[right])

    def test_test_indices_are_locked_until_configuration_is_frozen(self):
        from pi_jwm.v11_selector import SelectorProtocol

        protocol = SelectorProtocol(np.repeat(np.arange(60, dtype=np.int64), 2))
        with self.assertRaisesRegex(PermissionError, "freeze"):
            protocol.indices("matched_test")
        with self.assertRaisesRegex(PermissionError, "freeze"):
            protocol.indices("external_holdout")

        digest = protocol.freeze_configuration(
            {"hidden_dim": 64, "temperature": 0.25, "dropout": 0.1}
        )
        self.assertEqual(len(digest), 64)
        self.assertEqual(protocol.indices("matched_test").shape[0], 4)
        self.assertEqual(protocol.indices("external_holdout").shape[0], 0)
        with self.assertRaisesRegex(RuntimeError, "already frozen"):
            protocol.freeze_configuration({"hidden_dim": 128})

    def test_protocol_audit_rejects_future_or_identity_features(self):
        from pi_jwm.v11_selector import audit_selector_protocol

        good = audit_selector_protocol(
            feature_names=("predicted_activity_mean", "rb_total", "task_progress_proxy"),
            split_seed_sets={"train": {0, 1}, "validation": {2}, "matched_test": {3}},
        )
        self.assertTrue(good["passed"])

        bad = audit_selector_protocol(
            feature_names=("rb_total", "true_future_rate", "sample_seed"),
            split_seed_sets={"train": {0, 1}, "validation": {1, 2}},
        )
        self.assertFalse(bad["passed"])
        self.assertIn("true_future_rate", bad["forbidden_features"])
        self.assertIn("sample_seed", bad["forbidden_features"])
        self.assertEqual(bad["split_overlap_count"], 1)

    def test_sample_index_alignment_uses_sample_id_and_preserves_time(self):
        from pi_jwm.v11_selector import align_sample_index

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample_index.csv"
            path.write_text(
                "sample_id,seed,input_end_time,label_time\n"
                "2,0,1.0,1.5\n"
                "7,1,4.0,4.5\n",
                encoding="utf-8",
            )
            rows = align_sample_index(path, sample_ids=np.asarray([7, 2]))

        self.assertEqual(rows[0]["sample_id"], 7)
        self.assertEqual(rows[0]["seed"], 1)
        self.assertAlmostEqual(rows[0]["input_end_time"], 4.0)
        self.assertAlmostEqual(rows[1]["label_time"], 1.5)


class CandidateDataContractTest(unittest.TestCase):
    def test_selected_outcome_metrics_reconstruct_link_and_activity(self):
        from pi_jwm.v11_selector import CandidateOutcome, aggregate_selected_metrics

        outcome = CandidateOutcome(
            active_sse=np.asarray([[4.0, 1.0], [9.0, 16.0]], dtype=np.float32),
            active_count=np.asarray([1, 1]),
            link_sse=np.asarray([[10.0, 4.0], [6.0, 12.0]], dtype=np.float32),
            link_count=np.asarray([2, 2]),
            activity_tp=np.asarray([[1, 1], [1, 0]], dtype=np.int64),
            activity_fp=np.asarray([[1, 0], [0, 1]], dtype=np.int64),
            activity_fn=np.asarray([[0, 0], [0, 1]], dtype=np.int64),
            activity_tn=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        )
        metrics = aggregate_selected_metrics(outcome, np.asarray([1, 0]))

        self.assertAlmostEqual(metrics["active_rate_rmse"], np.sqrt(10 / 2))
        self.assertAlmostEqual(metrics["link_rmse"], np.sqrt(10 / 4))
        self.assertAlmostEqual(metrics["activity_f1"], 1.0)

    def test_candidate_batch_ablation_zeros_only_requested_observable_group(self):
        from pi_jwm.v11_selector import CandidateBatch, ablate_candidate_batch

        batch = CandidateBatch(
            context=np.ones((2, 4), dtype=np.float32),
            candidate_features=np.ones((2, 3, 4), dtype=np.float32),
            candidate_mask=np.ones((2, 3), dtype=bool),
            stage=np.asarray(["offload", "compute"]),
            feature_names=("rb_total_sum", "predicted_task_8", "predicted_energy_proxy", "predicted_rate_sum"),
        )
        task = ablate_candidate_batch(batch, "task")
        energy = ablate_candidate_batch(batch, "energy")
        stage = ablate_candidate_batch(batch, "stage")

        self.assertTrue(np.all(task.candidate_features[:, :, 1] == 0.0))
        self.assertTrue(np.all(task.candidate_features[:, :, 0] == 1.0))
        self.assertTrue(np.all(energy.candidate_features[:, :, 2] == 0.0))
        self.assertEqual(set(stage.stage.tolist()), {"unknown"})

    def test_candidate_batch_and_outcome_reject_shape_mismatch(self):
        from pi_jwm.v11_selector import CandidateBatch, CandidateOutcome

        with self.assertRaisesRegex(ValueError, "candidate"):
            CandidateBatch(
                context=np.zeros((2, 3), dtype=np.float32),
                candidate_features=np.zeros((2, 4, 5), dtype=np.float32),
                candidate_mask=np.ones((2, 3), dtype=bool),
                stage=np.asarray(["offload", "compute"]),
                feature_names=("a", "b", "c", "d", "e"),
            )
        with self.assertRaisesRegex(ValueError, "active_sse"):
            CandidateOutcome(
                active_sse=np.ones((2, 4), dtype=np.float32),
                active_count=np.ones((3,), dtype=np.int64),
            )

    def test_candidate_outcome_reconstructs_improvement_and_regret(self):
        from pi_jwm.v11_selector import CandidateOutcome

        outcome = CandidateOutcome(
            active_sse=np.asarray([[9.0, 4.0, 16.0], [0.0, 0.0, 0.0]], dtype=np.float32),
            active_count=np.asarray([1, 0], dtype=np.int64),
            default_index=0,
        )
        np.testing.assert_allclose(outcome.improvement[0], np.asarray([0.0, 1.0, -1.0]))
        np.testing.assert_allclose(outcome.regret[0], np.asarray([1.0, 0.0, 2.0]))
        self.assertTrue(np.isnan(outcome.sample_rmse[1]).all())

    def test_candidate_projection_enforces_resource_and_stage_constraints(self):
        from pi_jwm.v11_selector import project_candidate_actions

        baseline = np.zeros((2, 2, 1, 6), dtype=np.float32)
        actions = np.zeros((2, 3, 2, 1, 6), dtype=np.float32)
        actions[..., 1] = 1.0
        actions[..., 2] = -4.0
        actions[..., 3] = 1.0
        actions[..., 4] = 5.0
        families = ("rb_repair", "compute_cpu_low", "return_route_high")
        stages = np.asarray(["compute", "return"])

        projected, applied = project_candidate_actions(
            actions,
            baseline_actions=baseline,
            valid_element_mask=np.ones((2, 2, 1), dtype=bool),
            candidate_families=families,
            stages=stages,
        )

        self.assertTrue(np.all(projected >= 0.0))
        self.assertTrue(np.all(projected[..., 2][projected[..., 1] <= 0.0] == 0.0))
        self.assertTrue(np.all(projected[1, 1] == baseline[1]))
        self.assertTrue(np.all(projected[0, 2] == baseline[0]))
        self.assertFalse(applied[1, 1])
        self.assertFalse(applied[0, 2])

    def test_candidate_gate_reports_headroom_and_coverage(self):
        from pi_jwm.v11_selector import audit_candidate_library

        sse = np.asarray(
            [[100.0, 25.0, 64.0], [100.0, 81.0, 49.0], [100.0, 100.0, 100.0]],
            dtype=np.float32,
        )
        report = audit_candidate_library(
            active_sse=sse,
            active_count=np.ones((3,), dtype=np.int64),
            action_applied=np.asarray(
                [[False, True, True], [False, True, True], [False, True, True]], dtype=bool
            ),
            identity_index=0,
            oracle_rmse_threshold=10.0,
            min_nontrivial_ratio=0.6,
            max_identity_win_ratio=0.65,
        )
        self.assertAlmostEqual(report["sample_oracle_rmse"], np.sqrt((25 + 49 + 100) / 3))
        self.assertAlmostEqual(report["nontrivial_ratio"], 2 / 3)
        self.assertAlmostEqual(report["identity_oracle_win_ratio"], 1 / 3)
        self.assertEqual(report["action_applied_ratio"], 1.0)
        self.assertTrue(report["passed"])

    def test_candidate_gate_retains_zero_active_group_as_failed_audit(self):
        from pi_jwm.v11_selector import audit_candidate_library

        report = audit_candidate_library(
            active_sse=np.zeros((2, 3), dtype=np.float32),
            active_count=np.zeros((2,), dtype=np.int64),
            action_applied=np.asarray([[False, True, True], [False, True, True]], dtype=bool),
            identity_index=0,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["failure_reason"], "no_active_targets")
        self.assertEqual(report["num_valid_samples"], 0)

    def test_candidate_gate_counts_applicable_noop_actions_as_failures(self):
        from pi_jwm.v11_selector import audit_candidate_library

        report = audit_candidate_library(
            active_sse=np.asarray([[100.0, 25.0, 64.0]], dtype=np.float32),
            active_count=np.ones((1,), dtype=np.int64),
            action_applied=np.asarray([[False, True, False]], dtype=bool),
            applicability_mask=np.asarray([[True, True, True]], dtype=bool),
            candidate_mask=np.asarray([[True, True, False]], dtype=bool),
            identity_index=0,
            oracle_rmse_threshold=20.0,
            min_nontrivial_ratio=0.0,
            max_identity_win_ratio=1.0,
        )

        self.assertAlmostEqual(report["action_applied_ratio"], 0.5)
        self.assertFalse(report["passed"])


class SupportConstrainedCandidateLibraryTest(unittest.TestCase):
    def _inputs(self):
        baseline = np.zeros((2, 3, 40, 6), dtype=np.float32)
        baseline[..., 1] = 1.0
        baseline[..., 2] = 2.0
        score = np.broadcast_to(
            np.linspace(0.0, 1.0, 40, dtype=np.float32)[None, None, :], (2, 3, 40)
        ).copy()
        value_head = np.full((2, 3, 40), 8.0, dtype=np.float32)
        support = np.ones((2, 3, 40), dtype=bool)
        valid = np.ones((2, 3, 40), dtype=bool)
        stages = np.asarray(["compute", "return"])
        return baseline, score, value_head, support, valid, stages

    def test_candidate_library_has_fixed_bounded_composition(self):
        from pi_jwm.v11_candidates import build_support_constrained_candidates

        baseline, score, value_head, support, valid, stages = self._inputs()
        library = build_support_constrained_candidates(
            baseline_actions=baseline,
            selection_score=score,
            value_head=value_head,
            train_positive_quantiles={0.5: 5.0, 0.75: 9.0},
            support_mask=support,
            valid_element_mask=valid,
            stages=stages,
        )

        self.assertLessEqual(len(library.candidate_names), 32)
        self.assertEqual(library.candidate_names[0], "identity")
        self.assertIn("ranked_allocation_baseline", library.candidate_names)
        for k in (8, 16, 32):
            for magnitude in ("value_head", "q50", "q75"):
                for pattern in ("persistent", "decayed"):
                    self.assertIn(f"rb_repair__k{k}__{magnitude}__{pattern}", library.candidate_names)
        for name in (
            "offload_rb_low",
            "offload_rb_high",
            "compute_cpu_low",
            "compute_cpu_high",
            "return_route_low",
            "return_route_high",
        ):
            self.assertIn(name, library.candidate_names)
        for name in (
            "benefit_residual__expand25__k8",
            "benefit_residual__expand50__k16",
            "benefit_residual__shrink25__k8",
            "benefit_residual__shrink50__k16",
        ):
            self.assertIn(name, library.candidate_names)
        self.assertEqual(len(library.candidate_names), 32)
        self.assertEqual(library.actions.shape[:2], (2, len(library.candidate_names)))
        self.assertEqual(library.candidate_mask.shape, library.actions.shape[:2])
        self.assertEqual(library.applicability_mask.shape, library.actions.shape[:2])

    def test_decayed_repair_changes_later_step_less_than_persistent(self):
        from pi_jwm.v11_candidates import build_support_constrained_candidates

        baseline, score, value_head, support, valid, stages = self._inputs()
        library = build_support_constrained_candidates(
            baseline,
            score,
            value_head,
            {0.5: 5.0, 0.75: 9.0},
            support,
            valid,
            stages,
        )
        persistent = library.actions[:, library.candidate_names.index("rb_repair__k8__q75__persistent")]
        decayed = library.actions[:, library.candidate_names.index("rb_repair__k8__q75__decayed")]
        persistent_delta = np.abs(persistent[..., 2] - baseline[..., 2]).sum(axis=2)
        decayed_delta = np.abs(decayed[..., 2] - baseline[..., 2]).sum(axis=2)

        np.testing.assert_allclose(persistent_delta[:, 0], 0.0)
        np.testing.assert_allclose(decayed_delta[:, 0], 0.0)
        self.assertTrue(np.all(decayed_delta[:, 2] < persistent_delta[:, 2]))
        np.testing.assert_allclose(decayed_delta[:, 1], persistent_delta[:, 1])

    def test_stage_candidates_are_masked_when_inapplicable_and_valid_actions_apply(self):
        from pi_jwm.v11_candidates import build_support_constrained_candidates

        baseline, score, value_head, support, valid, stages = self._inputs()
        library = build_support_constrained_candidates(
            baseline,
            score,
            value_head,
            {0.5: 5.0, 0.75: 9.0},
            support,
            valid,
            stages,
        )
        compute_idx = library.candidate_names.index("compute_cpu_high")
        return_idx = library.candidate_names.index("return_route_high")

        self.assertTrue(library.candidate_mask[0, compute_idx])
        self.assertFalse(library.candidate_mask[1, compute_idx])
        self.assertFalse(library.candidate_mask[0, return_idx])
        self.assertTrue(library.candidate_mask[1, return_idx])
        self.assertTrue(library.action_applied[0, compute_idx])
        self.assertTrue(library.action_applied[1, return_idx])
        np.testing.assert_allclose(library.actions[1, compute_idx], baseline[1])
        np.testing.assert_allclose(library.actions[0, return_idx], baseline[0])

    def test_candidate_with_no_requested_delta_is_not_marked_applicable(self):
        from pi_jwm.v11_candidates import build_support_constrained_candidates

        baseline, score, value_head, support, valid, stages = self._inputs()
        library = build_support_constrained_candidates(
            baseline,
            score,
            value_head,
            {0.5: 2.0, 0.75: 9.0},
            support,
            valid,
            stages,
        )
        candidate = library.candidate_names.index("rb_repair__k8__q50__persistent")

        self.assertFalse(np.any(library.applicability_mask[:, candidate]))
        self.assertFalse(np.any(library.candidate_mask[:, candidate]))

    def test_benefit_residual_candidates_expand_and_shrink_selected_values(self):
        from pi_jwm.v11_candidates import build_support_constrained_candidates

        baseline, score, value_head, support, valid, stages = self._inputs()
        library = build_support_constrained_candidates(
            baseline,
            score,
            value_head,
            {0.5: 5.0, 0.75: 9.0},
            support,
            valid,
            stages,
        )
        expanded = library.actions[
            :, library.candidate_names.index("benefit_residual__expand25__k8"), 1:, :, 2
        ]
        shrunk = library.actions[
            :, library.candidate_names.index("benefit_residual__shrink25__k8"), 1:, :, 2
        ]

        self.assertTrue(np.any(expanded > baseline[:, 1:, :, 2]))
        self.assertTrue(np.any(shrunk < baseline[:, 1:, :, 2]))

    def test_training_quantiles_ignore_zero_and_are_deterministic(self):
        from pi_jwm.v11_candidates import positive_value_quantiles

        values = np.asarray([0.0, -1.0, 1.0, 3.0, 9.0, 20.0], dtype=np.float32)
        first = positive_value_quantiles(values, quantiles=(0.5, 0.75), min_value=1.0)
        second = positive_value_quantiles(values, quantiles=(0.5, 0.75), min_value=1.0)

        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0.5], 6.0)
        self.assertAlmostEqual(first[0.75], 11.75)


class CandidateSetRankerTest(unittest.TestCase):
    def test_listwise_loss_prefers_lower_regret_candidate(self):
        from pi_jwm.v11_selector import listwise_regret_loss

        regret = torch.tensor([[2.0, 0.0, 1.0]], dtype=torch.float32)
        mask = torch.ones_like(regret, dtype=torch.bool)
        good = listwise_regret_loss(
            torch.tensor([[0.0, 3.0, 1.0]], dtype=torch.float32), regret, mask, temperature=0.25
        )
        bad = listwise_regret_loss(
            torch.tensor([[3.0, 0.0, 1.0]], dtype=torch.float32), regret, mask, temperature=0.25
        )
        self.assertLess(float(good), float(bad))

    def test_candidate_set_ranker_is_permutation_equivariant(self):
        from pi_jwm.v11_selector import CandidateSetBenefitRanker

        torch.manual_seed(7)
        model = CandidateSetBenefitRanker(candidate_dim=5, context_dim=3, hidden_dim=8, dropout=0.0)
        model.eval()
        candidates = torch.randn(2, 4, 5)
        context = torch.randn(2, 3)
        mask = torch.ones(2, 4, dtype=torch.bool)
        permutation = torch.tensor([2, 0, 3, 1])

        original = model(candidates, context, mask)
        permuted = model(candidates[:, permutation], context, mask[:, permutation])

        inverse = torch.argsort(permutation)
        for key in ("score", "predicted_improvement", "uncertainty"):
            torch.testing.assert_close(original[key], permuted[key][:, inverse])

    def test_fit_listwise_selector_is_deterministic_for_same_seed(self):
        from pi_jwm.v11_selector import (
            CandidateBatch,
            CandidateOutcome,
            fit_listwise_selector,
            predict_fitted_selector,
        )

        rng = np.random.default_rng(4)
        batch = CandidateBatch(
            context=rng.normal(size=(8, 3)).astype(np.float32),
            candidate_features=rng.normal(size=(8, 4, 5)).astype(np.float32),
            candidate_mask=np.ones((8, 4), dtype=bool),
            stage=np.asarray(["offload"] * 8),
            feature_names=("a", "b", "c", "d", "e"),
        )
        outcome = CandidateOutcome(
            active_sse=np.abs(rng.normal(size=(8, 4))).astype(np.float32),
            active_count=np.ones((8,), dtype=np.int64),
        )
        first = fit_listwise_selector(batch, outcome, hidden_dim=8, epochs=2, seed=17)
        second = fit_listwise_selector(batch, outcome, hidden_dim=8, epochs=2, seed=17)
        self.assertGreater(first.target_scale, 0.0)
        self.assertAlmostEqual(first.target_scale, second.target_scale)
        for left, right in zip(first.model.parameters(), second.model.parameters()):
            torch.testing.assert_close(left, right)

        from pi_jwm.v11_selector import predict_fitted_selector

        prediction = predict_fitted_selector(first, batch)
        self.assertEqual(prediction["predicted_improvement"].shape, (8, 4))
        self.assertEqual(prediction["uncertainty"].shape, (8, 4))
        self.assertTrue(np.all(np.isfinite(prediction["predicted_improvement"])))

    def test_fit_selector_normalizes_train_features_and_reuses_stats_for_prediction(self):
        from pi_jwm.v11_selector import (
            CandidateBatch,
            CandidateOutcome,
            fit_listwise_selector,
            predict_fitted_selector,
        )

        batch = CandidateBatch(
            context=np.asarray([[1.0, 1000.0], [3.0, 3000.0], [5.0, 5000.0]], dtype=np.float32),
            candidate_features=np.asarray(
                [
                    [[1.0, 10000.0], [2.0, 20000.0]],
                    [[3.0, 30000.0], [4.0, 40000.0]],
                    [[5.0, 50000.0], [6.0, 60000.0]],
                ],
                dtype=np.float32,
            ),
            candidate_mask=np.ones((3, 2), dtype=bool),
            stage=np.asarray(["offload", "compute", "compute"]),
            feature_names=("small", "large"),
            context_feature_names=("state_small", "state_large"),
        )
        outcome = CandidateOutcome(
            active_sse=np.asarray([[4.0, 1.0], [9.0, 4.0], [16.0, 9.0]], dtype=np.float32),
            active_count=np.ones((3,), dtype=np.int64),
        )

        fitted = fit_listwise_selector(batch, outcome, hidden_dim=8, epochs=2, seed=17)
        prediction = predict_fitted_selector(fitted, batch)

        self.assertGreater(float(fitted.candidate_scale[1]), 1000.0)
        self.assertGreater(float(fitted.context_scale[1]), 100.0)
        self.assertTrue(np.all(np.isfinite(prediction["score"])))

    def test_selector_checkpoint_loader_restores_model_and_calibration(self):
        from pi_jwm.v11_selector import CandidateSetBenefitRanker, load_fitted_selector_checkpoint

        torch.manual_seed(5)
        model = CandidateSetBenefitRanker(5, 3, hidden_dim=8, dropout=0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "candidate_dim": 5,
                    "context_dim": 3,
                    "hidden_dim": 8,
                    "dropout": 0.0,
                    "temperature": 0.25,
                    "training_seed": 17,
                    "calibration_bias": 1.25,
                    "target_scale": 2.5,
                    "candidate_mean": torch.arange(5, dtype=torch.float32),
                    "candidate_scale": torch.arange(5, dtype=torch.float32) + 1.0,
                    "context_mean": torch.arange(3, dtype=torch.float32),
                    "context_scale": torch.arange(3, dtype=torch.float32) + 1.0,
                    "configuration_digest": "d" * 64,
                    "history": [],
                },
                path,
            )
            fitted, bias, metadata = load_fitted_selector_checkpoint(
                path, expected_configuration_digest="d" * 64
            )

        self.assertAlmostEqual(bias, 1.25)
        self.assertAlmostEqual(fitted.target_scale, 2.5)
        np.testing.assert_allclose(fitted.candidate_mean, np.arange(5, dtype=np.float32))
        np.testing.assert_allclose(fitted.context_scale, np.arange(3, dtype=np.float32) + 1.0)
        self.assertEqual(metadata["training_seed"], 17)
        for expected, actual in zip(model.parameters(), fitted.model.parameters()):
            torch.testing.assert_close(expected, actual)

    def test_fit_selector_validates_seed_group_ids(self):
        from pi_jwm.v11_selector import CandidateBatch, CandidateOutcome, fit_listwise_selector

        batch = CandidateBatch(
            context=np.ones((3, 2), dtype=np.float32),
            candidate_features=np.ones((3, 2, 2), dtype=np.float32),
            candidate_mask=np.ones((3, 2), dtype=bool),
            stage=np.asarray(["offload", "compute", "return"]),
            feature_names=("a", "b"),
        )
        outcome = CandidateOutcome(
            active_sse=np.asarray([[2.0, 1.0], [3.0, 1.0], [4.0, 2.0]], dtype=np.float32),
            active_count=np.ones((3,), dtype=np.int64),
        )
        with self.assertRaisesRegex(ValueError, "group_ids"):
            fit_listwise_selector(batch, outcome, epochs=1, group_ids=np.asarray([0, 1]))


class CandidateLabelCacheTest(unittest.TestCase):
    def _payload(self):
        from pi_jwm.v11_selector import CandidateBatch, CandidateOutcome

        batch = CandidateBatch(
            context=np.ones((2, 3), dtype=np.float32),
            candidate_features=np.arange(40, dtype=np.float32).reshape(2, 4, 5),
            candidate_mask=np.ones((2, 4), dtype=bool),
            stage=np.asarray(["offload", "compute"]),
            feature_names=("rb_total", "cpu_total", "predicted_activity", "task_proxy", "energy_proxy"),
            candidate_names=("identity", "a", "b", "c"),
            context_feature_names=("state_node", "state_link", "state_task"),
        )
        outcome = CandidateOutcome(
            active_sse=np.arange(8, dtype=np.float32).reshape(2, 4),
            active_count=np.asarray([2, 3], dtype=np.int64),
            link_sse=np.arange(8, dtype=np.float32).reshape(2, 4) + 1.0,
            link_count=np.asarray([12, 12], dtype=np.int64),
            activity_tp=np.ones((2, 4), dtype=np.int64),
            activity_fp=np.zeros((2, 4), dtype=np.int64),
            activity_fn=np.ones((2, 4), dtype=np.int64),
            activity_tn=np.full((2, 4), 9, dtype=np.int64),
            action_applied=np.asarray(
                [[True, True, True, False], [True, True, False, True]], dtype=bool
            ),
        )
        return batch, outcome

    def test_label_cache_round_trip_keeps_manifest_and_arrays(self):
        from pi_jwm.v11_labeling import load_candidate_label_cache, save_candidate_label_cache

        batch, outcome = self._payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation_labels.npz"
            manifest = save_candidate_label_cache(
                path,
                split_name="validation",
                sample_ids=np.asarray([4, 9]),
                sample_seed=np.asarray([50, 51]),
                batch=batch,
                outcome=outcome,
                configuration_digest="a" * 64,
            )
            loaded_batch, loaded_outcome, loaded_manifest = load_candidate_label_cache(
                path, expected_configuration_digest="a" * 64
            )

        self.assertEqual(manifest["result_kind"], "diagnostic_only")
        self.assertEqual(loaded_manifest["split_name"], "validation")
        np.testing.assert_allclose(loaded_batch.candidate_features, batch.candidate_features)
        self.assertEqual(loaded_batch.context_feature_names, batch.context_feature_names)
        self.assertEqual(loaded_manifest["schema_version"], 5)
        np.testing.assert_allclose(loaded_outcome.active_sse, outcome.active_sse)
        np.testing.assert_allclose(loaded_outcome.link_sse, outcome.link_sse)
        np.testing.assert_array_equal(loaded_outcome.link_count, outcome.link_count)
        np.testing.assert_array_equal(loaded_outcome.activity_tp, outcome.activity_tp)
        np.testing.assert_array_equal(loaded_outcome.action_applied, outcome.action_applied)

    def test_matched_test_cache_requires_frozen_configuration_digest(self):
        from pi_jwm.v11_labeling import save_candidate_label_cache

        batch, outcome = self._payload()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(PermissionError, "frozen"):
                save_candidate_label_cache(
                    Path(directory) / "test.npz",
                    split_name="matched_test",
                    sample_ids=np.asarray([1, 2]),
                    sample_seed=np.asarray([18, 19]),
                    batch=batch,
                    outcome=outcome,
                    configuration_digest=None,
                )

    def test_cache_rejects_configuration_mismatch(self):
        from pi_jwm.v11_labeling import load_candidate_label_cache, save_candidate_label_cache

        batch, outcome = self._payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.npz"
            save_candidate_label_cache(
                path,
                split_name="train",
                sample_ids=np.asarray([1, 2]),
                sample_seed=np.asarray([0, 1]),
                batch=batch,
                outcome=outcome,
                configuration_digest="b" * 64,
            )
            with self.assertRaisesRegex(ValueError, "digest"):
                load_candidate_label_cache(path, expected_configuration_digest="c" * 64)

    def test_candidate_feature_builder_uses_only_predictions_and_actions(self):
        from pi_jwm.v11_labeling import build_candidate_feature_batch
        from pi_jwm.v11_selector import audit_selector_protocol

        actions = np.zeros((2, 3, 3, 4, 6), dtype=np.float32)
        actions[:, 1, :, :, 2] = 2.0
        actions[:, 2, :, :, 4] = 3.0
        predictions = []
        for candidate in range(3):
            predictions.append(
                {
                    "link_activity_prob": np.full((2, 3, 4, 1), 0.2 + candidate * 0.1, dtype=np.float32),
                    "link_rate_pred": np.full((2, 3, 4, 1), 4.0 + candidate, dtype=np.float32),
                    "task_pred": np.full((2, 3, 9), float(candidate), dtype=np.float32),
                }
            )
        features, names = build_candidate_feature_batch(
            actions,
            predictions,
            action_families=("identity", "rb_repair", "compute_cpu"),
        )

        self.assertEqual(features.shape[:2], (2, 3))
        self.assertEqual(features.shape[2], len(names))
        self.assertTrue(np.all(np.isfinite(features)))
        audit = audit_selector_protocol(names, {"train": {0}, "validation": {1}})
        self.assertTrue(audit["passed"], audit)
        self.assertIn("predicted_energy_proxy", names)
        self.assertIn("predicted_task_delta_8", names)

    def test_candidate_features_use_ranked_default_and_selected_edge_context(self):
        from pi_jwm.v11_labeling import build_candidate_feature_batch

        actions = np.zeros((1, 3, 2, 2, 6), dtype=np.float32)
        actions[:, 1, :, :, 2] = 2.0
        actions[:, 2, :, 0, 2] = 3.0
        predictions = []
        for rate in (4.0, 7.0, 9.0):
            predictions.append(
                {
                    "link_activity_prob": np.full((1, 2, 2, 1), rate / 10.0, dtype=np.float32),
                    "link_rate_pred": np.full((1, 2, 2, 1), rate, dtype=np.float32),
                    "task_pred": np.full((1, 2, 9), rate, dtype=np.float32),
                }
            )
        current_link = np.asarray([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float32)

        features, names = build_candidate_feature_batch(
            actions,
            predictions,
            action_families=("identity", "ranked", "rb_repair"),
            default_index=1,
            current_link_features=current_link,
            current_link_feature_names=("distance", "rate_sum"),
        )

        rate_delta = names.index("predicted_rate_delta_mean")
        modified = names.index("selected_modified_edge_count")
        selected_rate = names.index("selected_current_rate_sum_mean")
        self.assertAlmostEqual(float(features[0, 1, rate_delta]), 0.0)
        self.assertAlmostEqual(float(features[0, 2, rate_delta]), 2.0)
        self.assertAlmostEqual(float(features[0, 1, modified]), 0.0)
        self.assertAlmostEqual(float(features[0, 2, modified]), 2.0)
        self.assertAlmostEqual(float(features[0, 2, selected_rate]), 30.0)

    def test_observable_state_context_uses_history_only(self):
        from pi_jwm.v11_labeling import build_observable_state_context
        from pi_jwm.v11_selector import audit_selector_protocol

        node = np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2)
        link = np.arange(2 * 3 * 3 * 2, dtype=np.float32).reshape(2, 3, 3, 2)
        task = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
        action = np.zeros((2, 3, 3, 2), dtype=np.float32)
        context, names = build_observable_state_context(
            node,
            link,
            task,
            action,
            valid_edge_mask=np.asarray([True, False, True]),
            node_feature_names=("speed", "cpu"),
            link_feature_names=("distance", "rate_sum"),
            task_feature_names=("num_tasks", "num_finished"),
            action_feature_names=("offload_count", "rb_total"),
        )

        self.assertEqual(context.shape, (2, len(names)))
        self.assertTrue(np.all(np.isfinite(context)))
        self.assertTrue(audit_selector_protocol(names, {"train": {0}})["passed"])
        self.assertIn("state_task_num_finished_last", names)
        self.assertIn("state_link_rate_sum_last_mean", names)

    def test_outcome_metric_builder_keeps_per_sample_link_and_activity_counts(self):
        from pi_jwm.v11_labeling import compute_rollout_outcome_metrics

        predictions = {
            "link_rate_true": np.asarray([[[[2.0], [0.0]]], [[[1.0], [3.0]]]], dtype=np.float32),
            "link_rate_pred": np.asarray([[[[1.0], [2.0]]], [[[2.0], [1.0]]]], dtype=np.float32),
            "link_activity_true": np.asarray([[[[1.0], [0.0]]], [[[1.0], [1.0]]]], dtype=np.float32),
            "link_activity_prob": np.asarray([[[[0.8], [0.6]]], [[[0.4], [0.9]]]], dtype=np.float32),
        }

        metrics = compute_rollout_outcome_metrics(predictions, activity_threshold=0.5)

        np.testing.assert_allclose(metrics["link_sse"], [5.0, 5.0])
        np.testing.assert_array_equal(metrics["link_count"], [2, 2])
        np.testing.assert_array_equal(metrics["activity_tp"], [1, 1])
        np.testing.assert_array_equal(metrics["activity_fp"], [1, 0])
        np.testing.assert_array_equal(metrics["activity_fn"], [0, 1])
        np.testing.assert_array_equal(metrics["activity_tn"], [0, 0])


class DeferAndParetoTest(unittest.TestCase):
    def test_deploy_ranks_by_listwise_score_but_gates_with_calibrated_improvement(self):
        from pi_jwm.v11_selector import select_with_defer

        rank_scores = np.asarray(
            [[[0.0, 9.0, 2.0]], [[0.0, 8.5, 2.5]], [[0.0, 9.5, 1.5]]],
            dtype=np.float32,
        )
        improvements = np.asarray(
            [[[0.0, 1.0, 20.0]], [[0.0, 1.1, 20.0]], [[0.0, 0.9, 20.0]]],
            dtype=np.float32,
        )

        decision = select_with_defer(
            rank_scores,
            default_index=0,
            ensemble_improvement=improvements,
            ensemble_uncertainty=np.zeros_like(improvements),
        )

        self.assertEqual(decision.candidate_index[0], 1)
        self.assertAlmostEqual(float(decision.predicted_improvement[0]), 1.0, places=5)

    def test_deploy_defers_when_top_ranked_candidate_has_nonpositive_gain(self):
        from pi_jwm.v11_selector import select_with_defer

        rank_scores = np.asarray([[[0.0, 10.0, 5.0]]] * 3, dtype=np.float32)
        improvements = np.asarray([[[0.0, -0.1, 4.0]]] * 3, dtype=np.float32)

        decision = select_with_defer(
            rank_scores,
            default_index=0,
            ensemble_improvement=improvements,
        )

        self.assertEqual(decision.candidate_index[0], 0)
        self.assertTrue(decision.deferred[0])

    def test_observable_pareto_deltas_use_task_finished_and_energy_proxy(self):
        from pi_jwm.v11_selector import CandidateBatch, observable_pareto_deltas

        batch = CandidateBatch(
            context=np.zeros((1, 3), dtype=np.float32),
            candidate_features=np.asarray([[[4.0, 1.0, 0.0], [7.0, 3.0, 1.0]]], dtype=np.float32),
            candidate_mask=np.ones((1, 2), dtype=bool),
            stage=np.asarray(["offload"]),
            feature_names=("predicted_energy_proxy", "predicted_task_delta_8", "x"),
        )
        task, energy = observable_pareto_deltas(batch, default_index=0)

        np.testing.assert_allclose(task, [[0.0, 2.0]])
        np.testing.assert_allclose(energy, [[0.0, 3.0]])

    def test_select_with_defer_uses_lower_confidence_and_pareto_rules(self):
        from pi_jwm.v11_selector import select_with_defer

        ensemble_improvement = np.asarray(
            [
                [[0.0, 4.0, 8.0], [0.0, 1.0, 3.0]],
                [[0.0, 4.2, 8.1], [0.0, -1.0, 3.1]],
                [[0.0, 3.8, 7.9], [0.0, 0.2, 2.9]],
            ],
            dtype=np.float32,
        )
        task_delta = np.asarray([[0.0, 1.0, -1.0], [0.0, 1.0, 2.0]], dtype=np.float32)
        energy_delta = np.asarray([[0.0, -1.0, 2.0], [0.0, 1.0, -1.0]], dtype=np.float32)

        decision = select_with_defer(
            ensemble_improvement,
            default_index=0,
            task_delta=task_delta,
            energy_delta=energy_delta,
            z_value=1.64,
        )

        self.assertEqual(decision.candidate_index[0], 1)
        self.assertFalse(decision.deferred[0])
        self.assertEqual(decision.candidate_index[1], 2)
        self.assertFalse(decision.deferred[1])

    def test_select_with_defer_falls_back_when_gain_is_not_positive(self):
        from pi_jwm.v11_selector import select_with_defer

        ensemble = np.asarray(
            [[[0.0, 0.2]], [[0.0, -0.2]], [[0.0, 0.0]]], dtype=np.float32
        )
        decision = select_with_defer(ensemble, default_index=0)
        self.assertEqual(decision.candidate_index[0], 0)
        self.assertTrue(decision.deferred[0])
        self.assertEqual(decision.defer_reason[0], "nonpositive_lower_confidence_bound")

    def test_selector_decision_is_json_serializable(self):
        from pi_jwm.v11_selector import select_with_defer

        decision = select_with_defer(
            np.asarray([[[0.0, 2.0]], [[0.0, 2.0]], [[0.0, 2.0]]], dtype=np.float32),
            default_index=0,
        )
        json.dumps(decision.to_records(sample_ids=[9]))


class CandidateLabelRunnerContractTest(unittest.TestCase):
    def test_matched_test_request_requires_frozen_manifest(self):
        from run_v11_selector_candidate_labels import validate_requested_splits

        with self.assertRaisesRegex(PermissionError, "frozen"):
            validate_requested_splits(("matched_test",), frozen_manifest=None)
        validate_requested_splits(
            ("matched_test",),
            frozen_manifest={"configuration_digest": "f" * 64, "configuration_frozen": True},
        )

    def test_seed_balanced_limit_does_not_take_only_early_seed(self):
        from run_v11_selector_candidate_labels import limit_indices_seed_balanced

        seeds = np.repeat(np.asarray([0, 1, 2, 3]), 4)
        indices = np.arange(seeds.shape[0])
        selected = limit_indices_seed_balanced(indices, seeds, limit=8)

        self.assertEqual(selected.shape[0], 8)
        self.assertEqual(set(seeds[selected].tolist()), {0, 1, 2, 3})

    def test_scatter_edge_values_preserves_baseline_outside_coordinates(self):
        from run_v11_selector_candidate_labels import scatter_edge_values

        baseline = np.ones((2, 3, 4), dtype=np.float32)
        coordinates = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        values = np.asarray([7.0, 9.0], dtype=np.float32)
        dense = scatter_edge_values(baseline, coordinates, values)

        self.assertEqual(dense[0, 1, 2], 7.0)
        self.assertEqual(dense[1, 2, 3], 9.0)
        self.assertEqual(dense[0, 0, 0], 1.0)


class SelectorTrainingRunnerContractTest(unittest.TestCase):
    def test_choice_metrics_preserves_no_active_target_as_unscored(self):
        from pi_jwm.v11_selector import CandidateOutcome
        from train_v11_candidate_set_selector import _choice_metrics

        outcome = CandidateOutcome(
            active_sse=np.zeros((2, 2), dtype=np.float32),
            active_count=np.zeros((2,), dtype=np.int64),
        )
        metrics = _choice_metrics(
            outcome,
            choice=np.zeros((2,), dtype=np.int64),
            sample_seed=np.asarray([44, 45]),
            candidate_mask=np.ones((2, 2), dtype=bool),
        )
        self.assertIsNone(metrics["rmse"])
        self.assertIsNone(metrics["improvement_vs_default"])

    def test_failed_gate_downgrades_non_oracle_rows_to_diagnostic(self):
        from train_v11_candidate_set_selector import enforce_result_kinds

        rows = [
            {"model": "baseline", "result_kind": "deployable"},
            {"model": "oracle", "result_kind": "sample_oracle"},
        ]
        result = enforce_result_kinds(rows, gate_passed=False)
        self.assertEqual(result[0]["result_kind"], "diagnostic_only")
        self.assertEqual(result[1]["result_kind"], "sample_oracle")

    def test_masked_oracle_never_selects_unavailable_candidate(self):
        from train_v11_candidate_set_selector import masked_oracle_choice

        sse = np.asarray([[5.0, 1.0, 3.0], [5.0, 4.0, 0.0]], dtype=np.float32)
        mask = np.asarray([[True, False, True], [True, True, False]], dtype=bool)
        np.testing.assert_array_equal(masked_oracle_choice(sse, mask), np.asarray([2, 1]))

    def test_cache_protocol_requires_same_digest_and_candidate_order(self):
        from train_v11_candidate_set_selector import validate_cache_protocol

        manifests = {
            "train": {
                "split_name": "train",
                "configuration_digest": "a" * 64,
                "candidate_names": ["identity", "repair"],
                "feature_names": ["rb_total"],
            },
            "calibration": {
                "split_name": "calibration",
                "configuration_digest": "a" * 64,
                "candidate_names": ["identity", "repair"],
                "feature_names": ["rb_total"],
            },
            "validation": {
                "split_name": "validation",
                "configuration_digest": "b" * 64,
                "candidate_names": ["repair", "identity"],
                "feature_names": ["rb_total"],
            },
        }
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_cache_protocol(manifests)

        manifests["validation"]["configuration_digest"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "candidate"):
            validate_cache_protocol(manifests)

        manifests["validation"]["candidate_names"] = ["identity", "repair"]
        for manifest in manifests.values():
            manifest["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_cache_protocol(manifests, required_schema_version=3)

    def test_calibration_bias_uses_only_available_candidates(self):
        from train_v11_candidate_set_selector import calibrate_improvement_bias

        predicted = np.asarray([[2.0, 9.0], [4.0, 8.0]], dtype=np.float32)
        actual = np.asarray([[1.0, -20.0], [3.0, -20.0]], dtype=np.float32)
        mask = np.asarray([[True, False], [True, False]], dtype=bool)
        bias = calibrate_improvement_bias(predicted, actual, mask)

        self.assertAlmostEqual(bias, 1.0)

    def test_empty_smoke_calibration_forces_conservative_defer(self):
        from train_v11_candidate_set_selector import calibrate_improvement_bias

        predicted = np.asarray([[0.0, 2.0]], dtype=np.float32)
        actual = np.asarray([[np.nan, np.nan]], dtype=np.float32)
        bias = calibrate_improvement_bias(
            predicted,
            actual,
            np.ones((1, 2), dtype=bool),
            allow_empty=True,
        )
        self.assertGreater(bias, float(np.max(predicted)))

    def test_validation_config_tie_breaks_by_worst_seed_then_std(self):
        from train_v11_candidate_set_selector import choose_best_validation_config

        rows = [
            {"config_id": "a", "validation_rmse": 190.0, "worst_seed_regret": 8.0, "seed_std": 1.0},
            {"config_id": "b", "validation_rmse": 190.0, "worst_seed_regret": 4.0, "seed_std": 3.0},
            {"config_id": "c", "validation_rmse": 191.0, "worst_seed_regret": 0.0, "seed_std": 0.0},
        ]
        self.assertEqual(choose_best_validation_config(rows)["config_id"], "b")


class FrozenSelectorEvaluationContractTest(unittest.TestCase):
    def test_candidate_family_names_are_stable_for_group_reporting(self):
        from evaluate_v11_frozen_selector import infer_candidate_family

        self.assertEqual(infer_candidate_family("rb_repair__k8__q50__decayed"), "rb_repair")
        self.assertEqual(infer_candidate_family("offload_rb_high"), "offload_rb")
        self.assertEqual(infer_candidate_family("compute_cpu_low"), "compute_cpu")
        self.assertEqual(infer_candidate_family("return_route_high"), "return_route")
        self.assertEqual(infer_candidate_family("ranked_allocation_baseline"), "baseline")

    def test_access_ledger_rejects_reopening_same_locked_split(self):
        from evaluate_v11_frozen_selector import record_locked_split_access

        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "test_access_ledger.jsonl"
            record_locked_split_access(
                ledger,
                split_name="matched_test",
                configuration_digest="e" * 64,
                selector_freeze_digest="f" * 64,
                cache_sha256="a" * 64,
            )
            with self.assertRaisesRegex(PermissionError, "already accessed"):
                record_locked_split_access(
                    ledger,
                    split_name="matched_test",
                    configuration_digest="e" * 64,
                    selector_freeze_digest="f" * 64,
                    cache_sha256="a" * 64,
                )

    def test_external_evidence_is_recomputed_from_all_holdout_seeds(self):
        from evaluate_v11_frozen_selector import validate_external_evidence

        summary = {
            "split_name": "external_holdout",
            "configuration_digest": "e" * 64,
            "selector_freeze_digest": "f" * 64,
            "per_seed": [
                {
                    "seed": seed,
                    "active_rate_rmse": 190.0 if seed < 67 else 210.0,
                    "default_active_rate_rmse": 200.0,
                }
                for seed in range(60, 70)
            ],
        }
        wins = validate_external_evidence(summary, "e" * 64, "f" * 64)
        self.assertEqual(wins, 7)

        summary["per_seed"] = summary["per_seed"][:-1]
        with self.assertRaisesRegex(ValueError, "60-69"):
            validate_external_evidence(summary, "e" * 64, "f" * 64)

    def test_safety_evidence_derives_violations_from_actual_physical_rows(self):
        from evaluate_v11_frozen_selector import validate_safety_evidence

        audit = {
            "result_kind": "actual_airfogsim_safety_audit",
            "configuration_digest": "e" * 64,
            "selector_freeze_digest": "f" * 64,
            "rows": [
                {"task_utility_delta_actual": 1.0, "energy_delta_actual": 2.0},
                {"task_utility_delta_actual": -1.0, "energy_delta_actual": 3.0},
            ],
        }
        self.assertEqual(validate_safety_evidence(audit, "e" * 64, "f" * 64), 1)

    def test_selector_freeze_digest_rejects_tampered_checkpoint_record(self):
        from pi_jwm.v11_selector import canonical_sha256, verify_selector_freeze_manifest

        payload = {
            "configuration_digest": "e" * 64,
            "selected_config": {"hidden_dim": 64, "temperature": 0.25},
            "checkpoint_records": [{"file": "selector.pt", "sha256": "a" * 64}],
        }
        manifest = {
            "selector_freeze_payload": payload,
            "selector_freeze_digest": canonical_sha256(payload),
        }
        verify_selector_freeze_manifest(manifest)

        manifest["selector_freeze_payload"]["checkpoint_records"][0]["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "freeze digest"):
            verify_selector_freeze_manifest(manifest)

    def test_matched_test_evaluator_requires_frozen_matching_configuration(self):
        from evaluate_v11_frozen_selector import validate_frozen_evaluation_inputs
        from pi_jwm.v11_selector import canonical_sha256

        payload = {
            "configuration_digest": "e" * 64,
            "checkpoint_records": [{"file": "selector_a.pt", "sha256": "a" * 64}],
        }
        frozen = {
            "configuration_frozen": True,
            "configuration_digest": "e" * 64,
            "selected_checkpoints": ["selector_a.pt"],
            "selected_checkpoint_records": payload["checkpoint_records"],
            "selector_freeze_payload": payload,
            "selector_freeze_digest": canonical_sha256(payload),
        }
        cache = {"split_name": "matched_test", "configuration_digest": "e" * 64}
        validate_frozen_evaluation_inputs(frozen, cache)
        cache["configuration_digest"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_frozen_evaluation_inputs(frozen, cache)

    def test_checkpoint_record_requires_matching_file_sha256(self):
        from evaluate_v11_frozen_selector import verify_checkpoint_record

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector.pt"
            path.write_bytes(b"frozen selector")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_checkpoint_record(path, {"file": path.name, "sha256": "0" * 64})

    def test_acceptance_report_requires_metric_and_robustness_gates_for_a(self):
        from evaluate_v11_frozen_selector import compute_acceptance_report

        report = compute_acceptance_report(
            {
                "active_rate_rmse": 198.0,
                "link_rmse": 51.0,
                "activity_f1": 0.931,
                "default_link_rmse": 50.0,
                "default_activity_f1": 0.932,
            },
            validation_seed_std=4.0,
            external_seed_wins=8,
            pareto_violations=0,
        )
        self.assertEqual(report["rmse_tier"], "A")
        self.assertEqual(report["final_tier"], "A")

        report = compute_acceptance_report(
            {
                "active_rate_rmse": 198.0,
                "link_rmse": 52.0,
                "activity_f1": 0.931,
                "default_link_rmse": 50.0,
                "default_activity_f1": 0.932,
            },
            validation_seed_std=4.0,
            external_seed_wins=None,
            pareto_violations=0,
        )
        self.assertEqual(report["rmse_tier"], "A")
        self.assertEqual(report["final_tier"], "pending_external_or_safety_gate")

    def test_gpu_batch_freezes_validation_before_matched_test(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_v11_selector_finalization_gpu.sh"
        content = script.read_text(encoding="utf-8")

        self.assertLess(content.index("train_v11_candidate_set_selector.py"), content.index("--splits matched_test"))
        self.assertIn("configuration_frozen", content)
        self.assertIn("evaluate_v11_frozen_selector.py", content)

    def test_gpu_batch_checks_validation_candidate_gate_before_full_train_labels(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_v11_selector_finalization_gpu.sh"
        content = script.read_text(encoding="utf-8")

        self.assertLess(content.index("--splits validation"), content.index("--splits train calibration"))
        self.assertLess(content.index('p["candidate_gate"]["passed"]'), content.index("--splits train calibration"))
        self.assertIn('OMP_NUM_THREADS="${PI_JWM_OMP_NUM_THREADS:-8}"', content)
        self.assertIn('MKL_NUM_THREADS="${PI_JWM_MKL_NUM_THREADS:-8}"', content)

    def test_refinement_gpu_batch_never_reopens_locked_evaluation_splits(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_v11_selector_refinement_gpu.sh"
        )
        content = script.read_text(encoding="utf-8")

        self.assertIn("--splits validation", content)
        self.assertIn("--splits train calibration", content)
        self.assertIn('EPOCHS="${EPOCHS:-200}"', content)
        self.assertNotIn("matched_test", content)
        self.assertNotIn("external_holdout", content)


class SelectorFinalReportContractTest(unittest.TestCase):
    def test_physical_safety_alignment_uses_exact_seed_time_and_candidate(self):
        from audit_v11_selector_physical_safety import build_physical_safety_rows

        rows = build_physical_safety_rows(
            decision_rows=[
                {"sample_id": "7", "seed": "1", "candidate_name": "repair", "deferred": "False"}
            ],
            sample_index_rows=[
                {"sample_id": "7", "seed": "1", "input_end_time": "4.0", "label_time": "4.5"}
            ],
            physical_rows=[
                {"seed": "1", "decision_time": "4.0", "candidate_id": "default", "task_utility": "3", "energy_total": "5"},
                {"seed": "1", "decision_time": "4.0", "candidate_id": "repair_cf", "task_utility": "2", "energy_total": "7"},
            ],
            candidate_mapping={"repair": "repair_cf"},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task_utility_delta_actual"], -1.0)
        self.assertEqual(rows[0]["energy_delta_actual"], 2.0)

    def test_sha256_manifest_is_relative_sorted_and_excludes_itself(self):
        from finalize_v11_selector_report import build_sha256_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.txt").write_text("b", encoding="utf-8")
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "sha256_manifest.json").write_text("old", encoding="utf-8")
            rows = build_sha256_manifest(root)

        self.assertEqual([row["path"] for row in rows], ["a.txt", "b.txt"])
        self.assertTrue(all(len(row["sha256"]) == 64 for row in rows))


if __name__ == "__main__":
    unittest.main()
