import sys
import unittest
import importlib.util
import tempfile
import warnings
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def synthetic_audit_payload():
    from pi_jwm.v11_selector import CandidateBatch, CandidateOutcome

    batch = CandidateBatch(
        context=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        candidate_features=np.asarray(
            [
                [[0.0, 0.1, 1.0], [1.0, 0.2, 2.0], [2.0, 0.3, 3.0]],
                [[0.0, 0.4, 4.0], [1.0, 0.5, 5.0], [2.0, 0.6, 6.0]],
                [[0.0, 0.7, 7.0], [1.0, 0.8, 8.0], [2.0, 0.9, 9.0]],
            ],
            dtype=np.float32,
        ),
        candidate_mask=np.asarray(
            [[True, True, True], [True, True, True], [True, True, True]], dtype=bool
        ),
        stage=np.asarray(["offload", "compute", "return"]),
        feature_names=(
            "action_family_rb",
            "predicted_rate_delta_mean",
            "selected_current_distance_mean",
        ),
        candidate_names=("identity", "ranked_allocation_baseline", "repair"),
        context_feature_names=("state_task_num_tasks_last", "state_link_rate_sum_last_mean"),
    )
    outcome = CandidateOutcome(
        active_sse=np.asarray(
            [[12.0, 10.0, 4.0], [5.0, 8.0, 12.0], [0.0, 0.0, 0.0]], dtype=np.float32
        ),
        active_count=np.asarray([2, 1, 0], dtype=np.int64),
        action_applicable=np.asarray(
            [[True, True, True], [True, True, True], [True, True, True]], dtype=bool
        ),
        action_applied=np.asarray(
            [[True, True, True], [True, True, False], [True, True, True]], dtype=bool
        ),
        default_index=1,
    )
    return batch, outcome


class BenefitAuditDatasetTest(unittest.TestCase):
    def test_targets_use_default_sse_and_exclude_unapplied_candidates(self):
        from pi_jwm.v11_benefit_identifiability import build_benefit_audit_dataset

        batch, outcome = synthetic_audit_payload()
        dataset = build_benefit_audit_dataset(
            batch,
            outcome,
            sample_ids=np.asarray([10, 11, 12]),
            sample_seed=np.asarray([0, 1, 2]),
        )

        np.testing.assert_allclose(dataset.candidate_benefit[0], [-2.0, 0.0, 6.0])
        self.assertFalse(dataset.legal_candidate[1, 2])
        self.assertTrue(np.isnan(dataset.candidate_benefit[1, 2]))
        np.testing.assert_allclose(dataset.opportunity[:2], [6.0, 3.0])
        self.assertEqual(dataset.candidate_positive[0].tolist(), [False, False, True])

    def test_zero_active_count_is_audited_but_not_flattened(self):
        from pi_jwm.v11_benefit_identifiability import build_benefit_audit_dataset

        batch, outcome = synthetic_audit_payload()
        dataset = build_benefit_audit_dataset(
            batch,
            outcome,
            sample_ids=np.asarray([10, 11, 12]),
            sample_seed=np.asarray([0, 1, 2]),
        )

        self.assertFalse(dataset.valid_sample[2])
        self.assertTrue(np.isnan(dataset.opportunity[2]))
        self.assertNotIn(2, dataset.flat_sample_index.tolist())

    def test_sample_and_candidate_indices_round_trip(self):
        from pi_jwm.v11_benefit_identifiability import build_benefit_audit_dataset

        batch, outcome = synthetic_audit_payload()
        dataset = build_benefit_audit_dataset(
            batch,
            outcome,
            sample_ids=np.asarray([10, 11, 12]),
            sample_seed=np.asarray([0, 1, 2]),
        )

        pairs = list(zip(dataset.flat_sample_index.tolist(), dataset.flat_candidate_index.tolist()))
        self.assertEqual(pairs, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)])


class BenefitFeatureGroupTest(unittest.TestCase):
    def _dataset(self):
        from pi_jwm.v11_benefit_identifiability import build_benefit_audit_dataset

        batch, outcome = synthetic_audit_payload()
        return build_benefit_audit_dataset(
            batch,
            outcome,
            sample_ids=np.asarray([10, 11, 12]),
            sample_seed=np.asarray([0, 1, 2]),
        )

    def test_feature_groups_are_fixed_and_leakage_safe(self):
        from pi_jwm.v11_benefit_identifiability import build_benefit_feature_groups

        groups = build_benefit_feature_groups(self._dataset())

        self.assertEqual(
            tuple(groups),
            (
                "prior_only",
                "context_only",
                "candidate_only",
                "forecast_delta",
                "selected_edge",
                "full_schema_v5",
            ),
        )
        self.assertIsNone(groups["context_only"].candidate_features)
        self.assertEqual(groups["full_schema_v5"].candidate_features.shape[0], 5)
        all_names = (
            groups["full_schema_v5"].opportunity_feature_names
            + groups["full_schema_v5"].candidate_feature_names
        )
        self.assertFalse(
            any(
                token in name.lower()
                for name in all_names
                for token in ("seed", "future", "oracle", "benefit", "regret", "sse")
            )
        )

    def test_selected_edge_group_contains_only_selected_observations_and_priors(self):
        from pi_jwm.v11_benefit_identifiability import build_benefit_feature_groups

        selected = build_benefit_feature_groups(self._dataset())["selected_edge"]

        self.assertTrue(
            any(name == "selected_current_distance_mean" for name in selected.candidate_feature_names)
        )
        self.assertTrue(
            all(
                name.startswith(("selected_", "stage_", "candidate_id_"))
                for name in selected.candidate_feature_names
            )
        )

    def test_schema6_adds_interaction_groups_without_changing_schema5_group(self):
        from pi_jwm.v11_benefit_identifiability import (
            build_benefit_audit_dataset,
            build_benefit_feature_groups,
        )
        from pi_jwm.v11_selector import CandidateBatch

        base_batch, outcome = synthetic_audit_payload()
        interaction = np.arange(18, dtype=np.float32).reshape(3, 3, 2)
        batch = CandidateBatch(
            context=base_batch.context,
            candidate_features=np.concatenate(
                [base_batch.candidate_features, interaction], axis=2
            ),
            candidate_mask=base_batch.candidate_mask,
            stage=base_batch.stage,
            feature_names=base_batch.feature_names
            + (
                "interaction_step_0__rb_total__count",
                "interaction_step_0__rb_total__delta_sum",
            ),
            candidate_names=base_batch.candidate_names,
            context_feature_names=base_batch.context_feature_names,
        )
        dataset = build_benefit_audit_dataset(
            batch,
            outcome,
            sample_ids=np.asarray([10, 11, 12]),
            sample_seed=np.asarray([0, 1, 2]),
        )
        groups = build_benefit_feature_groups(dataset)

        self.assertEqual(tuple(groups)[-2:], ("interaction_pooled_only", "full_schema_v6"))
        self.assertFalse(
            any(
                name.startswith("interaction_")
                for name in groups["full_schema_v5"].candidate_feature_names
            )
        )
        self.assertTrue(
            any(
                name.startswith("interaction_")
                for name in groups["interaction_pooled_only"].candidate_feature_names
            )
        )
        self.assertGreater(
            len(groups["full_schema_v6"].candidate_feature_names),
            len(groups["full_schema_v5"].candidate_feature_names),
        )

    def test_validation_values_do_not_change_train_normalization(self):
        from pi_jwm.v11_benefit_identifiability import fit_train_normalizer

        train = np.asarray([[0.0, 2.0], [2.0, 2.0]], dtype=np.float32)
        validation = np.asarray([[1000.0, -1000.0]], dtype=np.float32)
        normalizer = fit_train_normalizer(train)

        np.testing.assert_allclose(normalizer.mean, [1.0, 2.0])
        np.testing.assert_allclose(normalizer.scale, [1.0, 1.0])
        np.testing.assert_allclose(normalizer.transform(validation), [[999.0, -1002.0]])


class GroupAwareAuditTest(unittest.TestCase):
    def test_seed_group_folds_never_split_a_seed(self):
        from pi_jwm.v11_benefit_identifiability import seed_group_folds

        seeds = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
        folds = seed_group_folds(seeds, n_splits=2)

        self.assertEqual(len(folds), 2)
        for train, validation in folds:
            self.assertTrue(set(seeds[train]).isdisjoint(set(seeds[validation])))
            self.assertEqual(sorted(np.concatenate([train, validation]).tolist()), list(range(8)))

    def test_safe_threshold_selects_only_positive_benefit_candidates(self):
        from pi_jwm.v11_benefit_identifiability import (
            BenefitPredictions,
            build_benefit_audit_dataset,
            calibrate_safe_thresholds,
        )

        batch, outcome = synthetic_audit_payload()
        dataset = build_benefit_audit_dataset(
            batch,
            outcome,
            sample_ids=np.asarray([10, 11, 12]),
            sample_seed=np.asarray([0, 1, 2]),
        )
        predictions = BenefitPredictions(
            opportunity_probability=np.asarray([0.9, 0.9, np.nan], dtype=np.float32),
            candidate_sign_probability=np.asarray(
                [[0.1, 0.1, 0.9], [0.9, 0.1, 0.99], [np.nan, np.nan, np.nan]],
                dtype=np.float32,
            ),
            predicted_benefit=np.asarray(
                [[-2.0, 0.0, 5.0], [2.0, 0.0, 100.0], [np.nan, np.nan, np.nan]],
                dtype=np.float32,
            ),
        )

        selected = calibrate_safe_thresholds(dataset, predictions, thresholds=(0.65,))

        self.assertEqual(selected.status, "safe_threshold")
        self.assertEqual(selected.choice.tolist(), [2, 0, 1])
        self.assertEqual(selected.metrics["executed_positive_precision"], 1.0)
        self.assertEqual(selected.metrics["negative_selection_rate"], 0.0)
        self.assertLess(
            selected.metrics["active_rate_rmse"], selected.metrics["default_active_rate_rmse"]
        )

    def test_no_safe_threshold_falls_back_to_default(self):
        from pi_jwm.v11_benefit_identifiability import (
            BenefitPredictions,
            build_benefit_audit_dataset,
            calibrate_safe_thresholds,
        )

        batch, outcome = synthetic_audit_payload()
        dataset = build_benefit_audit_dataset(
            batch,
            outcome,
            sample_ids=np.asarray([10, 11, 12]),
            sample_seed=np.asarray([0, 1, 2]),
        )
        predictions = BenefitPredictions(
            opportunity_probability=np.asarray([0.9, 0.9, np.nan], dtype=np.float32),
            candidate_sign_probability=np.asarray(
                [[0.9, 0.1, 0.1], [0.1, 0.1, 0.1], [np.nan, np.nan, np.nan]],
                dtype=np.float32,
            ),
            predicted_benefit=np.asarray(
                [[10.0, 0.0, -1.0], [-1.0, 0.0, -1.0], [np.nan, np.nan, np.nan]],
                dtype=np.float32,
            ),
        )

        selected = calibrate_safe_thresholds(dataset, predictions, thresholds=(0.65,))

        self.assertEqual(selected.status, "no_safe_threshold")
        self.assertEqual(selected.choice.tolist(), [1, 1, 1])


class BenefitAuditModelTest(unittest.TestCase):
    def _dataset_and_groups(self):
        from pi_jwm.v11_benefit_identifiability import (
            build_benefit_audit_dataset,
            build_benefit_feature_groups,
        )

        batch, outcome = synthetic_audit_payload()
        dataset = build_benefit_audit_dataset(
            batch,
            outcome,
            sample_ids=np.asarray([10, 11, 12]),
            sample_seed=np.asarray([0, 1, 2]),
        )
        return dataset, build_benefit_feature_groups(dataset)

    def test_linear_model_is_deterministic_and_returns_full_shapes(self):
        from pi_jwm.v11_benefit_identifiability import (
            fit_benefit_audit_model,
            predict_benefit_audit_model,
        )

        dataset, groups = self._dataset_and_groups()
        first = fit_benefit_audit_model(
            dataset, groups["full_schema_v5"], model_kind="linear", random_seed=20260718
        )
        second = fit_benefit_audit_model(
            dataset, groups["full_schema_v5"], model_kind="linear", random_seed=20260718
        )
        first_predictions = predict_benefit_audit_model(
            first, dataset, groups["full_schema_v5"]
        )
        second_predictions = predict_benefit_audit_model(
            second, dataset, groups["full_schema_v5"]
        )

        self.assertEqual(first_predictions.opportunity_probability.shape, (3,))
        self.assertEqual(first_predictions.predicted_benefit.shape, (3, 3))
        np.testing.assert_allclose(
            first_predictions.opportunity_probability,
            second_predictions.opportunity_probability,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            first_predictions.predicted_benefit,
            second_predictions.predicted_benefit,
            equal_nan=True,
        )
        self.assertEqual(first.opportunity_status, "constant_prior")

    def test_context_only_model_records_candidate_task_as_unavailable(self):
        from pi_jwm.v11_benefit_identifiability import (
            fit_benefit_audit_model,
            predict_benefit_audit_model,
        )

        dataset, groups = self._dataset_and_groups()
        fitted = fit_benefit_audit_model(
            dataset, groups["context_only"], model_kind="linear", random_seed=20260718
        )
        predictions = predict_benefit_audit_model(fitted, dataset, groups["context_only"])

        self.assertEqual(fitted.candidate_status, "unavailable")
        self.assertTrue(
            np.all(predictions.predicted_benefit[dataset.legal_candidate & dataset.valid_sample[:, None]] == 0)
        )

    def test_linear_probability_model_uses_fixed_epoch_averaged_sgd(self):
        from pi_jwm.v11_benefit_identifiability import _model_factories

        classifier, _ = _model_factories("linear", random_seed=20260718)

        self.assertEqual(classifier.__class__.__name__, "SGDClassifier")
        self.assertEqual(classifier.loss, "log_loss")
        self.assertTrue(classifier.average)
        self.assertIsNone(classifier.tol)

    def test_prediction_metrics_are_recomputed_from_trace_arrays(self):
        from pi_jwm.v11_benefit_identifiability import (
            BenefitPredictions,
            build_benefit_audit_dataset,
            evaluate_benefit_predictions,
        )

        batch, outcome = synthetic_audit_payload()
        dataset = build_benefit_audit_dataset(
            batch,
            outcome,
            sample_ids=np.asarray([10, 11, 12]),
            sample_seed=np.asarray([0, 1, 2]),
        )
        predictions = BenefitPredictions(
            opportunity_probability=np.asarray([0.9, 0.9, np.nan], dtype=np.float32),
            candidate_sign_probability=np.asarray(
                [[0.1, 0.1, 0.9], [0.9, 0.1, np.nan], [np.nan, np.nan, np.nan]],
                dtype=np.float32,
            ),
            predicted_benefit=np.asarray(
                [[-2.0, 0.0, 5.0], [2.0, 0.0, np.nan], [np.nan, np.nan, np.nan]],
                dtype=np.float32,
            ),
        )

        metrics = evaluate_benefit_predictions(
            dataset, predictions, choice=np.asarray([2, 0, 1])
        )

        self.assertEqual(metrics["candidate_sign_pr_auc"], 1.0)
        self.assertEqual(metrics["top1_positive_ratio"], 1.0)
        self.assertEqual(metrics["improved_seed_count"], 2)
        self.assertLess(metrics["active_rate_rmse"], metrics["default_active_rate_rmse"])

    def test_constant_float32_scores_do_not_emit_spearman_warning(self):
        from scipy.stats import ConstantInputWarning
        from pi_jwm.v11_benefit_identifiability import (
            BenefitPredictions,
            build_benefit_audit_dataset,
            evaluate_benefit_predictions,
        )
        from pi_jwm.v11_selector import CandidateBatch, CandidateOutcome

        batch = CandidateBatch(
            context=np.ones((1, 1), dtype=np.float32),
            candidate_features=np.ones((1, 5, 1), dtype=np.float32),
            candidate_mask=np.ones((1, 5), dtype=bool),
            stage=np.asarray(["offload"]),
            feature_names=("rb_total_sum",),
            candidate_names=("identity", "ranked", "a", "b", "c"),
            context_feature_names=("state_task_num_tasks_last",),
        )
        outcome = CandidateOutcome(
            active_sse=np.asarray([[12.0, 10.0, 4.0, 8.0, 14.0]], dtype=np.float32),
            active_count=np.ones(1, dtype=np.int64),
            action_applied=np.ones((1, 5), dtype=bool),
            action_applicable=np.ones((1, 5), dtype=bool),
            default_index=1,
        )
        dataset = build_benefit_audit_dataset(
            batch,
            outcome,
            sample_ids=np.asarray([10]),
            sample_seed=np.asarray([0]),
        )
        constant = np.float32(6608.62158203125)
        predicted_benefit = np.full((1, 5), np.nan, dtype=np.float32)
        predicted_benefit[dataset.legal_candidate & dataset.valid_sample[:, None]] = constant
        sign = np.full((1, 5), np.nan, dtype=np.float32)
        sign[dataset.legal_candidate & dataset.valid_sample[:, None]] = 0.5
        predictions = BenefitPredictions(
            opportunity_probability=np.asarray([0.5], dtype=np.float32),
            candidate_sign_probability=sign,
            predicted_benefit=predicted_benefit,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = evaluate_benefit_predictions(
                dataset, predictions, choice=np.asarray([1])
            )

        self.assertIsNone(metrics["benefit_spearman"])
        self.assertIsNone(metrics["sample_rank_spearman"])
        self.assertFalse(any(isinstance(item.message, ConstantInputWarning) for item in caught))


class BenefitAuditCliTest(unittest.TestCase):
    @staticmethod
    def _load_cli_module():
        script = CODE_ROOT / "scripts" / "audit_v11_candidate_benefit_identifiability.py"
        spec = importlib.util.spec_from_file_location("benefit_audit_cli", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_cli_has_no_locked_split_arguments(self):
        module = self._load_cli_module()

        destinations = {action.dest for action in module.build_parser()._actions}

        self.assertIn("train_cache", destinations)
        self.assertIn("calibration_cache", destinations)
        self.assertIn("validation_cache", destinations)
        self.assertIn("required_schema_version", destinations)
        self.assertNotIn("matched_test_cache", destinations)
        self.assertNotIn("external_holdout_cache", destinations)

    def test_manifest_protocol_requires_schema5_matching_configuration_and_seed_sets(self):
        module = self._load_cli_module()
        candidate_names = ["identity", "ranked"]
        feature_names = ["rb_total"]
        context_names = ["state_task"]
        seed_spec = {
            "train": list(range(16)) + list(range(20, 44)),
            "calibration": list(range(44, 50)),
            "validation": list(range(50, 60)),
        }
        manifests = {
            split: {
                "schema_version": 5,
                "split_name": split,
                "configuration_digest": "a" * 64,
                "candidate_names": candidate_names,
                "feature_names": feature_names,
                "context_feature_names": context_names,
                "seed_values": seeds,
            }
            for split, seeds in seed_spec.items()
        }

        self.assertEqual(module.validate_audit_manifests(manifests), "a" * 64)
        schema6_manifests = {
            split: {
                **manifest,
                "schema_version": 6,
                "interaction": {
                    "token_capacity": 72,
                    "token_dimension": 25,
                    "pooled_dimension": 234,
                    "token_feature_names": ["step_0"],
                    "pooled_feature_names": ["step_0__rb_total__count"],
                    "action_feature_names": ["offload_count"],
                },
            }
            for split, manifest in manifests.items()
        }
        self.assertEqual(
            module.validate_audit_manifests(
                schema6_manifests, required_schema_version=6
            ),
            "a" * 64,
        )
        with self.assertRaisesRegex(ValueError, "schema 6"):
            module.validate_audit_manifests(manifests, required_schema_version=6)
        manifests["validation"]["seed_values"] = [50]
        with self.assertRaisesRegex(ValueError, "seed"):
            module.validate_audit_manifests(manifests)

    def test_tiny_end_to_end_run_writes_auditable_outputs(self):
        from pi_jwm.v11_labeling import save_candidate_label_cache
        from pi_jwm.v11_selector import CandidateBatch, CandidateOutcome

        module = self._load_cli_module()
        seed_spec = {
            "train": list(range(16)) + list(range(20, 44)),
            "calibration": list(range(44, 50)),
            "validation": list(range(50, 60)),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for split, seeds in seed_spec.items():
                count = len(seeds)
                features = np.zeros((count, 3, 3), dtype=np.float32)
                features[:, :, 0] = np.arange(3, dtype=np.float32)
                features[:, :, 1] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
                features[:, :, 2] = np.asarray([1.0, 0.0, 2.0], dtype=np.float32)
                batch = CandidateBatch(
                    context=np.stack(
                        [np.arange(count, dtype=np.float32), np.ones(count, dtype=np.float32)],
                        axis=1,
                    ),
                    candidate_features=features,
                    candidate_mask=np.ones((count, 3), dtype=bool),
                    stage=np.asarray(["offload", "compute"] * ((count + 1) // 2))[:count],
                    feature_names=(
                        "rb_total_sum",
                        "predicted_rate_delta_mean",
                        "selected_current_distance_mean",
                    ),
                    candidate_names=("identity", "ranked_allocation_baseline", "repair"),
                    context_feature_names=("state_task_num_tasks_last", "state_link_rate_sum_last_mean"),
                )
                active_sse = np.stack(
                    [
                        np.full(count, 9.0),
                        np.full(count, 10.0),
                        np.where(np.arange(count) % 2 == 0, 4.0, 12.0),
                    ],
                    axis=1,
                ).astype(np.float32)
                outcome = CandidateOutcome(
                    active_sse=active_sse,
                    active_count=np.ones(count, dtype=np.int64),
                    link_sse=active_sse.copy(),
                    link_count=np.ones(count, dtype=np.int64),
                    activity_tp=np.ones((count, 3), dtype=np.int64),
                    activity_fp=np.zeros((count, 3), dtype=np.int64),
                    activity_fn=np.zeros((count, 3), dtype=np.int64),
                    activity_tn=np.ones((count, 3), dtype=np.int64),
                    action_applied=np.ones((count, 3), dtype=bool),
                    action_applicable=np.ones((count, 3), dtype=bool),
                    default_index=1,
                )
                path = root / f"candidate_labels_{split}.npz"
                save_candidate_label_cache(
                    path,
                    split_name=split,
                    sample_ids=np.arange(count, dtype=np.int64),
                    sample_seed=np.asarray(seeds, dtype=np.int64),
                    batch=batch,
                    outcome=outcome,
                    configuration_digest="d" * 64,
                )
                paths[split] = path
            output = root / "report"
            args = module.build_parser().parse_args(
                [
                    "--train-cache", str(paths["train"]),
                    "--calibration-cache", str(paths["calibration"]),
                    "--validation-cache", str(paths["validation"]),
                    "--output-dir", str(output),
                    "--model-kinds", "linear",
                    "--feature-groups", "prior_only",
                ]
            )

            summary = module.run(args)

            self.assertFalse(summary["matched_test_accessed"])
            self.assertFalse(summary["external_holdout_accessed"])
            self.assertEqual(summary["group_cv"]["fold_count"], 3)
            for name in (
                "feature_group_results.csv",
                "seed_results.csv",
                "prediction_trace_validation.csv",
                "train_group_cv.csv",
                "summary.json",
                "sha256_manifest.txt",
            ):
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
