import sys
import unittest
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "src"))


class _FeatureSumModel:
    def __init__(self, scale=1.0):
        self.scale = float(scale)

    def predict(self, features):
        return self.scale * np.asarray(features, dtype=np.float32).sum(axis=1)


class PhysicalAlignmentTest(unittest.TestCase):
    def test_aligns_only_exact_input_end_times(self):
        from pi_jwm.v11_physical_benefit import align_decision_points

        aligned, rejected = align_decision_points(
            [
                {"seed": 0, "decision_time": 0.8},
                {"seed": 0, "decision_time": 0.7},
            ],
            [{"sample_id": 10, "seed": 0, "input_end_time": 0.8}],
        )

        self.assertEqual(aligned, [{"seed": 0, "decision_time": 0.8, "sample_id": 10}])
        self.assertEqual(rejected[0]["reason"], "no_exact_input_end_time")

    def test_rejects_duplicate_sample_time_keys(self):
        from pi_jwm.v11_physical_benefit import align_decision_points

        with self.assertRaisesRegex(ValueError, "duplicate sample time"):
            align_decision_points(
                [{"seed": 0, "decision_time": 0.8}],
                [
                    {"sample_id": 10, "seed": 0, "input_end_time": 0.8},
                    {"sample_id": 11, "seed": 0, "input_end_time": 0.8},
                ],
            )

    def test_normalizes_only_supported_physical_families(self):
        from pi_jwm.v11_physical_benefit import normalize_physical_family

        self.assertEqual(normalize_physical_family("default"), "identity_control")
        self.assertEqual(normalize_physical_family("rb_count"), "rb_repair")
        self.assertEqual(normalize_physical_family("mixed_offload_rb"), "offload_rb")
        self.assertEqual(normalize_physical_family("cpu_scale"), "compute_cpu")
        self.assertEqual(normalize_physical_family("return_route"), "return_route")
        self.assertIsNone(normalize_physical_family("offload_target"))


class PhysicalDescriptorTest(unittest.TestCase):
    def test_physical_descriptor_has_fixed_leakage_safe_order(self):
        from pi_jwm.v11_physical_benefit import (
            COMMON_DESCRIPTOR_NAMES,
            physical_action_descriptor,
        )

        values, names = physical_action_descriptor(
            {
                "action_family": "mixed_offload_rb",
                "rb_total": 40.0,
                "num_rb_tasks": 2,
                "cpu_total": 5.0,
                "num_cpu_overrides": 1,
                "num_offload_overrides": 1,
                "num_return_route_overrides": 0,
                "intervention_start_step": 1,
                "temporal_pattern": "persistent",
            }
        )

        self.assertEqual(names, COMMON_DESCRIPTOR_NAMES)
        self.assertEqual(values.shape, (len(COMMON_DESCRIPTOR_NAMES),))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values[names.index("family_offload")], 1.0)
        self.assertEqual(values[names.index("pattern_persistent")], 1.0)
        self.assertFalse(
            any(
                token in name
                for name in names
                for token in ("actual", "future", "oracle", "outcome", "seed")
            )
        )

    def test_physical_descriptor_rejects_unsupported_family(self):
        from pi_jwm.v11_physical_benefit import physical_action_descriptor

        with self.assertRaisesRegex(ValueError, "unsupported physical action family"):
            physical_action_descriptor({"action_family": "offload_target"})

    def test_selector_descriptor_matches_common_contract(self):
        from pi_jwm.v11_physical_benefit import (
            COMMON_DESCRIPTOR_NAMES,
            selector_action_descriptors,
        )
        from pi_jwm.v11_selector import CandidateBatch

        names = (
            "rb_total_sum",
            "rb_action_count",
            "cpu_total_sum",
            "cpu_action_count",
            "offload_action_count",
            "return_action_count",
            "action_family_identity",
            "action_family_rb",
            "action_family_offload",
            "action_family_compute",
            "action_family_return",
            "action_family_historical",
        )
        features = np.zeros((1, 2, len(names)), dtype=np.float32)
        features[0, 0, names.index("action_family_identity")] = 1.0
        features[0, 1, names.index("rb_total_sum")] = 12.0
        features[0, 1, names.index("rb_action_count")] = 2.0
        features[0, 1, names.index("action_family_rb")] = 1.0
        batch = CandidateBatch(
            context=np.zeros((1, 1), dtype=np.float32),
            candidate_features=features,
            candidate_mask=np.ones((1, 2), dtype=bool),
            stage=np.asarray(["offload"]),
            feature_names=names,
            candidate_names=("identity", "rb_repair__k8__q50__decayed"),
            context_feature_names=("context",),
        )

        descriptors, descriptor_names = selector_action_descriptors(batch)

        self.assertEqual(descriptor_names, COMMON_DESCRIPTOR_NAMES)
        self.assertEqual(descriptors.shape, (1, 2, len(COMMON_DESCRIPTOR_NAMES)))
        self.assertEqual(descriptors[0, 0, descriptor_names.index("family_control")], 1.0)
        self.assertEqual(descriptors[0, 1, descriptor_names.index("family_rb")], 1.0)
        self.assertEqual(descriptors[0, 1, descriptor_names.index("pattern_decayed")], 1.0)
        self.assertEqual(descriptors[0, 1, descriptor_names.index("intervention_start_step")], 1.0)


class PhysicalTrainingBatchTest(unittest.TestCase):
    def _selector_batch(self):
        from pi_jwm.v11_selector import CandidateBatch

        return CandidateBatch(
            context=np.asarray([[2.0, 3.0]], dtype=np.float32),
            candidate_features=np.zeros((1, 1, 1), dtype=np.float32),
            candidate_mask=np.ones((1, 1), dtype=bool),
            stage=np.asarray(["offload"]),
            feature_names=("placeholder",),
            candidate_names=("identity",),
            context_feature_names=("state_task_count", "state_link_rate"),
        )

    def _rows(self):
        base = {
            "seed": 0,
            "sample_id": 10,
            "decision_time": 0.8,
            "intervention_start_step": 1,
            "temporal_pattern": "persistent",
            "action_applied": True,
            "rb_total": 5.0,
            "num_rb_tasks": 1,
            "cpu_total": 0.0,
            "num_cpu_overrides": 0,
            "num_offload_overrides": 0,
            "num_return_route_overrides": 0,
        }
        return [
            {
                **base,
                "candidate_id": "default",
                "action_family": "default",
                "task_utility": 1.0,
                "energy_total": 5.0,
            },
            {
                **base,
                "candidate_id": "rb_2__persistent",
                "action_family": "rb_count",
                "task_utility": 1.5,
                "energy_total": 7.0,
                "rb_total": 10.0,
            },
        ]

    def test_builds_candidate_minus_default_targets(self):
        from pi_jwm.v11_physical_benefit import build_physical_training_batch

        batch = build_physical_training_batch(
            physical_rows=self._rows(),
            selector_sample_ids=np.asarray([10]),
            selector_sample_seed=np.asarray([0]),
            selector_batch=self._selector_batch(),
        )

        np.testing.assert_allclose(batch.task_delta, [0.5])
        np.testing.assert_allclose(batch.energy_delta, [2.0])
        np.testing.assert_allclose(batch.features[0, :2], [2.0, 3.0])
        self.assertEqual(batch.sample_ids.tolist(), [10])
        self.assertEqual(batch.normalized_family.tolist(), ["rb_repair"])
        self.assertEqual(batch.rejected_rows, ())

    def test_rejects_unsupported_and_unapplied_candidates_without_imputation(self):
        from pi_jwm.v11_physical_benefit import build_physical_training_batch

        rows = self._rows()
        rows.append(
            {
                **rows[1],
                "candidate_id": "offload_alt",
                "action_family": "offload_target",
            }
        )
        rows.append(
            {
                **rows[1],
                "candidate_id": "rb_unapplied",
                "action_applied": False,
            }
        )

        batch = build_physical_training_batch(
            rows,
            np.asarray([10]),
            np.asarray([0]),
            self._selector_batch(),
        )

        self.assertEqual(batch.features.shape[0], 1)
        self.assertEqual(
            {row["reason"] for row in batch.rejected_rows},
            {"unsupported_action_family", "action_not_applied"},
        )
        self.assertEqual(
            {row["action_family"] for row in batch.rejected_rows},
            {"offload_target", "rb_count"},
        )

    def test_rejects_missing_or_duplicate_default(self):
        from pi_jwm.v11_physical_benefit import build_physical_training_batch

        with self.assertRaisesRegex(ValueError, "exactly one default"):
            build_physical_training_batch(
                self._rows()[1:],
                np.asarray([10]),
                np.asarray([0]),
                self._selector_batch(),
            )
        with self.assertRaisesRegex(ValueError, "exactly one default"):
            build_physical_training_batch(
                [self._rows()[0], self._rows()[0], self._rows()[1]],
                np.asarray([10]),
                np.asarray([0]),
                self._selector_batch(),
            )

    def test_protocol_audit_rejects_forbidden_features_and_split_overlap(self):
        from pi_jwm.v11_physical_benefit import audit_physical_bridge_protocol

        result = audit_physical_bridge_protocol(
            feature_names=("state_task_count", "actual_energy_outcome"),
            split_seed_sets={"train": (0, 1), "calibration": (1, 2)},
            matched_test_accessed=False,
            external_holdout_accessed=False,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["forbidden_features"], ["actual_energy_outcome"])
        self.assertEqual(result["split_overlap_count"], 1)


class PhysicalBenefitModelTest(unittest.TestCase):
    def _batch(self, seeds):
        from pi_jwm.v11_physical_benefit import PhysicalBenefitTrainingBatch

        seed_values = np.repeat(np.asarray(seeds, dtype=np.int64), 2)
        within_seed = np.tile(np.asarray([-1.0, 1.0], dtype=np.float32), len(seeds))
        seed_signal = (seed_values % 7).astype(np.float32) / 7.0
        features = np.stack((within_seed, seed_signal), axis=1)
        task = 2.0 * within_seed + 0.2 * seed_signal
        energy = -1.5 * within_seed + 0.1 * seed_signal
        return PhysicalBenefitTrainingBatch(
            features=features,
            feature_names=("action_magnitude", "state_load"),
            task_delta=task,
            energy_delta=energy,
            sample_ids=np.arange(seed_values.size, dtype=np.int64),
            sample_seed=seed_values,
            group_ids=np.asarray(
                [f"{seed}:{index}" for index, seed in enumerate(seed_values)], dtype=str
            ),
            normalized_family=np.asarray(["rb_repair"] * seed_values.size),
            stage=np.asarray(["offload"] * seed_values.size),
        )

    def test_crossfit_is_seed_held_out_and_deterministic(self):
        from pi_jwm.v11_physical_benefit import fit_physical_benefit_bridge
        from pi_jwm.v11_selector import DEFAULT_SELECTOR_SEEDS

        train = self._batch(DEFAULT_SELECTOR_SEEDS["train"])
        calibration = self._batch(DEFAULT_SELECTOR_SEEDS["calibration"])

        first, first_report = fit_physical_benefit_bridge(train, calibration)
        second, second_report = fit_physical_benefit_bridge(train, calibration)

        np.testing.assert_allclose(first.oof_task_mean, second.oof_task_mean)
        np.testing.assert_allclose(first.oof_energy_mean, second.oof_energy_mean)
        self.assertEqual(first_report, second_report)
        for fold in first.fold_records:
            self.assertFalse(set(fold["held_out_seeds"]) & set(fold["model_train_seeds"]))
            self.assertFalse(
                set(DEFAULT_SELECTOR_SEEDS["calibration"]) & set(fold["model_train_seeds"])
            )
        self.assertEqual(set(first.oof_fold_id.tolist()), set(range(5)))

    def test_conformal_prediction_has_calibrated_intervals(self):
        from pi_jwm.v11_physical_benefit import (
            fit_physical_benefit_bridge,
            predict_physical_benefit,
        )
        from pi_jwm.v11_selector import DEFAULT_SELECTOR_SEEDS

        train = self._batch(DEFAULT_SELECTOR_SEEDS["train"])
        calibration = self._batch(DEFAULT_SELECTOR_SEEDS["calibration"])
        fitted, report = fit_physical_benefit_bridge(train, calibration)
        prediction = predict_physical_benefit(fitted, calibration.features)

        self.assertGreaterEqual(report["calibration_task_coverage"], 0.8)
        self.assertGreaterEqual(report["calibration_energy_coverage"], 0.8)
        self.assertTrue(np.all(prediction.task_lcb <= prediction.task_mean))
        self.assertTrue(np.all(prediction.task_mean <= prediction.task_ucb))
        self.assertTrue(np.all(prediction.energy_lcb <= prediction.energy_mean))
        self.assertTrue(np.all(prediction.energy_mean <= prediction.energy_ucb))
        self.assertLess(report["oof_task_mae"], report["baseline_task_mae"])
        self.assertLess(report["oof_energy_mae"], report["baseline_energy_mae"])

    def test_exposes_task_only_gate_when_energy_is_not_identifiable(self):
        from dataclasses import replace

        from pi_jwm.v11_physical_benefit import fit_physical_benefit_bridge
        from pi_jwm.v11_selector import DEFAULT_SELECTOR_SEEDS

        train = self._batch(DEFAULT_SELECTOR_SEEDS["train"])
        calibration = self._batch(DEFAULT_SELECTOR_SEEDS["calibration"])
        train = replace(train, energy_delta=np.zeros_like(train.energy_delta))
        calibration = replace(
            calibration, energy_delta=np.zeros_like(calibration.energy_delta)
        )

        _, report = fit_physical_benefit_bridge(train, calibration)

        self.assertTrue(report["task_model_passed"])
        self.assertFalse(report["energy_model_passed"])
        self.assertTrue(report["task_only_passed"])
        self.assertFalse(report["passed"])


class PhysicalBenefitAugmentationTest(unittest.TestCase):
    def _batch(self):
        from pi_jwm.v11_selector import CandidateBatch

        names = (
            "rb_total_sum",
            "rb_action_count",
            "cpu_total_sum",
            "cpu_action_count",
            "offload_action_count",
            "return_action_count",
            "action_family_identity",
            "action_family_rb",
            "action_family_offload",
            "action_family_compute",
            "action_family_return",
            "action_family_historical",
        )
        features = np.zeros((1, 2, len(names)), dtype=np.float32)
        features[0, 0, names.index("action_family_identity")] = 1.0
        features[0, 1, names.index("rb_total_sum")] = 8.0
        features[0, 1, names.index("rb_action_count")] = 2.0
        features[0, 1, names.index("action_family_rb")] = 1.0
        return CandidateBatch(
            context=np.asarray([[3.0]], dtype=np.float32),
            candidate_features=features,
            candidate_mask=np.asarray([[True, True]]),
            stage=np.asarray(["offload"]),
            feature_names=names,
            candidate_names=("identity", "rb_repair__k8__q50__persistent"),
            context_feature_names=("state_load",),
        )

    def _fitted(self):
        from pi_jwm.v11_physical_benefit import (
            COMMON_DESCRIPTOR_NAMES,
            FittedPhysicalBenefitBridge,
        )

        return FittedPhysicalBenefitBridge(
            feature_names=("state_load",) + COMMON_DESCRIPTOR_NAMES,
            task_models=(_FeatureSumModel(1.0),),
            energy_models=(_FeatureSumModel(2.0),),
            task_conformal_radius=0.5,
            energy_conformal_radius=1.0,
            fold_records=(),
            oof_task_mean=np.zeros((0,), dtype=np.float32),
            oof_energy_mean=np.zeros((0,), dtype=np.float32),
            oof_fold_id=np.zeros((0,), dtype=np.int16),
        )

    def test_appends_eight_physical_fields_without_mutating_source_contract(self):
        from pi_jwm.v11_physical_benefit import (
            PHYSICAL_PREDICTION_FEATURES,
            augment_candidate_batch_with_physical_benefit,
        )

        source = self._batch()
        augmented = augment_candidate_batch_with_physical_benefit(
            source, self._fitted(), default_index=1
        )

        self.assertIsNot(augmented, source)
        self.assertEqual(
            augmented.feature_names,
            source.feature_names + PHYSICAL_PREDICTION_FEATURES,
        )
        self.assertEqual(
            augmented.candidate_features.shape[-1],
            source.candidate_features.shape[-1] + 8,
        )
        np.testing.assert_array_equal(augmented.candidate_mask, source.candidate_mask)
        np.testing.assert_array_equal(augmented.stage, source.stage)
        self.assertEqual(augmented.candidate_names, source.candidate_names)
        self.assertEqual(augmented.context_feature_names, source.context_feature_names)
        np.testing.assert_array_equal(
            augmented.candidate_features[:, :, : len(source.feature_names)],
            source.candidate_features,
        )
        task_mean = augmented.feature_names.index("physical_task_delta_mean")
        task_lcb = augmented.feature_names.index("physical_task_delta_lcb")
        energy_mean = augmented.feature_names.index("physical_energy_delta_mean")
        energy_ucb = augmented.feature_names.index("physical_energy_delta_ucb")
        np.testing.assert_allclose(
            augmented.candidate_features[:, 0, task_lcb],
            augmented.candidate_features[:, 0, task_mean] - 0.5,
        )
        np.testing.assert_allclose(
            augmented.candidate_features[:, 0, energy_ucb],
            augmented.candidate_features[:, 0, energy_mean] + 1.0,
        )
        np.testing.assert_allclose(
            augmented.candidate_features[:, 1, -len(PHYSICAL_PREDICTION_FEATURES) :],
            0.0,
        )

    def test_rejects_duplicate_physical_fields(self):
        from pi_jwm.v11_physical_benefit import augment_candidate_batch_with_physical_benefit

        augmented = augment_candidate_batch_with_physical_benefit(
            self._batch(), self._fitted(), default_index=1
        )
        with self.assertRaisesRegex(ValueError, "already contains physical benefit"):
            augment_candidate_batch_with_physical_benefit(
                augmented, self._fitted(), default_index=1
            )

    def test_task_only_augmentation_omits_unidentifiable_energy_fields(self):
        from pi_jwm.v11_physical_benefit import (
            PHYSICAL_TASK_PREDICTION_FEATURES,
            augment_candidate_batch_with_physical_benefit,
        )

        source = self._batch()
        augmented = augment_candidate_batch_with_physical_benefit(
            source, self._fitted(), default_index=1, include_energy=False
        )

        self.assertEqual(
            augmented.feature_names,
            source.feature_names + PHYSICAL_TASK_PREDICTION_FEATURES,
        )
        self.assertFalse(any("physical_energy" in name for name in augmented.feature_names))


if __name__ == "__main__":
    unittest.main()
