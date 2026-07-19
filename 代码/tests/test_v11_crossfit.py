import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class SeedCrossfitProtocolTest(unittest.TestCase):
    def test_fixed_round_robin_folds_cover_train_without_overlap(self):
        from pi_jwm.v11_crossfit import (
            audit_seed_crossfit_folds,
            build_seed_crossfit_folds,
        )
        from pi_jwm.v11_selector import DEFAULT_SELECTOR_SEEDS

        folds = build_seed_crossfit_folds(DEFAULT_SELECTOR_SEEDS["train"])

        self.assertEqual(
            [fold.held_out_seeds for fold in folds],
            [
                (0, 5, 10, 15, 24, 29, 34, 39),
                (1, 6, 11, 20, 25, 30, 35, 40),
                (2, 7, 12, 21, 26, 31, 36, 41),
                (3, 8, 13, 22, 27, 32, 37, 42),
                (4, 9, 14, 23, 28, 33, 38, 43),
            ],
        )
        self.assertTrue(
            audit_seed_crossfit_folds(folds, DEFAULT_SELECTOR_SEEDS)["passed"]
        )

    def test_protocol_digest_is_canonical_and_has_no_execution_fold(self):
        from pi_jwm.v11_crossfit import build_crossfit_protocol_manifest

        left = build_crossfit_protocol_manifest({"rf_trees": 160, "schema": 6})
        right = build_crossfit_protocol_manifest({"schema": 6, "rf_trees": 160})

        self.assertEqual(
            left["crossfit_protocol_digest"], right["crossfit_protocol_digest"]
        )
        self.assertNotIn("execution_fold", left["crossfit_protocol_payload"])


class CrossfitExecutionTest(unittest.TestCase):
    @staticmethod
    def _sample_seed():
        all_seeds = (
            list(range(16))
            + list(range(20, 60))
            + list(range(60, 70))
        )
        return np.repeat(np.asarray(all_seeds, dtype=np.int64), 2)

    def test_train_fold_helper_never_reads_held_out_seed(self):
        from pi_jwm.v11_crossfit import resolve_crossfit_execution

        sample_seed = self._sample_seed()
        result = resolve_crossfit_execution(sample_seed, ("train",), fold_id=0)

        self.assertEqual(
            set(sample_seed[result.label_indices["train"]]),
            set(result.held_out_seeds),
        )
        self.assertFalse(
            set(sample_seed[result.helper_train_indices]) & set(result.held_out_seeds)
        )
        self.assertEqual(
            set(sample_seed[result.helper_train_indices]),
            set(result.helper_train_seeds),
        )

    def test_eval_uses_full_train_helper_and_rejects_fold_id(self):
        from pi_jwm.v11_crossfit import resolve_crossfit_execution
        from pi_jwm.v11_selector import DEFAULT_SELECTOR_SEEDS

        sample_seed = self._sample_seed()
        result = resolve_crossfit_execution(
            sample_seed, ("calibration", "validation"), fold_id=None
        )

        self.assertEqual(
            set(sample_seed[result.helper_train_indices]),
            set(DEFAULT_SELECTOR_SEEDS["train"]),
        )
        with self.assertRaisesRegex(ValueError, "fold"):
            resolve_crossfit_execution(sample_seed, ("validation",), fold_id=1)

    def test_locked_evaluation_split_uses_no_locked_seed_for_helper_training(self):
        from pi_jwm.v11_crossfit import resolve_crossfit_execution
        from pi_jwm.v11_selector import DEFAULT_SELECTOR_SEEDS

        sample_seed = self._sample_seed()
        result = resolve_crossfit_execution(
            sample_seed, ("external_holdout",), fold_id=None
        )

        self.assertEqual(
            set(sample_seed[result.helper_train_indices]),
            set(DEFAULT_SELECTOR_SEEDS["train"]),
        )
        self.assertEqual(
            set(sample_seed[result.label_indices["external_holdout"]]),
            set(DEFAULT_SELECTOR_SEEDS["external_holdout"]),
        )


class CrossfitCacheMergeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write_fold_cache(self, fold_id, sample_ids, sample_seed=None):
        from pi_jwm.v11_crossfit import build_seed_crossfit_folds
        from pi_jwm.v11_interactions import CandidateInteractionBatch
        from pi_jwm.v11_labeling import save_candidate_interaction_cache
        from pi_jwm.v11_selector import CandidateBatch, CandidateOutcome

        ids = np.asarray(sample_ids, dtype=np.int64)
        count = len(ids)
        fold = build_seed_crossfit_folds()[fold_id]
        seeds = np.asarray(
            sample_seed
            if sample_seed is not None
            else [fold.held_out_seeds[0]] * count,
            dtype=np.int64,
        )
        batch = CandidateBatch(
            context=np.stack([ids, ids + 1], axis=1).astype(np.float32),
            candidate_features=np.zeros((count, 2, 3), dtype=np.float32),
            candidate_mask=np.ones((count, 2), dtype=bool),
            stage=np.asarray(["offload"] * count),
            feature_names=("x", "y", "z"),
            candidate_names=("identity", "ranked_allocation_baseline"),
            context_feature_names=("ctx0", "ctx1"),
        )
        outcome = CandidateOutcome(
            active_sse=np.stack([ids + 1, ids + 2], axis=1).astype(np.float32),
            active_count=np.ones(count, dtype=np.int64),
            default_index=1,
        )
        interactions = CandidateInteractionBatch(
            tokens=np.zeros((count, 2, 72, 25), dtype=np.float32),
            token_mask=np.zeros((count, 2, 72), dtype=bool),
            edge_index=np.full((count, 2, 72), -1, dtype=np.int32),
            token_feature_names=tuple(f"token_{index}" for index in range(25)),
            pooled_features=np.zeros((count, 2, 234), dtype=np.float32),
            pooled_feature_names=tuple(f"pooled_{index}" for index in range(234)),
        )
        path = self.root / f"fold_{fold_id}_{ids[0]}.npz"
        save_candidate_interaction_cache(
            path,
            split_name="train",
            sample_ids=ids,
            sample_seed=seeds,
            batch=batch,
            outcome=outcome,
            interactions=interactions,
            action_feature_names=("a0", "a1", "a2", "a3", "a4", "a5"),
            configuration_digest="b" * 64,
            protocol_metadata={
                "crossfit_protocol_digest": "a" * 64,
                "helper_execution": {
                    "mode": "crossfit_train_fold",
                    "fold_id": fold.fold_id,
                    "held_out_seeds": list(fold.held_out_seeds),
                    "helper_train_seeds": list(fold.helper_train_seeds),
                },
            },
            sample_fold_id=np.full(count, fold_id, dtype=np.int16),
        )
        return path

    def test_schema6_cache_keeps_fold_provenance_outside_model_features(self):
        from pi_jwm.v11_labeling import (
            load_candidate_interaction_cache,
            load_candidate_label_metadata,
        )

        path = self._write_fold_cache(fold_id=2, sample_ids=[9, 10])
        metadata = load_candidate_label_metadata(path)
        batch, _, _, _ = load_candidate_interaction_cache(path)

        np.testing.assert_array_equal(metadata["sample_fold_id"], [2, 2])
        self.assertFalse(any("fold" in name for name in batch.feature_names))
        self.assertFalse(any("fold" in name for name in batch.context_feature_names))

    def test_merge_rejects_duplicate_sample_id(self):
        from pi_jwm.v11_crossfit import merge_crossfit_label_caches

        paths = [
            self._write_fold_cache(0, [0, 1]),
            self._write_fold_cache(1, [1, 2]),
        ]

        with self.assertRaisesRegex(ValueError, "duplicate sample"):
            merge_crossfit_label_caches(
                paths,
                self.root / "merged.npz",
                expected_sample_ids=np.asarray([0, 1, 2]),
                expected_sample_seed=np.asarray([0, 0, 1]),
                expected_crossfit_protocol_digest="a" * 64,
            )

    def test_merge_sorts_samples_and_preserves_all_fold_ids(self):
        from pi_jwm.v11_crossfit import merge_crossfit_label_caches
        from pi_jwm.v11_labeling import (
            load_candidate_interaction_cache,
            load_candidate_label_metadata,
        )

        folds = __import__(
            "pi_jwm.v11_crossfit", fromlist=["build_seed_crossfit_folds"]
        ).build_seed_crossfit_folds()
        ids = [40, 10, 30, 0, 20]
        paths = [
            self._write_fold_cache(
                fold_id,
                [ids[fold_id]],
                [folds[fold_id].held_out_seeds[0]],
            )
            for fold_id in range(5)
        ]
        expected_ids = np.asarray(sorted(ids), dtype=np.int64)
        seed_by_id = {
            ids[fold_id]: folds[fold_id].held_out_seeds[0]
            for fold_id in range(5)
        }
        expected_seed = np.asarray(
            [seed_by_id[int(sample_id)] for sample_id in expected_ids],
            dtype=np.int64,
        )

        manifest = merge_crossfit_label_caches(
            paths,
            self.root / "merged.npz",
            expected_sample_ids=expected_ids,
            expected_sample_seed=expected_seed,
            expected_crossfit_protocol_digest="a" * 64,
        )
        metadata = load_candidate_label_metadata(self.root / "merged.npz")
        batch, outcome, interactions, _ = load_candidate_interaction_cache(
            self.root / "merged.npz"
        )

        np.testing.assert_array_equal(metadata["sample_ids"], expected_ids)
        self.assertEqual(set(metadata["sample_fold_id"].tolist()), set(range(5)))
        np.testing.assert_array_equal(batch.context[:, 0], expected_ids)
        np.testing.assert_array_equal(outcome.active_sse[:, 0], expected_ids + 1)
        self.assertEqual(interactions.tokens.shape[0], 5)
        self.assertEqual(manifest["schema_version"], 6)
        self.assertEqual(len(manifest["protocol_metadata"]["source_fold_caches"]), 5)


class CrossfitLabelRunnerContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.world_checkpoint = root / "world.pt"
        self.policy_checkpoint = root / "policy.pt"
        self.world_checkpoint.write_bytes(b"world")
        self.policy_checkpoint.write_bytes(b"policy")
        self.root = root

    def tearDown(self):
        self.temp.cleanup()

    def _args(self, fold_id):
        return Namespace(
            world_experiment_dir=self.root,
            world_checkpoint=self.world_checkpoint,
            policy_checkpoint=self.policy_checkpoint,
            output_dir=self.root / "output",
            splits=["train"],
            frozen_config_manifest=None,
            device="cpu",
            batch_size=8,
            helper_train_limit=8,
            split_sample_limit=8,
            policy_threshold=0.4,
            value_scale=1.0,
            new_policy_threshold=0.37,
            new_value_scale=1.06,
            gate_feature="step_rb_cpu_total",
            gate_threshold=450.0,
            value_codebook_size=9,
            min_effective_rb_total=1.0,
            activity_threshold=0.5,
            rf_trees=5,
            seed=20260717,
            stats_chunk_size=64,
            cache_schema_version=6,
            helper_protocol="seed_crossfit_5fold",
            crossfit_fold=fold_id,
        )

    def test_crossfit_fold_is_execution_metadata_not_global_configuration(self):
        from run_v11_selector_candidate_labels import _canonical_configuration

        left = _canonical_configuration(self._args(0))
        right = _canonical_configuration(self._args(4))

        self.assertEqual(left, right)
        self.assertIn("helper_generation_protocol", left)

    def test_crossfit_train_requires_fold_and_eval_rejects_fold(self):
        from run_v11_selector_candidate_labels import validate_helper_execution_args

        with self.assertRaisesRegex(ValueError, "fold"):
            validate_helper_execution_args(
                "seed_crossfit_5fold", None, ("train",)
            )
        with self.assertRaisesRegex(ValueError, "fold"):
            validate_helper_execution_args(
                "seed_crossfit_5fold", 0, ("validation",)
            )

    def test_reproduction_command_contains_crossfit_protocol(self):
        import shlex

        from run_v11_selector_candidate_labels import build_reproduction_command

        tokens = shlex.split(build_reproduction_command(self._args(3)))

        self.assertEqual(
            tokens[tokens.index("--helper-protocol") + 1],
            "seed_crossfit_5fold",
        )
        self.assertEqual(tokens[tokens.index("--crossfit-fold") + 1], "3")


if __name__ == "__main__":
    unittest.main()
