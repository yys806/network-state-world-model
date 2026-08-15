import sys
import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


ACTION_NAMES = (
    "offload_count",
    "rb_task_count",
    "rb_total",
    "cpu_task_count",
    "cpu_total",
    "return_count",
)
LINK_NAMES = (
    "distance",
    "rate_sum",
    "csi_mean",
    "active_task_count",
    "allocated_rb_count",
)


def synthetic_interaction_inputs():
    sample_count, candidate_count, step_count, edge_count = 2, 3, 3, 4
    actions = np.zeros((sample_count, candidate_count, step_count, edge_count, 6), dtype=np.float32)
    actions[:, 1, 0, 1, 2] = 10.0
    actions[:, 1, 2, 3, 4] = 4.0
    actions[:, 0] = actions[:, 1]
    actions[:, 2] = actions[:, 1]
    actions[0, 2, 0, 1, 2] = 15.0
    actions[0, 2, 2, 3, 4] = 7.0
    actions[1, 2, 1, 2, 0] = 1.0

    predictions = []
    for candidate in range(candidate_count):
        activity = np.full((sample_count, step_count, edge_count), 0.2, dtype=np.float32)
        rate = np.full((sample_count, step_count, edge_count), 20.0, dtype=np.float32)
        if candidate == 2:
            activity[0, 0, 1] = 0.5
            rate[0, 0, 1] = 25.0
            activity[0, 2, 3] = 0.4
            rate[0, 2, 3] = 24.0
            activity[1, 1, 2] = 0.3
            rate[1, 1, 2] = 23.0
        predictions.append(
            {
                "link_activity_prob": activity,
                "link_rate_pred": rate,
                "task_pred": np.zeros((sample_count, step_count, 9), dtype=np.float32),
            }
        )
    current = np.zeros((sample_count, edge_count, 5), dtype=np.float32)
    for edge in range(edge_count):
        current[:, edge] = np.asarray([100 + edge, 10 + edge, 1 + edge, 2 + edge, 3 + edge])
    return actions, predictions, current


class CandidateInteractionTokenTest(unittest.TestCase):
    def test_builder_preserves_edge_step_action_and_prediction_pairing(self):
        from pi_jwm.v11_interactions import build_candidate_interaction_tokens

        actions, predictions, current = synthetic_interaction_inputs()
        batch = build_candidate_interaction_tokens(
            actions,
            predictions,
            current_link_features=current,
            action_feature_names=ACTION_NAMES,
            current_link_feature_names=LINK_NAMES,
            default_index=1,
            token_capacity=72,
        )

        self.assertEqual(batch.tokens.shape, (2, 3, 72, 25))
        self.assertEqual(batch.token_count[0, 2], 2)
        self.assertEqual(batch.edge_index[0, 2, :2].tolist(), [1, 3])
        names = list(batch.token_feature_names)
        first = batch.tokens[0, 2, 0]
        second = batch.tokens[0, 2, 1]
        self.assertEqual(first[names.index("step_0")], 1.0)
        self.assertEqual(second[names.index("step_2")], 1.0)
        self.assertEqual(first[names.index("current_distance")], 101.0)
        self.assertEqual(first[names.index("default_action_rb_total")], 10.0)
        self.assertEqual(first[names.index("action_delta_rb_total")], 5.0)
        self.assertEqual(first[names.index("default_predicted_rate")], 20.0)
        self.assertEqual(first[names.index("predicted_rate_delta")], 5.0)

    def test_padding_is_zero_and_uses_negative_edge_index(self):
        from pi_jwm.v11_interactions import build_candidate_interaction_tokens

        actions, predictions, current = synthetic_interaction_inputs()
        batch = build_candidate_interaction_tokens(
            actions,
            predictions,
            current,
            ACTION_NAMES,
            LINK_NAMES,
            default_index=1,
            token_capacity=72,
        )

        padding = ~batch.token_mask
        self.assertTrue(np.all(batch.tokens[padding] == 0.0))
        self.assertTrue(np.all(batch.edge_index[padding] == -1))
        self.assertTrue(np.all(batch.edge_index[batch.token_mask] >= 0))

    def test_action_and_prediction_values_reconstruct_candidate(self):
        from pi_jwm.v11_interactions import build_candidate_interaction_tokens

        actions, predictions, current = synthetic_interaction_inputs()
        batch = build_candidate_interaction_tokens(
            actions,
            predictions,
            current,
            ACTION_NAMES,
            LINK_NAMES,
            default_index=1,
            token_capacity=72,
        )
        names = list(batch.token_feature_names)
        token = batch.tokens[0, 2, 0]

        for action_name in ACTION_NAMES:
            reconstructed = token[names.index(f"default_action_{action_name}")] + token[
                names.index(f"action_delta_{action_name}")
            ]
            step = int(np.argmax(token[:3]))
            edge = int(batch.edge_index[0, 2, 0])
            self.assertEqual(
                reconstructed,
                actions[0, 2, step, edge, ACTION_NAMES.index(action_name)],
            )
        self.assertEqual(
            token[names.index("default_predicted_activity")]
            + token[names.index("predicted_activity_delta")],
            predictions[2]["link_activity_prob"][0, 0, 1],
        )

    def test_overflow_fails_instead_of_truncating(self):
        from pi_jwm.v11_interactions import build_candidate_interaction_tokens

        actions, predictions, current = synthetic_interaction_inputs()
        actions[0, 2, :, :, 2] = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0

        with self.assertRaisesRegex(ValueError, "capacity"):
            build_candidate_interaction_tokens(
                actions,
                predictions,
                current,
                ACTION_NAMES,
                LINK_NAMES,
                default_index=1,
                token_capacity=4,
            )

    def test_candidate_permutation_only_permutates_candidate_axis(self):
        from pi_jwm.v11_interactions import build_candidate_interaction_tokens

        actions, predictions, current = synthetic_interaction_inputs()
        original = build_candidate_interaction_tokens(
            actions,
            predictions,
            current,
            ACTION_NAMES,
            LINK_NAMES,
            default_index=1,
        )
        permutation = np.asarray([2, 1, 0])
        permuted = build_candidate_interaction_tokens(
            actions[:, permutation],
            [predictions[index] for index in permutation],
            current,
            ACTION_NAMES,
            LINK_NAMES,
            default_index=1,
        )

        np.testing.assert_array_equal(permuted.tokens, original.tokens[:, permutation])
        np.testing.assert_array_equal(permuted.token_mask, original.token_mask[:, permutation])
        np.testing.assert_array_equal(permuted.edge_index, original.edge_index[:, permutation])


class CandidateInteractionPoolingTest(unittest.TestCase):
    def test_pooling_has_fixed_234_dimension_and_recomputable_values(self):
        from pi_jwm.v11_interactions import (
            build_candidate_interaction_tokens,
            pool_candidate_interactions,
        )

        actions, predictions, current = synthetic_interaction_inputs()
        interactions = build_candidate_interaction_tokens(
            actions, predictions, current, ACTION_NAMES, LINK_NAMES, default_index=1
        )
        pooled = pool_candidate_interactions(interactions, ACTION_NAMES)

        self.assertEqual(pooled.pooled_features.shape, (2, 3, 234))
        self.assertEqual(len(pooled.pooled_feature_names), 234)
        names = list(pooled.pooled_feature_names)
        row = pooled.pooled_features[0, 2]
        self.assertEqual(row[names.index("step_0__rb_total__count")], 1.0)
        self.assertEqual(row[names.index("step_0__rb_total__delta_sum")], 5.0)
        self.assertEqual(row[names.index("step_0__rb_total__delta_abs_sum")], 5.0)
        self.assertEqual(row[names.index("step_0__rb_total__delta_abs_max")], 5.0)
        self.assertEqual(row[names.index("step_0__rb_total__weighted_current_distance")], 101.0)
        self.assertEqual(row[names.index("step_0__rb_total__weighted_predicted_rate_delta")], 5.0)

    def test_unmodified_step_action_block_is_all_zero(self):
        from pi_jwm.v11_interactions import (
            build_candidate_interaction_tokens,
            pool_candidate_interactions,
        )

        actions, predictions, current = synthetic_interaction_inputs()
        interactions = build_candidate_interaction_tokens(
            actions, predictions, current, ACTION_NAMES, LINK_NAMES, default_index=1
        )
        pooled = pool_candidate_interactions(interactions, ACTION_NAMES)
        names = np.asarray(pooled.pooled_feature_names)
        block = np.char.startswith(names, "step_1__rb_total__")

        self.assertEqual(int(block.sum()), 13)
        np.testing.assert_array_equal(pooled.pooled_features[0, 2, block], np.zeros(13))

    def test_append_returns_new_candidate_batch_with_prefixed_names(self):
        from pi_jwm.v11_interactions import (
            append_interaction_pooled_features,
            build_candidate_interaction_tokens,
            pool_candidate_interactions,
        )
        from pi_jwm.v11_selector import CandidateBatch

        actions, predictions, current = synthetic_interaction_inputs()
        interactions = pool_candidate_interactions(
            build_candidate_interaction_tokens(
                actions, predictions, current, ACTION_NAMES, LINK_NAMES, default_index=1
            ),
            ACTION_NAMES,
        )
        base = CandidateBatch(
            context=np.ones((2, 1), dtype=np.float32),
            candidate_features=np.ones((2, 3, 2), dtype=np.float32),
            candidate_mask=np.ones((2, 3), dtype=bool),
            stage=np.asarray(["offload", "compute"]),
            feature_names=("base_a", "base_b"),
            candidate_names=("a", "default", "c"),
            context_feature_names=("context",),
        )
        combined = append_interaction_pooled_features(base, interactions)

        self.assertIsNot(combined, base)
        self.assertEqual(combined.candidate_features.shape, (2, 3, 236))
        self.assertEqual(base.candidate_features.shape, (2, 3, 2))
        self.assertTrue(all(name.startswith("interaction_") for name in combined.feature_names[2:]))


class CandidateInteractionCacheTest(unittest.TestCase):
    def _payload(self):
        from pi_jwm.v11_interactions import (
            build_candidate_interaction_tokens,
            pool_candidate_interactions,
        )
        from pi_jwm.v11_selector import CandidateBatch, CandidateOutcome

        actions, predictions, current = synthetic_interaction_inputs()
        interactions = pool_candidate_interactions(
            build_candidate_interaction_tokens(
                actions, predictions, current, ACTION_NAMES, LINK_NAMES, default_index=1
            ),
            ACTION_NAMES,
        )
        batch = CandidateBatch(
            context=np.ones((2, 2), dtype=np.float32),
            candidate_features=np.ones((2, 3, 4), dtype=np.float32),
            candidate_mask=np.ones((2, 3), dtype=bool),
            stage=np.asarray(["offload", "compute"]),
            feature_names=("a", "b", "c", "d"),
            candidate_names=("candidate_0", "ranked_default", "candidate_2"),
            context_feature_names=("context_a", "context_b"),
        )
        outcome = CandidateOutcome(
            active_sse=np.arange(6, dtype=np.float32).reshape(2, 3),
            active_count=np.asarray([12, 12], dtype=np.int64),
            default_index=1,
        )
        return batch, outcome, interactions

    def test_schema6_cache_round_trip_keeps_interaction_contract(self):
        from pi_jwm.v11_labeling import (
            load_candidate_interaction_cache,
            load_candidate_label_cache,
            save_candidate_interaction_cache,
        )

        batch, outcome, interactions = self._payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema6.npz"
            manifest = save_candidate_interaction_cache(
                path,
                split_name="validation",
                sample_ids=np.asarray([10, 11]),
                sample_seed=np.asarray([50, 51]),
                batch=batch,
                outcome=outcome,
                interactions=interactions,
                action_feature_names=ACTION_NAMES,
                configuration_digest="6" * 64,
            )
            loaded_batch, loaded_outcome, loaded_interactions, loaded_manifest = (
                load_candidate_interaction_cache(path, expected_configuration_digest="6" * 64)
            )
            legacy_batch, legacy_outcome, legacy_manifest = load_candidate_label_cache(path)

        self.assertEqual(manifest["schema_version"], 6)
        self.assertEqual(manifest["interaction"]["token_capacity"], 72)
        self.assertEqual(manifest["interaction"]["token_dimension"], 25)
        self.assertEqual(manifest["interaction"]["pooled_dimension"], 234)
        self.assertEqual(manifest["interaction"]["overflow_count"], 0)
        self.assertEqual(manifest["interaction"]["action_feature_names"], list(ACTION_NAMES))
        self.assertIn("token_count_distribution", manifest["interaction"])
        np.testing.assert_array_equal(loaded_interactions.tokens, interactions.tokens)
        np.testing.assert_array_equal(loaded_interactions.token_mask, interactions.token_mask)
        np.testing.assert_array_equal(loaded_interactions.edge_index, interactions.edge_index)
        np.testing.assert_array_equal(
            loaded_interactions.pooled_features, interactions.pooled_features
        )
        np.testing.assert_array_equal(loaded_batch.candidate_features, batch.candidate_features)
        np.testing.assert_array_equal(legacy_batch.candidate_features, batch.candidate_features)
        np.testing.assert_array_equal(loaded_outcome.active_sse, legacy_outcome.active_sse)
        self.assertEqual(loaded_manifest["cache_sha256"], legacy_manifest["cache_sha256"])

    def test_schema6_loader_rejects_missing_interaction_array(self):
        from pi_jwm.v11_labeling import (
            load_candidate_interaction_cache,
            save_candidate_interaction_cache,
        )

        batch, outcome, interactions = self._payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema6.npz"
            save_candidate_interaction_cache(
                path,
                "validation",
                np.asarray([10, 11]),
                np.asarray([50, 51]),
                batch,
                outcome,
                interactions,
                ACTION_NAMES,
                "6" * 64,
            )
            with np.load(path, allow_pickle=False) as stored:
                payload = {name: stored[name] for name in stored.files if name != "interaction_token_mask"}
            np.savez_compressed(path, **payload)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path = path.with_suffix(path.suffix + ".manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cache_sha256"] = digest
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing interaction"):
                load_candidate_interaction_cache(path)


class CandidateInteractionRunnerTest(unittest.TestCase):
    def test_label_runner_defaults_to_schema5_and_accepts_schema6(self):
        from run_v11_selector_candidate_labels import parse_args

        with mock.patch.object(sys, "argv", ["runner"]):
            default_args = parse_args()
        with mock.patch.object(sys, "argv", ["runner", "--cache-schema-version", "6"]):
            schema6_args = parse_args()

        self.assertEqual(default_args.cache_schema_version, 5)
        self.assertEqual(schema6_args.cache_schema_version, 6)

    def test_gpu_handoff_script_only_generates_unlocked_schema6_labels(self):
        script_path = CODE_ROOT / "scripts" / "run_v11_schema6_labels_gpu.sh"

        text = script_path.read_text(encoding="utf-8")

        self.assertIn("--cache-schema-version 6", text)
        self.assertIn("--splits validation", text)
        self.assertIn("--splits train calibration", text)
        self.assertLess(
            text.index("--splits validation"),
            text.index("--splits train calibration"),
        )
        self.assertIn('summary["candidate_gate"]["passed"]', text)
        self.assertIn("torch.cuda.is_available", text)
        self.assertIn("PYTHONPATH", text)
        self.assertIn("label_cache_schema6", text)
        self.assertNotIn("matched_test", text)
        self.assertNotIn("external_holdout", text)
        self.assertNotIn("train_v11_candidate_set_selector", text)
        self.assertNotIn("evaluate_v11_frozen_selector", text)


if __name__ == "__main__":
    unittest.main()
