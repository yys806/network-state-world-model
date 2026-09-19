import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pi_jwm.step3_2_batch_preprocessing_v1 import RawSource, build_batch, fit_train_normalization_stats, apply_normalization
from pi_jwm.step3_3_model_input_tensor_v1 import (
    TensorContract,
    build_tensor_batch,
    load_tensor_batch,
    save_tensor_batch,
    validate_tensor_batch,
)


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/real_communication_outcome_semantics.json"


class Step33TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(RAW.read_text(encoding="utf-8"))

    def _raw(self, directory, trajectory_id, seed):
        payload = copy.deepcopy(self.raw)
        for decision in payload["decisions"]:
            decision["trajectory_id"] = trajectory_id
        payload["environment"]["seed"] = seed
        payload["environment"]["config_hash"] = f"config-{seed}"
        path = directory / f"{trajectory_id}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _bundle(self):
        with tempfile.TemporaryDirectory() as name:
            d = Path(name)
            a = self._raw(d, "train-a", 1)
            b = self._raw(d, "validation-a", 2)
            bundle = build_batch([RawSource(a, "dev_train"), RawSource(b, "dev_validation")])
            stats = fit_train_normalization_stats(bundle["samples"])
            return apply_normalization(bundle["samples"], stats), stats

    def test_fixed_shapes_identity_masks_and_action_indices(self):
        samples, stats = self._bundle()
        tensor = build_tensor_batch(samples, stats=stats)
        c = tensor["contract"]
        self.assertEqual(tensor["entity_features"].shape[:2], (len(samples), c.history_steps))
        self.assertEqual(tensor["past_action_present"].shape[:2], (len(samples), c.history_steps - 1))
        self.assertEqual(tensor["future_action_present"].shape[:2], (len(samples), c.horizon_steps))
        self.assertEqual(tensor["target_task_presence"].shape[:2], (len(samples), c.horizon_steps))
        self.assertTrue(np.all(tensor["entity_presence"][:, 0] >= 0))
        self.assertTrue(validate_tensor_batch(tensor))
        self.assertEqual(tensor["contract"].dag_direction, "source_task_j -> target_task_k; k depends on j")
        self.assertTrue(np.all(tensor["future_uav_index"] >= -1))
        for static in tensor["sample_static"]:
            for namespace, ids in static["target_only_objects"].items():
                self.assertTrue(set(ids).isdisjoint(static["input_entity_index"].get(namespace, {})))

    def test_missing_is_distinct_from_zero_and_padding_is_masked(self):
        samples, stats = self._bundle()
        sample = copy.deepcopy(samples[0])
        row = sample["history"][0]["entities"][0]
        row["speed_mps"] = {"value": 999999.0, "presence": True, "feature_mask": False}
        row["feature_mask"]["speed_mps"] = False
        sample["history"][0]["entities"] = sample["history"][0]["entities"][:-1]
        tensor = build_tensor_batch([sample], stats=stats)
        self.assertFalse(bool(tensor["entity_feature_mask"][0, 0, 0, 0]))
        self.assertEqual(float(tensor["entity_features"][0, 0, 0, 0]), 0.0)
        self.assertFalse(bool(tensor["entity_presence"][0, 0, -1]))
        self.assertFalse(bool(tensor["entity_feature_mask"][0, 0, -1, 0]))

    def test_capacity_overflow_and_round_trip(self):
        samples, stats = self._bundle()
        with self.assertRaises(ValueError):
            build_tensor_batch(samples, stats=stats, contract=TensorContract(max_entity=1))
        tensor = build_tensor_batch(samples, stats=stats)
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "tensor.npz"
            save_tensor_batch(tensor, path)
            loaded = load_tensor_batch(path)
        self.assertTrue(np.array_equal(tensor["entity_features"], loaded["entity_features"]))
        self.assertEqual(tensor["contract"].to_dict(), loaded["contract"].to_dict())


if __name__ == "__main__":
    unittest.main()
