import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pi_jwm.step3_2_batch_preprocessing_v1 import RawSource, build_batch, fit_train_normalization_stats, apply_normalization
from pi_jwm.step3_3_model_input_tensor_v1 import (
    ENTITY_TYPE_VOCAB,
    LIFECYCLE_VOCAB,
    ROUTE_KIND_VOCAB,
    TRANSPORT_VOCAB,
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

    def test_past_outcome_propagates_without_changing_observation(self):
        samples, stats = self._bundle()
        baseline = build_tensor_batch([samples[0]], stats=stats)
        changed = copy.deepcopy(samples[0])
        row = next(r for r in changed["history"][0]["outcome"]["tasks"] if r["presence"])
        row["transmitted_size"] = {"value": 123.5, "presence": True, "feature_mask": True}
        candidate = build_tensor_batch([changed], stats=stats)
        slot = row["task_index"]
        self.assertNotEqual(baseline["past_outcome_task_features"][0, 0, slot, 0], candidate["past_outcome_task_features"][0, 0, slot, 0])
        self.assertTrue(np.array_equal(baseline["task_features"], candidate["task_features"]))
        self.assertTrue(np.array_equal(baseline["entity_features"], candidate["entity_features"]))

    def test_target_propagation_isolated_from_input(self):
        samples, stats = self._bundle()
        source = copy.deepcopy(samples[0])
        task_id, task_slot = next(iter(source["static"]["target_index"]["task"].items()))
        nodes = list(source["static"]["target_index"]["physical"].items())
        flow_id = f"flow::{task_id}::wireless::{nodes[0][0]}->{nodes[1][0]}"
        source["static"]["target_index"]["flow"] = {flow_id: 0}
        source["target"][0]["flows"] = [{"flow_id": flow_id, "flow_index": 0, "task_id": task_id, "task_index": task_slot, "source_node_id": nodes[0][0], "source_node_index": nodes[0][1], "target_node_id": nodes[1][0], "target_node_index": nodes[1][1], "transport": "wireless", "service_volume": 5.0}]
        baseline = build_tensor_batch([source], stats=stats)
        changed = copy.deepcopy(source)
        entity = changed["target"][0]["entities"][0]
        entity["speed_mps"] = {"value": 321.25, "presence": True, "feature_mask": True}
        task = changed["target"][0]["tasks"][0]
        task["transmitted_size"] = {"value": 654.5, "presence": True, "feature_mask": True}
        service = changed["target"][0]["communication_service"]
        service["delivered_data_by_task"] = {task["task_id"]: 77.0}
        flow_frame = 0
        flow = changed["target"][0]["flows"][0]
        flow["service_volume"] = 88.0
        candidate = build_tensor_batch([changed], stats=stats)
        self.assertNotEqual(baseline["target_entity_features"][0, 0, entity["target_index"], 0], candidate["target_entity_features"][0, 0, entity["target_index"], 0])
        self.assertNotEqual(baseline["target_task_features"][0, 0, task["target_index"], 0], candidate["target_task_features"][0, 0, task["target_index"], 0])
        self.assertEqual(candidate["target_total_service"][0, 0, task["target_index"]], 77.0)
        self.assertEqual(candidate["target_flow_service"][0, flow_frame, flow["flow_index"]], 88.0)
        self.assertTrue(candidate["target_flow_feature_mask"][0, flow_frame, flow["flow_index"]])
        self.assertTrue(np.array_equal(baseline["entity_features"], candidate["entity_features"]))
        self.assertTrue(np.array_equal(baseline["task_features"], candidate["task_features"]))

    def test_comp_allocated_cpu_per_s_and_route_vocab(self):
        samples, stats = self._bundle()
        sample = copy.deepcopy(samples[0])
        task_id, task_index = next(iter(sample["static"]["input_entity_index"]["task"].items()))
        node_id, node_index = next(iter(sample["static"]["input_entity_index"]["physical"].items()))
        entry = {"task_id": task_id, "task_index": task_index, "node_id": node_id, "node_index": node_index, "allocated_cpu_per_s": 9.25}
        sample["future_action"][0]["comp"] = {"field_present": True, "empty": False, "missing": False, "entries": [entry]}
        past_route = sample["history"][0]["action"]["route"]["entries"][0]
        sample["future_action"][0]["route"] = {"field_present": True, "empty": False, "missing": False, "entries": [copy.deepcopy(past_route)]}
        tensor = build_tensor_batch([sample], stats=stats)
        self.assertEqual(tensor["future_task_index"][0, 0, 2, 0], task_index)
        self.assertEqual(tensor["future_comp_node_index"][0, 0, 0], node_index)
        self.assertEqual(tensor["future_comp_allocated_cpu"][0, 0, 0], 9.25)
        self.assertTrue(tensor["future_comp_allocated_cpu_mask"][0, 0, 0])
        code = ROUTE_KIND_VOCAB.index("offload")
        self.assertEqual(tensor["past_route_kind_index"][0, 0, 0], code)
        self.assertEqual(tensor["future_route_kind_index"][0, 0, 0], code)
        self.assertNotEqual(code, ROUTE_KIND_VOCAB.index("unknown"))

    def test_vocab_is_canonical_across_subset_reorder_and_entity_types_are_known(self):
        samples, stats = self._bundle()
        first = build_tensor_batch(samples, stats=stats)
        second = build_tensor_batch(list(reversed(samples[:2])), stats=stats)
        self.assertEqual(first["contract"].lifecycle_vocab, LIFECYCLE_VOCAB)
        self.assertEqual(first["contract"].entity_type_vocab, ENTITY_TYPE_VOCAB)
        self.assertEqual(first["contract"].route_kind_vocab, ROUTE_KIND_VOCAB)
        self.assertEqual(first["contract"].transport_vocab, TRANSPORT_VOCAB)
        self.assertEqual(first["contract"].to_dict()["lifecycle_vocab"], second["contract"].to_dict()["lifecycle_vocab"])
        present = first["entity_presence"]
        self.assertTrue(np.all(first["entity_type_index"][present] > ENTITY_TYPE_VOCAB.index("unknown")))
        self.assertTrue(np.all(first["task_lifecycle_index"][first["task_presence"]] > LIFECYCLE_VOCAB.index("unknown")))
        self.assertTrue(np.all(first["target_task_lifecycle_index"][first["target_task_presence"]] > LIFECYCLE_VOCAB.index("unknown")))
        if np.any(first["target_flow_presence"]):
            self.assertTrue(np.all(first["target_flow_transport_index"][first["target_flow_presence"]] > TRANSPORT_VOCAB.index("unknown")))

    def test_semantic_references_and_round_trip(self):
        samples, stats = self._bundle()
        tensor = build_tensor_batch([samples[0]], stats=stats)
        self.assertTrue(validate_tensor_batch(tensor))
        self.assertEqual(tensor["past_outcome_entity_presence"].shape[1], tensor["contract"].history_steps - 1)
        self.assertTrue(np.all(tensor["relation_endpoints"][tensor["relation_mask"]] >= 0))
        self.assertTrue(np.all(tensor["dag_edges"][tensor["dag_mask"]] >= 0))
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "semantic.npz"
            save_tensor_batch(tensor, path)
            loaded = load_tensor_batch(path)
        for key in ("past_outcome_task_features", "target_entity_features", "target_flow_task_index", "past_route_kind_index"):
            self.assertTrue(np.array_equal(tensor[key], loaded[key]))
        self.assertEqual(tensor["sample_static"], loaded["sample_static"])
        self.assertTrue(validate_tensor_batch(loaded))
        invalid = copy.deepcopy(loaded)
        active = np.argwhere(invalid["relation_mask"])
        self.assertGreater(len(active), 0)
        bi, hi, ri = active[0]
        invalid["relation_endpoints"][bi, hi, ri, 0] = -1
        self.assertFalse(validate_tensor_batch(invalid))


if __name__ == "__main__":
    unittest.main()
