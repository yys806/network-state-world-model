import json
import tempfile
import unittest
from pathlib import Path

from pi_jwm.model_ready_sample_contract_v1 import build_sample, load_sample, validate_sample, write_sample


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/real_communication_outcome_semantics.json"


class ModelReadySampleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(RAW.read_text(encoding="utf-8"))

    def test_real_raw_sample_has_causal_windows_and_four_actions(self):
        sample = build_sample(self.raw, anchor_step=2)
        checks = validate_sample(sample)
        self.assertTrue(all(checks.values()))
        self.assertEqual([1, 2], sample["metadata"]["history_frame_indices"])
        self.assertEqual([2, 3], sample["metadata"]["future_action_frame_indices"])
        self.assertEqual(2, sample["metadata"]["anchor_decision_frame"])
        self.assertEqual(2, sample["history"][-1]["frame_index"])
        self.assertEqual(2, sample["metadata"]["future_action_frame_indices"][0])
        self.assertEqual({"physical", "task", "flow"}, set(sample["static"]["target_index"]))
        self.assertTrue(sample["static"]["relation_endpoints"])
        self.assertTrue(sample["static"]["dag_relations"]["observed_mask"])
        self.assertTrue(all(len(row["entities"]) == len(sample["static"]["input_entity_index"]["physical"]) for row in sample["history"]))
        first_entity = sample["history"][0]["entities"][0]
        self.assertTrue(first_entity["presence"])
        if not first_entity["feature_mask"]["canonical_acceleration_mps2"]:
            self.assertTrue(first_entity["canonical_acceleration_mps2"]["presence"])
            self.assertFalse(first_entity["canonical_acceleration_mps2"]["feature_mask"])
        self.assertIn("communication_service", sample["target"][0])
        self.assertIn("wireless_delivered_data_by_task", sample["target"][0]["communication_service"])
        action_entries = [entry for action in sample["future_action"] for family in action.values() for entry in family["entries"]]
        self.assertFalse(any(value == -1 for entry in action_entries for key, value in entry.items() if key.endswith("_index")))
        for action in sample["future_action"]:
            for entry in action["route"]["entries"]:
                self.assertEqual(len(entry.get("route_node_ids", [])), len(entry["route_node_indices"]))
            for entry in action["comp"]["entries"]:
                self.assertIn("node_index", entry)

    def test_target_only_object_is_not_input_index(self):
        sample = build_sample(self.raw, anchor_step=2)
        target_only = sample["static"]["target_only_objects"]
        self.assertTrue(target_only["task"])
        for namespace in ("physical", "task", "flow"):
            self.assertTrue(set(target_only[namespace]).isdisjoint(sample["static"]["input_entity_index"][namespace]))

    def test_round_trip(self):
        sample = build_sample(self.raw, anchor_step=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json"
            write_sample(sample, path)
            self.assertEqual(sample, load_sample(path))

    def test_empty_noop_is_distinct_from_missing(self):
        sample = build_sample(self.raw, anchor_step=2)
        for action in sample["future_action"]:
            for family in action:
                self.assertFalse(action[family]["missing"])
                self.assertEqual(action[family]["empty"], not bool(action[family]["entries"]))
        sample["future_action"][0]["route"].update(
            {"missing": True, "empty": False, "field_present": False, "entries": [], "missing_reason": "test"}
        )
        self.assertNotEqual(
            sample["future_action"][0]["route"]["missing"],
            sample["future_action"][0]["route"]["empty"],
        )

    def test_history_object_missing_keeps_fixed_index_and_masks(self):
        raw = json.loads(json.dumps(self.raw))
        anchor = raw["decisions"][2]
        missing_entity = anchor["entities"][0]["entity_id"]
        missing_task = anchor["tasks"][0]["task_id"]
        raw["decisions"][1]["entities"] = [row for row in raw["decisions"][1]["entities"] if row["entity_id"] != missing_entity]
        raw["decisions"][1]["tasks"] = [row for row in raw["decisions"][1]["tasks"] if row["task_id"] != missing_task]
        sample = build_sample(raw, anchor_step=2)
        entity = next(row for row in sample["history"][0]["entities"] if row["entity_id"] == missing_entity)
        task = next(row for row in sample["history"][0]["tasks"] if row["task_id"] == missing_task)
        self.assertFalse(entity["presence"])
        self.assertFalse(entity["feature_mask"]["speed_mps"])
        self.assertIsNone(entity["speed_mps"]["value"])
        self.assertFalse(task["presence"])
        self.assertFalse(task["feature_mask"]["task_size"])
        self.assertIsNone(task["task_size"]["value"])

    def test_unresolved_action_reference_is_rejected(self):
        raw = json.loads(json.dumps(self.raw))
        raw["steps"][2]["action"]["route"] = {
            "field_present": True,
            "empty": False,
            "entries": [{"task_id": "future_only_task", "target_node_id": "UAV_0"}],
        }
        with self.assertRaisesRegex(ValueError, "RESEARCHER_DECISION_REQUIRED.*future_only_task"):
            build_sample(raw, anchor_step=2)
        raw = json.loads(json.dumps(self.raw))
        known_task = raw["decisions"][2]["tasks"][0]["task_id"]
        raw["steps"][2]["action"]["route"] = {
            "field_present": True,
            "empty": False,
            "entries": [{"task_id": known_task, "target_node_id": "future_only_node"}],
        }
        with self.assertRaisesRegex(ValueError, "RESEARCHER_DECISION_REQUIRED.*future_only_node"):
            build_sample(raw, anchor_step=2)


if __name__ == "__main__":
    unittest.main()
