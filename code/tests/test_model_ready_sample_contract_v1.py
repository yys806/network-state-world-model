import json
import tempfile
import unittest
from pathlib import Path

from pi_jwm.model_ready_sample_contract_v1 import build_sample, load_sample, validate_sample, write_sample


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v1_20260919/real_communication_outcome_semantics.json"


class ModelReadySampleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(RAW.read_text(encoding="utf-8"))

    def test_real_raw_sample_has_causal_windows_and_four_actions(self):
        sample = build_sample(self.raw, anchor_step=2)
        checks = validate_sample(sample)
        self.assertTrue(all(checks.values()))
        self.assertEqual([0, 1], sample["metadata"]["history_frame_indices"])
        self.assertEqual([2, 3], sample["metadata"]["future_action_frame_indices"])
        self.assertIn("communication_service", sample["target"][0])
        self.assertIn("wireless_delivered_data_by_task", sample["target"][0]["communication_service"])

    def test_target_only_object_is_not_input_index(self):
        sample = build_sample(self.raw, anchor_step=2)
        self.assertTrue(sample["static"]["target_only_objects"])
        self.assertTrue(set(sample["static"]["target_only_objects"]).isdisjoint(sample["static"]["input_entity_index"]["task"]))

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


if __name__ == "__main__":
    unittest.main()
