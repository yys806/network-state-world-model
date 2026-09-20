import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pi_jwm.step4_2a_graph_input_extension_v1 import (
    COMM_RELATION_TYPE_VOCAB,
    RAW_SCHEMA_VERSION,
    SAMPLE_SCHEMA_VERSION,
    TENSOR_SCHEMA_VERSION,
    amend_raw_graph_inputs,
    apply_extension_normalization,
    build_extended_sample,
    build_extended_tensor_batch,
    fit_extension_normalization_stats,
    load_extended_tensor_batch,
    save_extended_tensor_batch,
    validate_extended_sample_checks,
    validate_extended_tensor_checks,
)


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/real_communication_outcome_semantics.json"


class Step42AGraphInputExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(RAW.read_text(encoding="utf-8"))

    def _sample(self, raw=None, *, split="dev_train"):
        amended = amend_raw_graph_inputs(copy.deepcopy(raw or self.raw))
        sample = build_extended_sample(amended, anchor_step=2)
        sample["metadata"]["split"] = split
        sample["metadata"]["source_split"] = split
        sample["metadata"]["sample_id"] = f"{sample['metadata']['trajectory_id']}::anchor-0002::{split}"
        return amended, sample

    def _tensor(self, sample):
        stats = fit_extension_normalization_stats([sample])
        normalized = apply_extension_normalization([sample], stats)
        return build_extended_tensor_batch(normalized, stats=stats), stats

    def test_raw_amendment_materializes_bidirectional_wired_rows_before_action(self):
        amended, _ = self._sample()
        self.assertEqual(amended["schema_version"], RAW_SCHEMA_VERSION)
        for decision in amended["decisions"]:
            rows = decision["wired_relation_rows"]
            self.assertEqual({(row["source_id"], row["target_id"]) for row in rows}, {
                ("RSU_0", "cloudServer_4"), ("cloudServer_4", "RSU_0")
            })
            self.assertTrue(all(row["capture_phase"] == "decision_before_action" for row in rows))
            self.assertTrue(all(row["presence"] and row["validity"] for row in rows))
            self.assertTrue(all(row["csi_value"] is None and row["csi_feature_mask"] is False for row in rows))

    def test_future_target_position_does_not_change_history_position_tensor(self):
        _, sample = self._sample()
        baseline, _ = self._tensor(sample)
        changed = copy.deepcopy(sample)
        changed["target"][0]["entities"][0]["position_m"]["value"] = [1e9, -1e9, 5e8]
        candidate, _ = self._tensor(changed)
        self.assertTrue(np.array_equal(baseline["entity_position"], candidate["entity_position"]))
        self.assertTrue(np.array_equal(baseline["entity_position_mask"], candidate["entity_position_mask"]))

    def test_validation_csi_extreme_does_not_change_train_statistics(self):
        _, train = self._sample(split="dev_train")
        _, validation = self._sample(split="dev_validation")
        stats = fit_extension_normalization_stats([train, validation])
        changed = copy.deepcopy(validation)
        wireless = next(row for row in changed["history"][0]["communication_relations"] if row["relation_type"] == "wireless")
        wireless["csi_values"][0] = 1e12
        self.assertEqual(stats, fit_extension_normalization_stats([train, changed]))

    def test_wired_relation_can_be_valid_with_csi_mask_false(self):
        _, sample = self._sample()
        tensor, _ = self._tensor(sample)
        wired_code = COMM_RELATION_TYPE_VOCAB.index("wired")
        wired = tensor["comm_relation_type_index"] == wired_code
        self.assertTrue(np.any(wired & tensor["comm_relation_validity"]))
        self.assertFalse(np.any(tensor["comm_csi_mask"][wired]))

    def test_wired_service_outcome_does_not_change_decision_relation(self):
        _, baseline = self._sample()
        changed_raw = copy.deepcopy(self.raw)
        changed_raw["steps"][1]["outcome"]["wired_delivered_data_by_task"] = {"Task_1": 1e9}
        _, changed = self._sample(changed_raw)
        self.assertEqual(
            [frame["communication_relations"] for frame in baseline["history"]],
            [frame["communication_relations"] for frame in changed["history"]],
        )

    def test_cpu_capacity_changes_static_capability_not_comp_action(self):
        _, baseline_sample = self._sample()
        baseline, _ = self._tensor(baseline_sample)
        changed_raw = copy.deepcopy(self.raw)
        for decision in changed_raw["decisions"]:
            row = next(row for row in decision["node_cpu_capacity_observation_rows"] if row["node_id"] == "RSU_0")
            row["capacity_per_s"] += 7.0
        _, changed_sample = self._sample(changed_raw)
        changed, _ = self._tensor(changed_sample)
        self.assertFalse(np.array_equal(baseline["agent_cpu_capacity_raw"], changed["agent_cpu_capacity_raw"]))
        self.assertTrue(np.array_equal(baseline["future_comp_allocated_cpu"], changed["future_comp_allocated_cpu"]))

    def test_comp_allocation_does_not_change_static_capability(self):
        _, sample = self._sample()
        baseline, _ = self._tensor(sample)
        changed = copy.deepcopy(sample)
        entry = next((entry for frame in changed["future_action"] for entry in frame["comp"]["entries"]), None)
        if entry is None:
            task_id, task_index = next(iter(changed["static"]["input_entity_index"]["task"].items()))
            node_id, node_index = next(iter(changed["static"]["input_entity_index"]["physical"].items()))
            entry = {"task_id": task_id, "task_index": task_index, "node_id": node_id, "node_index": node_index, "allocated_cpu_per_s": 3.0}
            changed["future_action"][0]["comp"] = {"field_present": True, "empty": False, "missing": False, "entries": [entry]}
        else:
            entry["allocated_cpu_per_s"] = float(entry.get("allocated_cpu_per_s", 0.0)) + 3.0
        candidate, _ = self._tensor(changed)
        self.assertTrue(np.array_equal(baseline["agent_cpu_capacity_raw"], candidate["agent_cpu_capacity_raw"]))

    def test_cpu_service_outcome_does_not_change_static_capability(self):
        _, sample = self._sample()
        baseline, _ = self._tensor(sample)
        changed = copy.deepcopy(sample)
        changed["history"][0]["outcome"]["served_cpu_work_by_task"] = {"Task_1": 1e9}
        candidate, _ = self._tensor(changed)
        self.assertTrue(np.array_equal(baseline["agent_cpu_capacity_raw"], candidate["agent_cpu_capacity_raw"]))

    def test_current_task_progress_enters_history_tensor(self):
        amended, sample = self._sample()
        tensor, _ = self._tensor(sample)
        decision = amended["decisions"][sample["metadata"]["history_frame_indices"][0]]
        raw_task = decision["tasks"][0]
        slot = sample["static"]["input_entity_index"]["task"][raw_task["task_id"]]
        order = tensor["contract"]["task_history_feature_order"]
        self.assertEqual(tensor["task_history_extended_raw_features"][0, 0, slot, order.index("computed_cpu_work")], raw_task["computed_cpu_work"])
        self.assertEqual(tensor["task_history_extended_raw_features"][0, 0, slot, order.index("transmitted_size")], raw_task["transmitted_size"])

    def test_future_target_progress_does_not_leak_into_history(self):
        _, sample = self._sample()
        baseline, _ = self._tensor(sample)
        changed = copy.deepcopy(sample)
        changed["target"][0]["tasks"][0]["computed_cpu_work"]["value"] = 1e9
        changed["target"][0]["tasks"][0]["transmitted_size"]["value"] = 1e9
        candidate, _ = self._tensor(changed)
        self.assertTrue(np.array_equal(baseline["task_history_extended_features"], candidate["task_history_extended_features"]))

    def test_host_and_exec_do_not_use_future_route_target(self):
        _, sample = self._sample()
        baseline, _ = self._tensor(sample)
        changed = copy.deepcopy(sample)
        route = next((entry for frame in changed["future_action"] for entry in frame["route"]["entries"]), None)
        if route is not None:
            node_id, node_index = next((item for item in changed["static"]["input_entity_index"]["physical"].items() if item[0] != route["target_node_id"]))
            route["target_node_id"] = node_id
            route["target_node_index"] = node_index
        candidate, _ = self._tensor(changed)
        self.assertTrue(np.array_equal(baseline["task_agent_task_index"], candidate["task_agent_task_index"]))
        self.assertTrue(np.array_equal(baseline["task_agent_agent_index"], candidate["task_agent_agent_index"]))
        self.assertTrue(np.array_equal(baseline["task_agent_relation_type_index"], candidate["task_agent_relation_type_index"]))

    def test_relation_endpoint_id_index_and_tensor_slot_are_consistent(self):
        _, sample = self._sample()
        tensor, _ = self._tensor(sample)
        index = sample["static"]["input_entity_index"]["physical"]
        for history_index, frame in enumerate(sample["history"]):
            for relation_index, row in enumerate(frame["communication_relations"]):
                self.assertEqual(row["source_agent_index"], index[row["source_id"]])
                self.assertEqual(row["target_agent_index"], index[row["target_id"]])
                self.assertEqual(tensor["comm_source_index"][0, history_index, relation_index], index[row["source_id"]])
                self.assertEqual(tensor["comm_target_index"][0, history_index, relation_index], index[row["target_id"]])

    def test_late_entry_and_disappearance_keep_stable_slot(self):
        late_raw = copy.deepcopy(self.raw)
        late_raw["decisions"][1]["entities"] = [row for row in late_raw["decisions"][1]["entities"] if row["entity_id"] != "RSU_3"]
        _, late = self._sample(late_raw)
        late_slot = late["static"]["input_entity_index"]["physical"]["RSU_3"]
        self.assertFalse(late["history"][0]["entities"][late_slot]["presence"])
        self.assertTrue(late["history"][1]["entities"][late_slot]["presence"])

        gone_raw = copy.deepcopy(self.raw)
        gone_raw["decisions"][2]["entities"] = [row for row in gone_raw["decisions"][2]["entities"] if row["entity_id"] != "RSU_3"]
        _, gone = self._sample(gone_raw)
        gone_slot = gone["static"]["input_entity_index"]["physical"]["RSU_3"]
        self.assertTrue(gone["history"][0]["entities"][gone_slot]["presence"])
        self.assertFalse(gone["history"][1]["entities"][gone_slot]["presence"])
        self.assertEqual(late_slot, gone_slot)

    def test_versions_validation_and_round_trip(self):
        _, sample = self._sample()
        sample_checks = validate_extended_sample_checks(sample)
        self.assertTrue(sample_checks["passed"], sample_checks)
        tensor, _ = self._tensor(sample)
        self.assertEqual(sample["schema_version"], SAMPLE_SCHEMA_VERSION)
        self.assertEqual(tensor["schema_version"], TENSOR_SCHEMA_VERSION)
        self.assertTrue(validate_extended_tensor_checks(tensor)["passed"])
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "tensor.npz"
            save_extended_tensor_batch(tensor, path)
            loaded = load_extended_tensor_batch(path)
        self.assertEqual(tensor["contract"], loaded["contract"])
        for key in ("entity_position", "comm_csi", "agent_cpu_capacity", "task_agent_agent_index"):
            self.assertTrue(np.array_equal(tensor[key], loaded[key]))


if __name__ == "__main__":
    unittest.main()
