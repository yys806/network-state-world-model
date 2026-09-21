import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pi_jwm.step5_1a_motion_csi_target_contract_v1 import (
    COMM_RELATION_TYPE_VOCAB,
    SAMPLE_SCHEMA_VERSION,
    TENSOR_SCHEMA_VERSION,
    build_future_target_tensor_batch,
    denormalize_csi,
    denormalize_motion,
    extend_future_motion_csi_targets,
    future_target_digest,
    load_future_target_tensor_batch,
    normalize_csi,
    normalize_motion,
    save_future_target_tensor_batch,
    validate_future_target_sample_checks,
    validate_future_target_tensor_checks,
    validate_step5_1a_acceptance,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "code/artifacts/protocols/pi_jwm_step4_2a_graph_input_extension_v1_20260920"
STATS = ROOT / "code/artifacts/protocols/pi_jwm_step4_3b_dual_graph_encoder_v1_20260920/encoder_normalization_stats.json"


class Step51AMotionCsiTargetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = json.loads((ARTIFACT / "normalized_samples.json").read_text(encoding="utf-8"))
        cls.stats = json.loads(STATS.read_text(encoding="utf-8"))
        cls.raw_by_trajectory = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in (ARTIFACT / "raw_amendments").glob("*.json")
        }

    def _extended(self, index=0, *, sample=None, raw=None):
        source = copy.deepcopy(sample or self.samples[index])
        trajectory = source["metadata"]["trajectory_id"]
        raw_source = copy.deepcopy(raw or self.raw_by_trajectory[trajectory])
        return extend_future_motion_csi_targets(source, raw_source, self.stats)

    def _tensor(self, samples=None):
        rows = samples or [self._extended()]
        return build_future_target_tensor_batch(rows, self.stats)

    def test_vehicle_motion_is_delta_xyz_plus_next_speed_and_nonvehicle_is_masked(self):
        extended = self._extended()
        current = {row["entity_id"]: row for row in extended["history"][-1]["entities"]}
        first = extended["target"][0]
        vehicle = next(row for row in first["vehicle_motion_targets"] if row["entity_type"] == "vehicle")
        future = next(row for row in first["entities"] if row["entity_id"] == vehicle["entity_id"])
        expected_delta = np.asarray(future["position_m"]["value"]) - np.asarray(current[vehicle["entity_id"]]["position_m"]["value"])
        self.assertTrue(np.allclose(vehicle["raw_value"][:3], expected_delta))
        self.assertAlmostEqual(vehicle["raw_value"][3], future["speed_mps"]["value"])
        self.assertEqual(vehicle["semantic_order"], ["delta_x_m", "delta_y_m", "delta_z_m", "next_speed_mps"])
        non_vehicle = next(row for row in first["vehicle_motion_targets"] if row["entity_type"] != "vehicle")
        self.assertFalse(any(non_vehicle["mask"]))

    def test_motion_component_masks_follow_missing_position_and_speed(self):
        sample = copy.deepcopy(self.samples[0])
        vehicle = next(row for row in sample["target"][0]["entities"] if row["entity_type"] == "vehicle")
        vehicle["position_m"]["feature_mask"][1] = False
        vehicle["position_m"]["value"][1] = None
        vehicle["speed_mps"]["feature_mask"] = False
        vehicle["speed_mps"]["value"] = None
        extended = self._extended(sample=sample)
        row = next(row for row in extended["target"][0]["vehicle_motion_targets"] if row["entity_id"] == vehicle["entity_id"])
        self.assertEqual(row["mask"], [True, False, True, False])
        self.assertEqual(row["raw_value"][1], 0.0)
        self.assertEqual(row["normalized_value"][3], 0.0)

    def test_motion_normalization_uses_position_scale_not_mean_and_round_trips(self):
        raw = np.asarray([12.0, -7.0, 3.5, 9.0])
        mask = np.asarray([True, True, True, True])
        changed = copy.deepcopy(self.stats)
        changed["features"]["entity.position_x_m"]["mean"] = 1e9
        normalized = normalize_motion(raw, mask, changed)
        self.assertAlmostEqual(normalized[0], raw[0] / changed["features"]["entity.position_x_m"]["std"])
        self.assertTrue(np.allclose(denormalize_motion(normalized, mask, changed), raw))

    def test_future_csi_comes_from_future_outcome_and_history_csi_cannot_replace_it(self):
        baseline = self._extended()
        changed = copy.deepcopy(self.samples[0])
        relation = next(row for row in changed["history"][-1]["communication_relations"] if row["relation_type"] == "wireless")
        relation["csi_values"] = [value + 1e6 for value in relation["csi_values"]]
        candidate = self._extended(sample=changed)
        self.assertEqual(baseline["target"][0]["comm_csi_targets"], candidate["target"][0]["comm_csi_targets"])

        target = next(row for row in baseline["target"][0]["comm_csi_targets"] if row["relation_type"] == "wireless")
        raw = self.raw_by_trajectory[baseline["metadata"]["trajectory_id"]]
        outcome = next(step["outcome"] for step in raw["steps"] if step["frame_index"] == baseline["target"][0]["frame_index"])
        observed = next(row for row in outcome["channel_rows"] if f"comm::wireless::{row['physical_edge_id']}" == target["relation_id"])
        self.assertTrue(np.allclose(target["raw_value"], observed["channel_attenuation_db"]))
        self.assertEqual(target["source_capture_phase"], "outcome_after_action")

    def test_current_support_alignment_wired_and_missing_future_csi_masks(self):
        baseline = self._extended()
        support = baseline["static"]["future_target_current_comm_support"]
        for frame in baseline["target"]:
            self.assertEqual([row["relation_id"] for row in frame["comm_csi_targets"]], [row["relation_id"] for row in support])
            wired = [row for row in frame["comm_csi_targets"] if row["relation_type"] == "wired"]
            self.assertTrue(wired)
            self.assertTrue(all(not any(row["mask"]) for row in wired))

        raw = copy.deepcopy(self.raw_by_trajectory[baseline["metadata"]["trajectory_id"]])
        frame = baseline["target"][0]["frame_index"]
        outcome = next(step["outcome"] for step in raw["steps"] if step["frame_index"] == frame)
        missing = outcome["channel_rows"][0]
        missing_id = f"comm::wireless::{missing['physical_edge_id']}"
        missing["observed_mask"] = False
        missing["channel_attenuation_db"] = []
        missing["missing_reason"] = "COUNTERFACTUAL_FUTURE_CSI_MISSING"
        changed = self._extended(raw=raw)
        row = next(row for row in changed["target"][0]["comm_csi_targets"] if row["relation_id"] == missing_id)
        self.assertEqual(row["relation_id"], missing_id)
        self.assertFalse(any(row["mask"]))
        self.assertTrue(all(value == 0.0 for value in row["raw_value"]))

    def test_future_only_relation_is_counted_but_does_not_expand_current_support(self):
        baseline = self._extended()
        raw = copy.deepcopy(self.raw_by_trajectory[baseline["metadata"]["trajectory_id"]])
        frame = baseline["target"][0]["frame_index"]
        outcome = next(step["outcome"] for step in raw["steps"] if step["frame_index"] == frame)
        added = copy.deepcopy(outcome["channel_rows"][0])
        added["physical_edge_id"] = "physical::future_only_a::future_only_b::V2V"
        added["source_id"] = "future_only_a"
        added["target_id"] = "future_only_b"
        outcome["channel_rows"].append(added)
        changed = self._extended(raw=raw)
        self.assertEqual(len(changed["static"]["future_target_current_comm_support"]), len(baseline["static"]["future_target_current_comm_support"]))
        self.assertFalse(any(row["relation_id"].endswith("future_only_a::future_only_b::V2V") for row in changed["target"][0]["comm_csi_targets"]))
        self.assertEqual(changed["target"][0]["future_target_side_metadata"]["future_only_relation_count"], 1)

    def test_future_csi_reorders_by_rb_identity_not_source_list_position(self):
        baseline = self._extended()
        raw = copy.deepcopy(self.raw_by_trajectory[baseline["metadata"]["trajectory_id"]])
        frame = baseline["target"][0]["frame_index"]
        outcome = next(step["outcome"] for step in raw["steps"] if step["frame_index"] == frame)
        relation = next(row for row in outcome["channel_rows"] if row["channel_type"] == "I2I")
        relation["rb_indices"] = list(reversed(relation["rb_indices"]))
        relation["channel_attenuation_db"] = list(reversed(relation["channel_attenuation_db"]))
        changed = self._extended(raw=raw)
        baseline_row = next(row for row in baseline["target"][0]["comm_csi_targets"] if row["relation_id"] == f"comm::wireless::{relation['physical_edge_id']}")
        changed_row = next(row for row in changed["target"][0]["comm_csi_targets"] if row["relation_id"] == baseline_row["relation_id"])
        self.assertEqual(changed_row["rb_indices"], baseline_row["rb_indices"])
        self.assertTrue(np.allclose(changed_row["raw_value"], baseline_row["raw_value"]))

    def test_csi_normalization_uses_frozen_train_stats_and_round_trips(self):
        raw = np.asarray([80.0, 95.0, 110.0])
        mask = np.asarray([True, True, True])
        normalized = normalize_csi(raw, mask, self.stats)
        feature = self.stats["features"]["comm.channel_attenuation_db"]
        self.assertTrue(np.allclose(normalized, (raw - feature["mean"]) / feature["std"]))
        self.assertTrue(np.allclose(denormalize_csi(normalized, mask, self.stats), raw))

    def test_unsupported_return_birth_is_side_metadata_not_window_deletion(self):
        baseline = self._extended()
        sample = copy.deepcopy(self.samples[0])
        sample["target"][0]["unsupported_future_structure"] = {
            "unsupported": [{"kind": "future_return_birth"}],
            "unresolved": [{"kind": "return_requirement_unknown"}],
            "fixed_support_blocked": [{"kind": "future_return_birth"}],
        }
        changed = self._extended(sample=sample)
        self.assertEqual(baseline["target"][0]["vehicle_motion_targets"], changed["target"][0]["vehicle_motion_targets"])
        self.assertEqual(baseline["target"][0]["comm_csi_targets"], changed["target"][0]["comm_csi_targets"])
        side = changed["target"][0]["future_target_side_metadata"]
        self.assertEqual((side["unsupported_count"], side["unresolved_count"], side["fixed_support_blocked_count"]), (1, 1, 1))

    def test_additive_target_namespace_does_not_modify_history_or_input(self):
        source = copy.deepcopy(self.samples[0])
        history = copy.deepcopy(source["history"])
        static_input = copy.deepcopy(source["static"]["input_entity_index"])
        extended = self._extended(sample=source)
        self.assertEqual(extended["history"], history)
        self.assertEqual(extended["static"]["input_entity_index"], static_input)
        self.assertTrue(all("vehicle_motion_targets" not in frame for frame in extended["history"]))
        self.assertEqual(extended["schema_version"], SAMPLE_SCHEMA_VERSION)

    def test_tensor_schema_validation_round_trip_and_tamper_rejection(self):
        extended = self._extended()
        self.assertTrue(validate_future_target_sample_checks(extended)["passed"])
        tensor = self._tensor([extended])
        self.assertEqual(tensor["schema_version"], TENSOR_SCHEMA_VERSION)
        self.assertTrue(validate_future_target_tensor_checks(tensor)["passed"])
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "targets.npz"
            save_future_target_tensor_batch(tensor, path)
            loaded = load_future_target_tensor_batch(path)
        self.assertEqual(tensor["contract"], loaded["contract"])
        for key in ("target_vehicle_motion_raw", "target_vehicle_motion_normalized", "target_vehicle_motion_mask", "target_comm_csi_raw", "target_comm_csi_normalized", "target_comm_csi_mask"):
            self.assertTrue(np.array_equal(tensor[key], loaded[key]))

        tampered = copy.deepcopy(tensor)
        location = np.argwhere(tampered["target_vehicle_motion_mask"])[0]
        tampered["target_vehicle_motion_normalized"][tuple(location)] += 1.0
        checks = validate_future_target_tensor_checks(tampered)
        self.assertFalse(checks["motion_normalization_semantic_equality"])
        self.assertFalse(checks["passed"])

    def test_deterministic_digest_and_receipt_are_actual_and(self):
        first = [self._extended(index) for index in range(3)]
        second = [self._extended(index) for index in range(3)]
        tensor_a = self._tensor(first)
        tensor_b = self._tensor(second)
        self.assertEqual(future_target_digest(first, tensor_a), future_target_digest(second, tensor_b))
        receipt = validate_step5_1a_acceptance(first, tensor_a, deterministic_rebuild=True, real_trajectory_verified=True)
        self.assertTrue(receipt["passed"], receipt)
        failed = validate_step5_1a_acceptance(first, tensor_a, deterministic_rebuild=False, real_trajectory_verified=True)
        self.assertFalse(failed["passed"])
        self.assertTrue(all(receipt["required_checks"].values()))
        self.assertEqual(receipt["scope"], {"formal_dataset": False, "training": False, "gpu": False, "locked_test_accessed": False})


if __name__ == "__main__":
    unittest.main()
