"""Policy B tests: pure synthetic behavior and fixed five-frame evidence are separate."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

import audit_v2u4real_r2_04c_label_merge_policy_v1 as audit


def obj(value):
    return {"location": [value, 0, 0], "center": [0, 0, 0], "extent": [1, 1, 1], "angle": [0, 0, 0]}


class SyntheticMergeTests(unittest.TestCase):
    def test_shared_id_uav_overwrites_vehicle_and_keeps_raw_previous(self):
        vehicle = {"1": obj(1), "2": obj(2)}
        uav = {"1": obj(10), "3": obj(3)}
        before_v, before_u = copy.deepcopy(vehicle), copy.deepcopy(uav)
        merged, provenance = audit.merge_official_policy(vehicle, uav)
        self.assertEqual(set(merged), {"1", "2", "3"})
        self.assertEqual(merged["1"]["location"], [10, 0, 0])
        self.assertEqual(provenance["1"]["final_source"], "uav")
        self.assertEqual(provenance["1"]["overwritten_vehicle"]["location"], [1, 0, 0])
        self.assertEqual(vehicle, before_v)
        self.assertEqual(uav, before_u)

    def test_vehicle_only_and_uav_only_are_retained(self):
        merged, provenance = audit.merge_official_policy({"ego-only": obj(1)}, {"uav-only": obj(2)})
        self.assertEqual(set(merged), {"ego-only", "uav-only"})
        self.assertEqual(provenance["ego-only"]["final_source"], "vehicle")
        self.assertEqual(provenance["uav-only"]["final_source"], "uav")

    def test_empty_and_single_side_inputs(self):
        self.assertEqual(audit.merge_official_policy({}, {})[0], {})
        merged, provenance = audit.merge_official_policy({"1": obj(1)}, {})
        self.assertEqual(set(merged), {"1"})
        self.assertEqual(provenance["1"]["final_source"], "vehicle")
        merged, provenance = audit.merge_official_policy({}, {"1": obj(1)})
        self.assertEqual(set(merged), {"1"})
        self.assertEqual(provenance["1"]["final_source"], "uav")

    def test_keys_are_matched_without_string_or_integer_normalization(self):
        vehicle = {1: obj(1), "01": obj(2)}
        uav = {"1": obj(3), "01": obj(4)}
        merged, provenance = audit.merge_official_policy(vehicle, uav)
        self.assertEqual(set(merged), {1, "1", "01"})
        self.assertEqual(provenance[1]["final_source"], "vehicle")
        self.assertEqual(provenance["1"]["final_source"], "uav")
        self.assertEqual(provenance["01"]["final_source"], "uav")

    def test_set_union_and_shared_resolution(self):
        vehicle = {"a": obj(1), "b": obj(2)}
        uav = {"b": obj(3), "c": obj(4)}
        merged, provenance = audit.merge_official_policy(vehicle, uav)
        self.assertEqual(set(merged), set(vehicle) | set(uav))
        self.assertEqual(provenance["b"]["final_source"], "uav")

    def test_invalid_input_rejected_without_repair(self):
        with self.assertRaises(audit.InterfaceBlocked):
            audit.merge_official_policy([], {})
        with self.assertRaises(audit.InterfaceBlocked):
            audit.merge_official_policy({"1": None}, {})

    def test_source_identity_and_excerpt_are_fixed(self):
        identity = audit.verify_official_source()
        self.assertEqual(identity["commit"], audit.OFFICIAL_COMMIT)
        self.assertEqual(identity["sha256"], audit.OFFICIAL_SHA)
        self.assertTrue(any("merged_objs.update(ego_objs)" in line for line in identity["source_lines"]["text"]))
        self.assertTrue(any("merged_objs[oid] = obj" in line for line in identity["source_lines"]["text"]))

    def test_output_path_guard(self):
        with self.assertRaises(audit.InterfaceBlocked):
            audit.run(audit.ROOT / "code/artifacts/data/v2u4real_r2_01")
        with self.assertRaises(audit.InterfaceBlocked):
            audit.run(audit.ROOT / "code/artifacts/audit/v2u4real_r2_04b_20261026")

    def test_source_failure_produces_blocked_receipt(self):
        with tempfile.TemporaryDirectory(prefix="v2u4real_r2_04c_", dir=audit.OUT.parent) as folder:
            with patch.object(audit, "verify_official_source", side_effect=audit.InterfaceBlocked("source changed")):
                result = audit.run(Path(folder))
            self.assertEqual(result["task_execution_status"], "BLOCKED")
            self.assertEqual(result["source_contract_status"], "BLOCKED")
            self.assertEqual(result["frames"], [])
            self.assertTrue((Path(folder) / "result.json").is_file())

    def test_python_equal_keys_follow_official_dictionary_behavior(self):
        merged, provenance = audit.merge_official_policy({1: obj(1)}, {True: obj(2)})
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[1], obj(2))
        self.assertEqual(provenance[1]["final_source"], "uav")


class RealFiveFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="v2u4real_r2_04c_", dir=audit.OUT.parent)
        cls.result = audit.run(Path(cls.temp.name))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_status_and_five_keys(self):
        self.assertEqual(self.result["policy_status"], "FROZEN_BY_RESEARCHER_CHOICE")
        self.assertEqual(self.result["input_integrity_status"], "PASS")
        self.assertEqual(self.result["label_merge_status"], "PASS")
        self.assertEqual(self.result["task_execution_status"], "COMPLETED")
        self.assertEqual([f["frame_key"] for f in self.result["frames"]], list(audit.KEYS))

    def test_real_expected_sets_and_cover_counts(self):
        expected = [
            ({"1", "2", "3", "4", "5", "6"}, {"1", "2"}, {"1", "2"}, {"3", "4", "5", "6"}, set(), 2, 6),
            ({"1", "2", "3", "4", "6"}, {"1", "2"}, {"1", "2"}, {"3", "4", "6"}, set(), 2, 5),
            ({"1", "2", "3", "4", "6"}, {"1", "2"}, {"1", "2"}, {"3", "4", "6"}, set(), 2, 5),
            ({"1", "2", "3", "4", "5", "6"}, {"2"}, {"2"}, {"1", "3", "4", "5", "6"}, set(), 1, 6),
            ({"1", "2", "3", "4", "5", "6"}, {"2"}, {"2"}, {"1", "3", "4", "5", "6"}, set(), 1, 6),
        ]
        for frame, exp in zip(self.result["frames"], expected):
            vehicle, uav, shared, vehicle_only, uav_only, covered, output_count = exp
            self.assertEqual(set(frame["vehicle_ids"]), vehicle, frame["frame_key"])
            self.assertEqual(set(frame["uav_ids"]), uav, frame["frame_key"])
            self.assertEqual(set(frame["shared_ids"]), shared, frame["frame_key"])
            self.assertEqual(set(frame["vehicle_only_ids"]), vehicle_only, frame["frame_key"])
            self.assertEqual(set(frame["uav_only_ids"]), uav_only, frame["frame_key"])
            self.assertEqual(frame["counts"]["vehicle_overwritten_by_uav"], covered)
            self.assertEqual(frame["counts"]["merged_output"], output_count)
            self.assertTrue(all(frame["set_checks"].values()))

    def test_real_final_sources_and_overwritten_records(self):
        for frame in self.result["frames"]:
            for oid, record in frame["final_objects"].items():
                if oid in frame["shared_ids"]:
                    self.assertEqual(record["final_source"], "uav")
                    self.assertIsNotNone(record["overwritten_vehicle"])
                elif oid in frame["vehicle_only_ids"]:
                    self.assertEqual(record["final_source"], "vehicle")
                else:
                    self.assertEqual(record["final_source"], "uav")

    def test_real_sha_unchanged_and_no_label_write(self):
        self.assertEqual(len(self.result["input_hashes_before_after"]), 10)
        self.assertTrue(all(row["unchanged"] for row in self.result["input_hashes_before_after"]))
        self.assertFalse(self.result["runtime"]["labels_written"])

    def test_real_final_payloads_are_official_converted_objects(self):
        official, _ = audit.official_functions()
        for frame in self.result["frames"]:
            t = official.x1_to_x2(frame["raw_lidar_pose"]["uav"], frame["raw_lidar_pose"]["vehicle"])
            np.testing.assert_array_equal(t, frame["T_vehicle_from_uav"])
            ego = official.load_annotation_objects(str(audit.ROOT / frame["vehicle_input"]["path"]))
            uav = official.load_annotation_objects(str(audit.ROOT / frame["uav_input"]["path"]), transform=t)
            expected = dict(ego)
            expected.update(uav)
            for oid, record in frame["final_objects"].items():
                self.assertEqual(record["object"], expected[oid])
                self.assertEqual(record["interpretation"], "official_code_interpretation_only")
                if oid in frame["shared_ids"]:
                    self.assertEqual(record["overwritten_vehicle"], record["vehicle_input"])
                    self.assertEqual(record["overwritten_vehicle_official_object"], ego[oid])

    def test_missing_fields_nonfinite_and_unexpected_ids_block_without_repair(self):
        _, refs = audit.load_r2_04b_reference()
        official, _ = audit.official_functions()
        original = audit._read_yaml
        mutations = ("missing_field", "nonfinite", "unexpected_id")
        for mutation in mutations:
            def altered(path):
                value, metadata = original(path)
                objects = value["vehicles"]
                oid = next(iter(objects))
                if mutation == "missing_field":
                    del objects[oid]["extent"]
                elif mutation == "nonfinite":
                    objects[oid]["angle"][0] = float("nan")
                else:
                    objects[1] = objects.pop(oid)
                return value, metadata
            with self.subTest(mutation=mutation), patch.object(audit, "_read_yaml", side_effect=altered):
                with self.assertRaises((audit.InterfaceBlocked, KeyError)):
                    audit.audit_frame(audit.KEYS[0], refs[audit.KEYS[0]]["inputs"], official)


if __name__ == "__main__":
    unittest.main()
