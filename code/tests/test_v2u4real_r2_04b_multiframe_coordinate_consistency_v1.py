"""Synthetic semantics checks are separate from five-real-frame evidence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import audit_v2u4real_r2_04b_multiframe_coordinate_consistency_v1 as b


def obj(location, extent=None, center=None):
    return {"location": location, "center": center or [0, 0, 0],
            "extent": extent or [4, 2, 1], "angle": [0, 0, 0], "obj_type": "Car"}


class SyntheticTests(unittest.TestCase):
    def test_centers_use_location_plus_center_and_correct_direction(self):
        raw = {"vehicle": {"vehicles": {"1": obj([12, 24, 36], center=[1, 1, 1])}},
               "uav": {"vehicles": {"1": obj([1, 2, 3], center=[2, 3, 4])}}}
        t = np.eye(4); t[:3, 3] = [10, 20, 30]
        result = b.compare_annotations(raw, t)
        c = result["comparisons"][0]
        self.assertEqual(c["distance_3d_code_units"], 0)
        self.assertEqual(c["uav_center_local_code"], [3, 5, 7])
        self.assertEqual(c["uav_center_in_vehicle_code"], [13, 25, 37])
        self.assertTrue(result["no_physical_error_threshold"])

    def test_raw_extent_differences_not_hidden_or_reinterpreted(self):
        raw = {"vehicle": {"vehicles": {"1": obj([0, 0, 0], [4, 2, 1])}},
               "uav": {"vehicles": {"1": obj([0, 0, 0], [8, 1, 3])}}}
        before = copy.deepcopy(raw)
        c = b.compare_annotations(raw, np.eye(4))["comparisons"][0]
        self.assertEqual(c["extent_delta_uav_minus_vehicle"], [4, -1, 2])
        self.assertEqual(c["extent_relative_delta_over_vehicle"], [1, -.5, 2])
        self.assertEqual(raw, before)
        self.assertEqual(c["status"], "official_code_interpretation_only")

    def test_unshared_ids_are_retained_and_not_matched_by_position(self):
        raw = {"vehicle": {"vehicles": {"1": obj([0, 0, 0])}},
               "uav": {"vehicles": {"2": obj([0, 0, 0])}}}
        r = b.compare_annotations(raw, np.eye(4))
        self.assertEqual(r["common_count"], 0)
        self.assertEqual(r["only_vehicle_ids"], ["1"])
        self.assertEqual(r["only_uav_ids"], ["2"])

    def test_object_id_types_not_silently_normalized(self):
        raw = {"vehicle": {"vehicles": {1: obj([0, 0, 0])}},
               "uav": {"vehicles": {"1": obj([0, 0, 0])}}}
        self.assertEqual(b.compare_annotations(raw, np.eye(4))["common_count"], 0)

    def test_nonfinite_and_nonpositive_annotation_rejected(self):
        for field, value in [("location", [0, np.nan, 1]), ("angle", [0, np.inf, 0]), ("extent", [0, 1, 2])]:
            raw = {s: {"vehicles": {"1": obj([0, 0, 0])}} for s in ["vehicle", "uav"]}
            raw["uav"]["vehicles"]["1"][field] = value
            with self.subTest(field=field), self.assertRaises(b.a.InterfaceBlocked):
                b.compare_annotations(raw, np.eye(4))

    def test_duplicate_decoded_cloud_detected_even_with_different_bytes(self):
        frames = [{"frame_key": key, "inputs": {side: {"pcd": {
            "sha256_before": key+side, "decoded_xyz_sha256": "shared" if side == "vehicle" else key}}
            for side in ["vehicle", "uav"]}} for key in ["a", "b"]]
        result = b.duplicate_check(frames, {"duplicate_pcd_groups": [], "pairs": []})
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["within_selected_five_duplicates"][0]["identity"], "decoded_xyz_sha256")

    def test_known_duplicate_payload_detected(self):
        frames = [{"frame_key": "a", "inputs": {side: {"pcd": {
            "sha256_before": "known" if side == "vehicle" else "other", "decoded_xyz_sha256": side}}
            for side in ["vehicle", "uav"]}}]
        result = b.duplicate_check(frames, {"duplicate_pcd_groups": [{"sha256": "known"}], "pairs": []})
        self.assertEqual(result["status"], "BLOCKED")

    def test_distribution_no_comparison_is_explicit(self):
        r = b.scalar_distribution([])
        self.assertEqual(r["count"], 0)
        self.assertIsNone(r["max"])
        self.assertEqual(b.scalar_distribution([0, 1, 3])["mean"], 4/3)

    def test_no_auto_substitution_for_missing_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(b.a.InterfaceBlocked):
                b.read_frame(b.KEYS[0], {}, Path(tmp))
        with self.assertRaises(b.a.InterfaceBlocked):
            b.read_frame("1752814590.3", {})

    def test_tolerances_identical_to_r2_04a(self):
        previous = b.load_json(b.ROOT/"code/artifacts/audit/v2u4real_r2_04a_20261026/config.json")
        self.assertEqual(b.CONFIG, previous)
        self.assertEqual(b.CONFIG["inverse_max_abs_atol"], 1e-8)
        self.assertEqual(b.CONFIG["distance_max_abs_atol"], 1e-8)

    def test_output_cannot_write_to_raw_data(self):
        with self.assertRaises(b.a.InterfaceBlocked):
            b.run(output=b.ROOT/"code/artifacts/data/v2u4real_r2_01")


class RealFiveFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = b.load_json(b.ROOT/b.a.R2_03)
        cls.by_key = {r["frame_key"]: r for r in cls.reference["pairs"]}
        cls.data = [b.read_frame(k, cls.by_key[k]) for k in b.KEYS]
        cls.official, _ = b.a.official_functions()

    def test_exact_five_frame_read_and_duplicate_status(self):
        self.assertEqual([r[0]["frame_key"] for r in self.data], list(b.KEYS))
        self.assertEqual(b.duplicate_check([r[0] for r in self.data], self.reference)["status"], "PASS")
        self.assertEqual([len(r[1]["vehicle"]) for r in self.data], [141177, 142866, 143252, 144406, 144002])
        self.assertEqual([len(r[1]["uav"]) for r in self.data], [70532, 70751, 70732, 70579, 70597])

    def test_independent_homogeneous_all_points_five_frames(self):
        for row, arrays, raw in self.data:
            with self.subTest(key=row["frame_key"]):
                v, u = raw["vehicle"]["lidar_pose"], raw["uav"]["lidar_pose"]
                t = self.official.x1_to_x2(u, v)
                p = arrays["uav"]; h = np.column_stack([p, np.ones(len(p))])
                direct = np.linalg.solve(self.official.x_to_world(v), self.official.x_to_world(u)@h.T).T
                projected, checks = b.a.geometry_checks(p, t)
                np.testing.assert_allclose(projected, direct[:, :3], rtol=0, atol=1e-10)
                self.assertEqual(checks["status"], "PASS")

    def test_centers_independent_column_vector_calculation(self):
        counts = []
        for row, _, raw in self.data:
            t = self.official.x1_to_x2(raw["uav"]["lidar_pose"], raw["vehicle"]["lidar_pose"])
            ann = b.compare_annotations(raw, t); counts.append(ann["common_count"])
            for c in ann["comparisons"]:
                u = c["uav_raw"]; v = c["vehicle_raw"]
                uc = np.array(u["location"])+u["center"]
                expected = t@np.r_[uc, 1]
                np.testing.assert_allclose(c["uav_center_in_vehicle_code"], expected[:3], rtol=0, atol=1e-12)
                self.assertAlmostEqual(c["distance_3d_code_units"], np.linalg.norm(expected[:3]-(np.array(v["location"])+v["center"])))
        self.assertEqual(counts, [2, 2, 2, 1, 1])

    def test_source_contract_and_runtime_boundaries(self):
        contract = b.source_contract()
        self.assertFalse(contract["B_official_runtime"]["separate_fixed_axis_matrix_in_audited_path"])
        self.assertEqual(contract["A_paper_contract"]["zotero_attachment"], "JHB4X5KT")
        self.assertFalse(contract["repository_inspection"]["tree_truncated"])


if __name__ == "__main__":
    unittest.main()
