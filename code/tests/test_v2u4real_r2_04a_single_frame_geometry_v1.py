"""Artificial assertions and real-sample checks remain distinct evidence classes."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import audit_v2u4real_r2_04a_single_frame_geometry_v1 as audit
from pi_jwm.v2u4real_interface_v1 import InterfaceBlocked, UniqueKeyLoader
import yaml


class SyntheticGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official, cls.identity = audit.official_functions()

    def test_known_rotation_translation_and_direction(self):
        source = [10, 20, 30, 0, 90, 0]
        target = [2, 3, 4, 0, 0, 0]
        t = self.official.x1_to_x2(source, target)
        expected = np.array([[0, -1, 0, 8], [1, 0, 0, 17], [0, 0, 1, 26], [0, 0, 0, 1]])
        np.testing.assert_allclose(t, expected, atol=1e-12, rtol=0)
        point = audit.transform_points([[1, 0, 0]], t)
        np.testing.assert_allclose(point, [[8, 18, 26]], atol=1e-12, rtol=0)

    def test_inverse_and_distances_nonplanar_pose(self):
        t = self.official.x1_to_x2([11, -7, 29, 13, 72, -34], [-8, 4, 1, -22, 49, 18])
        p = np.random.default_rng(204).normal(size=(257, 3)) * 100
        _, checks = audit.geometry_checks(p, t)
        self.assertEqual(checks["status"], "PASS")
        self.assertEqual(checks["distance_pair_count"], 8128)
        self.assertLessEqual(checks["roundtrip_max_abs"], 1e-8)
        self.assertLessEqual(checks["distance_max_abs_error"], 1e-8)

    def test_nonfinite_points_refused(self):
        for invalid in [np.nan, np.inf, -np.inf]:
            with self.subTest(invalid=invalid), self.assertRaises(InterfaceBlocked):
                audit.geometry_checks([[0, 1, invalid]], np.eye(4))

    def test_invalid_matrix_refused(self):
        reflection = np.diag([-1., 1., 1., 1.])
        scale = np.diag([2., 1., 1., 1.])
        bottom = np.eye(4); bottom[3, 0] = 1
        nonfinite = np.eye(4); nonfinite[0, 0] = np.nan
        for bad in [np.eye(3), reflection, scale, bottom, nonfinite]:
            with self.subTest(matrix=bad), self.assertRaises(InterfaceBlocked):
                audit.geometry_checks([[1, 2, 3]], bad)

    def test_bad_pose_refused_without_coercing_list(self):
        for bad in [(1, 2, 3, 4, 5, 6), [1]*5, [1, 2, 3, 4, 5, np.inf], [True]*6, ["0"]*6]:
            with self.subTest(pose=bad), self.assertRaises(InterfaceBlocked):
                audit.strict_vector(bad, 6, "pose")

    def test_deterministic_subset(self):
        np.testing.assert_array_equal(audit.selected_indices(10, 4), [0, 3, 6, 9])
        np.testing.assert_array_equal(audit.selected_indices(3, 128), [0, 1, 2])

    def test_output_cannot_write_into_raw_inputs(self):
        with self.assertRaises(InterfaceBlocked):
            audit.run(output=audit.ROOT/"code/artifacts/data/v2u4real_r2_01/selected_samples")

    def test_machine_evidence_uses_canonical_lf_for_git_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"result.json"
            audit.write_json(path, {"line": "one\ntwo"})
            self.assertNotIn(b"\r", path.read_bytes())
            self.assertEqual(json.loads(path.read_text())["line"], "one\ntwo")

    def test_official_box_primitives_known_extent(self):
        corners = self.official.create_bbx([4, 2, 6])
        np.testing.assert_array_equal(corners.max(0)-corners.min(0), [4, 2, 6])
        center = self.official.corner_to_center(corners[None], order="lwh")[0]
        np.testing.assert_allclose(center, [0, 0, 0, 4, 2, 6, 0], atol=1e-12, rtol=0)

    def test_exact_converter_body_artificial_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"sample.yaml"
            path.write_text(yaml.safe_dump({"vehicles": {"x": {"location": [1, 2, 3], "center": [4, 5, 6],
                            "extent": [4, 2, 6], "angle": [0, 90, 0], "obj_type": "Car"}}}), encoding="utf8")
            t = np.eye(4); t[:3, 3] = [8, 9, 10]
            result = self.official.load_annotation_objects(str(path), transform=t)["x"]
            np.testing.assert_allclose(result["loc"], [13, 16, 19], atol=1e-12, rtol=0)
            np.testing.assert_allclose(result["dims"], [4, 2, 6], atol=1e-12, rtol=0)
            self.assertAlmostEqual(result["heading"], np.pi/2)

    def test_duplicate_yaml_key_refused(self):
        with self.assertRaises(ValueError):
            yaml.load("lidar_pose: [0]\nlidar_pose: [1]\n", Loader=UniqueKeyLoader)

    def test_source_change_stops_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(audit.ROOT/"code/reference/V2U4Real", root/"code/reference/V2U4Real")
            shutil.copy(audit.ROOT/"code/reference/V2U4Real.SOURCE.json", root/"code/reference/V2U4Real.SOURCE.json")
            path = root/"code/reference/V2U4Real/opencood/utils/transformation_utils.py"
            path.write_bytes(path.read_bytes()+b"\n# tampered fixture\n")
            with self.assertRaises(InterfaceBlocked):
                audit.official_functions(root)


class RealSampleTests(unittest.TestCase):
    """No skip: fixed local real files must be available for this task's acceptance."""
    @classmethod
    def setUpClass(cls):
        cls.inputs, cls.arrays, cls.raw, cls.evidence = audit.read_inputs()
        cls.official, _ = audit.official_functions()

    def test_fixed_real_sample_strict_read(self):
        self.assertEqual(len(self.arrays["vehicle"]), 141177)
        self.assertEqual(len(self.arrays["uav"]), 70532)
        self.assertFalse(self.evidence["fixed_vehicle_in_known_duplicate_groups"])
        self.assertEqual(self.inputs["vehicle"]["yaml"]["object_count"], 6)
        self.assertEqual(self.inputs["uav"]["yaml"]["object_count"], 2)

    def test_real_all_points_roundtrip_and_distance(self):
        t = self.official.x1_to_x2(self.raw["uav"]["lidar_pose"], self.raw["vehicle"]["lidar_pose"])
        _, result = audit.geometry_checks(self.arrays["uav"], t)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["before"]["count"], result["after"]["count"])

    def test_real_independent_homogeneous_application(self):
        tv = self.official.x_to_world(self.raw["vehicle"]["lidar_pose"])
        tu = self.official.x_to_world(self.raw["uav"]["lidar_pose"])
        t = self.official.x1_to_x2(self.raw["uav"]["lidar_pose"], self.raw["vehicle"]["lidar_pose"])
        p = self.arrays["uav"]
        homogeneous = np.column_stack([p, np.ones(len(p))])
        independent = np.linalg.solve(tv, tu @ homogeneous.T).T
        actual = audit.transform_points(p, t)
        np.testing.assert_allclose(actual, independent[:, :3], atol=1e-10, rtol=0)
        recovered = np.linalg.solve(t, independent.T).T
        np.testing.assert_allclose(recovered[:, :3], p, atol=1e-8, rtol=0)

    def test_real_annotation_exact_converter_and_raw_preserved(self):
        t = self.official.x1_to_x2(self.raw["uav"]["lidar_pose"], self.raw["vehicle"]["lidar_pose"])
        paths = {s: audit.ROOT / r["yaml"]["path"] for s, r in self.inputs.items()}
        original = json.dumps(self.raw, sort_keys=True)
        result, corners = audit.annotation_diagnostic(self.raw, paths, t, self.official)
        self.assertEqual(result["status"], "official_code_interpretation_only")
        self.assertEqual({s:len(v) for s,v in corners.items()}, {"vehicle": 6, "uav": 2})
        self.assertEqual(json.dumps(self.raw, sort_keys=True), original)

    def test_missing_fixed_sample_does_not_select_another(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root/audit.R2_03
            dest.parent.mkdir(parents=True)
            shutil.copy(audit.ROOT/audit.R2_03, dest)
            with self.assertRaises(InterfaceBlocked):
                audit.read_inputs(root)


if __name__ == "__main__":
    unittest.main()
