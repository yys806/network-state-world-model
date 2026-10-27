"""R2-06 contract tests: no detection training and no old geometry reruns."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
import audit_v2u4real_r2_06_modality_v1 as audit


class AuditContractTests(unittest.TestCase):
    def test_reject_other_split(self):
        for split in ("validation", "test", "locked_test"):
            with self.assertRaises(ValueError):
                audit.checked_train_member(f"{split}/{audit.SCENE}/1/ouster/a.pcd", "1")

    def test_reject_traversal_and_other_scene_agent(self):
        for p in (f"train/{audit.SCENE}/1/../2/a.jpg", "train/other/1/a.jpg",
                  f"train/{audit.SCENE}/2/a.jpg", "C:/bad.jpg"):
            with self.assertRaises(ValueError):
                audit.checked_train_member(p, "1")

    def test_valid_train_member(self):
        self.assertEqual(audit.checked_train_member(f"train/{audit.SCENE}/1/ouster/a.pcd", "1").suffix, ".pcd")

    def test_camera_paths_and_unsupported_agent(self):
        self.assertEqual(len(audit.camera_paths("1", "1752814590.2")), 3)
        self.assertEqual(len(audit.camera_paths("2", "1752814590.2")), 1)
        with self.assertRaises(ValueError):
            audit.camera_paths("3", "x")

    def test_no_rgb_never_decode_pass(self):
        gate = audit.geometry_gate(0, False, False)
        self.assertEqual(gate["status"], "BLOCKED")
        self.assertEqual(len(gate["blocking_reasons"]), 3)
        self.assertFalse(gate["projection_executed"])

    def test_rgb_presence_does_not_establish_sync(self):
        self.assertEqual(audit.geometry_gate(12, True, False)["status"], "BLOCKED")

    def test_rgb_presence_does_not_establish_direction(self):
        self.assertEqual(audit.geometry_gate(12, False, True)["status"], "BLOCKED")

    def test_gate_is_not_projection_execution(self):
        self.assertFalse(audit.geometry_gate(12, True, True)["projection_executed"])

    def test_head_identity_records_line_endings(self):
        self.assertEqual(audit.head_identity(b'a\r\n', b'a\n'), "CRLF_LF_ONLY")
        self.assertEqual(audit.head_identity(b'a\n', b'a\n'), "EXACT_BYTES")
        with self.assertRaises(ValueError):
            audit.head_identity(b'{"a":1}\n', b'{"a":2}\n')

    def test_matrix_shape_and_finiteness(self):
        c = {"intrinsic": {"values": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
             "extrinsic": {"values": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}}
        d = audit.matrix_diagnostics(c)
        self.assertEqual(d["direction_status"], "BLOCKED")
        self.assertIsNone(d["image_resolution"])
        bad = copy.deepcopy(c)
        bad["intrinsic"]["values"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            audit.matrix_diagnostics(bad)
        bad["intrinsic"]["values"] = [[1]]
        with self.assertRaises(ValueError):
            audit.matrix_diagnostics(bad)

    def test_inventory_rejects_duplicate_names(self):
        m = {"name": f"train/{audit.SCENE}/1/ouster/a.pcd", "type": "file"}
        with self.assertRaises(ValueError):
            audit.inventory_summary({"members": [m, m], "member_count": 2}, "1")

    def test_seven_uses_and_no_full_system_claim(self):
        rows = audit.uses()
        self.assertEqual([r["id"] for r in rows], list(range(1, 8)))
        self.assertEqual([r["status"] for r in rows],
                         ["PARTIAL", "PARTIAL", "PARTIAL", "BLOCKED", "BLOCKED", "PARTIAL", "NOT_APPLICABLE"])


class RealReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((ROOT / audit.OUT / "result.json").read_text(encoding="utf-8"))

    def test_full_archive_and_local_no_rgb(self):
        r = self.result
        self.assertEqual([a["member_count"] for a in r["archive_inventory"]], [1314, 438])
        self.assertEqual(r["rgb_inventory"]["actual_rgb_count"], 0)
        self.assertEqual(r["rgb_inventory"]["expected_reference_count"], 64)
        self.assertEqual(r["rgb_inventory"]["decode_status"], "BLOCKED_NO_SAMPLE")

    def test_metadata_counts_not_physical_pass(self):
        r = self.result
        self.assertEqual(r["camera_calibration"]["instance_count"], 64)
        self.assertEqual(len(r["camera_calibration"]["yaml_anchor_checks"]), 2)
        self.assertEqual(r["geometry_diagnostic"]["status"], "BLOCKED")
        self.assertEqual(r["time_association"]["status"], "BLOCKED")
        self.assertIsNone(r["time_association"]["timestamp_unit"])

    def test_old_findings_reused_not_cleared(self):
        r = self.result["inherited_lidar"]
        self.assertEqual(r["r2_04b_software_transform"], "PASS")
        self.assertEqual(r["physical_alignment"], "NOT_ESTABLISHED")
        self.assertEqual(len(r["duplicate_pcd_groups"]), 2)
        self.assertEqual(r["pcd_decoded_this_step"], 0)
        self.assertFalse(r["old_five_frame_geometry_reexecuted"])

    def test_boundaries_and_raw_identity(self):
        self.assertEqual(self.result["split"], "train")
        self.assertTrue(all(value is False for value in self.result["boundaries"].values()))
        self.assertEqual(len(self.result["raw_input_integrity"]["files"]), 64)
        for row in self.result["raw_input_integrity"]["files"]:
            self.assertEqual(audit.sha256(ROOT / row["path"]), row["sha256"])

    def test_refuse_overwriting_receipt(self):
        with self.assertRaises(FileExistsError):
            audit.run(ROOT / audit.OUT)


if __name__ == "__main__":
    unittest.main()
