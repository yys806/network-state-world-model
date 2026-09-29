"""Planner-v1 projection labels historical rows without changing TRAIN data."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
from run_step6_3bc_train_self_replay_v1 import project_comm, comm_projection_reason, semantic_replay_reason
from pi_jwm.step6_3b_candidate_support_v1 import TrainStructuralSupportCatalog


class ProjectionTests(unittest.TestCase):
    def test_route_created_row_excluded_and_raw_unchanged(self):
        raw = {"route": {"entries": [{"task_id": "pending"}]},
               "comm": {"entries": [
                   {"task_id": "pending", "rb_indices": [0]},
                   {"task_id": "active", "rb_indices": [1]}]}}
        original = copy.deepcopy(raw)
        projected, excluded = project_comm(raw, {"active": 7})
        self.assertEqual(projected, [raw["comm"]["entries"][1]])
        self.assertEqual(excluded[0]["reason"],
                         "ROUTE_CREATED_FLOW_NOT_AVAILABLE_IN_CURRENT_STATE")
        self.assertEqual(raw, original)

    def test_unbound_row_without_route_has_distinct_residual(self):
        raw = {"route": {"entries": []},
               "comm": {"entries": [{"task_id": "unknown", "rb_indices": [0]}]}}
        projected, excluded = project_comm(raw, {})
        self.assertEqual(projected, [])
        self.assertEqual(excluded[0]["reason"],
                         "NO_UNIQUE_CURRENT_WIRELESS_FLOW_WITHOUT_ROUTE_ROW")

    def test_projected_noop_and_selected_count_use_train_catalog(self):
        path = ROOT / "code/artifacts/protocols/pi_jwm_step6_3b_candidate_grammar_v1_20260929/02_train_structural_support_catalog.json"
        catalog = TrainStructuralSupportCatalog.from_json(path)
        self.assertIsNone(comm_projection_reason([], {"active": 0}, catalog, 50))
        self.assertIsNone(comm_projection_reason(
            [{"task_id": "active", "rb_indices": [1]}],
            {"active": 0, "other": 1}, catalog, 50))
        self.assertEqual(comm_projection_reason(
            [{"task_id": "active", "rb_indices": [1]},
             {"task_id": "other", "rb_indices": [2]}],
            {"active": 0, "other": 1}, catalog, 50), None)

    def test_route_projection_can_leave_unseen_comm_structure(self):
        path = ROOT / "code/artifacts/protocols/pi_jwm_step6_3b_candidate_grammar_v1_20260929/02_train_structural_support_catalog.json"
        catalog = TrainStructuralSupportCatalog.from_json(path)
        rows = [{"task_id": "a", "rb_indices": [48]},
                {"task_id": "a", "rb_indices": [46, 47]},
                {"task_id": "b", "rb_indices": [0, 49]}]
        self.assertEqual(comm_projection_reason(rows, {"a": 0, "b": 1}, catalog, 50),
                         "COMM_STRUCTURAL_SIGNATURE_UNSEEN_AFTER_PROJECTION")

    def test_semantic_replay_compares_projected_rows_not_just_structure(self):
        projected = {"route": [],
                     "comm": [{"task_id": "a", "task_index": 0, "rb_indices": [0, 49]}],
                     "comp": [{"task_id": "b", "node_id": "RSU", "allocated_cpu_per_s": 2.0}],
                     "mobility": [{"uav_index": 1, "azimuth_rad": 0.1,
                                   "elevation_rad": 0.0, "speed_mps": 8.0}]}
        bound = SimpleNamespace(action=SimpleNamespace(
            route=(), comm=({"task_id": "a", "task_index": 0, "rb_indices": [0, 49]},),
            comp=({"task_id": "b", "node_id": "RSU", "allocated_cpu_per_s": 2.0},),
            mob=({"uav_index": 1, "azimuth_rad": 0.1,
                  "elevation_rad": 0.0, "speed_mps": 8.0},)))
        self.assertIsNone(semantic_replay_reason(projected, bound))
        changed = copy.deepcopy(projected)
        changed["comp"][0]["allocated_cpu_per_s"] = 2.1
        self.assertEqual(semantic_replay_reason(changed, bound), "PROJECTED_COMP_AMOUNT_DIFFERS")


if __name__ == "__main__":
    unittest.main()
