from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.full_dual_graph_coverage_v1 import (  # noqa: E402
    WirelessFlowRequest,
    allocate_rb_coverage,
    choose_resource_arm,
    choose_target_family,
    has_same_transmitter_rb_conflict,
    overlapping_rbs,
    target_family_for_ordinal,
)


class CoveragePolicyTests(unittest.TestCase):
    def test_target_family_is_stable_and_sorting_is_explicit(self):
        candidates = (
            {"node_id": "rsu2", "is_local": False, "distance": 20.0, "available_cpu": 9.0},
            {"node_id": "uav1", "is_local": False, "distance": 10.0, "available_cpu": 4.0},
            {"node_id": "uav0", "is_local": True, "distance": 0.0, "available_cpu": 2.0},
        )
        first = choose_target_family(
            trajectory_id="traj0", task_id="task0", route_revision=0, candidates=candidates
        )
        second = choose_target_family(
            trajectory_id="traj0", task_id="task0", route_revision=0, candidates=tuple(reversed(candidates))
        )
        self.assertEqual(first, second)
        self.assertIn(first.executed_family, {"local", "nearest_remote", "capacity_remote"})

    def test_stable_ordinal_rotation_covers_three_requested_families(self):
        candidates = (
            {"node_id": "uav0", "is_local": True, "distance": 0.0, "available_cpu": 2.0},
            {"node_id": "rsu0", "is_local": False, "distance": 10.0, "available_cpu": 4.0},
        )
        choices = [
            choose_target_family(
                trajectory_id="traj0",
                task_id=f"task{i}",
                route_revision=0,
                candidates=candidates,
                requested_family=target_family_for_ordinal(i),
            )
            for i in range(3)
        ]
        self.assertEqual({choice.requested_family for choice in choices}, {"local", "nearest_remote", "capacity_remote"})

    def test_nearest_and_capacity_tie_breakers_are_stable(self):
        candidates = (
            {"node_id": "z", "is_local": False, "distance": 10.0, "available_cpu": 4.0},
            {"node_id": "a", "is_local": False, "distance": 10.0, "available_cpu": 4.0},
            {"node_id": "b", "is_local": False, "distance": 30.0, "available_cpu": 9.0},
        )
        nearest = choose_target_family(
            trajectory_id="traj0", task_id="task1", route_revision=0,
            candidates=candidates, requested_family="nearest_remote"
        )
        capacity = choose_target_family(
            trajectory_id="traj0", task_id="task1", route_revision=0,
            candidates=candidates, requested_family="capacity_remote"
        )
        self.assertEqual("a", nearest.target_node_id)
        self.assertEqual("b", capacity.target_node_id)

    def test_empty_requested_family_uses_fixed_fallback(self):
        candidates = (
            {"node_id": "uav0", "is_local": True, "distance": 0.0, "available_cpu": 2.0},
        )
        choice = choose_target_family(
            trajectory_id="traj0", task_id="task0", route_revision=0,
            candidates=candidates, requested_family="nearest_remote"
        )
        self.assertTrue(choice.fallback)
        self.assertEqual("local", choice.executed_family)
        self.assertEqual("uav0", choice.target_node_id)

    def test_cross_process_target_choice_is_identical(self):
        script = (
            "import sys; sys.path.insert(0, r'"
            + str(SRC_ROOT)
            + "'); from pi_jwm.full_dual_graph_coverage_v1 import choose_target_family; "
            "print(choose_target_family(trajectory_id='traj0', task_id='task0', "
            "route_revision=0, candidates=[{'node_id':'uav0','is_local':True,'distance':0.0,'available_cpu':2.0}, "
            "{'node_id':'rsu0','is_local':False,'distance':10.0,'available_cpu':4.0}]))"
        )
        first = subprocess.check_output([sys.executable, "-c", script], text=True)
        second = subprocess.check_output([sys.executable, "-c", script], text=True)
        self.assertEqual(first, second)


class ResourceCoverageTests(unittest.TestCase):
    def setUp(self):
        self.requests = (
            WirelessFlowRequest("flow0", "hop0", "uav0", "rsu0"),
            WirelessFlowRequest("flow1", "hop1", "uav1", "rsu0"),
            WirelessFlowRequest("flow2", "hop2", "uav0", "rsu1"),
        )

    def test_orthogonal_arm_has_no_rb_overlap(self):
        decisions = allocate_rb_coverage(self.requests[:2], n_rb=6, arm="orthogonal")
        self.assertEqual((), overlapping_rbs(decisions))
        self.assertFalse(has_same_transmitter_rb_conflict(decisions))
        self.assertTrue(all(row.selected and row.rb_indices for row in decisions))

    def test_reuse_arm_covers_cross_transmitter_overlap(self):
        decisions = allocate_rb_coverage(self.requests[:2], n_rb=2, arm="interference_reuse")
        self.assertTrue(set(decisions[0].rb_indices) & set(decisions[1].rb_indices))
        self.assertFalse(has_same_transmitter_rb_conflict(decisions))

    def test_same_transmitter_never_reuses_an_rb(self):
        decisions = allocate_rb_coverage(self.requests, n_rb=2, arm="interference_reuse")
        self.assertFalse(has_same_transmitter_rb_conflict(decisions))

    def test_shortage_is_explicit_and_selected_wireless_has_rb(self):
        requests = self.requests[:2]
        decisions = allocate_rb_coverage(requests, n_rb=1, arm="orthogonal")
        self.assertEqual({"flow0", "flow1"}, {row.flow_id for row in decisions})
        unselected = [row for row in decisions if not row.selected]
        self.assertEqual(1, len(unselected))
        self.assertEqual("rb_budget_exhausted", unselected[0].reason)
        self.assertTrue(all(row.rb_indices for row in decisions if row.selected))

    def test_invalid_arm_and_capacity_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "arm"):
            allocate_rb_coverage(self.requests, n_rb=2, arm="unknown")
        with self.assertRaisesRegex(ValueError, "n_rb"):
            allocate_rb_coverage(self.requests, n_rb=0, arm="orthogonal")

    def test_resource_arm_is_deterministic(self):
        self.assertEqual(choose_resource_arm("traj0", 0), choose_resource_arm("traj0", 0))
        self.assertIn(choose_resource_arm("traj0", 0), {"orthogonal", "interference_reuse"})


if __name__ == "__main__":
    unittest.main()
