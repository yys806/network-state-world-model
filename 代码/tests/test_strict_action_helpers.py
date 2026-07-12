import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class StrictActionHelperTest(unittest.TestCase):
    def test_cpu_allocation_uses_each_assigned_nodes_capacity(self):
        from strict_action_helpers import allocate_cpu_by_assigned_node

        tasks = [
            {"task_id": "a1", "assigned_to": "node-a"},
            {"task_id": "a2", "assigned_to": "node-a"},
            {"task_id": "b1", "assigned_to": "node-b"},
            {"task_id": "b2", "assigned_to": "node-b"},
        ]
        node_info = {
            "node-a": {"fog_profile": {"cpu": 10.0}},
            "node-b": {"fog_profile": {"cpu": 30.0}},
        }

        allocations, accepted = allocate_cpu_by_assigned_node(tasks, node_info.get)

        self.assertEqual(allocations, {"a1": 5.0, "a2": 5.0, "b1": 15.0, "b2": 15.0})
        self.assertEqual([row["task_id"] for row in accepted], ["a1", "a2", "b1", "b2"])

    def test_cpu_allocation_limits_each_node_independently(self):
        from strict_action_helpers import allocate_cpu_by_assigned_node

        tasks = [
            *({"task_id": f"a{i}", "assigned_to": "node-a"} for i in range(4)),
            {"task_id": "b0", "assigned_to": "node-b"},
        ]
        node_info = {
            "node-a": {"fog_profile": {"cpu": 12.0}},
            "node-b": {"fog_profile": {"cpu": 20.0}},
        }

        allocations, accepted = allocate_cpu_by_assigned_node(tasks, node_info.get, max_tasks_per_node=3)

        self.assertEqual(allocations, {"a0": 4.0, "a1": 4.0, "a2": 4.0, "b0": 20.0})
        self.assertEqual(len(accepted), 4)


if __name__ == "__main__":
    unittest.main()
