from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_airfogsim_collector_adapter_v2 import (  # noqa: E402
    CollectorBundleContractError,
    build_formal_bundles,
)


def _frame(*, trajectory_id: str = "traj-0") -> dict[str, object]:
    return {
        "trajectory_id": trajectory_id,
        "frame_index": 0,
        "decision_snapshot": {
            "simulation_time": 0.0,
            "nodes": [
                {
                    "node_id": "v0",
                    "node_type": "vehicle",
                    "present": True,
                    "position": [1.0, 2.0, 0.0],
                }
            ],
            "physical_edges": [],
            "tasks": [],
            "dag_edges": [],
            "channel_rows": [],
        },
        "execution_snapshot": {
            "simulation_time": 0.1,
            "nodes": [
                {
                    "node_id": "v0",
                    "node_type": "vehicle",
                    "present": True,
                    "position": [1.0, 2.0, 0.0],
                }
            ],
            "physical_edges": [],
            "tasks": [],
            "dag_edges": [],
            "channel_rows": [],
        },
        "outcome_snapshot": {
            "simulation_time": 0.1,
            "nodes": [
                {
                    "node_id": "v0",
                    "node_type": "vehicle",
                    "present": True,
                    "position": [1.0, 2.0, 0.0],
                }
            ],
            "physical_edges": [],
            "tasks": [],
            "dag_edges": [],
            "channel_rows": [],
        },
        "action": {
            "frame_index": 0,
            "decisions": [],
            "flows": [],
            "hops": [],
            "rb_allocations": [],
        },
        "lifecycle_rows": [],
        "transfer_rows": [],
        "cpu_rows": [],
        "energy_rows": [],
    }


class FormalCollectorAdapterTests(unittest.TestCase):
    def test_builds_source_and_resource_bundles_from_complete_frame_payload(self):
        source, resource = build_formal_bundles(
            [_frame()],
            task_records=[],
            n_rb=4,
        )

        self.assertEqual("traj-0", source["trajectory_id"])
        self.assertEqual("v0", source["physical_nodes"][0]["id"])
        self.assertEqual(3, len(source["physical_node_snapshots"]))
        self.assertIn("task_ledger", resource)
        self.assertIn("rb_ledger", resource)
        self.assertIn("cpu_ledger", resource)
        self.assertIn("uav_energy_ledger", resource)

    def test_missing_outcome_snapshot_blocks_formal_bundle(self):
        frame = _frame()
        frame["outcome_snapshot"] = None

        with self.assertRaises(CollectorBundleContractError):
            build_formal_bundles([frame], task_records=[], n_rb=4)


if __name__ == "__main__":
    unittest.main()
