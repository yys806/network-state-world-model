from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.full_dual_graph_collector_contract_v1 import (  # noqa: E402
    CarryingHop,
    DagEdge,
    LogicalFlow,
    PhysicalEdge,
    PhysicalNode,
    TaskLifecycle,
    TaskSnapshot,
)
from pi_jwm.full_dual_graph_vocabulary_v1 import (  # noqa: E402
    FullTrajectoryVocabulary,
    RouteRevisionLedger,
)


def task(task_id: str, node_id: str = "uav0") -> TaskSnapshot:
    return TaskSnapshot(
        task_id,
        node_id,
        TaskLifecycle.WAITING_TO_OFFLOAD,
        node_id,
        (),
        node_id,
        0.0,
    )


class FullTrajectoryVocabularyTests(unittest.TestCase):
    def test_six_vocabularies_are_append_only_across_disappearance(self):
        vocabulary = FullTrajectoryVocabulary()
        edge = PhysicalEdge("pe0", "uav0", "rsu0", "U2I", True)
        flow = LogicalFlow("flow0", "traj0", "task0", "offload", 0)
        hop = CarryingHop("hop0", "flow0", 0, "uav0", "rsu0", "pe0", "wireless")
        first = vocabulary.observe(
            nodes=(PhysicalNode("uav0", "U", True), PhysicalNode("rsu0", "I", True)),
            physical_edges=(edge,),
            tasks=(task("task0"),),
            dag_edges=(DagEdge("dag0", "task0", "task1"),),
            flows=(flow,),
            hops=(hop,),
        )
        second = vocabulary.observe(
            nodes=(PhysicalNode("rsu0", "I", True),),
            physical_edges=(),
            tasks=(),
            dag_edges=(),
            flows=(),
            hops=(),
        )
        third = vocabulary.observe(
            nodes=(
                PhysicalNode("uav0", "U", True),
                PhysicalNode("rsu0", "I", True),
                PhysicalNode("rsu1", "I", True),
            ),
            physical_edges=(edge,),
            tasks=(task("task1", "rsu1"),),
            dag_edges=(DagEdge("dag1", "task1", "task0"),),
            flows=(LogicalFlow("flow1", "traj0", "task1", "offload", 0),),
            hops=(
                CarryingHop("hop1", "flow1", 0, "rsu1", "rsu0", "pe1", "wired"),
            ),
        )
        self.assertEqual(0, first.node_indices["rsu0"])
        self.assertEqual(1, first.node_indices["uav0"])
        self.assertFalse(second.node_presence[first.node_indices["uav0"]])
        self.assertEqual(first.node_indices["uav0"], third.node_indices["uav0"])
        self.assertEqual(2, third.node_indices["rsu1"])
        self.assertEqual(first.physical_edge_indices["pe0"], third.physical_edge_indices["pe0"])
        self.assertEqual(0, first.task_indices["task0"])
        self.assertEqual(1, third.task_indices["task1"])
        self.assertEqual(0, first.dag_edge_indices["dag0"])
        self.assertEqual(1, third.dag_edge_indices["dag1"])
        self.assertEqual(0, first.flow_indices["flow0"])
        self.assertEqual(1, third.flow_indices["flow1"])
        self.assertEqual(0, first.hop_indices["hop0"])
        self.assertEqual(1, third.hop_indices["hop1"])

    def test_observation_is_deterministic_and_rejects_duplicate_ids(self):
        vocabulary = FullTrajectoryVocabulary()
        snapshot = vocabulary.observe(
            nodes=(PhysicalNode("uav1", "U", True), PhysicalNode("uav0", "U", True)),
            physical_edges=(),
            tasks=(task("task1"), task("task0")),
            dag_edges=(),
            flows=(LogicalFlow("flow1", "traj0", "task1", "offload", 0), LogicalFlow("flow0", "traj0", "task0", "offload", 0)),
            hops=(),
        )
        self.assertEqual({"uav0": 0, "uav1": 1}, snapshot.node_indices)
        self.assertEqual({"task0": 0, "task1": 1}, snapshot.task_indices)
        self.assertEqual({"flow0": 0, "flow1": 1}, snapshot.flow_indices)
        with self.assertRaisesRegex(ValueError, "duplicate node"):
            vocabulary.observe(
                nodes=(PhysicalNode("uav0", "U", True), PhysicalNode("uav0", "U", True)),
                physical_edges=(), tasks=(), dag_edges=(), flows=(), hops=()
            )

    def test_binding_conflict_does_not_partially_mutate(self):
        vocabulary = FullTrajectoryVocabulary()
        edge = PhysicalEdge("pe0", "uav0", "rsu0", "U2I", True)
        vocabulary.observe(
            nodes=(PhysicalNode("uav0", "U", True), PhysicalNode("rsu0", "I", True)),
            physical_edges=(edge,), tasks=(), dag_edges=(), flows=(), hops=()
        )
        before = vocabulary.snapshot()
        with self.assertRaisesRegex(ValueError, "physical edge binding"):
            vocabulary.observe(
                nodes=(PhysicalNode("uav0", "U", True), PhysicalNode("rsu1", "I", True)),
                physical_edges=(PhysicalEdge("pe0", "uav0", "rsu1", "U2I", True),),
                tasks=(), dag_edges=(), flows=(), hops=()
            )
        self.assertEqual(before, vocabulary.snapshot())

    def test_cross_space_binding_conflict_is_rejected(self):
        vocabulary = FullTrajectoryVocabulary()
        with self.assertRaisesRegex(ValueError, "DAG edge"):
            vocabulary.observe(
                nodes=(PhysicalNode("uav0", "U", True),),
                physical_edges=(),
                tasks=(task("task0"),),
                dag_edges=(DagEdge("shared", "task0", "task0"),),
                flows=(LogicalFlow("shared", "traj0", "task0", "offload", 0),),
                hops=(),
            )


class RouteRevisionLedgerTests(unittest.TestCase):
    def test_unchanged_route_reuses_revision_and_changed_route_increments(self):
        ledger = RouteRevisionLedger()
        self.assertEqual(0, ledger.assign("traj0", "task0", "offload", ("rsu0",)))
        self.assertEqual(0, ledger.assign("traj0", "task0", "offload", ("rsu0",)))
        self.assertEqual(1, ledger.assign("traj0", "task0", "offload", ("uav1", "rsu0")))
        self.assertEqual(0, ledger.assign("traj0", "task0", "return", ("uav0",)))

    def test_invalid_route_and_phase_are_rejected_without_mutation(self):
        ledger = RouteRevisionLedger()
        with self.assertRaisesRegex(ValueError, "non-empty"):
            ledger.assign("traj0", "task0", "offload", ())
        with self.assertRaisesRegex(ValueError, "phase"):
            ledger.assign("traj0", "task0", "compute", ("uav0",))
        self.assertEqual({}, ledger.snapshot())

    def test_imported_revision_must_be_contiguous(self):
        ledger = RouteRevisionLedger()
        with self.assertRaisesRegex(ValueError, "revision"):
            ledger.import_revision("traj0", "task0", "offload", 2, ("rsu0",))
        self.assertEqual({}, ledger.snapshot())


if __name__ == "__main__":
    unittest.main()
