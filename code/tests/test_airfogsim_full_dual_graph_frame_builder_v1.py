from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_full_dual_graph_frame_builder_v1 import (  # noqa: E402
    build_carrying_hop_id,
    build_frame_decision,
    build_logical_flow_id,
)
from pi_jwm.airfogsim_full_dual_graph_observer_v1 import AirFogSimSnapshot  # noqa: E402
from pi_jwm.full_dual_graph_collector_contract_v1 import (  # noqa: E402
    DagEdge,
    PhysicalEdge,
    PhysicalNode,
    SnapshotPhase,
    TaskLifecycle,
    TaskSnapshot,
)
from pi_jwm.full_dual_graph_vocabulary_v1 import (  # noqa: E402
    FullTrajectoryVocabulary,
    RouteRevisionLedger,
)


def task(
    task_id: str,
    lifecycle: TaskLifecycle,
    *,
    source: str = "uav0",
    current: str | None = None,
    route=(),
    destination: str | None = "uav0",
    arrival: float = 0.0,
):
    return TaskSnapshot(
        task_id=task_id,
        task_node_id=source,
        lifecycle=lifecycle,
        current_node_id=current or source,
        route_nodes=tuple(route),
        return_destination_id=destination,
        arrival_time=arrival,
    )


def snapshot(tasks, *, include_cloud=True):
    nodes = [
        PhysicalNode("uav0", "U", True, (0.0, 0.0, 0.0)),
        PhysicalNode("uav1", "U", True, (10.0, 0.0, 0.0)),
        PhysicalNode("rsu0", "I", True, (20.0, 0.0, 0.0)),
    ]
    edges = [
        PhysicalEdge("physical::uav0::uav1::U2U", "uav0", "uav1", "U2U", True),
        PhysicalEdge("physical::uav0::rsu0::U2I", "uav0", "rsu0", "U2I", True),
        PhysicalEdge("physical::uav1::uav0::U2U", "uav1", "uav0", "U2U", True),
        PhysicalEdge("physical::rsu0::uav0::I2U", "rsu0", "uav0", "I2U", True),
    ]
    if include_cloud:
        nodes.append(PhysicalNode("cloud0", "C", True, (30.0, 0.0, 0.0)))
        edges.extend(
            [
                PhysicalEdge("physical::rsu0::cloud0::wired", "rsu0", "cloud0", "wired", True),
                PhysicalEdge("physical::cloud0::rsu0::wired", "cloud0", "rsu0", "wired", True),
            ]
        )
    return AirFogSimSnapshot(
        phase=SnapshotPhase.DECISION,
        simulation_time=1.0,
        nodes=tuple(nodes),
        physical_edges=tuple(edges),
        tasks=tuple(tasks),
        dag_edges=(DagEdge("dag::uav0::parent::child", "parent", "child"),),
        channel_rows=(),
    )


NODE_CPU = {"uav0": 1.0, "uav1": 2.0, "rsu0": 5.0, "cloud0": 10.0}
DISTANCE = {
    ("uav0", "uav1"): 10.0,
    ("uav0", "rsu0"): 20.0,
    ("uav1", "uav0"): 10.0,
    ("rsu0", "uav0"): 20.0,
    ("rsu0", "cloud0"): 10.0,
    ("cloud0", "rsu0"): 10.0,
}


def build(snap, *, frame_index=0, n_rb=4, trajectory="traj0"):
    return build_frame_decision(
        snap,
        trajectory_id=trajectory,
        frame_index=frame_index,
        seed=7,
        n_rb=n_rb,
        vocabulary=FullTrajectoryVocabulary(),
        route_revisions=RouteRevisionLedger(),
        node_cpu=NODE_CPU,
        node_distance=DISTANCE,
    )


class IdentityTests(unittest.TestCase):
    def test_flow_and_hop_ids_are_phase_and_route_specific(self):
        flow_id = build_logical_flow_id("traj0", "task0", "offload", 0)
        self.assertEqual("flow::traj0::task0::offload::0", flow_id)
        self.assertEqual(
            "hop::flow::traj0::task0::offload::0::0::uav0::rsu0",
            build_carrying_hop_id(flow_id, 0, "uav0", "rsu0"),
        )
        self.assertNotEqual(
            flow_id,
            build_logical_flow_id("traj0", "task0", "return", 0),
        )

    def test_route_change_increments_revision(self):
        vocabulary = FullTrajectoryVocabulary()
        revisions = RouteRevisionLedger()
        kwargs = dict(
            trajectory_id="traj0",
            seed=7,
            n_rb=4,
            vocabulary=vocabulary,
            route_revisions=revisions,
            node_cpu=NODE_CPU,
            node_distance=DISTANCE,
        )
        first = build_frame_decision(
            snapshot([task("task0", TaskLifecycle.OFFLOADING, route=("uav1",))]),
            frame_index=0,
            **kwargs,
        )
        second = build_frame_decision(
            snapshot([task("task0", TaskLifecycle.OFFLOADING, route=("rsu0",))]),
            frame_index=1,
            **kwargs,
        )
        self.assertEqual(0, first.action.flows[0].route_revision)
        self.assertEqual(1, second.action.flows[0].route_revision)


class LifecycleDecisionTests(unittest.TestCase):
    def test_waiting_offload_uses_explicit_balanced_family_rotation(self):
        local = build(snapshot([task("task0", TaskLifecycle.WAITING_TO_OFFLOAD)]), frame_index=0)
        nearest = build(snapshot([task("task0", TaskLifecycle.WAITING_TO_OFFLOAD)]), frame_index=1)
        capacity = build(snapshot([task("task0", TaskLifecycle.WAITING_TO_OFFLOAD)]), frame_index=2)

        self.assertEqual("uav0", local.action.decisions[0].target_node_id)
        self.assertEqual("uav1", nearest.action.decisions[0].target_node_id)
        self.assertEqual("rsu0", capacity.action.decisions[0].target_node_id)
        self.assertEqual(
            ["local", "nearest_remote", "capacity_remote"],
            [
                local.action.decisions[0].requested_target_family,
                nearest.action.decisions[0].requested_target_family,
                capacity.action.decisions[0].requested_target_family,
            ],
        )

    def test_waiting_offload_with_unfinished_or_failed_parent_is_not_sent_to_setter(self):
        unfinished = build(
            snapshot(
                [
                    task("parent", TaskLifecycle.TO_GENERATE),
                    task("child", TaskLifecycle.WAITING_TO_OFFLOAD),
                ]
            ),
            frame_index=1,
        )
        failed = build(
            snapshot(
                [
                    task("parent", TaskLifecycle.FAILED),
                    task("child", TaskLifecycle.WAITING_TO_OFFLOAD),
                ]
            ),
            frame_index=1,
        )

        unfinished_child = next(
            row for row in unfinished.action.decisions if row.task_id == "child"
        )
        failed_child = next(
            row for row in failed.action.decisions if row.task_id == "child"
        )
        self.assertFalse(unfinished_child.selected)
        self.assertEqual("dependency_not_satisfied", unfinished_child.reason)
        self.assertFalse(failed_child.selected)
        self.assertEqual("dependency_failed", failed_child.reason)
        self.assertFalse(
            next(
                row for row in unfinished.lifecycle_rows if row["task_id"] == "child"
            )["requires_route_setter"]
        )

    def test_offloading_and_returning_preserve_current_route_without_new_route_action(self):
        built = build(
            snapshot(
                [
                    task("off", TaskLifecycle.OFFLOADING, route=("uav1",)),
                    task(
                        "ret",
                        TaskLifecycle.RETURNING,
                        source="uav0",
                        current="rsu0",
                        route=("uav0",),
                        destination="uav0",
                    ),
                ]
            )
        )
        decisions = {row.task_id: row for row in built.action.decisions}
        lifecycle = {row["task_id"]: row for row in built.lifecycle_rows}
        self.assertEqual(("uav1",), decisions["off"].route_nodes)
        self.assertEqual(("uav0",), decisions["ret"].route_nodes)
        self.assertFalse(lifecycle["off"]["requires_route_setter"])
        self.assertFalse(lifecycle["ret"]["requires_route_setter"])

    def test_waiting_return_ends_at_frozen_destination_or_is_explicitly_unselected(self):
        selected = build(
            snapshot(
                [
                    task(
                        "ret",
                        TaskLifecycle.WAITING_TO_RETURN,
                        current="rsu0",
                        destination="uav0",
                    )
                ]
            )
        )
        absent = build(
            snapshot(
                [
                    task(
                        "ret",
                        TaskLifecycle.WAITING_TO_RETURN,
                        current="rsu0",
                        destination="missing",
                    )
                ]
            )
        )
        self.assertEqual(("uav0",), selected.action.decisions[0].route_nodes)
        self.assertEqual("return_destination_absent", absent.action.decisions[0].reason)
        self.assertFalse(absent.action.decisions[0].selected)

    def test_nonactionable_tasks_have_lifecycle_rows_but_no_decisions(self):
        lifecycles = (
            TaskLifecycle.COMPUTING,
            TaskLifecycle.DONE,
            TaskLifecycle.FAILED,
            TaskLifecycle.TO_GENERATE,
        )
        built = build(snapshot([task(state.value, state) for state in lifecycles]))
        self.assertEqual((), built.action.decisions)
        self.assertEqual(set(lifecycles), {row["lifecycle"] for row in built.lifecycle_rows})

    def test_rb_shortage_is_explicit_and_keeps_unserved_task(self):
        built = build(
            snapshot(
                [
                    task("a", TaskLifecycle.OFFLOADING, route=("uav1",), arrival=0.0),
                    task("b", TaskLifecycle.OFFLOADING, route=("rsu0",), arrival=1.0),
                ]
            ),
            n_rb=1,
        )
        decisions = {row.task_id: row for row in built.action.decisions}
        self.assertTrue(decisions["a"].selected)
        self.assertFalse(decisions["b"].selected)
        self.assertEqual("rb_budget_exhausted", decisions["b"].reason)
        self.assertEqual({"a", "b"}, {row["task_id"] for row in built.lifecycle_rows})

    def test_local_and_wired_actions_emit_no_rb(self):
        local = build(snapshot([task("local", TaskLifecycle.WAITING_TO_OFFLOAD)]), frame_index=0)
        wired = build(
            snapshot(
                [
                    task(
                        "wired",
                        TaskLifecycle.OFFLOADING,
                        source="rsu0",
                        current="rsu0",
                        route=("cloud0",),
                    )
                ]
            )
        )
        self.assertEqual((), local.action.rb_allocations)
        self.assertEqual((), local.action.flows)
        self.assertEqual((), wired.action.rb_allocations)
        self.assertEqual("wired", wired.action.hops[0].transport)


if __name__ == "__main__":
    unittest.main()
