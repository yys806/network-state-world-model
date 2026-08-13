from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.full_dual_graph_collector_contract_v1 import (  # noqa: E402
    CarryingHop,
    CollectorContractError,
    DagEdge,
    DecisionRow,
    JointFrameAction,
    LogicalFlow,
    PhysicalEdge,
    PhysicalNode,
    RbAllocation,
    SnapshotPhase,
    TaskLifecycle,
    TaskSnapshot,
    validate_joint_frame_action,
)


def wireless_fixture():
    nodes = (
        PhysicalNode("uav0", "U", True),
        PhysicalNode("rsu0", "I", True),
    )
    edges = (
        PhysicalEdge("pe::uav0::rsu0", "uav0", "rsu0", "U2I", True),
    )
    tasks = (
        TaskSnapshot(
            task_id="task0",
            task_node_id="uav0",
            lifecycle=TaskLifecycle.WAITING_TO_OFFLOAD,
            current_node_id="uav0",
            route_nodes=(),
            return_destination_id="uav0",
            arrival_time=0.0,
        ),
    )
    flow = LogicalFlow(
        "flow::traj0::task0::offload::0",
        "traj0",
        "task0",
        "offload",
        0,
    )
    hop = CarryingHop(
        "hop::flow0::0",
        flow.flow_id,
        0,
        "uav0",
        "rsu0",
        "pe::uav0::rsu0",
        "wireless",
    )
    decision = DecisionRow(
        task_id="task0",
        lifecycle=TaskLifecycle.WAITING_TO_OFFLOAD,
        selected=True,
        reason="selected",
        target_node_id="rsu0",
        route_nodes=("rsu0",),
        flow_id=flow.flow_id,
        hop_id=hop.hop_id,
        requested_target_family="nearest_remote",
        executed_target_family="nearest_remote",
        target_family_fallback=False,
    )
    action = JointFrameAction(
        frame_index=0,
        decisions=(decision,),
        flows=(flow,),
        hops=(hop,),
        rb_allocations=(RbAllocation(flow.flow_id, hop.hop_id, 0),),
    )
    return nodes, edges, tasks, action


def validate(nodes, edges, tasks, action, *, dag_edges=(), n_rb=4, sources=None):
    return validate_joint_frame_action(
        action,
        phase=SnapshotPhase.DECISION,
        nodes=nodes,
        physical_edges=edges,
        tasks=tasks,
        dag_edges=dag_edges,
        n_rb=n_rb,
        input_source_phases=sources,
    )


class ValidActionTests(unittest.TestCase):
    def test_valid_wireless_action_passes(self):
        nodes, edges, tasks, action = wireless_fixture()
        self.assertIs(action, validate(nodes, edges, tasks, action))

    def test_local_action_has_no_flow_hop_or_rb(self):
        nodes, _, tasks, _ = wireless_fixture()
        decision = DecisionRow(
            task_id="task0",
            lifecycle=TaskLifecycle.WAITING_TO_OFFLOAD,
            selected=True,
            reason="selected",
            target_node_id="uav0",
            route_nodes=("uav0",),
            flow_id=None,
            hop_id=None,
            requested_target_family="local",
            executed_target_family="local",
            target_family_fallback=False,
        )
        action = JointFrameAction(0, (decision,), (), (), ())
        self.assertIs(action, validate(nodes, (), tasks, action))

    def test_wired_hop_has_cep_but_no_rb(self):
        nodes = (
            PhysicalNode("rsu0", "I", True),
            PhysicalNode("cloud0", "C", True),
        )
        edges = (
            PhysicalEdge(
                "pe::rsu0::cloud0", "rsu0", "cloud0", "wired", True
            ),
        )
        tasks = (
            TaskSnapshot(
                "task0",
                "rsu0",
                TaskLifecycle.WAITING_TO_OFFLOAD,
                "rsu0",
                (),
                "rsu0",
                0.0,
            ),
        )
        flow = LogicalFlow(
            "flow::traj0::task0::offload::0", "traj0", "task0", "offload", 0
        )
        hop = CarryingHop(
            "hop::flow0::0",
            flow.flow_id,
            0,
            "rsu0",
            "cloud0",
            "pe::rsu0::cloud0",
            "wired",
        )
        decision = DecisionRow(
            "task0",
            TaskLifecycle.WAITING_TO_OFFLOAD,
            True,
            "selected",
            "cloud0",
            ("cloud0",),
            flow.flow_id,
            hop.hop_id,
            "capacity_remote",
            "capacity_remote",
            False,
        )
        action = JointFrameAction(0, (decision,), (flow,), (hop,), ())
        self.assertIs(action, validate(nodes, edges, tasks, action))

    def test_unselected_actionable_task_is_explicit(self):
        nodes, edges, tasks, _ = wireless_fixture()
        decision = DecisionRow(
            "task0",
            TaskLifecycle.WAITING_TO_OFFLOAD,
            False,
            "rb_budget_exhausted",
            None,
            (),
            None,
            None,
            "nearest_remote",
            None,
            False,
        )
        action = JointFrameAction(0, (decision,), (), (), ())
        self.assertIs(action, validate(nodes, edges, tasks, action))


class RejectionTests(unittest.TestCase):
    def assert_code(self, code, nodes, edges, tasks, action, **kwargs):
        with self.assertRaises(CollectorContractError) as caught:
            validate(nodes, edges, tasks, action, **kwargs)
        self.assertEqual(code, caught.exception.code)

    def test_rb_out_of_range(self):
        nodes, edges, tasks, action = wireless_fixture()
        bad = replace(action, rb_allocations=(replace(action.rb_allocations[0], rb_index=4),))
        self.assert_code("rb_out_of_range", nodes, edges, tasks, bad)

    def test_duplicate_rb_allocation(self):
        nodes, edges, tasks, action = wireless_fixture()
        bad = replace(action, rb_allocations=action.rb_allocations * 2)
        self.assert_code("duplicate_rb_allocation", nodes, edges, tasks, bad)

    def test_route_first_hop_mismatch(self):
        nodes, edges, tasks, action = wireless_fixture()
        bad_decision = replace(action.decisions[0], route_nodes=("uav0", "rsu0"))
        bad = replace(action, decisions=(bad_decision,))
        self.assert_code("route_first_hop_mismatch", nodes, edges, tasks, bad)

    def test_cep_endpoint_mismatch(self):
        nodes, _, tasks, action = wireless_fixture()
        nodes = nodes + (PhysicalNode("rsu1", "I", True),)
        edges = (PhysicalEdge("pe::uav0::rsu0", "uav0", "rsu1", "U2I", True),)
        self.assert_code("cep_endpoint_mismatch", nodes, edges, tasks, action)

    def test_absent_node_is_rejected(self):
        nodes, edges, tasks, action = wireless_fixture()
        nodes = (nodes[0], replace(nodes[1], present=False))
        self.assert_code("node_absent_at_decision", nodes, edges, tasks, action)

    def test_selected_nonactionable_task_is_rejected(self):
        nodes, edges, tasks, action = wireless_fixture()
        tasks = (replace(tasks[0], lifecycle=TaskLifecycle.COMPUTING),)
        decision = replace(action.decisions[0], lifecycle=TaskLifecycle.COMPUTING)
        self.assert_code(
            "offload_wrong_lifecycle",
            nodes,
            edges,
            tasks,
            replace(action, decisions=(decision,)),
        )

    def test_duplicate_task_decision_is_rejected(self):
        nodes, edges, tasks, action = wireless_fixture()
        bad = replace(action, decisions=action.decisions * 2)
        self.assert_code("duplicate_task_decision", nodes, edges, tasks, bad)

    def test_return_destination_mismatch_is_rejected(self):
        nodes, edges, tasks, action = wireless_fixture()
        tasks = (
            replace(
                tasks[0],
                lifecycle=TaskLifecycle.WAITING_TO_RETURN,
                current_node_id="uav0",
                return_destination_id="uav0",
            ),
        )
        flow = replace(action.flows[0], phase="return")
        decision = replace(
            action.decisions[0],
            lifecycle=TaskLifecycle.WAITING_TO_RETURN,
            flow_id=flow.flow_id,
        )
        bad = replace(action, decisions=(decision,), flows=(flow,))
        self.assert_code("return_destination_mismatch", nodes, edges, tasks, bad)

    def test_same_transmitter_rb_conflict_is_rejected(self):
        nodes, edges, tasks, action = wireless_fixture()
        tasks = tasks + (
            replace(tasks[0], task_id="task1", arrival_time=1.0),
        )
        flow1 = LogicalFlow(
            "flow::traj0::task1::offload::0", "traj0", "task1", "offload", 0
        )
        hop1 = CarryingHop(
            "hop::flow1::0",
            flow1.flow_id,
            0,
            "uav0",
            "rsu0",
            "pe::uav0::rsu0",
            "wireless",
        )
        decision1 = replace(
            action.decisions[0],
            task_id="task1",
            flow_id=flow1.flow_id,
            hop_id=hop1.hop_id,
        )
        bad = replace(
            action,
            decisions=action.decisions + (decision1,),
            flows=action.flows + (flow1,),
            hops=action.hops + (hop1,),
            rb_allocations=action.rb_allocations
            + (RbAllocation(flow1.flow_id, hop1.hop_id, 0),),
        )
        self.assert_code("same_transmitter_rb_conflict", nodes, edges, tasks, bad)

    def test_wireless_flow_without_rb_is_rejected(self):
        nodes, edges, tasks, action = wireless_fixture()
        self.assert_code(
            "wireless_flow_without_rb",
            nodes,
            edges,
            tasks,
            replace(action, rb_allocations=()),
        )

    def test_wired_flow_with_rb_is_rejected(self):
        nodes, edges, tasks, action = wireless_fixture()
        wired_edge = replace(edges[0], edge_type="wired")
        wired_hop = replace(action.hops[0], transport="wired")
        bad = replace(action, hops=(wired_hop,))
        self.assert_code("nonwireless_flow_has_rb", nodes, (wired_edge,), tasks, bad)

    def test_dag_edge_cannot_be_used_as_communication_hop(self):
        nodes, edges, tasks, action = wireless_fixture()
        dag = DagEdge(action.hops[0].hop_id, "task0", "task1")
        self.assert_code(
            "dag_edge_used_as_communication_hop",
            nodes,
            edges,
            tasks,
            action,
            dag_edges=(dag,),
        )

    def test_same_slot_outcome_source_is_rejected(self):
        nodes, edges, tasks, action = wireless_fixture()
        self.assert_code(
            "same_slot_outcome_leak",
            nodes,
            edges,
            tasks,
            action,
            sources={"channel": SnapshotPhase.OUTCOME},
        )

    def test_unknown_flow_identity_is_rejected(self):
        nodes, edges, tasks, action = wireless_fixture()
        bad_rb = replace(action.rb_allocations[0], flow_id="missing-flow")
        self.assert_code(
            "unknown_identity",
            nodes,
            edges,
            tasks,
            replace(action, rb_allocations=(bad_rb,)),
        )

    def test_missing_actionable_task_decision_is_rejected(self):
        nodes, edges, tasks, action = wireless_fixture()
        self.assert_code(
            "missing_task_decision",
            nodes,
            edges,
            tasks,
            replace(action, decisions=(), flows=(), hops=(), rb_allocations=()),
        )


if __name__ == "__main__":
    unittest.main()
