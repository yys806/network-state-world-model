import json
import unittest
from pathlib import Path

import torch

from pi_jwm.step6_4m_route_comm_flow_contract_v1 import (
    ContractError,
    ContractState,
    TaskContract,
    assert_common_anchor,
    bind_new_flow_to_model_state,
    contract_metadata,
)


def make_task(**overrides):
    values = dict(task_id="Task_5", lifecycle="waiting_to_return",
                  current_node_id="UAV_0", return_destination_id="UAV_1",
                  required_returned_size=0.076, task_cpu_work=1.0,
                  computed_cpu_work=1.0, last_transmission_time=1.2,
                  last_return_time=-1.0, arrival_time=0.6, deadline=8.5,
                  task_size=1.0)
    values.update(overrides)
    return TaskContract(**values)


def make_state(**overrides):
    values = dict(now_s=1.8, tasks={"Task_5": make_task()},
                  links=frozenset({("UAV_0", "UAV_1")}), rb_count=50,
                  slot_capacity=4, tti_threshold=0.5)
    values.update(overrides)
    return ContractState(**values)


class Step64MContractTests(unittest.TestCase):
    def test_independent_acceptance_receipt_is_complete(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "code/artifacts/audit/pi_jwm_step6_4m_20261011_v2/07_independent_acceptance.json"
        self.assertTrue(path.is_file())
        receipt = json.loads(path.read_text(encoding="utf8"))
        self.assertEqual(receipt["status"], "PASS_WITH_DECISION_GATES")
        self.assertEqual(receipt["passed_count"], receipt["check_count"])
        self.assertFalse(receipt["GPU"])
        self.assertFalse(receipt["locked_test"])

    def test_metadata_is_isolated(self):
        self.assertFalse(contract_metadata()["formal_semantics_modified"])
        self.assertTrue(contract_metadata()["single_hop_only"])

    def test_common_anchor_and_route_birth(self):
        left = make_state(); right = left.clone()
        assert_common_anchor(left, right)
        self.assertEqual(left.anchor_fingerprint(), right.anchor_fingerprint())
        self.assertEqual(left.route_noop().flows, {})
        fid = right.apply_route("Task_5", ["UAV_1"])
        self.assertEqual(fid, "flow::Task_5::Return::0")
        self.assertEqual(right.flows[fid].slot, 0)
        with self.assertRaises(ContractError):
            assert_common_anchor(left, right)

    def test_tti_compatibility_fails_before_service(self):
        state = make_state()
        fid = state.apply_route("Task_5", ["UAV_1"])
        state.assign_comm("Task_5", [0])
        out = state.advance_tick(0.1, service_events=[
            {"task_id": "Task_5", "source_id": "UAV_0", "target_id": "UAV_1",
             "delivered_data": 0.076, "stage_or_hop_completed": True}])
        self.assertEqual(fid, "flow::Task_5::Return::0")
        self.assertEqual(out[0]["event"], "TASK_FAIL_OUT_OF_TTI")
        self.assertEqual(state.tasks["Task_5"].lifecycle, "failed")
        self.assertTrue(state.flows[fid].presence)

    def test_legal_service_completes_flow_and_task(self):
        state = make_state(now_s=0.0, tti_threshold=0.5,
                           tasks={"Task_5": make_task(last_transmission_time=0.0,
                                                       arrival_time=0.0)})
        fid = state.apply_route("Task_5", ["UAV_1"])
        state.assign_comm("Task_5", [0, 1])
        out = state.advance_tick(0.1, service_events=[
            {"task_id": "Task_5", "source_id": "UAV_0", "target_id": "UAV_1",
             "delivered_data": 0.076, "stage_or_hop_completed": True}])
        self.assertEqual(out[0]["task_id"], "Task_5")
        self.assertEqual(state.flows[fid].status, "COMPLETED")
        self.assertEqual(state.tasks["Task_5"].lifecycle, "done")
        self.assertFalse(state.flows[fid].presence)

    def test_no_return_requirement_does_not_birth(self):
        state = make_state(tasks={"Task_5": make_task(required_returned_size=0.0)})
        self.assertIsNone(state.apply_route("Task_5", ["UAV_1"]))
        self.assertFalse(state.flows)

    def test_duplicate_birth_and_active_reroute_fail_closed(self):
        state = make_state(); state.apply_route("Task_5", ["UAV_1"])
        with self.assertRaisesRegex(ContractError, "DUPLICATE_FLOW_BIRTH"):
            state.apply_route("Task_5", ["UAV_1"])

    def test_invalid_route_and_missing_link(self):
        with self.assertRaisesRegex(ContractError, "INVALID_ROUTE_DESTINATION"):
            make_state().apply_route("Task_5", ["UAV_0"])
        with self.assertRaisesRegex(ContractError, "ROUTE_LINK_UNAVAILABLE"):
            make_state(links=frozenset()).apply_route("Task_5", ["UAV_1"])
        with self.assertRaisesRegex(ContractError, "MULTIHOP"):
            make_state(links=frozenset({("UAV_0", "UAV_2"), ("UAV_2", "UAV_1")})).apply_route(
                "Task_5", ["UAV_2", "UAV_1"])

    def test_comm_invalid_qualification_and_rb(self):
        state = make_state(); state.apply_route("Task_5", ["UAV_1"])
        with self.assertRaisesRegex(ContractError, "COMM_RB_LIST_INVALID"):
            state.assign_comm("Task_5", [])
        with self.assertRaisesRegex(ContractError, "COMM_RB_OUT_OF_RANGE"):
            state.assign_comm("Task_5", [50])
        with self.assertRaisesRegex(ContractError, "COMM_FLOW_NOT_ACTIVE"):
            make_state().assign_comm("Task_5", [0])

    def test_missing_transfer_fields_and_future_truth(self):
        state = make_state(now_s=0.0, tasks={"Task_5": make_task(last_transmission_time=0.0)})
        state.apply_route("Task_5", ["UAV_1"])
        with self.assertRaisesRegex(ContractError, "TRANSFER_FIELDS_MISSING"):
            state.apply_transfer({"task_id": "Task_5"})
        with self.assertRaisesRegex(ContractError, "FUTURE_TRUTH"):
            state.apply_transfer({"task_id": "Task_5", "source_id": "UAV_0",
                                  "target_id": "UAV_1", "delivered_data": 0.076,
                                  "stage_or_hop_completed": True, "future_outcome": True})

    def test_slot_capacity_and_unknown_identity(self):
        state = make_state(slot_capacity=1)
        state.apply_route("Task_5", ["UAV_1"])
        other = make_task(task_id="Task_6", lifecycle="waiting_to_return")
        state.tasks["Task_6"] = other
        with self.assertRaisesRegex(ContractError, "FLOW_SLOT_CAPACITY"):
            state.apply_route("Task_6", ["UAV_1"])
        with self.assertRaisesRegex(ContractError, "TASK_ID_UNKNOWN"):
            make_state().apply_route("Missing", ["UAV_1"])

    def test_task15_fp_boundary_is_not_rewritten(self):
        total = 0.4836374411490477
        near = make_task(task_id="Task_15", task_cpu_work=total,
                         computed_cpu_work=total - 1.11e-16)
        exact = make_task(task_id="Task_15", task_cpu_work=total,
                          computed_cpu_work=total)
        self.assertFalse(near.completed_exact)
        self.assertTrue(exact.completed_exact)

    def test_model_binding_requires_integer_indices_and_existing_relation(self):
        state = {
            "flow_presence": torch.zeros((1, 2), dtype=torch.bool),
            "flow_known": torch.zeros((1, 2), dtype=torch.bool),
            "comm_presence": torch.tensor([[True]]),
            "comm_validity": torch.tensor([[True]]),
            "comm_source_index": torch.tensor([[4]]),
            "comm_target_index": torch.tensor([[5]]),
        }
        row = bind_new_flow_to_model_state(state=state, source_index=4,
                                           destination_index=5, flow_slot=2,
                                           task_index=0, route=[5])
        self.assertEqual(row["relation_index"], 0)
        self.assertTrue(row["append_required"])
        with self.assertRaisesRegex(ContractError, "MUST_BE_INT"):
            bind_new_flow_to_model_state(state=state, source_index="UAV_0",
                                         destination_index=5, flow_slot=2,
                                         task_index=0, route=[5])
        with self.assertRaisesRegex(ContractError, "SUPPORT_MISSING"):
            bind_new_flow_to_model_state(state=state, source_index=5,
                                         destination_index=4, flow_slot=2,
                                         task_index=0, route=[4])

    def test_input_route_uses_same_identity_and_conservation_contract(self):
        task = make_task(task_id="InputTask", lifecycle="waiting_to_offload",
                         current_node_id="UAV_0", return_destination_id="UAV_1",
                         required_returned_size=0.0, task_size=2.0,
                         task_cpu_work=2.0, computed_cpu_work=0.0)
        state = make_state(tasks={"InputTask": task})
        fid = state.apply_route("InputTask", ["UAV_1"], flow_type="Input",
                                logical_destination="UAV_1")
        self.assertEqual(fid, "flow::InputTask::Input::0")
        flow = state.flows[fid]
        self.assertAlmostEqual(flow.total_data, 2.0)
        with self.assertRaisesRegex(ContractError, "TRANSFER_HOLDER"):
            state.apply_transfer({"task_id": "InputTask", "source_id": "UAV_1",
                                  "target_id": "UAV_1", "delivered_data": 2.0,
                                  "stage_or_hop_completed": True})


if __name__ == "__main__":
    unittest.main()
