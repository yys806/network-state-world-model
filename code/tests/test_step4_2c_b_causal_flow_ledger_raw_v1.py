import copy
import json
import unittest
from pathlib import Path

from pi_jwm.step4_2c_b_causal_flow_ledger_raw_v1 import (
    FLOW_TYPES,
    CausalFlowLedger,
    amend_raw_with_causal_flow_ledger,
    build_flow_id,
    parse_flow_id,
    validate_epoch_transition,
    validate_flow_transition,
    validate_flow_ledger_receipt,
    validate_logical_destination_continuity,
    validate_real_multihop_single_flow,
    validate_step4_2c_b_acceptance,
)


def task(task_id="T", source="A", holder="A", size=10.0, lifecycle="waiting_to_offload", return_destination="A"):
    return {"task_id": task_id, "task_node_id": source, "current_node_id": holder, "task_size": size, "lifecycle": lifecycle, "return_destination_id": return_destination}


def event(task_id="T", phase="offload", source="A", target="B", amount=10.0, completed=True):
    return {"task_id": task_id, "phase": phase, "source_id": source, "target_id": target, "delivered_data": amount, "stage_or_hop_completed": completed}


class Step42CBLedgerTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]
    def test_flow_id_is_stable_reversible_and_not_route_dependent(self):
        first = build_flow_id("Task::7", "Input", 0)
        self.assertEqual(("Task::7", "Input", 0), parse_flow_id(first))
        self.assertEqual(first, build_flow_id("Task::7", "Input", 0))
        self.assertEqual({"Input", "Return", "DepData"}, set(FLOW_TYPES))

    def test_input_multihop_progress_holder_and_completion(self):
        ledger = CausalFlowLedger()
        flow_id = ledger.create_input_flow(task(), logical_destination="C", route=["B", "C"])
        ledger.apply_transfer_event(event(target="B"), observed_task_current_node_id="B")
        row = ledger.flow(flow_id)
        self.assertEqual(0.0, row["e2e_delivered"])
        self.assertEqual(10.0, row["e2e_remaining"])
        self.assertEqual("B", ledger.carrying(flow_id)["current_holder"])
        ledger.apply_transfer_event(event(source="B", target="C", amount=3.0, completed=False), observed_task_current_node_id="B")
        self.assertEqual(3.0, ledger.flow(flow_id)["e2e_delivered"])
        self.assertEqual(7.0, ledger.flow(flow_id)["e2e_remaining"])
        ledger.apply_transfer_event(event(source="B", target="C", amount=7.0), observed_task_current_node_id="C")
        self.assertEqual("COMPLETED", ledger.flow(flow_id)["status"])
        self.assertFalse(ledger.flow(flow_id)["presence"])

    def test_normal_hop_advancement_keeps_flow_epoch_destination_and_route_revision(self):
        ledger = CausalFlowLedger()
        flow_id = ledger.create_input_flow(
            task(),
            logical_destination="C",
            route=["B", "C"],
            logical_destination_source="established_offload_route_terminal",
            logical_destination_capture_phase="after_route_action_established_before_outcome",
        )
        before = ledger.flow(flow_id)
        ledger.apply_transfer_event(event(target="B"), observed_task_current_node_id="B")
        after = ledger.flow(flow_id)
        carrying = ledger.carrying(flow_id)
        self.assertEqual(flow_id, after["flow_id"])
        self.assertEqual(before["epoch"], after["epoch"])
        self.assertEqual("C", after["logical_destination"])
        self.assertEqual(0, carrying["route_revision"])
        self.assertEqual(1, carrying["current_hop_index"])

    def test_return_is_independent_and_multihop(self):
        ledger = CausalFlowLedger()
        flow_id = ledger.create_return_flow(task(holder="C", lifecycle="waiting_to_return"), logical_destination="A", total_data=4.0, route=["B", "A"])
        ledger.apply_transfer_event(event(phase="return", source="C", target="B", amount=4.0), observed_task_current_node_id="B")
        self.assertEqual(4.0, ledger.flow(flow_id)["e2e_remaining"])
        ledger.apply_transfer_event(event(phase="return", source="B", target="A", amount=4.0), observed_task_current_node_id="A")
        self.assertEqual("COMPLETED", ledger.flow(flow_id)["status"])

    def test_local_execution_creates_no_fake_input_flow(self):
        ledger = CausalFlowLedger()
        self.assertIsNone(ledger.create_input_flow(task(), logical_destination="A", route=[]))
        self.assertEqual([], ledger.current_flow_rows())

    def test_same_destination_reroute_keeps_epoch_and_flow_id(self):
        ledger = CausalFlowLedger()
        flow_id = ledger.create_input_flow(task(), logical_destination="C", route=["B", "C"])
        before = copy.deepcopy(ledger.flow(flow_id))
        ledger.reroute(flow_id, new_route=["D", "C"], new_logical_destination="C")
        self.assertEqual(flow_id, ledger.flow(flow_id)["flow_id"])
        self.assertEqual(before["epoch"], ledger.flow(flow_id)["epoch"])
        self.assertEqual(1, ledger.carrying(flow_id)["route_revision"])

    def test_destination_change_at_clean_boundary_creates_lineage(self):
        ledger = CausalFlowLedger()
        old_id = ledger.create_input_flow(task(), logical_destination="C", route=["B", "C"])
        new_id = ledger.reroute(old_id, new_route=["D"], new_logical_destination="D")
        self.assertNotEqual(old_id, new_id)
        self.assertEqual("SUPERSEDED", ledger.flow(old_id)["status"])
        self.assertEqual(old_id, ledger.flow(new_id)["previous_flow_id"])
        self.assertEqual(new_id, ledger.flow(old_id)["superseded_by_flow_id"])
        self.assertEqual(ledger.flow(old_id)["e2e_remaining"], ledger.flow(new_id)["total_data"])
        self.assertEqual(1, ledger.flow(new_id)["epoch"])
        self.assertEqual(0, ledger.carrying(new_id)["route_revision"])

    def test_destination_change_during_partial_hop_is_rejected(self):
        ledger = CausalFlowLedger()
        flow_id = ledger.create_input_flow(task(), logical_destination="C", route=["B", "C"])
        ledger.apply_transfer_event(event(amount=3.0, completed=False), observed_task_current_node_id="A")
        with self.assertRaisesRegex(ValueError, "NOT_AT_CLEAN_HOP_BOUNDARY"):
            ledger.reroute(flow_id, new_route=["D"], new_logical_destination="D")

    def test_holder_event_inconsistency_and_legacy_completion_are_rejected(self):
        ledger = CausalFlowLedger()
        flow_id = ledger.create_input_flow(task(), logical_destination="B", route=["B"])
        bad = event(target="B")
        bad["flow_completed"] = True
        with self.assertRaisesRegex(ValueError, "legacy flow_completed"):
            ledger.apply_transfer_event(bad, observed_task_current_node_id="B")
        with self.assertRaisesRegex(ValueError, "HOLDER_EVENT_INCONSISTENCY"):
            ledger.apply_transfer_event(event(target="B"), observed_task_current_node_id="A")
        self.assertEqual(10.0, ledger.flow(flow_id)["e2e_remaining"])

    def test_future_action_does_not_change_current_snapshot(self):
        ledger = CausalFlowLedger()
        ledger.create_input_flow(task(), logical_destination="C", route=["B", "C"])
        before = ledger.snapshot()
        future_action = {"target": "D", "route": ["D"]}
        after = ledger.snapshot()
        self.assertEqual(before, after)
        self.assertEqual("D", future_action["target"])

    def test_depdata_vocabulary_has_zero_runtime_instances(self):
        ledger = CausalFlowLedger()
        ledger.create_input_flow(task(), logical_destination="B", route=["B"])
        self.assertEqual(0, sum(row["flow_type"] == "DepData" for row in ledger.all_flow_rows()))

    def test_raw_amendment_is_causal_and_legacy_schema_is_preserved(self):
        raw = {
            "schema_version": "base-v1",
            "decisions": [
                {"frame_index": 0, "tasks": [task()]},
                {"frame_index": 1, "tasks": [task(holder="B")]},
            ],
            "steps": [{
                "frame_index": 0,
                "action": {"route": {"entries": [{"task_id": "T", "route_kind": "offload", "target_node_id": "B", "route_node_ids": ["B"]}]}},
                "outcome": {"tasks": [task(holder="B")], "slot_transfer_events": [{"task_id": "T", "phase": "offload", "source_id": "A", "target_id": "B", "delivered_data": 10.0, "flow_completed": True}]},
            }],
        }
        amended, receipt = amend_raw_with_causal_flow_ledger(raw)
        self.assertEqual("base-v1", amended["base_schema_version"])
        self.assertEqual([], amended["decisions"][0]["logical_flow_rows"])
        self.assertEqual("COMPLETED", amended["flow_ledger_history"][0]["status"])
        self.assertFalse(amended["decisions"][1]["logical_flow_rows"][0]["presence"])
        self.assertEqual("stage_or_hop_completed", amended["steps"][0]["outcome"]["flow_ledger_transition_evidence"][0]["legacy_flow_completed_semantics"])
        self.assertTrue(receipt["passed"])

    def test_raw_amendment_does_not_use_current_next_hop_as_input_destination(self):
        raw = {
            "schema_version": "base-v1",
            "decisions": [{"frame_index": 0, "tasks": [task()]}],
            "steps": [{
                "frame_index": 0,
                "action": {"route": {"entries": [{
                    "task_id": "T", "route_kind": "offload",
                    "target_node_id": "B", "route_node_ids": ["B", "C"],
                }]}},
                "outcome": {"tasks": [task(holder="B")], "slot_transfer_events": []},
            }],
        }
        amended, receipt = amend_raw_with_causal_flow_ledger(raw)
        flow = amended["flow_ledger_history"][0]
        action = amended["steps"][0]["action"]["route"]["entries"][0]
        self.assertEqual("B", action["target_node_id"])
        self.assertEqual("C", flow["logical_destination"])
        self.assertEqual("established_offload_route_terminal", flow["logical_destination_source"])
        self.assertEqual("after_route_action_established_before_outcome", flow["logical_destination_capture_phase"])
        self.assertEqual("C", action["logical_destination_id"])
        self.assertTrue(receipt["passed"])

    def test_return_destination_comes_from_task_return_destination_not_current_hop(self):
        return_task = task(holder="C", lifecycle="waiting_to_return", return_destination="A")
        return_task["return_size"] = 4.0
        raw = {
            "schema_version": "base-v1",
            "decisions": [{"frame_index": 0, "tasks": [return_task]}],
            "steps": [{
                "frame_index": 0,
                "action": {"route": {"entries": [{
                    "task_id": "T", "route_kind": "return",
                    "target_node_id": "B", "route_node_ids": ["B", "A"],
                }]}},
                "outcome": {"tasks": [return_task], "slot_transfer_events": []},
            }],
        }
        amended, receipt = amend_raw_with_causal_flow_ledger(raw)
        flow = amended["flow_ledger_history"][0]
        self.assertEqual("A", flow["logical_destination"])
        self.assertEqual("decision.tasks[].return_destination_id", flow["logical_destination_source"])
        self.assertTrue(receipt["passed"])

    def test_receipt_tamper_fails_required_and_top_level_and(self):
        ledger = CausalFlowLedger()
        receipt = ledger.validation_receipt(runtime_depdata_instances=0)
        self.assertTrue(receipt["passed"])
        tampered = copy.deepcopy(receipt)
        tampered["checks"]["flow_conservation"] = False
        self.assertFalse(validate_flow_ledger_receipt(tampered)["passed"])

    def test_real_non_locked_trace_updates_input_and_return_ledger(self):
        path = self.ROOT / "code/artifacts/protocols/pi_jwm_step4_2c_b_real_flow_trace_source_v1_20260920/real_raw_contract_finalization.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        amended, receipt = amend_raw_with_causal_flow_ledger(raw)
        rows = amended["flow_ledger_history"]
        completed_types = {row["flow_type"] for row in rows if row["status"] == "COMPLETED"}
        transitions = [item for step in amended["steps"] for item in step["outcome"]["flow_ledger_transition_evidence"]]
        self.assertEqual({"Input", "Return"}, completed_types)
        self.assertTrue(transitions)
        self.assertFalse(any("error" in item for item in transitions))
        self.assertTrue(receipt["passed"])
        self.assertFalse(raw["scope"]["locked_test"])
        self.assertFalse(raw["scope"]["training"])
        self.assertFalse(raw["scope"]["gpu"])

    def test_negative_transition_tampers_are_rejected(self):
        before = {"logical_destination": "C", "total_data": 10.0, "e2e_delivered": 0.0, "e2e_remaining": 10.0}
        intermediate = {"target_id": "B", "delivered_data": 10.0}
        wrongly_counted = {**before, "e2e_delivered": 10.0, "e2e_remaining": 0.0}
        self.assertFalse(validate_flow_transition(before, intermediate, wrongly_counted)["passed"])
        final = {"target_id": "C", "delivered_data": 3.0}
        bad_remaining = {**before, "e2e_delivered": 3.0, "e2e_remaining": 8.0}
        self.assertFalse(validate_flow_transition(before, final, bad_remaining)["passed"])
        legacy = {**final, "flow_completed": True}
        correct = {**before, "e2e_delivered": 3.0, "e2e_remaining": 7.0}
        self.assertFalse(validate_flow_transition(before, legacy, correct)["passed"])

    def test_negative_epoch_identity_tampers_are_rejected(self):
        old = {"task_id": "T", "flow_type": "Input", "flow_id": build_flow_id("T", "Input", 0), "epoch": 0, "e2e_remaining": 7.0}
        same_destination_bad = {**old, "flow_id": build_flow_id("T", "Input", 1), "epoch": 1, "total_data": 7.0}
        self.assertFalse(validate_epoch_transition(old, same_destination_bad, destination_changed=False)["passed"])
        changed_without_epoch = {**old, "total_data": 7.0}
        self.assertFalse(validate_epoch_transition(old, changed_without_epoch, destination_changed=True)["passed"])

    def test_logical_destination_mutation_in_same_epoch_is_rejected(self):
        rows = [
            {"task_id": "T", "flow_type": "Input", "epoch": 0, "logical_destination": "C"},
            {"task_id": "T", "flow_type": "Input", "epoch": 0, "logical_destination": "D"},
        ]
        self.assertFalse(validate_logical_destination_continuity(rows)["passed"])

    def test_fake_multihop_from_two_independent_flows_is_rejected(self):
        fake = [
            {"task_id": "T1", "flow_id": "F1", "epoch": 0, "logical_destination": "B", "source_id": "A", "target_id": "B", "logical_delivery_delta": 10.0, "e2e_delivered_after": 10.0, "e2e_remaining_after": 0.0, "route_revision_before": 0, "route_revision_after": 0},
            {"task_id": "T2", "flow_id": "F2", "epoch": 0, "logical_destination": "C", "source_id": "B", "target_id": "C", "logical_delivery_delta": 10.0, "e2e_delivered_after": 10.0, "e2e_remaining_after": 0.0, "route_revision_before": 0, "route_revision_after": 0},
        ]
        result = validate_real_multihop_single_flow(fake)
        self.assertFalse(result["checks"]["real_multihop_single_flow_id"])
        self.assertFalse(result["passed"])

    def test_fake_multihop_required_check_forces_top_level_acceptance_false(self):
        from pi_jwm.step4_2c_b_causal_flow_ledger_raw_v1 import STEP42CB_PATCH_REQUIRED_CHECKS

        required = {name: True for name in STEP42CB_PATCH_REQUIRED_CHECKS}
        required["real_multihop_single_flow_id"] = False
        receipt = {
            "required_checks": required,
            "passed": True,
            "scope": {"training": False, "gpu": False, "locked_test": False, "formal_dataset": False, "sample_tensor": False, "graph_builder": False},
        }
        validation = validate_step4_2c_b_acceptance(receipt)
        self.assertFalse(validation["expected_passed"])
        self.assertFalse(validation["declared_matches_expected"])
        self.assertFalse(validation["passed"])

    def test_multihop_identity_epoch_destination_and_revision_tampers_are_rejected(self):
        rows = [
            {"task_id": "T", "flow_id": "F", "epoch": 0, "logical_destination": "C", "source_id": "A", "target_id": "B", "delivered_data": 10.0, "logical_delivery_delta": 0.0, "e2e_delivered_after": 0.0, "e2e_remaining_after": 10.0, "route_revision_before": 0, "route_revision_after": 0},
            {"task_id": "T", "flow_id": "F", "epoch": 0, "logical_destination": "C", "source_id": "B", "target_id": "C", "delivered_data": 10.0, "logical_delivery_delta": 10.0, "e2e_delivered_after": 10.0, "e2e_remaining_after": 0.0, "route_revision_before": 0, "route_revision_after": 0},
        ]
        self.assertTrue(validate_real_multihop_single_flow(rows)["passed"])
        for field, value, check in (
            ("flow_id", "F2", "real_multihop_single_flow_id"),
            ("epoch", 1, "real_multihop_single_epoch"),
            ("logical_destination", "D", "real_multihop_logical_destination_constant"),
            ("route_revision_after", 1, "normal_hop_advancement_not_reroute"),
        ):
            tampered = copy.deepcopy(rows)
            tampered[1][field] = value
            result = validate_real_multihop_single_flow(tampered)
            self.assertFalse(result["checks"][check])
            self.assertFalse(result["passed"])

    def test_real_two_hop_trace_is_one_flow_one_epoch_with_final_only_e2e_delivery(self):
        path = self.ROOT / "code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/real_communication_outcome_semantics.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        amended, receipt = amend_raw_with_causal_flow_ledger(raw)
        transitions = [
            item
            for step in amended["steps"]
            for item in step["outcome"].get("flow_ledger_transition_evidence", [])
            if item.get("task_id") == "Task_1" and item.get("flow_type") == "Input" and item.get("applied", True)
        ]
        result = validate_real_multihop_single_flow(transitions)
        self.assertTrue(result["passed"])
        self.assertEqual(["UAV_0", "RSU_0"], [row["source_id"] for row in result["hop_sequence"]])
        self.assertEqual(["RSU_0", "cloudServer_4"], [row["target_id"] for row in result["hop_sequence"]])
        self.assertEqual([0.0, raw["decisions"][0]["tasks"][0]["task_size"]], result["e2e_delivered_sequence"])
        self.assertTrue(receipt["passed"])

    def test_future_action_mutation_does_not_change_raw_current_state(self):
        raw = {"schema_version": "base", "decisions": [{"frame_index": 0, "tasks": [task()]}], "steps": []}
        changed = copy.deepcopy(raw)
        changed["future_action"] = {"target": "D", "route": ["D"]}
        first, _ = amend_raw_with_causal_flow_ledger(raw)
        second, _ = amend_raw_with_causal_flow_ledger(changed)
        self.assertEqual(first["decisions"][0]["logical_flow_rows"], second["decisions"][0]["logical_flow_rows"])

    def test_depdata_cannot_be_fabricated_from_dag(self):
        with self.assertRaisesRegex(ValueError, "DEPDATA_FROM_DAG_FABRICATION_FORBIDDEN"):
            CausalFlowLedger().create_depdata_from_dag({"source_task": "A", "target_task": "B"})


if __name__ == "__main__":
    unittest.main()
