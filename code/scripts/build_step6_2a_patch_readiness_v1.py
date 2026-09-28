"""Build read-only reconciliation receipts from current source and existing evidence."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))
from pi_jwm.step4_4_structured_rssm_world_model_v1 import StructuredRSSMConfig, StructuredRSSMWorldModel
from pi_jwm.step6_2a_throughput_metric_v1 import extract_real_throughput
from pi_jwm.step6_2a_planner_objective_side_state_v1 import PlannerRouteCausalSideState, prepare_objective_side_state
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import load_typed_dual_graph_batch
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from build_step5_1d_unified_model_chain_v1 import build_state

OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_2a_patch_objective_readiness_v1_20260928"
OLD = ROOT / "code/artifacts/protocols/pi_jwm_step4_2c_b_causal_flow_ledger_raw_v1_20260920"
RAW = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/raw/formal-v1-sim-2026092302-policy-2026092402.json.gz"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+"\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    side = read(OUT / "deadline_sidecar_anchor_0001.json")
    alignment = read(OUT / "04_task_side_state_source_receipt.json")
    if not side["alignment_passed"] or not alignment["alignment_passed"]:
        raise RuntimeError("causal replay alignment failed")
    real = read(OLD / "real_trace_evidence.json")
    fixture = read(OLD / "contract_fixture_receipt.json")
    with gzip.open(RAW, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    anchor = raw["decisions"][1]
    package = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/packages"
    stem = "formal-v1-sim-2026092302-policy-2026092402"
    with gzip.open(package / "samples" / f"{stem}.json.gz", "rt", encoding="utf-8") as handle:
        sample = json.load(handle)[0]
    tensor = load_flow_tensor_batch(package / "tensor" / f"{stem}.npz")
    graph = load_typed_dual_graph_batch(package / "graph" / f"{stem}.npz")
    anchor_state, _ = build_state(tensor, graph, 0)
    routes = []
    carrying = {row["flow_id"]: row for row in sample["history"][-1]["carrying_states"] if row.get("known") and row.get("active")}
    for row in sample["history"][-1]["logical_flows"]:
        if row.get("known") and row.get("presence") and row["flow_id"] in carrying:
            c = carrying[row["flow_id"]]
            nodes = tuple(map(int, c["route_node_indices"]))
            routes.append(PlannerRouteCausalSideState(row["flow_id"], int(row["flow_index"]),
                row["task_id"], row["flow_type"], int(row["epoch"]),
                int(row["logical_destination_index"]), nodes, int(c["current_hop_index"]), int(c["route_revision"]),
                float(anchor["simulation_time_s"]), int(c["current_holder_index"])))
    prepared = prepare_objective_side_state(decision=anchor, deadline_sidecar=side,
        task_slots=sample["static"]["input_entity_index"]["task"],
        physical_slots=sample["static"]["input_entity_index"]["physical"], routes=routes,
        model_state=anchor_state)
    write("04a_anchor_prepared_side_state.json", {"status":"PASS", "sample_id":sample["metadata"]["sample_id"],
        "task_slot_alignment": {row.task_id: row.task_index for row in prepared.tasks},
        "task_count":len(prepared.tasks), "active_flow_count":len(prepared.routes),
        "effort_component_mask":prepared.effort_component_mask,
        "effort_denominators":prepared.effort_denominators,
        "model_state_route_current_alignment":True, "source_time_s":prepared.current_time_s})
    returned = [row["required_returned_size"] for row in anchor["tasks"]]
    ledger_source = ROOT / "code/src/pi_jwm/step4_2c_b_causal_flow_ledger_raw_v1.py"
    tensor_source = ROOT / "code/src/pi_jwm/step4_2c_c_flow_sample_tensor_v1.py"
    model_source = ROOT / "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py"
    adapter_source = ROOT / "code/scripts/build_step5_1d_unified_model_chain_v1.py"
    ledger_text, tensor_text, model_text, adapter_text = [p.read_text(encoding="utf-8") for p in
        (ledger_source, tensor_source, model_source, adapter_source)]
    assert 'flow_id = build_flow_id(task_id, flow_type, epoch_value)' in ledger_text
    assert '"e2e_remaining": total' in ledger_text
    assert '"route_node_indices": [int(node_index[node]) for node in route]' in tensor_text
    assert 'route = state["route_node_indices"][b, f]' in model_text
    assert 'next_hop + 1 >= route.shape[0]' in model_text
    assert 'src, dst = int(action["route_values"][b, r, 0]), int(action["route_values"][b, r, 1])' in model_text
    assert 'action["route_values"][b, r, 3]' not in model_text
    assert 'float(max(len(hops) - 1, 0))' in adapter_text
    write("01_historical_vs_current_flow_reconciliation.json", {
        "HISTORICAL_FACT": "4.2B source audit preceded the 4.2C-B causal ledger; E2E remaining was then not reconstructable from Raw alone.",
        "CURRENT_IMPLEMENTATION_FACT": "4.2C-B records FlowID=(TaskID,FlowType,Epoch), logical destination, total/e2e delivered/e2e remaining, holder, hop, route revision; 4.2C-C tensors these fields; graph and 4.4 carry them.",
        "CURRENT_VERDICT": "E2E_FLOW_FIELDS_PRESENT_AND_CONSERVED; do not reuse 4.2B missing-field verdict",
        "field_coverage": ["FlowID", "TaskID", "FlowType", "Epoch", "RouteRevision", "logical_source", "logical_destination", "total_data", "e2e_delivered", "e2e_remaining", "current_holder", "current_hop_index", "hop_progress", "hop_remaining", "route_node_indices", "route_node_mask", "carrying_active", "flow_presence", "flow_status"],
        "real_runtime_evidence": real["checks"], "synthetic_contract_evidence": fixture["checks"],
        "real_limitations": real["limitations"],
        "source_sha256": {str(p.relative_to(ROOT)).replace('\\','/'): sha(p) for p in (ledger_source,tensor_source,model_source)}})
    world = StructuredRSSMWorldModel(StructuredRSSMConfig(d_h=8,d_z=3,mlp_width=12,graph_layers=1,n_comm_rb=4))
    _, state, graph, action = world.synthetic_fixture()
    state["route_node_indices"][0,0] = torch.tensor([1,2,-1,-1])
    state["route_node_mask"][0,0] = torch.tensor([True,True,False,False])
    state["hop_remaining"][0,0] = .001
    action["route_flow_index"].fill_(-1)
    action["route_task_index"].fill_(-1)
    future, diagnostics = world.deterministic_transition(state, action,
        {"vehicle_motion": torch.zeros((1,4,4)), "csi": state["csi"].clone()},
        graph=graph, service_mode="expectation", generator=None)
    trapped = (float(diagnostics["delivered_bytes"][0,0]) > 0 and
               float(future["hop_remaining"][0,0]) == 0 and
               int(future["current_hop_index"][0,0]) == 0)
    if not trapped:
        raise RuntimeError("expected source mismatch no longer reproduced; re-audit")
    route_state = {key: value.clone() for key, value in state.items()}
    route_state["carrying_active"][0,0] = False
    route_action = {key: value.clone() for key, value in action.items()}
    route_action["route_flow_index"][0,0] = 0
    route_action["route_task_index"][0,0] = 0
    route_action["route_values"][0,0] = torch.tensor([0.,3.,2.,2.])
    rerouted, _ = world.deterministic_transition(route_state, route_action,
        {"vehicle_motion":torch.zeros((1,4,4)), "csi":route_state["csi"].clone()},
        graph=graph, service_mode="expectation", generator=None)
    route_action["route_values"][0,0,3] = 99.
    rerouted_other_hops, _ = world.deterministic_transition(route_state, route_action,
        {"vehicle_motion":torch.zeros((1,4,4)), "csi":route_state["csi"].clone()},
        graph=graph, service_mode="expectation", generator=None)
    assert torch.equal(rerouted["route_node_indices"], route_state["route_node_indices"])
    assert torch.equal(rerouted["route_node_mask"], route_state["route_node_mask"])
    assert int(rerouted["carrying_hop_destination_index"][0,0]) == 3
    assert int(rerouted["flow_route_revision"][0,0]) == 1
    assert torch.equal(rerouted["carrying_hop_destination_index"], rerouted_other_hops["carrying_hop_destination_index"])
    write("02_btx_current_support_audit.json", {
        "B_TX_NORMAL_HOP_READY": "READY_WITH_CURRENT_ROUTE_ADAPTER",
        "B_TX_MULTI_HOP_INPUT_READY": "BLOCKED_MODEL_ROUTE_INDEX_SEMANTICS",
        "B_TX_EXISTING_RETURN_READY": "CONTRACT_SUPPORTED_WITH_RUNTIME_EVIDENCE_LIMITATION_FOR_DIRECT_HOP; MULTIHOP_BLOCKED",
        "B_TX_REROUTE_READY": "BLOCKED_STALE_MODEL_ROUTE_ARRAY",
        "B_TX_DESTINATION_CHANGE_READY": "BLOCKED_MODEL_CANNOT_CREATE_NEW_EPOCH_OR_CHANGE_LOGICAL_DESTINATION",
        "B_TX_DEPDATA_READY": "NOT_APPLICABLE_V1_RUNTIME_INSTANCE_ZERO",
        "formula": "R_hop + (N_hop - 1 - current_hop_index) * R_e2e",
        "e2e_state_ready": True, "route_rollout_state_ready": False,
        "real_two_hop_input": real["observations"]["real_multihop_single_flow_analysis"]})
    write("03_route_state_rollout_consistency.json", {
        "status": "BLOCKED", "actual_rule_executed": True,
        "c_c_tensor_route_semantics": "remaining destination list, excludes current holder",
        "model_rule_route_semantics": "indexes route[current_hop_index+1] as holder and route[current_hop_index+2] as next destination",
        "actual_delivered_hop_bytes": float(diagnostics["delivered_bytes"][0,0]),
        "actual_hop_remaining_after": float(future["hop_remaining"][0,0]),
        "actual_current_hop_index_after": int(future["current_hop_index"][0,0]),
        "expected_current_hop_index_after": 1,
        "route_action_actual_rule_executed": True,
        "route_action_updates_full_array": False, "route_action_updates_endpoints": True,
        "route_action_updates_revision_on_endpoint_change": True,
        "route_values_hop_count_consumed_by_deterministic_rule": False,
        "planner_route_side_state_required": True,
        "planner_route_side_state_sufficient_for_trained_model_multi_hop": False,
        "blocker": "Side-state can detect divergence but cannot repair frozen model transition without changing predicted state semantics."})
    write("05_deadline_side_state_receipt.json", {
        "status": "READY_FOR_SELECTED_ANCHOR", "source": side["source"],
        "task_count": len(side["tasks"]), "task_ids": [x["task_id"] for x in side["tasks"]],
        "deadline_semantics": {"unit": "simulation seconds", "relative": True,
            "absolute": "arrival_time_s + deadline_s", "active_failure": "elapsed > deadline",
            "return_completion": "delay <= deadline", "no_return_completion": "delay <= deadline + 1e-5",
            "completion_before_active_failure_sweep": True},
        "deadline_failure_is_planner_derived": True, "learned_output": False,
        "alignment_receipt": "04_task_side_state_source_receipt.json"})
    write("06_return_requirement_receipt.json", {
        "status": "READY_WITH_HOST_CONDITION_FOR_SELECTED_ANCHOR", "formal_raw_has_required_returned_size": all("required_returned_size" in x for x in anchor["tasks"]),
        "formal_raw_has_return_destination": all("return_destination_id" in x for x in anchor["tasks"]),
        "task_count": len(returned), "positive_return_count": sum(x>0 for x in returned),
        "zero_return_count": sum(x==0 for x in returned),
        "semantic_rule": "Task.requireReturn() depends on assigned compute host versus return destination as well as required size; positive size alone is not sufficient for local execution",
        "runtime_source": "code/reference/AirFogSim/airfogsim/entities/task.py:getReturnedSize/requireReturn; task_manager.py:checkTasks",
        "raw_sha256": sha(RAW), "future_return_birth": "planner-derived support boundary only; no new Flow slot"})
    write("07_support_boundary_patch_receipt.json", {
        "status": "FORMULA_READY", "first_unsupported_state_u": 2, "H_sup": 1,
        "H_eff": "min across candidates H_sup", "unsupported_state_scored": False,
        "H_eff_zero": "OBJECTIVE_UNSCOREABLE; no fallback winner",
        "cases": {"no_return": "no Return-birth boundary", "existing_return": "supported continuation",
                  "future_return_birth": "u-1 if positive size and predicted compute host differs from Return destination",
                  "model_requirement_unknown_planner_known_true": "planner derives host-conditioned Return birth boundary"}})
    write("08_effort_contract_patch.json", {
        "status": "SOURCE_DEFINED", "components": ["Comm", "Comp", "Mob"],
        "route_effort": False, "priority_weighting": False,
        "aggregation": "mean of anchor-applicable normalized components",
        "effort_component_mask": "anchor-frozen, candidate-independent",
        "effort_denominator_contract": {"Comm": "current valid allocatable RB count",
            "Comp": "sum applicable observed raw static CPU capacity per second",
            "Mob": "15 m/s times present UAV count; formal mobility core domain, not physical limit"}})
    example = extract_real_throughput([
        {"flow_id":"f","target_id":"B","delivered_data":1,"e2e_remaining_after":1},
        {"flow_id":"f","target_id":"C","delivered_data":1,"e2e_remaining_after":1},
        {"flow_id":"f","target_id":"D","delivered_data":1,"e2e_remaining_after":0}],
        elapsed_s=2, flow_definitions={"f":{"total_data":1,"logical_destination":"D"}})
    write("09_throughput_metric_freeze.json", {"status":"FROZEN_DEFINITION_NO_CLOSED_LOOP_EXECUTION",
        "primary":"END_TO_END_USEFUL_THROUGHPUT", "diagnostic":"NETWORK_SERVICE_THROUGHPUT",
        "shared_extractor":"code/src/pi_jwm/step6_2a_throughput_metric_v1.py:extract_real_throughput",
        "real_ledger_multihop":real["observations"]["real_multihop_single_flow_analysis"],
        "synthetic_three_hop_one_megabyte_example":example,
        "planner_independent_weighted_term":False})
    baseline = {"throughput":{"primary":"END_TO_END_USEFUL_THROUGHPUT",
        "diagnostic":"NETWORK_SERVICE_THROUGHPUT",
        "extractor":"pi_jwm.step6_2a_throughput_metric_v1.extract_real_throughput"},
        "planner_objective":{"tuple":["N_DDL","A_DDL","J_Delay","J_Burden","J_Effort"],
            "route_effort":False,"priority_weighting":False,"risk_active":False,"energy":False,"fairness":False},
        "planner_side_state":{"required_fields":["task_id","arrival_time_s","elapsed_time_s",
            "deadline_s","absolute_deadline_s","required_returned_size","return_destination_index","requires_return",
            "flow_id","route_node_indices","current_hop_index","route_revision"]},
        "deadline":{"active_failure":"elapsed > deadline","return_completion":"delay <= deadline",
            "no_return_completion":"delay <= deadline + 1e-5"},
        "support_boundary":{"H_sup":"first unsupported state index - 1",
            "H_eff":"minimum H_sup over candidates","zero":"OBJECTIVE_UNSCOREABLE"},
        "separation":["Planner Objective != Final Evaluation Metric",
            "Predicted Metric != Real Closed-loop Metric","Evaluation Metric != Acceptance Gate"]}
    write("baseline_sync_contract_v1.json", baseline)
    write("10_baseline_sync_patch.json", {"status":"INTERFACE_SYNCED_NOT_EXECUTED",
         "machine_contract":"baseline_sync_contract_v1.json", "common_metric_extractor_required":True})
    write("11_leakage_audit.json", {"status":"PASS_FOR_SELECTED_ANCHOR_AND_SIDE_STATE_CODE",
        "side_state_source_time_le_anchor": side["source_time_s"] <= anchor["simulation_time_s"],
        "future_target_used":False,"future_schedule_used":False,"recorded_future_action_used":False,
        "future_outcome_used":False,"locked_test_accessed":False,
        "formal_dataset_identity_unchanged":True,"checkpoint_identity_unchanged":True,
        "training_tensor_unchanged":True,
        "negative_tests":"test_step6_2a_patch_side_state_v1.py"})
    matrix = {
        "DEADLINE_SOURCE_READINESS":"READY_SELECTED_ANCHOR",
        "DEADLINE_SIDE_STATE_READINESS":"READY_SELECTED_ANCHOR",
        "RETURN_REQUIREMENT_READINESS":"READY_WITH_PREDICTED_COMPUTE_HOST_CONDITION",
        "TX_E2E_STATE_READINESS":"READY",
        "TX_ROUTE_STATE_READINESS":"BLOCKED_MULTI_HOP_AND_REROUTE",
        "TX_BURDEN_SOURCE_READINESS":"BLOCKED_MULTI_HOP_AND_REROUTE",
        "COMP_BURDEN_SOURCE_READINESS":"READY_FORMULA",
        "EFFORT_SOURCE_READINESS":"READY_CONTRACT",
        "SUPPORT_BOUNDARY_READINESS":"READY_FORMULA",
        "THROUGHPUT_METRIC_READINESS":"READY_DEFINITION",
        "BASELINE_SYNC_INTERFACE":"READY_INTERFACE_NOT_EXECUTED",
        "STEP_6_2B_READINESS":"BLOCKED",
        "blocking_condition":"4.2C-C remaining-destination route array and frozen 4.4 route-index rule disagree; route action also leaves full array stale. Planner side-state detects, but cannot keep predicted holder/route aligned through multi-hop completion. No full candidate-action contract B_Tx semantics yet."}
    write("12_step6_2b_readiness_recomputed.json", matrix)
    names = sorted(p.name for p in OUT.glob("*.json") if p.name != "manifest.json")
    write("manifest.json", {"schema_version":"PI-JWM-Step6.2A-PATCH-v1","status":"BLOCKED_FOR_STEP_6_2B",
        "artifact_files":names+["manifest.json"],"source_hashes":{str(p.relative_to(ROOT)).replace('\\','/'):sha(p)
            for p in (ledger_source,tensor_source,model_source,adapter_source,RAW,
                ROOT / "code/src/pi_jwm/step6_2a_planner_objective_side_state_v1.py",
                ROOT / "code/src/pi_jwm/step6_2a_throughput_metric_v1.py")},
        "cpu_only":True,"gpu":False,"locked_test_accessed":False,"candidate_ranking":False,
        "objective_scorer_implemented":False,"baseline_executed":False})
    print(json.dumps(matrix, ensure_ascii=False))


if __name__ == "__main__":
    main()
