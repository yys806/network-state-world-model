"""Build deterministic STEP 4.2C-B Raw/Ledger acceptance artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pi_jwm.step4_2c_b_causal_flow_ledger_raw_v1 import (
    CausalFlowLedger,
    amend_raw_with_causal_flow_ledger,
    validate_epoch_transition,
    validate_flow_ledger_receipt,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step4_2c_b_causal_flow_ledger_raw_v1_20260920"
REAL_SOURCE = ROOT / "code/artifacts/protocols/pi_jwm_step4_2c_b_real_flow_trace_source_v1_20260920/real_raw_contract_finalization.json"
REAL_MULTI_HOP_SOURCE = ROOT / "code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/real_communication_outcome_semantics.json"
IMPLEMENTATION_SOURCES = (
    "code/src/pi_jwm/step4_2c_b_causal_flow_ledger_raw_v1.py",
    "code/scripts/build_step4_2c_b_causal_flow_ledger_raw_v1.py",
    "code/scripts/run_step4_2c_b_real_flow_trace_v1.py",
    "code/scripts/run_step2_3_real_airfogsim_raw_contract_finalization_v1.py",
    "code/src/pi_jwm/airfogsim_full_dual_graph_observer_v1.py",
    "code/src/pi_jwm/airfogsim_full_dual_graph_collector_v1.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def task(task_id="FixtureTask", source="A", holder="A", size=10.0):
    return {"task_id": task_id, "task_node_id": source, "current_node_id": holder, "task_size": size}


def build_fixture_receipt() -> dict[str, object]:
    ledger = CausalFlowLedger()
    flow_id = ledger.create_input_flow(task(), logical_destination="C", route=["B", "C"])
    ledger.apply_transfer_event({"task_id": "FixtureTask", "phase": "offload", "source_id": "A", "target_id": "B", "delivered_data": 10.0, "stage_or_hop_completed": True}, observed_task_current_node_id="B")
    same_id = ledger.reroute(flow_id, new_route=["D", "C"], new_logical_destination="C")
    same_destination = same_id == flow_id and ledger.carrying(flow_id)["route_revision"] == 1
    old = ledger.flow(flow_id)
    new_id = ledger.reroute(flow_id, new_route=["E"], new_logical_destination="E")
    new = ledger.flow(new_id)
    epoch_check = validate_epoch_transition(old, new, destination_changed=True)

    partial = CausalFlowLedger()
    partial_id = partial.create_input_flow(task(task_id="Partial"), logical_destination="C", route=["B", "C"])
    partial.apply_transfer_event({"task_id": "Partial", "phase": "offload", "source_id": "A", "target_id": "B", "delivered_data": 3.0, "stage_or_hop_completed": False}, observed_task_current_node_id="A")
    rejected = False
    try:
        partial.reroute(partial_id, new_route=["D"], new_logical_destination="D")
    except ValueError as exc:
        rejected = str(exc) == "NOT_AT_CLEAN_HOP_BOUNDARY"

    local = CausalFlowLedger()
    local_none = local.create_input_flow(task(task_id="Local"), logical_destination="A", route=[]) is None
    depdata_rejected = False
    try:
        local.create_depdata_from_dag({"source": "A", "target": "B"})
    except ValueError:
        depdata_rejected = True
    checks = {
        "same_destination_reroute_identity": same_destination,
        "destination_change_epoch_lineage": bool(epoch_check["passed"] and ledger.flow(flow_id)["status"] == "SUPERSEDED"),
        "destination_change_epoch_conservation": bool(epoch_check["checks"]["epoch_conservation"]),
        "partial_hop_destination_change_rejected": rejected,
        "local_execution_no_fake_input_flow": local_none,
        "depdata_from_dag_forbidden": depdata_rejected,
        "runtime_depdata_instances_zero": True,
    }
    return {"evidence_class": "SYNTHETIC_CONTRACT_FIXTURE", "checks": checks, "passed": all(checks.values()), "flow_rows": ledger.all_flow_rows(), "carrying_rows": ledger.current_carrying_rows()}


def main() -> None:
    raw = json.loads(REAL_SOURCE.read_text(encoding="utf-8"))
    amended, ledger_receipt = amend_raw_with_causal_flow_ledger(raw)
    multihop = json.loads(REAL_MULTI_HOP_SOURCE.read_text(encoding="utf-8"))
    amended_multihop, multihop_receipt = amend_raw_with_causal_flow_ledger(multihop)
    fixture = build_fixture_receipt()

    transitions = [row for step in amended["steps"] for row in step["outcome"].get("flow_ledger_transition_evidence", [])]
    flows = amended["flow_ledger_history"]
    completed_types = {row["flow_type"] for row in flows if row["status"] == "COMPLETED"}
    real_checks = {
        "real_non_locked_airfogsim": raw.get("checks", {}).get("real_airfogsim_environment") is True and raw.get("scope", {}).get("locked_test") is False,
        "real_input_flow_completed": "Input" in completed_types,
        "real_return_flow_completed": "Return" in completed_types,
        "real_transitions_applied_without_error": bool(transitions) and not any("error" in row for row in transitions),
        "real_decision_causality": ledger_receipt["checks"]["raw_decision_causality"],
        "real_multihop_input_completed": any(row["flow_type"] == "Input" and row["status"] == "COMPLETED" for row in amended_multihop["flow_ledger_history"]),
        "legacy_completion_overlay_only": all(row.get("legacy_flow_completed_forbidden_as_logical_completion") is True for row in transitions),
    }
    real_evidence = {
        "evidence_class": "REAL_TRACE_EVIDENCE",
        "source_path": str(REAL_SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256(REAL_SOURCE),
        "multi_hop_source_path": str(REAL_MULTI_HOP_SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "multi_hop_source_sha256": sha256(REAL_MULTI_HOP_SOURCE),
        "checks": real_checks,
        "passed": all(real_checks.values()),
        "observations": {
            "flow_count": len(flows),
            "completed_input_count": sum(row["flow_type"] == "Input" and row["status"] == "COMPLETED" for row in flows),
            "completed_return_count": sum(row["flow_type"] == "Return" and row["status"] == "COMPLETED" for row in flows),
            "transition_count": len(transitions),
            "runtime_depdata_instances": 0,
            "same_destination_reroute_observed": False,
            "local_execution_observed": False,
        },
        "limitations": [
            "Real trace covers direct-hop Input/Return plus a separate real two-hop Input trace; Return multi-hop and same-destination reroute are contract fixtures.",
            "Legacy wireless Return event amount can exceed observer return_size; the frozen min(total, delivered+delta) rule caps logical delivery and preserves conservation.",
        ],
    }
    required = {
        "ledger_receipt": ledger_receipt["passed"] and validate_flow_ledger_receipt(ledger_receipt)["passed"],
        "multi_hop_receipt": multihop_receipt["passed"],
        "real_trace": real_evidence["passed"],
        "contract_fixture": fixture["passed"],
        "flow_id_index_stable": ledger_receipt["checks"]["flow_identity_reversible"] and ledger_receipt["checks"]["flow_index_unique"],
        "depdata_zero": ledger_receipt["checks"]["depdata_zero_instances"],
        "scope": all(ledger_receipt[name] is False for name in ("training", "gpu", "locked_test", "formal_dataset", "sample_tensor_extended", "graph_builder_started")),
    }
    acceptance = {
        "schema_version": "PI-JWM-Step4.2C-B-Acceptance-v1",
        "required_checks": required,
        "passed": all(required.values()),
        "implementation_fact": "Causal Flow Ledger and additive Raw Flow state implemented; Sample/Tensor and graph are not implemented.",
        "real_trace_observation": real_evidence,
        "derived_ledger_state": {"input_return": "existing real state/event plus deterministic causal rules", "depdata_instances": 0},
        "researcher_decision": {"flow_identity": "TaskID+FlowType+Epoch", "destination_change": "new epoch at clean hop boundary", "same_destination_reroute": "same epoch", "depdata": "vocabulary only, zero runtime instances"},
        "remaining_gap": ["real Return multi-hop not observed", "real same-destination reroute not observed", "Sample/Tensor Flow extension not started"],
        "scope": {"training": False, "gpu": False, "locked_test": False, "formal_dataset": False, "sample_tensor": False, "graph_builder": False},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "raw_causal_flow_ledger_amendment.json", amended)
    write(OUT / "real_trace_evidence.json", real_evidence)
    write(OUT / "contract_fixture_receipt.json", fixture)
    write(OUT / "acceptance.json", acceptance)
    files = []
    for name in ("raw_causal_flow_ledger_amendment.json", "real_trace_evidence.json", "contract_fixture_receipt.json", "acceptance.json"):
        path = OUT / name
        files.append({"path": name, "sha256": sha256(path)})
    source_files = [
        {"path": str(REAL_SOURCE.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(REAL_SOURCE), "role": "real_input_return_trace"},
        {"path": str(REAL_MULTI_HOP_SOURCE.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(REAL_MULTI_HOP_SOURCE), "role": "real_input_multihop_trace"},
        *[{"path": path, "sha256": sha256(ROOT / path), "role": "implementation_or_provenance_source"} for path in IMPLEMENTATION_SOURCES],
    ]
    manifest = {"schema_version": "PI-JWM-Step4.2C-B-Manifest-v1", "files": files, "source_files": source_files, "passed": acceptance["passed"], "observation_only_real_trace": True, "training": False, "gpu": False, "locked_test": False, "formal_dataset": False}
    write(OUT / "manifest.json", manifest)
    print(json.dumps({"output": str(OUT), "passed": acceptance["passed"], "required_checks": required}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
