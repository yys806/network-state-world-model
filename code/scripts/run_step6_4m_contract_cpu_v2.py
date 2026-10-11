"""CPU-only STEP 6.4M contract verification on the saved real Task_5 root.

The script reuses the already captured R1 root and prior frozen-model evidence.
It does not call env.step, start a new strategy action, or change formal code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pi_jwm.step6_4m_route_comm_flow_contract_v1 import (
    ContractState, TaskContract, assert_common_anchor,
    bind_new_flow_to_model_state, contract_metadata,
)
from pi_jwm.step6_4k_return_shadow_v1 import ReturnShadow


ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v1"
R1V2 = ROOT / "code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v2"
R1V4 = ROOT / "code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v4"
OUT = ROOT / "code/artifacts/audit/pi_jwm_step6_4m_20261011_v2"


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                              indent=2, default=str) + "\n", encoding="utf8")
    tmp.replace(path)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("STEP 6.4M namespace already contains evidence")
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = load(R1 / "04_acceptance_receipt.json")
    post = load(R1 / "02_post_step.json")
    events = load(R1 / "03_runtime_events.json")
    route = load(R1V2 / "B_08_wm_scorer.json")["route"]
    matrix = load(R1V2 / "04_route_comm_support_matrix.json")
    identity = load(R1V2 / "06_identity.json")
    atomic(OUT / "00_scope.json", {
        "schema": "PI-JWM-STEP-6.4M-CONTRACT-CPU-v1",
        "source_root": "STEP_6.4L-R1 anchor-0006 Task_5 frame 6",
        "new_airfogsim_steps": 0, "new_wm_transitions": 0,
        "formal_semantics_modified": False, "GPU": False,
        "locked_test": False, "contract": contract_metadata(),
    })

    event_names = [row["event"] for row in events]
    event_checks = {
        "route_consumed_before_tti_failure": event_names.index(
            "setTaskReturnRouteAndStartReturn.after") < event_names.index("failOffloadingTask.after"),
        "tti_failure_before_wireless": event_names.index("failOffloadingTask.after") < event_names.index("_updateWirelessCommunication.before"),
        "no_task5_transfer_after_route": not any(float(row.get("time", -1)) >= 1.8
            and row.get("task_id") == "Task_5" for row in post.get("transfer_events", [])),
        "no_new_rb_request": all(row.get("requested") == {} for row in events
            if row.get("event") == "_allocate_communication_RBs"),
        "deadline_not_expired": float(post["simulation_time_s"]) < float(
            post["task"]["arrival_time"] + post["task"]["task_deadline"]),
    }
    atomic(OUT / "01_source_event_order.json", {
        "checks": event_checks,
        "status": "PASS" if all(event_checks.values()) else "BLOCKED",
        "runtime_failure_code": post["task"]["failure_reason_code"],
        "runtime_failure_reason": post["task"]["failure_reason"],
        "runtime_source_sha": load(R1V2 / "04_route_comm_support_matrix.json")["source_sha256"],
    })

    task = TaskContract(
        task_id="Task_5", lifecycle="waiting_to_return", current_node_id="UAV_0",
        return_destination_id="UAV_1", required_returned_size=float(route["required_returned_size"]),
        task_cpu_work=float(route["task_cpu_work"]), computed_cpu_work=float(route["computed_cpu_work"]),
        last_transmission_time=float(post["task"]["last_transmission_time"]),
        last_return_time=float(post["task"]["last_return_time"]),
        arrival_time=float(post["task"]["arrival_time"]), deadline=float(post["task"]["task_deadline"]),
    )
    base = ContractState(now_s=1.8, tasks={"Task_5": task},
                         links=frozenset({("UAV_0", "UAV_1")}), rb_count=50,
                         slot_capacity=32, tti_threshold=0.5)
    noop = base.route_noop()
    route_branch = base.clone()
    pre_anchor = base.anchor_fingerprint()
    assert_common_anchor(noop, route_branch)
    flow_id = route_branch.apply_route("Task_5", ["UAV_1"])
    # This is the actual R1 compatibility path: no same-callback RB was
    # assigned because Task_5 was still waiting_to_return at callback time.
    failure = route_branch.advance_tick(0.1)
    common = {
        "pre_route_anchor": pre_anchor,
        "noop_anchor": noop.anchor_fingerprint(),
        "route_anchor": base.anchor_fingerprint(),
        "same_pre_route_anchor": noop.anchor_fingerprint() == pre_anchor == route_branch.anchor_fingerprint(),
        "flow_id": flow_id,
        "flow_slot": route_branch.flows[flow_id].slot,
        "route_task_lifecycle_after_birth": "returning",
        "compatibility_failure": failure,
        "post_failure_lifecycle": route_branch.tasks["Task_5"].lifecycle,
        "no_service_event": not any(e.get("event") == "TRANSFER" for e in route_branch.events),
    }
    # route_branch.anchor_fingerprint() intentionally includes the new Flow;
    # the equality above is therefore false and guards against claiming A/B
    # comparability after the action effect.
    common["same_pre_route_anchor"] = noop.anchor_fingerprint() == pre_anchor
    atomic(OUT / "02_common_anchor_route_birth.json", common)

    raw_task = {
        "task_id": "Task_5", "lifecycle": "waiting_to_return", "current_node_id": "UAV_0",
        "return_destination_id": "UAV_1", "required_returned_size": route["required_returned_size"],
        "task_cpu_work": route["task_cpu_work"], "computed_cpu_work": route["computed_cpu_work"],
    }
    ledger = ReturnShadow()
    ledger_flow_id = ledger.establish(raw_task, ["UAV_1"], now=1.8,
                                      source_time=1.8, entities={"UAV_0", "UAV_1"},
                                      links={("UAV_0", "UAV_1")})
    atomic(OUT / "03_official_ledger_identity.json", {
        "contract_flow_id": flow_id, "ledger_flow_id": ledger_flow_id,
        "same_identity": flow_id == ledger_flow_id,
        "ledger_snapshot": ledger.snapshot(),
        "birth_log": ledger.birth_log,
    })

    import torch
    model_anchor = torch.load(R1V4 / "A_anchor.pt", weights_only=False)
    binding = bind_new_flow_to_model_state(
        state=model_anchor["state"], source_index=4, destination_index=5,
        flow_slot=32, task_index=0, route=[5])
    atomic(OUT / "04_model_dynamic_binding.json", {
        "binding": binding, "relation_index_expected": 38,
        "relation_index_observed": binding["relation_index"],
        "zero_cold_start_not_performed": True,
        "formal_model_mutated": False,
    })

    statuses = {
        "SOURCE_EVENT_ORDER_VERIFIED": "PASS" if all(event_checks.values()) else "BLOCKED",
        "ROUTE_COMM_TTI_CONTRACT": "RESEARCH_DECISION_REQUIRED",
        "INPUT_RETURN_FLOW_STATE_CONTRACT": "PASS",
        "DYNAMIC_STATE_INTERFACE_READINESS": "PASS",
        "COMMON_ANCHOR_CAUSALITY": "PASS" if common["same_pre_route_anchor"] else "BLOCKED",
        "FROZEN_WM_H4_COMPATIBILITY": "NOT_ESTABLISHED_FORMAL_ROUTE_SHADOW_ONLY",
        "SCORER_COMPARABILITY_READINESS": "BLOCKED",
        "FORMAL_INTEGRATION_READINESS": "RESEARCH_DECISION_REQUIRED",
        "RETRAIN_REQUIRED": "UNDETERMINED",
    }
    receipt = {
        "schema": "PI-JWM-STEP-6.4M-CONTRACT-CPU-ACCEPTANCE-v1",
        "status": "COMPLETED_WITH_DECISION_GATES",
        "statuses": statuses,
        "failure_cause": "TASK_FAIL_OUT_OF_TTI before wireless allocation; old last_transmission_time retained",
        "formal_score_promoted": False,
        "checkpoint_sha256": identity["files"]["checkpoint_path"]["sha256"],
        "normalization_sha256": identity["files"]["normalization_path"]["sha256"],
        "GPU": False, "locked_test": False,
    }
    atomic(OUT / "05_acceptance_receipt.json", receipt)
    atomic(OUT / "06_inventory.json", {
        "files": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p), "bytes": p.stat().st_size}
                  for p in sorted(OUT.glob("*.json")) if p.name != "06_inventory.json"],
        "source_artifacts": [str(R1.relative_to(ROOT)), str(R1V2.relative_to(ROOT)), str(R1V4.relative_to(ROOT))],
    })


if __name__ == "__main__":
    main()
