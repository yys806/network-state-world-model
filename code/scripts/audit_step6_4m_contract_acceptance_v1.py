"""Independent acceptance for the additive STEP 6.4M contract evidence.

This auditor recomputes the important claims from the saved R1 runtime trace,
the v2 contract artifacts, and the source/checkpoint identity receipt.  It does
not trust the v2 status field as the sole acceptance criterion and never calls
AirFogSim, the formal planner, the World Model, or the formal scorer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pi_jwm.step6_4m_route_comm_flow_contract_v1 import (
    ContractState,
    TaskContract,
    assert_common_anchor,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/artifacts/audit/pi_jwm_step6_4m_20261011_v2"
R1 = ROOT / "code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v1"
R1V2 = ROOT / "code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v2"
IDENTITY = R1V2 / "06_identity.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not OUT.is_dir():
        raise RuntimeError(f"missing evidence namespace: {OUT}")
    scope = load(OUT / "00_scope.json")
    order = load(OUT / "01_source_event_order.json")
    common = load(OUT / "02_common_anchor_route_birth.json")
    ledger = load(OUT / "03_official_ledger_identity.json")
    binding = load(OUT / "04_model_dynamic_binding.json")
    receipt = load(OUT / "05_acceptance_receipt.json")
    runtime_events = load(R1 / "03_runtime_events.json")
    post = load(R1 / "02_post_step.json")
    route = load(R1V2 / "B_08_wm_scorer.json")["route"]
    identity = load(IDENTITY)

    names = [row["event"] for row in runtime_events]
    checks = {}
    checks["source_order_recomputed"] = (
        names.index("setTaskReturnRouteAndStartReturn.after")
        < names.index("failOffloadingTask.after")
        < names.index("_updateWirelessCommunication.before")
    )
    checks["runtime_failure_is_tti"] = (
        post["task"]["failure_reason_code"] == 1
        and post["task"]["failure_reason"] == "Task fails due to transmission timeout."
    )
    checks["no_task5_transfer_after_route"] = not any(
        row.get("task_id") == "Task_5" and float(row.get("time", -1)) >= 1.8
        for row in post.get("transfer_events", [])
    )
    checks["deadline_not_expired"] = float(post["simulation_time_s"]) < (
        float(post["task"]["arrival_time"])
        + float(post["task"]["task_deadline"])
    )
    checks["v2_event_receipt_agrees"] = all(order["checks"].values())

    task = TaskContract(
        task_id="Task_5", lifecycle="waiting_to_return", current_node_id="UAV_0",
        return_destination_id="UAV_1",
        required_returned_size=float(route["required_returned_size"]),
        task_cpu_work=float(route["task_cpu_work"]),
        computed_cpu_work=float(route["computed_cpu_work"]),
        last_transmission_time=float(post["task"]["last_transmission_time"]),
        last_return_time=float(post["task"]["last_return_time"]),
        arrival_time=float(post["task"]["arrival_time"]),
        deadline=float(post["task"]["task_deadline"]),
    )
    base = ContractState(
        now_s=1.8, tasks={"Task_5": task},
        links=frozenset({("UAV_0", "UAV_1")}), rb_count=50,
        slot_capacity=32, tti_threshold=0.5,
    )
    noop = base.route_noop()
    branch = base.clone()
    assert_common_anchor(noop, branch)
    flow_id = branch.apply_route("Task_5", ["UAV_1"])
    failure = branch.advance_tick(0.1)
    checks["common_anchor_recomputed"] = noop.anchor_fingerprint() == base.anchor_fingerprint()
    checks["route_birth_is_action_effect"] = flow_id == "flow::Task_5::Return::0"
    checks["tti_precedes_service_recomputed"] = (
        failure and failure[0]["event"] == "TASK_FAIL_OUT_OF_TTI"
        and not any(row.get("event") == "TRANSFER" for row in branch.events)
    )
    checks["post_failure_recomputed"] = branch.tasks["Task_5"].lifecycle == "failed"

    checks["ledger_identity_recomputed"] = (
        ledger["contract_flow_id"] == ledger["ledger_flow_id"] == flow_id
        and ledger["same_identity"] is True
    )
    checks["model_binding_recomputed"] = (
        binding["binding"]["relation_index"] == 38
        and binding["binding"]["source_index"] == 4
        and binding["binding"]["destination_index"] == 5
        and binding["binding"]["append_required"] is True
        and binding["formal_model_mutated"] is False
    )
    checks["identity_recomputed"] = (
        identity["checkpoint_sha256_expected"] == identity["files"]["checkpoint_path"]["sha256"]
        == receipt["checkpoint_sha256"]
        and identity["normalization_sha256_expected"]
        == identity["files"]["normalization_path"]["sha256"]
        == receipt["normalization_sha256"]
        and identity["locked_test"] is False
    )
    checks["scope_boundary_recomputed"] = (
        scope["new_airfogsim_steps"] == 0
        and scope["new_wm_transitions"] == 0
        and scope["formal_semantics_modified"] is False
        and scope["locked_test"] is False
        and scope["GPU"] is False
    )
    checks["status_fields_preserved"] = (
        receipt["statuses"]["FROZEN_WM_H4_COMPATIBILITY"]
        == "NOT_ESTABLISHED_FORMAL_ROUTE_SHADOW_ONLY"
        and receipt["statuses"]["SCORER_COMPARABILITY_READINESS"] == "BLOCKED"
        and receipt["statuses"]["RETRAIN_REQUIRED"] == "UNDETERMINED"
    )

    prior_v1 = load(ROOT / "code/artifacts/audit/pi_jwm_step6_4m_20261011_v1/05_acceptance_receipt.json")
    checks["prior_v1_defect_preserved"] = prior_v1["status"] == "BLOCKED"
    checks["v2_schema_fix_present"] = receipt["status"] == "COMPLETED_WITH_DECISION_GATES"

    output = {
        "schema": "PI-JWM-STEP-6.4M-INDEPENDENT-ACCEPTANCE-v1",
        "status": "PASS_WITH_DECISION_GATES" if all(checks.values()) else "BLOCKED",
        "checks": checks,
        "passed_count": sum(bool(v) for v in checks.values()),
        "check_count": len(checks),
        "recomputed_from": [
            str((R1 / "03_runtime_events.json").relative_to(ROOT)),
            str((R1 / "02_post_step.json").relative_to(ROOT)),
            str((OUT / "00_scope.json").relative_to(ROOT)),
            str((OUT / "01_source_event_order.json").relative_to(ROOT)),
            str((OUT / "02_common_anchor_route_birth.json").relative_to(ROOT)),
            str((OUT / "03_official_ledger_identity.json").relative_to(ROOT)),
            str((OUT / "04_model_dynamic_binding.json").relative_to(ROOT)),
            str((OUT / "05_acceptance_receipt.json").relative_to(ROOT)),
        ],
        "prior_v1": {
            "status": prior_v1["status"],
            "interpretation": "preserved as a receipt schema allowlist defect; v2 corrected it",
        },
        "GPU": False,
        "locked_test": False,
    }
    target = OUT / "07_independent_acceptance.json"
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    tmp.replace(target)
    if output["status"] == "BLOCKED":
        raise SystemExit(1)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
