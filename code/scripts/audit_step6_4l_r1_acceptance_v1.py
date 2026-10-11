"""Independent acceptance audit for STEP 6.4L-R1 evidence."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v1"
WM = ROOT / "code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v2"
V4 = ROOT / "code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v4"
OUT = WM / "05_independent_acceptance.json"

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def validate_observed_identity(row):
    """Fail closed on actual recorded admission, parameter or recursive defects."""
    if (not row.get("official_admission_passed") or
            row.get("candidate", {}).get("generation_metadata", {}).get("planner_action_domain") != "V1"):
        raise ValueError("official candidate admission evidence missing")
    if row.get("parameter_before") != row.get("parameter_after"):
        raise ValueError("frozen parameters changed")
    inputs, outputs = row["inputs"], row["outputs"]
    if not outputs or len(inputs) != len(outputs) or len(row["finite"]) != len(outputs):
        raise ValueError("incomplete observed horizons")
    if inputs[0] != row["initial"] or any(inputs[h] != outputs[h - 1] for h in range(1, len(inputs))):
        raise ValueError("recursive input identity mismatch")
    if not all(row["finite"]):
        raise ValueError("nonfinite rollout evidence")

def main():
    runtime = load(RUNTIME / "04_acceptance_receipt.json")
    events = load(RUNTIME / "03_runtime_events.json")
    post = load(RUNTIME / "02_post_step.json")
    a = load(WM / "A_08_wm_scorer.json")
    b = load(WM / "B_08_wm_scorer.json")
    matrix = load(WM / "04_route_comm_support_matrix.json")
    checks = {}
    checks["runtime_completed"] = runtime.get("status") == "COMPLETED"
    checks["task5_tti_failure"] = post["task"].get("failure_reason_code") == 1 and post["task"].get("failure_reason") == "Task fails due to transmission timeout."
    checks["deadline_not_expired"] = float(post["simulation_time_s"]) < float(post["task"]["arrival_time"] + post["task"]["task_deadline"])
    names = [e.get("event") for e in events]
    checks["route_consumed_before_failure"] = names.index("setTaskReturnRouteAndStartReturn.after") < names.index("failOffloadingTask.after")
    checks["failure_before_wireless"] = names.index("failOffloadingTask.after") < names.index("_updateWirelessCommunication.before")
    alloc = [e for e in events if e.get("event") == "_allocate_communication_RBs"]
    checks["no_new_rb_request"] = bool(alloc) and alloc[-1].get("requested") == {}
    checks["no_new_task5_transfer"] = not any(float(e.get("time", -1)) >= 1.8 for e in post.get("transfer_events", []) if e.get("task_id") == "Task_5")
    checks["a_official_domain_qualification"] = a.get("stage") == "PASS" and a.get("wm_scorer", {}).get("horizons") == 4
    checks["a_return_birth_boundary"] = a.get("wm_scorer", {}).get("H_eff") == 0 and "UNSUPPORTED_FUTURE_RETURN_BIRTH:Task_5:H1" in json.dumps(a.get("wm_scorer", {}).get("diagnostics", {}))
    checks["b_shadow_h4"] = b.get("stage") == "PASS" and b.get("wm_scorer", {}).get("horizons") == 4
    checks["b_shadow_scoreable"] = b.get("wm_scorer", {}).get("scorer_status") == "SCOREABLE" and b.get("wm_scorer", {}).get("H_eff") == 4
    checks["b_formal_not_promoted"] = b.get("wm_scorer", {}).get("formal_score_promoted") is False and b.get("shadow_flow", {}).get("formal_score_not_promoted") is True
    checks["model_uav_hop_supported"] = bool(matrix["model"]["matching_relation_indices"]) and not matrix["model"]["raw_graph_support_missing"]
    checks["classification_temporary_no_service"] = matrix["classification"]["primary"] == "TEMPORARILY_NO_SERVICE"
    checks["same_formal_candidate_anchor_not_proven"] = a["wm_scorer"]["trace_initial"]["anchor"] != b["wm_scorer"]["trace_initial"]["anchor"]
    checks["no_gpu_locked_test"] = not runtime.get("GPU") and not runtime.get("locked_test") and not load(WM / "09_wm_acceptance_receipt.json").get("GPU")
    if V4.exists():
        completion = load(V4 / "10_observation_completion.json")
        va, vb = load(V4 / "A_observed.json"), load(V4 / "B_observed.json")
        vra, vrb = load(V4 / "A_raw_scorer.json"), load(V4 / "B_raw_scorer.json")
        checks["v4_completed_within_scope"] = (completion.get("status") == "COMPLETED" and
            completion.get("new_wm_transitions") == 8 and completion.get("GPU") is False and
            completion.get("locked_test") is False)
        checks["v4_a_admission_and_recursive_identity"] = (va.get("official_admission_passed") is True and
            va.get("candidate", {}).get("generation_metadata", {}).get("planner_action_domain") == "V1" and
            va.get("parameter_before") == va.get("parameter_after") and all(va.get("finite", [])) and
            va.get("inputs", [])[0] == va.get("initial") and
            all(va["inputs"][h] == va["outputs"][h - 1] for h in range(1, len(va["inputs"]))))
        checks["v4_b_admission_and_recursive_identity"] = (vb.get("official_admission_passed") is True and
            vb.get("parameter_before") == vb.get("parameter_after") and all(vb.get("finite", [])) and
            vb.get("inputs", [])[0] == vb.get("initial") and
            all(vb["inputs"][h] == vb["outputs"][h - 1] for h in range(1, len(vb["inputs"]))))
        checks["v4_raw_scorer_preserved"] = (vra["raw_result"]["status"] == "OBJECTIVE_UNSCOREABLE" and
            vra["raw_result"]["H_eff"] == 0 and
            vrb["raw_result"]["status"] == "SCOREABLE" and vrb["raw_result"]["H_eff"] == 4 and
            vrb["raw_result"]["scores"][0]["objective_tuple"] == [0, 0.0, 1.0, 1.0, 0.0] and
            vrb["raw_result"]["scores"][0]["diagnostics"]["future_truth_used"] is False)
        checks["v4_formal_scores_not_promoted"] = vra["formal_score_promoted"] is False and vrb["formal_score_promoted"] is False
    artifact_paths = sorted(p for base in (RUNTIME, WM, V4) if base.exists()
                            for pattern in ("*.json", "*.pt") for p in base.glob(pattern)
                            if p.name != OUT.name)
    inventory = [{"path": str(p.relative_to(ROOT)), "sha256": sha(p), "bytes": p.stat().st_size} for p in artifact_paths]
    receipt = {
        "schema": "PI-JWM-STEP-6.4L-R1-INDEPENDENT-ACCEPTANCE-v1",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checks": checks,
        "interpretation": {
            "task5_failure_root_cause": "TASK_FAIL_OUT_OF_TTI before wireless RB allocation",
            "route_comm_support": "TEMPORARILY_NO_SERVICE; current raw/model U2U relation exists",
            "frozen_wm_h4": "B shadow recursion PASS after integer index fix; this is not formal Route planner evidence",
            "scorer": "A exact fixed-support rejection; B shadow SCOREABLE; A/B objective comparison not established",
            "retrain_required": "UNDETERMINED",
        },
        "inventory": inventory,
        "GPU": False, "locked_test": False,
    }
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    if receipt["status"] != "PASS":
        raise SystemExit("independent acceptance checks failed")

if __name__ == "__main__":
    main()
