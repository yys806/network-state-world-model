"""Close the CPU-only Comm selection and projected TRAIN H1 audit receipts."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "code/artifacts/protocols/pi_jwm_step6_3b_candidate_grammar_v1_20260929"
C = ROOT / "code/artifacts/protocols/pi_jwm_step6_3c_candidate_search_protocol_v1_20260929"
BASE = "f95bf2e5b9430666f325af3e6d6846d68b97e6e8"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def main() -> None:
    train_catalog = read(B / "02_train_structural_support_catalog.json")
    support = read(B / "01_train_comm_support_scope_and_duplicate_rows.json")
    train = read(C / "01_formal_train_domain_feasibility.json")
    validation = read(C / "02_formal_validation_domain_descriptive.json")
    replay_path = C / "06_train_h1_projected_self_replay.json"
    replay = read(replay_path)
    comp_delta = read(C / "08_comp_amount_residual_diagnostic.json")
    if len(train_catalog["joint_structural_signatures"]) != 251:
        raise ValueError("joint structural support changed")
    if len(train_catalog["comm_start_width_pairs"]) != 145:
        raise ValueError("Comm RB pair support changed")
    if support["comm_selected_task_counts"] != train_catalog["comm_selected_task_counts"]:
        raise ValueError("selected-task-count support mismatch")
    if train["anchor_count"] != replay["anchor_count"] != 4416:
        raise ValueError("TRAIN anchor coverage mismatch")
    if validation["anchor_count"] != 1104:
        raise ValueError("validation anchor coverage mismatch")
    if len(replay["per_anchor"]) != 4416 or len({r["sample_id"] for r in replay["per_anchor"]}) != 4416:
        raise ValueError("projected replay has missing or duplicate anchors")
    if sum(replay["excluded_comm_row_reasons"].values()) != (
            replay["raw_comm_row_count"] - replay["projected_comm_row_count"]):
        raise ValueError("excluded Comm row accounting mismatch")
    for anchor in replay["per_anchor"]:
        raw_action = anchor["raw_historical_action"]
        projected = anchor["planner_v1_projected_action"]
        if projected["route"] or projected["comp"] != raw_action["comp"]["entries"] \
                or projected["mobility"] != raw_action["mobility"]["entries"]:
            raise ValueError("Planner-v1 projection altered Route/Comp/Mob unexpectedly")
        key = lambda row: json.dumps(row, sort_keys=True, separators=(",", ":"))
        raw_comm = collections.Counter(map(key, raw_action["comm"]["entries"]))
        split_comm = collections.Counter(map(key, projected["comm"]))
        split_comm.update(key(item["row"]) for item in anchor["excluded_comm_rows"])
        if raw_comm != split_comm:
            raise ValueError("projected and excluded Comm rows do not reconstruct raw action")
        route_kinds = {str(row["task_id"]): row.get("route_kind")
                       for row in raw_action["route"]["entries"]}
        for excluded in anchor["excluded_comm_rows"]:
            if excluded["reason"] == "ROUTE_CREATED_FLOW_NOT_AVAILABLE_IN_CURRENT_STATE":
                if route_kinds.get(str(excluded["row"]["task_id"])) != "offload":
                    raise ValueError("excluded Route-created Comm row lacks same-decision offload")
    if replay["comm_task_selection_rejections"] != 0:
        raise ValueError("COMM_TASK_SELECTION projected rejection remains")
    if replay["comm_projection_admitted_count"] + sum(replay["comm_projection_failures"].values()) != 4416:
        raise ValueError("Comm projection verdict accounting mismatch")
    if replay["full_projected_admitted_count"] + sum(replay["full_projected_failures"].values()) != 4416:
        raise ValueError("full projected verdict accounting mismatch")
    if replay["projected_semantic_replay_pass"] + sum(
            replay["projected_semantic_replay_mismatches"].values()) != replay["full_projected_admitted_count"]:
        raise ValueError("projected semantic replay accounting mismatch")
    if comp_delta["source_receipt_sha256"] != hashlib.sha256(replay_path.read_bytes()).hexdigest() \
            or comp_delta["affected_anchor_count"] != replay["projected_semantic_replay_mismatches"].get(
                "PROJECTED_COMP_AMOUNT_DIFFERS", 0):
        raise ValueError("Comp residual diagnostic does not match projected replay")
    if any(obj.get("gpu") or obj.get("training") or obj.get("locked_test")
           or obj.get("future_target_used") for obj in (train, validation, replay, comp_delta)):
        raise ValueError("out-of-scope execution in patch receipt")
    if replay["validation_used"] or not replay["route_projected_empty"]:
        raise ValueError("projected replay violated split or Route policy")

    previous = {
        "source_commit": BASE,
        "train_empty_domains": 1935,
        "validation_empty_domains": 550,
        "train_exact_concrete_median": 6,
        "train_exact_concrete_max": 313949952,
    }
    current = {
        "train_empty_domains": train["empty_domain_count"],
        "validation_empty_domains": validation["empty_domain_count"],
        "train_exact_concrete_median": train["quantiles"]["exact_unique_concrete_candidates"]["median"],
        "train_exact_concrete_max": train["quantiles"]["exact_unique_concrete_candidates"]["max"],
        "validation_exact_concrete_median": validation["quantiles"]["exact_unique_concrete_candidates"]["median"],
        "validation_exact_concrete_max": validation["quantiles"]["exact_unique_concrete_candidates"]["max"],
    }
    patch = {
        "step": "STEP_6_3BC_PATCH",
        "STEP_6_3BC_PATCH": "PASS_FOR_COMM_TASK_SELECTION_AND_PROJECTED_H1_AUDIT",
        "researcher_decision": "TRAIN_SIGNATURE_CONDITIONAL_SELECTED_TASK_COUNT; EXISTING_UNIQUE_WIRELESS_FLOW_ONLY",
        "historical_before_patch": previous,
        "after_patch": current,
        "train_joint_structural_signature_count": 251,
        "train_comm_start_width_pair_count": 145,
        "train_selected_task_counts_by_comm_signature": train_catalog["comm_selected_task_counts"],
        "projected_h1": {
            "anchor_count": 4416,
            "raw_comm_rows": replay["raw_comm_row_count"],
            "projected_comm_rows": replay["projected_comm_row_count"],
            "excluded_comm_row_reasons": replay["excluded_comm_row_reasons"],
            "comm_task_selection_rejections": 0,
            "comm_projection_admitted": replay["comm_projection_admitted_count"],
            "comm_projection_residuals": replay["comm_projection_failures"],
            "full_projected_admitted": replay["full_projected_admitted_count"],
            "full_projected_residuals": replay["full_projected_failures"],
            "full_projected_pass_meaning": replay["full_projected_pass_meaning"],
            "projected_semantic_replay_pass": replay["projected_semantic_replay_pass"],
            "projected_semantic_replay_mismatches": replay["projected_semantic_replay_mismatches"],
            "semantic_comparison": replay["semantic_comparison"],
            "comp_amount_residual_diagnostic": {
                "affected_anchors": comp_delta["affected_anchor_count"],
                "differing_rows": comp_delta["differing_comp_row_count"],
                "absolute_delta_min": comp_delta["absolute_delta_min"],
                "absolute_delta_median": comp_delta["absolute_delta_median"],
                "absolute_delta_max": comp_delta["absolute_delta_max"],
                "relative_delta_max": comp_delta["relative_delta_max"],
                "acceptance_tolerance_changed": False,
            },
        },
        "no_policy_widening_for_residuals": True,
        "formal_raw_modified": False,
        "validation_used_for_support": False,
        "optimizer_implemented": False,
        "candidate_ranking": False,
        "baseline": False,
        "closed_loop": False,
        "gpu": False,
        "training": False,
        "locked_test": False,
        "performance_claim": False,
    }
    write(C / "07_step6_3bc_patch_acceptance.json", patch)
    previous_c = read(C / "05_step6_3c_acceptance.json")
    previous_c["grammar_policy_unchanged"] = False
    previous_c["formal_train_domain_audit"]["empty_domains_observed"] = train["empty_domain_count"]
    previous_c["formal_validation_descriptive_only"]["empty_domains_observed"] = validation["empty_domain_count"]
    previous_c["superseding_patch_receipt"] = "07_step6_3bc_patch_acceptance.json"
    write(C / "05_step6_3c_acceptance.json", previous_c)
    files = sorted(path for path in C.glob("*.json") if path.name != "manifest.json")
    manifest = {
        "step": "STEP_6_3BC_PATCH",
        "status": patch["STEP_6_3BC_PATCH"],
        "files": [{"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "bytes": path.stat().st_size} for path in files],
    }
    write(C / "manifest.json", manifest)
    print(json.dumps({"status": patch["STEP_6_3BC_PATCH"], "before": previous,
                      "after": current, "projected_h1": patch["projected_h1"]}, sort_keys=True))


if __name__ == "__main__":
    main()
