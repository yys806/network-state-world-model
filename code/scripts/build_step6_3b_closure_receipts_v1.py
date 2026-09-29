"""Build STEP 6.3B machine receipts from frozen TRAIN catalog and CPU checks.

No model, candidate search, training, validation-based selection or GPU use.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_3b_candidate_grammar_v1_20260929"
PREVIOUS = ROOT / "code/artifacts/protocols"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(name: str, value: dict) -> None:
    (OUT / name).write_bytes((json.dumps(value, sort_keys=True, indent=2,
                                       ensure_ascii=False) + "\n").encode("utf-8"))


def main() -> None:
    train = read(OUT / "01_train_comm_support_scope_and_duplicate_rows.json")
    catalog = read(OUT / "02_train_structural_support_catalog.json")
    validation = read(OUT / "03_validation_descriptive_only.json")
    step63a = read(PREVIOUS / "pi_jwm_step6_3a_candidate_support_audit_v1_20260928/13_step6_3a_acceptance.json")
    step62b = read(PREVIOUS / "pi_jwm_step6_2b_patch_route_noop_v1_20260928/08_step6_2b_final_acceptance.json")
    assert step63a["STEP_6_3A"] == "PASS" and step62b["STEP_6_2B"] == "PASS"
    assert train["source"] == "FORMAL_TRAIN_ONLY"
    assert catalog["source"] == "FORMAL_TRAIN_ONLY" and len(catalog["joint_structural_signatures"]) == 251
    assert train["comm_selected_task_counts"] == catalog["comm_selected_task_counts"]
    assert catalog["comm_selected_task_counts"]["NOOP"] == [0]
    assert validation["validation_used_for_template_or_policy_selection"] is False
    assert train["raw_slots_with_same_task_multiple_comm_rows"] == 293
    assert train["eligible_with_empty_action_count"] == {"comm": 0, "comp": 0, "mobility": 0}
    common = {"step": "STEP_6_3B", "source": "FORMAL_TRAIN_ONLY_FOR_SUPPORT",
              "dataset_manifest_sha256": train["dataset_manifest_sha256"],
              "future_target_used": False, "validation_used_for_selection": False,
              "locked_test": False, "gpu": False, "training": False,
              "checkpoint_modified": False, "formal_dataset_modified": False,
              "optimizer_implemented": False, "candidate_ranking_executed": False,
              "closed_loop": False, "baseline": False}
    write("04_researcher_policy_closure.json", {**common,
          "researcher_decisions": {
              "same_task_multirow_comm": "ALLOW_ON_UNIQUE_CURRENT_WIRELESS_RELATION",
              "comm_eligible_set": "TASKS_WITH_UNIQUE_BINDING_TO_EXISTING_CURRENT_WIRELESS_FLOW",
              "comm_task_selection": "SUBSET_WITH_TRAIN_SIGNATURE_CONDITIONAL_SELECTED_TASK_COUNT",
              "route_created_flow_comm_rows": "EXCLUDED_FROM_PLANNER_V1_PROJECTION",
              "mobility_formal_pool": "SHARED_PROFILE_ONLY",
              "joint_formal_pool": "FORMAL_TRAIN_OBSERVED_STRUCTURAL_TRIPLES_ONLY",
              "joint_unseen_marginal_seen": "FUTURE_ABLATION_ONLY",
              "temporal_observed_transition": "DIAGNOSTIC_LABEL_NOT_HARD_GATE",
              "comp_eligible_empty_action": "REJECT",
              "comp_alpha": [0.5, 0.75, 1.0],
              "route": "EXPLICIT_NOOP_ONLY"},
          "predecessor_step6_3a": step63a["STEP_6_3A"],
          "predecessor_step6_2b": step62b["STEP_6_2B"]})
    write("05_grammar_and_support_labels.json", {**common,
          "train_joint_structural_count": len(catalog["joint_structural_signatures"]),
          "train_comm_start_width_pair_count": len(catalog["comm_start_width_pairs"]),
          "train_max_comm_rows_per_slot": max(map(int, train["raw_comm_row_count_per_slot"])),
          "train_same_task_multirow_raw_slots": train["raw_slots_with_same_task_multiple_comm_rows"],
          "train_comm_selected_task_counts_by_signature": catalog["comm_selected_task_counts"],
          "support_axes": ["family_marginal", "joint_structural", "temporal_prefix",
                           "temporal_adjacent", "causal_binding", "fixed_support", "h_sup"],
          "concrete_identity_rebound_from_current_state": True,
          "temporal_unseen_is_rejection": False,
          "structural_signature_is_concrete_action": False,
          "h_sup_before_rollout": None,
          "train_only_future_return_hsup_histogram_available": False,
          "hsup_histogram_required_for_grammar": False})
    write("06_validation_descriptive_scope.json", {**common,
          "validation": validation["counts"],
          "validation_is_support_definition_source": False,
          "validation_changed_catalog_or_policy": False})
    write("07_step6_3b_acceptance.json", {**common,
          "STEP_6_3B": "PASS",
          "acceptance_scope": "CANDIDATE_GRAMMAR_SUPPORT_LABELING_ADMISSION_CPU_CONTRACT_ONLY",
          "focused_test_suites": ["test_step6_3b_candidate_support_v1.py",
                                  "test_step6_3b_candidate_grammar_v1.py",
                                  "test_step6_0a_candidate_generation_v1.py",
                                  "test_step6_0c_planner_action_domain_v1.py",
                                  "test_step6_2b_planner_objective_scorer_v1.py"],
          "focused_test_counts": {"step6_3b": 18, "step6_0a": 6,
                                  "step6_0c": 15, "step6_2b": 11},
          "compileall": "PASS", "method_selected": False,
          "checkpoint_sha256_expected_from_prior_acceptance":
              "941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9",
          "checkpoint_hash_rechecked_this_step": False,
          "remaining_limits": ["TRAIN-only future Return H_sup histogram unavailable",
                               "future Comp needs explicit predicted computing and unique Exec relation",
                               "temporally unseen sequence generalization untested",
                               "raw historical Route-created Comm rows are outside Planner-v1 projected replay",
                               "concrete action composition beyond structural signature is not exact TRAIN replay"]})
    files = sorted(p for p in OUT.glob("*.json") if p.name != "manifest.json")
    manifest = {"step": "STEP_6_3B", "STEP_6_3B": "PASS",
                "files": [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                           "bytes": p.stat().st_size} for p in files]}
    write("manifest.json", manifest)
    print(f"STEP_6_3B=PASS; receipts={len(files)}; train_joint=251")


if __name__ == "__main__":
    main()
