"""Verify bounded STEP 6.3D preflight receipts without changing research policy."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929"
FILES = (
    "15_train_anchor_manifest_objective_eligible.json",
    "16_validation_anchor_manifest_objective_eligible.json",
    "17_replacement_deadline_sidecars.json",
    "18_replacement_deadline_alignment_receipt.json",
    "19_formal_selected_deadline_sidecars.json",
    "20_objective_anchor_eligibility_receipt.json",
    "21_batch_one_step_cpu_gate.json",
    "22_train_h4_scoreability_preflight.json",
    "23_h4_batch_equivalence_cpu_gate.json",
)
CODE_FILES = (
    "code/src/pi_jwm/step6_3d_objective_anchor_patch_v1.py",
    "code/src/pi_jwm/step6_1_trained_candidate_rollout_v1.py",
    "code/src/pi_jwm/step6_3d_fixed_budget_search_v1.py",
    "code/scripts/build_step6_3d_objective_eligible_anchors_v1.py",
    "code/scripts/replay_step6_3d_replacement_deadline_sidecars_v1.py",
    "code/scripts/audit_step6_3d_objective_eligible_anchors_v1.py",
    "code/scripts/run_step6_3d_one_cpu_solve_v1.py",
    "code/scripts/run_step6_3d_train_h4_scoreability_preflight_v1.py",
    "code/scripts/run_step6_3d_batch_transition_cpu_gate_v1.py",
    "code/scripts/run_step6_3d_h4_batch_equivalence_v1.py",
    "code/scripts/build_step6_3d_preflight_patch_acceptance_v1.py",
)


def read(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def identity(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    train, validation = read(FILES[0]), read(FILES[1])
    replacement_sides = read(FILES[2])
    alignment, selected, eligibility = read(FILES[3]), read(FILES[4]), read(FILES[5])
    one_step, h4, four_step = read(FILES[6]), read(FILES[7]), read(FILES[8])
    if len(train["selected"]) != 32 or len(validation["selected"]) != 64:
        raise ValueError("corrected manifest sizes mismatch")
    replacement_ids = {row["new_sample_id"] for manifest in (train, validation)
                       for row in manifest["replacements"]}
    if len(replacement_ids) != 8 or set(replacement_sides) != replacement_ids or \
            alignment["anchor_count"] != 8 or not alignment["alignment_passed"]:
        raise ValueError("replacement deadline replay incomplete")
    if len(selected) != 96 or eligibility["cohort_zero_count"] != 0 or \
            eligibility["static_empty_count"] != 0 or eligibility["replacement_count"] != 8 or \
            eligibility["verdict"] != "OBJECTIVE_ANCHOR_ELIGIBILITY_PASS" or \
            eligibility["selection_uses_rollout_or_search_outcome"]:
        raise ValueError("Objective eligibility incomplete")
    if not all(row["alignment_passed"] for row in selected.values()):
        raise ValueError("selected deadline sidecar alignment incomplete")
    if not all(row["one_step_equivalence_pass"] for row in one_step["equivalence"]) or \
            not all(row["B_WM_accounting_pass"] for row in one_step["throughput"]) or \
            len(one_step["throughput"]) != 4 or four_step["verdict"] != "PASS" or \
            not all(row["pass"] for row in four_step["rows"]):
        raise ValueError("serial/batch equivalence or throughput gate failed")
    if h4["selected_train_anchor_count"] != 32 or len(h4["rows"]) != 32 or \
            h4["diagnostic_seed"] != 6391 or h4["B_WM_per_anchor"] != 64 or \
            h4["actual_unique_transitions"] != 2048 or \
            {row["sample_id"] for row in h4["rows"]} != \
            {row["sample_id"] for row in train["selected"]}:
        raise ValueError("TRAIN-only H4 preflight incomplete")
    if h4["anchors_with_h4_scoreable_candidate"] + \
            h4["anchors_without_h4_scoreable_candidate"] != 32:
        raise ValueError("H4 anchor accounting mismatch")
    if h4["validation_search_executed"] or h4["optimizer_comparison"] or \
            one_step["validation_search_executed"] or four_step["validation_search_executed"]:
        raise ValueError("formal comparison was executed in preflight")
    reasons = Counter()
    anchor_reasons = Counter()
    for row in h4["rows"]:
        categories = set()
        detailed = any(key.startswith("H4_SUPPORT_BOUNDARY:") or
                       key.startswith("ScorerStateInconsistency:") or
                       key.startswith("BurdenSemanticsBlocked:")
                       for key in row["scorer_residual_reasons"])
        for reason, count in row["scorer_residual_reasons"].items():
            if reason.startswith("H4_UNSCOREABLE:") and detailed:
                continue
            category = ("UNSUPPORTED_FUTURE_RETURN_BIRTH" if
                        "UNSUPPORTED_FUTURE_RETURN_BIRTH" in reason else
                        "OBJECTIVE_UNSCOREABLE_EMPTY_COHORT" if
                        "OBJECTIVE_UNSCOREABLE_EMPTY_COHORT" in reason else
                        "SCORER_INCONSISTENCY" if
                        "ScorerStateInconsistency" in reason else "OTHER_SCORER_OR_SUPPORT")
            reasons[category] += count
            categories.add(category)
        if row["grammar_dead_end_count"]:
            reasons["GRAMMAR_DEAD_END"] += row["grammar_dead_end_count"]
            categories.add("GRAMMAR_DEAD_END")
        anchor_reasons.update(categories)
    success = h4["anchors_with_h4_scoreable_candidate"]
    readiness = ("SUPPORTED" if success > 16 else
                 "PENDING_RESEARCHER_DECISION_ON_MODEL_OBJECTIVE_SUPPORT")
    tracked = [OUT / name for name in FILES]
    tracked += [ROOT / row["result_file"] for row in h4["rows"]]
    tracked += [ROOT / name for name in CODE_FILES]
    entries = [identity(path) for path in tracked]
    if len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("duplicate evidence path")
    manifest = {"step": "STEP_6_3D_PREFLIGHT_PATCH", "files": entries,
                "locked_test": False, "gpu": False,
                "formal_train_tuning": False, "validation_comparison": False}
    manifest_path = OUT / "24_preflight_patch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    receipt = {
        "STEP_6_3D_PREFLIGHT_PATCH": "PASS",
        "H4_SEARCH_COMPARISON_READINESS": readiness,
        "historical_train_selected": 32, "historical_validation_selected": 64,
        "corrected_train_selected": 32, "corrected_validation_selected": 64,
        "train_replacements": len(train["replacements"]),
        "validation_replacements": len(validation["replacements"]),
        "selected_cohort_zero_count": 0, "selected_static_empty_count": 0,
        "train_anchors_h4_scoreable": success,
        "train_anchors_h4_unscoreable": 32 - success,
        "support_reason_event_counts": dict(sorted(reasons.items())),
        "support_reason_anchor_counts": dict(sorted(anchor_reasons.items())),
        "serial_batch_equivalence": "PASS",
        "recommended_cpu_batch_size": max(one_step["throughput"],
            key=lambda row: row["transitions_per_second"])["batch_size"],
        "gpu_batch_probe": "NOT_AVAILABLE",
        "manifest": identity(manifest_path),
        "formal_train_tuning": False, "validation_comparison": False,
        "search_method_selected": False, "locked_test": False,
    }
    (OUT / "25_preflight_patch_acceptance.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({key: receipt[key] for key in
        ("STEP_6_3D_PREFLIGHT_PATCH", "H4_SEARCH_COMPARISON_READINESS",
         "train_anchors_h4_scoreable", "train_anchors_h4_unscoreable",
         "recommended_cpu_batch_size")}, sort_keys=True))


if __name__ == "__main__":
    main()
