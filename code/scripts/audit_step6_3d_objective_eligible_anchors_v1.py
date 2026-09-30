"""Close static CandidateDomain and frozen scorer cohort eligibility for 32+64."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset
from pi_jwm.step6_3b_candidate_support_v1 import TrainStructuralSupportCatalog
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from run_step6_3d_one_cpu_solve_v1 import CATALOG, DATASET, OUT, load_selected, objective_side


def read(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main() -> None:
    manifests = [read("15_train_anchor_manifest_objective_eligible.json"),
                 read("16_validation_anchor_manifest_objective_eligible.json")]
    prior = read("09_selected_side_state_readiness.json")
    old_counts = {row["sample_id"]: int(row["cohort_count"]) for row in prior["rows"]}
    old_side = read("03_selected_deadline_sidecars.json")
    merged = read("19_formal_selected_deadline_sidecars.json")
    replacements = {row["new_sample_id"] for manifest in manifests
                    for row in manifest["replacements"]}
    if len(replacements) != 8:
        raise ValueError("replacement count changed")
    interface = FormalTrainingInterface.from_manifest(DATASET)
    shards = FullFormalShardDataset(interface)
    catalog = TrainStructuralSupportCatalog.from_json(CATALOG)
    rows = []
    for manifest in manifests:
        for selected in manifest["selected"]:
            sample_id = selected["sample_id"]
            if int(selected["concrete_count"]) <= 0 or sample_id not in merged:
                raise ValueError("selected anchor lacks static domain or aligned sidecar")
            if sample_id in replacements:
                sample, _, _, state, _, context, operational, decision, meta = load_selected(
                    sample_id, interface, shards)
                side = objective_side(sample, state, context, decision, merged[sample_id])
                cohort = sum(not task.terminal_at_anchor and
                             bool(state["task_presence"][0, task.task_index]) and
                             not bool(state["task_completed"][0, task.task_index])
                             for task in side.tasks)
                domain = CandidateDomain.from_state(context, operational, state,
                    operational.mobility_states, catalog)
                if domain.is_empty or domain.exact_unique_single_step_count != selected["concrete_count"]:
                    raise ValueError("replacement static CandidateDomain differs from source receipt")
                source = "new_exact_replay_and_frozen_scorer_side_state"
            else:
                if merged[sample_id] != old_side[sample_id]:
                    raise ValueError("retained sidecar changed")
                cohort = old_counts[sample_id]
                source = "accepted_prior_side_state_and_exact_replay"
            if cohort <= 0 or not merged[sample_id]["alignment_passed"]:
                raise ValueError("formal selection contains Objective-ineligible anchor")
            rows.append({"sample_id": sample_id, "split": manifest["split"],
                         "stratum": selected["stratum"],
                         "static_candidate_count": int(selected["concrete_count"]),
                         "cohort_count": cohort, "evidence_source": source})
    if len(rows) != 96 or len({row["sample_id"] for row in rows}) != 96:
        raise ValueError("formal selection count/uniqueness mismatch")
    result = {
        "verdict": "OBJECTIVE_ANCHOR_ELIGIBILITY_PASS",
        "train_count": sum(row["split"] == "dev_train" for row in rows),
        "validation_count": sum(row["split"] == "dev_validation" for row in rows),
        "cohort_zero_count": 0,
        "static_empty_count": 0,
        "replacement_count": len(replacements),
        "selection_uses_rollout_or_search_outcome": False,
        "validation_search_executed": False,
        "locked_test": False,
        "rows": rows,
        "source_sha256": {
            name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
            for name in ("15_train_anchor_manifest_objective_eligible.json",
                         "16_validation_anchor_manifest_objective_eligible.json",
                         "17_replacement_deadline_sidecars.json",
                         "18_replacement_deadline_alignment_receipt.json",
                         "19_formal_selected_deadline_sidecars.json")},
    }
    (OUT / "20_objective_anchor_eligibility_receipt.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({key: result[key] for key in
                      ("verdict", "train_count", "validation_count", "cohort_zero_count")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
