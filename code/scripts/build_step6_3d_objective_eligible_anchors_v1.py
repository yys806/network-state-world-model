"""Replace old cohort-empty anchors using only current-time static facts."""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset
from pi_jwm.step6_3d_objective_anchor_patch_v1 import replace_empty_cohorts
from run_step6_3d_one_cpu_solve_v1 import DATASET, OUT, load_selected, sha

SOURCE = ROOT / "code/artifacts/protocols/pi_jwm_step6_3c_candidate_search_protocol_v1_20260929"
TERMINAL = {"done", "failed", "completed"}


def main() -> None:
    interface = FormalTrainingInterface.from_manifest(DATASET)
    shards = FullFormalShardDataset(interface)
    indexed = {row["metadata"]["sample_id"]: row["metadata"] for row in interface.samples}
    if len(indexed) != len(interface.samples):
        raise ValueError("sample identity not unique")
    old_side = json.loads((OUT / "09_selected_side_state_readiness.json").read_text(encoding="utf-8"))
    old_counts = {row["sample_id"]: int(row["cohort_count"]) for row in old_side["rows"]}
    raw_cache = {}
    eligibility_cache = {}

    def static_cohort_count(sample_id: str) -> int:
        if sample_id in eligibility_cache:
            return eligibility_cache[sample_id]
        meta = indexed[sample_id]
        tid = meta["trajectory_id"]
        if tid not in raw_cache:
            path = ROOT / meta["source_path"]
            if sha(path) != meta["source_sha256"]:
                raise ValueError("Formal Raw identity changed")
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                raw_cache[tid] = json.load(stream)
        decision = raw_cache[tid]["decisions"][int(meta["anchor_decision_frame"])]
        if decision["frame_index"] != meta["anchor_decision_frame"] or decision["trajectory_id"] != tid:
            raise ValueError("Raw decision identity mismatch")
        if not any(str(row["lifecycle"]).lower() not in TERMINAL for row in decision["tasks"]):
            eligibility_cache[sample_id] = 0
            return 0
        _, _, _, state, _, context, domain, exact, _ = load_selected(sample_id, interface, shards)
        if domain.causal_provenance != context.causal_provenance or exact["capture_event_id"] != decision["capture_event_id"]:
            raise ValueError("candidate causal state mismatch")
        tasks = {str(row["task_id"]): row for row in decision["tasks"]}
        count = sum(str(tasks[task_id]["lifecycle"]).lower() not in TERMINAL and
                    bool(state["task_presence"][0, slot]) and
                    not bool(state["task_completed"][0, slot])
                    for task_id, slot in context.static["input_entity_index"]["task"].items())
        eligibility_cache[sample_id] = count
        return count

    outputs = []
    for split, old_name, new_name, source_name, expected in (
        ("dev_train", "01_train_anchor_manifest.json",
         "15_train_anchor_manifest_objective_eligible.json",
         "01_formal_train_domain_feasibility.json", 32),
        ("dev_validation", "02_validation_anchor_manifest.json",
         "16_validation_anchor_manifest_objective_eligible.json",
         "02_formal_validation_domain_descriptive.json", 64),
    ):
        old_path = OUT / old_name
        old = json.loads(old_path.read_text(encoding="utf-8"))
        source_path = SOURCE / source_name
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if old["split"] != source["split"] or old["split"] != split or len(old["selected"]) != expected:
            raise ValueError("split or old selection count mismatch")
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != old["source_sha256"]:
            raise ValueError("static-domain source receipt changed")
        counts = {row["sample_id"]: old_counts[row["sample_id"]] for row in old["selected"]}
        result = replace_empty_cohorts(old, source["anchors"], counts, static_cohort_count)
        if any(row["sample_id"] not in indexed or indexed[row["sample_id"]]["split"] != split
               for row in result["selected"]):
            raise ValueError("replacement crossed Formal split")
        new = dict(old)
        new.update(result)
        new.update({"historical_manifest": str(old_path.relative_to(ROOT)).replace("\\", "/"),
                    "historical_manifest_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
                    "objective_eligibility": "nonempty CandidateDomain and current nonterminal/present/not-completed task cohort",
                    "replacement_uses_rollout_or_search_outcome": False,
                    "sidecar_alignment_pending": True})
        (OUT / new_name).write_text(json.dumps(new, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                                    encoding="utf-8", newline="\n")
        outputs.append({"split": split, "old_count": expected,
                        "new_count": len(result["selected"]),
                        "replacements": result["replacements"]})
    print(json.dumps(outputs, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
