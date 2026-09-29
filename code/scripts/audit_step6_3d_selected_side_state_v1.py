"""Check every selected anchor can form the frozen 6.2B causal side-state."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts")]
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalShardDataset
from run_step6_3d_one_cpu_solve_v1 import DATASET, OUT, load_selected, objective_side


def main() -> None:
    interface = FormalTrainingInterface.from_manifest(DATASET)
    shards = FullFormalShardDataset(interface)
    sidecars = json.loads((OUT / "03_selected_deadline_sidecars.json").read_text(encoding="utf-8"))
    rows = []
    for name in ("01_train_anchor_manifest.json", "02_validation_anchor_manifest.json"):
        manifest = json.loads((OUT / name).read_text(encoding="utf-8"))
        for selected in manifest["selected"]:
            sample_id = selected["sample_id"]
            try:
                sample, _, _, state, _, context, _, decision, meta = load_selected(
                    sample_id, interface, shards)
                side = objective_side(sample, state, context, decision, sidecars[sample_id])
                rows.append({"sample_id": sample_id, "split": meta["split"],
                             "status": "READY", "task_count": len(side.tasks),
                             "effort_component_mask": side.effort_component_mask,
                             "cohort_count": sum(not task.terminal_at_anchor for task in side.tasks)})
            except (ValueError, KeyError, IndexError) as exc:
                rows.append({"sample_id": sample_id, "status": "REJECTED",
                             "reason": type(exc).__name__ + ":" + str(exc)})
        print(f"{name}: side-state checked", flush=True)
    result = {"selected_anchor_count": len(rows),
              "ready_count": sum(row["status"] == "READY" for row in rows),
              "rejected_count": sum(row["status"] == "REJECTED" for row in rows),
              "rows": rows, "future_target_used": False, "locked_test": False,
              "gpu": False, "training": False}
    (OUT / "09_selected_side_state_readiness.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({key: result[key] for key in ("selected_anchor_count", "ready_count", "rejected_count")}))


if __name__ == "__main__":
    main()
