"""Replay one accepted non-locked Formal Validation trajectory without writing Raw."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code" / "scripts"), str(ROOT / "code" / "src")]
import collect_step5_5_formal_raw_v1 as collector


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main():
    path = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/raw/formal-v1-sim-2026092302-policy-2026092402.json.gz"
    output = ROOT / "code/artifacts/protocols/pi_jwm_step6_2a_patch_objective_readiness_v1_20260928"
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        original = json.load(handle)
    anchor_frame = 1
    sidecar_rows = []
    capture = collector.step23._capture
    def capture_with_deadline(env, *args, **kwargs):
        row = capture(env, *args, **kwargs)
        if kwargs.get("phase") == "loop_start_decision" and kwargs.get("frame") == anchor_frame:
            current = collector.step23._all_runtime_tasks(env)
            sidecar_rows.extend({"task_id": task_id,
                                 "arrival_time_s": float(task.getTaskArrivalTime()),
                                 "deadline_s": float(task.getTaskDeadline())}
                                for task_id, task in sorted(current.items())
                                if task_id in {str(x["task_id"]) for x in row["tasks"]})
        return row
    collector.step23._capture = capture_with_deadline
    try:
        replay = collector.collect_trajectory(2026092302, 2026092402,
            "formal-v1-sim-2026092302-policy-2026092402")
    finally:
        collector.step23._capture = capture
    old, new = original["decisions"][anchor_frame], replay["decisions"][anchor_frame]
    comparisons = {
        "trajectory_id": old["trajectory_id"] == new["trajectory_id"],
        "frame": old["frame_index"] == new["frame_index"],
        "simulation_time": old["simulation_time_s"] == new["simulation_time_s"],
        "capture_event_id": old["capture_event_id"] == new["capture_event_id"],
        "task_rows": canonical(old["tasks"]) == canonical(new["tasks"]),
        "entity_rows": canonical(old["entities"]) == canonical(new["entities"]),
        "physical_state": canonical({k: old[k] for k in ("entities", "channel_rows", "node_cpu_capacity_observation_rows")}) == canonical({k: new[k] for k in ("entities", "channel_rows", "node_cpu_capacity_observation_rows")}),
        "action_prefix": canonical(original["steps"][0]["action"]) == canonical(replay["steps"][0]["action"]),
        "current_task_ids": {x["task_id"] for x in old["tasks"]} == {x["task_id"] for x in sidecar_rows},
    }
    # Flow state may live in the additive ledger amendment rather than base Raw.
    for key in ("flow_ledger", "flow_ledger_history", "carrying_states"):
        if key in old or key in new:
            comparisons[key] = canonical(old.get(key)) == canonical(new.get(key))
    passed = all(comparisons.values())
    receipt = {"status": "PASS" if passed else "BLOCKED", "anchor_frame": anchor_frame,
               "raw_path": str(path.relative_to(ROOT)).replace("\\", "/"),
               "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "simulator_seed": 2026092302, "behavior_seed": 2026092402,
               "config_hash_equal": original["environment"]["config_hash"] == replay["environment"]["config_hash"],
               "checks": comparisons, "source": "live Task at decision-before-action",
               "future_target_used": False, "locked_test_accessed": False,
               "formal_dataset_identity_unchanged": True, "checkpoint_identity_unchanged": True,
               "training_tensor_unchanged": True}
    receipt["alignment_passed"] = passed and receipt["config_hash_equal"]
    (output / "04_task_side_state_source_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    sidecar = {"trajectory_id": old["trajectory_id"], "frame_index": old["frame_index"],
               "capture_event_id": old["capture_event_id"], "simulation_time_s": old["simulation_time_s"],
               "source_time_s": old["simulation_time_s"], "alignment_passed": receipt["alignment_passed"],
               "tasks": sidecar_rows, "source": "live Task.getTaskDeadline at decision-before-action"}
    (output / "deadline_sidecar_anchor_0001.json").write_text(json.dumps(sidecar, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"alignment_passed": receipt["alignment_passed"], "checks": comparisons,
                      "task_count": len(sidecar_rows)}, ensure_ascii=False))
    return 0 if receipt["alignment_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
