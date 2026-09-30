"""Read-only decision-before-action replay for selected Formal anchors.

No Formal Raw/package or World Model file is written. Fail closed when any
selected decision or its preceding action prefix differs from frozen Raw.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/scripts"), str(ROOT / "code/src")]
import collect_step5_5_formal_raw_v1 as collector
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface

OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929"
DATASET = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/formal_dataset_manifest.json"
PATTERN = re.compile(r"formal-v1-sim-(\d+)-policy-(\d+)")


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def replay_selected(*, manifest_names=("01_train_anchor_manifest.json",
                                 "02_validation_anchor_manifest.json"),
                    selected_ids=None, expected_count=96,
                    sidecar_name="03_selected_deadline_sidecars.json",
                    evidence_name="04_selected_deadline_alignment_receipt.json") -> None:
    if len(manifest_names) != 2:
        raise ValueError("TRAIN and Validation manifest pair required")
    interface = FormalTrainingInterface.from_manifest(DATASET)
    indexed = {row["metadata"]["sample_id"]: row["metadata"] for row in interface.samples}
    selected = defaultdict(list)
    for name, split in zip(manifest_names, ("dev_train", "dev_validation")):
        manifest = json.loads((OUT / name).read_text(encoding="utf-8"))
        if manifest["split"] != split:
            raise ValueError("selected manifest split mismatch")
        for row in manifest["selected"]:
            if selected_ids is not None and row["sample_id"] not in selected_ids:
                continue
            meta = indexed[row["sample_id"]]
            if meta["split"] != split:
                raise ValueError("selected sample crosses split")
            selected[meta["trajectory_id"]].append(meta)
    sidecars = {}
    evidence = []
    for trajectory_id, metas in sorted(selected.items()):
        match = PATTERN.fullmatch(trajectory_id)
        if match is None:
            raise ValueError("trajectory seed identity unresolved")
        raw_path = ROOT / metas[0]["source_path"]
        raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        if any(meta["source_sha256"] != raw_sha for meta in metas):
            raise ValueError("Formal Raw identity mismatch")
        with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
            original = json.load(stream)
        target_frames = {int(meta["anchor_decision_frame"]) for meta in metas}
        captured = {}
        old_capture = collector.step23._capture

        def capture_with_deadline(env, *args, **kwargs):
            decision = old_capture(env, *args, **kwargs)
            frame = int(kwargs.get("frame", -1))
            if kwargs.get("phase") == "loop_start_decision" and frame in target_frames:
                runtime = collector.step23._all_runtime_tasks(env)
                task_ids = {str(row["task_id"]) for row in decision["tasks"]}
                captured[frame] = [{"task_id": task_id,
                                    "arrival_time_s": float(task.getTaskArrivalTime()),
                                    "deadline_s": float(task.getTaskDeadline())}
                                   for task_id, task in sorted(runtime.items()) if task_id in task_ids]
            return decision

        collector.step23._capture = capture_with_deadline
        try:
            replay = collector.collect_trajectory(int(match.group(1)), int(match.group(2)), trajectory_id)
        finally:
            collector.step23._capture = old_capture
        if original["environment"]["config_hash"] != replay["environment"]["config_hash"]:
            raise ValueError(f"environment config hash differs: {trajectory_id}")
        for meta in metas:
            frame = int(meta["anchor_decision_frame"])
            old = original["decisions"][frame]
            new = replay["decisions"][frame]
            rows = captured.get(frame)
            if canonical(old) != canonical(new):
                raise ValueError(f"current decision differs on replay: {meta['sample_id']}")
            if canonical([step["action"] for step in original["steps"][:frame]]) != canonical(
                    [step["action"] for step in replay["steps"][:frame]]):
                raise ValueError(f"causal action prefix differs: {meta['sample_id']}")
            if rows is None or {row["task_id"] for row in rows} != {row["task_id"] for row in old["tasks"]}:
                raise ValueError(f"deadline task IDs differ: {meta['sample_id']}")
            sidecars[meta["sample_id"]] = {
                "trajectory_id": trajectory_id, "frame_index": frame,
                "capture_event_id": old["capture_event_id"],
                "simulation_time_s": old["simulation_time_s"],
                "source_time_s": old["simulation_time_s"], "alignment_passed": True,
                "tasks": rows, "source": "live Task.getTaskDeadline at decision-before-action"}
            evidence.append({"sample_id": meta["sample_id"], "raw_sha256": raw_sha,
                             "decision_equal": True, "action_prefix_equal": True,
                             "deadline_task_ids_equal": True})
        print(f"aligned {trajectory_id}: {len(metas)} selected anchors", flush=True)
    if len(sidecars) != expected_count:
        raise ValueError("selected sidecar count differs from expected")
    for name, value in ((sidecar_name, sidecars),
                        (evidence_name, {
                            "anchor_count": len(sidecars), "trajectory_count": len(selected),
                            "checks": evidence, "alignment_passed": True,
                            "source": "read-only exact replay of frozen Formal Raw",
                            "future_target_used": False, "locked_test": False})):
        (OUT / name).write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8", newline="\n")


def main() -> None:
    replay_selected()


if __name__ == "__main__":
    main()
