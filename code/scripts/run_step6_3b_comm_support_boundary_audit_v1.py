"""Read-only TRAIN action-index audit for the STEP 6.3B grammar boundary.

This measures support identities; it does not generate candidates or run a model.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
from run_step6_3a_candidate_support_audit_v1 import DATASET, read_trajectory, structural_signature


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _comm_summary(step: dict) -> tuple[int, bool]:
    rows = step["action"]["comm"]["entries"]
    task_counts = collections.Counter(str(row["task_id"]) for row in rows)
    return len(rows), any(count > 1 for count in task_counts.values())


def main() -> None:
    manifest_path = DATASET / "formal_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads((DATASET / manifest["provenance"]["split_manifest"]).read_text(encoding="utf-8"))
    index_path = DATASET / "packages/samples/index.json.gz"
    if index_path.exists():
        with gzip.open(index_path, "rt", encoding="utf-8") as stream:
            index = json.load(stream)
    else:
        index = json.loads((DATASET / "packages/samples/index.json").read_text(encoding="utf-8"))
    by_trajectory: dict[str, list[dict]] = collections.defaultdict(list)
    for sample in index:
        if sample["metadata"]["split"] == "dev_train":
            by_trajectory[str(sample["metadata"]["trajectory_id"])].append(sample)

    raw_slots = h1_windows = future_window_positions = 0
    raw_duplicate_slots = h1_duplicate_windows = future_duplicate_positions = 0
    raw_row_counts: collections.Counter[int] = collections.Counter()
    policy_noop_with_rows: collections.Counter[str] = collections.Counter()
    eligible_with_empty_action: collections.Counter[str] = collections.Counter()
    raw_structures: set[tuple[str, str, str]] = set()
    h1_structures: set[tuple[str, str, str]] = set()
    future_structures: set[tuple[str, str, str]] = set()
    future_joint_counts: collections.Counter[tuple[str, str, str]] = collections.Counter()
    comm_block_pair_counts: collections.Counter[tuple[int, int]] = collections.Counter()
    comm_selected_task_counts: dict[str, set[int]] = collections.defaultdict(set)
    temporal_counts: dict[int, collections.Counter[tuple[tuple[str, str, str], ...]]] = {
        h: collections.Counter() for h in range(1, 5)
    }
    unique_future_frames = 0
    for tid in split["dev_train"]:
        raw = read_trajectory(tid)
        decisions, steps = raw["decisions"], raw["steps"]
        duration = float(raw["environment"]["slot_duration_s"])
        structures = [structural_signature(decision, step, duration)
                      for decision, step in zip(decisions, steps)]
        seen_frames: set[int] = set()
        for frame, step in enumerate(steps):
            row_count, duplicate = _comm_summary(step)
            raw_slots += 1
            raw_duplicate_slots += int(duplicate)
            raw_row_counts[row_count] += 1
            raw_structures.add(structures[frame])
            for family in ("comm", "comp", "mobility"):
                audit = step["behavior_policy_audit"]["families"][family]
                action_rows = step["action"][family]["entries"]
                policy_noop_with_rows[family] += int(bool(audit["no_op"]) and bool(action_rows))
                eligible_with_empty_action[family] += int(bool(audit["eligible"]) and not action_rows)
        for sample in by_trajectory[tid]:
            frames = list(map(int, sample["metadata"]["future_action_frame_indices"]))
            assert len(frames) == 4
            h1_windows += 1
            h1_duplicate_windows += int(_comm_summary(steps[frames[0]])[1])
            h1_structures.add(structures[frames[0]])
            for h in range(1, 5):
                temporal_counts[h][tuple(structures[frame] for frame in frames[:h])] += 1
            for frame in frames:
                future_window_positions += 1
                future_duplicate_positions += int(_comm_summary(steps[frame])[1])
                future_structures.add(structures[frame])
                future_joint_counts[structures[frame]] += 1
                comm_signature = structures[frame][0]
                comm_selected_task_counts[comm_signature].add(
                    len({str(row["task_id"]) for row in steps[frame]["action"]["comm"]["entries"]}))
                n_rb = int(decisions[frame]["n_rb"])
                for row in steps[frame]["action"]["comm"]["entries"]:
                    rb_set = set(map(int, row["rb_indices"]))
                    starts = [start for start in range(n_rb)
                              if {(start + offset) % n_rb for offset in range(len(rb_set))} == rb_set]
                    assert len(starts) == 1
                    comm_block_pair_counts[(starts[0], len(rb_set))] += 1
                seen_frames.add(frame)
        unique_future_frames += len(seen_frames)

    result = {
        "step": "STEP_6_3B_PREDECISION_EVIDENCE",
        "source": "FORMAL_TRAIN_ONLY",
        "dataset_manifest_sha256": _digest(manifest_path),
        "raw_trajectory_decision_slots": raw_slots,
        "raw_comm_row_count_per_slot": dict(sorted(raw_row_counts.items())),
        "raw_slots_with_same_task_multiple_comm_rows": raw_duplicate_slots,
        "formal_h1_windows": h1_windows,
        "formal_h1_windows_with_same_task_multiple_comm_rows": h1_duplicate_windows,
        "formal_h1_h4_window_positions": future_window_positions,
        "formal_h1_h4_positions_with_same_task_multiple_comm_rows": future_duplicate_positions,
        "unique_train_future_action_frames": unique_future_frames,
        "raw_trajectory_structural_joint_count": len(raw_structures),
        "formal_h1_structural_joint_count": len(h1_structures),
        "formal_h1_h4_union_structural_joint_count": len(future_structures),
        "raw_only_structural_joint_count": len(raw_structures - future_structures),
        "h1_h4_only_structural_joint_count": len(future_structures - raw_structures),
        "formal_future_comm_start_width_pair_count": len(comm_block_pair_counts),
        "comm_selected_task_counts": {
            key: sorted(values) for key, values in sorted(comm_selected_task_counts.items())
        },
        "policy_noop_with_nonempty_action_count": dict(policy_noop_with_rows),
        "eligible_with_empty_action_count": dict(eligible_with_empty_action),
        "candidate_validator_duplicate_task_rule_before_6_3b_fix": "step6_0a_candidate_generation_v1.validate_step rejected a second Comm row with the same task_id",
        "scientific_interpretation": "Policy no-op flags are distinct from empty action families; formal future-action positions define the train structural catalog.",
        "future_target_used": False,
        "validation_used": False,
        "locked_test": False,
        "gpu": False,
        "training": False,
        "formal_dataset_modified": False,
        "checkpoint_modified": False,
    }
    out = ROOT / "code/artifacts/protocols/pi_jwm_step6_3b_candidate_grammar_v1_20260929"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "01_train_comm_support_scope_and_duplicate_rows.json"
    path.write_bytes((json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    catalog = {
        "schema_version": "PI-JWM-STEP-6.3B-TRAIN-STRUCTURAL-SUPPORT-V1",
        "source": "FORMAL_TRAIN_ONLY",
        "dataset_manifest_sha256": result["dataset_manifest_sha256"],
        "joint_structural_signatures": [
            {"signature": list(signature), "count_over_formal_window_positions": count}
            for signature, count in sorted(future_joint_counts.items())
        ],
        "comm_start_width_pairs": [
            {"start": start, "width": width, "count_over_formal_window_positions": count}
            for (start, width), count in sorted(comm_block_pair_counts.items())
        ],
        "comm_selected_task_counts": {
            key: sorted(values) for key, values in sorted(comm_selected_task_counts.items())
        },
        "temporal_structural_prefixes": {
            str(h): [{"sequence": [list(signature) for signature in sequence], "count": count}
                     for sequence, count in sorted(temporal_counts[h].items())]
            for h in range(1, 5)
        },
        "no_optimizer": True,
        "validation_used": False,
        "future_target_used": False,
    }
    (out / "02_train_structural_support_catalog.json").write_bytes(
        (json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    # Descriptive validation check follows the completed TRAIN catalog.
    validation = collections.Counter()
    train_joint = set(future_structures)
    train_pairs = set(comm_block_pair_counts)
    for tid in split["dev_validation"]:
        raw = read_trajectory(tid)
        duration = float(raw["environment"]["slot_duration_s"])
        for decision, step in zip(raw["decisions"], raw["steps"]):
            validation["raw_slots"] += 1
            signature = structural_signature(decision, step, duration)
            validation["joint_structural_observed_in_train"] += int(signature in train_joint)
            validation["joint_structural_unseen_in_train"] += int(signature not in train_joint)
            n_rb = int(decision["n_rb"])
            for row in step["action"]["comm"]["entries"]:
                rb = set(map(int, row["rb_indices"]))
                starts = [s for s in range(n_rb)
                          if {(s + i) % n_rb for i in range(len(rb))} == rb]
                assert len(starts) == 1
                validation["comm_rows"] += 1
                validation["comm_pair_unseen_in_train"] += int((starts[0], len(rb)) not in train_pairs)
    (out / "03_validation_descriptive_only.json").write_bytes(
        (json.dumps({"source": "FORMAL_VALIDATION_DESCRIPTIVE_ONLY",
                    "train_catalog_frozen_before_validation_scan": True,
                    "validation_used_for_template_or_policy_selection": False,
                    "counts": dict(sorted(validation.items())),
                    "future_target_used": False, "locked_test": False},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"receipt": str(path.relative_to(ROOT)),
                      "raw_duplicate_slots": raw_duplicate_slots,
                      "h1_duplicate_windows": h1_duplicate_windows,
                      "raw_structures": len(raw_structures),
                      "formal_future_structures": len(future_structures)}))


if __name__ == "__main__":
    main()
