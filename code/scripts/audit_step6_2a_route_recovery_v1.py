"""Read-only, CPU-only Formal Dataset route activation audit.

The sample package is read one trajectory at a time. No checkpoint, optimizer,
GPU device, or locked-test package is opened by this static pass.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))
SAMPLES = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/packages/samples"
TENSORS = SAMPLES.parent / "tensor"


def scan() -> dict:
    totals = {split: Counter() for split in ("dev_train", "dev_validation")}
    flow_ids = {split: set() for split in totals}
    trajectory_ids = {split: set() for split in totals}
    examples = {split: [] for split in totals}
    potential_ids = {split: set() for split in totals}
    for path in sorted(SAMPLES.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            samples = json.load(stream)
        if len(samples) != 92:
            raise ValueError(f"noncanonical Formal shard count: {path}")
        for sample in samples:
            metadata = sample["metadata"]
            split = metadata["split"]
            if split not in totals:
                raise ValueError("locked or unknown split rejected")
            c = totals[split]
            c["windows"] += 1
            trajectory_ids[split].add(metadata["trajectory_id"])
            frame = sample["history"][-1]
            active_flows = {int(f["flow_index"]): f for f in frame["logical_flows"]
                            if f["presence"] and f["status"] == "ACTIVE"}
            multihop = [f for f in frame["carrying_states"]
                        if f["active"] and len(f["route_node_indices"]) > 1
                        and int(f["flow_index"]) in active_flows]
            if multihop:
                c["active_multihop_windows"] += 1
                c["active_multihop_flow_window_occurrences"] += len(multihop)
                for f in multihop:
                    logical = active_flows[int(f["flow_index"])]
                    flow_ids[split].add((metadata["trajectory_id"], f["flow_id"], int(logical["epoch"])))
            for horizon, action in enumerate(sample["future_action"][:4], 1):
                # This is offline activation accounting only. At H2-H4 the
                # pre-action true state is the previous target outcome; it is
                # never passed to a model rollout or planner runtime.
                pre = frame if horizon == 1 else sample["target"][horizon - 2]
                pre_flows = pre.get("logical_flows", [])
                pre_active = {f["task_id"]: f for f in pre_flows
                              if f["presence"] and f["status"] == "ACTIVE"}
                for route in action["route"]["entries"]:
                    c["route_action_nonempty_occurrences"] += 1
                    hops = route.get("route_node_indices", [])
                    if len(hops) > 1:
                        c["route_action_multihop_occurrences"] += 1
                    task_id = route.get("task_id")
                    if any(f["task_id"] == task_id for f in active_flows.values()):
                        c["route_action_anchor_flow_overlap_occurrences"] += 1
                        c[f"route_action_anchor_flow_overlap_H{horizon}_occurrences"] += 1
                        potential_ids[split].add(metadata["sample_id"])
                    current = pre_active.get(task_id)
                    if current is not None:
                        c["route_action_existing_flow_occurrences"] += 1
                        if hops and int(hops[-1]) == int(current["logical_destination_index"]):
                            c["route_action_same_destination_occurrences"] += 1
                        else:
                            c["route_action_destination_change_occurrences"] += 1
                            if len(examples[split]) < 5:
                                examples[split].append({"sample_id": metadata["sample_id"], "horizon": horizon,
                                    "route": hops, "logical_destination_index": current["logical_destination_index"]})
                    else:
                        c["route_action_no_unique_existing_flow_occurrences"] += 1
                    c[f"route_action_H{horizon}_occurrences"] += 1
    return {split: {**dict(c), "active_multihop_distinct_flows": len(flow_ids[split]),
                    "trajectories": len(trajectory_ids[split]),
                    "destination_change_examples": examples[split],
                    "potential_route_overlap_window_count": len(potential_ids[split]),
                    "potential_route_overlap_sample_ids": sorted(potential_ids[split])} for split, c in totals.items()}


def scan_route_tensor_capacity() -> dict:
    """Count actual multi-hop rows in anchor/target/action tensors, all 60 shards."""
    counts = {split: Counter() for split in ("dev_train", "dev_validation")}
    from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
    for path in sorted(TENSORS.glob("*.npz")):
        tensor = load_flow_tensor_batch(path)
        anchor = tensor["route_node_mask"]
        target = tensor["target_route_node_mask"]
        action = tensor["future_route_hop_mask"]
        split = tensor["sample_metadata"][0]["split"]
        c = counts[split]
        c["shards"] += 1
        c["windows"] += anchor.shape[0]
        c["anchor_route_gt1_flow_occurrences"] += int((anchor.sum(-1) > 1).sum())
        c["target_route_gt1_flow_occurrences"] += int((target.sum(-1) > 1).sum())
        c["future_route_gt1_action_occurrences"] += int((action.sum(-1) > 1).sum())
        c["max_anchor_route_width"] = max(c["max_anchor_route_width"], anchor.shape[-1])
        c["max_target_route_width"] = max(c["max_target_route_width"], target.shape[-1])
        c["max_future_route_width"] = max(c["max_future_route_width"], action.shape[-1])
    return {split: dict(c) for split, c in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"sample": scan(), "tensor": scan_route_tensor_capacity()}
    if any(result[k][split]["windows"] != expected for k in ("sample", "tensor")
           for split, expected in (("dev_train", 4416), ("dev_validation", 1104))):
        raise ValueError("Formal split window count mismatch")
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
