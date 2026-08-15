import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "dataset_multiseed_v0"
STRICT_ACTION_DIR = ROOT / "reports" / "strict_action_v0"
OUTPUT_DIR = ROOT / "datasets" / "edge_action_v0"

EDGE_ACTION_FEATURES = [
    "offload_count",
    "rb_task_count",
    "rb_total",
    "cpu_task_count",
    "cpu_total",
    "return_count",
]


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Build edge-level action tensors from strict scheduler action logs.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--strict-action-dir", type=Path, default=STRICT_ACTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--min-core-match-rate", type=float, default=0.05)
    return parser.parse_args()


def read_action_file(seed, filename):
    path = STRICT_ACTION_DIR / f"seed_{seed:03d}" / filename
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required action log is missing or empty: {path}")
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = df["time"].round(3)
    return df


def edge_lookup(edge_vocab):
    lookup = {}
    for row in edge_vocab.itertuples(index=False):
        lookup[(row.tx_id, row.rx_id)] = int(row.edge_index)
    return lookup


def add_edge_event(tensor, time_to_idx, edge_to_idx, time_value, tx, rx, values):
    key = (tx, rx)
    if key not in edge_to_idx:
        return False
    time_key = round(float(time_value), 3)
    if time_key not in time_to_idx:
        return False
    ti = time_to_idx[time_key]
    ei = edge_to_idx[key]
    tensor[ti, ei, :] += np.asarray(values, dtype=np.float32)
    return True


def build_seed_edge_action(seed, seed_samples, edge_vocab):
    min_time = min(seed_samples["input_start_time"].min(), seed_samples["label_start_time"].min())
    max_time = max(seed_samples["input_end_time"].max(), seed_samples["label_end_time"].max())
    times = np.round(np.arange(min_time, max_time + 1e-6, 0.1), 3)
    time_to_idx = {round(float(t), 3): i for i, t in enumerate(times)}
    edge_to_idx = edge_lookup(edge_vocab)
    tensor = np.zeros((len(times), len(edge_vocab), len(EDGE_ACTION_FEATURES)), dtype=np.float32)
    matched = {name: 0 for name in ["offload", "rb", "cpu", "return"]}
    total = {name: 0 for name in ["offload", "rb", "cpu", "return"]}

    offload = read_action_file(seed, "offload_actions.csv")
    for row in offload.itertuples(index=False):
        total["offload"] += 1
        ok = add_edge_event(
            tensor,
            time_to_idx,
            edge_to_idx,
            row.time,
            row.source_node_id,
            row.target_node_id,
            [1, 0, 0, 0, 0, 0],
        )
        matched["offload"] += int(ok)

    rb = read_action_file(seed, "rb_actions.csv")
    for row in rb.itertuples(index=False):
        total["rb"] += 1
        ok = add_edge_event(
            tensor,
            time_to_idx,
            edge_to_idx,
            row.time,
            row.current_node_id,
            row.assigned_to,
            [0, 1, float(row.rb_count), 0, 0, 0],
        )
        matched["rb"] += int(ok)

    cpu = read_action_file(seed, "cpu_actions.csv")
    for row in cpu.itertuples(index=False):
        total["cpu"] += 1
        ok = add_edge_event(
            tensor,
            time_to_idx,
            edge_to_idx,
            row.time,
            row.task_node_id,
            row.assigned_to,
            [0, 0, 0, 1, float(row.allocated_cpu), 0],
        )
        matched["cpu"] += int(ok)

    returns = read_action_file(seed, "return_actions.csv")
    for row in returns.itertuples(index=False):
        total["return"] += 1
        ok = add_edge_event(
            tensor,
            time_to_idx,
            edge_to_idx,
            row.time,
            row.current_node_id,
            row.return_target_id,
            [0, 0, 0, 0, 0, 1],
        )
        matched["return"] += int(ok)

    return times, tensor, matched, total


def slice_edge_actions(tensor, sample_index, start_col, end_col):
    samples = []
    for row in sample_index.itertuples(index=False):
        start = int(getattr(row, start_col))
        end = int(getattr(row, end_col))
        samples.append(tensor[start : end + 1])
    return np.stack(samples, axis=0).astype(np.float32)


def validate_core_match_rates(summary_rows, minimum_rate=0.05):
    if not 0.0 <= minimum_rate <= 1.0:
        raise ValueError("minimum_rate must be between 0 and 1")
    aggregate = {}
    for action_name in ("offload", "rb", "cpu"):
        total = sum(int(row[f"{action_name}_total"]) for row in summary_rows)
        matched = sum(int(row[f"{action_name}_matched"]) for row in summary_rows)
        rate = matched / total if total else 0.0
        aggregate[action_name] = {"total": total, "matched": matched, "match_rate": rate}
        if rate < minimum_rate:
            raise ValueError(
                f"{action_name} action match rate {rate:.6f} is below required floor {minimum_rate:.6f}."
            )
    return aggregate


def build_edge_action_dataset(
    dataset_dir=DATASET_DIR,
    strict_action_dir=STRICT_ACTION_DIR,
    output_dir=OUTPUT_DIR,
    min_core_match_rate=0.05,
):
    global STRICT_ACTION_DIR
    dataset_dir = Path(dataset_dir)
    strict_action_dir = Path(strict_action_dir)
    output_dir = Path(output_dir)
    previous_strict_action_dir = STRICT_ACTION_DIR
    STRICT_ACTION_DIR = strict_action_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_index = pd.read_csv(dataset_dir / "sample_index.csv")
    edge_vocab = pd.read_csv(dataset_dir / "edge_vocab.csv")
    hist_parts = []
    future_parts = []
    summary_rows = []

    try:
        for seed in sorted(sample_index["seed"].unique()):
            seed = int(seed)
            seed_samples = sample_index[sample_index["seed"] == seed].reset_index(drop=True)
            times, tensor, matched, total = build_seed_edge_action(seed, seed_samples, edge_vocab)
            hist_parts.append(slice_edge_actions(tensor, seed_samples, "input_start_idx", "input_end_idx"))
            future_parts.append(slice_edge_actions(tensor, seed_samples, "label_start_idx", "label_end_idx"))
            row = {"seed": seed, "num_times": len(times), "num_edges": len(edge_vocab)}
            for key in total:
                row[f"{key}_total"] = int(total[key])
                row[f"{key}_matched"] = int(matched[key])
                row[f"{key}_match_rate"] = float(matched[key] / total[key]) if total[key] else 0.0
            summary_rows.append(row)
    finally:
        STRICT_ACTION_DIR = previous_strict_action_dir

    edge_a_hist = np.concatenate(hist_parts, axis=0)
    edge_a_future = np.concatenate(future_parts, axis=0)
    aggregate_match = validate_core_match_rates(summary_rows, minimum_rate=min_core_match_rate)
    if not np.any(edge_a_future):
        raise ValueError(
            "edge_a_future is all zero; strict action logs are missing, unmatched, or misaligned. "
            "Refusing to create an action-conditioned dataset."
        )
    npz_path = output_dir / "edge_action_v0_samples.npz"
    np.savez_compressed(
        npz_path,
        edge_action_features=np.array(EDGE_ACTION_FEATURES),
        edge_a_hist=edge_a_hist,
        edge_a_future=edge_a_future,
        sample_seed=sample_index["seed"].to_numpy(dtype=np.int32),
        sample_id=sample_index["sample_id"].to_numpy(dtype=np.int64),
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "edge_action_match_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary = {
        "dataset_dir": str(dataset_dir),
        "strict_action_dir": str(strict_action_dir),
        "output_dir": str(output_dir),
        "features": EDGE_ACTION_FEATURES,
        "shapes": {
            "edge_a_hist": list(edge_a_hist.shape),
            "edge_a_future": list(edge_a_future.shape),
        },
        "match_summary": summary_rows,
        "aggregate_core_match": aggregate_match,
        "outputs": {
            "npz": str(npz_path),
            "match_summary_csv": str(summary_path),
        },
    }
    summary_path_json = output_dir / "edge_action_v0_summary.json"
    summary_path_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    args = parse_args()
    summary = build_edge_action_dataset(
        dataset_dir=args.dataset_dir,
        strict_action_dir=args.strict_action_dir,
        output_dir=args.output_dir,
        min_core_match_rate=args.min_core_match_rate,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
