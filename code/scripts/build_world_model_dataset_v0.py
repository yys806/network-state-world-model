import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "datasets" / "dataset_multiseed_v0"
EDGE_ACTION_DIR = ROOT / "datasets" / "edge_action_v0"
OUTPUT_DIR = ROOT / "datasets" / "world_model_dataset_v0"

STATE_SAMPLE_KEYS = ("x_node", "x_link", "x_task", "y_node", "y_link", "y_task")
ACTION_SAMPLE_KEYS = ("edge_a_hist", "edge_a_future")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Build integrated world-model tensors from state and edge-action tensors.")
    parser.add_argument("--state-dir", type=Path, default=STATE_DIR)
    parser.add_argument("--edge-action-dir", type=Path, default=EDGE_ACTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def validate_world_model_inputs(state, edge_action):
    required_state = ("sample_seed", *STATE_SAMPLE_KEYS)
    required_action = ("sample_seed", *ACTION_SAMPLE_KEYS)
    for owner, arrays, required in (
        ("state", state, required_state),
        ("edge action", edge_action, required_action),
    ):
        missing = [key for key in required if key not in arrays]
        if missing:
            raise ValueError(f"Missing {owner} arrays: {', '.join(missing)}")

    if not np.array_equal(state["sample_seed"], edge_action["sample_seed"]):
        raise ValueError("State samples and edge action samples are not aligned by sample_seed.")

    state_has_ids = "sample_id" in state
    action_has_ids = "sample_id" in edge_action
    if state_has_ids != action_has_ids:
        raise ValueError("State and edge action datasets must either both contain sample_id or both omit it.")
    if state_has_ids and not np.array_equal(state["sample_id"], edge_action["sample_id"]):
        raise ValueError("State samples and edge action samples are not aligned by sample_id.")

    num_samples = len(state["sample_seed"])
    for key in (*STATE_SAMPLE_KEYS, *ACTION_SAMPLE_KEYS):
        arrays = state if key in state else edge_action
        value = np.asarray(arrays[key])
        if value.shape[0] != num_samples:
            raise ValueError(f"{key} sample dimension does not match sample_seed.")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains non-finite values.")

    history = state["x_node"].shape[1]
    if any(state[key].shape[1] != history for key in ("x_link", "x_task")):
        raise ValueError("State history dimensions are inconsistent.")
    if edge_action["edge_a_hist"].shape[1] != history:
        raise ValueError("State and action history dimensions are inconsistent.")

    horizon = state["y_node"].shape[1]
    if any(state[key].shape[1] != horizon for key in ("y_link", "y_task")):
        raise ValueError("State target horizon dimensions are inconsistent.")
    if edge_action["edge_a_future"].shape[1] != horizon:
        raise ValueError("State and future action horizon dimensions are inconsistent.")

    edge_count = state["x_link"].shape[2]
    if state["y_link"].shape[2] != edge_count:
        raise ValueError("State link edge dimensions are inconsistent.")
    if any(edge_action[key].shape[2] != edge_count for key in ACTION_SAMPLE_KEYS):
        raise ValueError("State and action edge dimensions are inconsistent.")
    if state["x_node"].shape[2] != state["y_node"].shape[2]:
        raise ValueError("State node dimensions are inconsistent.")

    future_nonzero = int(np.count_nonzero(edge_action["edge_a_future"]))
    if future_nonzero == 0:
        raise ValueError(
            "edge_a_future is all zero; refusing to create an action-conditioned world-model dataset."
        )
    return {
        "num_samples": int(num_samples),
        "history": int(history),
        "horizon": int(horizon),
        "num_edges": int(edge_count),
        "future_action_nonzero": future_nonzero,
        "sample_id_checked": bool(state_has_ids),
    }


def build_world_model_dataset(state_dir=STATE_DIR, edge_action_dir=EDGE_ACTION_DIR, output_dir=OUTPUT_DIR):
    state_dir = Path(state_dir)
    edge_action_dir = Path(edge_action_dir)
    output_dir = Path(output_dir)
    with np.load(state_dir / "dataset_multiseed_v0_samples.npz", allow_pickle=True) as data:
        state = {key: data[key] for key in data.files}
    with np.load(edge_action_dir / "edge_action_v0_samples.npz", allow_pickle=True) as data:
        edge_action = {key: data[key] for key in data.files}

    validation = validate_world_model_inputs(state, edge_action)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_vocab = pd.read_csv(state_dir / "node_vocab.csv")
    edge_vocab = pd.read_csv(state_dir / "edge_vocab.csv")
    node_to_idx = dict(zip(node_vocab["node_id"], node_vocab["node_index"]))
    src_idx = []
    dst_idx = []
    valid_edge_node = []
    for row in edge_vocab.itertuples(index=False):
        src = node_to_idx.get(row.tx_id, -1)
        dst = node_to_idx.get(row.rx_id, -1)
        src_idx.append(src)
        dst_idx.append(dst)
        valid_edge_node.append(int(src >= 0 and dst >= 0))
    src_idx = np.asarray(src_idx, dtype=np.int32)
    dst_idx = np.asarray(dst_idx, dtype=np.int32)
    valid_edge_node = np.asarray(valid_edge_node, dtype=np.int32)

    link_features = list(state["link_features"])
    rate_idx = link_features.index("rate_sum")
    y_link_rate = state["y_link"][..., rate_idx].astype(np.float32)
    y_link_active = (y_link_rate > 1e-6).astype(np.float32)

    output_path = output_dir / "world_model_dataset_v0_samples.npz"
    payload = dict(
        node_features=state["node_features"],
        link_features=state["link_features"],
        task_features=state["task_features"],
        edge_action_features=edge_action["edge_action_features"],
        x_node=state["x_node"].astype(np.float32),
        x_link=state["x_link"].astype(np.float32),
        x_task=state["x_task"].astype(np.float32),
        edge_a_hist=edge_action["edge_a_hist"].astype(np.float32),
        edge_a_future=edge_action["edge_a_future"].astype(np.float32),
        y_node=state["y_node"].astype(np.float32),
        y_link=state["y_link"].astype(np.float32),
        y_link_rate=y_link_rate,
        y_link_active=y_link_active,
        y_task=state["y_task"].astype(np.float32),
        sample_seed=state["sample_seed"].astype(np.int32),
        edge_src_idx=src_idx,
        edge_dst_idx=dst_idx,
        valid_edge_node=valid_edge_node,
    )
    if "sample_id" in state:
        payload["sample_id"] = state["sample_id"].astype(np.int64)
    np.savez_compressed(output_path, **payload)
    node_vocab.to_csv(output_dir / "node_vocab.csv", index=False, encoding="utf-8-sig")
    edge_vocab.to_csv(output_dir / "edge_vocab.csv", index=False, encoding="utf-8-sig")
    summary = {
        "source_state_dir": str(state_dir),
        "source_edge_action_dir": str(edge_action_dir),
        "output_dir": str(output_dir),
        "shapes": {
            "x_node": list(state["x_node"].shape),
            "x_link": list(state["x_link"].shape),
            "x_task": list(state["x_task"].shape),
            "edge_a_hist": list(edge_action["edge_a_hist"].shape),
            "edge_a_future": list(edge_action["edge_a_future"].shape),
            "y_link_rate": list(y_link_rate.shape),
            "y_link_active": list(y_link_active.shape),
            "y_task": list(state["y_task"].shape),
        },
        "num_edges": int(len(edge_vocab)),
        "num_nodes": int(len(node_vocab)),
        "valid_edge_node_ratio": float(valid_edge_node.mean()),
        "active_link_item_ratio": float(y_link_active.any(axis=1).mean()),
        "input_validation": validation,
        "outputs": {
            "npz": str(output_path),
            "node_vocab": str(output_dir / "node_vocab.csv"),
            "edge_vocab": str(output_dir / "edge_vocab.csv"),
        },
    }
    summary_path = output_dir / "world_model_dataset_v0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    args = parse_args()
    summary = build_world_model_dataset(
        state_dir=args.state_dir,
        edge_action_dir=args.edge_action_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
