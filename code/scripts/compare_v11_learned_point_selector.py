"""Train a CPU-only learned selector over PI-JWM v11 bridge operating points."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch, load_world_model_arrays, make_normalization_stats
from pi_jwm.v8_training import collect_v8_predictions

from diagnose_v11_adaptive_candidate_oracle import (
    DEFAULT_POINTS,
    PointSpec,
    flat_metric_row,
    metrics_from_arrays,
    oracle_select_by_step,
    parse_point_spec,
    squeeze_last_channel,
)
from diagnose_v11_bridge_operating_point import collect_raw_actions
from evaluate_v10_policy_bridge import load_policy
from evaluate_v11_adaptive_bridge import PointConfig, choose_device, make_point_dataset
from evaluate_v9_action_ablation import load_model_for_experiment, resolve_project_path
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, write_csv
from run_world_model_v8_full_training import add_event_memory_features, resolve_seed_splits


def limit_indices(indices: np.ndarray, limit: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if limit and limit > 0:
        return indices[: min(int(limit), len(indices))]
    return indices


def make_step_feature_matrix(actions_by_point: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions_by_point, dtype=np.float32)
    if actions.ndim != 5:
        raise ValueError("actions_by_point must have shape [point, sample, step, edge, action_dim]")
    point_count, sample_count, horizon, _edge_count, _action_dim = actions.shape
    step_index = np.broadcast_to(
        (np.arange(horizon, dtype=np.float32) / max(float(horizon - 1), 1.0)).reshape(1, horizon, 1),
        (sample_count, horizon, 1),
    )
    feature_blocks = [step_index.reshape(sample_count * horizon, 1)]
    for point_idx in range(point_count):
        point_actions = actions[point_idx]
        active_count = np.count_nonzero(np.abs(point_actions) > 1e-9, axis=(2, 3)).astype(np.float32)
        rb_total = np.clip(point_actions[..., 2], 0.0, None).sum(axis=2)
        cpu_total = np.clip(point_actions[..., 4], 0.0, None).sum(axis=2)
        block = np.stack([active_count, rb_total, cpu_total], axis=-1).reshape(sample_count * horizon, 3)
        feature_blocks.append(block)
    return np.concatenate(feature_blocks, axis=1).astype(np.float32)


def mix_actions_by_step_labels(actions_by_point: np.ndarray, labels: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions_by_point, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if actions.ndim != 5:
        raise ValueError("actions_by_point must have shape [point, sample, step, edge, action_dim]")
    if labels.shape != actions.shape[1:3]:
        raise ValueError("labels must have shape [sample, step]")
    point_count, sample_count, horizon, edge_count, action_dim = actions.shape
    labels = np.clip(labels, 0, point_count - 1)
    mixed = np.zeros((sample_count, horizon, edge_count, action_dim), dtype=np.float32)
    for point_idx in range(point_count):
        mask = labels == point_idx
        if np.any(mask):
            mixed[mask] = actions[point_idx][mask]
    return mixed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-experiment-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline")
    parser.add_argument("--world-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt")
    parser.add_argument("--policy-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_learned_point_selector_cpu_20260629")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=384)
    parser.add_argument("--max-test-samples", type=int, default=384)
    parser.add_argument("--value-codebook-size", type=int, default=9)
    parser.add_argument("--point", action="append", default=None, help="Full bridge point: threshold:scale or name:threshold:scale.")
    parser.add_argument("--selector-trees", type=int, default=200)
    parser.add_argument("--selector-max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260629)
    return parser.parse_args()


def point_specs(args: argparse.Namespace) -> list[PointSpec]:
    return [parse_point_spec(text) for text in (args.point or DEFAULT_POINTS)]


def load_context(args: argparse.Namespace, device: torch.device):
    summary = json.loads((args.world_experiment_dir / "v8_full_training_summary.json").read_text(encoding="utf-8"))
    dataset_dir = resolve_project_path(summary["dataset_dir"])
    arrays = load_world_model_arrays(dataset_dir)
    if summary["config"].get("use_event_memory_features", False):
        arrays = add_event_memory_features(arrays)
    split = summary["split_seed_spec"]
    full_train_idx, val_idx, test_idx, _ = resolve_seed_splits(
        arrays["sample_seed"],
        train_seeds=split["train_seeds"],
        val_seeds=split["val_seeds"],
        test_seeds=split["test_seeds"],
    )
    train_idx = limit_indices(full_train_idx, args.max_train_samples)
    val_idx = limit_indices(val_idx, args.max_val_samples)
    test_idx = limit_indices(test_idx, args.max_test_samples)
    stats = make_normalization_stats(arrays, full_train_idx)
    world_model = load_model_for_experiment(summary, arrays, args.world_checkpoint, device)
    policy_model, action_scale, _, value_vocab = load_policy(args.policy_checkpoint, device)
    return summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, {"train": train_idx, "val": val_idx, "test": test_idx}


def collect_point_bundle(args, split_name: str, indices: np.ndarray, context: tuple, specs: list[PointSpec], device: torch.device) -> dict:
    summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, splits = context
    base = V6WorldModelDataset(arrays, indices, stats)
    rates = []
    probs = []
    actions = []
    rows = []
    truth = None
    active = None
    for spec in specs:
        point = PointConfig(spec.threshold, spec.value_scale, value_codebook_size=args.value_codebook_size)
        dataset = make_point_dataset(arrays, indices, stats, policy_model, action_scale, value_vocab, device, splits["train"], point)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_v6_world_model_batch)
        predictions = collect_v8_predictions(
            world_model,
            loader,
            device,
            stats,
            rate_output_mode=summary["config"].get("rate_output_mode", "main"),
            inactive_rate_value=float(summary["config"].get("inactive_rate_value", 0.0)),
        )
        rate = squeeze_last_channel(predictions["link_rate_pred"])
        prob = squeeze_last_channel(predictions["link_activity_prob"])
        truth = squeeze_last_channel(predictions["link_rate_true"])
        active = squeeze_last_channel(predictions["link_activity_true"]) > 0.5
        rates.append(rate)
        probs.append(prob)
        actions.append(collect_raw_actions(dataset))
        metrics = metrics_from_arrays(rate, prob, truth, active)
        rows.append({"split": split_name, "selector": "candidate_point", **flat_metric_row({"name": spec.name, **metrics})})
    assert truth is not None and active is not None
    rate_stack = np.stack(rates, axis=0)
    actions_stack = np.stack(actions, axis=0)
    oracle_rate, oracle_labels = oracle_select_by_step(rate_stack, truth, active)
    return {
        "base": base,
        "actions": actions_stack,
        "features": make_step_feature_matrix(actions_stack),
        "oracle_labels": oracle_labels.reshape(-1),
        "oracle_label_matrix": oracle_labels,
        "oracle_rate": oracle_rate,
        "truth": truth,
        "active": active,
        "rows": rows,
    }


def train_selector(features: np.ndarray, labels: np.ndarray, args: argparse.Namespace) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=int(args.selector_trees),
        max_depth=int(args.selector_max_depth) if args.selector_max_depth > 0 else None,
        class_weight="balanced_subsample",
        random_state=int(args.seed),
        n_jobs=-1,
    )
    model.fit(features, labels)
    return model


def evaluate_selector(split_name: str, bundle: dict, model: RandomForestClassifier, context: tuple, args: argparse.Namespace, device: torch.device) -> dict:
    summary, _arrays, stats, world_model, _policy_model, _action_scale, _value_vocab, _splits = context
    actions = bundle["actions"]
    sample_count, horizon = actions.shape[1:3]
    predicted_labels = model.predict(bundle["features"]).astype(np.int64).reshape(sample_count, horizon)
    mixed_actions = mix_actions_by_step_labels(actions, predicted_labels)
    predictions = evaluate_raw_actions(mixed_actions, bundle["base"], stats, world_model, summary["config"], device, args.batch_size)
    row = active_rate_row("learned_point_selector", split_name, predictions, float("nan"))
    row["selector_label_counts_json"] = json.dumps(label_counts(predicted_labels), sort_keys=True)
    row["oracle_label_counts_json"] = json.dumps(label_counts(bundle["oracle_label_matrix"]), sort_keys=True)
    return row


def label_counts(labels: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    return {str(int(value)): int(np.sum(labels == value)) for value in np.unique(labels)}


def row_label(row: dict) -> str:
    return str(row.get("candidate", row.get("name", "unknown")))


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    device = choose_device(args.device)
    specs = point_specs(args)
    context = load_context(args, device)
    _summary, _arrays, _stats, _world_model, _policy_model, _action_scale, _value_vocab, splits = context
    bundles = {
        split_name: collect_point_bundle(args, split_name, splits[split_name], context, specs, device)
        for split_name in ("train", "val", "test")
    }
    selector = train_selector(bundles["train"]["features"], bundles["train"]["oracle_labels"], args)
    rows = []
    for split_name in ("val", "test"):
        rows.extend(bundles[split_name]["rows"])
        rows.append(evaluate_selector(split_name, bundles[split_name], selector, context, args, device))
    write_csv(args.output_dir / "learned_point_selector_results.csv", rows)
    val_rows = [row for row in rows if row.get("split") == "val"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "learned_point_selector",
        "point_specs": [spec.__dict__ for spec in specs],
        "rows": rows,
        "best_val": min(val_rows, key=lambda row: float(row["active_rate_rmse"])),
        "best_test": min(test_rows, key=lambda row: float(row["active_rate_rmse"])),
        "runtime_seconds": float(time.time() - started),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    result = run(parse_args())
    best_val = result["best_val"]
    best_test = result["best_test"]
    print(
        f"best_val={row_label(best_val)} active_rmse={float(best_val['active_rate_rmse']):.6f} "
        f"best_test={row_label(best_test)} test_active_rmse={float(best_test['active_rate_rmse']):.6f}"
    )
    print(f"wrote {result['output_dir']}")


if __name__ == "__main__":
    main()
