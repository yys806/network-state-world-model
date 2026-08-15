"""Diagnose coupled-token value policy errors for PI-JWM v11 candidates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.paths import ARTIFACTS_DIR
from pi_jwm.v6_data import load_world_model_arrays, make_normalization_stats
from pi_jwm.v7_action_policy import V7ActionPolicy
from run_v11_discrete_value_policy import (
    V11DiscreteValuePolicyDataset,
    collate_discrete_policy_batch,
    collect_predictions,
)
from run_v7_action_policy import resolve_policy_seed_splits


DEFAULT_DATASET_DIR = (
    ARTIFACTS_DIR
    / "experiments"
    / "airfogsim_v0"
    / "datasets"
    / "world_model_dataset_active_heavy_v2_60seed_20260619"
)
DEFAULT_CHECKPOINT = (
    ARTIFACTS_DIR
    / "experiments"
    / "pi_jwm_v11_coupled_tokens_gpu_early_h64_e5_20260620"
    / "checkpoints"
    / "v11_discrete_value_policy_cross_attention_best.pt"
)
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "reports" / "v11_next_plan_20260620"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PI-JWM v11 coupled-token policy predictions.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def make_confusion_rows(
    token_true: np.ndarray,
    token_pred: np.ndarray,
    group_name: str,
    vocab_size: int,
) -> list[dict[str, int | str | float]]:
    true_flat = np.asarray(token_true, dtype=np.int64).reshape(-1)
    pred_flat = np.asarray(token_pred, dtype=np.int64).reshape(-1)
    if true_flat.shape != pred_flat.shape:
        raise ValueError("token_true and token_pred must have the same shape")
    valid = (true_flat >= 0) & (true_flat < int(vocab_size)) & (pred_flat >= 0) & (pred_flat < int(vocab_size))
    true_flat = true_flat[valid]
    pred_flat = pred_flat[valid]
    rows: list[dict[str, int | str | float]] = []
    total = int(true_flat.size)
    for true_token in range(int(vocab_size)):
        for pred_token in range(int(vocab_size)):
            count = int(np.sum((true_flat == true_token) & (pred_flat == pred_token)))
            if count:
                rows.append(
                    {
                        "group": group_name,
                        "true_token": true_token,
                        "pred_token": pred_token,
                        "count": count,
                        "share": float(count / total) if total else float("nan"),
                    }
                )
    return rows


def make_token_histogram_rows(
    token_ids: np.ndarray,
    vocab_values: np.ndarray,
    group_name: str,
    prefix: str,
) -> list[dict[str, int | str]]:
    token_flat = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    vocab_values = np.asarray(vocab_values, dtype=np.float32)
    rows: list[dict[str, int | str]] = []
    for token in range(vocab_values.shape[0]):
        count = int(np.sum(token_flat == token))
        if count:
            rows.append(
                {
                    "group": group_name,
                    "token": token,
                    f"{prefix}_count": count,
                    "value": format_token_value(vocab_values[token]),
                }
            )
    rows.sort(key=lambda row: int(row[f"{prefix}_count"]), reverse=True)
    return rows


def make_aggregate_rows(
    true_value: np.ndarray,
    pred_value: np.ndarray,
    action_features: list[str],
) -> list[dict[str, float | int | str]]:
    true_value = np.asarray(true_value, dtype=np.float32)
    pred_value = np.asarray(pred_value, dtype=np.float32)
    if true_value.shape != pred_value.shape:
        raise ValueError("true_value and pred_value must have the same shape")
    if true_value.ndim != 4:
        raise ValueError("values must have shape [sample, horizon, edge, action_dim]")
    rows: list[dict[str, float | int | str]] = []
    horizon = true_value.shape[1]
    action_dim = true_value.shape[-1]
    for h in range(horizon):
        for dim in range(action_dim):
            true_slice = true_value[:, h, :, dim]
            pred_slice = pred_value[:, h, :, dim]
            true_total = float(np.sum(true_slice))
            pred_total = float(np.sum(pred_slice))
            rows.append(
                {
                    "horizon": h,
                    "dim": dim,
                    "name": action_features[dim] if dim < len(action_features) else f"dim_{dim}",
                    "true_total": true_total,
                    "pred_total": pred_total,
                    "total_error": pred_total - true_total,
                    "active_count_true": int(np.sum(true_slice > 1e-9)),
                    "active_count_pred": int(np.sum(pred_slice > 1e-9)),
                }
            )
    return rows


def apply_activity_mask(value: np.ndarray, prob: np.ndarray, threshold: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    prob = np.asarray(prob, dtype=np.float32)
    if value.shape != prob.shape:
        raise ValueError("value and prob must have the same shape")
    return np.where(prob >= float(threshold), value, 0.0).astype(np.float32)


def analyze(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    arrays = load_world_model_arrays(args.dataset_dir)
    train_idx, val_idx, test_idx, split_spec = resolve_policy_seed_splits(
        arrays["sample_seed"],
        train_seeds=[*range(16), *range(20, 60)],
        val_seeds=[16, 17],
        test_seeds=[18, 19],
    )
    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}[args.split]
    stats = make_normalization_stats(arrays, train_idx)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = type("Config", (), checkpoint["config"])()
    if getattr(config, "value_mode", None) != "coupled_tokens":
        raise ValueError("checkpoint must use value_mode=coupled_tokens")
    vocab = checkpoint.get("value_vocab")
    if not vocab or vocab.get("mode") != "coupled_tokens":
        raise ValueError("checkpoint is missing coupled_tokens value_vocab")

    model = V7ActionPolicy(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    ds = V11DiscreteValuePolicyDataset(arrays, split_indices, stats, vocab)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_discrete_policy_batch)
    pred = collect_predictions(model, loader, device, vocab)

    features = [str(x) for x in arrays["edge_action_features"].tolist()]
    groups = [[int(dim) for dim in group] for group in vocab["groups"]]
    sizes = np.asarray(vocab["sizes"], dtype=np.int64)
    values = np.asarray(vocab["values"], dtype=np.float32)
    group_names = [group_name(group, features) for group in groups]

    token_true = np.asarray(pred["token_true"], dtype=np.int64)
    token_pred = np.asarray(pred["token_pred"], dtype=np.int64)
    confusion_rows: list[dict[str, int | str | float]] = []
    hist_rows: list[dict[str, int | str]] = []
    group_summary: list[dict[str, float | int | str]] = []
    for group_idx, name in enumerate(group_names):
        mask = token_true[..., group_idx] >= 0
        group_true = token_true[..., group_idx][mask]
        group_pred = token_pred[..., group_idx][mask]
        vocab_size = int(sizes[group_idx])
        confusion_rows.extend(make_confusion_rows(group_true, group_pred, name, vocab_size))
        hist_rows.extend(make_token_histogram_rows(group_true, values[group_idx, :vocab_size], name, "true"))
        hist_rows.extend(make_token_histogram_rows(group_pred, values[group_idx, :vocab_size], name, "pred"))
        group_summary.append(
            {
                "group": name,
                "active_positions": int(group_true.size),
                "accuracy": float(np.mean(group_true == group_pred)) if group_true.size else float("nan"),
                "vocab_size": vocab_size,
                "top_true_token": int(most_common_token(group_true)),
                "top_pred_token": int(most_common_token(group_pred)),
            }
        )

    activity_threshold = float(checkpoint.get("activity_threshold", 0.5))
    masked_value_pred = apply_activity_mask(pred["value_pred"], pred["prob"], activity_threshold)
    aggregate_rows = make_aggregate_rows(pred["value_true"], masked_value_pred, features)
    result = {
        "framework": "PI-JWM",
        "module": "v11_coupled_token_policy_analysis",
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "split_seed_spec": split_spec,
        "sample_count": int(len(split_indices)),
        "activity_threshold": activity_threshold,
        "group_summary": group_summary,
        "aggregate_summary": summarize_aggregates(aggregate_rows),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"coupled_token_confusion_{args.split}.csv", confusion_rows)
    write_csv(args.output_dir / f"coupled_token_histogram_{args.split}.csv", hist_rows)
    write_csv(args.output_dir / f"coupled_token_aggregate_{args.split}.csv", aggregate_rows)
    (args.output_dir / f"coupled_token_diagnosis_{args.split}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / f"coupled_token_diagnosis_{args.split}.md").write_text(
        render_markdown(result, group_summary, aggregate_rows),
        encoding="utf-8",
    )
    return result


def format_token_value(value: np.ndarray) -> str:
    return ";".join(f"{idx}:{float(v):.6f}" for idx, v in enumerate(np.asarray(value).reshape(-1)))


def group_name(group: list[int], features: list[str]) -> str:
    return "+".join(features[dim] if dim < len(features) else f"dim_{dim}" for dim in group)


def most_common_token(tokens: np.ndarray) -> int:
    if tokens.size == 0:
        return -1
    unique, counts = np.unique(tokens, return_counts=True)
    return int(unique[int(np.argmax(counts))])


def summarize_aggregates(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    rb_errors = [float(row["total_error"]) for row in rows if row["name"] == "rb_total"]
    cpu_errors = [float(row["total_error"]) for row in rows if row["name"] == "cpu_total"]
    return {
        "rb_total_abs_error_sum": float(np.sum(np.abs(rb_errors))) if rb_errors else float("nan"),
        "cpu_total_abs_error_sum": float(np.sum(np.abs(cpu_errors))) if cpu_errors else float("nan"),
    }


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "group",
        "token",
        "true_count",
        "pred_count",
        "value",
        "true_token",
        "pred_token",
        "count",
        "share",
        "horizon",
        "dim",
        "name",
        "true_total",
        "pred_total",
        "total_error",
        "active_count_true",
        "active_count_pred",
    ]
    keys = set().union(*(row.keys() for row in rows))
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys.difference(fieldnames)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(
    result: dict,
    group_summary: list[dict[str, float | int | str]],
    aggregate_rows: list[dict[str, float | int | str]],
) -> str:
    lines = [
        "# PI-JWM v11 Coupled Token Diagnosis",
        "",
        f"- Split: {result['split']}",
        f"- Sample count: {result['sample_count']}",
        f"- Checkpoint: {result['checkpoint']}",
        "",
        "## Group Token Accuracy",
        "",
        "| group | active positions | vocab size | accuracy | top true token | top pred token |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in group_summary:
        lines.append(
            "| {group} | {active_positions} | {vocab_size} | {accuracy:.6f} | {top_true_token} | {top_pred_token} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate Totals By Horizon",
            "",
            "| horizon | dim | name | true total | pred total | error | true active | pred active |",
            "|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {horizon} | {dim} | {name} | {true_total:.6f} | {pred_total:.6f} | {total_error:.6f} | {active_count_true} | {active_count_pred} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Prompt",
            "",
            "- If RB/CPU groups have low token accuracy or majority-token collapse, prioritize conditional/hierarchical token heads.",
            "- If token accuracy is moderate but aggregate totals are badly biased, prioritize no-training world-model-aware calibration.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    result = analyze(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
