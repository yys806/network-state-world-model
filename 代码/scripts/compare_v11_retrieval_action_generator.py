"""CPU-first retrieval action generator for PI-JWM v11 candidate.

The goal is to test an in-distribution alternative to direct BC/token policies:
retrieve future-action templates from similar training states, optionally
average several neighbors, then evaluate the generated actions through the
frozen PI-JWM world model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v11_rollout_value_calibrator import freeze_module

from diagnose_v11_counterfactual_value_attribution import collect_raw_actions
from run_v11_rb_total_repair import active_rate_row, evaluate_raw_actions, make_adaptive_dataset, write_csv
from run_v11_rb_total_value_head import limit_indices, load_context_limited


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_retrieval_action_generator_20260628"
RB_DIM = 2
CPU_DIM = 4
EPS = 1e-9


def resolve_torch_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if device_arg not in {"cpu", "cuda"}:
        raise ValueError(f"unknown device: {device_arg}")
    return torch.device(device_arg)


def standardize_features(train_features: np.ndarray, query_features: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    train_features = np.asarray(train_features, dtype=np.float32)
    query_features = np.asarray(query_features, dtype=np.float32)
    if train_features.ndim != 2 or query_features.ndim != 2:
        raise ValueError("features must be 2D")
    if train_features.shape[1] != query_features.shape[1]:
        raise ValueError("train and query feature dimensions must match")
    mean = np.mean(train_features, axis=0, keepdims=True).astype(np.float32)
    std = np.std(train_features, axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return (
        ((train_features - mean) / std).astype(np.float32),
        ((query_features - mean) / std).astype(np.float32),
        {"mean": mean.reshape(-1), "std": std.reshape(-1)},
    )


def retrieve_knn_indices(
    query_features: np.ndarray,
    prototype_features: np.ndarray,
    k: int,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    query_features = np.asarray(query_features, dtype=np.float32)
    prototype_features = np.asarray(prototype_features, dtype=np.float32)
    if query_features.ndim != 2 or prototype_features.ndim != 2:
        raise ValueError("features must be 2D")
    if prototype_features.shape[0] == 0:
        raise ValueError("at least one prototype is required")
    if query_features.shape[1] != prototype_features.shape[1]:
        raise ValueError("query and prototype feature dimensions must match")
    k = max(1, min(int(k), prototype_features.shape[0]))
    nearest = np.empty((query_features.shape[0], k), dtype=np.int64)
    distances = np.empty((query_features.shape[0], k), dtype=np.float32)
    proto_norm = np.sum(prototype_features * prototype_features, axis=1, keepdims=True).T
    for start in range(0, query_features.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), query_features.shape[0])
        query = query_features[start:stop]
        distance = np.sum(query * query, axis=1, keepdims=True) + proto_norm - 2.0 * (query @ prototype_features.T)
        distance = np.maximum(distance, 0.0)
        part = np.argpartition(distance, kth=k - 1, axis=1)[:, :k]
        part_distance = np.take_along_axis(distance, part, axis=1)
        order = np.argsort(part_distance, axis=1)
        nearest[start:stop] = np.take_along_axis(part, order, axis=1).astype(np.int64)
        distances[start:stop] = np.take_along_axis(part_distance, order, axis=1).astype(np.float32)
    return nearest, distances


def aggregate_retrieved_actions(
    prototype_actions: np.ndarray,
    nearest_indices: np.ndarray,
    distances: np.ndarray,
    mode: str,
) -> np.ndarray:
    prototype_actions = np.asarray(prototype_actions, dtype=np.float32)
    nearest_indices = np.asarray(nearest_indices, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float32)
    if nearest_indices.shape != distances.shape:
        raise ValueError("nearest_indices and distances must share shape")
    selected = prototype_actions[nearest_indices]
    if mode == "nearest":
        return selected[:, 0].astype(np.float32)
    if mode == "mean":
        return np.mean(selected, axis=1).astype(np.float32)
    if mode == "median":
        return np.median(selected, axis=1).astype(np.float32)
    if mode == "inverse_distance":
        weights = 1.0 / np.maximum(distances, EPS)
        weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), EPS)
        return np.sum(selected * weights[:, :, None, None, None], axis=1).astype(np.float32)
    raise ValueError(f"unknown aggregation mode: {mode}")


def _replacement_dims(mode: str) -> list[int]:
    if mode == "all":
        return [0, 1, 2, 3, 4, 5]
    if mode == "rb_cpu":
        return [1, RB_DIM, 3, CPU_DIM]
    if mode == "rb_only":
        return [1, RB_DIM]
    raise ValueError(f"unknown replacement mode: {mode}")


def _cap_step_totals(repaired: np.ndarray, baseline: np.ndarray, dims: list[int], cap_scale: float) -> None:
    if float(cap_scale) <= 0.0:
        return
    for dim in (RB_DIM, CPU_DIM):
        if dim not in dims:
            continue
        for step in range(1, repaired.shape[1]):
            current = np.sum(np.clip(repaired[:, step, :, dim], 0.0, None), axis=1)
            cap = np.sum(np.clip(baseline[:, step, :, dim], 0.0, None), axis=1) * float(cap_scale)
            scale = np.ones_like(current, dtype=np.float32)
            mask = current > np.maximum(cap, EPS)
            scale[mask] = (cap[mask] / np.maximum(current[mask], EPS)).astype(np.float32)
            repaired[:, step, :, dim] *= scale[:, None]


def apply_retrieved_action(
    baseline_actions: np.ndarray,
    retrieved_actions: np.ndarray,
    alpha: float,
    replacement_mode: str,
    preserve_step0: bool,
    step_total_cap_scale: float,
) -> np.ndarray:
    baseline_actions = np.asarray(baseline_actions, dtype=np.float32)
    retrieved_actions = np.asarray(retrieved_actions, dtype=np.float32)
    if baseline_actions.shape != retrieved_actions.shape:
        raise ValueError("baseline and retrieved actions must share shape")
    repaired = baseline_actions.copy()
    dims = _replacement_dims(replacement_mode)
    alpha = float(alpha)
    repaired[..., dims] = (1.0 - alpha) * baseline_actions[..., dims] + alpha * retrieved_actions[..., dims]
    if preserve_step0:
        repaired[:, 0] = baseline_actions[:, 0]
    _cap_step_totals(repaired, baseline_actions, dims, float(step_total_cap_scale))
    return np.clip(repaired, 0.0, None).astype(np.float32)


def make_sample_retrieval_features(arrays: dict[str, np.ndarray], indices: np.ndarray, mode: str) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    parts: list[np.ndarray] = []
    if mode in {"load", "state_load"}:
        hist = np.asarray(arrays["edge_a_hist"][indices], dtype=np.float32)
        totals = np.sum(hist, axis=2)
        counts = np.sum(hist > EPS, axis=2).astype(np.float32)
        parts.extend(
            [
                totals[:, -1],
                np.mean(totals, axis=1),
                np.max(totals, axis=1),
                totals[:, -1] - totals[:, 0],
                counts[:, -1],
                np.mean(counts, axis=1),
            ]
        )
    if mode in {"state", "state_load"}:
        task = np.asarray(arrays["x_task"][indices], dtype=np.float32)
        link = np.asarray(arrays["x_link"][indices], dtype=np.float32)
        node = np.asarray(arrays["x_node"][indices], dtype=np.float32)
        parts.extend(
            [
                task[:, -1],
                np.mean(task, axis=1),
                task[:, -1] - task[:, 0],
                np.mean(link[:, -1], axis=1),
                np.std(link[:, -1], axis=1),
                np.max(link[:, -1], axis=1),
                np.mean(node[:, -1], axis=1),
                np.std(node[:, -1], axis=1),
            ]
        )
    if not parts:
        raise ValueError(f"unknown feature mode: {mode}")
    return np.concatenate([part.reshape(len(indices), -1) for part in parts], axis=1).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-experiment-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline")
    parser.add_argument("--world-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt")
    parser.add_argument("--policy-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--policy-threshold", type=float, default=0.4)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--new-policy-threshold", type=float, default=0.37)
    parser.add_argument("--new-value-scale", type=float, default=1.06)
    parser.add_argument("--gate-feature", choices=("step_rb_total", "step_cpu_total", "step_rb_cpu_total", "step_active_count"), default="step_rb_cpu_total")
    parser.add_argument("--gate-threshold", type=float, default=450.0)
    parser.add_argument("--value-codebook-size", type=int, default=9)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=256)
    parser.add_argument("--max-test-samples", type=int, default=256)
    parser.add_argument("--limit-after-stats", action="store_true")
    parser.add_argument("--streaming-stats", action="store_true")
    parser.add_argument("--stats-chunk-size", type=int, default=512)
    parser.add_argument("--feature-modes", choices=("load", "state", "state_load"), nargs="+", default=["state_load"])
    parser.add_argument("--neighbor-k", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--aggregation-modes", choices=("nearest", "mean", "median", "inverse_distance"), nargs="+", default=["nearest", "inverse_distance"])
    parser.add_argument("--replacement-modes", choices=("all", "rb_cpu", "rb_only"), nargs="+", default=["rb_cpu", "rb_only"])
    parser.add_argument("--blend-alpha", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--step-total-cap-scale", type=float, nargs="+", default=[0.0, 1.1, 1.25])
    parser.add_argument("--preserve-step0", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=20260628)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    np.random.seed(int(args.seed))
    device = resolve_torch_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, splits = load_context_limited(args, device)
    splits = dict(splits)
    if args.limit_after_stats:
        splits["train"] = limit_indices(splits["train"], args.max_train_samples)
        splits["val"] = limit_indices(splits["val"], args.max_val_samples)
        splits["test"] = limit_indices(splits["test"], args.max_test_samples)
    world_model = freeze_module(world_model)
    policy_model = freeze_module(policy_model)
    train_actions = np.asarray(arrays["edge_a_future"][splits["train"]], dtype=np.float32)

    rows: list[dict] = []
    diagnostics: dict[str, object] = {
        "train_samples": int(len(splits["train"])),
        "feature_modes": list(args.feature_modes),
        "neighbor_k": [int(item) for item in args.neighbor_k],
        "aggregation_modes": list(args.aggregation_modes),
        "replacement_modes": list(args.replacement_modes),
        "blend_alpha": [float(item) for item in args.blend_alpha],
        "step_total_cap_scale": [float(item) for item in args.step_total_cap_scale],
    }

    feature_cache: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for feature_mode in args.feature_modes:
        train_features = make_sample_retrieval_features(arrays, splits["train"], feature_mode)
        feature_cache[feature_mode] = (train_features, {})
        diagnostics[f"{feature_mode}_feature_dim"] = int(train_features.shape[1])

    for split_name in ("val", "test"):
        base_dataset, adaptive_dataset = make_adaptive_dataset(
            args,
            arrays,
            splits[split_name],
            stats,
            policy_model,
            action_scale,
            value_vocab,
            device,
            splits["train"],
        )
        baseline_actions, truth_actions = collect_raw_actions(adaptive_dataset, stats)
        baseline_predictions = evaluate_raw_actions(baseline_actions, base_dataset, stats, world_model, summary["config"], device, args.batch_size)
        baseline_row = active_rate_row("identity", split_name, baseline_predictions, float("nan"))
        baseline_rmse = float(baseline_row["active_rate_rmse"])
        baseline_row.update(
            {
                "feature_mode": "none",
                "neighbor_k": 0,
                "aggregation_mode": "none",
                "replacement_mode": "none",
                "alpha": 0.0,
                "step_total_cap_scale": 0.0,
            }
        )
        rows.append(baseline_row)

        for feature_mode in args.feature_modes:
            train_features = feature_cache[feature_mode][0]
            query_features = make_sample_retrieval_features(arrays, splits[split_name], feature_mode)
            train_z, query_z, _ = standardize_features(train_features, query_features)
            max_k = max(int(k) for k in args.neighbor_k)
            nearest, distances = retrieve_knn_indices(query_z, train_z, k=max_k)
            diagnostics[f"{split_name}_{feature_mode}_mean_nearest_distance"] = float(np.mean(distances[:, 0]))
            for neighbor_k in args.neighbor_k:
                k = int(neighbor_k)
                selected_idx = nearest[:, :k]
                selected_dist = distances[:, :k]
                for aggregation_mode in args.aggregation_modes:
                    retrieved = aggregate_retrieved_actions(train_actions, selected_idx, selected_dist, aggregation_mode)
                    for replacement_mode in args.replacement_modes:
                        for alpha in args.blend_alpha:
                            for cap_scale in args.step_total_cap_scale:
                                actions = apply_retrieved_action(
                                    baseline_actions,
                                    retrieved,
                                    alpha=float(alpha),
                                    replacement_mode=replacement_mode,
                                    preserve_step0=bool(args.preserve_step0),
                                    step_total_cap_scale=float(cap_scale),
                                )
                                predictions = evaluate_raw_actions(actions, base_dataset, stats, world_model, summary["config"], device, args.batch_size)
                                candidate = (
                                    f"retrieval__{feature_mode}__k{int(k)}__{aggregation_mode}"
                                    f"__{replacement_mode}__alpha{float(alpha):g}__cap{float(cap_scale):g}"
                                )
                                row = active_rate_row(candidate, split_name, predictions, baseline_rmse)
                                row.update(
                                    {
                                        "feature_mode": feature_mode,
                                        "neighbor_k": int(k),
                                        "aggregation_mode": aggregation_mode,
                                        "replacement_mode": replacement_mode,
                                        "alpha": float(alpha),
                                        "step_total_cap_scale": float(cap_scale),
                                    }
                                )
                                rows.append(row)

    write_csv(args.output_dir / "retrieval_action_results.csv", rows)
    val_ranked = sorted(
        [row for row in rows if row["split"] == "val"],
        key=lambda row: (float(row["active_rate_rmse"]), str(row["candidate"])),
    )
    write_csv(args.output_dir / "retrieval_action_val_ranked.csv", val_ranked)
    test_by_candidate = {str(row["candidate"]): row for row in rows if row["split"] == "test"}
    best_overall_val = val_ranked[0] if val_ranked else None
    best_non_identity_val = next((row for row in val_ranked if str(row["candidate"]) != "identity"), None)
    identity_val = next((row for row in val_ranked if str(row["candidate"]) == "identity"), None)
    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "retrieval_action_generator_cpu",
        "output_dir": str(args.output_dir),
        "command": " ".join(sys.argv),
        "device": str(device),
        "diagnostics": {**diagnostics, "runtime_seconds": float(time.time() - started)},
        "identity_val": identity_val,
        "matched_test_for_identity_val": None if identity_val is None else test_by_candidate.get(str(identity_val["candidate"])),
        "best_overall_val": best_overall_val,
        "matched_test_for_best_overall_val": None
        if best_overall_val is None
        else test_by_candidate.get(str(best_overall_val["candidate"])),
        "best_non_identity_val": best_non_identity_val,
        "matched_test_for_best_non_identity_val": None
        if best_non_identity_val is None
        else test_by_candidate.get(str(best_non_identity_val["candidate"])),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
