"""Diagnose oracle headroom for PI-JWM v11 candidate adaptive bridge policies.

This script is diagnostic only. It evaluates a small portfolio of deployable
old/new bridge gate candidates, then reports how much active-rate RMSE could be
reduced by an oracle sample-level or step-level selector. If the oracle mixture
has little headroom, a learned selector is unlikely to justify GPU time.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pi_jwm.v6_data import V6WorldModelDataset, collate_v6_world_model_batch, load_world_model_arrays, make_normalization_stats
from pi_jwm.v6_dual_graph import V6DualGraphBatch
from pi_jwm.v6_metrics import active_rate_metrics, activity_metrics, regression_metrics
from pi_jwm.v8_training import choose_activity_threshold, collect_v8_predictions

from evaluate_v10_policy_bridge import load_policy
from evaluate_v11_adaptive_bridge import PointConfig, choose_device, make_point_dataset, mix_actions_by_step_gate
from evaluate_v9_action_ablation import load_model_for_experiment, resolve_project_path
from run_world_model_v8_full_training import add_event_memory_features, resolve_seed_splits
from sweep_v11_adaptive_bridge_gate import CachedProposal, GateRule, cache_proposals, compute_rule_gate, parse_rule


DEFAULT_RULES = (
    "step_active_count:16",
    "step_active_count:18",
    "step_rb_total:425:12",
    "step_rb_total:425:15",
    "step_rb_cpu_total:450:12",
    "step_rb_cpu_total:450:18",
)

DEFAULT_POINTS = (
    "p0p34_s0p8:0.34:0.8",
    "p0p34_s0p95:0.34:0.95",
    "p0p37_s0p9:0.37:0.9",
    "p0p37_s1p06:0.37:1.06",
    "p0p4_s0p8:0.4:0.8",
    "p0p4_s1p0:0.4:1.0",
    "p0p43_s0p9:0.43:0.9",
    "p0p46_s0p75:0.46:0.75",
)


@dataclass(frozen=True)
class PointSpec:
    name: str
    threshold: float
    value_scale: float


def float_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def parse_point_spec(text: str) -> PointSpec:
    parts = str(text).split(":")
    if len(parts) == 2:
        threshold = float(parts[0])
        value_scale = float(parts[1])
        name = f"point_thr{float_token(threshold)}_scale{float_token(value_scale)}"
    elif len(parts) == 3:
        name = parts[0]
        threshold = float(parts[1])
        value_scale = float(parts[2])
    else:
        raise ValueError("point spec must be threshold:scale or name:threshold:scale")
    if not name:
        raise ValueError("point name must not be empty")
    if value_scale < 0.0:
        raise ValueError("point value_scale must be non-negative")
    return PointSpec(name=name, threshold=threshold, value_scale=value_scale)


def squeeze_last_channel(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim >= 1 and values.shape[-1] == 1:
        return np.squeeze(values, axis=-1)
    return values


def oracle_select_by_sample(
    candidate_rates: np.ndarray,
    truth: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rates = np.asarray(candidate_rates, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if rates.ndim != 4:
        raise ValueError("candidate_rates must have shape [candidate, sample, step, edge]")
    if rates.shape[1:] != truth.shape or truth.shape != active.shape:
        raise ValueError("truth/active must match candidate_rates sample-step-edge shape")
    squared = ((rates - truth[None, ...]) ** 2) * active[None, ...]
    candidate_sse = squared.reshape(rates.shape[0], rates.shape[1], -1).sum(axis=2)
    indices = np.argmin(candidate_sse, axis=0).astype(np.int64)
    selected = rates[indices, np.arange(rates.shape[1])]
    return selected.astype(np.float32), indices, candidate_sse.astype(np.float64)


def oracle_select_by_step(
    candidate_rates: np.ndarray,
    truth: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rates = np.asarray(candidate_rates, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if rates.ndim != 4:
        raise ValueError("candidate_rates must have shape [candidate, sample, step, edge]")
    if rates.shape[1:] != truth.shape or truth.shape != active.shape:
        raise ValueError("truth/active must match candidate_rates sample-step-edge shape")
    squared = ((rates - truth[None, ...]) ** 2) * active[None, ...]
    step_sse = squared.sum(axis=3)
    indices = np.argmin(step_sse, axis=0).astype(np.int64)
    selected = np.take_along_axis(rates, indices[None, ..., None], axis=0).squeeze(axis=0)
    return selected.astype(np.float32), indices


def select_prob_by_sample(candidate_prob: np.ndarray, indices: np.ndarray) -> np.ndarray:
    prob = np.asarray(candidate_prob, dtype=np.float64)
    if prob.ndim != 4:
        raise ValueError("candidate_prob must have shape [candidate, sample, step, edge]")
    return prob[indices, np.arange(prob.shape[1])].astype(np.float32)


def select_prob_by_step(candidate_prob: np.ndarray, indices: np.ndarray) -> np.ndarray:
    prob = np.asarray(candidate_prob, dtype=np.float64)
    if prob.ndim != 4:
        raise ValueError("candidate_prob must have shape [candidate, sample, step, edge]")
    return np.take_along_axis(prob, indices[None, ..., None], axis=0).squeeze(axis=0).astype(np.float32)


def metrics_from_arrays(rate_pred: np.ndarray, prob: np.ndarray, truth: np.ndarray, active: np.ndarray) -> dict:
    threshold = choose_activity_threshold(prob, active)
    return {
        "link_rate": regression_metrics(rate_pred, truth),
        "active_rate": active_rate_metrics(rate_pred, truth, active),
        "activity": activity_metrics(prob, active, threshold=float(threshold)),
    }


def count_indices(indices: np.ndarray, names: list[str]) -> dict[str, int]:
    flat = np.asarray(indices, dtype=np.int64).reshape(-1)
    return {name: int(np.sum(flat == idx)) for idx, name in enumerate(names)}


class CandidatePolicyDataset(Dataset):
    def __init__(self, proposals: list[CachedProposal], normalizer, kind: str, rule: GateRule | None = None) -> None:
        self.proposals = proposals
        self.normalizer = normalizer
        self.kind = kind
        self.rule = rule
        self.gate_true_count = 0
        self.gate_total_count = 0

    def __len__(self) -> int:
        return len(self.proposals)

    def __getitem__(self, item: int):
        proposal = self.proposals[item]
        if self.kind == "old":
            mixed = proposal.old_actions
            gate = torch.zeros((proposal.old_actions.shape[0],), dtype=torch.bool)
        elif self.kind == "new":
            mixed = proposal.new_actions
            gate = torch.ones((proposal.new_actions.shape[0],), dtype=torch.bool)
        elif self.kind == "rule":
            if self.rule is None:
                raise ValueError("rule candidate requires a GateRule")
            gate = compute_rule_gate(proposal.new_actions, self.rule)
            mixed = mix_actions_by_step_gate(proposal.old_actions, proposal.new_actions, gate)
        else:
            raise ValueError(f"unknown candidate kind: {self.kind}")
        self.gate_true_count += int(gate.sum().item())
        self.gate_total_count += int(gate.numel())
        normalized_future = self.normalizer.normalize_future_actions(mixed)
        world_batch = proposal.world_batch
        bridged = V6DualGraphBatch(
            node_history=world_batch.node_history,
            physical_edge_history=world_batch.physical_edge_history,
            info_edge_history=world_batch.info_edge_history,
            action_history=world_batch.action_history,
            future_actions=normalized_future,
            task_history=world_batch.task_history,
            link_rate_baseline=world_batch.link_rate_baseline,
        )
        return bridged, proposal.target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-experiment-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline")
    parser.add_argument("--world-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v9_expanded_v2_gpu_20260619/v2_hurdle_baseline/checkpoints/v8_dual_best.pt")
    parser.add_argument("--policy-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v10_policy_bridge_gpu_20260620/v10_action_policy_bc/checkpoints/v7_action_policy_cross_attention_best.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/pi_jwm_v11_adaptive_candidate_oracle_cpu_20260629")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--old-threshold", type=float, default=0.4)
    parser.add_argument("--old-scale", type=float, default=1.0)
    parser.add_argument("--new-threshold", type=float, default=0.37)
    parser.add_argument("--new-scale", type=float, default=1.06)
    parser.add_argument("--value-codebook-size", type=int, default=9)
    parser.add_argument("--rule", action="append", default=None, help="Candidate rule: feature:threshold[:min_active[:max_active]].")
    parser.add_argument("--disable-rules", action="store_true", help="Evaluate only full bridge points; skip old/new gate-rule candidates.")
    parser.add_argument("--point", action="append", default=None, help="Full bridge point: threshold:scale or name:threshold:scale.")
    parser.add_argument("--split", choices=("val", "test", "both"), default="both")
    parser.add_argument("--max-val-samples", type=int, default=384)
    parser.add_argument("--max-test-samples", type=int, default=384)
    return parser.parse_args()


def limited_indices(indices: np.ndarray, limit: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if limit and limit > 0:
        return indices[: min(int(limit), len(indices))]
    return indices


def load_context(args: argparse.Namespace, device: torch.device):
    summary = json.loads((args.world_experiment_dir / "v8_full_training_summary.json").read_text(encoding="utf-8"))
    dataset_dir = resolve_project_path(summary["dataset_dir"])
    arrays = load_world_model_arrays(dataset_dir)
    if summary["config"].get("use_event_memory_features", False):
        arrays = add_event_memory_features(arrays)
    split = summary["split_seed_spec"]
    train_idx, val_idx, test_idx, _ = resolve_seed_splits(
        arrays["sample_seed"],
        train_seeds=split["train_seeds"],
        val_seeds=split["val_seeds"],
        test_seeds=split["test_seeds"],
    )
    val_idx = limited_indices(val_idx, args.max_val_samples)
    test_idx = limited_indices(test_idx, args.max_test_samples)
    stats = make_normalization_stats(arrays, train_idx)
    world_model = load_model_for_experiment(summary, arrays, args.world_checkpoint, device)
    policy_model, action_scale, _, value_vocab = load_policy(args.policy_checkpoint, device)
    return summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, train_idx, {"val": val_idx, "test": test_idx}


def candidate_specs(args: argparse.Namespace) -> list[tuple[str, str, GateRule | None]]:
    if bool(args.disable_rules):
        return []
    rules = [parse_rule(text) for text in (args.rule or DEFAULT_RULES)]
    specs: list[tuple[str, str, GateRule | None]] = [("old_all", "old", None), ("new_all", "new", None)]
    specs.extend((f"rule_{rule.slug}", "rule", rule) for rule in rules)
    return specs


def point_specs(args: argparse.Namespace) -> list[PointSpec]:
    seen = {"old_all", "new_all"}
    rows = []
    for text in (args.point or DEFAULT_POINTS):
        spec = parse_point_spec(text)
        if spec.name in seen:
            continue
        seen.add(spec.name)
        rows.append(spec)
    return rows


def collect_candidate(
    name: str,
    kind: str,
    rule: GateRule | None,
    proposals: list[CachedProposal],
    normalizer,
    world_model,
    device: torch.device,
    stats: dict,
    config: dict,
    batch_size: int,
) -> tuple[dict, dict]:
    dataset = CandidatePolicyDataset(proposals, normalizer, kind, rule)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_v6_world_model_batch)
    predictions = collect_v8_predictions(
        world_model,
        loader,
        device,
        stats,
        rate_output_mode=config.get("rate_output_mode", "main"),
        inactive_rate_value=float(config.get("inactive_rate_value", 0.0)),
    )
    rate = squeeze_last_channel(predictions["link_rate_pred"])
    truth = squeeze_last_channel(predictions["link_rate_true"])
    prob = squeeze_last_channel(predictions["link_activity_prob"])
    active = squeeze_last_channel(predictions["link_activity_true"]) > 0.5
    metrics = metrics_from_arrays(rate, prob, truth, active)
    metrics["adaptive_gate"] = {
        "true_count": int(dataset.gate_true_count),
        "total_count": int(dataset.gate_total_count),
        "fraction": float(dataset.gate_true_count / max(dataset.gate_total_count, 1)),
    }
    metrics["name"] = name
    return predictions, metrics


def collect_point_candidate(
    spec: PointSpec,
    arrays: dict,
    indices: np.ndarray,
    stats: dict,
    policy_model,
    action_scale: np.ndarray,
    value_vocab: dict | None,
    device: torch.device,
    train_idx: np.ndarray,
    world_model,
    config: dict,
    value_codebook_size: int,
    batch_size: int,
) -> tuple[dict, dict]:
    point = PointConfig(spec.threshold, spec.value_scale, value_codebook_size=value_codebook_size)
    dataset = make_point_dataset(arrays, indices, stats, policy_model, action_scale, value_vocab, device, train_idx, point)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_v6_world_model_batch)
    predictions = collect_v8_predictions(
        world_model,
        loader,
        device,
        stats,
        rate_output_mode=config.get("rate_output_mode", "main"),
        inactive_rate_value=float(config.get("inactive_rate_value", 0.0)),
    )
    rate = squeeze_last_channel(predictions["link_rate_pred"])
    truth = squeeze_last_channel(predictions["link_rate_true"])
    prob = squeeze_last_channel(predictions["link_activity_prob"])
    active = squeeze_last_channel(predictions["link_activity_true"]) > 0.5
    metrics = metrics_from_arrays(rate, prob, truth, active)
    metrics["adaptive_gate"] = {"true_count": 0, "total_count": 0, "fraction": float("nan")}
    metrics["name"] = spec.name
    metrics["point"] = {"threshold": float(spec.threshold), "value_scale": float(spec.value_scale)}
    return predictions, metrics


def run_split(args: argparse.Namespace, split_name: str, context: tuple, device: torch.device) -> tuple[list[dict], dict]:
    summary, arrays, stats, world_model, policy_model, action_scale, value_vocab, train_idx, splits = context
    indices = splits[split_name]
    old_point = PointConfig(args.old_threshold, args.old_scale, value_codebook_size=args.value_codebook_size)
    new_point = PointConfig(args.new_threshold, args.new_scale, value_codebook_size=args.value_codebook_size)
    base = V6WorldModelDataset(arrays, indices, stats)
    old_dataset = make_point_dataset(arrays, indices, stats, policy_model, action_scale, value_vocab, device, train_idx, old_point)
    new_dataset = make_point_dataset(arrays, indices, stats, policy_model, action_scale, value_vocab, device, train_idx, new_point)
    proposals = cache_proposals(base, old_dataset, new_dataset)
    names = []
    rates = []
    probs = []
    rows = []
    truth = None
    active = None
    for spec in point_specs(args):
        predictions, metrics = collect_point_candidate(
            spec,
            arrays,
            indices,
            stats,
            policy_model,
            action_scale,
            value_vocab,
            device,
            train_idx,
            world_model,
            summary["config"],
            args.value_codebook_size,
            args.batch_size,
        )
        names.append(spec.name)
        rates.append(squeeze_last_channel(predictions["link_rate_pred"]))
        probs.append(squeeze_last_channel(predictions["link_activity_prob"]))
        truth = squeeze_last_channel(predictions["link_rate_true"])
        active = squeeze_last_channel(predictions["link_activity_true"]) > 0.5
        rows.append({"split": split_name, "selector": "candidate_point", **flat_metric_row(metrics)})
    for name, kind, rule in candidate_specs(args):
        predictions, metrics = collect_candidate(
            name,
            kind,
            rule,
            proposals,
            old_dataset,
            world_model,
            device,
            stats,
            summary["config"],
            args.batch_size,
        )
        names.append(name)
        rates.append(squeeze_last_channel(predictions["link_rate_pred"]))
        probs.append(squeeze_last_channel(predictions["link_activity_prob"]))
        truth = squeeze_last_channel(predictions["link_rate_true"])
        active = squeeze_last_channel(predictions["link_activity_true"]) > 0.5
        rows.append({"split": split_name, "selector": "candidate", **flat_metric_row(metrics)})
    assert truth is not None and active is not None
    rate_stack = np.stack(rates, axis=0)
    prob_stack = np.stack(probs, axis=0)
    sample_rate, sample_idx, sample_sse = oracle_select_by_sample(rate_stack, truth, active)
    sample_prob = select_prob_by_sample(prob_stack, sample_idx)
    sample_metrics = metrics_from_arrays(sample_rate, sample_prob, truth, active)
    rows.append({"split": split_name, "selector": "oracle_sample", **flat_metric_row({"name": "oracle_sample", **sample_metrics})})
    step_rate, step_idx = oracle_select_by_step(rate_stack, truth, active)
    step_prob = select_prob_by_step(prob_stack, step_idx)
    step_metrics = metrics_from_arrays(step_rate, step_prob, truth, active)
    rows.append({"split": split_name, "selector": "oracle_step", **flat_metric_row({"name": "oracle_step", **step_metrics})})
    summary_payload = {
        "split": split_name,
        "sample_count": int(len(indices)),
        "candidate_names": names,
        "oracle_sample_selection_counts": count_indices(sample_idx, names),
        "oracle_step_selection_counts": count_indices(step_idx, names),
        "candidate_sse": {
            name: {
                "mean": float(np.mean(sample_sse[idx])),
                "median": float(np.median(sample_sse[idx])),
            }
            for idx, name in enumerate(names)
        },
        "oracle_sample": sample_metrics,
        "oracle_step": step_metrics,
    }
    return rows, summary_payload


def flat_metric_row(metrics: dict) -> dict:
    return {
        "name": metrics["name"],
        "active_rate_rmse": float(metrics["active_rate"]["active_rmse"]),
        "active_rate_mae": float(metrics["active_rate"]["active_mae"]),
        "active_count": int(metrics["active_rate"]["active_count"]),
        "link_rmse": float(metrics["link_rate"]["rmse"]),
        "link_mae": float(metrics["link_rate"]["mae"]),
        "activity_f1": float(metrics["activity"]["f1"]),
        "activity_precision": float(metrics["activity"]["precision"]),
        "activity_recall": float(metrics["activity"]["recall"]),
        "activity_threshold": float(metrics["activity"]["threshold"]),
        "gate_fraction": float(metrics.get("adaptive_gate", {}).get("fraction", float("nan"))),
    }


def candidate_selector_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if str(row.get("selector")) in {"candidate", "candidate_point"}]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    device = choose_device(args.device)
    context = load_context(args, device)
    split_names = ["val", "test"] if args.split == "both" else [args.split]
    rows = []
    split_summaries = {}
    for split_name in split_names:
        split_rows, split_summary = run_split(args, split_name, context, device)
        rows.extend(split_rows)
        split_summaries[split_name] = split_summary
        best_candidate = min(candidate_selector_rows(split_rows), key=lambda row: float(row["active_rate_rmse"]))
        oracle_sample = next(row for row in split_rows if row["selector"] == "oracle_sample")
        oracle_step = next(row for row in split_rows if row["selector"] == "oracle_step")
        print(
            f"{split_name} best_candidate={best_candidate['name']} "
            f"active_rmse={best_candidate['active_rate_rmse']:.6f} "
            f"oracle_sample={oracle_sample['active_rate_rmse']:.6f} "
            f"oracle_step={oracle_step['active_rate_rmse']:.6f}"
        )
    write_csv(args.output_dir / "candidate_oracle_results.csv", rows)
    result = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "mode": "adaptive_candidate_oracle_diagnostic",
        "output_dir": str(args.output_dir),
        "splits": split_summaries,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
