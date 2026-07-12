"""Evaluate PI-JWM v9 checkpoints with action-input ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pi_jwm.v6_data import (
    V6WorldModelDataset,
    collate_v6_world_model_batch,
    load_world_model_arrays,
    make_normalization_stats,
)
from pi_jwm.v6_dual_graph import V6DualGraphBatch
from pi_jwm.v8_training import build_v8_model_from_arrays, evaluate_v8_model

from run_world_model_v8_full_training import add_event_memory_features, resolve_seed_splits


DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "pi_jwm_v9_expanded_v2_gpu_20260619"
    / "v2_hurdle_baseline"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PI-JWM checkpoint with action ablations.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--modes",
        default="normal_actions,zero_future_actions,zero_history_actions",
        help="Comma/space separated modes to evaluate.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_mode_list(value: str) -> list[str]:
    return [part for part in value.replace(",", " ").split() if part]


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalized_raw_zero(stats: dict, key: str) -> torch.Tensor:
    mean, std = stats[key]
    zero = (0.0 - mean) / std
    return torch.as_tensor(zero[0], dtype=torch.float32)


def apply_action_ablation(
    batch: V6DualGraphBatch,
    stats: dict,
    mode: str,
) -> V6DualGraphBatch:
    if mode == "normal_actions":
        return batch
    if mode == "zero_future_actions":
        future_actions = normalized_raw_zero(stats, "edge_a_future").clone()
        action_history = batch.action_history
    elif mode == "zero_history_actions":
        future_actions = batch.future_actions
        action_history = normalized_raw_zero(stats, "edge_a_hist").clone()
    else:
        raise ValueError(f"Unknown action ablation mode: {mode}")
    return V6DualGraphBatch(
        node_history=batch.node_history,
        physical_edge_history=batch.physical_edge_history,
        info_edge_history=batch.info_edge_history,
        action_history=action_history,
        future_actions=future_actions,
        task_history=batch.task_history,
        link_rate_baseline=batch.link_rate_baseline,
    )


class ActionAblationDataset(Dataset):
    def __init__(self, base_dataset: Dataset, stats: dict, mode: str):
        self.base_dataset = base_dataset
        self.stats = stats
        self.mode = mode

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, item: int):
        batch, target = self.base_dataset[item]
        return apply_action_ablation(batch, self.stats, self.mode), target


def load_model_for_experiment(summary: dict, arrays: dict[str, np.ndarray], checkpoint_path: Path, device: torch.device):
    config = summary["config"]
    activity_memory_dim = int(config.get("activity_memory_dim", 0))
    model = build_v8_model_from_arrays(
        arrays,
        hidden_dim=int(config["hidden_dim"]),
        graph_mode=config["graph_mode"],
        fusion_mode=config["fusion_mode"],
        fusion_num_heads=int(config.get("fusion_num_heads", 4)),
        active_rate_auxiliary=bool(config.get("active_rate_auxiliary", False)),
        active_rate_head_mode=config.get("active_rate_head_mode", "mlp"),
        num_rate_experts=int(config.get("num_rate_experts", 4)),
        rate_output_mode=config.get("model_rate_output_mode", "direct"),
        history_encoder=config.get("history_encoder", "mean"),
        latent_transition_mode=config.get("latent_transition_mode", "message_passing"),
        adaptive_edge_context=config.get("adaptive_edge_context", "none"),
        adaptive_edge_topk=int(config.get("adaptive_edge_topk", 8)),
        activity_memory_dim=activity_memory_dim,
        activity_memory_routing="activity_only" if activity_memory_dim > 0 else "none",
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def evaluate_action_ablation(
    experiment_dir: Path,
    checkpoint_path: Path | None = None,
    modes: list[str] | None = None,
    device: torch.device | None = None,
    batch_size: int = 64,
) -> list[dict]:
    experiment_dir = Path(experiment_dir)
    summary = json.loads((experiment_dir / "v8_full_training_summary.json").read_text(encoding="utf-8"))
    dataset_dir = resolve_project_path(summary["dataset_dir"])
    checkpoint_path = checkpoint_path or experiment_dir / "checkpoints" / "v8_dual_best.pt"
    modes = modes or ["normal_actions", "zero_future_actions", "zero_history_actions"]
    device = device or torch.device("cpu")

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
    stats = make_normalization_stats(arrays, train_idx)
    val_base = V6WorldModelDataset(arrays, val_idx, stats)
    test_base = V6WorldModelDataset(arrays, test_idx, stats)
    model = load_model_for_experiment(summary, arrays, checkpoint_path, device)
    config = summary["config"]

    results = []
    for mode in modes:
        val_loader = DataLoader(
            ActionAblationDataset(val_base, stats, mode),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_v6_world_model_batch,
        )
        test_loader = DataLoader(
            ActionAblationDataset(test_base, stats, mode),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_v6_world_model_batch,
        )
        val_metrics = evaluate_v8_model(
            model,
            val_loader,
            device,
            stats,
            rate_output_mode=config.get("rate_output_mode", "main"),
            inactive_rate_value=float(config.get("inactive_rate_value", 0.0)),
            hurdle_gate_temperature=float(config.get("eval_hurdle_gate_temperature", 1.0)),
            hurdle_gate_power=float(config.get("eval_hurdle_gate_power", 1.0)),
        )
        threshold = val_metrics["activity"]["threshold"]
        test_metrics = evaluate_v8_model(
            model,
            test_loader,
            device,
            stats,
            activity_threshold=threshold,
            rate_output_mode=config.get("rate_output_mode", "main"),
            inactive_rate_value=float(config.get("inactive_rate_value", 0.0)),
            hurdle_gate_temperature=float(config.get("eval_hurdle_gate_temperature", 1.0)),
            hurdle_gate_power=float(config.get("eval_hurdle_gate_power", 1.0)),
        )
        results.append(
            {
                "label": mode,
                "val_threshold": float(threshold),
                "val": val_metrics,
                "test": test_metrics,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    results = evaluate_action_ablation(
        args.experiment_dir,
        checkpoint_path=args.checkpoint_path,
        modes=parse_mode_list(args.modes),
        device=device,
        batch_size=args.batch_size,
    )
    output_json = args.output_json or args.experiment_dir.parent / f"{args.experiment_dir.name}_action_ablation_eval.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in results:
        test = row["test"]
        print(
            f"{row['label']}: "
            f"threshold={row['val_threshold']:.6f} "
            f"active_rate_rmse={test['active_rate']['active_rmse']:.6f} "
            f"f1={test['activity']['f1']:.6f} "
            f"link_rmse={test['link_rate']['rmse']:.6f}"
        )
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
