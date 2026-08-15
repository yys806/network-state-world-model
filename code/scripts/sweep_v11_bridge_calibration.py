"""CPU-only PI-JWM v11 bridge calibration sweep.

This script does not train a policy. It evaluates existing policy checkpoints
through the frozen PI-JWM bridge under conservative action/value decoders, then
ranks configurations by validation active-rate RMSE.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORLD_EXPERIMENT_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "pi_jwm_v9_expanded_v2_gpu_20260619" / "v2_hurdle_baseline"
DEFAULT_WORLD_CHECKPOINT = DEFAULT_WORLD_EXPERIMENT_DIR / "checkpoints" / "v8_dual_best.pt"
DEFAULT_POLICY_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "pi_jwm_v11_hierarchical_tokens_cpu_probe_1024x3_h64_20260620"
    / "checkpoints"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "pi_jwm_v11_bridge_calibration_sweep_20260620"


@dataclass(frozen=True)
class SweepConfig:
    checkpoint_name: str
    action_decoder: str
    value_decoder: str
    value_scale: float
    budget_quantile: float

    def slug(self) -> str:
        return "__".join(
            [
                _safe_token(self.checkpoint_name),
                _safe_token(self.action_decoder),
                _safe_token(self.value_decoder),
                f"scale_{_float_token(self.value_scale)}",
                f"bq_{_float_token(self.budget_quantile)}",
            ]
        )


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))


def _float_token(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep conservative PI-JWM v11 bridge calibration settings.")
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--world-experiment-dir", type=Path, default=DEFAULT_WORLD_EXPERIMENT_DIR)
    parser.add_argument("--world-checkpoint", type=Path, default=DEFAULT_WORLD_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint", action="append", dest="checkpoints", default=None)
    parser.add_argument("--action-decoder", action="append", dest="action_decoders", default=None)
    parser.add_argument("--value-decoder", action="append", dest="value_decoders", default=None)
    parser.add_argument("--value-scale", action="append", type=float, dest="value_scales", default=None)
    parser.add_argument("--budget-quantile", action="append", type=float, dest="budget_quantiles", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N configs after enumeration; 0 means all.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write planned commands but do not execute bridge runs.")
    return parser.parse_args()


def enumerate_sweep_configs(
    checkpoints: Iterable[str],
    action_decoders: Iterable[str],
    value_decoders: Iterable[str],
    value_scales: Iterable[float],
    budget_quantiles: Iterable[float],
) -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    for checkpoint_name in checkpoints:
        for action_decoder in action_decoders:
            for value_decoder in value_decoders:
                for value_scale in value_scales:
                    for budget_quantile in budget_quantiles:
                        configs.append(
                            SweepConfig(
                                checkpoint_name=str(checkpoint_name),
                                action_decoder=str(action_decoder),
                                value_decoder=str(value_decoder),
                                value_scale=float(value_scale),
                                budget_quantile=float(budget_quantile),
                            )
                        )
    return configs


def default_configs(args: argparse.Namespace) -> list[SweepConfig]:
    checkpoints = args.checkpoints or ["best", "val_bin_accuracy", "val_activity_f1", "last"]
    action_decoders = args.action_decoders or ["threshold", "probability_mass_topk", "val_mean_topk", "val_quantile_topk"]
    value_decoders = args.value_decoders or [
        "train_median",
        "train_q75",
        "train_median_dim_scaled",
        "train_median_step_scaled",
        "train_codebook_quantile",
        "policy",
    ]
    value_scales = args.value_scales or [0.5, 0.75, 1.0, 1.25, 1.5]
    budget_quantiles = args.budget_quantiles or [0.25, 0.5, 0.75]
    configs = enumerate_sweep_configs(checkpoints, action_decoders, value_decoders, value_scales, budget_quantiles)
    if args.limit and args.limit > 0:
        return configs[: args.limit]
    return configs


def checkpoint_path(policy_dir: Path, checkpoint_name: str) -> Path:
    suffix = "best" if checkpoint_name == "best" else checkpoint_name
    return policy_dir / f"v11_discrete_value_policy_cross_attention_{suffix}.pt"


def should_run(output_json: Path, overwrite: bool) -> bool:
    return bool(overwrite or not output_json.exists())


def build_bridge_command(
    python_executable: str,
    bridge_script: Path,
    world_experiment_dir: Path,
    world_checkpoint: Path,
    policy_checkpoint: Path,
    output_json: Path,
    device: str,
    batch_size: int,
    config: SweepConfig,
) -> list[str]:
    return [
        python_executable,
        str(bridge_script),
        "--world-experiment-dir",
        str(world_experiment_dir),
        "--world-checkpoint",
        str(world_checkpoint),
        "--policy-checkpoint",
        str(policy_checkpoint),
        "--output-json",
        str(output_json),
        "--device",
        str(device),
        "--batch-size",
        str(int(batch_size)),
        "--action-decoder",
        config.action_decoder,
        "--mode",
        "true_first_pred_rest",
        "--action-generator",
        "policy",
        "--value-decoder",
        config.value_decoder,
        "--value-scale",
        f"{config.value_scale:g}",
        "--budget-quantile",
        f"{config.budget_quantile:g}",
    ]


def extract_bridge_metrics(payload: dict) -> dict[str, float]:
    return {
        "val_active_rate_rmse": _metric(payload, "val", "active_rate", "active_rmse"),
        "val_activity_f1": _metric(payload, "val", "activity", "f1"),
        "val_link_rmse": _metric(payload, "val", "link_rate", "rmse"),
        "test_active_rate_rmse": _metric(payload, "test", "active_rate", "active_rmse"),
        "test_activity_f1": _metric(payload, "test", "activity", "f1"),
        "test_link_rmse": _metric(payload, "test", "link_rate", "rmse"),
    }


def _metric(payload: dict, split: str, group: str, name: str) -> float:
    value = payload.get(split, {}).get(group, {}).get(name, float("nan"))
    return float(value)


def row_from_result(config: SweepConfig, output_json: Path, payload: dict) -> dict:
    return {
        "slug": config.slug(),
        "checkpoint": config.checkpoint_name,
        "action_decoder": config.action_decoder,
        "value_decoder": config.value_decoder,
        "value_scale": float(config.value_scale),
        "budget_quantile": float(config.budget_quantile),
        "output_json": str(output_json),
        **extract_bridge_metrics(payload),
    }


def rank_results(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            _nan_to_inf(row.get("val_active_rate_rmse")),
            _nan_to_inf(row.get("test_active_rate_rmse")),
            str(row.get("slug", "")),
        ),
    )


def _nan_to_inf(value) -> float:
    value = float(value)
    return float("inf") if math.isnan(value) else value


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "slug",
        "checkpoint",
        "action_decoder",
        "value_decoder",
        "value_scale",
        "budget_quantile",
        "val_active_rate_rmse",
        "test_active_rate_rmse",
        "val_activity_f1",
        "test_activity_f1",
        "val_link_rmse",
        "test_link_rmse",
        "output_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_ranked_markdown(rows: list[dict], path: Path, top_k: int = 20) -> None:
    lines = [
        "# PI-JWM v11 Bridge Calibration Sweep",
        "",
        "Ranked by validation active-rate RMSE, then test active-rate RMSE.",
        "",
        "| rank | checkpoint | action_decoder | value_decoder | scale | budget_q | val_rmse | test_rmse | test_f1 | test_link_rmse |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows[:top_k], start=1):
        lines.append(
            "| {rank} | {checkpoint} | {action_decoder} | {value_decoder} | {scale:g} | {bq:g} | {val:.6f} | {test:.6f} | {f1:.6f} | {link:.6f} |".format(
                rank=idx,
                checkpoint=row["checkpoint"],
                action_decoder=row["action_decoder"],
                value_decoder=row["value_decoder"],
                scale=float(row["value_scale"]),
                bq=float(row["budget_quantile"]),
                val=float(row["val_active_rate_rmse"]),
                test=float(row["test_active_rate_rmse"]),
                f1=float(row["test_activity_f1"]),
                link=float(row["test_link_rmse"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bridge(command: list[str]) -> None:
    subprocess.run(command, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = args.output_dir / "bridge_json"
    json_dir.mkdir(parents=True, exist_ok=True)
    configs = default_configs(args)
    commands_path = args.output_dir / "planned_commands.txt"
    rows: list[dict] = []
    planned_lines: list[str] = []

    for config in configs:
        policy_checkpoint = checkpoint_path(args.policy_dir, config.checkpoint_name)
        output_json = json_dir / f"{config.slug()}.json"
        command = build_bridge_command(
            python_executable=sys.executable,
            bridge_script=PROJECT_ROOT / "scripts" / "evaluate_v10_policy_bridge.py",
            world_experiment_dir=args.world_experiment_dir,
            world_checkpoint=args.world_checkpoint,
            policy_checkpoint=policy_checkpoint,
            output_json=output_json,
            device=args.device,
            batch_size=args.batch_size,
            config=config,
        )
        planned_lines.append(" ".join(command))
        if args.dry_run:
            continue
        if not policy_checkpoint.exists():
            raise FileNotFoundError(f"Missing policy checkpoint: {policy_checkpoint}")
        if should_run(output_json, args.overwrite):
            run_bridge(command)
        payload = load_json(output_json)
        rows.append(row_from_result(config, output_json, payload))

    commands_path.write_text("\n".join(planned_lines) + "\n", encoding="utf-8")
    if rows:
        ranked = rank_results(rows)
        write_csv(ranked, args.output_dir / "sweep_results.csv")
        write_ranked_markdown(ranked, args.output_dir / "sweep_results_ranked.md")
        best = ranked[0]
        print(
            "best "
            f"checkpoint={best['checkpoint']} action_decoder={best['action_decoder']} "
            f"value_decoder={best['value_decoder']} scale={best['value_scale']} "
            f"budget_quantile={best['budget_quantile']} "
            f"val_active_rate_rmse={best['val_active_rate_rmse']:.6f} "
            f"test_active_rate_rmse={best['test_active_rate_rmse']:.6f}"
        )
    print(f"planned_commands={commands_path}")


if __name__ == "__main__":
    main()
