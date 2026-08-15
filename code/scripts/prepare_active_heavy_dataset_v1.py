"""Prepare an expanded active-heavy PI-JWM world-model dataset pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ARTIFACT_DATASET_ROOT = PROJECT_ROOT / "artifacts" / "experiments" / "airfogsim_v0" / "datasets"
ARTIFACT_REPORT_ROOT = PROJECT_ROOT / "artifacts" / "experiments" / "airfogsim_v0" / "reports"
AIRFOGSIM_REFERENCE_ROOT = PROJECT_ROOT / "reference" / "AirFogSim"
AIRFOGSIM_EXAMPLE_ROOT = AIRFOGSIM_REFERENCE_ROOT / "examples"


DEFAULT_SEEDS = tuple(range(20))
DEFAULT_TAG = "active_heavy_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or dry-run the PI-JWM active-heavy dataset v1 pipeline.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-time", type=float, default=40.0)
    parser.add_argument("--output-tag", default=DEFAULT_TAG)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run", action="store_true", help="execute the generated commands; default is dry-run only")
    parser.add_argument("--high-rate-threshold", type=float, default=600.0)
    return parser.parse_args()


def build_paths(output_tag: str) -> dict[str, Path]:
    return {
        "raw_root": ARTIFACT_REPORT_ROOT / f"multiseed_raw_{output_tag}",
        "state_dataset": ARTIFACT_DATASET_ROOT / f"dataset_multiseed_{output_tag}",
        "strict_actions": ARTIFACT_REPORT_ROOT / f"strict_action_{output_tag}",
        "edge_action": ARTIFACT_DATASET_ROOT / f"edge_action_{output_tag}",
        "world_model": ARTIFACT_DATASET_ROOT / f"world_model_dataset_{output_tag}",
    }


def build_pipeline_commands(
    seeds: list[int],
    max_time: float,
    output_tag: str = DEFAULT_TAG,
    python_executable: str = sys.executable,
) -> list[list[str]]:
    paths = build_paths(output_tag)
    seed_args = [str(seed) for seed in seeds]
    return [
        [
            python_executable,
            str(SCRIPTS_DIR / "export_multiseed_dataset_v0.py"),
            "--seeds",
            *seed_args,
            "--max-time",
            str(float(max_time)),
            "--output-root",
            str(paths["raw_root"]),
        ],
        [
            python_executable,
            str(SCRIPTS_DIR / "build_dataset_multiseed_v0.py"),
            "--raw-root",
            str(paths["raw_root"]),
            "--output-dir",
            str(paths["state_dataset"]),
        ],
        [
            python_executable,
            str(SCRIPTS_DIR / "export_strict_actions_v0.py"),
            "--seeds",
            *seed_args,
            "--max-time",
            str(float(max_time)),
            "--dataset-dir",
            str(paths["state_dataset"]),
            "--output-dir",
            str(paths["strict_actions"]),
        ],
        [
            python_executable,
            str(SCRIPTS_DIR / "build_edge_action_v0.py"),
            "--dataset-dir",
            str(paths["state_dataset"]),
            "--strict-action-dir",
            str(paths["strict_actions"]),
            "--output-dir",
            str(paths["edge_action"]),
        ],
        [
            python_executable,
            str(SCRIPTS_DIR / "build_world_model_dataset_v0.py"),
            "--state-dir",
            str(paths["state_dataset"]),
            "--edge-action-dir",
            str(paths["edge_action"]),
            "--output-dir",
            str(paths["world_model"]),
        ],
    ]


def summarize_active_rate_distribution(arrays: dict[str, np.ndarray], high_rate_threshold: float = 600.0) -> dict:
    active = np.asarray(arrays["y_link_active"]) > 0.5
    rate = np.asarray(arrays["y_link_rate"], dtype=np.float32)
    active_rate = rate[active]
    sample_seed = np.asarray(arrays["sample_seed"])
    summary = {
        "num_samples": int(active.shape[0]),
        "horizon": int(active.shape[1]),
        "num_edges": int(active.shape[2]),
        "total_link_steps": int(active.size),
        "active_link_steps": int(active.sum()),
        "active_ratio": float(active.mean()),
        "high_rate_threshold": float(high_rate_threshold),
        "high_rate_active_steps": int((active_rate >= high_rate_threshold).sum()),
        "active_rate_mean": float(active_rate.mean()) if active_rate.size else 0.0,
        "active_rate_p90": float(np.percentile(active_rate, 90)) if active_rate.size else 0.0,
        "active_edges": int((active.sum(axis=(0, 1)) > 0).sum()),
        "per_seed": [],
    }
    for seed in sorted(int(seed) for seed in np.unique(sample_seed)):
        idx = np.where(sample_seed == seed)[0]
        seed_active = active[idx]
        seed_rate = rate[idx][seed_active]
        summary["per_seed"].append(
            {
                "seed": int(seed),
                "samples": int(len(idx)),
                "active_link_steps": int(seed_active.sum()),
                "high_rate_active_steps": int((seed_rate >= high_rate_threshold).sum()),
                "active_edges": int((seed_active.sum(axis=(0, 1)) > 0).sum()),
            }
        )
    return summary


def load_world_model_arrays(dataset_dir: Path) -> dict[str, np.ndarray]:
    with np.load(dataset_dir / "world_model_dataset_v0_samples.npz", allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def write_plan(path: Path, commands: list[list[str]], paths: dict[str, Path], args: argparse.Namespace) -> dict:
    plan = {
        "framework": "PI-JWM",
        "purpose": "Expanded active-heavy dataset for active-rate amplitude modeling.",
        "seeds": [int(seed) for seed in args.seeds],
        "max_time": float(args.max_time),
        "output_tag": args.output_tag,
        "paths": {key: str(value) for key, value in paths.items()},
        "commands": commands,
        "run_requested": bool(args.run),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def build_subprocess_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    existing = env.get("PYTHONPATH", "")
    paths = [str(AIRFOGSIM_REFERENCE_ROOT)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PI_JWM_AIRFOGSIM_EXAMPLE_DIR"] = str(AIRFOGSIM_EXAMPLE_ROOT)
    return env


def build_subprocess_workdir() -> Path:
    return AIRFOGSIM_EXAMPLE_ROOT


def run_commands(commands: list[list[str]]) -> None:
    env = build_subprocess_env()
    workdir = build_subprocess_workdir()
    for command in commands:
        print("[active-heavy] running", " ".join(command), flush=True)
        subprocess.run(command, cwd=str(workdir), check=True, env=env)


def main() -> None:
    args = parse_args()
    paths = build_paths(args.output_tag)
    commands = build_pipeline_commands(args.seeds, args.max_time, args.output_tag, args.python)
    plan_path = ARTIFACT_DATASET_ROOT / f"{args.output_tag}_pipeline_plan.json"
    plan = write_plan(plan_path, commands, paths, args)
    if args.run:
        run_commands(commands)
    summary = None
    world_model_dir = paths["world_model"]
    if (world_model_dir / "world_model_dataset_v0_samples.npz").exists():
        arrays = load_world_model_arrays(world_model_dir)
        summary = summarize_active_rate_distribution(arrays, high_rate_threshold=args.high_rate_threshold)
        summary_path = world_model_dir / "active_rate_distribution_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"plan_path": str(plan_path), "run": bool(args.run), "summary": summary, "commands": plan["commands"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
