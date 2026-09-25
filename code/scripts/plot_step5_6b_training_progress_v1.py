"""Render a read-only snapshot of a STEP 5.6B formal training run.

The input directory contains copies of the live run logs. This script never
opens the live run directory for writing and does not infer final performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A copy can catch the active writer during its last append.
            if line_number == len(lines):
                break
            raise
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rolling_mean(values: list[float], width: int) -> list[float]:
    sums = 0.0
    result = []
    for index, value in enumerate(values):
        sums += value
        if index >= width:
            sums -= values[index - width]
        result.append(sums / min(index + 1, width))
    return result


def stage_lines(axis) -> None:
    for step, label in [(552, "prior H1"), (1104, "H2"), (2208, "H4")]:
        axis.axvline(step, color="0.5", linestyle=":", linewidth=1)
        axis.text(step + 20, 0.97, label, transform=axis.get_xaxis_transform(),
                  fontsize=8, va="top", color="0.35")


def render_training(rows: list[dict], output: Path) -> None:
    steps = [row["completed_steps"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    for axis, fields in zip(axes, [("L_Total", "L_Mot", "L_CSI"), ("L_KL", "gradient_norm_before_clip")]):
        for field in fields:
            values = [float(row[field]) for row in rows]
            axis.plot(steps, values, alpha=0.13, linewidth=0.7)
            axis.plot(steps, rolling_mean(values, 50), label=f"{field} (50-step mean)", linewidth=1.7)
        stage_lines(axis)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Training loss")
    axes[1].set_ylabel("KL / gradient norm")
    axes[1].set_xlabel("Completed optimizer steps")
    axes[0].set_title(f"Formal training progress through step {steps[-1]} / 5520 — interim snapshot")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def render_validation(rows: list[dict], output: Path) -> None:
    steps = [row["completed_steps"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    axes[0].plot(steps, [row["L_Val"] for row in rows], "o-", linewidth=2, label="L_Val")
    axes[0].legend()
    for horizon in range(1, 5):
        axes[1].plot(steps, [next(part["L_Pred"] for part in row["per_horizon"]
                                   if part["horizon"] == horizon) for row in rows],
                     "o-", label=f"H{horizon}")
    axes[1].legend(ncol=4)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlim(0, 5520)
    axes[0].set_ylabel("Full validation L_Val")
    axes[1].set_ylabel("Per-horizon L_Pred")
    axes[1].set_xlabel("Completed optimizer steps")
    fig.suptitle(f"Prior-only validation: {len(rows)} completed passes, 1104 windows each — interim")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def render_raw_errors(rows: list[dict], output: Path) -> None:
    steps = [row["completed_steps"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, layout="constrained")
    for column, family in enumerate(("motion", "csi")):
        for row_index, metric in enumerate(("raw_mae", "raw_rmse")):
            axis = axes[row_index, column]
            for horizon in range(1, 5):
                axis.plot(steps, [next(part[metric] for part in row["raw_metrics"][family]
                                       if part["horizon"] == horizon) for row in rows],
                          "o-", label=f"H{horizon}")
            axis.set_title(f"{family.upper() if family == 'csi' else 'Motion'} {metric[4:].upper()}")
            axis.set_ylabel("dB" if family == "csi" else "Aggregate raw error (mixed units)")
            axis.set_xlim(0, 5520)
            axis.grid(alpha=0.2)
            axis.legend(ncol=4, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Completed optimizer steps")
    fig.suptitle("Validation raw errors by rollout horizon — interim snapshot")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = args.snapshot_dir.resolve()
    output = (args.output_dir or source / "plots").resolve()
    output.mkdir(parents=True, exist_ok=True)

    train = read_jsonl(source / "train_metrics.jsonl")
    validation = read_jsonl(source / "validation_metrics.jsonl")
    heartbeat = json.loads((source / "heartbeat.json").read_text(encoding="utf-8"))
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    if not train or not validation:
        raise ValueError("Training and validation logs must both contain completed rows")
    if any(row["run_id"] != manifest["run_id"] for row in train + validation):
        raise ValueError("Run identity mismatch between logs and run manifest")
    if len({row["completed_steps"] for row in train}) != len(train):
        raise ValueError("Duplicate training step in copied log")
    if train[-1]["completed_steps"] != len(train):
        raise ValueError("Training log has missing step rows")
    for row in validation:
        if row["validation_windows"] != 1104 or not row["prior_only"] or not row["parameter_unchanged"]:
            raise ValueError("Validation provenance is incomplete")

    figures = {
        "training_loss": output / "training_loss.png",
        "validation_loss": output / "validation_loss.png",
        "validation_raw_errors": output / "validation_raw_errors.png",
    }
    render_training(train, figures["training_loss"])
    render_validation(validation, figures["validation_loss"])
    render_raw_errors(validation, figures["validation_raw_errors"])
    input_names = ("train_metrics.jsonl", "validation_metrics.jsonl", "heartbeat.json", "run_manifest.json")
    receipt = {
        "schema_version": "step5_6b_progress_snapshot_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest["run_id"],
        "run_source_git_sha": manifest["git_commit"],
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "formal_config_sha256": manifest["formal_config_sha256"],
        "copied_heartbeat_timestamp": heartbeat["timestamp"],
        "copied_heartbeat_status": heartbeat["status"],
        "last_logged_completed_step": train[-1]["completed_steps"],
        "validation_completed_steps": [row["completed_steps"] for row in validation],
        "validation_L_Val": [row["L_Val"] for row in validation],
        "input_sha256": {name: sha256(source / name) for name in input_names},
        "figure_sha256": {name: sha256(path) for name, path in figures.items()},
        "scope": {"interim_snapshot": True, "formal_result": False, "training_modified": False,
                  "locked_test_accessed": False, "performance_claim": False},
    }
    (output / "snapshot_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
