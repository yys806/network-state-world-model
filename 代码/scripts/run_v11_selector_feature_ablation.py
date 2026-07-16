"""Run validation-only coupling feature ablations for the frozen v11 selector."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.v11_labeling import load_candidate_label_cache
from pi_jwm.v11_selector import (
    ablate_candidate_batch,
    fit_listwise_selector,
    load_fitted_selector_checkpoint,
    observable_pareto_deltas,
    predict_fitted_selector,
    select_with_defer,
)
from train_v11_candidate_set_selector import (
    _choice_metrics,
    _metadata,
    calibrate_improvement_bias,
    validate_cache_protocol,
)


def _checkpoint_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    fallback = manifest_path.parent / path.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(value)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    cache_paths = {
        "train": args.train_cache,
        "calibration": args.calibration_cache,
        "validation": args.validation_cache,
    }
    loaded = {name: load_candidate_label_cache(path) for name, path in cache_paths.items()}
    digest = validate_cache_protocol({name: values[2] for name, values in loaded.items()})
    batches = {name: values[0] for name, values in loaded.items()}
    outcomes = {name: values[1] for name, values in loaded.items()}
    validation_seed = _metadata(args.validation_cache)["sample_seed"]
    train_seed = _metadata(args.train_cache)["sample_seed"]
    frozen = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
    if not bool(frozen.get("configuration_frozen")) or str(frozen.get("configuration_digest")) != digest:
        raise ValueError("ablation requires a matching validation-frozen selector manifest")
    selected = frozen["selected_config"]
    rows = []
    main_rank_scores = []
    main_improvements = []
    main_uncertainties = []
    for value in frozen["selected_checkpoints"]:
        fitted, bias, _ = load_fitted_selector_checkpoint(
            _checkpoint_path(str(value), args.frozen_manifest), digest, args.device
        )
        heads = predict_fitted_selector(fitted, batches["validation"])
        main_rank_scores.append(heads["score"])
        main_improvements.append(heads["predicted_improvement"] - bias)
        main_uncertainties.append(heads["uncertainty"])
    main_ensemble = np.stack(main_rank_scores, axis=0)
    for name, z_value in (("full", 1.64), ("without_uncertainty", 0.0)):
        task_delta, energy_delta = observable_pareto_deltas(
            batches["validation"], outcomes["validation"].default_index
        )
        decision = select_with_defer(
            main_ensemble,
            default_index=outcomes["validation"].default_index,
            ensemble_improvement=np.stack(main_improvements, axis=0),
            ensemble_uncertainty=(
                None if name == "without_uncertainty" else np.stack(main_uncertainties, axis=0)
            ),
            candidate_mask=batches["validation"].candidate_mask,
            z_value=z_value,
            task_delta=task_delta,
            energy_delta=energy_delta,
        )
        metrics = _choice_metrics(
            outcomes["validation"], decision.candidate_index, validation_seed, batches["validation"].candidate_mask
        )
        rows.append(
            {
                "ablation": name,
                "validation_rmse": metrics["rmse"],
                "validation_link_rmse": metrics["link_rmse"],
                "validation_activity_f1": metrics["activity_f1"],
                "improvement_vs_default": metrics["improvement_vs_default"],
                "defer_ratio": float(np.mean(decision.deferred)),
                "worst_seed_regret": metrics["worst_seed_regret"],
                "result_kind": "diagnostic_only" if name != "full" else "deployable",
            }
        )
    for group in ("stage", "task", "resource", "energy"):
        ablated = {name: ablate_candidate_batch(batch, group) for name, batch in batches.items()}
        rank_scores = []
        improvements = []
        uncertainties = []
        for training_seed in args.training_seeds:
            fitted = fit_listwise_selector(
                ablated["train"],
                outcomes["train"],
                hidden_dim=int(selected["hidden_dim"]),
                temperature=float(selected["temperature"]),
                dropout=float(selected["dropout"]),
                epochs=int(args.epochs),
                learning_rate=float(args.learning_rate),
                seed=int(training_seed),
                device=args.device,
                group_ids=train_seed,
            )
            calibration = predict_fitted_selector(fitted, ablated["calibration"])[
                "predicted_improvement"
            ]
            bias = calibrate_improvement_bias(
                calibration,
                outcomes["calibration"].improvement / fitted.target_scale,
                ablated["calibration"].candidate_mask,
            )
            heads = predict_fitted_selector(fitted, ablated["validation"])
            rank_scores.append(heads["score"])
            improvements.append(heads["predicted_improvement"] - bias)
            uncertainties.append(heads["uncertainty"])
        task_delta, energy_delta = observable_pareto_deltas(
            ablated["validation"], outcomes["validation"].default_index
        )
        decision = select_with_defer(
            np.stack(rank_scores, axis=0),
            default_index=outcomes["validation"].default_index,
            ensemble_improvement=np.stack(improvements, axis=0),
            ensemble_uncertainty=np.stack(uncertainties, axis=0),
            candidate_mask=ablated["validation"].candidate_mask,
            task_delta=task_delta,
            energy_delta=energy_delta,
        )
        metrics = _choice_metrics(
            outcomes["validation"], decision.candidate_index, validation_seed, ablated["validation"].candidate_mask
        )
        rows.append(
            {
                "ablation": f"without_{group}",
                "validation_rmse": metrics["rmse"],
                "validation_link_rmse": metrics["link_rmse"],
                "validation_activity_f1": metrics["activity_f1"],
                "improvement_vs_default": metrics["improvement_vs_default"],
                "defer_ratio": float(np.mean(decision.deferred)),
                "worst_seed_regret": metrics["worst_seed_regret"],
                "result_kind": "diagnostic_only",
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "feature_group_ablation.csv"
    _write_csv(output, rows)
    summary = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "selection_split": "validation",
        "result_kind": "diagnostic_only",
        "configuration_digest": digest,
        "rows": rows,
        "output": str(output),
    }
    (args.output_dir / "feature_group_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
