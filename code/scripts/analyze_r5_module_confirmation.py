from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r4_gpu_screening import build_validation_windows  # noqa: E402
from pi_jwm.r5_analysis import MetricSpec  # noqa: E402
from pi_jwm.r5_checkpoint import load_r5_checkpoint  # noqa: E402
from pi_jwm.r5_confirmation_analysis import (  # noqa: E402
    analyze_confirmation_reports,
    analyze_horizon_records,
    evaluate_model_by_horizon_cpu,
    freeze_r6_candidate_set,
    merge_confirmation_reports,
    validate_window_schedule,
    write_confirmation_analysis_bundle,
)
from pi_jwm.r5_confirmation_checkpoint import load_confirmation_checkpoint  # noqa: E402
from pi_jwm.r5_protocol import load_r5_protocol  # noqa: E402
from pi_jwm.r4_gpu_screening import SELECTION_COMPONENTS  # noqa: E402


EXPECTED_COMBINATIONS = ("B", "F", "G", "H", "J")
EXPECTED_SEEDS = (20260803, 20260804, 20260805)
EXPECTED_HORIZONS = (1, 5, 20)
METRIC_SPECS = (
    MetricSpec("protocol_score", "lower"),
    MetricSpec("event.information_link_activity.auprc", "higher"),
    MetricSpec("link.active_only_rate.mae", "lower"),
    MetricSpec("task.lifecycle.macro_f1", "higher"),
    MetricSpec("selection.required_continuous.normalized_error", "lower"),
    *(MetricSpec(metric_id, "lower") for metric_id in SELECTION_COMPONENTS),
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_new_output_directory(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    return output


def _safe_relative_path(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"checkpoint path must be a safe relative path: {path}")
    return path


def resolve_checkpoint_path(
    report: Mapping[str, object],
    confirmation_root: str | Path,
    existing_r5_root: str | Path,
) -> Path:
    relative = _safe_relative_path(report["checkpoint"])
    root = Path(existing_r5_root) if str(report["combination_id"]) == "B" else Path(confirmation_root)
    return root / relative


def load_confirmation_bundle_inputs(root: str | Path) -> dict[str, object]:
    root = Path(root).resolve()
    required = (
        "trained_run_reports.json",
        "reused_run_reports.json",
        "validation_windows.json",
        "input_provenance.json",
        "manifest.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"confirmation bundle is missing required files: {missing}")
    provenance = _read_json(root / "input_provenance.json")
    if provenance.get("locked_test_accessed") is not False:
        raise ValueError("confirmation bundle provenance accessed locked-test")
    reports = merge_confirmation_reports(
        _read_json(root / "trained_run_reports.json"),
        _read_json(root / "reused_run_reports.json"),
        expected_combinations=EXPECTED_COMBINATIONS,
        expected_seeds=EXPECTED_SEEDS,
    )
    windows = _read_json(root / "validation_windows.json")
    if set(windows) != {str(seed) for seed in EXPECTED_SEEDS}:
        raise ValueError("confirmation bundle validation-window seed matrix is incomplete")
    return {
        "reports": reports,
        "validation_windows": windows,
        "provenance": provenance,
        "manifest_sha256": _sha256(root / "manifest.json"),
    }


def _load_selection_scales(evaluation_root: Path) -> dict[str, float]:
    payload = _read_json(evaluation_root / "checkpoint_selection_scales.json")
    if payload.get("source_split") != "train":
        raise ValueError("selection scales must be train-only")
    scales = {str(key): float(value) for key, value in payload.get("scales", {}).items()}
    if set(scales) != set(SELECTION_COMPONENTS):
        raise ValueError("selection scales are incomplete")
    return scales


def _load_checkpoint_model(
    report: Mapping[str, object],
    *,
    confirmation_root: Path,
    existing_r5_root: Path,
    confirmation_bindings: Mapping[str, str],
    existing_bindings: Mapping[str, str],
    protocol: object,
):
    path = resolve_checkpoint_path(report, confirmation_root, existing_r5_root)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {path}")
    actual_sha = _sha256(path)
    if actual_sha != str(report["checkpoint_sha256"]):
        raise ValueError(f"checkpoint SHA-256 mismatch: {path}")
    if str(report["combination_id"]) == "B":
        loaded = load_r5_checkpoint(
            path,
            expected_bindings=existing_bindings,
            expected_protocol=protocol,
        )
    else:
        loaded = load_confirmation_checkpoint(
            path,
            expected_bindings=confirmation_bindings,
            expected_protocol=protocol,
        )
    if loaded.seed != int(report["training_seed"]):
        raise ValueError("checkpoint seed differs from its report")
    return loaded.model


def run_analysis(
    *,
    confirmation_root: str | Path,
    existing_r5_root: str | Path,
    dataset_root: str | Path,
    evaluation_root: str | Path,
    output_dir: str | Path,
    micro_batch_size: int = 4,
) -> dict[str, object]:
    confirmation_root = Path(confirmation_root).resolve()
    existing_r5_root = Path(existing_r5_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    output = require_new_output_directory(output_dir)
    bundle = load_confirmation_bundle_inputs(confirmation_root)
    existing_provenance_path = existing_r5_root / "input_provenance.json"
    existing_manifest_path = existing_r5_root / "manifest.json"
    if not existing_provenance_path.is_file() or not existing_manifest_path.is_file():
        raise FileNotFoundError("existing R5 root is missing provenance or manifest")
    existing_provenance = _read_json(existing_provenance_path)
    if existing_provenance.get("locked_test_accessed") is not False:
        raise ValueError("existing R5 provenance accessed locked-test")
    protocol = load_r5_protocol(evaluation_root)
    normalization_stats = _read_json(evaluation_root / "evaluation_normalization_stats.json")
    selection_scales = _load_selection_scales(evaluation_root)
    confirmation_bindings = dict(bundle["provenance"]["bindings"])  # type: ignore[index]
    existing_bindings = dict(existing_provenance["bindings"])

    rebuilt_windows: dict[int, list[object]] = {}
    for seed in EXPECTED_SEEDS:
        windows = build_validation_windows(
            dataset_root,
            split="validation",
            horizons=EXPECTED_HORIZONS,
            seed=seed,
        )
        validate_window_schedule(
            [window.to_dict() for window in windows],
            bundle["validation_windows"][str(seed)],  # type: ignore[index]
        )
        rebuilt_windows[seed] = windows

    reports = list(bundle["reports"])  # type: ignore[arg-type]
    horizon_records: list[dict[str, object]] = []
    reproduction: list[dict[str, object]] = []
    for run_index, report in enumerate(reports, start=1):
        combination = str(report["combination_id"])
        seed = int(report["training_seed"])
        print(f"[{run_index}/{len(reports)}] CPU replay {combination} seed={seed}", flush=True)
        model = _load_checkpoint_model(
            report,
            confirmation_root=confirmation_root,
            existing_r5_root=existing_r5_root,
            confirmation_bindings=confirmation_bindings,
            existing_bindings=existing_bindings,
            protocol=protocol,
        )
        evaluation = evaluate_model_by_horizon_cpu(
            model,
            rebuilt_windows[seed],  # type: ignore[arg-type]
            normalization_stats,
            selection_scales,
            micro_batch_size=micro_batch_size,
        )
        expected_score = float(report["best_validation_protocol_score"])
        actual_score = float(evaluation["overall"]["validation_protocol_score"])  # type: ignore[index]
        delta = abs(actual_score - expected_score)
        tolerance = float(report.get("checkpoint_reproduction_tolerance", 0.0001))
        if delta > tolerance:
            raise ValueError(
                f"CPU checkpoint reproduction failed for {combination}/{seed}: "
                f"expected={expected_score}, actual={actual_score}, delta={delta}, "
                f"tolerance={tolerance}"
            )
        reproduction.append(
            {
                "combination_id": combination,
                "training_seed": seed,
                "expected_score": expected_score,
                "cpu_score": actual_score,
                "absolute_delta": delta,
                "tolerance": tolerance,
            }
        )
        for horizon in EXPECTED_HORIZONS:
            horizon_report = evaluation["per_horizon"][str(horizon)]  # type: ignore[index]
            horizon_records.append(
                {
                    "combination_id": combination,
                    "training_seed": seed,
                    "horizon_steps": horizon,
                    "window_count": horizon_report["window_count"],
                    "validation_protocol_score": horizon_report[
                        "validation_protocol_score"
                    ],
                    "metrics": horizon_report["metrics"],
                }
            )

    aggregate = analyze_confirmation_reports(
        reports,
        metric_specs=METRIC_SPECS,
        expected_combinations=EXPECTED_COMBINATIONS,
        expected_seeds=EXPECTED_SEEDS,
        reference_combination="B",
    )
    aggregate["cpu_checkpoint_reproduction"] = reproduction
    horizons = analyze_horizon_records(
        horizon_records,
        metric_specs=METRIC_SPECS,
        expected_combinations=EXPECTED_COMBINATIONS,
        expected_seeds=EXPECTED_SEEDS,
        expected_horizons=EXPECTED_HORIZONS,
        reference_combination="B",
    )
    frozen = freeze_r6_candidate_set(
        aggregate,
        horizons,
        expected_combinations=EXPECTED_COMBINATIONS,
        reference_combination="B",
        ablation_combinations=("F",),
    )
    write_confirmation_analysis_bundle(
        output,
        aggregate_analysis=aggregate,
        horizon_analysis=horizons,
        horizon_records=horizon_records,
        candidate_freeze=frozen,
        input_binding={
            "confirmation_manifest_sha256": str(bundle["manifest_sha256"]),
            "existing_r5_manifest_sha256": _sha256(existing_manifest_path),
            "dataset_protocol_sha256": _sha256(dataset_root / "protocol.json"),
            "evaluation_protocol_sha256": _sha256(
                evaluation_root / "fair_experiment_protocol.json"
            ),
        },
    )
    print(json.dumps(frozen, ensure_ascii=False, indent=2), flush=True)
    return frozen


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict local R5.1 three-seed and 1/5/20-step analysis"
    )
    parser.add_argument("--confirmation-root", required=True)
    parser.add_argument("--existing-r5-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_analysis(
        confirmation_root=args.confirmation_root,
        existing_r5_root=args.existing_r5_root,
        dataset_root=args.dataset_root,
        evaluation_root=args.evaluation_root,
        output_dir=args.output_dir,
        micro_batch_size=args.micro_batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
