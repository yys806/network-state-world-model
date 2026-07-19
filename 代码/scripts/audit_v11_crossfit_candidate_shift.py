"""Audit candidate-distribution shift after PI-JWM helper seed cross-fitting."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_v11_frozen_selector import infer_candidate_family
from pi_jwm.v11_labeling import (
    load_candidate_interaction_cache,
    load_candidate_label_metadata,
)


def candidate_shift_rows(
    active_sse: np.ndarray,
    active_count: np.ndarray,
    sample_seed: np.ndarray,
    candidate_names: tuple[str, ...],
    default_index: int,
) -> list[dict[str, object]]:
    """Recompute fixed-candidate RMSE and benefit direction from physical totals."""
    sse = np.asarray(active_sse, dtype=np.float64)
    count = np.asarray(active_count, dtype=np.int64).reshape(-1)
    seeds = np.asarray(sample_seed, dtype=np.int64).reshape(-1)
    if sse.ndim != 2 or sse.shape[0] != count.size or count.shape != seeds.shape:
        raise ValueError("candidate shift arrays have incompatible sample dimensions")
    if sse.shape[1] != len(candidate_names) or not 0 <= int(default_index) < sse.shape[1]:
        raise ValueError("candidate names/default index do not match candidate outcomes")
    scored = count > 0
    total_count = int(count[scored].sum())
    rows = []
    for candidate_index, candidate_name in enumerate(candidate_names):
        benefit = sse[:, int(default_index)] - sse[:, candidate_index]
        seed_directions = []
        for seed in sorted(int(value) for value in np.unique(seeds[scored])):
            mask = scored & (seeds == seed)
            seed_directions.append(float(benefit[mask].sum()) > 0.0)
        rows.append(
            {
                "candidate_index": candidate_index,
                "candidate_name": str(candidate_name),
                "action_family": infer_candidate_family(str(candidate_name)),
                "scored": total_count > 0,
                "rmse": (
                    float(np.sqrt(sse[scored, candidate_index].sum() / total_count))
                    if total_count > 0
                    else None
                ),
                "improvement_sse": float(benefit[scored].sum()) if np.any(scored) else 0.0,
                "positive_pair_rate": (
                    float(np.mean(benefit[scored] > 0.0)) if np.any(scored) else None
                ),
                "improved_seed_count": int(sum(seed_directions)),
                "seed_count": len(seed_directions),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stage_family_rows(batch, outcome) -> list[dict[str, Any]]:
    available = np.asarray(batch.candidate_mask, dtype=bool)
    masked_sse = np.where(available, outcome.active_sse, np.inf)
    oracle_choice = np.argmin(masked_sse, axis=1)
    default_sse = outcome.active_sse[:, outcome.default_index]
    oracle_sse = outcome.active_sse[np.arange(oracle_choice.size), oracle_choice]
    opportunity = (outcome.active_count > 0) & (oracle_sse < default_sse)
    oracle_family = np.asarray(
        [infer_candidate_family(batch.candidate_names[int(index)]) for index in oracle_choice]
    )
    rows = []
    for stage in sorted(str(value) for value in np.unique(batch.stage.astype(str))):
        for family in sorted(str(value) for value in np.unique(oracle_family)):
            keep = (batch.stage.astype(str) == stage) & (oracle_family == family)
            if not np.any(keep):
                continue
            rows.append(
                {
                    "stage": stage,
                    "oracle_action_family": family,
                    "sample_count": int(np.sum(keep)),
                    "opportunity_count": int(np.sum(opportunity[keep])),
                    "opportunity_rate": float(np.mean(opportunity[keep])),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-train-cache", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    cache_paths = {
        "old_in_sample_train": args.old_train_cache,
        "oof_train": args.train_cache,
        "calibration": args.calibration_cache,
        "validation": args.validation_cache,
    }
    loaded = {
        name: load_candidate_interaction_cache(path)
        for name, path in cache_paths.items()
    }
    metadata = {
        name: load_candidate_label_metadata(path) for name, path in cache_paths.items()
    }
    for name, (_, _, _, manifest) in loaded.items():
        if str(manifest.get("split_name")) in {"matched_test", "external_holdout"}:
            raise PermissionError(f"locked split is forbidden in crossfit audit: {name}")
    new_manifests = [loaded[name][3] for name in ("oof_train", "calibration", "validation")]
    new_digests = {str(manifest.get("configuration_digest")) for manifest in new_manifests}
    if len(new_digests) != 1:
        raise ValueError("new crossfit cache configuration digests differ")
    old_digest = str(loaded["old_in_sample_train"][3].get("configuration_digest"))
    if old_digest in new_digests:
        raise ValueError("old and new train caches unexpectedly share a configuration digest")
    candidate_orders = {
        tuple(value[0].candidate_names) for value in loaded.values()
    }
    if len(candidate_orders) != 1:
        raise ValueError("candidate order differs across shift-audit caches")

    all_rows = []
    rows_by_split = {}
    stage_family = []
    for split_name, (batch, outcome, _, _) in loaded.items():
        rows = candidate_shift_rows(
            outcome.active_sse,
            outcome.active_count,
            metadata[split_name]["sample_seed"],
            batch.candidate_names,
            outcome.default_index,
        )
        for row in rows:
            row["split_name"] = split_name
            all_rows.append(row)
        rows_by_split[split_name] = rows
        for row in _stage_family_rows(batch, outcome):
            row["split_name"] = split_name
            stage_family.append(row)

    helper_rows = [
        row
        for row in all_rows
        if any(token in str(row["candidate_name"]) for token in ("value_head", "q50", "q75"))
    ]
    old_persistent = [
        row
        for row in rows_by_split["old_in_sample_train"]
        if "value_head" in str(row["candidate_name"])
        and "persistent" in str(row["candidate_name"])
    ]
    new_persistent = [
        row
        for row in rows_by_split["oof_train"]
        if "value_head" in str(row["candidate_name"])
        and "persistent" in str(row["candidate_name"])
    ]
    old_max = max((int(row["improved_seed_count"]) for row in old_persistent), default=0)
    new_max = max((int(row["improved_seed_count"]) for row in new_persistent), default=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "candidate_metrics_by_split.csv", all_rows)
    _write_csv(args.output_dir / "helper_candidate_shift.csv", helper_rows)
    _write_csv(args.output_dir / "stage_family_opportunity.csv", stage_family)
    summary = {
        "framework": "PI-JWM",
        "mode": "v11_selector_helper_crossfit_candidate_shift",
        "result_kind": "diagnostic_only",
        "old_configuration_digest": old_digest,
        "crossfit_configuration_digest": next(iter(new_digests)),
        "old_value_head_persistent_max_improved_seeds": old_max,
        "oof_value_head_persistent_max_improved_seeds": new_max,
        "in_sample_40_of_40_anomaly_removed": old_max == 40 and new_max < 40,
        "matched_test_accessed": False,
        "external_holdout_accessed": False,
        "outputs": {
            "candidate_metrics": "candidate_metrics_by_split.csv",
            "helper_shift": "helper_candidate_shift.csv",
            "stage_family": "stage_family_opportunity.csv",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
