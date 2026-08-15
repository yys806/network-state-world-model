"""Merge five PI-JWM v11 out-of-fold schema-v6 train label caches."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.v11_crossfit import (
    build_seed_crossfit_folds,
    merge_crossfit_label_caches,
)
from pi_jwm.v11_labeling import load_candidate_label_metadata


def validate_fold_cache_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    """Require exactly five distinct cache paths."""
    resolved = tuple(Path(path) for path in paths)
    if len(resolved) != 5:
        raise ValueError("crossfit merge requires exactly five fold cache paths")
    if len({str(path.resolve()) for path in resolved}) != 5:
        raise ValueError("crossfit fold cache paths must be distinct")
    return resolved


def _limit_fold_rows(
    rows: list[tuple[int, int]], held_out_seeds: tuple[int, ...], limit: int
) -> list[tuple[int, int]]:
    selected_rows = [row for row in rows if row[1] in set(held_out_seeds)]
    if int(limit) <= 0 or len(selected_rows) <= int(limit):
        return selected_rows
    buckets = {
        seed: [row for row in selected_rows if row[1] == seed]
        for seed in held_out_seeds
    }
    selected = []
    offset = 0
    while len(selected) < int(limit):
        changed = False
        for seed in held_out_seeds:
            if offset < len(buckets[seed]) and len(selected) < int(limit):
                selected.append(buckets[seed][offset])
                changed = True
        if not changed:
            break
        offset += 1
    return sorted(selected)


def expected_train_samples(
    sample_index_csv: Path, sample_limit_per_fold: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the exact formal or smoke OOF sample coverage."""
    with Path(sample_index_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            (int(row["sample_id"]), int(row["seed"]))
            for row in csv.DictReader(handle)
        ]
    selected = []
    for fold in build_seed_crossfit_folds():
        selected.extend(
            _limit_fold_rows(rows, fold.held_out_seeds, int(sample_limit_per_fold))
        )
    selected.sort()
    sample_ids = np.asarray([row[0] for row in selected], dtype=np.int64)
    sample_seed = np.asarray([row[1] for row in selected], dtype=np.int64)
    if int(sample_limit_per_fold) <= 0 and sample_ids.size != 15600:
        raise ValueError(
            f"formal crossfit sample index must contain 15600 train samples; found {sample_ids.size}"
        )
    return sample_ids, sample_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-cache", type=Path, action="append", required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--sample-index-csv", type=Path, required=True)
    parser.add_argument("--sample-limit-per-fold", type=int, default=0)
    parser.add_argument("--expected-configuration-digest", required=True)
    parser.add_argument("--expected-crossfit-protocol-digest", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    paths = validate_fold_cache_paths(args.fold_cache)
    sample_ids, sample_seed = expected_train_samples(
        args.sample_index_csv, args.sample_limit_per_fold
    )
    manifest = merge_crossfit_label_caches(
        paths,
        args.output_cache,
        expected_sample_ids=sample_ids,
        expected_sample_seed=sample_seed,
        expected_crossfit_protocol_digest=args.expected_crossfit_protocol_digest,
        expected_configuration_digest=args.expected_configuration_digest,
    )
    metadata = load_candidate_label_metadata(args.output_cache)
    summary = {
        "framework": "PI-JWM",
        "mode": "v11_selector_helper_seed_crossfit_merge",
        "result_kind": "diagnostic_only",
        "sample_count": int(metadata["sample_ids"].size),
        "seed_values": sorted(int(value) for value in np.unique(metadata["sample_seed"])),
        "fold_ids": sorted(int(value) for value in np.unique(metadata["sample_fold_id"])),
        "configuration_digest": str(manifest["configuration_digest"]),
        "crossfit_protocol_digest": str(args.expected_crossfit_protocol_digest),
        "merged_cache_sha256": str(manifest["cache_sha256"]),
        "source_fold_caches": manifest["protocol_metadata"]["source_fold_caches"],
    }
    summary_path = Path(args.output_cache).parent / "merge_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
