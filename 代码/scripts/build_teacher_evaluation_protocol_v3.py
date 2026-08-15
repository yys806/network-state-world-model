"""Build the frozen PI-JWM R2 evaluation protocol and baseline bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.evaluation_bundle_v3 import build_evaluation_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=CODE_ROOT / "artifacts" / "datasets" / "airfogsim_teacher_aligned_v3",
    )
    parser.add_argument(
        "--factual-metrics-csv",
        type=Path,
        default=CODE_ROOT / "artifacts" / "datasets" / "airfogsim_formal_v1" / "metrics_by_trajectory.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_ROOT / "artifacts" / "evaluation" / "pi_jwm_eval_protocol_v3",
    )
    args = parser.parse_args()
    result = build_evaluation_bundle(
        args.dataset_root, args.factual_metrics_csv, args.output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
