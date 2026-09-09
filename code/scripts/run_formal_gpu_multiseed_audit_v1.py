"""Generate a reproducible audit for completed formal non-locked GPU seeds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_gpu_multiseed_audit_v1 import audit_gpu_multiseed_runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit_gpu_multiseed_runs(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gpu_multiseed_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "per_seed_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "seed",
            "manifest_mismatch_count",
            "validation_link_f1_delta",
            "calibration_link_f1_delta",
            "validation_node_x_mae_delta",
            "validation_throughput_mae_delta",
            "validation_rb_occupancy_mae_delta",
            "validation_task_delay_mae_delta",
            "threshold_selection_split",
            "gpu_execution",
            "locked_test_accessed",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["records"]:
            writer.writerow(
                {
                    "seed": row["seed"],
                    "manifest_mismatch_count": row["manifest_mismatch_count"],
                    "validation_link_f1_delta": row["validation_deltas"]["link_f1"],
                    "calibration_link_f1_delta": row["calibration_link_f1_delta"],
                    "validation_node_x_mae_delta": row["validation_deltas"]["node_x_mae"],
                    "validation_throughput_mae_delta": row["validation_deltas"]["throughput_mae"],
                    "validation_rb_occupancy_mae_delta": row["validation_deltas"]["rb_occupancy_mae"],
                    "validation_task_delay_mae_delta": row["validation_deltas"]["task_delay_mae"],
                    "threshold_selection_split": row["threshold_selection_split"],
                    "gpu_execution": row["gpu_execution"],
                    "locked_test_accessed": row["locked_test_accessed"],
                }
            )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
