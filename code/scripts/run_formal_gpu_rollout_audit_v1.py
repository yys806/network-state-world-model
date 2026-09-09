"""Audit horizon-wise rollout metrics from completed formal GPU runs."""

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

from pi_jwm.formal_gpu_rollout_audit_v1 import audit_gpu_rollout_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for run_dir in args.run_dir:
        metrics_dir = Path(run_dir) / "metrics"
        report = audit_gpu_rollout_metrics(metrics_dir)
        report["run_dir"] = str(Path(run_dir).resolve())
        reports.append(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "PI-JWM-formal-gpu-rollout-multiseed-audit-v1",
        "audit_passed": all(report["audit_passed"] for report in reports),
        "formal_performance_claim_ready": False,
        "seed_count": len(reports),
        "reports": reports,
        "execution_policy": {"locked_test_accessed": False},
    }
    (args.output_dir / "gpu_rollout_multiseed_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "horizon_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "run_dir", "horizon", "node_x_mae_delta", "link_active_rate_mae_delta",
            "throughput_mae_delta", "rb_occupancy_mae_delta", "node_x_coverage_95_delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for horizon, row in report["per_horizon"].items():
                writer.writerow({"run_dir": report["run_dir"], "horizon": horizon, **{field: row[field] for field in fields[2:]}})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
