from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_cpu_gpu_gate_v1 import audit_cpu_to_gpu_gate


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"missing numeric comparison field: {key}")
    return float(value)


def _load_seed_row(run_dir: Path, learned_method: str) -> dict[str, Any]:
    with (run_dir / "comparison.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_method = {row["method"]: row for row in rows}
    learned = by_method[learned_method]
    persistence = by_method["last_persistence"]
    threshold_report = json.loads(
        (run_dir / "metrics" / f"{learned_method}__threshold_selection.json").read_text(encoding="utf-8")
    )
    result: dict[str, Any] = {
        "seed": json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["seed"],
        "threshold_selection_split": threshold_report.get("selection_split"),
    }
    for metric in ("link_f1", "node_x_mae", "throughput_mae", "rb_occupancy_mae", "task_delay_mae"):
        result[f"validation_{metric}"] = _float(learned, f"validation_{metric}")
        result[f"validation_persistence_{metric}"] = _float(persistence, f"validation_{metric}")
    result["calibration_link_f1"] = _float(learned, "calibration_link_f1")
    result["calibration_persistence_link_f1"] = _float(persistence, "calibration_link_f1")
    return result


def run_gate(*, run_dirs: list[str | Path], output_dir: str | Path, learned_method: str) -> dict[str, Any]:
    rows = [_load_seed_row(Path(run_dir), learned_method) for run_dir in run_dirs]
    report = audit_cpu_to_gpu_gate(rows)
    report["sources"] = [str(Path(run_dir).resolve()) for run_dir in run_dirs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cpu_to_gpu_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the frozen non-locked CPU-to-GPU launch gate.")
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learned-method", default="coupled_dual_gnn_residual")
    args = parser.parse_args()
    report = run_gate(run_dirs=args.run_dir, output_dir=args.output_dir, learned_method=args.learned_method)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["gpu_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
