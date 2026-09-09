"""Run reproducible CPU key-wise input-perturbation diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from run_formal_gpu_robustness_v1 import METRIC_NAMES


KEYWISE_PERTURBATION_KEYS = (
    "node_state",
    "physical_edge_state",
    "flow_state",
    "task_state",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    reports: list[dict] = []
    single_key_script = Path(__file__).with_name("run_formal_gpu_robustness_v1.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for perturb_key in KEYWISE_PERTURBATION_KEYS:
        key_output_dir = args.output_dir / f"perturb_{perturb_key}"
        command = [
            sys.executable,
            str(single_key_script),
            "--tensor-root",
            str(args.tensor_root),
            "--output-dir",
            str(key_output_dir),
            "--device",
            args.device,
            "--perturb-key",
            perturb_key,
        ]
        for run_dir in args.run_dir:
            command.extend(("--run-dir", str(run_dir)))
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"key-wise perturbation failed for {perturb_key}: {completed.stderr[-2000:]}"
            )
        key_payload = json.loads(
            (key_output_dir / "gpu_robustness_multiseed.json").read_text(encoding="utf-8")
        )
        reports.extend(
            {"perturb_key": perturb_key, **report}
            for report in key_payload["reports"]
        )

    output = {
        "schema_version": "PI-JWM-formal-gpu-robustness-keywise-multiseed-v1",
        "audit_passed": all(report["audit_passed"] for report in reports),
        "formal_performance_claim_ready": False,
        "perturbation_keys": list(KEYWISE_PERTURBATION_KEYS),
        "seed_count": len(args.run_dir),
        "reports": reports,
        "execution_policy": {
            "locked_test_accessed": False,
            "evaluation_devices": sorted(
                {report["execution_policy"]["evaluation_device"] for report in reports}
            ),
            "gpu_execution": any(
                report["execution_policy"]["gpu_execution"] for report in reports
            ),
        },
    }
    (args.output_dir / "gpu_robustness_keywise_multiseed.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = [
        "perturb_key",
        "seed",
        "noise_scale",
        *METRIC_NAMES,
        *[f"delta_{name}" for name in METRIC_NAMES],
    ]
    with (args.output_dir / "robustness_keywise_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for row in report["reports"]:
                writer.writerow(
                    {
                        "perturb_key": report["perturb_key"],
                        "seed": report["seed"],
                        "noise_scale": row["noise_scale"],
                        **row["metrics"],
                        **{
                            f"delta_{name}": value
                            for name, value in row["metric_deltas_from_clean"].items()
                        },
                    }
                )
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if output["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
