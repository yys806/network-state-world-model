"""Analyze the completed PI-JWM R6 10k online-policy health-gate matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_10k_analysis import (  # noqa: E402
    analyze_r6_10k_records,
    write_r6_10k_analysis_bundle,
)
from pi_jwm.r6_gpu_training_protocol import build_default_gpu_training_protocol  # noqa: E402


DEFAULT_INPUT = (
    CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r6_gpu_training_v3"
)
DEFAULT_OUTPUT = (
    CODE_ROOT / "artifacts" / "analysis" / "pi_jwm_r6_10k_gate_analysis_v1"
)
CONTROL_FILES = ("run_records.json", "matrix_summary.json", "launch_manifest.json")


def build_input_binding(input_root: str | Path) -> dict[str, str]:
    root = Path(input_root)
    binding: dict[str, str] = {}
    for name in CONTROL_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        binding[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return binding


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    records = json.loads((input_root / "run_records.json").read_text(encoding="utf-8"))
    matrix = json.loads((input_root / "matrix_summary.json").read_text(encoding="utf-8"))
    protocol = build_default_gpu_training_protocol()
    if int(matrix.get("expected_count", -1)) != len(protocol.formal_runs()):
        raise ValueError("matrix expected_count differs from frozen R6 protocol")
    if int(matrix.get("complete_count", -1)) != len(protocol.formal_runs()):
        raise ValueError("R6 10k matrix is incomplete")
    if int(matrix.get("failed_count", -1)) != 0:
        raise ValueError("R6 10k matrix contains failed runs")
    if matrix.get("locked_test_accessed") is not False:
        raise PermissionError("R6 10k matrix accessed locked-test")
    target_steps = int(matrix["target_environment_steps"])
    analysis = analyze_r6_10k_records(
        records,
        expected_methods=protocol.methods,
        expected_state_modes=protocol.state_modes,
        expected_seeds=protocol.seeds,
        target_environment_steps=target_steps,
    )
    write_r6_10k_analysis_bundle(
        analysis,
        output_dir,
        input_binding=build_input_binding(input_root),
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "completed_run_count": analysis["integrity"]["completed_run_count"],
                "continuation_gate": analysis["continuation_gate"]["status"],
                "locked_test_accessed": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
