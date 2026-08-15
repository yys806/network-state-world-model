"""Freeze PI-JWM R6 service-first reward scales from train system targets only."""

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

from pi_jwm.r6_reward_protocol import (  # noqa: E402
    R6_REWARD_PROTOCOL_VERSION,
    freeze_train_reward_scale,
)


DEFAULT_DATASET_ROOT = (
    CODE_ROOT / "artifacts" / "datasets" / "airfogsim_formal_system_targets_v1"
)
DEFAULT_OUTPUT_DIR = CODE_ROOT / "artifacts" / "protocols" / "pi_jwm_r6_reward_protocol_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-train-trajectories", type=int, default=36)
    return parser.parse_args(argv)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scale = freeze_train_reward_scale(args.dataset_root)
    if scale.train_trajectory_count != int(args.expected_train_trajectories):
        raise ValueError(
            "train trajectory count differs from the frozen R2 protocol: "
            f"{scale.train_trajectory_count} != {args.expected_train_trajectories}"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "reward_scale.json", scale.to_dict())
    _write_json(
        output / "reward_protocol.json",
        {
            "schema_version": R6_REWARD_PROTOCOL_VERSION,
            "primary_terms": {
                "on_time_completion_count": 1.0,
                "failure_count": -1.0,
            },
            "secondary_terms": {
                "completed_delay_sum": -0.1,
                "delivered_data_delta": 0.1,
                "energy_delta": -0.1,
            },
            "secondary_normalization": "train-only positive-step P95 then clip to [0,1]",
            "hard_constraint_semantics": "invalid_transition_not_scalar_penalty",
            "episode_only_metrics": [
                "latency_p95",
                "latency_p99",
                "rb_utilization",
                "cpu_utilization",
                "jain_fairness",
                "action_regret",
                "uncertainty_calibration",
            ],
            "locked_test_accessed": False,
            "gpu_used": False,
        },
    )
    files = [output / "reward_protocol.json", output / "reward_scale.json"]
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "PIJWM-R6-reward-protocol-bundle-v1",
            "files": [
                {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in files
            ],
            "source_dataset": str(args.dataset_root.resolve()),
            "source_manifest_sha256": scale.source_manifest_sha256,
            "locked_test_accessed": False,
            "gpu_used": False,
        },
    )
    print(json.dumps(scale.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
