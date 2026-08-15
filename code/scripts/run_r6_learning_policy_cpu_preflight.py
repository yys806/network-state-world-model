"""Run the formal CPU-only learning-policy preflight for PI-JWM R6."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_learning_policy_preflight import (  # noqa: E402
    load_real_frozen_policy_state,
    run_policy_cpu_smoke,
    write_preflight_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=CODE_ROOT / "artifacts/datasets/airfogsim_teacher_aligned_v3",
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=CODE_ROOT / "artifacts/evaluation/pi_jwm_eval_protocol_v3",
    )
    parser.add_argument(
        "--r5-training-root",
        type=Path,
        default=CODE_ROOT / "artifacts/formal_training/pi_jwm_r5_gpu_training_v1",
    )
    parser.add_argument(
        "--r5-analysis-root",
        type=Path,
        default=CODE_ROOT / "artifacts/formal_training/pi_jwm_r5_module_confirmation_analysis_v1",
    )
    parser.add_argument(
        "--r6-paired-root",
        type=Path,
        default=CODE_ROOT / "artifacts/formal_training/pi_jwm_r6_cpu_paired_closed_loop_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_ROOT / "artifacts/preflight/pi_jwm_r6_learning_policy_cpu_preflight_v1",
    )
    parser.add_argument("--training-seed", type=int, default=20260803)
    parser.add_argument("--policy-seed", type=int, default=20260808)
    parser.add_argument("--hidden-dim", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state, state_audit, bindings = load_real_frozen_policy_state(
        dataset_root=args.dataset_root,
        evaluation_root=args.evaluation_root,
        r5_training_root=args.r5_training_root,
        r5_analysis_root=args.r5_analysis_root,
        r6_paired_root=args.r6_paired_root,
        training_seed=args.training_seed,
    )
    summary = run_policy_cpu_smoke(
        state,
        hidden_dim=args.hidden_dim,
        seed=args.policy_seed,
    )
    summary.update(
        {
            "formal_real_state_used": True,
            "world_model_candidate": "B",
            "world_model_training_seed": int(args.training_seed),
            "offload_training_enabled": False,
            "rb_training_enabled": False,
            "cpu_training_interface_enabled": True,
            "claim_boundary": (
                "CPU interface, mask, projection, finite-gradient and frozen-world-model "
                "preflight only; no policy performance or final-method claim."
            ),
        }
    )
    write_preflight_bundle(
        args.output_dir,
        summary=summary,
        bindings=bindings,
        state_audit=state_audit,
        action_rows=list(summary["action_rows"]),
        failures=[],
    )
    print(args.output_dir.resolve())
    print(f"r6_learning_policy_cpu_ready={summary['r6_learning_policy_cpu_ready']}")
    return 0 if summary["r6_learning_policy_cpu_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
