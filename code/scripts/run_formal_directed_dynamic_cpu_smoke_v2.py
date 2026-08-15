"""Run a nonlocked CPU smoke for PI-JWM directed-dynamic v2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from run_formal_directed_dynamic_gpu_train_v2 import (
    DIRECTED_DYNAMIC_METHODS,
    run_directed_dynamic_training,
)
from run_formal_dual_graph_gpu_train_v1 import NONLOCKED_SPLITS


def validate_cpu_protocol_v2(splits: Iterable[str], device: str) -> None:
    requested = tuple(str(split) for split in splits)
    if "locked_test" in requested:
        raise ValueError("locked_test cannot be used by the directed-dynamic CPU smoke")
    unknown = set(requested) - set(NONLOCKED_SPLITS)
    if unknown:
        raise ValueError(f"unsupported CPU splits: {sorted(unknown)}")
    if str(device) != "cpu":
        raise ValueError("the directed-dynamic CPU smoke requires the CPU device")


def run_cpu_smoke_v2(
    *,
    tensor_root: str | Path,
    output_dir: str | Path,
    seed: int = 20260802,
    train_limit: int = 4,
    evaluation_limit: int = 2,
    hidden_dim: int = 8,
    epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
) -> dict[str, Any]:
    validate_cpu_protocol_v2(NONLOCKED_SPLITS, "cpu")
    return run_directed_dynamic_training(
        tensor_root=tensor_root,
        output_dir=output_dir,
        device="cpu",
        learned_methods=DIRECTED_DYNAMIC_METHODS,
        seed=seed,
        train_limit=train_limit,
        evaluation_limit=evaluation_limit,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_workers=0,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--train-limit", type=int, default=4)
    parser.add_argument("--evaluation-limit", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_cpu_smoke_v2(
        tensor_root=args.tensor_root,
        output_dir=args.output_dir,
        seed=args.seed,
        train_limit=args.train_limit,
        evaluation_limit=args.evaluation_limit,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print(result)


if __name__ == "__main__":
    main()
