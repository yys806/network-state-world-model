"""Train PI-JWM directed-dynamic v2 models with the formal nonlocked protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from run_formal_dual_graph_gpu_train_v1 import (
    NONLOCKED_SPLITS,
    run_formal_training,
    validate_gpu_protocol,
)


DIRECTED_DYNAMIC_METHODS = (
    "coupled_directed_dynamic_v2",
    "coupled_directed_dynamic_residual_v2",
)


def validate_directed_dynamic_gpu_protocol(
    splits: Iterable[str], device: str
) -> None:
    validate_gpu_protocol(splits, device)


def run_directed_dynamic_training(
    *,
    learned_methods: Sequence[str] = DIRECTED_DYNAMIC_METHODS,
    **kwargs: Any,
) -> dict[str, Any]:
    unknown = set(learned_methods) - set(DIRECTED_DYNAMIC_METHODS)
    if unknown:
        raise ValueError(f"unsupported directed-dynamic methods: {sorted(unknown)}")
    if not learned_methods:
        raise ValueError("at least one directed-dynamic method is required")
    return run_formal_training(learned_methods=tuple(learned_methods), **kwargs)


def run_directed_dynamic_gpu_training(**kwargs: Any) -> dict[str, Any]:
    device = str(kwargs.get("device", "cuda"))
    validate_directed_dynamic_gpu_protocol(NONLOCKED_SPLITS, device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    kwargs["device"] = device
    return run_directed_dynamic_training(**kwargs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--system-root", type=Path)
    parser.add_argument("--use-system-energy-head", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--learned-methods",
        nargs="+",
        choices=DIRECTED_DYNAMIC_METHODS,
        default=list(DIRECTED_DYNAMIC_METHODS),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_directed_dynamic_gpu_training(
        tensor_root=args.tensor_root,
        system_root=args.system_root,
        use_system_energy_head=args.use_system_energy_head,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        train_limit=args.train_limit,
        evaluation_limit=args.evaluation_limit,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        learned_methods=args.learned_methods,
    )
    print(result)


if __name__ == "__main__":
    main()
