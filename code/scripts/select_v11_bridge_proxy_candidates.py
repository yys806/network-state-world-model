"""Select PI-JWM v11 policy checkpoints with multi-objective bridge-proxy gates."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select bridge-proxy candidates from bridge JSON metrics.")
    parser.add_argument("--input-glob", action="append", required=True, help="Glob for bridge JSON files. Can be repeated.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--min-f1", type=float, default=0.23)
    parser.add_argument("--max-link-rmse", type=float, default=40.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--allow-fallback", action="store_true")
    return parser.parse_args()


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        paths.extend(Path(item) for item in matched)
    deduped = sorted({str(path): path for path in paths}.values(), key=lambda path: str(path))
    return deduped


def read_candidate(path: Path, split: str, min_f1: float, max_link_rmse: float) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    metrics = payload[split]
    active_rate = metrics["active_rate"]
    activity = metrics["activity"]
    link_rate = metrics["link_rate"]
    positive_rate = metrics.get("positive_rate_active", {})
    active_rmse = float(active_rate["active_rmse"])
    f1 = float(activity["f1"])
    link_rmse = float(link_rate["rmse"])
    return {
        "path": str(path),
        "policy_checkpoint": payload.get("policy_checkpoint"),
        "policy_threshold": payload.get("policy_threshold"),
        "split": split,
        "active_rate_rmse": active_rmse,
        "active_rate_mae": float(active_rate.get("active_mae", float("nan"))),
        "activity_f1": f1,
        "activity_precision": float(activity.get("precision", float("nan"))),
        "activity_recall": float(activity.get("recall", float("nan"))),
        "activity_tp": float(activity.get("tp", float("nan"))),
        "activity_fp": float(activity.get("fp", float("nan"))),
        "activity_fn": float(activity.get("fn", float("nan"))),
        "link_rate_rmse": link_rmse,
        "positive_rate_active_rmse": float(positive_rate.get("active_rmse", float("nan"))),
        "passes_gate": f1 >= float(min_f1) and link_rmse <= float(max_link_rmse),
    }


def sort_key(candidate: dict) -> tuple[float, float, float]:
    return (
        float(candidate["active_rate_rmse"]),
        float(candidate["link_rate_rmse"]),
        -float(candidate["activity_f1"]),
    )


def select_candidates(
    paths: Iterable[Path],
    split: str = "test",
    min_f1: float = 0.23,
    max_link_rmse: float = 40.0,
    top_k: int = 3,
    allow_fallback: bool = False,
) -> list[dict]:
    candidates = [read_candidate(Path(path), split=split, min_f1=min_f1, max_link_rmse=max_link_rmse) for path in paths]
    gated = [candidate for candidate in candidates if candidate["passes_gate"]]
    pool = gated
    if not pool and allow_fallback:
        pool = candidates
    return sorted(pool, key=sort_key)[: max(int(top_k), 0)]


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.input_glob)
    selected = select_candidates(
        paths,
        split=args.split,
        min_f1=args.min_f1,
        max_link_rmse=args.max_link_rmse,
        top_k=args.top_k,
        allow_fallback=args.allow_fallback,
    )
    result = {
        "input_count": len(paths),
        "split": args.split,
        "min_f1": float(args.min_f1),
        "max_link_rmse": float(args.max_link_rmse),
        "top_k": int(args.top_k),
        "allow_fallback": bool(args.allow_fallback),
        "selected": selected,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
