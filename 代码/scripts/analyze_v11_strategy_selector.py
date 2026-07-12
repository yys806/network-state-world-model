"""Offline selector diagnostics for PI-JWM v11 candidate scheduler grids.

The graph-support experiments can contain many operating points.  This script
audits whether the validation split is selecting candidates that generalize to
the matched test rows, and compares simple robust selector rules without
rerunning the world model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_RESULTS = Path("artifacts/experiments/pi_jwm_v11_graph_support_pred_value_ops_small_256_20260628/graph_support_generator_results.csv")


def _float(row: dict, key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, default)
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rows(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def paired_rows(rows: Iterable[dict], family: str = "graph_support_generator") -> list[dict]:
    val_rows = {
        str(row.get("candidate")): row
        for row in rows
        if str(row.get("split")) == "val" and (not family or str(row.get("family")) == family)
    }
    test_rows = {
        str(row.get("candidate")): row
        for row in rows
        if str(row.get("split")) == "test" and (not family or str(row.get("family")) == family)
    }
    pairs = []
    for candidate, val in val_rows.items():
        test = test_rows.get(candidate)
        if test is None:
            continue
        pairs.append({"candidate": candidate, "val": val, "test": test})
    return pairs


def pearson(xs: list[float], ys: list[float]) -> float:
    clean = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(clean) < 2:
        return math.nan
    xs = [x for x, _ in clean]
    ys = [y for _, y in clean]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
    return float(numerator / denominator) if denominator > 0.0 else math.nan


Selector = Callable[[dict], float]


def selector_scores() -> dict[str, Selector]:
    return {
        "val_active": lambda row: _float(row["val"], "active_rate_rmse"),
        "val_active_link_penalty": lambda row: _float(row["val"], "active_rate_rmse") + 0.25 * max(0.0, _float(row["val"], "link_rmse") - 82.0),
        "val_active_f1_bonus": lambda row: _float(row["val"], "active_rate_rmse") - 50.0 * _float(row["val"], "activity_f1", 0.0),
        "val_active_link_f1": lambda row: _float(row["val"], "active_rate_rmse")
        + 0.25 * max(0.0, _float(row["val"], "link_rmse") - 82.0)
        - 50.0 * _float(row["val"], "activity_f1", 0.0),
        "val_link_constrained": lambda row: _float(row["val"], "active_rate_rmse")
        if _float(row["val"], "link_rmse") <= 82.0
        else _float(row["val"], "active_rate_rmse") + 1000.0,
        "val_f1_constrained": lambda row: _float(row["val"], "active_rate_rmse")
        if _float(row["val"], "activity_f1") >= 0.024
        else _float(row["val"], "active_rate_rmse") + 1000.0,
    }


def _candidate_summary(pair: dict) -> dict:
    val = pair["val"]
    test = pair["test"]
    return {
        "candidate": pair["candidate"],
        "selection_group_mode": val.get("selection_group_mode"),
        "top_k": _float(val, "top_k"),
        "alpha": _float(val, "alpha"),
        "new_edge_value_cap": _float(val, "new_edge_value_cap"),
        "val_active_rate_rmse": _float(val, "active_rate_rmse"),
        "val_link_rmse": _float(val, "link_rmse"),
        "val_activity_f1": _float(val, "activity_f1"),
        "test_active_rate_rmse": _float(test, "active_rate_rmse"),
        "test_link_rmse": _float(test, "link_rmse"),
        "test_activity_f1": _float(test, "activity_f1"),
    }


def select_by_rules(pairs: list[dict]) -> dict[str, dict]:
    selections = {}
    for name, scorer in selector_scores().items():
        finite_pairs = [pair for pair in pairs if math.isfinite(scorer(pair))]
        if not finite_pairs:
            continue
        best = min(finite_pairs, key=lambda pair: (scorer(pair), str(pair["candidate"])))
        item = _candidate_summary(best)
        item["selector_score"] = float(scorer(best))
        selections[name] = item
    return selections


def group_summary(pairs: list[dict], fields: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for pair in pairs:
        key = tuple(str(pair["val"].get(field, "")) for field in fields)
        groups[key].append(pair)
    rows = []
    for key, group_pairs in groups.items():
        val_values = [_float(pair["val"], "active_rate_rmse") for pair in group_pairs]
        test_values = [_float(pair["test"], "active_rate_rmse") for pair in group_pairs]
        rows.append(
            {
                **{field: key[idx] for idx, field in enumerate(fields)},
                "count": len(group_pairs),
                "mean_val_active_rate_rmse": float(sum(val_values) / len(val_values)),
                "mean_test_active_rate_rmse": float(sum(test_values) / len(test_values)),
                "best_test_active_rate_rmse": float(min(test_values)),
                "best_val_active_rate_rmse": float(min(val_values)),
            }
        )
    return sorted(rows, key=lambda row: (row["mean_test_active_rate_rmse"], row["mean_val_active_rate_rmse"], str(row)))


def analyze(rows: list[dict], family: str = "graph_support_generator") -> dict:
    pairs = paired_rows(rows, family=family)
    val_active = [_float(pair["val"], "active_rate_rmse") for pair in pairs]
    test_active = [_float(pair["test"], "active_rate_rmse") for pair in pairs]
    best_val = min(pairs, key=lambda pair: (_float(pair["val"], "active_rate_rmse"), str(pair["candidate"]))) if pairs else None
    best_test = min(pairs, key=lambda pair: (_float(pair["test"], "active_rate_rmse"), str(pair["candidate"]))) if pairs else None
    return {
        "paired_count": len(pairs),
        "val_test_active_pearson": pearson(val_active, test_active),
        "best_val_active": _candidate_summary(best_val) if best_val else None,
        "best_test_active": _candidate_summary(best_test) if best_test else None,
        "selector_results": select_by_rules(pairs),
        "group_by_alpha": group_summary(pairs, ("alpha",)),
        "group_by_selection_alpha": group_summary(pairs, ("selection_group_mode", "alpha")),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--family", default="graph_support_generator")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results_csv)
    report = analyze(rows, family=str(args.family))
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "selector_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(args.output_dir / "selector_results.csv", list(report["selector_results"].values()))
        write_csv(args.output_dir / "group_by_alpha.csv", report["group_by_alpha"])
        write_csv(args.output_dir / "group_by_selection_alpha.csv", report["group_by_selection_alpha"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
