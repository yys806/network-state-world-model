"""Align frozen selector decisions with actual AirFogSim task/energy evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _time_key(value: Any) -> float:
    return round(float(value), 6)


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def build_physical_safety_rows(
    decision_rows: Sequence[Mapping[str, Any]],
    sample_index_rows: Sequence[Mapping[str, Any]],
    physical_rows: Sequence[Mapping[str, Any]],
    candidate_mapping: Mapping[str, str],
    default_candidate_id: str = "default",
) -> list[dict[str, Any]]:
    """Join only exact sample/seed/time/candidate matches; never impute physical outcomes."""
    sample_index = {int(row["sample_id"]): row for row in sample_index_rows}
    physical_index = {
        (int(row["seed"]), _time_key(row["decision_time"]), str(row["candidate_id"])): row
        for row in physical_rows
    }
    result = []
    missing = []
    for decision in decision_rows:
        sample_id = int(decision["sample_id"])
        index_row = sample_index.get(sample_id)
        if index_row is None:
            missing.append(f"sample_id={sample_id}:sample_index")
            continue
        seed = int(decision["seed"])
        if int(index_row["seed"]) != seed:
            raise ValueError(f"sample {sample_id} seed differs between decision trace and sample index")
        decision_time = _time_key(index_row.get("input_end_time", index_row.get("decision_time")))
        selected_name = str(decision["candidate_name"])
        selected_id = (
            str(default_candidate_id)
            if _is_true(decision.get("deferred", False))
            else candidate_mapping.get(selected_name)
        )
        if selected_id is None:
            missing.append(f"sample_id={sample_id}:mapping:{selected_name}")
            continue
        baseline = physical_index.get((seed, decision_time, str(default_candidate_id)))
        selected = physical_index.get((seed, decision_time, str(selected_id)))
        if baseline is None or selected is None:
            missing.append(f"sample_id={sample_id}:physical:{selected_id}")
            continue
        task_delta = float(selected["task_utility"]) - float(baseline["task_utility"])
        energy_delta = float(selected["energy_total"]) - float(baseline["energy_total"])
        result.append(
            {
                "sample_id": sample_id,
                "seed": seed,
                "decision_time": decision_time,
                "candidate_name": selected_name,
                "physical_candidate_id": str(selected_id),
                "task_utility_delta_actual": task_delta,
                "energy_delta_actual": energy_delta,
                "unsafe_task_down_energy_up": bool(task_delta < 0.0 and energy_delta > 0.0),
            }
        )
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"physical safety alignment is incomplete ({len(missing)} rows): {preview}")
    if not result:
        raise ValueError("physical safety alignment produced no rows")
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run(args: argparse.Namespace) -> dict[str, Any]:
    frozen = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
    mapping = json.loads(args.candidate_mapping.read_text(encoding="utf-8"))
    rows = build_physical_safety_rows(
        _read_csv(args.decision_trace),
        _read_csv(args.sample_index),
        _read_csv(args.physical_candidate_summary),
        mapping,
        default_candidate_id=args.default_candidate_id,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "framework": "PI-JWM",
        "result_kind": "actual_airfogsim_safety_audit",
        "configuration_digest": frozen["configuration_digest"],
        "selector_freeze_digest": frozen["selector_freeze_digest"],
        "num_aligned_rows": len(rows),
        "pareto_violations": sum(bool(row["unsafe_task_down_energy_up"]) for row in rows),
        "rows": rows,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--decision-trace", type=Path, required=True)
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument("--physical-candidate-summary", type=Path, required=True)
    parser.add_argument("--candidate-mapping", type=Path, required=True)
    parser.add_argument("--default-candidate-id", default="default")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
