from __future__ import annotations

"""Build the non-training P2-A CPU inner-rule preflight evidence bundle."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


CODE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.airfogsim_cpu_inner_rule_v1 import (  # noqa: E402
    allocate_airfogsim_cpu,
    load_airfogsim_task_class_from_source,
)
from pi_jwm.cpu_inner_rule_v1 import (  # noqa: E402
    CPU_INNER_RULE_VERSION,
    CpuTaskDemand,
    allocate_work_conserving_cpu,
)


PREFLIGHT_VERSION = "PIJWM-P2-A-CPU-Preflight-v1"
DEFAULT_OUTPUT_DIR = CODE_ROOT / "artifacts" / "preflight" / "pi_jwm_cpu_inner_rule_v1"
DEFAULT_DESIGN_PATH = (
    WORKSPACE_ROOT
    / "记录"
    / "研究进展"
    / "2026-08-13-PI-JWM-P1-A-CPU动作边界冻结设计.md"
)
DEFAULT_AIRFOGSIM_ROOT = CODE_ROOT / "reference" / "AirFogSim"
CORE_MODULE_PATH = SRC_ROOT / "pi_jwm" / "cpu_inner_rule_v1.py"
ADAPTER_MODULE_PATH = SRC_ROOT / "pi_jwm" / "airfogsim_cpu_inner_rule_v1.py"
PURE_TEST_PATH = CODE_ROOT / "tests" / "test_cpu_inner_rule_v1.py"
ADAPTER_TEST_PATH = CODE_ROOT / "tests" / "test_airfogsim_cpu_inner_rule_v1.py"
PREFLIGHT_TEST_PATH = CODE_ROOT / "tests" / "test_run_cpu_inner_rule_preflight_v1.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": resolved.as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def published_file_metadata(source_path: Path, published_path: Path) -> dict[str, Any]:
    metadata = file_metadata(source_path)
    metadata["path"] = Path(published_path).resolve().as_posix()
    return metadata


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sample_row(
    *,
    case_id: str,
    source: str,
    node_id: str,
    task_id: str,
    remaining_work: float,
    capacity: float,
    slot_seconds: float,
    allocated_cpu: float,
    served_work: float,
    passed: bool,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "source": source,
        "node_id": node_id,
        "task_id": task_id,
        "remaining_work": remaining_work,
        "capacity": capacity,
        "slot_seconds": slot_seconds,
        "allocated_cpu": allocated_cpu,
        "served_work": served_work,
        "sample_origin": "contract_fixture",
        "training_eligible": False,
        "passed": passed,
    }


def _run_case(
    case_id: str,
    tasks: list[CpuTaskDemand],
    capacities: dict[str, float],
    slot_seconds: float,
    expected: dict[str, float],
) -> list[dict[str, Any]]:
    first = allocate_work_conserving_cpu(tasks, capacities, slot_seconds)
    second = allocate_work_conserving_cpu(reversed(tasks), capacities, slot_seconds)
    if first != second:
        raise AssertionError(f"nondeterministic output for case: {case_id}")
    allocations = first.as_allocation_dict()
    if set(allocations) != set(expected):
        raise AssertionError(f"task set mismatch for case: {case_id}")
    for task_id, expected_value in expected.items():
        if not math.isclose(allocations[task_id], expected_value, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(
                f"allocation mismatch for {case_id}/{task_id}: "
                f"{allocations[task_id]} != {expected_value}"
            )
    for summary in first.node_summaries:
        target = min(summary.capacity, summary.total_demand_rate)
        if not math.isclose(
            summary.total_allocated_cpu,
            target,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"work conservation mismatch for case: {case_id}")

    rows = []
    if not first.allocations:
        rows.append(
            _sample_row(
                case_id=case_id,
                source="pure_rule",
                node_id="",
                task_id="",
                remaining_work=0.0,
                capacity=0.0,
                slot_seconds=slot_seconds,
                allocated_cpu=0.0,
                served_work=0.0,
                passed=True,
            )
        )
    for allocation in first.allocations:
        rows.append(
            _sample_row(
                case_id=case_id,
                source="pure_rule",
                node_id=allocation.node_id,
                task_id=allocation.task_id,
                remaining_work=allocation.remaining_work,
                capacity=capacities[allocation.node_id],
                slot_seconds=slot_seconds,
                allocated_cpu=allocation.allocated_cpu,
                served_work=allocation.served_work,
                passed=True,
            )
        )
    return rows


class _Node:
    def __init__(self, capacity: float) -> None:
        self.capacity = capacity

    def getFogProfile(self) -> dict[str, float]:
        return {"cpu": self.capacity}


class _Env:
    simulation_interval = 0.5

    def __init__(self, capacities: dict[str, float]) -> None:
        self.nodes = {node_id: _Node(capacity) for node_id, capacity in capacities.items()}

    def _getNodeById(self, node_id: str):
        return self.nodes.get(node_id)


def _real_airfogsim_task(Task, task_id: str, node_id: str, total_cpu: float, computed_cpu: float):
    task = Task(
        task_id=task_id,
        task_node_id=node_id,
        task_cpu=total_cpu,
        task_size=1.0,
        task_deadline=10.0,
        task_priority=1.0,
        task_arrival_time=0.0,
    )
    task.setAssignedTo(node_id)
    task.setAttribute("_routes", [node_id])
    task.setAttribute("_computed_size", computed_cpu)
    task.startToCompute(0.0)
    return task


def _callback_parity_rows(airfogsim_root: Path) -> list[dict[str, Any]]:
    Task = load_airfogsim_task_class_from_source(airfogsim_root)
    env = _Env({"RSU_0": 6.0})
    task_a = _real_airfogsim_task(Task, "task_a", "RSU_0", 1.0, 0.0)
    task_b = _real_airfogsim_task(Task, "task_b", "RSU_0", 5.0, 1.0)
    adapted = allocate_airfogsim_cpu(
        env,
        {"RSU_0": [task_b, task_a]},
        slot_seconds=0.5,
    )
    pure = allocate_work_conserving_cpu(
        [
            CpuTaskDemand("task_b", "RSU_0", 4.0),
            CpuTaskDemand("task_a", "RSU_0", 1.0),
        ],
        {"RSU_0": 6.0},
        0.5,
    )
    if adapted.decision != pure or adapted.allocations != pure.as_allocation_dict():
        raise AssertionError("AirFogSim Task callback adapter differs from pure CPU rule")
    if adapted.source_task_classes != ("airfogsim.entities.task.Task",):
        raise AssertionError(f"unexpected AirFogSim source task class: {adapted.source_task_classes}")
    return [
        _sample_row(
            case_id="airfogsim_callback_parity",
            source="airfogsim_real_task_source_interface",
            node_id=row.node_id,
            task_id=row.task_id,
            remaining_work=row.remaining_work,
            capacity=6.0,
            slot_seconds=0.5,
            allocated_cpu=row.allocated_cpu,
            served_work=row.served_work,
            passed=True,
        )
        for row in adapted.decision.allocations
    ]


def _expected_rejections(airfogsim_root: Path) -> list[dict[str, Any]]:
    Task = load_airfogsim_task_class_from_source(airfogsim_root)
    mismatch_task = _real_airfogsim_task(Task, "task_mismatch", "RSU_0", 1.0, 0.0)
    mismatch_task.setAssignedTo("RSU_1")
    fixtures: list[tuple[str, str, Callable[[], Any]]] = [
        (
            "negative_remaining_work",
            "invalid_numeric_value",
            lambda: allocate_work_conserving_cpu(
                [CpuTaskDemand("task_a", "node_0", -1.0)], {"node_0": 1.0}, 1.0
            ),
        ),
        (
            "missing_node_capacity",
            "missing_required_relation",
            lambda: allocate_work_conserving_cpu(
                [CpuTaskDemand("task_a", "node_0", 1.0)], {}, 1.0
            ),
        ),
        (
            "duplicate_task_id",
            "identity_violation",
            lambda: allocate_work_conserving_cpu(
                [
                    CpuTaskDemand("task_a", "node_0", 1.0),
                    CpuTaskDemand("task_a", "node_1", 1.0),
                ],
                {"node_0": 1.0, "node_1": 1.0},
                1.0,
            ),
        ),
        (
            "invalid_slot_seconds",
            "invalid_numeric_value",
            lambda: allocate_work_conserving_cpu([], {}, 0.0),
        ),
        (
            "nonfinite_demand_rate",
            "numeric_overflow",
            lambda: allocate_work_conserving_cpu(
                [CpuTaskDemand("task_a", "node_0", 1e308)],
                {"node_0": 1.0},
                1e-308,
            ),
        ),
        (
            "airfogsim_assignment_mismatch",
            "callback_relation_violation",
            lambda: allocate_airfogsim_cpu(
                _Env({"RSU_0": 1.0}), {"RSU_0": [mismatch_task]}, slot_seconds=0.5
            ),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for case_id, reason_type, fixture in fixtures:
        try:
            fixture()
        except (TypeError, ValueError) as error:
            rows.append(
                {
                    "case_id": case_id,
                    "reason_type": reason_type,
                    "exception_type": type(error).__name__,
                    "detail": str(error),
                    "sample_origin": "contract_fixture",
                    "training_eligible": False,
                    "expected_rejection": True,
                }
            )
        else:
            raise AssertionError(f"invalid fixture was accepted: {case_id}")
    return rows


def build_evidence(*, airfogsim_root: Path) -> dict[str, Any]:
    sample_rows: list[dict[str, Any]] = []
    sample_rows.extend(_run_case("empty", [], {}, 0.5, {}))
    sample_rows.extend(
        _run_case(
            "zero_capacity",
            [CpuTaskDemand("task_a", "node_0", 1.0), CpuTaskDemand("task_b", "node_0", 2.0)],
            {"node_0": 0.0},
            1.0,
            {"task_a": 0.0, "task_b": 0.0},
        )
    )
    sample_rows.extend(
        _run_case(
            "demand_below_capacity",
            [CpuTaskDemand("task_a", "node_0", 1.0), CpuTaskDemand("task_b", "node_0", 2.0)],
            {"node_0": 10.0},
            1.0,
            {"task_a": 1.0, "task_b": 2.0},
        )
    )
    sample_rows.extend(
        _run_case(
            "demand_above_capacity",
            [CpuTaskDemand("task_a", "node_0", 10.0), CpuTaskDemand("task_b", "node_0", 10.0)],
            {"node_0": 6.0},
            1.0,
            {"task_a": 3.0, "task_b": 3.0},
        )
    )
    sample_rows.extend(
        _run_case(
            "unequal_demand_water_fill",
            [
                CpuTaskDemand("task_a", "node_0", 1.0),
                CpuTaskDemand("task_b", "node_0", 10.0),
                CpuTaskDemand("task_c", "node_0", 10.0),
            ],
            {"node_0": 7.0},
            1.0,
            {"task_a": 1.0, "task_b": 3.0, "task_c": 3.0},
        )
    )
    sample_rows.extend(
        _run_case(
            "candidate_local_post_communication",
            [CpuTaskDemand("task_a", "vehicle_0", 4.0)],
            {"vehicle_0": 1.0, "rsu_0": 3.0},
            1.0,
            {"task_a": 1.0},
        )
    )
    sample_rows.extend(
        _run_case(
            "candidate_offload_post_communication",
            [CpuTaskDemand("task_a", "rsu_0", 4.0)],
            {"vehicle_0": 1.0, "rsu_0": 3.0},
            1.0,
            {"task_a": 3.0},
        )
    )
    sample_rows.extend(_callback_parity_rows(airfogsim_root))
    rejected_rows = _expected_rejections(airfogsim_root)

    rule_contract = {
        "preflight_version": PREFLIGHT_VERSION,
        "rule_version": CPU_INNER_RULE_VERSION,
        "status": "implemented_and_preflight_verified",
        "claim_scope": "CPU inner rule and AirFogSim Task callback interface only",
        "core_action_dimensions": ["offload", "resource_block_allocation"],
        "cpu_is_action_dimension": False,
        "cpu_is_candidate_independent_constant": False,
        "invocation": "after each candidate communication update at every rollout step",
        "input": ["post_communication_compute_task_set", "remaining_work", "node_cpu_capacity", "slot_seconds"],
        "formula": "f_m=min(remaining_work_m/slot_seconds, lambda_i); sum(f_m)=min(node_capacity_i,sum(demand_m))",
        "policy": "deterministic_work_conserving_capacity_capped_equal_sharing",
        "uses_deadline": False,
        "uses_priority": False,
        "uses_future_information": False,
        "uses_learned_weights": False,
        "v4_collector_implemented": False,
        "v4_dataset_complete": False,
        "candidate_rollout_planner_complete": False,
        "final_method_frozen": False,
    }
    summary = {
        "preflight_version": PREFLIGHT_VERSION,
        "rule_version": CPU_INNER_RULE_VERSION,
        "p2_a_cpu_inner_rule_preflight_verified": True,
        "airfogsim_task_callback_interface_parity": True,
        "full_airfogsim_trajectory_executed": False,
        "contract_case_count": len({row["case_id"] for row in sample_rows}),
        "sample_row_count": len(sample_rows),
        "expected_rejection_count": len(rejected_rows),
        "all_samples_training_eligible": False,
        "v4_collector_implemented": False,
        "v4_dataset_complete": False,
        "model_training_started": False,
        "gpu_started": False,
        "locked_test_accessed": False,
        "candidate_rollout_planner_complete": False,
        "final_method_frozen": False,
    }
    return {
        "rule_contract": rule_contract,
        "sample_rows": sample_rows,
        "rejected_rows": rejected_rows,
        "summary": summary,
    }


def run_preflight(
    *,
    output_dir: Path,
    design_path: Path,
    airfogsim_root: Path,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    design = Path(design_path).resolve()
    airfogsim = Path(airfogsim_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing preflight bundle: {output}")
    if not design.is_file():
        raise FileNotFoundError(f"frozen design input not found: {design}")

    required_files = [
        CORE_MODULE_PATH,
        ADAPTER_MODULE_PATH,
        Path(__file__).resolve(),
        PURE_TEST_PATH,
        ADAPTER_TEST_PATH,
        PREFLIGHT_TEST_PATH,
    ]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"required implementation evidence not found: {path}")

    airfogsim_sources = {
        "airfogsim/enum_const.py": airfogsim / "airfogsim" / "enum_const.py",
        "airfogsim/entities/mission.py": airfogsim / "airfogsim" / "entities" / "mission.py",
        "airfogsim/entities/task.py": airfogsim / "airfogsim" / "entities" / "task.py",
        "airfogsim/manager/task_manager.py": airfogsim / "airfogsim" / "manager" / "task_manager.py",
    }
    for path in airfogsim_sources.values():
        if not path.is_file():
            raise FileNotFoundError(f"required AirFogSim source not found: {path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        evidence = build_evidence(airfogsim_root=airfogsim)
        write_json(temporary / "rule_contract.json", evidence["rule_contract"])
        write_csv(
            temporary / "sample_cases.csv",
            evidence["sample_rows"],
            [
                "case_id",
                "source",
                "node_id",
                "task_id",
                "remaining_work",
                "capacity",
                "slot_seconds",
                "allocated_cpu",
                "served_work",
                "sample_origin",
                "training_eligible",
                "passed",
            ],
        )
        write_csv(
            temporary / "rejected_records.csv",
            evidence["rejected_rows"],
            [
                "case_id",
                "reason_type",
                "exception_type",
                "detail",
                "sample_origin",
                "training_eligible",
                "expected_rejection",
            ],
        )
        write_json(temporary / "summary.json", evidence["summary"])

        manifest = {
            "preflight_version": PREFLIGHT_VERSION,
            "rule_version": CPU_INNER_RULE_VERSION,
            "design_input": file_metadata(design),
            "code_and_test_files": {
                path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix(): file_metadata(path)
                for path in required_files
            },
            "airfogsim_source_files": {
                name: file_metadata(path) for name, path in airfogsim_sources.items()
            },
            "output_files": {
                name: published_file_metadata(temporary / name, output / name)
                for name in (
                    "rule_contract.json",
                    "sample_cases.csv",
                    "rejected_records.csv",
                    "summary.json",
                )
            },
            "locked_test_accessed": False,
            "gpu_started": False,
        }
        write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return evidence["summary"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--design-path", type=Path, default=DEFAULT_DESIGN_PATH)
    parser.add_argument("--airfogsim-root", type=Path, default=DEFAULT_AIRFOGSIM_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_preflight(
        output_dir=args.output_dir,
        design_path=args.design_path,
        airfogsim_root=args.airfogsim_root,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
