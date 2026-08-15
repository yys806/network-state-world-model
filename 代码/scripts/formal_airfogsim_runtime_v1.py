from __future__ import annotations

"""Run one formal PI-JWM trajectory through existing AirFogSim observers."""

import copy
import sys
from pathlib import Path
from typing import Any, Callable


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SMALL_EXPERIMENT_DIR = Path(__file__).resolve().parent / "small_experiments"
for path in (SRC_ROOT, SMALL_EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_jwm.airfogsim_cpu_policy_v1 import CpuPolicyAllocator
from pi_jwm.formal_airfogsim_dataset_v1 import (
    TrajectorySpec,
    apply_formal_scenario_overrides,
)


def _rewrite_trajectory_id(value: Any, trajectory_id: str) -> None:
    if isinstance(value, dict):
        if "trajectory_id" in value:
            value["trajectory_id"] = str(trajectory_id)
        for child in value.values():
            _rewrite_trajectory_id(child, trajectory_id)
    elif isinstance(value, list):
        for child in value:
            _rewrite_trajectory_id(child, trajectory_id)


def _decision_key(row: dict[str, Any]) -> tuple[float, str, str]:
    return (
        round(float(row.get("time", 0.0)), 9),
        str(row.get("task_id", "")),
        str(row.get("node_id", "")),
    )


def _enrich_cpu_rows(
    rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> None:
    decision_by_key = {_decision_key(row): row for row in decisions}
    for row in rows:
        decision = decision_by_key.get(_decision_key(row))
        if decision is None:
            raise ValueError(f"missing formal CPU decision metadata for {_decision_key(row)}")
        for field in (
            "policy_id",
            "policy_weight",
            "deadline_remaining",
            "queue_size",
            "allocated_fraction",
        ):
            row[field] = decision[field]


def _strip_synthetic_dependency_payloads(result: dict[str, Any]) -> None:
    source = result["source_bundle"]
    for edge in source.get("information_edges", []):
        if edge.get("data_mb") is not None:
            raise ValueError("formal v1 forbids non-null DAG dependency payloads")
        edge["semantic"] = "precedence_only"
        edge["payload_status"] = "not_modeled"
    source["dependency_flows"] = []
    source["ep_relations"] = []
    result["bundle"]["dependency_ledger"] = []


def _runtime_components():
    import airfogsim_cross_graph_evidence_closure as evidence_module
    import airfogsim_strict_dual_graph_preflight as preflight_module
    import task_resource_conservation_audit as conservation_module

    return (
        evidence_module,
        preflight_module,
        conservation_module.run_airfogsim_conservation_seed,
    )


def run_formal_airfogsim_trajectory(
    spec: TrajectorySpec,
    *,
    max_time: float,
    evidence_module: Any | None = None,
    preflight_module: Any | None = None,
    conservation_runner: Callable[[int, float], dict[str, Any]] | None = None,
    allocator_factory: Callable[[str, int], Any] | None = None,
) -> dict[str, Any]:
    """Inject one scenario and CPU policy without changing AirFogSim core code."""

    if evidence_module is None or preflight_module is None or conservation_runner is None:
        evidence_module, preflight_module, conservation_runner = _runtime_components()

    original_build_config = preflight_module.build_preflight_config
    original_install_cpu = evidence_module.install_capacity_safe_cpu_callback
    if allocator_factory is None:
        allocator = CpuPolicyAllocator(spec.cpu_policy, seed=spec.seed)
    else:
        allocator = allocator_factory(str(spec.cpu_policy), int(spec.seed))
    decision_rows: list[dict[str, Any]] = []

    def formal_build_config(config: dict[str, Any], seed: int, runtime: float):
        configured = original_build_config(config, seed, runtime)
        return apply_formal_scenario_overrides(configured, spec.scenario)

    def formal_install_cpu(env: Any, computation_scheduler: Any) -> None:
        def callback(computing_tasks: dict[str, list[Any]], **_: Any) -> dict[str, float]:
            decision = allocator.allocate(env, computing_tasks)
            time_value = float(env.simulation_time)
            decision_rows.extend({**row, "time": time_value} for row in decision.rows)
            return decision.allocations

        computation_scheduler.setComputingCallBack(env, callback)

    preflight_module.build_preflight_config = formal_build_config
    evidence_module.install_capacity_safe_cpu_callback = formal_install_cpu
    try:
        result = copy.deepcopy(conservation_runner(int(spec.seed), float(max_time)))
    finally:
        preflight_module.build_preflight_config = original_build_config
        evidence_module.install_capacity_safe_cpu_callback = original_install_cpu

    cpu_rows = result["bundle"].get("cpu_ledger", [])
    _enrich_cpu_rows(cpu_rows, decision_rows)
    compute_task_rows = [
        row
        for row in result["bundle"].get("task_ledger", [])
        if row.get("kind") == "compute"
    ]
    if compute_task_rows:
        _enrich_cpu_rows(compute_task_rows, decision_rows)
    _strip_synthetic_dependency_payloads(result)
    _rewrite_trajectory_id(result["source_bundle"], spec.trajectory_id)
    result["config"]["pi_jwm_formal_v1"] = {
        "trajectory_id": spec.trajectory_id,
        "seed": int(spec.seed),
        "split": spec.split,
        "cpu_policy": spec.cpu_policy,
        "scenario": spec.scenario.to_dict(),
        "dag_semantics": "airfogsim_precedence_only",
        "dependency_payload": "not_modeled",
    }
    result["runtime_summary"].update(
        {
            "trajectory_id": spec.trajectory_id,
            "split": spec.split,
            "cpu_policy": spec.cpu_policy,
            "cpu_decision_rows": len(cpu_rows),
            "dependency_semantics": "airfogsim_precedence_only",
        }
    )
    return result
