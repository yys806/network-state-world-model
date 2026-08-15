"""Frozen PI-JWM v1 control slice and machine-readable readiness gates."""

from __future__ import annotations

from typing import Any, Mapping

from .airfogsim_contract_adapter import ADAPTER_VERSION, encode_optional_value


CONTRACT_VERSION = "PI-JWM-main-experiment-contract-v2"


def _field(field_id: str, group: str, source: str, availability: str = "direct") -> dict[str, str]:
    return {
        "field_id": field_id,
        "group": group,
        "source": source,
        "availability": availability,
    }


def _target(target_id: str, scope: str) -> dict[str, str]:
    return {"target_id": target_id, "scope": scope}


def _metric(metric_id: str, layer: str, mask_policy: str = "observed_only") -> dict[str, str]:
    return {"metric_id": metric_id, "layer": layer, "mask_policy": mask_policy}


def build_frozen_contract() -> dict[str, Any]:
    """Return the first formal control slice without simulator-only extras."""

    state_fields = [
        _field("node_identity_type", "physical_node", "AirFogSim/direct"),
        _field("position_velocity_acceleration", "physical_node", "AirFogSim/direct"),
        _field("cpu_capacity", "physical_node", "AirFogSim/direct_after_adapter"),
        _field("uav_energy", "physical_node", "AirFogSim/direct_for_uav", "conditional"),
        _field("physical_edge_endpoints_type", "physical_edge", "AirFogSim/direct"),
        _field("link_activity", "physical_edge", "AirFogSim/direct_runtime_event"),
        _field("link_rate_by_rb", "physical_edge", "AirFogSim/direct_runtime_event"),
        _field("agent_identity_type_attachment", "information_agent", "PI-JWM/CIP_from_active_physical_node"),
        _field("information_flow_endpoints_type", "information_flow", "AirFogSim/direct_runtime_event"),
        _field("information_flow_amount_status", "information_flow", "AirFogSim/direct_runtime_event"),
        _field("task_identity_placement", "task_auxiliary", "AirFogSim/direct"),
        _field("task_size_cpu_deadline_priority", "task_auxiliary", "AirFogSim/direct"),
        _field("task_lifecycle_progress_outcome", "task_auxiliary", "AirFogSim/direct"),
        _field("dag_precedence", "dag_auxiliary", "AirFogSim/direct"),
        _field("dag_dependency_payload", "dag_auxiliary", "not_modeled_without_explicit_data_mb", "conditional"),
        _field("cip_agent_attachment", "cross_graph", "PI-JWM/deterministic_attachment"),
        _field("cfe_flow_bearer", "cross_graph", "AirFogSim/direct_runtime_event"),
        _field("total_rb_and_allocation", "resource", "AirFogSim/direct_runtime"),
        _field("cpu_capacity_and_allocation", "resource", "PI-JWM/capacity_safe_adapter"),
        _field("uav_energy_components", "resource", "PI-JWM/direct_event_accounting_adapter", "conditional"),
    ]
    actions = [
        {"field_id": "offload_target_path", "constraint": "reachable_direct_path"},
        {"field_id": "rb_allocation", "constraint": "valid_indices_and_capacity"},
    ]
    prediction_targets = [
        _target("next_physical_node_state", "one_step_and_rollout"),
        _target("next_physical_edge_state", "activity_rate_and_topology"),
        _target("next_information_agent_state", "agent_queue_and_service_state"),
        _target("next_information_flow_state", "flow_activity_amount_and_completion"),
        _target("next_task_state", "lifecycle_dag_readiness_and_progress"),
        _target("next_resource_state", "rb_cpu_and_masked_uav_energy"),
        _target("task_terminal_outcome", "completion_delay_deadline"),
    ]
    metrics = [
        _metric("state_mae", "state_prediction"),
        _metric("state_rmse", "state_prediction"),
        _metric("link_activity_f1", "state_prediction"),
        _metric("rollout_horizon_error", "state_prediction_k_1_5_20"),
        _metric("nll", "uncertainty"),
        _metric("calibration_error", "uncertainty"),
        _metric("conformal_coverage", "uncertainty"),
        _metric("prediction_interval_width", "uncertainty"),
        _metric("action_ranking_regret", "decision"),
        _metric("task_completion_rate", "system"),
        _metric("latency_mean", "system"),
        _metric("latency_p95", "system"),
        _metric("latency_p99", "system"),
        _metric("deadline_violation_rate", "system"),
        _metric("energy_consumption", "system", "available_energy_fields_only"),
        _metric("resource_utilization", "system"),
        _metric("fairness", "system"),
        _metric("constraint_violation_rate", "safety"),
        _metric("ood_fallback_rate", "safety"),
        _metric("inference_latency", "deployment"),
    ]
    unmodelled_fields = [
        {"field_id": field_id, **encode_optional_value(None, status="not_modeled")}
        for field_id in (
            "vehicle_energy",
            "rsu_energy",
            "cpu_compute_energy",
            "storage_occupancy",
        )
    ]
    return {
        "contract_version": CONTRACT_VERSION,
        "contract_adapter_version": ADAPTER_VERSION,
        "control_slice": "offload_rb_v2",
        "state_fields": state_fields,
        "actions": actions,
        "fixed_rules": ["cpu_allocation"],
        "excluded_actions": [
            "uav_trajectory",
            "transmit_power",
            "mcs",
            "cache_placement",
            "mission_control",
        ],
        "prediction_targets": prediction_targets,
        "metrics": metrics,
        "unmodelled_fields": unmodelled_fields,
        "missing_value_policy": "never_zero_fill",
        "rollout_horizons": [1, 5, 20],
        "evidence_boundary": {
            "airfogsim": "reference_simulator_and_data_source",
            "task_dag": "precedence_only_without_explicit_payload",
            "dependency_data": "not_modeled_unless_data_mb_is_explicit",
        },
    }


def _contract_is_valid(contract: Mapping[str, Any]) -> bool:
    actions = [row.get("field_id") for row in contract.get("actions", [])]
    target_ids = [row.get("target_id") for row in contract.get("prediction_targets", [])]
    metric_ids = [row.get("metric_id") for row in contract.get("metrics", [])]
    missing = list(contract.get("unmodelled_fields", []))
    return bool(
        actions == ["offload_target_path", "rb_allocation"]
        and contract.get("fixed_rules") == ["cpu_allocation"]
        and len(target_ids) == len(set(target_ids)) >= 6
        and len(metric_ids) == len(set(metric_ids)) >= 10
        and missing
        and all(row.get("value") is None and row.get("observed_mask") == 0 for row in missing)
        and contract.get("missing_value_policy") == "never_zero_fill"
    )


def _all_true(report: Mapping[str, Any] | None, names: tuple[str, ...]) -> bool:
    return bool(report) and all(bool(report.get(name, False)) for name in names)


def _manifest_ready(manifest: Mapping[str, Any] | None, fields: tuple[str, ...]) -> bool:
    return bool(manifest) and all(bool(manifest.get(field, False)) for field in fields)


def build_readiness_report(
    exp03: Mapping[str, Any],
    exp04: Mapping[str, Any],
    exp05: Mapping[str, Any],
    *,
    formal_dataset_manifest: Mapping[str, Any] | None = None,
    external_validation_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate simulator preflight from dataset and real-data completion."""

    contract = build_frozen_contract()
    total_pairs = int(exp05.get("total_pairs", 0))
    accepted_pairs = int(exp05.get("accepted_pairs", 0))
    checks = {
        "contract_valid": _contract_is_valid(contract),
        "exp03_completed": bool(exp03.get("experiment_completed", False)),
        "exp03_strict_dual_graph_ready": bool(exp03.get("strict_dual_graph_ready", False)),
        "exp03_reproducible": bool(exp03.get("reproducibility_passed", False)),
        "exp03_corruption_detection": bool(exp03.get("corruption_detection_passed", False)),
        "exp04_completed": bool(exp04.get("experiment_completed", False)),
        "exp04_conservation_ready": bool(exp04.get("conservation_ready", False)),
        "exp04_reproducible": bool(exp04.get("reproducibility_passed", False)),
        "exp04_corruption_detection": bool(exp04.get("corruption_detection_passed", False)),
        "exp05_completed": bool(exp05.get("experiment_completed", False)),
        "exp05_action_sensitivity_ready": bool(exp05.get("action_sensitivity_ready", False)),
        "exp05_pairs_valid": bool(
            total_pairs > 0
            and accepted_pairs == total_pairs
            and not list(exp05.get("failed_pair_ids", []))
        ),
        "exp05_corruption_detection": bool(
            exp05.get("corruption_detection_passed", False)
        ),
    }
    for gate_name, passed in dict(exp04.get("gates", {})).items():
        checks[f"exp04_gate_{gate_name}"] = bool(passed)

    blocking_checks = sorted(name for name, passed in checks.items() if not passed)
    contract_ready = checks["contract_valid"]
    simulator_preflight_ready = contract_ready and not blocking_checks
    formal_dataset_ready = simulator_preflight_ready and _manifest_ready(
        formal_dataset_manifest,
        ("generation_completed", "field_masks_valid", "splits_frozen", "source_manifest_present"),
    )
    external_validation_ready = _manifest_ready(
        external_validation_manifest,
        ("local_data_verified", "license_verified", "field_semantics_verified", "holdout_split_frozen"),
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "contract_ready": contract_ready,
        "simulator_preflight_ready": simulator_preflight_ready,
        "simulation_training_ready": simulator_preflight_ready,
        "formal_dataset_ready": formal_dataset_ready,
        "external_validation_ready": external_validation_ready,
        "checks": checks,
        "blocking_checks": blocking_checks,
        "scope": {
            "simulation_training_ready": "safe_to_launch_formal_simulation_data_generation_and_training_smoke",
            "formal_dataset_ready": "formal_scale_train_calibration_test_trajectories_exist_and_pass_contract",
            "external_validation_ready": "verified_real_measurement_holdout_is_locally_available",
        },
    }
