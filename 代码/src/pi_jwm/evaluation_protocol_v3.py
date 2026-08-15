"""Frozen, teacher-aligned evaluation and fairness protocol for PI-JWM."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "PIJWM-Eval-Protocol-v3"
FAIR_PROTOCOL_VERSION = "PIJWM-Fair-Experiment-v3"


def _metric(
    metric_id: str,
    layer: str,
    direction: str,
    formula: str,
    unit: str,
    numerator: str,
    denominator: str,
    source_fields: Sequence[str],
    mask_policy: str,
    aggregation: str,
    computable_when: str,
    na_rule: str,
) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "layer": layer,
        "direction": direction,
        "formula": formula,
        "unit": unit,
        "numerator": numerator,
        "denominator": denominator,
        "source_fields": list(source_fields),
        "mask_policy": mask_policy,
        "aggregation": aggregation,
        "computable_when": computable_when,
        "na_rule": na_rule,
    }


def build_metric_registry() -> list[dict[str, Any]]:
    """Return the complete R2 metric registry without model-specific values."""

    observed = "target present AND target feature_mask; persistence additionally requires previous target present AND previous feature_mask"
    nonempty = "emit not_computable when denominator is zero"
    metrics = [
        _metric("state.physical_node.position.rmse", "state_prediction", "minimize", "sqrt(sum(||p_hat-p||_2^2)/N)", "m", "sum squared 3-D position error", "valid physical-node observations", ["physical_node_state.x", "physical_node_state.y", "physical_node_state.z", "physical_node_present", "physical_node_feature_mask"], observed, "micro within seed; macro over seeds", "position components are observed", nonempty),
        _metric("state.physical_node.motion.rmse", "state_prediction", "minimize", "sqrt(sum((v_hat-v)^2)/N)", "m/s", "sum squared speed error", "valid speed observations", ["physical_node_state.speed", "physical_node_present", "physical_node_feature_mask"], observed, "micro within seed; macro over seeds", "speed is observed", nonempty),
        _metric("state.physical_edge.distance.rmse", "state_prediction", "minimize", "sqrt(sum((d_hat-d)^2)/N)", "m", "sum squared distance error", "valid physical spatial edges", ["physical_edge_state.distance", "physical_edge_present", "physical_edge_feature_mask"], observed, "micro within seed; macro over seeds", "distance is observed", nonempty),
        _metric("state.physical_edge.relative_speed.rmse", "state_prediction", "minimize", "sqrt(sum((vrel_hat-vrel)^2)/N)", "m/s", "sum squared relative-speed error", "valid physical spatial edges", ["physical_edge_state.relative_speed", "physical_edge_present", "physical_edge_feature_mask"], observed, "micro within seed; macro over seeds", "relative speed is observed", nonempty),
        _metric("state.information_node.queue.mae", "state_prediction", "minimize", "sum(|q_hat-q|)/N", "count", "sum absolute queue-state error", "valid information-agent queue features", ["information_node_state.unassigned_queue_count", "information_node_state.tx_queue_count", "information_node_state.return_queue_count", "information_node_present", "information_node_feature_mask"], observed, "micro within seed; macro over seeds", "at least one queue feature is observed", nonempty),
        _metric("state.information_node.cpu_backlog.mae", "state_prediction", "minimize", "sum(|b_hat-b|)/N", "cycles", "sum absolute backlog error", "valid information-agent backlog observations", ["information_node_state.cpu_backlog", "information_node_present", "information_node_feature_mask"], observed, "micro within seed; macro over seeds", "CPU backlog is observed", nonempty),
        _metric("state.information_edge.rate.rmse", "state_prediction", "minimize", "sqrt(sum((r_hat-r)^2)/N)", "Mbps", "sum squared rate error", "valid observed information-link rates", ["information_edge_state.outcome.rate_sum", "information_edge_present", "information_edge_feature_mask"], observed, "micro within seed; macro over seeds", "outcome.rate_sum is observed", nonempty),
        _metric("link.active_only_rate.mae", "state_prediction", "minimize", "sum(|r_hat-r| * active_true)/N_active", "Mbps", "absolute rate error on true active links", "true active links with observed rate", ["information_edge_state.outcome.rate_sum", "information_edge_state.outcome.active_task_count", "information_edge_present", "information_edge_feature_mask"], "true information-link activity AND observed rate", "micro within seed; macro over seeds", "at least one true active link has observed rate", nonempty),
        _metric("state.flow.remaining_data.mae", "state_prediction", "minimize", "sum(|x_hat-x|)/N", "MB", "sum absolute remaining-data error", "valid present business flows", ["data_flow_state", "data_flow_present", "data_flow_valid"], "data_flow_present AND data_flow_valid", "micro within seed; macro over seeds", "at least one flow is present", nonempty),
        _metric("state.task.deadline_remaining.mae", "state_prediction", "minimize", "sum(|d_hat-d|)/N", "s", "sum absolute deadline-remaining error", "valid present tasks", ["task_state.deadline_remaining", "task_present", "task_valid"], "task_present AND task_valid", "micro within seed; macro over seeds", "at least one task is present", nonempty),
        _metric("state.dag.unfinished_parent_count.mae", "state_prediction", "minimize", "sum(|u_hat-u|)/N", "count", "sum absolute unfinished-parent error", "valid DAG task states", ["task_dag_state.unfinished_parent_count", "task_dag_state_present"], "task_dag_state_present", "micro within seed; macro over seeds", "at least one DAG task state is present", nonempty),
        _metric("selection.required_continuous.normalized_error", "checkpoint_selection", "minimize", "first pool each error metric over its frozen valid validation targets, then mean_j(error_metric_j/train_scale_j) over the ten frozen continuous target groups", "normalized ratio", "sum of ten train-scale-normalized aggregate errors", "10 frozen continuous target groups", ["state.physical_node.position.rmse", "state.physical_node.motion.rmse", "state.physical_edge.distance.rmse", "state.physical_edge.relative_speed.rmse", "state.information_node.queue.mae", "state.information_node.cpu_backlog.mae", "state.information_edge.rate.rmse", "state.flow.remaining_data.mae", "state.task.deadline_remaining.mae", "state.dag.unfinished_parent_count.mae", "train_only_normalization_stats"], "each component uses its frozen target mask; do not require every individual trajectory to contain every target type", "validation checkpoint score computed after component aggregation; never a final scientific metric", "all ten validation-aggregate component metrics and strictly positive train-only scales exist", "candidate_ineligible when an aggregate component or scale is missing",),
        _metric("event.information_link_activity.f1", "sparse_event", "maximize", "2TP/(2TP+FP+FN)", "ratio", "2TP", "2TP+FP+FN", ["information_edge_state.outcome.active_task_count", "information_edge_present", "information_edge_feature_mask"], "information_edge_present AND active_task_count observed", "micro within seed; macro over seeds", "positive truth or prediction exists", nonempty),
        _metric("event.information_link_activity.auprc", "sparse_event", "maximize", "average precision: sum over grouped score thresholds of recall increment times precision", "ratio", "stepwise precision-recall accumulation", "valid link labels", ["information_edge_state.outcome.active_task_count", "information_edge_present", "information_edge_feature_mask"], "information_edge_present AND active_task_count observed; persistence additionally requires the previous label observed", "pool labels within environment trajectory; macro over complete environment trajectories", "at least one positive truth label exists", nonempty),
        _metric("event.flow_present.f1", "sparse_event", "maximize", "2TP/(2TP+FP+FN)", "ratio", "2TP", "2TP+FP+FN", ["data_flow_present", "data_flow_valid"], "data_flow_valid", "micro within seed; macro over seeds", "positive truth or prediction exists", nonempty),
        _metric("event.task_present.f1", "sparse_event", "maximize", "2TP/(2TP+FP+FN)", "ratio", "2TP", "2TP+FP+FN", ["task_present", "task_valid"], "task_valid", "micro within seed; macro over seeds", "positive truth or prediction exists", nonempty),
        _metric("task.lifecycle.macro_f1", "task_prediction", "maximize", "mean over the frozen labels {to_offload, computing, returning, finished, failed}; absent-class F1 is zero", "ratio", "sum of five classwise F1 values", "5 frozen lifecycle classes", ["task_lifecycle_index", "task_present", "task_valid"], "task_present AND task_valid AND lifecycle in [0,4]; persistence additionally requires the previous lifecycle observed", "five-class macro within environment trajectory; macro over complete environment trajectories", "at least one valid lifecycle label exists", nonempty),
        _metric("system.task_completion_rate", "system", "maximize", "completed evaluable tasks/evaluable tasks", "ratio", "completed evaluable tasks", "evaluable tasks", ["AirFogSim.task.terminal_status", "AirFogSim.task.deadline_time"], "completed, failed, or deadline-reached tasks", "per trajectory; macro over seeds", "at least one task is evaluable", nonempty),
        _metric("system.latency.mean", "system", "minimize", "sum completed-task delay/N_completed", "s", "sum completed-task delay", "completed tasks with delay", ["AirFogSim.task.task_delay", "AirFogSim.task.terminal_status"], "completed tasks with direct delay", "per trajectory; macro over seeds", "at least one completed delay exists", nonempty),
        _metric("system.latency.p95", "system", "minimize", "linear-interpolated 0.95 quantile(completed-task delay)", "s", "ordered completed-task delays", "completed tasks with delay", ["AirFogSim.task.task_delay", "AirFogSim.task.terminal_status"], "completed tasks with direct delay", "per trajectory; macro over seeds", "at least one completed delay exists", nonempty),
        _metric("system.latency.p99", "system", "minimize", "linear-interpolated 0.99 quantile(completed-task delay)", "s", "ordered completed-task delays", "completed tasks with delay", ["AirFogSim.task.task_delay", "AirFogSim.task.terminal_status"], "completed tasks with direct delay", "per trajectory; macro over seeds", "at least one completed delay exists", nonempty),
        _metric("system.deadline_violation_rate", "system", "minimize", "deadline violations/evaluable tasks", "ratio", "deadline violations", "evaluable tasks", ["AirFogSim.task.failure_reason", "AirFogSim.task.deadline_time"], "evaluable tasks", "per trajectory; macro over seeds", "at least one task is evaluable", nonempty),
        _metric("system.task_failure_rate", "system", "minimize", "failed or right-censored evaluable tasks/evaluable tasks", "ratio", "failed or expired tasks", "evaluable tasks", ["AirFogSim.task.terminal_status", "AirFogSim.task.deadline_time"], "evaluable tasks", "per trajectory; macro over environment trajectories", "at least one task is evaluable", nonempty),
        _metric("system.priority_weighted_completion_rate", "system", "maximize", "sum(priority of completed evaluable tasks)/sum(priority of evaluable tasks)", "ratio", "completed-task priority", "evaluable-task priority", ["AirFogSim.task.priority", "AirFogSim.task.terminal_status"], "evaluable tasks with nonnegative priority", "per trajectory; macro over environment trajectories", "positive evaluable-task priority exists", nonempty),
        _metric("system.application_throughput", "system", "maximize", "sum delivered data/evaluation seconds", "MB/s", "delivered MB", "evaluation seconds", ["AirFogSim.transfer_events.delivered_data", "AirFogSim.evaluation_end_time"], "runtime transfer events", "per trajectory; macro over seeds", "positive evaluation duration exists", nonempty),
        _metric("system.information_link_active_ratio", "system", "report", "sum(1[outcome.active_task_count>0])/N_observed_information_edges", "ratio", "active observed information-edge states", "observed information-edge states", ["information_edge_state.outcome.active_task_count", "information_edge_present", "information_edge_feature_mask"], "information_edge_present AND active_task_count observed", "per complete environment trajectory; macro over environment trajectories", "observed information-edge states exist", nonempty),
        _metric("resource.rb_utilization", "resource", "report", "used RB slots/(n_RB * observed slots)", "ratio", "used RB slots", "available RB slots", ["AirFogSim.rb_ledger.rb_indices", "AirFogSim.rb_ledger.n_rb"], "consistent positive RB capacity", "per trajectory; macro over seeds", "one consistent n_rb exists", nonempty),
        _metric("resource.cpu_utilization", "resource", "report", "sum allocated_cpu*dt/sum capacity*dt", "ratio", "CPU allocation-time", "CPU capacity-time", ["AirFogSim.cpu_ledger", "AirFogSim.physical_node_snapshots.cpu"], "observed CPU ledger and capacities", "per trajectory; macro over seeds", "positive CPU capacity-time exists", nonempty),
        _metric("system.uav_energy_total", "system", "minimize", "sum(energy_before-energy_after)", "AirFogSim energy unit", "observed UAV energy decrease", "none", ["AirFogSim.uav_energy_ledger.energy_before", "AirFogSim.uav_energy_ledger.energy_after"], "observed UAV energy rows", "per trajectory; macro over seeds", "UAV energy rows exist", nonempty),
        _metric("system.uav_energy_per_completed_task", "system", "minimize", "total UAV energy/completed evaluable tasks", "AirFogSim energy unit/task", "total UAV energy", "completed evaluable tasks", ["AirFogSim.uav_energy_ledger", "AirFogSim.task.terminal_status"], "energy observed AND completed task", "per trajectory; macro over seeds", "energy and a completion exist", nonempty),
        _metric("system.completion_fairness_jain", "system", "maximize", "(sum source completion rates)^2/(U*sum squared rates)", "ratio", "squared sum of source rates", "source count times squared-rate sum", ["AirFogSim.task.source", "AirFogSim.task.terminal_status"], "fixed evaluable task-source population", "per trajectory; macro over seeds", "nonempty source population and positive denominator", nonempty),
        _metric("system.dependency_payload_coverage", "system", "report", "explicit dependency-data flows/DAG precedence edges", "ratio", "DAG edges with explicit dependency payload", "DAG precedence edges", ["AirFogSim.task_dag_edges", "AirFogSim.dependency_data_flows"], "DAG edges and explicit dependency-data flow IDs", "per trajectory; macro over environment trajectories", "DAG edges exist", nonempty),
        _metric("system.dependency_data_delivery_rate", "system", "report", "completed explicit dependency-data flows/explicit dependency-data flows", "ratio", "completed dependency-data flows", "explicit dependency-data flows", ["AirFogSim.dependency_data_flows.status"], "explicit dependency-data flow rows", "per trajectory; macro over environment trajectories", "explicit dependency-data flows exist", nonempty),
        _metric("safety.task_flow_conservation_violation_rate", "safety", "minimize", "violating task-flow rows/task-flow rows", "ratio", "rows with residual above 1e-8", "task-flow ledger rows", ["AirFogSim.task_ledger"], "observed task-flow ledger", "per trajectory; macro over seeds", "ledger is nonempty", nonempty),
        _metric("safety.cpu_capacity_violation_rate", "safety", "minimize", "violating node-slots/node-slots", "ratio", "node-slots exceeding capacity", "observed node-slots", ["AirFogSim.cpu_ledger"], "grouped by time and node", "per trajectory; macro over seeds", "CPU ledger is nonempty", nonempty),
        _metric("safety.energy_equation_violation_rate", "safety", "minimize", "violating UAV energy rows/UAV energy rows", "ratio", "rows with residual above 1e-8", "UAV energy rows", ["AirFogSim.uav_energy_ledger"], "observed UAV energy rows", "per trajectory; macro over seeds", "energy ledger is nonempty", nonempty),
        _metric("uncertainty.nll", "uncertainty", "minimize", "mean negative log predictive density", "nats", "sum negative log density", "valid scalar targets", ["predictive_distribution", "target", "target_mask"], "observed targets with model distribution", "micro within seed; macro over seeds", "model outputs a valid predictive distribution", "emit not_computable for deterministic methods"),
        _metric("uncertainty.coverage_95", "uncertainty", "target_0.95", "covered valid targets/valid targets", "ratio", "targets inside frozen 95% interval", "valid targets", ["prediction_interval_95", "target", "target_mask"], "observed targets with calibrated interval", "micro within seed; macro over seeds", "interval is calibrated without evaluation leakage", "emit not_computable when no interval exists"),
        _metric("uncertainty.interval_width_95", "uncertainty", "minimize_at_valid_coverage", "mean(upper-lower)", "target unit", "sum interval width", "valid targets", ["prediction_interval_95", "target_mask"], "same mask as coverage_95", "micro within seed; macro over seeds", "95% interval exists", "emit not_computable when no interval exists"),
        _metric("decision.action_regret", "decision", "minimize", "utility(best factual candidate)-utility(selected candidate)", "utility", "sum factual candidate regret", "states with complete counterfactual candidates", ["counterfactual_action_outcomes", "selected_action"], "complete legal candidate outcomes", "per decision; macro over seeds", "same-state counterfactual outcomes exist", "emit not_computable for factual-only trajectories"),
        _metric("safety.ood_fallback_rate", "safety", "report", "OOD-triggered fallbacks/OOD-evaluated decisions", "ratio", "OOD-triggered fallbacks", "OOD-evaluated decisions", ["ood_indicator", "fallback_indicator"], "frozen OOD split", "per decision; macro over seeds", "OOD split and fallback policy exist", "emit not_computable before transfer evaluation"),
        _metric("safety.ood_transfer_score", "safety", "report", "external-OOD score under frozen transfer protocol", "ratio", "external-OOD score numerator", "external-OOD score denominator", ["ood_indicator", "external_holdout_metric"], "frozen external holdout split", "per holdout trajectory; macro over environment trajectories", "external holdout and transfer protocol exist", "emit not_computable before transfer evaluation"),
        _metric("deployment.inference_latency.p95", "deployment", "minimize", "0.95 quantile wall-clock inference time", "ms", "ordered synchronized inference times", "timed inference calls", ["inference_wall_time_ms"], "warmup excluded; device synchronized", "per run; macro over seeds", "timed deployment run exists", "emit not_computable for offline data-only reports"),
    ]
    return metrics


def build_factual_metric_mapping() -> list[dict[str, Any]]:
    """Map every factual sidecar name to one canonical R2 metric ID."""

    mappings = {
        "action_regret": ("decision.action_regret", "direct", None),
        "completion_fairness_jain": ("system.completion_fairness_jain", "direct", None),
        "cpu_capacity_violation_rate": ("safety.cpu_capacity_violation_rate", "direct", None),
        "cpu_utilization": ("resource.cpu_utilization", "direct", None),
        "deadline_violation_rate": ("system.deadline_violation_rate", "direct", None),
        "dependency_data_delivery_rate": ("system.dependency_data_delivery_rate", "direct", None),
        "dependency_payload_coverage": ("system.dependency_payload_coverage", "direct", None),
        "energy_equation_violation_rate": ("safety.energy_equation_violation_rate", "direct", None),
        "information_throughput": ("system.application_throughput", "alias", "canonical system throughput; sidecar keeps source name"),
        "ood_transfer_score": ("safety.ood_transfer_score", "direct", None),
        "physical_link_active_ratio": ("system.information_link_active_ratio", "semantic_alias", "legacy source name; interpreted as information-link observation activity, not a physical-graph edge definition"),
        "priority_weighted_completion_rate": ("system.priority_weighted_completion_rate", "direct", None),
        "rb_utilization": ("resource.rb_utilization", "direct", None),
        "successful_task_delay_mean": ("system.latency.mean", "alias", "completed-task delay mean"),
        "successful_task_delay_p95": ("system.latency.p95", "alias", "completed-task delay p95"),
        "successful_task_delay_p99": ("system.latency.p99", "alias", "completed-task delay p99"),
        "task_completion_rate": ("system.task_completion_rate", "direct", None),
        "task_failure_rate": ("system.task_failure_rate", "direct", None),
        "task_flow_conservation_violation_rate": ("safety.task_flow_conservation_violation_rate", "direct", None),
        "uav_energy_per_completed_task": ("system.uav_energy_per_completed_task", "direct", None),
        "uav_energy_total": ("system.uav_energy_total", "direct", None),
        "uncertainty_coverage": ("uncertainty.coverage_95", "alias", "sidecar status is not_computable until a calibrated interval exists"),
    }
    return [
        {
            "source_metric_name": source,
            "canonical_metric_id": canonical,
            "mapping_kind": kind,
            "note": note,
        }
        for source, (canonical, kind, note) in sorted(mappings.items())
    ]


def build_fair_experiment_protocol(
    *,
    environment_splits: Mapping[str, Sequence[str]] | None = None,
    normalization_stats_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": FAIR_PROTOCOL_VERSION,
        "data_binding": {
            "environment_splits": {
                split: sorted(str(item) for item in items)
                for split, items in (environment_splits or {}).items()
            },
            "normalization_stats_sha256": normalization_stats_sha256,
            "locked_test_content_read": False,
        },
        "split_roles": {
            "fit": ["train"],
            "checkpoint_selection": "validation",
            "architecture_selection": "validation",
            "threshold_calibration": "calibration",
            "uncertainty_calibration": "calibration",
            "final_evaluation": "locked_test",
        },
        "method_budget_policy": "equal_optimizer_steps_and_windows",
        "budgets": {
            "module_screening": {
                "training_seeds": [20260803],
                "max_epochs": 30,
                "early_stopping_patience": 5,
            },
            "formal_comparison": {
                "training_seeds": [20260803, 20260804, 20260805],
                "max_epochs": 100,
                "early_stopping_patience": 10,
            },
        },
        "common_training": {
            "batch_size": 32,
            "minimum_improvement": 1e-4,
            "same_train_windows": True,
            "same_optimizer_step_cap": True,
            "same_common_output_heads": True,
            "failed_seeds_are_reported": True,
        },
        "checkpoint_selection": {
            "split": "validation",
            "direction": "minimize",
            "name": "validation_protocol_score",
            "formula": "equal mean of four frozen normalized terms",
            "terms": [
                {"metric_id": "event.information_link_activity.auprc", "transform": "1-value", "weight": 0.25},
                {"metric_id": "link.active_only_rate.mae", "transform": "divide_by_train_scale", "weight": 0.25},
                {"metric_id": "task.lifecycle.macro_f1", "transform": "1-value", "weight": 0.25},
                {"metric_id": "selection.required_continuous.normalized_error", "transform": "identity", "weight": 0.25},
            ],
            "missing_common_term": "candidate_ineligible",
            "tie_break": "lower_inference_latency_then_lexical_method_id; missing_latency_is_positive_infinity",
        },
        "calibration_boundary": {
            "may_set": ["event_thresholds", "distribution_temperature", "conformal_quantiles"],
            "may_not_set": ["architecture", "checkpoint", "loss_weights", "training_budget"],
        },
        "event_threshold_policy": {
            "split": "calibration",
            "scope": "one global threshold per probabilistic event head and method",
            "objective": "maximize pooled F1 over calibration trajectories",
            "candidates": "sorted unique predicted probabilities plus 0 and 1",
            "comparison": "prediction_is_positive_when_probability_greater_than_or_equal_to_threshold",
            "tie_break": "higher_threshold",
            "locked_test_reuse": "forbidden",
            "deterministic_baselines": "use their fixed binary outputs without threshold fitting",
        },
        "literature_basis": {
            "10.1371/journal.pone.0118432": "AUPRC/average precision for imbalanced event prediction",
            "10.1198/016214506000001437": "proper scoring rules and logarithmic score",
            "10.1111/j.1467-9868.2007.00587.x": "calibration and sharpness of probabilistic forecasts",
            "CQR-NeurIPS-2019": "calibration-set conformal coverage and interval width",
            "DEC-TR-301": "Jain fairness index",
            "ITU-T-Y.1540": "IP performance quantity terminology",
            "JMLR-11-2079": "avoid model-selection overfitting by separating selection and final evaluation",
        },
        "reporting": {
            "primary_aggregation": "macro_mean_over_complete_environment_trajectories",
            "secondary_aggregation": "pooled_micro_for_additive_metrics_with_explicit_numerator_denominator; recompute_nonadditive_rank_metrics_from_pooled_labels",
            "experimental_units": {
                "environment_trajectory_seed": "AirFogSim scene trajectory; paired across methods",
                "training_seed": "independent model initialization/optimization replicate",
                "summary_order": "aggregate environment trajectories within each training seed, then summarize paired training-seed differences",
            },
            "uncertainty": "paired training-seed mean, sample std, paired t 95% CI when assumptions are declared; otherwise paired bootstrap over environment trajectories",
            "quantiles": "linear_interpolation_within_trajectory_then_macro_over_seeds",
            "retain_failures": True,
            "locked_test_policy": "single_use_in_R9_after_method_freeze",
        },
    }


def validate_evaluation_protocol(
    registry: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "metric_id", "layer", "direction", "formula", "unit", "numerator",
        "denominator", "source_fields", "mask_policy", "aggregation",
        "computable_when", "na_rule",
    }
    ids = [str(row.get("metric_id", "")) for row in registry]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("metric ids must be nonempty and unique")
    for row in registry:
        if not required <= set(row) or any(row.get(name) in (None, "") for name in required):
            raise ValueError(f"metric definition is incomplete: {row.get('metric_id')}")
    allowed_physical_edge_sources = {
        "physical_edge_state.distance",
        "physical_edge_state.relative_speed",
        "physical_edge_present",
        "physical_edge_feature_mask",
    }
    invalid_physical_sources = [
        source
        for row in registry
        for source in row.get("source_fields", [])
        if str(source).startswith("physical_edge")
        and str(source) not in allowed_physical_edge_sources
    ]
    if invalid_physical_sources:
        raise ValueError(
            "physical-edge sources must use the frozen geometry-only whitelist: "
            + ", ".join(sorted(set(invalid_physical_sources)))
        )
    split_roles = dict(protocol.get("split_roles", {}))
    fit = list(split_roles.get("fit", []))
    if "locked_test" in fit:
        raise ValueError("locked_test must never be used for fit")
    if split_roles.get("architecture_selection") == "calibration":
        raise ValueError("calibration must not select architecture")
    if protocol.get("method_budget_policy") != "equal_optimizer_steps_and_windows":
        raise ValueError("all methods require equal training budget")
    may_not_set = set(protocol.get("calibration_boundary", {}).get("may_not_set", []))
    if not {"architecture", "checkpoint", "loss_weights", "training_budget"} <= may_not_set:
        raise ValueError("calibration boundary is incomplete")
    metric_ids = set(ids)
    terms = protocol.get("checkpoint_selection", {}).get("terms", [])
    if any(row.get("metric_id") not in metric_ids for row in terms):
        raise ValueError("checkpoint score references an unknown metric")
    term_ids = [str(row.get("metric_id")) for row in terms]
    if len(term_ids) != len(set(term_ids)):
        raise ValueError("checkpoint terms must be unique")
    expected_transforms = {
        "event.information_link_activity.auprc": "1-value",
        "link.active_only_rate.mae": "divide_by_train_scale",
        "task.lifecycle.macro_f1": "1-value",
        "selection.required_continuous.normalized_error": "identity",
    }
    if len(terms) != 4 or set(term_ids) != set(expected_transforms):
        raise ValueError("checkpoint terms must match the four frozen protocol terms")
    if any(row.get("transform") != expected_transforms[row["metric_id"]] for row in terms):
        raise ValueError("checkpoint transforms must match the frozen protocol")
    weights = [float(row.get("weight", -1.0)) for row in terms]
    if any(abs(weight - 0.25) > 1e-12 for weight in weights) or abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("checkpoint weights must be four equal 0.25 weights summing to one")
    threshold_policy = dict(protocol.get("event_threshold_policy", {}))
    if threshold_policy.get("split") != "calibration":
        raise ValueError("event thresholds must be fitted only on calibration")
    binding = dict(protocol.get("data_binding", {}))
    environment_splits = dict(binding.get("environment_splits", {}))
    required_splits = {"train", "validation", "calibration", "locked_test"}
    if set(environment_splits) != required_splits or any(
        not environment_splits.get(split) for split in required_splits
    ):
        raise ValueError("data binding must freeze nonempty train, validation, calibration, and locked_test splits")
    flat_split_ids = [
        str(item)
        for split in required_splits
        for item in environment_splits[split]
    ]
    if len(flat_split_ids) != len(set(flat_split_ids)):
        raise ValueError("data binding environment trajectories must be split-disjoint")
    normalization_sha = str(binding.get("normalization_stats_sha256") or "")
    if len(normalization_sha) != 64 or any(char not in "0123456789abcdef" for char in normalization_sha):
        raise ValueError("data binding requires a lowercase SHA-256 for train-only normalization stats")
    if binding.get("locked_test_content_read") is not False:
        raise ValueError("locked_test content must remain unread before method freeze")
    mapping = build_factual_metric_mapping()
    mapping_ids = {row["canonical_metric_id"] for row in mapping}
    if not mapping_ids <= metric_ids:
        raise ValueError("factual metric mapping references an unknown canonical metric")
    checks = {
        "metric_definitions_complete": True,
        "metric_ids_unique": True,
        "teacher_aligned_information_edge_semantics": True,
        "locked_test_excluded_from_fit": True,
        "calibration_does_not_select_models": True,
        "equal_method_budget": True,
        "checkpoint_score_frozen": True,
        "event_threshold_policy_frozen": True,
        "data_binding_frozen": True,
        "factual_metric_mapping_complete": len(mapping) == 22 and mapping_ids <= metric_ids,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_protocol_ready": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


__all__ = [
    "SCHEMA_VERSION",
    "FAIR_PROTOCOL_VERSION",
    "build_metric_registry",
    "build_fair_experiment_protocol",
    "build_factual_metric_mapping",
    "validate_evaluation_protocol",
]
