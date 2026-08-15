"""Teacher-aligned one-step baseline evaluation for PI-JWM v3 tensors."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .airfogsim_sparse_diagnostics_v2 import average_precision
from .airfogsim_teacher_tensor_v3 import (
    INFORMATION_EDGE_FEATURES,
    INFORMATION_NODE_FEATURES,
    PHYSICAL_EDGE_FEATURES,
    PHYSICAL_NODE_FEATURES,
)
from .airfogsim_tensor_v2 import FLOW_FEATURES, TASK_FEATURES
from .formal_airfogsim_graph_v1 import FORMAL_DAG_STATE_FEATURES


SCHEMA_VERSION = "PIJWM-Teacher-Evaluation-v3"
METHODS = ("zero_state", "last_persistence")
SELECTION_COMPONENTS = (
    "state.physical_node.position.rmse",
    "state.physical_node.motion.rmse",
    "state.physical_edge.distance.rmse",
    "state.physical_edge.relative_speed.rmse",
    "state.information_node.queue.mae",
    "state.information_node.cpu_backlog.mae",
    "state.information_edge.rate.rmse",
    "state.flow.remaining_data.mae",
    "state.task.deadline_remaining.mae",
    "state.dag.unfinished_parent_count.mae",
)


def _computed(
    value: float,
    *,
    count: int,
    numerator: float,
    denominator: float,
    unit: str,
    source_fields: Sequence[str],
) -> dict[str, Any]:
    return {
        "status": "computed",
        "value": float(value),
        "count": int(count),
        "numerator": float(numerator),
        "denominator": float(denominator),
        "unit": unit,
        "source_fields": list(source_fields),
        "reason": None,
    }


def _not_computable(
    *, unit: str, source_fields: Sequence[str], reason: str
) -> dict[str, Any]:
    return {
        "status": "not_computable",
        "value": None,
        "count": 0,
        "numerator": None,
        "denominator": 0.0,
        "unit": unit,
        "source_fields": list(source_fields),
        "reason": reason,
    }


def _continuous_metric(
    target: np.ndarray,
    previous: np.ndarray,
    valid: np.ndarray,
    *,
    method: str,
    previous_valid: np.ndarray | None = None,
    kind: str,
    unit: str,
    source_fields: Sequence[str],
    zero_value: Any = 0.0,
) -> dict[str, Any]:
    valid = np.asarray(valid, dtype=bool)
    if method == "last_persistence":
        if previous_valid is None:
            raise ValueError("last_persistence requires an explicit previous_valid mask")
        valid = valid & np.asarray(previous_valid, dtype=bool)
    count = int(valid.sum())
    if count == 0:
        return _not_computable(
            unit=unit,
            source_fields=source_fields,
            reason="no valid target observations",
        )
    prediction = previous if method == "last_persistence" else zero_value
    error = np.asarray(prediction - target, dtype=np.float64)
    if kind == "rmse_vector":
        squared = np.sum(np.square(error), axis=-1)
        numerator = float(squared[valid].sum())
        value = float(np.sqrt(numerator / count))
    elif kind == "rmse":
        squared = np.square(error)
        numerator = float(squared[valid].sum())
        value = float(np.sqrt(numerator / count))
    elif kind == "mae":
        absolute = np.abs(error)
        numerator = float(absolute[valid].sum())
        value = float(numerator / count)
    else:  # pragma: no cover - internal misuse guard
        raise ValueError(f"unknown continuous metric kind: {kind}")
    return _computed(
        value,
        count=count,
        numerator=numerator,
        denominator=float(count),
        unit=unit,
        source_fields=source_fields,
    )


def _binary_metric(
    target: np.ndarray,
    previous: np.ndarray,
    valid: np.ndarray,
    *,
    method: str,
    previous_valid: np.ndarray | None = None,
    source_fields: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    valid = np.asarray(valid, dtype=bool)
    if method == "last_persistence":
        if previous_valid is None:
            raise ValueError("last_persistence requires an explicit previous_valid mask")
        valid = valid & np.asarray(previous_valid, dtype=bool)
    labels = np.asarray(target, dtype=bool)[valid]
    predictions = (
        np.asarray(previous, dtype=bool)[valid]
        if method == "last_persistence"
        else np.zeros(int(valid.sum()), dtype=bool)
    )
    count = int(labels.size)
    if count == 0:
        reason = "no valid event labels"
        return (
            _not_computable(unit="ratio", source_fields=source_fields, reason=reason),
            _not_computable(unit="ratio", source_fields=source_fields, reason=reason),
        )
    tp = int(np.logical_and(predictions, labels).sum())
    fp = int(np.logical_and(predictions, ~labels).sum())
    fn = int(np.logical_and(~predictions, labels).sum())
    f1_denominator = 2 * tp + fp + fn
    if f1_denominator == 0:
        f1 = _not_computable(
            unit="ratio",
            source_fields=source_fields,
            reason="truth and prediction contain no positive event",
        )
        f1.update({"tp": tp, "fp": fp, "fn": fn})
    else:
        f1 = _computed(
            2.0 * tp / f1_denominator,
            count=count,
            numerator=float(2 * tp),
            denominator=float(f1_denominator),
            unit="ratio",
            source_fields=source_fields,
        )
        f1.update({"tp": tp, "fp": fp, "fn": fn})
    ap_value = average_precision(predictions.astype(np.float64), labels)
    if ap_value is None:
        auprc = _not_computable(
            unit="ratio",
            source_fields=source_fields,
            reason="truth contains no positive event",
        )
    else:
        auprc = _computed(
            ap_value,
            count=count,
            numerator=float(ap_value),
            denominator=1.0,
            unit="ratio",
            source_fields=source_fields,
        )
    score_bins = []
    prediction_scores = predictions.astype(np.float64)
    for score in sorted(np.unique(prediction_scores), reverse=True):
        selected = prediction_scores == score
        score_bins.append(
            {
                "score": float(score),
                "positive": int(np.logical_and(selected, labels).sum()),
                "negative": int(np.logical_and(selected, ~labels).sum()),
            }
        )
    auprc["score_bins"] = score_bins
    return f1, auprc


def _macro_f1(
    target: np.ndarray,
    previous: np.ndarray,
    valid: np.ndarray,
    *,
    method: str,
    previous_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    valid = np.asarray(valid, dtype=bool)
    if method == "last_persistence":
        if previous_valid is None:
            raise ValueError("last_persistence requires an explicit previous_valid mask")
        valid = valid & np.asarray(previous_valid, dtype=bool)
    labels = np.asarray(target, dtype=np.int64)[valid]
    predictions = (
        np.asarray(previous, dtype=np.int64)[valid]
        if method == "last_persistence"
        else np.zeros(labels.shape, dtype=np.int64)
    )
    if labels.size == 0:
        return _not_computable(
            unit="ratio",
            source_fields=["task_lifecycle_index", "task_present", "task_valid"],
            reason="no valid task lifecycle labels",
        )
    class_f1 = []
    confusion = np.zeros((5, 5), dtype=np.int64)
    for truth, predicted in zip(labels, predictions):
        if 0 <= truth < 5 and 0 <= predicted < 5:
            confusion[int(truth), int(predicted)] += 1
    for class_index in range(5):
        truth = labels == class_index
        predicted = predictions == class_index
        tp = int(np.logical_and(truth, predicted).sum())
        fp = int(np.logical_and(~truth, predicted).sum())
        fn = int(np.logical_and(truth, ~predicted).sum())
        denominator = 2 * tp + fp + fn
        class_f1.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
    value = float(np.mean(class_f1))
    result = _computed(
        value,
        count=int(labels.size),
        numerator=float(sum(class_f1)),
        denominator=float(len(class_f1)),
        unit="ratio",
        source_fields=["task_lifecycle_index", "task_present", "task_valid"],
    )
    result["confusion_matrix"] = confusion.tolist()
    return result


def _validate_arrays(arrays: Mapping[str, np.ndarray], method: str) -> int:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    required = (
        "time",
        "physical_node_state",
        "physical_node_feature_mask",
        "physical_node_present",
        "physical_edge_state",
        "physical_edge_feature_mask",
        "physical_edge_present",
        "information_node_state",
        "information_node_feature_mask",
        "information_node_present",
        "information_edge_state",
        "information_edge_feature_mask",
        "information_edge_present",
        "data_flow_state",
        "data_flow_present",
        "data_flow_valid",
        "task_state",
        "task_present",
        "task_valid",
        "task_lifecycle_index",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"missing teacher tensor arrays: {missing}")
    time_count = int(np.asarray(arrays["time"]).shape[0])
    if time_count < 2:
        raise ValueError("trajectory must contain at least two time steps")
    temporal = [name for name in required if name not in {"data_flow_valid", "task_valid"}]
    inconsistent = [
        name for name in temporal if int(np.asarray(arrays[name]).shape[0]) != time_count
    ]
    if inconsistent:
        raise ValueError(f"temporal arrays have inconsistent length: {inconsistent}")
    lifecycle = np.asarray(arrays["task_lifecycle_index"], dtype=np.int64)
    task_present = np.asarray(arrays["task_present"], dtype=bool)
    task_valid = np.asarray(arrays["task_valid"], dtype=bool)[None, :]
    lifecycle_observed = task_present & np.broadcast_to(task_valid, task_present.shape)
    if np.any(lifecycle_observed & ((lifecycle < -1) | (lifecycle >= 5))):
        raise ValueError(
            "observed task lifecycle labels must be -1 (unknown) or in the frozen range [0, 4]"
        )
    return time_count


def _train_mean(
    normalization_stats: Mapping[str, Any] | None,
    feature_group: str,
    indices: int | Sequence[int],
) -> float | np.ndarray:
    if normalization_stats is None:
        return 0.0
    if normalization_stats.get("source_split") != "train":
        raise ValueError("normalization_stats must be computed from the train split")
    features = normalization_stats.get("features", {})
    if feature_group not in features or "mean" not in features[feature_group]:
        raise ValueError(f"normalization_stats is missing train mean for {feature_group}")
    mean = np.asarray(features[feature_group]["mean"], dtype=np.float64)
    selected = mean[indices]
    return float(selected) if np.ndim(selected) == 0 else selected


def evaluate_teacher_trajectory(
    arrays: Mapping[str, np.ndarray],
    *,
    method: str,
    normalization_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a deterministic one-step baseline on one v3 trajectory."""

    time_count = _validate_arrays(arrays, method)
    metrics: dict[str, dict[str, Any]] = {}

    pn = np.asarray(arrays["physical_node_state"])
    pn_mask = np.asarray(arrays["physical_node_feature_mask"], dtype=bool)
    pn_present = np.asarray(arrays["physical_node_present"], dtype=bool)
    position_indices = [PHYSICAL_NODE_FEATURES.index(name) for name in ("x", "y", "z")]
    position_valid = pn_present[1:] & np.all(pn_mask[1:, :, position_indices], axis=-1)
    position_previous_valid = pn_present[:-1] & np.all(
        pn_mask[:-1, :, position_indices], axis=-1
    )
    metrics["state.physical_node.position.rmse"] = _continuous_metric(
        pn[1:, :, position_indices],
        pn[:-1, :, position_indices],
        position_valid,
        method=method,
        previous_valid=position_previous_valid,
        kind="rmse_vector",
        unit="m",
        source_fields=[
            "physical_node_state.x",
            "physical_node_state.y",
            "physical_node_state.z",
        ],
        zero_value=_train_mean(
            normalization_stats, "physical_node_state", position_indices
        ),
    )
    speed = PHYSICAL_NODE_FEATURES.index("speed")
    metrics["state.physical_node.motion.rmse"] = _continuous_metric(
        pn[1:, :, speed],
        pn[:-1, :, speed],
        pn_present[1:] & pn_mask[1:, :, speed],
        method=method,
        previous_valid=pn_present[:-1] & pn_mask[:-1, :, speed],
        kind="rmse",
        unit="m/s",
        source_fields=["physical_node_state.speed"],
        zero_value=_train_mean(normalization_stats, "physical_node_state", speed),
    )

    pe = np.asarray(arrays["physical_edge_state"])
    pe_mask = np.asarray(arrays["physical_edge_feature_mask"], dtype=bool)
    pe_present = np.asarray(arrays["physical_edge_present"], dtype=bool)
    for metric_id, feature, unit in (
        ("state.physical_edge.distance.rmse", "distance", "m"),
        ("state.physical_edge.relative_speed.rmse", "relative_speed", "m/s"),
    ):
        index = PHYSICAL_EDGE_FEATURES.index(feature)
        metrics[metric_id] = _continuous_metric(
            pe[1:, :, index],
            pe[:-1, :, index],
            pe_present[1:] & pe_mask[1:, :, index],
            method=method,
            previous_valid=pe_present[:-1] & pe_mask[:-1, :, index],
            kind="rmse",
            unit=unit,
            source_fields=[f"physical_edge_state.{feature}"],
            zero_value=_train_mean(
                normalization_stats, "physical_edge_state", index
            ),
        )

    information_node = np.asarray(arrays["information_node_state"])
    information_node_mask = np.asarray(arrays["information_node_feature_mask"], dtype=bool)
    information_node_present = np.asarray(arrays["information_node_present"], dtype=bool)
    queue_indices = [
        INFORMATION_NODE_FEATURES.index(name)
        for name in ("unassigned_queue_count", "tx_queue_count", "return_queue_count")
    ]
    queue_valid = (
        information_node_present[1:, :, None]
        & information_node_mask[1:, :, queue_indices]
    )
    queue_previous_valid = (
        information_node_present[:-1, :, None]
        & information_node_mask[:-1, :, queue_indices]
    )
    metrics["state.information_node.queue.mae"] = _continuous_metric(
        information_node[1:, :, queue_indices],
        information_node[:-1, :, queue_indices],
        queue_valid,
        method=method,
        previous_valid=queue_previous_valid,
        kind="mae",
        unit="count",
        source_fields=[f"information_node_state.{INFORMATION_NODE_FEATURES[index]}" for index in queue_indices],
        zero_value=_train_mean(
            normalization_stats, "information_node_state", queue_indices
        ),
    )
    backlog = INFORMATION_NODE_FEATURES.index("cpu_backlog")
    metrics["state.information_node.cpu_backlog.mae"] = _continuous_metric(
        information_node[1:, :, backlog],
        information_node[:-1, :, backlog],
        information_node_present[1:] & information_node_mask[1:, :, backlog],
        method=method,
        previous_valid=(
            information_node_present[:-1]
            & information_node_mask[:-1, :, backlog]
        ),
        kind="mae",
        unit="cycles",
        source_fields=["information_node_state.cpu_backlog"],
        zero_value=_train_mean(
            normalization_stats, "information_node_state", backlog
        ),
    )

    information_edge = np.asarray(arrays["information_edge_state"])
    information_edge_mask = np.asarray(arrays["information_edge_feature_mask"], dtype=bool)
    information_edge_present = np.asarray(arrays["information_edge_present"], dtype=bool)
    active = INFORMATION_EDGE_FEATURES.index("outcome.active_task_count")
    rate = INFORMATION_EDGE_FEATURES.index("outcome.rate_sum")
    link_valid = information_edge_present[1:] & information_edge_mask[1:, :, active]
    activity_f1, activity_auprc = _binary_metric(
        information_edge[1:, :, active] > 0,
        information_edge[:-1, :, active] > 0,
        link_valid,
        method=method,
        previous_valid=(
            information_edge_present[:-1]
            & information_edge_mask[:-1, :, active]
        ),
        source_fields=[
            "information_edge_state.outcome.active_task_count",
            "information_edge_present",
            "information_edge_feature_mask",
        ],
    )
    metrics["event.information_link_activity.f1"] = activity_f1
    metrics["event.information_link_activity.auprc"] = activity_auprc
    rate_valid = information_edge_present[1:] & information_edge_mask[1:, :, rate]
    metrics["state.information_edge.rate.rmse"] = _continuous_metric(
        information_edge[1:, :, rate],
        information_edge[:-1, :, rate],
        rate_valid,
        method=method,
        previous_valid=(
            information_edge_present[:-1]
            & information_edge_mask[:-1, :, rate]
        ),
        kind="rmse",
        unit="Mbps",
        source_fields=["information_edge_state.outcome.rate_sum"],
        zero_value=_train_mean(
            normalization_stats, "information_edge_state", rate
        ),
    )
    active_rate_valid = rate_valid & link_valid & (information_edge[1:, :, active] > 0)
    metrics["link.active_only_rate.mae"] = _continuous_metric(
        information_edge[1:, :, rate],
        information_edge[:-1, :, rate],
        active_rate_valid,
        method=method,
        previous_valid=(
            information_edge_present[:-1]
            & information_edge_mask[:-1, :, rate]
            & information_edge_mask[:-1, :, active]
        ),
        kind="mae",
        unit="Mbps",
        source_fields=[
            "information_edge_state.outcome.rate_sum",
            "information_edge_state.outcome.active_task_count",
        ],
        zero_value=_train_mean(
            normalization_stats, "information_edge_state", rate
        ),
    )

    flow_present = np.asarray(arrays["data_flow_present"], dtype=bool)
    flow_valid = np.asarray(arrays["data_flow_valid"], dtype=bool)[None, :]
    flow_f1, _ = _binary_metric(
        flow_present[1:],
        flow_present[:-1],
        np.broadcast_to(flow_valid, flow_present[1:].shape),
        method=method,
        previous_valid=np.broadcast_to(flow_valid, flow_present[:-1].shape),
        source_fields=["data_flow_present", "data_flow_valid"],
    )
    metrics["event.flow_present.f1"] = flow_f1
    flow_state = np.asarray(arrays["data_flow_state"])
    remaining = FLOW_FEATURES.index("remaining_data")
    metrics["state.flow.remaining_data.mae"] = _continuous_metric(
        flow_state[1:, :, remaining],
        flow_state[:-1, :, remaining],
        flow_present[1:] & np.broadcast_to(flow_valid, flow_present[1:].shape),
        method=method,
        previous_valid=(
            flow_present[:-1]
            & np.broadcast_to(flow_valid, flow_present[:-1].shape)
        ),
        kind="mae",
        unit="MB",
        source_fields=["data_flow_state.remaining_data"],
        zero_value=_train_mean(normalization_stats, "data_flow_state", remaining),
    )

    task_present = np.asarray(arrays["task_present"], dtype=bool)
    task_valid = np.asarray(arrays["task_valid"], dtype=bool)[None, :]
    broadcast_task_valid = np.broadcast_to(task_valid, task_present[1:].shape)
    task_f1, _ = _binary_metric(
        task_present[1:],
        task_present[:-1],
        broadcast_task_valid,
        method=method,
        previous_valid=np.broadcast_to(task_valid, task_present[:-1].shape),
        source_fields=["task_present", "task_valid"],
    )
    metrics["event.task_present.f1"] = task_f1
    task_state = np.asarray(arrays["task_state"])
    deadline = TASK_FEATURES.index("deadline_remaining")
    metrics["state.task.deadline_remaining.mae"] = _continuous_metric(
        task_state[1:, :, deadline],
        task_state[:-1, :, deadline],
        task_present[1:] & broadcast_task_valid,
        method=method,
        previous_valid=(
            task_present[:-1]
            & np.broadcast_to(task_valid, task_present[:-1].shape)
        ),
        kind="mae",
        unit="s",
        source_fields=["task_state.deadline_remaining"],
        zero_value=_train_mean(normalization_stats, "task_state", deadline),
    )
    lifecycle = np.asarray(arrays["task_lifecycle_index"])
    lifecycle_valid = (
        task_present[1:]
        & broadcast_task_valid
        & (lifecycle[1:] >= 0)
        & (lifecycle[1:] < 5)
    )
    metrics["task.lifecycle.macro_f1"] = _macro_f1(
        lifecycle[1:],
        lifecycle[:-1],
        lifecycle_valid,
        previous_valid=(
            task_present[:-1]
            & np.broadcast_to(task_valid, task_present[:-1].shape)
            & (lifecycle[:-1] >= 0)
            & (lifecycle[:-1] < 5)
        ),
        method=method,
    )

    if "task_dag_state" in arrays and "task_dag_state_present" in arrays:
        dag = np.asarray(arrays["task_dag_state"])
        dag_present = np.asarray(arrays["task_dag_state_present"], dtype=bool)
        unfinished_parent_count = FORMAL_DAG_STATE_FEATURES.index(
            "unfinished_parent_count"
        )
        metrics["state.dag.unfinished_parent_count.mae"] = _continuous_metric(
            dag[1:, :, unfinished_parent_count],
            dag[:-1, :, unfinished_parent_count],
            dag_present[1:],
            method=method,
            previous_valid=dag_present[:-1],
            kind="mae",
            unit="count",
            source_fields=["task_dag_state.unfinished_parent_count"],
            zero_value=_train_mean(
                normalization_stats,
                "task_dag_state",
                unfinished_parent_count,
            ),
        )
    else:
        metrics["state.dag.unfinished_parent_count.mae"] = _not_computable(
            unit="count",
            source_fields=["task_dag_state.unfinished_parent_count"],
            reason="DAG state is absent from this trajectory",
        )

    selection_sources = [*SELECTION_COMPONENTS, "train_only_selection_scales"]
    metrics["selection.required_continuous.normalized_error"] = _not_computable(
        unit="normalized ratio",
        source_fields=selection_sources,
        reason="defined only after split-level pooling of the ten component metrics",
    )

    for metric_id, unit in (
        ("uncertainty.nll", "nats"),
        ("uncertainty.coverage_95", "ratio"),
        ("uncertainty.interval_width_95", "target unit"),
    ):
        metrics[metric_id] = _not_computable(
            unit=unit,
            source_fields=["predictive_distribution"],
            reason=f"{method} is deterministic and emits no predictive distribution",
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "prediction_offset_steps": 1,
        "time_step_count": time_count,
        "evaluation_transition_count": time_count - 1,
        "continuous_baseline": (
            "train_feature_mean"
            if method == "zero_state" and normalization_stats is not None
            else "literal_zero_fallback"
            if method == "zero_state"
            else "previous_time_step"
        ),
        "metrics": metrics,
    }


def _pooled_binary_f1(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows or any(key not in row for row in rows for key in ("tp", "fp", "fn")):
        return {"status": "not_computable", "value": None, "reason": "confusion counts unavailable"}
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return {"status": "not_computable", "value": None, "numerator": 0.0, "denominator": 0.0, "reason": "pooled truth and prediction contain no positive event"}
    return {"status": "computed", "value": 2.0 * tp / denominator, "numerator": float(2 * tp), "denominator": float(denominator), "tp": tp, "fp": fp, "fn": fn, "reason": None}


def _pooled_average_precision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bins: dict[float, list[int]] = {}
    for row in rows:
        for item in row.get("score_bins", []):
            slot = bins.setdefault(float(item["score"]), [0, 0])
            slot[0] += int(item["positive"])
            slot[1] += int(item["negative"])
    positives = sum(value[0] for value in bins.values())
    if positives == 0:
        return {"status": "not_computable", "value": None, "reason": "pooled truth contains no positive event"}
    cumulative_positive = 0
    cumulative_total = 0
    previous_recall = 0.0
    ap = 0.0
    for score in sorted(bins, reverse=True):
        positive, negative = bins[score]
        cumulative_positive += positive
        cumulative_total += positive + negative
        recall = cumulative_positive / positives
        precision = cumulative_positive / cumulative_total
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return {"status": "computed", "value": float(ap), "numerator": float(ap), "denominator": 1.0, "reason": None, "score_bin_count": len(bins)}


def _pooled_lifecycle_macro_f1(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matrices = [np.asarray(row["confusion_matrix"], dtype=np.int64) for row in rows if "confusion_matrix" in row]
    if not matrices:
        return {"status": "not_computable", "value": None, "reason": "five-class confusion matrices unavailable"}
    confusion = np.sum(matrices, axis=0)
    class_f1 = []
    for index in range(5):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        denominator = 2 * tp + fp + fn
        class_f1.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
    return {"status": "computed", "value": float(np.mean(class_f1)), "numerator": float(sum(class_f1)), "denominator": 5.0, "reason": None, "confusion_matrix": confusion.tolist()}


def compute_selection_continuous_from_summary(
    summary_metrics: Mapping[str, Mapping[str, Any]],
    selection_scales: Mapping[str, float],
) -> dict[str, Any]:
    invalid = []
    normalized = []
    for metric_id in SELECTION_COMPONENTS:
        pooled = summary_metrics.get(metric_id, {}).get("pooled_micro", {})
        scale = selection_scales.get(metric_id)
        if pooled.get("status") != "computed" or scale is None or not np.isfinite(float(scale)) or float(scale) <= 0.0:
            invalid.append(metric_id)
        else:
            normalized.append(float(pooled["value"]) / float(scale))
    if invalid:
        return _not_computable(
            unit="normalized ratio",
            source_fields=[*SELECTION_COMPONENTS, "train_only_selection_scales"],
            reason="missing pooled component or train scale: " + ", ".join(invalid),
        )
    return _computed(
        float(np.mean(normalized)),
        count=len(normalized),
        numerator=float(sum(normalized)),
        denominator=float(len(normalized)),
        unit="normalized ratio",
        source_fields=[*SELECTION_COMPONENTS, "train_only_selection_scales"],
    )


def summarize_teacher_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    selection_scales: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one teacher evaluation report is required")
    metric_ids = sorted(
        {metric_id for report in reports for metric_id in report.get("metrics", {})}
    )
    summary: dict[str, dict[str, Any]] = {}
    for metric_id in metric_ids:
        rows = [report.get("metrics", {}).get(metric_id) for report in reports]
        rows = [row for row in rows if row is not None]
        values = [float(row["value"]) for row in rows if row.get("status") == "computed"]
        computed_rows = [row for row in rows if row.get("status") == "computed"]
        pooled_micro: dict[str, Any]
        if metric_id.endswith(".auprc"):
            pooled_micro = _pooled_average_precision(rows)
        elif metric_id.endswith(".macro_f1"):
            pooled_micro = _pooled_lifecycle_macro_f1(rows)
        elif metric_id.endswith(".f1"):
            pooled_micro = _pooled_binary_f1(rows)
        elif not computed_rows or any(
            row.get("numerator") is None or row.get("denominator") in (None, 0)
            for row in computed_rows
        ):
            pooled_micro = {
                "status": "not_computable",
                "value": None,
                "numerator": None,
                "denominator": None,
                "reason": "additive numerator and denominator are unavailable",
            }
        else:
            numerator = float(sum(float(row["numerator"]) for row in computed_rows))
            denominator = float(sum(float(row["denominator"]) for row in computed_rows))
            pooled_value = (
                float(np.sqrt(numerator / denominator))
                if metric_id.endswith(".rmse")
                else float(numerator / denominator)
            )
            pooled_micro = {
                "status": "computed",
                "value": pooled_value,
                "numerator": numerator,
                "denominator": denominator,
                "reason": None,
            }
        if not values:
            reasons = sorted({str(row.get("reason")) for row in rows if row.get("reason")})
            summary[metric_id] = {
                "status": "not_computable",
                "macro_mean": None,
                "sample_std": None,
                "bootstrap_95_ci": [None, None],
                "computed_seed_count": 0,
                "computed_trajectory_count": 0,
                "not_computable_seed_count": len(rows),
                "not_computable_trajectory_count": len(rows),
                "reasons": reasons,
                "pooled_micro": pooled_micro,
            }
            continue
        if len(values) > 1:
            rng = np.random.default_rng(20260803)
            value_array = np.asarray(values, dtype=np.float64)
            bootstrap_means = np.mean(
                rng.choice(value_array, size=(2000, len(values)), replace=True),
                axis=1,
            )
            bootstrap_ci = [
                float(np.quantile(bootstrap_means, 0.025, method="linear")),
                float(np.quantile(bootstrap_means, 0.975, method="linear")),
            ]
        else:
            bootstrap_ci = [None, None]
        summary[metric_id] = {
            "status": "computed",
            "macro_mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "bootstrap_95_ci": bootstrap_ci,
            "computed_seed_count": len(values),
            "computed_trajectory_count": len(values),
            "not_computable_seed_count": len(rows) - len(values),
            "not_computable_trajectory_count": len(rows) - len(values),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "pooled_micro": pooled_micro,
        }
    if selection_scales is not None:
        selection = compute_selection_continuous_from_summary(summary, selection_scales)
        summary["selection.required_continuous.normalized_error"] = {
            "status": selection["status"],
            "macro_mean": selection["value"],
            "sample_std": None,
            "bootstrap_95_ci": [None, None],
            "computed_seed_count": 0,
            "computed_trajectory_count": 0,
            "not_computable_seed_count": 0,
            "not_computable_trajectory_count": 0,
            "pooled_micro": selection,
            "aggregation_level": "split_level_after_component_pooling",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "report_count": len(reports),
        "environment_trajectory_count": len(reports),
        "methods": sorted({str(report.get("method")) for report in reports}),
        "metrics": summary,
    }


__all__ = [
    "METHODS",
    "SELECTION_COMPONENTS",
    "SCHEMA_VERSION",
    "evaluate_teacher_trajectory",
    "compute_selection_continuous_from_summary",
    "summarize_teacher_reports",
]
