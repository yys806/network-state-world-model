"""Evaluation metrics for PI-JWM v6."""

from __future__ import annotations

import numpy as np


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    err = pred - true
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
    }


def activity_metrics(prob: np.ndarray, true: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    prob = np.asarray(prob, dtype=np.float64)
    true = np.asarray(true, dtype=np.int32)
    pred = (prob >= threshold).astype(np.int32)

    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + fp + fn + tn)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def active_rate_metrics(pred: np.ndarray, true: np.ndarray, active: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    active = np.asarray(active) > 0.5
    active_count = int(active.sum())
    if active_count == 0:
        return {"active_count": 0, "active_rmse": float("nan"), "active_mae": float("nan")}
    metrics = regression_metrics(pred[active], true[active])
    return {
        "active_count": active_count,
        "active_rmse": metrics["rmse"],
        "active_mae": metrics["mae"],
    }


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
