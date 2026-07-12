"""Active-link rate teacher utilities for PI-JWM v7.

The teacher is a classical active-rate specialist trained only on true active
train link-step samples. Its predictions are used as optional auxiliary targets
for the neural world model; validation and test metrics still use ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pi_jwm.v6_data import build_physical_edge_history


@dataclass(frozen=True)
class ActiveRateTeacherResult:
    arrays: dict[str, np.ndarray]
    summary: dict[str, float | int | str]


def add_random_forest_active_rate_teacher(
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
    seed: int,
    n_estimators: int = 400,
    min_samples_leaf: int = 2,
    max_features: float = 0.8,
) -> ActiveRateTeacherResult:
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:
        raise RuntimeError(
            "random_forest_active teacher requires scikit-learn. Install scikit-learn on the training environment "
            "or run without --rate-teacher-mode random_forest_active."
        ) from exc

    features, target, active = build_active_rate_teacher_table(arrays)
    flat_train_idx = sample_indices_to_flat_indices(
        train_idx,
        arrays["y_link_active"].shape[1],
        arrays["y_link_active"].shape[2],
    )
    train_active_idx = flat_train_idx[active[flat_train_idx]]
    if len(train_active_idx) == 0:
        raise ValueError("No active train link-step samples are available for rate teacher training.")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(features[train_active_idx], target[train_active_idx])
    teacher = np.clip(model.predict(features), 0.0, None).astype(np.float32)
    teacher = teacher.reshape(arrays["y_link_rate"].shape)

    teacher_arrays = dict(arrays)
    teacher_arrays["y_link_rate_teacher"] = teacher
    teacher_arrays["y_link_rate_teacher_mask"] = arrays["y_link_active"].astype(np.float32)
    summary = {
        "rate_teacher_mode": "random_forest_active",
        "rate_teacher_train_active_count": int(len(train_active_idx)),
        "rate_teacher_feature_dim": int(features.shape[1]),
        "rate_teacher_n_estimators": int(n_estimators),
        "rate_teacher_min_samples_leaf": int(min_samples_leaf),
        "rate_teacher_max_features": float(max_features),
    }
    return ActiveRateTeacherResult(arrays=teacher_arrays, summary=summary)


def add_ridge_active_rate_teacher(
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
    ridge_lambda: float = 10.0,
) -> ActiveRateTeacherResult:
    features, target, active = build_active_rate_teacher_table(arrays)
    flat_train_idx = sample_indices_to_flat_indices(
        train_idx,
        arrays["y_link_active"].shape[1],
        arrays["y_link_active"].shape[2],
    )
    train_active_idx = flat_train_idx[active[flat_train_idx]]
    if len(train_active_idx) == 0:
        raise ValueError("No active train link-step samples are available for rate teacher training.")

    x_train = features[train_active_idx].astype(np.float64)
    y_train = target[train_active_idx].astype(np.float64)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train_n = add_bias((x_train - mean) / std)
    x_all_n = add_bias((features.astype(np.float64) - mean) / std)
    weights = fit_ridge(x_train_n, y_train, ridge_lambda)
    teacher = np.clip(x_all_n @ weights, 0.0, None).astype(np.float32)
    teacher = teacher.reshape(arrays["y_link_rate"].shape)

    teacher_arrays = dict(arrays)
    teacher_arrays["y_link_rate_teacher"] = teacher
    teacher_arrays["y_link_rate_teacher_mask"] = arrays["y_link_active"].astype(np.float32)
    summary = {
        "rate_teacher_mode": "ridge_active",
        "rate_teacher_train_active_count": int(len(train_active_idx)),
        "rate_teacher_feature_dim": int(features.shape[1]),
        "rate_teacher_ridge_lambda": float(ridge_lambda),
    }
    return ActiveRateTeacherResult(arrays=teacher_arrays, summary=summary)


def build_active_rate_teacher_table(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_samples, horizon, num_edges = arrays["y_link_active"].shape
    physical = build_physical_edge_history(
        arrays["x_node"],
        arrays["edge_src_idx"],
        arrays["edge_dst_idx"],
        arrays["valid_edge_node"],
    ).numpy()

    base_parts = [
        arrays["x_link"][:, -1],
        arrays["x_link"].mean(axis=1),
        physical[:, -1],
        physical.mean(axis=1),
        arrays["edge_a_hist"][:, -1],
        arrays["edge_a_hist"].sum(axis=1),
    ]
    task_features = np.concatenate([arrays["x_task"][:, -1], arrays["x_task"].mean(axis=1)], axis=-1)
    task_repeated = np.repeat(task_features[:, None, :], num_edges, axis=1)

    rows = []
    for step in range(horizon):
        step_col = np.full((num_samples, num_edges, 1), float(step), dtype=np.float32)
        rows.append(
            np.concatenate(
                [*base_parts, arrays["edge_a_future"][:, step], task_repeated, step_col],
                axis=-1,
            )
        )
    features = np.stack(rows, axis=1).reshape(num_samples * horizon * num_edges, -1).astype(np.float32)
    target = arrays["y_link_rate"].reshape(-1).astype(np.float32)
    active = arrays["y_link_active"].reshape(-1) > 0.5
    return features, target, active


def sample_indices_to_flat_indices(sample_idx: np.ndarray, horizon: int, num_edges: int) -> np.ndarray:
    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    offsets = sample_idx[:, None, None] * horizon * num_edges
    h = np.arange(horizon, dtype=np.int64)[None, :, None] * num_edges
    e = np.arange(num_edges, dtype=np.int64)[None, None, :]
    return (offsets + h + e).reshape(-1)


def add_bias(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values, np.ones((values.shape[0], 1), dtype=values.dtype)], axis=1)


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    regularizer = ridge_lambda * np.eye(x.shape[1], dtype=x.dtype)
    regularizer[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y)
