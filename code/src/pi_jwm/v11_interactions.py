"""Local edge-step interaction features for PI-JWM v11 schema-v6 caches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .v11_selector import CandidateBatch


DEFAULT_TOKEN_CAPACITY = 72
REQUIRED_CURRENT_LINK_FEATURES = (
    "distance",
    "rate_sum",
    "csi_mean",
    "active_task_count",
    "allocated_rb_count",
)
POOL_STATISTIC_NAMES = (
    "count",
    "delta_sum",
    "delta_abs_sum",
    "delta_abs_max",
    *(f"weighted_current_{name}" for name in REQUIRED_CURRENT_LINK_FEATURES),
    "weighted_default_predicted_activity",
    "weighted_default_predicted_rate",
    "weighted_predicted_activity_delta",
    "weighted_predicted_rate_delta",
)


@dataclass(frozen=True)
class CandidateInteractionBatch:
    tokens: np.ndarray
    token_mask: np.ndarray
    edge_index: np.ndarray
    token_feature_names: tuple[str, ...]
    pooled_features: np.ndarray | None = None
    pooled_feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        tokens = np.asarray(self.tokens, dtype=np.float32)
        mask = np.asarray(self.token_mask, dtype=bool)
        edge_index = np.asarray(self.edge_index, dtype=np.int32)
        if tokens.ndim != 4:
            raise ValueError("interaction tokens must be [sample,candidate,token,feature]")
        if mask.shape != tokens.shape[:3] or edge_index.shape != mask.shape:
            raise ValueError("interaction token mask and edge index must match token dimensions")
        if len(self.token_feature_names) != tokens.shape[3]:
            raise ValueError("interaction token feature names must match feature dimension")
        if not np.all(np.isfinite(tokens)):
            raise ValueError("interaction tokens must be finite")
        if np.any(tokens[~mask] != 0.0):
            raise ValueError("padding interaction tokens must be zero")
        if np.any(edge_index[~mask] != -1) or np.any(edge_index[mask] < 0):
            raise ValueError("interaction edge indices must use -1 only for padding")
        pooled = self.pooled_features
        if pooled is not None:
            pooled_array = np.asarray(pooled, dtype=np.float32)
            if pooled_array.ndim != 3 or pooled_array.shape[:2] != tokens.shape[:2]:
                raise ValueError("pooled interaction features must be [sample,candidate,feature]")
            if len(self.pooled_feature_names) != pooled_array.shape[2]:
                raise ValueError("pooled interaction names must match feature dimension")
            if not np.all(np.isfinite(pooled_array)):
                raise ValueError("pooled interaction features must be finite")
            object.__setattr__(self, "pooled_features", pooled_array)
        elif self.pooled_feature_names:
            raise ValueError("pooled interaction names require pooled features")
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "token_mask", mask)
        object.__setattr__(self, "edge_index", edge_index)

    @property
    def token_count(self) -> np.ndarray:
        return self.token_mask.sum(axis=2).astype(np.int32)


def _prediction_tensor(
    predictions: dict[str, np.ndarray],
    name: str,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    values = np.asarray(predictions[name], dtype=np.float32)
    if values.shape[-1:] == (1,):
        values = values[..., 0]
    if values.shape != expected_shape:
        raise ValueError(f"{name} must match sample, step, and edge dimensions")
    return values


def build_candidate_interaction_tokens(
    candidate_actions: np.ndarray,
    predictions_by_candidate: list[dict[str, np.ndarray]],
    current_link_features: np.ndarray,
    action_feature_names: tuple[str, ...] | list[str],
    current_link_feature_names: tuple[str, ...] | list[str],
    default_index: int,
    token_capacity: int = DEFAULT_TOKEN_CAPACITY,
) -> CandidateInteractionBatch:
    """Build deterministic tokens for every modified candidate edge-step."""
    actions = np.asarray(candidate_actions, dtype=np.float32)
    if actions.ndim != 5 or actions.shape[-1] != 6:
        raise ValueError("candidate_actions must be [sample,candidate,step,edge,6]")
    sample_count, candidate_count, step_count, edge_count, action_dim = actions.shape
    action_names = tuple(str(value) for value in action_feature_names)
    if len(action_names) != action_dim or len(set(action_names)) != action_dim:
        raise ValueError("action feature names must contain six unique names in tensor order")
    if step_count != 3:
        raise ValueError("schema-v6 interaction tokens require exactly three rollout steps")
    default = int(default_index)
    if not 0 <= default < candidate_count:
        raise ValueError("default_index outside candidate dimension")
    capacity = int(token_capacity)
    if capacity < 1:
        raise ValueError("token capacity must be positive")
    if len(predictions_by_candidate) != candidate_count:
        raise ValueError("candidate predictions must match candidate dimension")

    current = np.asarray(current_link_features, dtype=np.float32)
    current_names = tuple(str(value) for value in current_link_feature_names)
    if current.ndim != 3 or current.shape[:2] != (sample_count, edge_count):
        raise ValueError("current_link_features must be [sample,edge,feature]")
    if len(current_names) != current.shape[2]:
        raise ValueError("current link feature names must match feature dimension")
    missing = [name for name in REQUIRED_CURRENT_LINK_FEATURES if name not in current_names]
    if missing:
        raise ValueError(f"current link features missing required fields: {missing}")
    current_indices = [current_names.index(name) for name in REQUIRED_CURRENT_LINK_FEATURES]
    current = current[:, :, current_indices]

    expected_prediction_shape = (sample_count, step_count, edge_count)
    activity = np.stack(
        [
            _prediction_tensor(predictions, "link_activity_prob", expected_prediction_shape)
            for predictions in predictions_by_candidate
        ],
        axis=1,
    )
    rate = np.stack(
        [
            _prediction_tensor(predictions, "link_rate_pred", expected_prediction_shape)
            for predictions in predictions_by_candidate
        ],
        axis=1,
    )
    action_delta = actions - actions[:, default : default + 1]
    modified = np.any(np.abs(action_delta) > 1e-8, axis=-1)

    token_names = (
        "step_0",
        "step_1",
        "step_2",
        *(f"default_action_{name}" for name in action_names),
        *(f"action_delta_{name}" for name in action_names),
        *(f"current_{name}" for name in REQUIRED_CURRENT_LINK_FEATURES),
        "default_predicted_activity",
        "default_predicted_rate",
        "predicted_activity_delta",
        "predicted_rate_delta",
        "action_delta_l1",
    )
    if len(token_names) != 25:
        raise RuntimeError("schema-v6 token feature contract must contain 25 fields")
    tokens = np.zeros((sample_count, candidate_count, capacity, 25), dtype=np.float32)
    token_mask = np.zeros((sample_count, candidate_count, capacity), dtype=bool)
    edge_index = np.full((sample_count, candidate_count, capacity), -1, dtype=np.int32)
    for sample_index in range(sample_count):
        for candidate_index in range(candidate_count):
            positions = np.argwhere(modified[sample_index, candidate_index])
            if positions.shape[0] > capacity:
                raise ValueError(
                    "modified edge-step token count exceeds fixed token capacity: "
                    f"sample={sample_index}, candidate={candidate_index}, "
                    f"count={positions.shape[0]}, capacity={capacity}"
                )
            for token_index, (step_index, edge) in enumerate(positions):
                row = tokens[sample_index, candidate_index, token_index]
                row[int(step_index)] = 1.0
                row[3:9] = actions[sample_index, default, step_index, edge]
                row[9:15] = action_delta[sample_index, candidate_index, step_index, edge]
                row[15:20] = current[sample_index, edge]
                default_activity = activity[sample_index, default, step_index, edge]
                default_rate = rate[sample_index, default, step_index, edge]
                row[20] = default_activity
                row[21] = default_rate
                row[22] = (
                    activity[sample_index, candidate_index, step_index, edge] - default_activity
                )
                row[23] = rate[sample_index, candidate_index, step_index, edge] - default_rate
                row[24] = np.sum(np.abs(row[9:15]))
                token_mask[sample_index, candidate_index, token_index] = True
                edge_index[sample_index, candidate_index, token_index] = int(edge)
    return CandidateInteractionBatch(
        tokens=tokens,
        token_mask=token_mask,
        edge_index=edge_index,
        token_feature_names=tuple(token_names),
    )


def pool_candidate_interactions(
    interactions: CandidateInteractionBatch,
    action_feature_names: tuple[str, ...] | list[str],
    epsilon: float = 1e-8,
) -> CandidateInteractionBatch:
    """Pool sparse edge-step tokens into fixed, directly auditable statistics."""
    action_names = tuple(str(value) for value in action_feature_names)
    if len(action_names) != 6 or len(set(action_names)) != 6:
        raise ValueError("pooling requires six unique action feature names")
    if len(POOL_STATISTIC_NAMES) != 13:
        raise RuntimeError("schema-v6 pooling contract must contain 13 statistics")
    names = tuple(
        f"step_{step}__{action_name}__{statistic}"
        for step in range(3)
        for action_name in action_names
        for statistic in POOL_STATISTIC_NAMES
    )
    tokens = interactions.tokens
    mask = interactions.token_mask
    sample_count, candidate_count = tokens.shape[:2]
    pooled = np.zeros((sample_count, candidate_count, len(names)), dtype=np.float32)
    output_index = 0
    for step in range(3):
        step_mask = mask & (tokens[..., step] > 0.5)
        for action_index in range(6):
            delta = tokens[..., 9 + action_index]
            weights = np.abs(delta)
            selected = step_mask & (weights > float(epsilon))
            selected_float = selected.astype(np.float32)
            selected_weights = np.where(selected, weights, 0.0)
            denominator = selected_weights.sum(axis=2)
            safe_denominator = np.where(denominator > 0.0, denominator, 1.0)
            block = np.zeros((sample_count, candidate_count, 13), dtype=np.float32)
            block[..., 0] = selected_float.sum(axis=2)
            block[..., 1] = np.where(selected, delta, 0.0).sum(axis=2)
            block[..., 2] = selected_weights.sum(axis=2)
            block[..., 3] = selected_weights.max(axis=2)
            weighted_sources = tokens[..., 15:24]
            block[..., 4:] = (
                (weighted_sources * selected_weights[..., None]).sum(axis=2)
                / safe_denominator[..., None]
            )
            block[denominator <= 0.0, 4:] = 0.0
            pooled[..., output_index : output_index + 13] = block
            output_index += 13
    return CandidateInteractionBatch(
        tokens=interactions.tokens,
        token_mask=interactions.token_mask,
        edge_index=interactions.edge_index,
        token_feature_names=interactions.token_feature_names,
        pooled_features=pooled,
        pooled_feature_names=names,
    )


def append_interaction_pooled_features(
    batch: CandidateBatch,
    interactions: CandidateInteractionBatch,
) -> CandidateBatch:
    """Return a CandidateBatch augmented with schema-v6 pooled features."""
    if interactions.pooled_features is None:
        raise ValueError("interaction batch must contain pooled features")
    if interactions.pooled_features.shape[:2] != batch.candidate_features.shape[:2]:
        raise ValueError("interaction and candidate batches must share sample/candidate dimensions")
    names = tuple(f"interaction_{name}" for name in interactions.pooled_feature_names)
    return CandidateBatch(
        context=batch.context,
        candidate_features=np.concatenate(
            [batch.candidate_features, interactions.pooled_features], axis=2
        ),
        candidate_mask=batch.candidate_mask,
        stage=batch.stage,
        feature_names=tuple(batch.feature_names) + names,
        candidate_names=batch.candidate_names,
        context_feature_names=batch.context_feature_names,
    )
