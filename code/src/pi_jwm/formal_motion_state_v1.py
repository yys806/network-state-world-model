"""Causal motion features derived from formal node-position histories."""

from __future__ import annotations

from typing import Any

import numpy as np

from .airfogsim_tensor_v2 import NODE_FEATURES


MOTION_FEATURES = ("vx", "vy", "vz", "ax", "ay", "az")
MOTION_SCHEMA_VERSION = "PI-JWM-causal-node-motion-v1"


def derive_causal_node_motion(
    node_state: np.ndarray,
    node_present: np.ndarray,
    *,
    slot_seconds: float,
) -> dict[str, Any]:
    """Derive backward-difference motion without reading a future position."""

    state = np.asarray(node_state)
    present = np.asarray(node_present, dtype=bool)
    if state.ndim != 3 or state.shape[:2] != present.shape:
        raise ValueError("node_state and node_present shapes are incompatible")
    if state.shape[-1] != len(NODE_FEATURES):
        raise ValueError("node_state feature width does not match NODE_FEATURES")
    step_seconds = float(slot_seconds)
    if not np.isfinite(step_seconds) or step_seconds <= 0.0:
        raise ValueError("slot_seconds must be finite and positive")

    motion = np.zeros((*present.shape, len(MOTION_FEATURES)), dtype=np.float32)
    mask = np.zeros_like(motion, dtype=bool)
    position = state[..., :3].astype(np.float64, copy=False)
    velocity_valid = present[1:] & present[:-1]
    velocity = (position[1:] - position[:-1]) / step_seconds
    motion[1:, :, :3] = np.where(velocity_valid[..., None], velocity, 0.0)
    mask[1:, :, :3] = velocity_valid[..., None]

    acceleration_valid = velocity_valid[1:] & velocity_valid[:-1]
    acceleration = (velocity[1:] - velocity[:-1]) / step_seconds
    motion[2:, :, 3:] = np.where(
        acceleration_valid[..., None], acceleration, 0.0
    )
    mask[2:, :, 3:] = acceleration_valid[..., None]

    updated_state = state.astype(np.float32, copy=True)
    speed_index = list(NODE_FEATURES).index("speed")
    acceleration_index = list(NODE_FEATURES).index("acceleration")
    updated_state[..., speed_index] = np.where(
        mask[..., :3].all(axis=-1), np.linalg.norm(motion[..., :3], axis=-1), 0.0
    )
    updated_state[..., acceleration_index] = np.where(
        mask[..., 3:].all(axis=-1), np.linalg.norm(motion[..., 3:], axis=-1), 0.0
    )
    return {
        "schema_version": MOTION_SCHEMA_VERSION,
        "node_state": updated_state,
        "node_motion_state": motion,
        "node_motion_mask": mask,
    }


__all__ = [
    "MOTION_FEATURES",
    "MOTION_SCHEMA_VERSION",
    "derive_causal_node_motion",
]
