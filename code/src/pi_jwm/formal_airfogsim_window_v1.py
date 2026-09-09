"""Leak-safe windows for the formal PI-JWM AirFogSim tensor dataset."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .airfogsim_window_dataset_v2 import AirFogSimTensorWindowDataset, _to_tensor
from .formal_airfogsim_dataset_v1 import require_split_access
from .formal_rb_targets_v1 import validate_label_window_alignment
from .airfogsim_tensor_v2 import EDGE_FEATURES


@dataclass(frozen=True)
class FormalWindowConfig:
    history_steps: int = 8
    horizon_steps: int = 3
    allow_locked_test: bool = False


def select_stratified_window_ids(
    index_rows: Sequence[Mapping[str, Any]],
    limit: int,
    seed: int,
) -> list[str]:
    """Select deterministic, approximately seed-balanced window IDs."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    grouped: dict[int, list[str]] = defaultdict(list)
    for row in index_rows:
        grouped[int(row["seed"])].append(str(row["sample_id"]))
    generator = random.Random(int(seed))
    for sample_ids in grouped.values():
        sample_ids.sort()
        generator.shuffle(sample_ids)

    selected: list[str] = []
    group_order = sorted(grouped)
    generator.shuffle(group_order)
    offset = 0
    while len(selected) < min(limit, len(index_rows)):
        added = False
        for group in group_order:
            if offset < len(grouped[group]):
                selected.append(grouped[group][offset])
                added = True
                if len(selected) == min(limit, len(index_rows)):
                    break
        if not added:
            break
        offset += 1
    return selected


class FormalAirFogSimWindowDataset(Dataset):
    """Add formal future-action and DAG fields to the existing lazy loader."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        config: FormalWindowConfig = FormalWindowConfig(),
        stats: Mapping[str, Any] | None = None,
        normalize: bool = False,
    ) -> None:
        require_split_access(split, allow_locked_test=config.allow_locked_test)
        self.config = config
        self._base = AirFogSimTensorWindowDataset(
            root,
            split=split,
            stats=stats,
            normalize=normalize,
        )
        contract = self._base.contract
        if int(contract.get("history_steps", -1)) != config.history_steps:
            raise ValueError("history_steps does not match the formal tensor contract")
        if int(contract.get("horizon_steps", -1)) != config.horizon_steps:
            raise ValueError("horizon_steps does not match the formal tensor contract")

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self._base.rows

    @property
    def contract(self) -> dict[str, Any]:
        return self._base.contract

    @property
    def loaded_seed_count(self) -> int:
        return self._base.loaded_seed_count

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._base[index]
        row = self._base.rows[index]
        arrays = self._base._load_seed(row["seed"])
        input_start, input_end = row["input_start_index"], row["input_end_index"]
        label_start, label_end = row["label_start_index"], row["label_end_index"]
        if input_end - input_start != self.config.history_steps:
            raise ValueError("window history_steps does not match the configured length")
        if label_end - label_start != self.config.horizon_steps:
            raise ValueError("window horizon_steps does not match the configured length")
        validate_label_window_alignment(
            history_times=arrays["time"][input_start:input_end],
            label_times=arrays["time"][label_start:label_end],
        )

        for key in ("task_dag_state", "task_dag_state_present", "dag_edge_present"):
            sample["history"][key] = _to_tensor(arrays[key][input_start:input_end])
            sample["target"][key] = _to_tensor(arrays[key][label_start:label_end])

        motion_keys = ("node_motion_state", "node_motion_mask")
        if any(key in arrays for key in motion_keys) and not all(
            key in arrays for key in motion_keys
        ):
            raise ValueError("causal node-motion arrays are incomplete")
        if all(key in arrays for key in motion_keys):
            # Motion is an input-side causal feature.  Future motion remains a
            # label-side fact and is deliberately not exposed to the model.
            sample["history"]["node_motion_state"] = _to_tensor(
                arrays["node_motion_state"][input_start:input_end]
            )
            sample["history"]["node_motion_mask"] = _to_tensor(
                arrays["node_motion_mask"][input_start:input_end]
            )
            sample["metadata"]["node_motion_contract"] = self.contract.get(
                "node_motion_contract"
            )

        sample["future_action"] = {
            key: _to_tensor(arrays[key][label_start:label_end])
            for key in (
                "task_action",
                "task_action_present",
                "task_action_node_index",
                "task_action_source_node_index",
            )
        }
        edge_present = arrays["physical_edge_present"].astype(bool)
        edge_feature_mask = arrays.get("physical_edge_feature_mask")
        if edge_feature_mask is None:
            edge_feature_mask = np.broadcast_to(
                edge_present[..., None], arrays["physical_edge_state"].shape
            )
        edge_feature_mask = edge_feature_mask.astype(bool)
        activity_index = EDGE_FEATURES.index("active_task_count")
        rate_index = EDGE_FEATURES.index("rate_sum")
        rb_index = EDGE_FEATURES.index("allocated_rb_count")

        def aggregate_slice(start: int, end: int) -> dict[str, np.ndarray]:
            state = arrays["physical_edge_state"][start:end]
            present = edge_present[start:end]
            feature_mask = edge_feature_mask[start:end]
            activity_mask = present & feature_mask[..., activity_index]
            rate_mask = present & feature_mask[..., rate_index]
            rb_mask = present & feature_mask[..., rb_index]
            return {
                "aggregate_link_activity": (
                    (state[..., activity_index] > 0) & activity_mask
                ),
                "aggregate_link_activity_mask": activity_mask,
                "aggregate_link_rate_sum": np.where(
                    rate_mask, state[..., rate_index], 0.0
                ).astype(np.float32),
                "aggregate_link_rate_sum_mask": rate_mask,
                "aggregate_rb_occupancy": np.where(
                    rb_mask, state[..., rb_index], 0.0
                ).astype(np.float32),
                "aggregate_rb_occupancy_mask": rb_mask,
            }

        for key, value in aggregate_slice(input_start, input_end).items():
            sample["history"][key] = _to_tensor(value)
        for key, value in aggregate_slice(label_start, label_end).items():
            sample["target"][key] = _to_tensor(value)
        node_present = arrays["node_present"][input_end - 1].astype(bool)
        edge_present = arrays["physical_edge_present"][input_end - 1].astype(bool)
        flow_present = arrays["flow_present"][input_end - 1].astype(bool)
        task_present = arrays["task_present"][input_end - 1].astype(bool)
        deterministic_node_mask = np.zeros((node_present.shape[0], 7), dtype=bool)
        deterministic_edge_mask = np.zeros((edge_present.shape[0], 5), dtype=bool)
        deterministic_edge_mask[:, 4] = edge_present
        deterministic_flow_mask = np.zeros((flow_present.shape[0], 5), dtype=bool)
        deterministic_flow_mask[:] = flow_present[:, None]
        deterministic_task_mask = np.zeros((task_present.shape[0], 8), dtype=bool)
        deterministic_task_mask[:] = task_present[:, None]
        sample["static"]["deterministic_node_mask"] = _to_tensor(deterministic_node_mask)
        sample["static"]["deterministic_edge_mask"] = _to_tensor(deterministic_edge_mask)
        sample["static"]["deterministic_flow_mask"] = _to_tensor(deterministic_flow_mask)
        sample["static"]["deterministic_task_mask"] = _to_tensor(deterministic_task_mask)
        sample["metadata"]["rule_layer_contract"] = {
            "source_endpoint_mapping": "task_action_source_node_index_from_action_records",
            "service_outcome_source": "learned_service_heads",
            "deterministic_target_masks": True,
            "units": "physical_inside_rule_layer_normalized_at_model_boundary",
        }
        per_rb_keys = {
            "link_activity": "link_activity_by_rb",
            "link_activity_mask": "link_activity_mask_by_rb",
            "link_rate_by_rb": "link_rate_by_rb",
            "link_rate_by_rb_mask": "link_rate_by_rb_mask",
        }
        for key, sample_key in per_rb_keys.items():
            if key not in arrays:
                continue
            sample["history"][sample_key] = _to_tensor(arrays[key][input_start:input_end])
            sample["target"][sample_key] = _to_tensor(arrays[key][label_start:label_end])
        return sample


__all__ = [
    "FormalAirFogSimWindowDataset",
    "FormalWindowConfig",
    "select_stratified_window_ids",
]
