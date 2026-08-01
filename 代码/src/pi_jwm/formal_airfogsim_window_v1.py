"""Leak-safe windows for the formal PI-JWM AirFogSim tensor dataset."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .airfogsim_window_dataset_v2 import AirFogSimTensorWindowDataset, _to_tensor
from .formal_airfogsim_dataset_v1 import require_split_access


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

        for key in ("task_dag_state", "task_dag_state_present", "dag_edge_present"):
            sample["history"][key] = _to_tensor(arrays[key][input_start:input_end])
            sample["target"][key] = _to_tensor(arrays[key][label_start:label_end])

        sample["future_action"] = {
            key: _to_tensor(arrays[key][label_start:label_end])
            for key in ("task_action", "task_action_present", "task_action_node_index")
        }
        return sample


__all__ = [
    "FormalAirFogSimWindowDataset",
    "FormalWindowConfig",
    "select_stratified_window_ids",
]
