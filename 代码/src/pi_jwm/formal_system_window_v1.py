"""Aligned real system-outcome sidecars for formal PI-JWM windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from torch.utils.data import Dataset

from .airfogsim_window_dataset_v2 import _to_tensor
from .formal_airfogsim_window_v1 import FormalAirFogSimWindowDataset, FormalWindowConfig


SYSTEM_TIME_KEYS = (
    "task_completion_event",
    "completed_task_delay",
    "completed_task_delay_valid",
    "uav_energy_delta",
    "uav_energy_valid",
    "source_service_delta",
    "delivered_data_total",
)
SYSTEM_STATIC_KEYS = ("source_population_valid",)
TASK_SYSTEM_KEYS = (
    "task_completion_event",
    "completed_task_delay",
    "completed_task_delay_valid",
)
NODE_SYSTEM_KEYS = (
    "uav_energy_delta",
    "uav_energy_valid",
    "source_service_delta",
)


def _pad_entity_axis(value: np.ndarray, target_count: int, name: str) -> np.ndarray:
    current_count = int(value.shape[-1])
    if current_count > target_count:
        raise ValueError(f"system array {name} exceeds the formal tensor contract")
    if current_count == target_count:
        return value
    padding = [(0, 0)] * value.ndim
    padding[-1] = (0, target_count - current_count)
    return np.pad(value, padding, mode="constant", constant_values=0)


class FormalSystemWindowDataset(Dataset):
    """Attach directly measured/simulated outcome labels to formal graph windows."""

    def __init__(
        self,
        tensor_root: str | Path,
        *,
        system_root: str | Path,
        split: str,
        config: FormalWindowConfig = FormalWindowConfig(),
        stats: Mapping[str, Any] | None = None,
        normalize: bool = False,
    ) -> None:
        self.system_root = Path(system_root)
        self._base = FormalAirFogSimWindowDataset(
            tensor_root,
            split=split,
            config=config,
            stats=stats,
            normalize=normalize,
        )
        self._system_cache: dict[int, dict[str, np.ndarray]] = {}

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self._base.rows

    @property
    def contract(self) -> dict[str, Any]:
        return self._base.contract

    @property
    def loaded_system_seed_count(self) -> int:
        return len(self._system_cache)

    def __len__(self) -> int:
        return len(self._base)

    def _load_system_seed(self, row: Mapping[str, Any]) -> dict[str, np.ndarray]:
        seed = int(row["seed"])
        if seed in self._system_cache:
            return self._system_cache[seed]
        seed_dir = self.system_root / f"seed_{seed:03d}"
        report_path = seed_dir / "system_target_report.json"
        npz_path = seed_dir / "system_targets.npz"
        if not report_path.exists() or not npz_path.exists():
            raise FileNotFoundError(f"missing system sidecar for seed {seed:03d}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("seed", -1)) != seed or str(report.get("split")) != str(row["split"]):
            raise ValueError(f"system sidecar identity mismatch for seed {seed:03d}")
        with np.load(npz_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        required = {"time", *SYSTEM_TIME_KEYS, *SYSTEM_STATIC_KEYS}
        missing = sorted(required.difference(arrays))
        if missing:
            raise ValueError(f"system sidecar is missing arrays: {missing}")
        tensor_arrays = self._base._base._load_seed(seed)
        tensor_time = np.asarray(tensor_arrays["time"], dtype=np.float64)
        system_time = np.asarray(arrays["time"], dtype=np.float64)
        if tensor_time.shape != system_time.shape or not np.allclose(
            tensor_time, system_time, rtol=0.0, atol=1e-6
        ):
            raise ValueError(f"system sidecar time grid mismatch for seed {seed:03d}")
        for key in SYSTEM_TIME_KEYS:
            if arrays[key].shape[0] != len(system_time):
                raise ValueError(f"system array {key} does not match the time grid")
        max_tasks = int(self.contract["max_tasks"])
        max_nodes = int(self.contract["max_nodes"])
        for key in TASK_SYSTEM_KEYS:
            arrays[key] = _pad_entity_axis(arrays[key], max_tasks, key)
        for key in NODE_SYSTEM_KEYS:
            arrays[key] = _pad_entity_axis(arrays[key], max_nodes, key)
        arrays["source_population_valid"] = _pad_entity_axis(
            arrays["source_population_valid"], max_nodes, "source_population_valid"
        )
        self._system_cache[seed] = arrays
        return arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._base[index]
        row = self.rows[index]
        arrays = self._load_system_seed(row)
        input_start, input_end = int(row["input_start_index"]), int(row["input_end_index"])
        label_start, label_end = int(row["label_start_index"]), int(row["label_end_index"])
        sample["system_history"] = {
            key: _to_tensor(arrays[key][input_start:input_end]) for key in SYSTEM_TIME_KEYS
        }
        sample["system_target"] = {
            key: _to_tensor(arrays[key][label_start:label_end]) for key in SYSTEM_TIME_KEYS
        }
        sample["system_static"] = {
            key: _to_tensor(arrays[key]) for key in SYSTEM_STATIC_KEYS
        }
        return sample


__all__ = [
    "FormalSystemWindowDataset",
    "SYSTEM_STATIC_KEYS",
    "SYSTEM_TIME_KEYS",
    "NODE_SYSTEM_KEYS",
    "TASK_SYSTEM_KEYS",
]
