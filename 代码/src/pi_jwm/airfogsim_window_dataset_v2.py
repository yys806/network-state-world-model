"""Lazy window dataset for the PI-JWM AirFogSim tensor-v2 artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


STATE_KEYS = {
    "node_state": "node_present",
    "physical_edge_state": "physical_edge_present",
    "flow_state": "flow_present",
    "task_state": "task_present",
}
HISTORY_KEYS = (
    "node_state",
    "node_present",
    "physical_edge_state",
    "physical_edge_present",
    "flow_state",
    "flow_present",
    "task_state",
    "task_present",
    "task_lifecycle_index",
    "task_action",
    "task_action_present",
    "task_node_index",
    "task_action_node_index",
    "flow_bearer_mask",
    "flow_bearer_edge_index",
)
TARGET_KEYS = (
    "node_state",
    "node_present",
    "physical_edge_state",
    "physical_edge_present",
    "flow_state",
    "flow_present",
    "flow_completed",
    "task_state",
    "task_present",
    "task_lifecycle_index",
)
STATIC_KEYS = (
    "node_kind_index",
    "physical_edge_endpoint_index",
    "physical_edge_kind_index",
    "flow_endpoint_index",
    "flow_type_index",
    "flow_valid",
    "task_valid",
    "dag_edge_index",
    "dag_edge_valid",
    "agent_node_index",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _read_window_rows(root: Path, split: str | None) -> list[dict[str, Any]]:
    path = root / "window_index.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        if split is not None and str(row.get("split")) != str(split):
            continue
        parsed.append(
            {
                **row,
                "seed": int(row["seed"]),
                "input_start_index": int(row["input_start_index"]),
                "input_end_index": int(row["input_end_index"]),
                "label_start_index": int(row["label_start_index"]),
                "label_end_index": int(row["label_end_index"]),
            }
        )
    return parsed


def _to_tensor(value: np.ndarray) -> torch.Tensor:
    if value.dtype == np.bool_:
        return torch.from_numpy(value.astype(np.bool_, copy=False))
    if np.issubdtype(value.dtype, np.integer):
        return torch.from_numpy(value.astype(np.int64, copy=False))
    return torch.from_numpy(value.astype(np.float32, copy=False))


def _masked_statistics(values: list[np.ndarray], masks: list[np.ndarray]) -> dict[str, Any]:
    if not values:
        return {"mean": [], "scale": [], "count": 0}
    feature_count = values[0].shape[-1]
    total = np.zeros((feature_count,), dtype=np.float64)
    total_sq = np.zeros((feature_count,), dtype=np.float64)
    count = 0
    for value, mask in zip(values, masks):
        flat = np.asarray(value, dtype=np.float64).reshape(-1, feature_count)
        valid = np.asarray(mask, dtype=bool).reshape(-1)
        selected = flat[valid]
        if selected.size == 0:
            continue
        total += selected.sum(axis=0)
        total_sq += np.square(selected).sum(axis=0)
        count += selected.shape[0]
    if count == 0:
        return {"mean": [0.0] * feature_count, "scale": [1.0] * feature_count, "count": 0}
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    scale = np.sqrt(variance)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "count": int(count)}


def fit_training_stats(root: str | Path, *, split: str = "dev_train") -> dict[str, Any]:
    """Fit masked feature statistics using only windows from ``split``."""

    root = Path(root)
    rows = _read_window_rows(root, split)
    cache: dict[int, dict[str, np.ndarray]] = {}
    values: dict[str, list[np.ndarray]] = {key: [] for key in STATE_KEYS}
    masks: dict[str, list[np.ndarray]] = {key: [] for key in STATE_KEYS}
    for row in rows:
        seed = row["seed"]
        if seed not in cache:
            path = root / f"seed_{seed:03d}" / "trajectory_tensors.npz"
            with np.load(path, allow_pickle=False) as loaded:
                cache[seed] = {key: loaded[key] for key in loaded.files}
        arrays = cache[seed]
        start, end = row["input_start_index"], row["input_end_index"]
        for key, mask_key in STATE_KEYS.items():
            values[key].append(arrays[key][start:end])
            masks[key].append(arrays[mask_key][start:end])
    return {
        "schema_version": "PI-JWM-AirFogSim-normalization-v2",
        "source_split": split,
        "sample_count": len(rows),
        "features": {key: _masked_statistics(values[key], masks[key]) for key in STATE_KEYS},
    }


def _normalize(value: np.ndarray, mask: np.ndarray, stat: Mapping[str, Any]) -> np.ndarray:
    mean = np.asarray(stat.get("mean", []), dtype=np.float32)
    scale = np.asarray(stat.get("scale", []), dtype=np.float32)
    if mean.size != value.shape[-1] or scale.size != value.shape[-1]:
        raise ValueError("normalization feature dimension mismatch")
    normalized = (value.astype(np.float32) - mean) / np.maximum(scale, 1e-6)
    normalized = normalized.copy()
    normalized[~mask.astype(bool)] = 0.0
    return normalized


class AirFogSimTensorWindowDataset(Dataset):
    """Load only the seed trajectory needed by each requested window."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str | None = None,
        stats: Mapping[str, Any] | None = None,
        normalize: bool = False,
    ) -> None:
        self.root = Path(root)
        self.contract = _read_json(self.root / "tensor_contract.json")
        self.rows = _read_window_rows(self.root, split)
        self.stats = stats
        self.normalize = bool(normalize)
        if self.normalize and self.stats is None:
            raise ValueError("normalize=True requires training stats")
        self._seed_cache: dict[int, dict[str, np.ndarray]] = {}

    @property
    def loaded_seed_count(self) -> int:
        return len(self._seed_cache)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_seed(self, seed: int) -> dict[str, np.ndarray]:
        if seed not in self._seed_cache:
            path = self.root / f"seed_{seed:03d}" / "trajectory_tensors.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as loaded:
                self._seed_cache[seed] = {key: loaded[key] for key in loaded.files}
        return self._seed_cache[seed]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        arrays = self._load_seed(row["seed"])
        input_start, input_end = row["input_start_index"], row["input_end_index"]
        label_start, label_end = row["label_start_index"], row["label_end_index"]
        history: dict[str, torch.Tensor] = {}
        target: dict[str, torch.Tensor] = {}
        for key in HISTORY_KEYS:
            value = arrays[key][input_start:input_end]
            if self.normalize and key in STATE_KEYS:
                value = _normalize(value, arrays[STATE_KEYS[key]][input_start:input_end], self.stats["features"][key])
            history[key] = _to_tensor(value)
        for key in TARGET_KEYS:
            value = arrays[key][label_start:label_end]
            if self.normalize and key in STATE_KEYS:
                value = _normalize(value, arrays[STATE_KEYS[key]][label_start:label_end], self.stats["features"][key])
            target[key] = _to_tensor(value)
        static = {key: _to_tensor(arrays[key]) for key in STATIC_KEYS if key in arrays}
        return {
            "sample_id": str(row["sample_id"]),
            "seed": int(row["seed"]),
            "split": str(row.get("split", "")),
            "history": history,
            "target": target,
            "static": static,
            "metadata": {
                "decision_time": float(row.get("decision_time", "nan")),
                "label_start_time": float(row.get("label_start_time", "nan")),
            },
        }
