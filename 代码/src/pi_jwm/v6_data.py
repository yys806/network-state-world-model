"""Dataset utilities for PI-JWM v6 dual-graph rollout."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from pi_jwm.v6_dual_graph import V6DualGraphBatch


PHYSICAL_EDGE_FEATURES = (
    "dx",
    "dy",
    "dz",
    "distance_3d",
    "abs_speed_delta",
    "src_speed",
    "dst_speed",
    "abs_dz",
)


def load_world_model_arrays(dataset_dir: Path) -> dict[str, np.ndarray]:
    dataset_dir = Path(dataset_dir)
    with np.load(dataset_dir / "world_model_dataset_v0_samples.npz", allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def split_by_seed(
    sample_seed: np.ndarray,
    train_seeds: Iterable[int] = tuple(range(0, 8)),
    val_seed: int = 8,
    test_seed: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_seed = np.asarray(sample_seed)
    train_idx = np.where(np.isin(sample_seed, list(train_seeds)))[0]
    val_idx = np.where(sample_seed == val_seed)[0]
    test_idx = np.where(sample_seed == test_seed)[0]
    return train_idx, val_idx, test_idx


def build_physical_edge_history(
    x_node: np.ndarray | Tensor,
    edge_src_idx: np.ndarray | Tensor,
    edge_dst_idx: np.ndarray | Tensor,
    valid_edge_node: np.ndarray | Tensor,
) -> Tensor:
    x_node_t = torch.as_tensor(x_node, dtype=torch.float32)
    src_idx = torch.as_tensor(edge_src_idx, dtype=torch.long).clamp(min=0)
    dst_idx = torch.as_tensor(edge_dst_idx, dtype=torch.long).clamp(min=0)
    valid = torch.as_tensor(valid_edge_node, dtype=torch.float32).reshape(1, 1, -1, 1)

    src = x_node_t[:, :, src_idx, :]
    dst = x_node_t[:, :, dst_idx, :]
    delta_xyz = dst[..., :3] - src[..., :3]
    distance = torch.linalg.norm(delta_xyz, dim=-1, keepdim=True)
    src_speed = src[..., 3:4]
    dst_speed = dst[..., 3:4]
    speed_delta = torch.abs(dst_speed - src_speed)
    abs_dz = torch.abs(delta_xyz[..., 2:3])
    physical = torch.cat(
        [delta_xyz, distance, speed_delta, src_speed, dst_speed, abs_dz],
        dim=-1,
    )
    return physical * valid


def fit_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(values: np.ndarray, stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = stats
    return ((values - mean) / std).astype(np.float32)


def inverse_normalize(values: np.ndarray, stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = stats
    return (values * std + mean).astype(np.float32)


def transform_link_rate(values: np.ndarray, rate_target_transform: str = "raw") -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if rate_target_transform == "raw":
        return values
    if rate_target_transform == "log1p_raw":
        return np.log1p(np.clip(values, a_min=0.0, a_max=None)).astype(np.float32)
    if rate_target_transform == "residual_last_rate":
        return values
    raise ValueError("rate_target_transform must be one of: raw, log1p_raw, residual_last_rate")


def inverse_transform_link_rate(
    values: np.ndarray,
    rate_target_transform: str = "raw",
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if clip_min is not None or clip_max is not None:
        values = np.clip(values, a_min=clip_min, a_max=clip_max)
    if rate_target_transform == "raw":
        return values
    if rate_target_transform == "log1p_raw":
        return np.expm1(values).astype(np.float32)
    if rate_target_transform == "residual_last_rate":
        return values
    raise ValueError("rate_target_transform must be one of: raw, log1p_raw, residual_last_rate")


def find_link_rate_feature_index(arrays: dict[str, np.ndarray]) -> int:
    if "link_features" not in arrays:
        return 1
    names = [str(name) for name in arrays["link_features"]]
    if "rate_sum" not in names:
        raise ValueError("link_features must include rate_sum for residual_last_rate targets")
    return names.index("rate_sum")


def build_last_rate_baseline(arrays: dict[str, np.ndarray], indices: Iterable[int]) -> np.ndarray:
    indices = np.asarray(list(indices), dtype=np.int64)
    rate_idx = find_link_rate_feature_index(arrays)
    return arrays["x_link"][indices, -1, :, rate_idx].astype(np.float32)


def build_link_rate_target(
    arrays: dict[str, np.ndarray],
    indices: Iterable[int],
    rate_target_transform: str = "raw",
) -> np.ndarray:
    indices = np.asarray(list(indices), dtype=np.int64)
    raw_rate = arrays["y_link_rate"][indices][..., None]
    if rate_target_transform == "residual_last_rate":
        baseline = build_last_rate_baseline(arrays, indices)[:, None, :, None]
        return (raw_rate - baseline).astype(np.float32)
    return transform_link_rate(raw_rate, rate_target_transform)


def make_normalization_stats(
    arrays: dict[str, np.ndarray],
    train_idx: Iterable[int],
    rate_target_transform: str = "raw",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    train_idx = np.asarray(list(train_idx), dtype=np.int64)
    physical_edges = build_physical_edge_history(
        arrays["x_node"][train_idx],
        arrays["edge_src_idx"],
        arrays["edge_dst_idx"],
        arrays["valid_edge_node"],
    ).numpy()
    return {
        "rate_target_transform": rate_target_transform,
        "x_node": fit_stats(arrays["x_node"][train_idx]),
        "x_link": fit_stats(arrays["x_link"][train_idx]),
        "x_physical_edge": fit_stats(physical_edges),
        "x_task": fit_stats(arrays["x_task"][train_idx]),
        "edge_a_hist": fit_stats(arrays["edge_a_hist"][train_idx]),
        "edge_a_future": fit_stats(arrays["edge_a_future"][train_idx]),
        "y_node": fit_stats(arrays["y_node"][train_idx]),
        "y_task": fit_stats(arrays["y_task"][train_idx]),
        "y_link_rate": fit_stats(build_link_rate_target(arrays, train_idx, rate_target_transform)),
    }


class V6WorldModelDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        indices: Iterable[int],
        stats: dict[str, tuple[np.ndarray, np.ndarray]],
        rate_target_transform: str = "raw",
        future_action_mode: str = "full",
    ):
        if future_action_mode not in {"full", "first_step_only", "none"}:
            raise ValueError("future_action_mode must be one of: full, first_step_only, none")
        self.arrays = arrays
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.stats = stats
        self.rate_target_transform = rate_target_transform
        self.future_action_mode = future_action_mode
        self.link_rate_baseline = build_last_rate_baseline(arrays, self.indices)
        self.physical_edge_history = build_physical_edge_history(
            arrays["x_node"][self.indices],
            arrays["edge_src_idx"],
            arrays["edge_dst_idx"],
            arrays["valid_edge_node"],
        ).numpy()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[V6DualGraphBatch, dict[str, Tensor]]:
        source_idx = self.indices[item]
        local_idx = item

        future_actions = self.arrays["edge_a_future"][source_idx]
        if self.future_action_mode != "full":
            future_actions = future_actions.copy()
            start = 1 if self.future_action_mode == "first_step_only" else 0
            future_actions[start:] = 0.0

        batch = V6DualGraphBatch(
            node_history=torch.from_numpy(normalize(self.arrays["x_node"][source_idx], self.stats["x_node"])[0]),
            physical_edge_history=torch.from_numpy(
                normalize(self.physical_edge_history[local_idx], self.stats["x_physical_edge"])[0]
            ),
            info_edge_history=torch.from_numpy(normalize(self.arrays["x_link"][source_idx], self.stats["x_link"])[0]),
            action_history=torch.from_numpy(
                normalize(self.arrays["edge_a_hist"][source_idx], self.stats["edge_a_hist"])[0]
            ),
            future_actions=torch.from_numpy(
                normalize(future_actions, self.stats["edge_a_future"])[0]
            ),
            task_history=torch.from_numpy(normalize(self.arrays["x_task"][source_idx], self.stats["x_task"])[0]),
            link_rate_baseline=torch.from_numpy(
                np.repeat(
                    self.link_rate_baseline[local_idx][None, :, None],
                    self.arrays["y_link_rate"].shape[1],
                    axis=0,
                )
            ),
        )
        link_rate_target = build_link_rate_target(
            self.arrays,
            [source_idx],
            self.rate_target_transform,
        )[0]
        target = {
            "node": torch.from_numpy(normalize(self.arrays["y_node"][source_idx], self.stats["y_node"])[0]),
            "link_activity": torch.from_numpy(self.arrays["y_link_active"][source_idx, ..., None].astype(np.float32)),
            "link_rate": torch.from_numpy(normalize(link_rate_target, self.stats["y_link_rate"])[0]),
            "link_rate_raw": torch.from_numpy(self.arrays["y_link_rate"][source_idx, ..., None].astype(np.float32)),
            "task": torch.from_numpy(normalize(self.arrays["y_task"][source_idx], self.stats["y_task"])[0]),
        }
        if "y_link_rate_teacher" in self.arrays and "y_link_rate_teacher_mask" in self.arrays:
            target["link_rate_teacher"] = torch.from_numpy(
                normalize(
                    transform_link_rate(
                        self.arrays["y_link_rate_teacher"][source_idx, ..., None],
                        self.rate_target_transform,
                    ),
                    self.stats["y_link_rate"],
                )[0]
            )
            target["link_rate_teacher_mask"] = torch.from_numpy(
                self.arrays["y_link_rate_teacher_mask"][source_idx, ..., None].astype(np.float32)
            )
        return batch, target


def collate_v6_world_model_batch(
    items: list[tuple[V6DualGraphBatch, dict[str, Tensor]]],
) -> tuple[V6DualGraphBatch, dict[str, Tensor]]:
    batches, targets = zip(*items)
    batch = V6DualGraphBatch(
        node_history=torch.stack([item.node_history for item in batches]),
        physical_edge_history=torch.stack([item.physical_edge_history for item in batches]),
        info_edge_history=torch.stack([item.info_edge_history for item in batches]),
        action_history=torch.stack([item.action_history for item in batches]),
        future_actions=torch.stack([item.future_actions for item in batches]),
        task_history=torch.stack([item.task_history for item in batches]),
        link_rate_baseline=(
            torch.stack([item.link_rate_baseline for item in batches])
            if batches[0].link_rate_baseline is not None
            else None
        ),
    )
    target = {
        "node": torch.stack([item["node"] for item in targets]),
        "link_activity": torch.stack([item["link_activity"] for item in targets]),
        "link_rate": torch.stack([item["link_rate"] for item in targets]),
        "link_rate_raw": torch.stack([item["link_rate_raw"] for item in targets]),
        "task": torch.stack([item["task"] for item in targets]),
    }
    if "link_rate_teacher" in targets[0] and "link_rate_teacher_mask" in targets[0]:
        target["link_rate_teacher"] = torch.stack([item["link_rate_teacher"] for item in targets])
        target["link_rate_teacher_mask"] = torch.stack([item["link_rate_teacher_mask"] for item in targets])
    return batch, target
