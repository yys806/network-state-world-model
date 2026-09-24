"""Bounded-memory consumption of the accepted Formal Dataset v1 shards.

Only the requested trajectory shards are decompressed for a batch.  The
manifest/index stays resident; sample, tensor, graph and target payloads do
not.  This module does not generate data or run formal training.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import load_typed_dual_graph_batch
from pi_jwm.step5_1a_motion_csi_target_contract_v1 import load_future_target_tensor_batch
from pi_jwm.step5_2_training_loop_v1 import DevelopmentBundle, Step52Trainer, Step52TrainingConfig, ROOT, _jsonable, _sha, aggregate_validation_horizon_rows
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface, sha256_path


# The frozen Formal v1 collector uses this one bidirectional wired edge in all
# 60 Raw trajectories. A separate audit checks the exact Raw values against
# this source constant; this lets the Training Bundle omit Raw without changing
# any current state or package content.
FORMAL_V1_WIRED_EDGES = [{"u": "RSU_0", "v": "cloudServer_4", "capacity_mbps": 100.0, "bidirectional": True}]


def _one(value: Mapping[str, Any], index: int, count: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, np.ndarray) and item.ndim and item.shape[0] == count:
            result[key] = item[index:index + 1].copy()
        elif key in {"sample_metadata", "sample_static"} and isinstance(item, list):
            result[key] = [item[index]]
        elif key == "blocks" and isinstance(item, Mapping):
            result[key] = {name: _one(block, index, count) for name, block in item.items()}
        elif key == "contract" and isinstance(item, Mapping):
            result[key] = {
                name: [rows[index]] if name in {"motion_identity", "comm_identity"} and isinstance(rows, list) and len(rows) == count else rows
                for name, rows in item.items()
            }
        else:
            result[key] = item
    return result


class FullFormalShardDataset:
    def __init__(self, interface: FormalTrainingInterface):
        self.interface = interface
        self.paths = interface.package_paths
        if sha256_path(self.paths["normalization"]) != interface.manifest["hashes"]["normalization"]:
            raise ValueError("formal train-only normalization package hash mismatch")
        self.index = interface.samples
        self.shards = {row["trajectory_id"]: row for row in json.loads((self.paths["tensor"] / "index.json").read_text(encoding="utf-8"))["shards"]}
        split = json.loads((interface.manifest_path.parent / interface.manifest["provenance"]["split_manifest"]).read_text(encoding="utf-8"))
        self.train_ids = set(split["dev_train"])
        self.validation_ids = set(split["dev_validation"])
        if len(self.index) != 5520 or len(interface.train_indices) != 4416 or len(interface.validation_indices) != 1104:
            raise ValueError("Formal Dataset v1 requires exactly 4416/1104 indexed windows")
        if len(self.shards) != 60 or len(self.train_ids) != 48 or len(self.validation_ids) != 12 or self.train_ids & self.validation_ids:
            raise ValueError("formal trajectory split differs from frozen 48/12 contract")
        seen: set[str] = set()
        for row in self.index:
            metadata = row["metadata"]
            sample_id = metadata["sample_id"]
            trajectory_id = metadata["trajectory_id"]
            expected = "dev_train" if trajectory_id in self.train_ids else "dev_validation" if trajectory_id in self.validation_ids else None
            if sample_id in seen or expected != metadata["split"] or row["shard"] != f"{trajectory_id}.json.gz" or not 0 <= row["shard_index"] < 92:
                raise ValueError(f"invalid or duplicate formal sample index: {sample_id}")
            seen.add(sample_id)
        if len(seen) != 5520:
            raise ValueError("formal sample index is incomplete")
        self.identity = {
            "dataset_id": interface.manifest["dataset_id"],
            "dataset_manifest_hash": interface.dataset_manifest_hash,
            "formal_package_hashes": dict(interface.manifest["hashes"]),
            "train_sample_count": 4416,
            "validation_sample_count": 1104,
            "sample_index_sha256": _sha(self.paths["samples"] / "index.json"),
        }
        self.peak_loaded_shards = 0
        self._verified_shards: set[str] = set()
        self._cached_trajectory: str | None = None
        self._cached_shard: tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]] | None = None

    def encoder_normalization_stats(self) -> dict[str, Any]:
        """Use frozen train statistics and stream only train physical edges."""
        stats = json.loads((self.paths["normalization"] / "stats.json").read_text(encoding="utf-8"))
        features = {
            **stats["base_step3_2"]["features"],
            **stats["extension_step4_2a"]["features"],
            **stats["flow_step4_2c"]["features"],
        }
        names = (
            "physical_relation.delta_x_m", "physical_relation.delta_y_m", "physical_relation.delta_z_m",
            "physical_relation.distance_m", "physical_relation.relative_speed_mps",
            "physical_relation.relative_acceleration_mps2",
        )
        units = ("m", "m", "m", "m", "m/s", "m/s^2")
        source = ("delta_x", "delta_y", "delta_z", "distance", "relative_speed", "relative_acceleration")
        totals = np.zeros(6, dtype=np.float64)
        squares = np.zeros(6, dtype=np.float64)
        counts = np.zeros(6, dtype=np.int64)
        for trajectory_id in sorted(self.train_ids):
            graph_path = self.paths["graph"] / f"{trajectory_id}.npz"
            if _sha(graph_path) != self.shards[trajectory_id]["files"]["graph"]["sha256"]:
                raise ValueError(f"formal train graph hash mismatch: {trajectory_id}")
            graph = load_typed_dual_graph_batch(graph_path)
            block = graph["blocks"]["physical_relations"]
            values = np.asarray(block["features"], dtype=np.float64)
            active = np.asarray(block["presence"], dtype=bool) & np.asarray(block["validity"], dtype=bool)
            mask = np.asarray(block["feature_mask"], dtype=bool) & active[..., None] & np.isfinite(values)
            for feature_index in range(6):
                selected = values[..., feature_index][mask[..., feature_index]]
                totals[feature_index] += selected.sum()
                squares[feature_index] += np.square(selected).sum()
                counts[feature_index] += selected.size
            del graph
        for index, name in enumerate(names):
            if counts[index] == 0:
                raise ValueError(f"no formal train physical-relation values for {name}")
            mean = float(totals[index] / counts[index])
            variance = max(float(squares[index] / counts[index] - mean * mean), 0.0)
            std = float(np.sqrt(variance)) if variance > 1e-12 else 1.0
            features[name] = {
                "count": int(counts[index]), "mean": mean, "std": std,
                "source_field": f"physical_relations.features[{index}]/{source[index]}",
                "unit": units[index], "mask_policy": "dev_train only AND presence AND validity AND feature_mask AND finite",
                "zero_variance_handling": "population_std" if variance > 1e-12 else "scale=1.0",
                "preprocessing_policy": "z_score_on_valid_train_values",
            }
        return {"schema_version": "PI-JWM-Step-5.5-Formal-Encoder-Normalization-v1", "source_split": "dev_train", "fit_trajectory_count": 48, "features": features}

    def _load_shard(self, trajectory_id: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
        if trajectory_id == self._cached_trajectory and self._cached_shard is not None:
            return self._cached_shard
        row = self.shards[trajectory_id]
        name = f"{trajectory_id}.json.gz"
        if trajectory_id not in self._verified_shards:
            for family in ("samples", "tensor", "graph", "target"):
                entry = row["files"][family]
                if _sha(self.paths[family] / entry["path"]) != entry["sha256"]:
                    raise ValueError(f"formal shard hash mismatch: {trajectory_id}/{family}")
            self._verified_shards.add(trajectory_id)
        with gzip.open(self.paths["samples"] / name, "rt", encoding="utf-8") as handle:
            samples = json.load(handle)
        tensor = load_flow_tensor_batch(self.paths["tensor"] / f"{trajectory_id}.npz")
        graph = load_typed_dual_graph_batch(self.paths["graph"] / f"{trajectory_id}.npz")
        target = load_future_target_tensor_batch(self.paths["target"] / f"{trajectory_id}.npz")
        count = row["sample_count"]
        if count != 92 or len(samples) != count or len(tensor["sample_metadata"]) != count:
            raise ValueError(f"shard count mismatch: {trajectory_id}")
        if [s["metadata"]["sample_id"] for s in samples] != [m["sample_id"] for m in tensor["sample_metadata"]]:
            raise ValueError(f"sample/tensor identity mismatch: {trajectory_id}")
        for block in graph["blocks"].values():
            if any(isinstance(v, np.ndarray) and v.ndim and v.shape[0] != count for v in block.values()):
                raise ValueError(f"graph identity/count mismatch: {trajectory_id}")
        if any(isinstance(v, np.ndarray) and v.ndim and v.shape[0] != count for v in target.values()):
            raise ValueError(f"target identity/count mismatch: {trajectory_id}")
        self._cached_trajectory = trajectory_id
        self._cached_shard = (samples, tensor, graph, target)
        return self._cached_shard

    def load_batch(self, indices: Sequence[int]) -> DevelopmentBundle:
        if not indices:
            raise ValueError("empty formal batch")
        from build_step5_1d_unified_model_chain_v1 import build_state
        from build_step5_5_formal_dataset_v1 import merge_array_batches, merge_graph_batches, merge_target_batches, synchronize_tensor_contract
        grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for position, global_index in enumerate(indices):
            row = self.index[global_index]
            grouped[row["metadata"]["trajectory_id"]].append((position, row["shard_index"]))
        self.peak_loaded_shards = max(self.peak_loaded_shards, len(grouped))
        ordered: list[Any] = [None] * len(indices)
        for trajectory_id, positions in grouped.items():
            samples, tensor, graph, target = self._load_shard(trajectory_id)
            for position, local_index in positions:
                sample = samples[local_index]
                if sample["metadata"]["sample_id"] != self.index[indices[position]]["metadata"]["sample_id"]:
                    raise ValueError("formal sample/shard identity mismatch")
                ordered[position] = (sample, _one(tensor, local_index, 92), _one(graph, local_index, 92), _one(target, local_index, 92))
            del samples, tensor, graph, target
        batch_samples = [row[0] for row in ordered]
        batch_tensor = synchronize_tensor_contract(merge_array_batches([row[1] for row in ordered]))
        batch_graph = merge_graph_batches([row[2] for row in ordered])
        batch_target = merge_target_batches([row[3] for row in ordered])
        metadata = batch_tensor["sample_metadata"]
        if [s["metadata"]["sample_id"] for s in batch_samples] != [m["sample_id"] for m in metadata]:
            raise ValueError("merged formal batch identity mismatch")
        tensor_for_state = dict(batch_tensor)
        tensor_for_state["sample_metadata"] = [
            {**m, "source_path": str((ROOT / m["source_path"]).resolve()) if not Path(m["source_path"]).is_absolute() else m["source_path"]}
            for m in metadata
        ]
        states, graphs = [], []
        for index in range(len(batch_samples)):
            state, graph_state = build_state(tensor_for_state, batch_graph, index, wired_edges=FORMAL_V1_WIRED_EDGES)
            states.append(state)
            graphs.append(graph_state)
        target_tensors = {
            key: torch.from_numpy(value.copy()).bool() if value.dtype == np.bool_ else torch.from_numpy(value.copy()).float()
            for key, value in batch_target.items() if key.startswith("target_") and isinstance(value, np.ndarray)
        }
        target_normalization = batch_target["contract"]["normalization_parameters"]
        local_train = [i for i, sample in enumerate(batch_samples) if sample["metadata"]["split"] == "dev_train"]
        local_validation = [i for i, sample in enumerate(batch_samples) if sample["metadata"]["split"] == "dev_validation"]
        return DevelopmentBundle(batch_samples, [{} for _ in batch_samples], batch_tensor, batch_graph, target_tensors, target_normalization, states, graphs, local_train, local_validation, self.identity)

    def audit_all_shards(self) -> dict[str, Any]:
        """Walk every real payload shard and compare it to the frozen index."""
        expected = defaultdict(dict)
        for row in self.index:
            expected[row["metadata"]["trajectory_id"]][row["shard_index"]] = row["metadata"]["sample_id"]
        seen: set[str] = set()
        split_counts = {"dev_train": 0, "dev_validation": 0}
        for trajectory_id in self.shards:
            samples, _, _, _ = self._load_shard(trajectory_id)
            if len(expected[trajectory_id]) != len(samples):
                raise ValueError(f"shard/index length mismatch: {trajectory_id}")
            for local_index, sample in enumerate(samples):
                sample_id = sample["metadata"]["sample_id"]
                if expected[trajectory_id].get(local_index) != sample_id or sample_id in seen:
                    raise ValueError(f"shard/index identity missing or duplicated: {sample_id}")
                seen.add(sample_id)
                split_counts[sample["metadata"]["split"]] += 1
        if len(seen) != 5520 or split_counts != {"dev_train": 4416, "dev_validation": 1104}:
            raise ValueError("full formal shard traversal did not match 4416/1104")
        return {"shards": 60, "unique_windows": len(seen), "split_counts": split_counts, "all_payload_hashes_verified": len(self._verified_shards) == 60, "bounded_cached_trajectory_shards": 1}


class FormalTrajectorySampler:
    """Deterministic, trajectory-aware epoch order for Formal Dataset training.

    Each epoch visits every frozen train window exactly once. Trajectories are
    shuffled first and windows within each trajectory are shuffled second, so
    shard locality is bounded while the original trajectory order is not reused.
    """

    def __init__(self, index: Sequence[Mapping[str, Any]], train_indices: Sequence[int], *, seed: int):
        self.seed = int(seed)
        self.train_indices = tuple(int(i) for i in train_indices)
        groups: dict[str, list[int]] = defaultdict(list)
        for global_index in self.train_indices:
            metadata = index[global_index]["metadata"]
            if metadata.get("split") != "dev_train":
                raise ValueError("formal sampler received a non-train window")
            groups[str(metadata["trajectory_id"])].append(global_index)
        if set(self.train_indices) != {i for values in groups.values() for i in values} or len(self.train_indices) != 4416:
            raise ValueError("formal sampler must cover exactly 4416 train windows")
        if len(groups) != 48:
            raise ValueError("formal sampler must cover exactly 48 train trajectories")
        self._groups = {trajectory: tuple(values) for trajectory, values in groups.items()}

    def order_for_epoch(self, epoch: int) -> tuple[int, ...]:
        rng = random.Random(self.seed + int(epoch))
        trajectories = list(self._groups)
        rng.shuffle(trajectories)
        order: list[int] = []
        for trajectory in trajectories:
            windows = list(self._groups[trajectory])
            rng.shuffle(windows)
            order.extend(windows)
        if len(order) != len(self.train_indices) or len(set(order)) != len(order):
            raise AssertionError("formal sampler epoch order is incomplete or duplicated")
        return tuple(order)

    def batch_for_global_step(self, global_step: int, batch_size: int) -> list[int]:
        if global_step < 0 or batch_size <= 0:
            raise ValueError("global_step and batch_size must be positive")
        steps_per_epoch = (len(self.train_indices) + batch_size - 1) // batch_size
        epoch, step = divmod(int(global_step), steps_per_epoch)
        order = self.order_for_epoch(epoch)
        start = step * batch_size
        return list(order[start:min(start + batch_size, len(order))])

    def state_dict(self) -> dict[str, Any]:
        return {"schema_version": "PI-JWM-FormalTrajectorySampler-v1", "seed": self.seed, "train_windows": len(self.train_indices), "train_trajectories": len(self._groups)}


class FullFormalTrainer(Step52Trainer):
    """Existing training loop with trajectory-aware sampling and transient batches."""

    @classmethod
    def from_interface(cls, interface: FormalTrainingInterface, config: Step52TrainingConfig) -> "FullFormalTrainer":
        shards = FullFormalShardDataset(interface)
        representative = shards.load_batch([interface.train_indices[0]])
        encoder_stats = shards.encoder_normalization_stats()
        shards.identity["encoder_normalization_sha256"] = hashlib.sha256(json.dumps(encoder_stats, sort_keys=True).encode("utf-8")).hexdigest()
        trainer = cls(config, representative, encoder_stats=encoder_stats)
        trainer.shards = shards
        trainer.formal_sampler = FormalTrajectorySampler(shards.index, interface.train_indices, seed=config.seed)
        trainer.data.samples = list(interface.samples)  # index metadata only
        trainer.data.train_indices = list(interface.train_indices)
        trainer.data.validation_indices = list(interface.validation_indices)
        trainer.data.identity = shards.identity
        return trainer

    def _select_train_indices(self, global_step: int) -> list[int]:
        return self.formal_sampler.batch_for_global_step(global_step, self.config.batch_size)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        try:
            payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(Path(path), map_location="cpu")
        if payload.get("config") != _jsonable(asdict(self.config)):
            raise ValueError("incompatible formal checkpoint training config")
        return super().load_checkpoint(path)

    def _run_batch(self, indices: Sequence[int], horizon: int, *, stage: str, beta_kl: float, training: bool) -> dict[str, Any]:
        batch = self.shards.load_batch(indices)
        original = self.data
        try:
            self.data = batch
            self._move_data_to_device()
            return super()._run_batch(list(range(len(indices))), horizon, stage=stage, beta_kl=beta_kl, training=training)
        finally:
            self.data = original

    def validate_indices(self, indices: Sequence[int]) -> dict[str, Any]:
        if any(index not in self.data.validation_indices for index in indices):
            raise ValueError("validation batch contains a non-validation index")
        self.eval()
        with torch.no_grad():
            result = self._run_batch(indices, self.config.max_horizon, stage="prior_dominant_recursive", beta_kl=0.0, training=False)
        return {"prior_only": not result["posterior_teacher_used"], "horizon_rows": result["horizon_rows"], "sample_ids": [self.shards.index[i]["metadata"]["sample_id"] for i in indices]}

    def validate(self) -> dict[str, Any]:
        """Traverse every validation index in bounded batches, preserving L_Val."""
        was_training = self.training
        self.eval()
        self.posterior_teacher_calls = 0
        self.posterior_rollout_calls = 0
        self.future_target_encoder_calls = 0
        self.current_posterior_calls = 0
        before = self._parameter_snapshot()
        rows: dict[int, list[dict[str, Any]]] = {}
        with torch.no_grad():
            indices = self.data.validation_indices
            for start in range(0, len(indices), self.config.batch_size):
                result = self._run_batch(indices[start:start + self.config.batch_size], self.config.max_horizon, stage="prior_dominant_recursive", beta_kl=0.0, training=False)
                for row in result["horizon_rows"]:
                    rows.setdefault(int(row["horizon"]), []).append(row)
        _, changed_count = self._changed(before, self._parameter_snapshot())
        per_horizon = aggregate_validation_horizon_rows(rows)
        l_val = float(np.mean([row["L_Pred"] for row in per_horizon])) if per_horizon else float("nan")
        result = {
            "model_eval": True, "no_grad": True, "prior_only_rollout": True,
            "posterior_rollout_calls": self.posterior_rollout_calls,
            "future_posterior_teacher_calls": self.posterior_teacher_calls,
            "future_target_encoder_calls": self.future_target_encoder_calls,
            "current_observation_posterior_calls": self.current_posterior_calls,
            "per_horizon": per_horizon, "L_Val": l_val,
            "l_val_finite": bool(np.isfinite(l_val)), "selector_uses_l_val_only": True,
            "parameter_changed_count": changed_count, "validation_no_parameter_update": changed_count == 0,
            "future_state_not_rollout_input": True,
        }
        self.last_validation = result
        if was_training:
            self.train()
        return result
