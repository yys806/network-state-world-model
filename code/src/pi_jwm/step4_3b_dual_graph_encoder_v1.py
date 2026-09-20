"""STEP 4.3B typed dual-graph encoder contract.

This module ends at the aligned same-time graph representation ``Z_t^{PI,L_g}``.
It deliberately contains no world-model state, prediction head, action input,
loss, optimizer, or training loop.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


SCHEMA_VERSION = "PI-JWM-Joint-Graph-Representation-v1-step4.3B"
NORMALIZATION_SCHEMA_VERSION = "PI-JWM-Step-4.3B-Fixed-Train-Only-Normalization-v1"
REQUIRED_ACCEPTANCE_CHECKS = frozenset({
    "typed_feature_encoding", "explicit_feature_mask_input", "objectwise_temporal_encoding",
    "presence_gated_gru", "current_relation_encoding", "no_phy_edge_temporal_memory",
    "no_comm_temporal_memory", "logical_carrying_separate_then_fuse", "one_flow_relation_latent",
    "relation_family_specific_processing", "semantic_computational_direction_separated",
    "reverse_comm", "reverse_flow", "reverse_task_agent", "dag_forward_only",
    "direction_embedding", "relation_wise_aggregation", "masked_mean", "residual_node_update",
    "residual_relation_update", "p2a_gated_align_only", "p2c_gated_geocomm_only",
    "wired_p2c_disabled", "no_matching_physical_edge_requirement", "no_info_to_physical",
    "no_shortcut_relations", "history_causal", "future_target_isolation", "index_not_numeric_feature",
    "permutation_equivariance", "inactive_padding_isolation", "output_alignment",
    "p2a_joint_context_value_processor", "p2c_joint_context_value_processor",
    "p2a_gate_value_separated", "p2c_gate_value_separated", "complete_structural_interface",
    "node_alignment_preserved", "relation_alignment_preserved", "cross_alignment_preserved",
    "structural_presence_validity_preserved", "complete_output_digest", "comm_width_from_tensor_contract",
    "z_pi_not_world_model_latent", "deterministic_forward", "serialize_load", "autograd_smoke", "scope",
})


@dataclass(frozen=True)
class DualGraphEncoderConfig:
    d_h: int = 16
    mlp_hidden_width: int = 24
    type_embedding_dim: int = 4
    lifecycle_embedding_dim: int = 4
    direction_embedding_dim: int = 4
    graph_layers: int = 2
    enable_p2a: bool = True
    enable_p2c: bool = True
    enable_reverse_comm: bool = True
    enable_reverse_flow: bool = True
    enable_reverse_task_agent: bool = True
    enable_reverse_dag: bool = False
    csi_history_encoder: bool = False
    aggregator: str = "masked_mean"
    update_rule: str = "residual_layernorm"
    initialization_seed: int = 431
    development_only: bool = True
    research_frozen: bool = False

    def __post_init__(self) -> None:
        if min(self.d_h, self.mlp_hidden_width, self.type_embedding_dim, self.lifecycle_embedding_dim, self.direction_embedding_dim, self.graph_layers) <= 0:
            raise ValueError("encoder dimensions and graph_layers must be positive")
        if self.aggregator != "masked_mean" or self.update_rule != "residual_layernorm":
            raise ValueError("STEP 4.3B freezes masked_mean and residual_layernorm")
        if self.enable_reverse_dag:
            raise ValueError("DAG reverse computation is forbidden")


class TypeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.normalization = nn.LayerNorm(output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.normalization(self.linear2(self.activation(self.linear1(values))))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sample_splits(tensor: Mapping[str, Any]) -> list[str]:
    return [str(row.get("split", "missing")) for row in tensor.get("sample_metadata", [])]


def _feature_stat(features: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name not in features:
        raise ValueError(f"missing normalization statistic: {name}")
    return features[name]


def fit_encoder_normalization_stats(
    tensor: Mapping[str, Any], graph: Mapping[str, Any], upstream_stats_path: str | Path
) -> dict[str, Any]:
    """Compose frozen upstream statistics and fit only new physical-edge fields."""
    upstream = json.loads(Path(upstream_stats_path).read_text(encoding="utf-8"))
    features: dict[str, Any] = {}
    features.update(upstream["base_step3_2_stats"]["features"])
    features.update(upstream["features"])
    features.update(tensor["flow_normalization_stats"]["features"])
    names = [
        "physical_relation.delta_x_m", "physical_relation.delta_y_m", "physical_relation.delta_z_m",
        "physical_relation.distance_m", "physical_relation.relative_speed_mps",
        "physical_relation.relative_acceleration_mps2",
    ]
    units = ["m", "m", "m", "m", "m/s", "m/s^2"]
    source_fields = ["delta_x", "delta_y", "delta_z", "distance", "relative_speed", "relative_acceleration"]
    values = np.asarray(graph["blocks"]["physical_relations"]["features"], dtype=np.float64)
    masks = np.asarray(graph["blocks"]["physical_relations"]["feature_mask"], dtype=bool)
    active = np.asarray(graph["blocks"]["physical_relations"]["presence"], dtype=bool) & np.asarray(graph["blocks"]["physical_relations"]["validity"], dtype=bool)
    train = np.asarray([split == "dev_train" for split in _sample_splits(tensor)], dtype=bool)
    if train.shape[0] != values.shape[0]:
        raise ValueError("sample split metadata does not align with graph batch")
    for fi, name in enumerate(names):
        valid = train[:, None] & active & masks[..., fi] & np.isfinite(values[..., fi])
        selected = values[..., fi][valid]
        if selected.size == 0:
            raise ValueError(f"no dev_train values for {name}")
        std = float(selected.std())
        features[name] = {
            "count": int(selected.size), "mean": float(selected.mean()), "std": std if std > 0 else 1.0,
            "zero_variance_handling": "population_std" if std > 0 else "scale=1.0",
            "source_field": f"physical_relations.features[{fi}]/{source_fields[fi]}", "unit": units[fi],
            "mask_policy": "dev_train only AND presence=true AND validity=true AND feature_mask=true AND value finite",
            "preprocessing_policy": "z_score_on_valid_train_values",
        }
    return {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "source_split": "dev_train",
        "fit_policy": "reuse frozen Step 3.2/4.2A/4.2C-C train statistics; fit only new 4.3A physical-relation fields on dev_train",
        "features": features,
    }


def build_encoder_input_audit(tensor: Mapping[str, Any], graph: Mapping[str, Any], upstream_stats_path: str | Path, graph_manifest_path: str | Path | None = None, tensor_path: str | Path | None = None) -> dict[str, Any]:
    stats = fit_encoder_normalization_stats(tensor, graph, upstream_stats_path)
    non_flow_pairs = [
        ("entity_position_raw", "entity_position"), ("entity_raw_features", "entity_features"),
        ("task_raw_features", "task_features"), ("agent_cpu_capacity_raw", "agent_cpu_capacity"),
        ("task_history_extended_raw_features", "task_history_extended_features"), ("comm_csi_raw", "comm_csi"),
    ]
    pre_normalized = any(not np.array_equal(np.asarray(tensor[a]), np.asarray(tensor[b])) for a, b in non_flow_pairs)
    source_hash_traceable = True
    source_hash_evidence: dict[str, Any] = {"mode": "schema_only_no_manifest_supplied"}
    if graph_manifest_path is not None and tensor_path is not None:
        manifest = json.loads(Path(graph_manifest_path).read_text(encoding="utf-8"))
        actual_sha = hashlib.sha256(Path(tensor_path).read_bytes()).hexdigest()
        recorded_sha = manifest.get("source", {}).get("sha256")
        source_hash_traceable = recorded_sha == actual_sha
        source_hash_evidence = {"graph_manifest": Path(graph_manifest_path).as_posix(), "tensor_path": Path(tensor_path).as_posix(), "recorded_tensor_sha256": recorded_sha, "actual_tensor_sha256": actual_sha}
    audit = {
        "schema_version": "PI-JWM-Step-4.3B-Encoder-Input-Audit-v1",
        "source_tensor_schema_version": tensor.get("schema_version"),
        "source_graph_schema_version": graph.get("schema_version"),
        "current_non_flow_tensor_features_are_pre_normalized": pre_normalized,
        "normalization_schema_version": stats["schema_version"],
        "source_hash_evidence": source_hash_evidence,
        "continuous_inputs": {
            "physical_node": {"source": "History raw position/speed/acceleration", "mask": "entity presence and per-feature mask", "normalization": "fixed dev_train statistics"},
            "physical_relation": {"source": "4.3A current derived relation features", "mask": "relation presence/validity and per-feature mask", "normalization": "4.3B dev_train-only additive statistics"},
            "agent": {"source": "static CPU capacity", "mask": "agent presence and CPU mask", "normalization": "fixed dev_train statistics"},
            "task": {"source": "History task raw fields", "mask": "task presence and per-feature mask", "normalization": "fixed dev_train statistics"},
            "comm": {"source": "current CSI", "mask": "relation presence/validity and CSI mask", "normalization": "fixed dev_train statistics"},
            "logical_flow": {"source": "History logical Flow raw fields", "learned_fields": ["logical.total_data", "logical.e2e_remaining"], "excluded_redundant_fields": ["logical.e2e_delivered"], "excluded_structural_fields": ["epoch", "flow_index"]},
            "carrying": {"source": "History carrying raw fields", "learned_fields": ["carrying.hop_progress", "carrying.hop_remaining", "carrying.active"], "excluded_structural_fields": ["route_revision", "current_hop_index", "holder_index", "hop_source_index", "hop_destination_index", "route_node_indices"]},
        },
        "unavailable_fields": {
            "agent.dynamic_available_cpu": "UPSTREAM_FIELD_NOT_AVAILABLE",
            "agent.storage": "UPSTREAM_FIELD_NOT_AVAILABLE",
            "comm.queue_or_service_load": "UPSTREAM_FIELD_NOT_AVAILABLE",
            "task.return_size": "UPSTREAM_FIELD_NOT_AVAILABLE",
            "task.priority": "UPSTREAM_FIELD_NOT_AVAILABLE",
            "task.deadline": "UPSTREAM_FIELD_NOT_AVAILABLE",
            "physical.heading_elevation_posture": "UPSTREAM_FIELD_NOT_AVAILABLE",
        },
        "checks": {
            "tensor_graph_schema_lineage": graph.get("contract", {}).get("source_tensor_schema_version") == tensor.get("schema_version"),
            "tensor_graph_source_hash_lineage": source_hash_traceable,
            "normalization_source_dev_train": stats.get("source_split") == "dev_train",
            "no_target_namespace_consumed": graph.get("contract", {}).get("target_namespace_consumed") is False,
            "raw_non_flow_normalization_truth_recorded": pre_normalized is False,
            "required_statistics_available": len(stats["features"]) >= 20,
        },
    }
    audit["passed"] = bool(all(audit["checks"].values()))
    return audit


def _t(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype)


class PIJointGraphEncoder(nn.Module):
    def __init__(self, config: DualGraphEncoderConfig, tensor_contract: Mapping[str, Any], graph_contract: Mapping[str, Any], normalization_stats: Mapping[str, Any]):
        super().__init__()
        self.config = config
        self.tensor_contract = _jsonable(tensor_contract)
        self.graph_contract = _jsonable(graph_contract)
        self.normalization_stats = _jsonable(normalization_stats)
        if graph_contract.get("source_tensor_schema_version") != tensor_contract.get("schema_version", tensor_contract.get("source_schema_version")) and graph_contract.get("source_tensor_schema_version") != "PI-JWM-Model-Input-Tensor-Collation-v5-step4.2C-C-PATCH":
            raise ValueError("tensor/graph schema lineage mismatch")
        previous = torch.random.get_rng_state()
        torch.manual_seed(config.initialization_seed)
        d, w, te, le, de = config.d_h, config.mlp_hidden_width, config.type_embedding_dim, config.lifecycle_embedding_dim, config.direction_embedding_dim
        self.entity_type_embedding = nn.Embedding(16, te)
        self.agent_type_embedding = nn.Embedding(16, te)
        self.lifecycle_embedding = nn.Embedding(16, le)
        self.comm_type_embedding = nn.Embedding(8, te)
        self.flow_type_embedding = nn.Embedding(8, te)
        self.task_agent_type_embedding = nn.Embedding(8, te)
        self.dag_type_embedding = nn.Embedding(2, te)
        self.direction_embedding = nn.Embedding(2, de)
        self.physical_node_encoder = TypeMLP(5 + 5 + te, w, d)
        self.agent_encoder = TypeMLP(1 + 1 + te, w, d)
        self.task_encoder = TypeMLP(5 + 5 + le, w, d)
        self.physical_relation_encoder = TypeMLP(6 + 6, w, d)
        n_comm_rb = int(tensor_contract.get("n_comm_rb", 0))
        if n_comm_rb <= 0:
            raise ValueError("tensor contract must provide positive n_comm_rb")
        self.n_comm_rb = n_comm_rb
        self.comm_relation_encoder = TypeMLP(n_comm_rb + n_comm_rb + te, w, d)
        self.logical_flow_encoder = TypeMLP(2 + 2 + te, w, d)
        self.carrying_encoder = TypeMLP(2 + 2 + 1, w, d)
        self.flow_fuse_encoder = TypeMLP(2 * d, w, d)
        self.task_agent_relation_encoder = TypeMLP(te, w, d)
        self.dag_relation_encoder = TypeMLP(te, w, d)
        self.physical_gru = nn.GRUCell(d, d)
        self.agent_gru = nn.GRUCell(d, d)
        self.task_gru = nn.GRUCell(d, d)
        self.flow_gru = nn.GRUCell(d, d)
        relation_input = 3 * d + de
        self.relation_processors = nn.ModuleDict({name: TypeMLP(relation_input, w, d) for name in ("physical", "comm", "flow", "task_agent", "dag")})
        self.agent_family_fuse = TypeMLP(3 * d + 3, w, d)
        self.task_family_fuse = TypeMLP(2 * d + 2, w, d)
        self.physical_node_update = TypeMLP(2 * d, w, d)
        self.agent_cross_fuse = TypeMLP(2 * d + 2, w, d)
        self.agent_node_update = TypeMLP(2 * d, w, d)
        self.task_node_update = TypeMLP(2 * d, w, d)
        self.p2a_gate = nn.Linear(2 * d, d)
        self.p2a_value = TypeMLP(2 * d, w, d)
        self.p2c_gate = nn.Linear(3 * d, d)
        self.p2c_value = TypeMLP(3 * d, w, d)
        self.physical_node_norm = nn.LayerNorm(d)
        self.agent_node_norm = nn.LayerNorm(d)
        self.task_node_norm = nn.LayerNorm(d)
        self.relation_norms = nn.ModuleDict({name: nn.LayerNorm(d) for name in ("physical", "comm", "flow", "task_agent", "dag")})
        torch.random.set_rng_state(previous)

    def _stats(self, names: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.normalization_stats["features"]
        mean = torch.tensor([float(_feature_stat(features, n)["mean"]) for n in names], dtype=torch.float32, device=device)
        std = torch.tensor([float(_feature_stat(features, n)["std"]) for n in names], dtype=torch.float32, device=device)
        return mean, std

    def _normalize(self, raw: torch.Tensor, mask: torch.Tensor, names: list[str]) -> torch.Tensor:
        mean, std = self._stats(names, raw.device)
        normalized = (raw - mean) / std
        return torch.where(mask, normalized, torch.zeros_like(normalized))

    @staticmethod
    def _presence_gru(encoded: torch.Tensor, presence: torch.Tensor, cell: nn.GRUCell) -> torch.Tensor:
        batch, history, count, dim = encoded.shape
        hidden = torch.zeros((batch, count, dim), dtype=encoded.dtype, device=encoded.device)
        for hi in range(history):
            candidate = cell(encoded[:, hi].reshape(batch * count, dim), hidden.reshape(batch * count, dim)).reshape(batch, count, dim)
            hidden = torch.where(presence[:, hi, :, None], candidate, hidden)
        return hidden

    @staticmethod
    def _safe_gather(nodes: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        safe = indices.clamp(min=0, max=max(nodes.shape[1] - 1, 0))
        return torch.gather(nodes, 1, safe[..., None].expand(*safe.shape, nodes.shape[-1]))

    @staticmethod
    def _masked_mean(messages: torch.Tensor, indices: torch.Tensor, valid: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, dim = messages.shape
        total = torch.zeros((batch, count, dim), dtype=messages.dtype, device=messages.device)
        denom = torch.zeros((batch, count, 1), dtype=messages.dtype, device=messages.device)
        for bi in range(batch):
            chosen = valid[bi] & (indices[bi] >= 0) & (indices[bi] < count)
            if bool(chosen.any()):
                target = indices[bi, chosen]
                total[bi].index_add_(0, target, messages[bi, chosen])
                denom[bi].index_add_(0, target, torch.ones((int(chosen.sum()), 1), dtype=messages.dtype, device=messages.device))
        return total / denom.clamp_min(1.0), denom.squeeze(-1) > 0

    def _relation_pass(self, family: str, relation: torch.Tensor, source_nodes: torch.Tensor, target_nodes: torch.Tensor, source_index: torch.Tensor, target_index: torch.Tensor, valid: torch.Tensor, reverse: bool, extra_delta: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src = self._safe_gather(source_nodes, source_index)
        dst = self._safe_gather(target_nodes, target_index)
        forward_direction = self.direction_embedding(torch.zeros_like(source_index).clamp(min=0))
        forward = self.relation_processors[family](torch.cat((src, dst, relation, forward_direction), dim=-1))
        forward = forward * valid[..., None]
        delta = forward if extra_delta is None else forward + extra_delta
        updated = self.relation_norms[family](relation + delta) * valid[..., None]
        target_msg, target_has = self._masked_mean(updated, target_index, valid, target_nodes.shape[1])
        source_msg = torch.zeros_like(source_nodes)
        source_has = torch.zeros(source_nodes.shape[:2], dtype=torch.bool, device=source_nodes.device)
        if reverse:
            reverse_direction = self.direction_embedding(torch.ones_like(source_index))
            reverse_msg = self.relation_processors[family](torch.cat((dst, src, relation, reverse_direction), dim=-1)) * valid[..., None]
            source_msg, source_has = self._masked_mean(reverse_msg, source_index, valid, source_nodes.shape[1])
        return updated, source_msg, source_has, target_msg, target_has

    def forward(self, tensor: Mapping[str, Any], graph: Mapping[str, Any], *, validate_inputs: bool = True) -> dict[str, Any]:
        if validate_inputs:
            if graph.get("contract", {}).get("source_tensor_schema_version") != tensor.get("schema_version"):
                raise ValueError("tensor and graph lineage mismatch")
            if graph.get("contract", {}).get("target_namespace_consumed") is not False:
                raise ValueError("graph must not consume future target namespace")
        device = next(self.parameters()).device
        f32 = lambda key: _t(tensor[key], dtype=torch.float32).to(device)
        boo = lambda key: _t(tensor[key], dtype=torch.bool).to(device)
        lng = lambda key: _t(tensor[key], dtype=torch.long).to(device)

        entity_presence = boo("entity_presence")
        pos_mask, entity_mask = boo("entity_position_mask"), boo("entity_feature_mask")
        pos = self._normalize(f32("entity_position_raw"), pos_mask, ["entity.position_x_m", "entity.position_y_m", "entity.position_z_m"])
        dyn = self._normalize(f32("entity_raw_features"), entity_mask, ["entity.speed_mps", "entity.canonical_acceleration_mps2"])
        entity_types = self.entity_type_embedding(lng("entity_type_index").clamp(0, 15))
        physical_encoded = self.physical_node_encoder(torch.cat((pos, dyn, pos_mask.float(), entity_mask.float(), entity_types), -1))
        physical_hidden = self._presence_gru(physical_encoded, entity_presence, self.physical_gru)

        cpu_mask = boo("agent_cpu_capacity_mask")[:, None, :].expand_as(entity_presence)
        cpu_raw = f32("agent_cpu_capacity_raw")[:, None, :].expand_as(entity_presence.float())
        cpu = self._normalize(cpu_raw[..., None], cpu_mask[..., None], ["agent.cpu_capacity_per_s"])
        agent_types = self.agent_type_embedding(lng("entity_type_index").clamp(0, 15))
        agent_encoded = self.agent_encoder(torch.cat((cpu, cpu_mask[..., None].float(), agent_types), -1))
        agent_hidden = self._presence_gru(agent_encoded, entity_presence, self.agent_gru)

        task_presence, task_mask = boo("task_presence"), torch.cat((boo("task_feature_mask"), boo("task_history_extended_feature_mask")), -1)
        task_raw = torch.cat((f32("task_raw_features"), f32("task_history_extended_raw_features")), -1)
        task_values = self._normalize(task_raw, task_mask, ["task.task_size", "task.task_cpu_work", "task.computed_cpu_work", "task.transmitted_size", "task.elapsed_time_s"])
        lifecycle = self.lifecycle_embedding(lng("task_lifecycle_index").clamp(0, 15))
        task_encoded = self.task_encoder(torch.cat((task_values, task_mask.float(), lifecycle), -1))
        task_hidden = self._presence_gru(task_encoded, task_presence, self.task_gru)

        flow_presence, flow_mask_all = boo("logical_flow_presence"), boo("logical_flow_feature_mask")
        flow_mask = flow_mask_all[..., [0, 2]]
        flow_raw = f32("logical_flow_raw_features")[..., [0, 2]]
        flow_values = self._normalize(flow_raw, flow_mask, ["logical.total_data", "logical.e2e_remaining"])
        flow_types = self.flow_type_embedding(lng("logical_flow_type_index").clamp(0, 7))
        logical_encoded = self.logical_flow_encoder(torch.cat((flow_values, flow_mask.float(), flow_types), -1))
        carrying_mask = boo("carrying_feature_mask")
        carrying_values = self._normalize(f32("carrying_raw_features"), carrying_mask, ["carrying.hop_progress", "carrying.hop_remaining"])
        carrying_encoded = self.carrying_encoder(torch.cat((carrying_values, carrying_mask.float(), boo("carrying_active")[..., None].float()), -1))
        fused_flow = self.flow_fuse_encoder(torch.cat((logical_encoded, carrying_encoded), -1))
        flow_hidden = self._presence_gru(fused_flow, flow_presence, self.flow_gru)

        blocks = graph["blocks"]
        physical_rel = blocks["physical_relations"]
        pr_raw, pr_mask = _t(physical_rel["features"], dtype=torch.float32).to(device), _t(physical_rel["feature_mask"], dtype=torch.bool).to(device)
        pr_values = self._normalize(pr_raw, pr_mask, ["physical_relation.delta_x_m", "physical_relation.delta_y_m", "physical_relation.delta_z_m", "physical_relation.distance_m", "physical_relation.relative_speed_mps", "physical_relation.relative_acceleration_mps2"])
        phy_rel_latent = self.physical_relation_encoder(torch.cat((pr_values, pr_mask.float()), -1))
        comm = blocks["comm_relations"]
        csi_mask = _t(comm["csi_mask"], dtype=torch.bool).to(device)
        if int(comm["csi"].shape[-1]) != self.n_comm_rb:
            raise ValueError(f"communication CSI width {comm['csi'].shape[-1]} != tensor contract n_comm_rb {self.n_comm_rb}")
        csi = self._normalize(_t(comm["csi"], dtype=torch.float32).to(device), csi_mask, ["comm.channel_attenuation_db"] * int(csi_mask.shape[-1]))
        comm_type_idx = _t(comm["relation_type_index"], dtype=torch.long).to(device)
        comm_latent = self.comm_relation_encoder(torch.cat((csi, csi_mask.float(), self.comm_type_embedding(comm_type_idx.clamp(0, 7))), -1))
        ta = blocks["task_agent_relations"]
        ta_type_idx = _t(ta["relation_type_index"], dtype=torch.long).to(device)
        ta_latent = self.task_agent_relation_encoder(self.task_agent_type_embedding(ta_type_idx.clamp(0, 7)))
        dag = blocks["dag_relations"]
        dag_latent = self.dag_relation_encoder(self.dag_type_embedding(torch.ones_like(_t(dag["source_task_index"], dtype=torch.long).to(device))))

        phy_presence = _t(blocks["physical_nodes"]["presence"], dtype=torch.bool).to(device)
        agent_presence = _t(blocks["agent_nodes"]["presence"], dtype=torch.bool).to(device)
        current_task_presence = _t(blocks["task_nodes"]["presence"], dtype=torch.bool).to(device)
        phy_valid = _t(physical_rel["presence"], dtype=torch.bool).to(device) & _t(physical_rel["validity"], dtype=torch.bool).to(device)
        comm_valid = _t(comm["presence"], dtype=torch.bool).to(device) & _t(comm["validity"], dtype=torch.bool).to(device)
        flow_block = blocks["flow_relations"]
        flow_valid = _t(flow_block["presence"], dtype=torch.bool).to(device) & _t(flow_block["validity"], dtype=torch.bool).to(device)
        ta_valid = _t(ta["presence"], dtype=torch.bool).to(device) & _t(ta["validity"], dtype=torch.bool).to(device)
        dag_valid = _t(dag["presence"], dtype=torch.bool).to(device) & _t(dag["validity"], dtype=torch.bool).to(device)

        align = blocks["align_relations"]
        align_valid = _t(align["presence"], dtype=torch.bool).to(device) & _t(align["validity"], dtype=torch.bool).to(device)
        align_phy = _t(align["physical_entity_index"], dtype=torch.long).to(device)
        align_agent = _t(align["agent_index"], dtype=torch.long).to(device)
        geo = blocks["geo_comm_relations"]
        geo_valid = _t(geo["presence"], dtype=torch.bool).to(device) & _t(geo["validity"], dtype=torch.bool).to(device) & (comm_type_idx == 2)
        geo_src_index = _t(geo["source_physical_index"], dtype=torch.long).to(device)
        geo_dst_index = _t(geo["target_physical_index"], dtype=torch.long).to(device)

        initial_p2c_message = None
        for layer_index in range(self.config.graph_layers):
            aligned_phy = self._safe_gather(physical_hidden, align_phy)
            aligned_agent = self._safe_gather(agent_hidden, align_agent)
            p2a_gate = torch.sigmoid(self.p2a_gate(torch.cat((aligned_phy, aligned_agent), -1))) * align_valid[..., None]
            p2a_value_message = self.p2a_value(torch.cat((aligned_phy, aligned_agent), -1))
            p2a_edge_message = p2a_gate * p2a_value_message * align_valid[..., None]
            p2a_message, p2a_has = self._masked_mean(p2a_edge_message, align_agent, align_valid, agent_hidden.shape[1])
            geo_src = self._safe_gather(physical_hidden, geo_src_index)
            geo_dst = self._safe_gather(physical_hidden, geo_dst_index)
            p2c_gate = torch.sigmoid(self.p2c_gate(torch.cat((geo_src, geo_dst, comm_latent), -1))) * geo_valid[..., None]
            p2c_value_message = self.p2c_value(torch.cat((geo_src, geo_dst, comm_latent), -1))
            p2c_message = p2c_gate * p2c_value_message * geo_valid[..., None]
            if layer_index == 0:
                initial_p2c_message = p2c_message
            phy_rel_latent, phy_reverse, phy_reverse_has, phy_forward, phy_forward_has = self._relation_pass(
                "physical", phy_rel_latent, physical_hidden, physical_hidden,
                _t(physical_rel["source_index"], dtype=torch.long).to(device), _t(physical_rel["target_index"], dtype=torch.long).to(device), phy_valid, False)
            physical_message = phy_forward
            physical_has = phy_forward_has
            physical_delta = self.physical_node_update(torch.cat((physical_hidden, physical_message), -1))
            physical_hidden = torch.where(phy_presence[..., None], self.physical_node_norm(physical_hidden + physical_delta), torch.zeros_like(physical_hidden))

            comm_latent, comm_src, comm_src_has, comm_dst, comm_dst_has = self._relation_pass(
                "comm", comm_latent, agent_hidden, agent_hidden, _t(comm["source_index"], dtype=torch.long).to(device), _t(comm["target_index"], dtype=torch.long).to(device), comm_valid, self.config.enable_reverse_comm, p2c_message if self.config.enable_p2c else None)
            flow_latent, flow_src, flow_src_has, flow_dst, flow_dst_has = self._relation_pass(
                "flow", flow_hidden, agent_hidden, agent_hidden, _t(flow_block["source_index"], dtype=torch.long).to(device), _t(flow_block["destination_index"], dtype=torch.long).to(device), flow_valid, self.config.enable_reverse_flow)
            ta_latent, ta_task, ta_task_has, ta_agent, ta_agent_has = self._relation_pass(
                "task_agent", ta_latent, task_hidden, agent_hidden, _t(ta["task_index"], dtype=torch.long).to(device), _t(ta["agent_index"], dtype=torch.long).to(device), ta_valid, self.config.enable_reverse_task_agent)
            dag_latent, _, _, dag_task, dag_task_has = self._relation_pass(
                "dag", dag_latent, task_hidden, task_hidden, _t(dag["source_task_index"], dtype=torch.long).to(device), _t(dag["target_task_index"], dtype=torch.long).to(device), dag_valid, False)
            comm_msg, comm_has = comm_src + comm_dst, comm_src_has | comm_dst_has
            flow_msg, flow_has = flow_src + flow_dst, flow_src_has | flow_dst_has
            agent_msg = self.agent_family_fuse(torch.cat((comm_msg, flow_msg, ta_agent, comm_has[..., None].float(), flow_has[..., None].float(), ta_agent_has[..., None].float()), -1))
            cross_message = p2a_message if self.config.enable_p2a else torch.zeros_like(p2a_message)
            all_agent_message = self.agent_cross_fuse(torch.cat((agent_msg, cross_message, (comm_has | flow_has | ta_agent_has)[..., None].float(), p2a_has[..., None].float()), -1))
            agent_delta = self.agent_node_update(torch.cat((agent_hidden, all_agent_message), -1))
            agent_hidden = torch.where(agent_presence[..., None], self.agent_node_norm(agent_hidden + agent_delta), torch.zeros_like(agent_hidden))
            task_msg = self.task_family_fuse(torch.cat((ta_task, dag_task, ta_task_has[..., None].float(), dag_task_has[..., None].float()), -1))
            task_delta = self.task_node_update(torch.cat((task_hidden, task_msg), -1))
            task_hidden = torch.where(current_task_presence[..., None], self.task_node_norm(task_hidden + task_delta), torch.zeros_like(task_hidden))

        phy_rel_latent = phy_rel_latent * phy_valid[..., None]
        comm_latent = comm_latent * comm_valid[..., None]
        flow_latent = flow_latent * flow_valid[..., None]
        ta_latent = ta_latent * ta_valid[..., None]
        dag_latent = dag_latent * dag_valid[..., None]
        output = {
            "schema_version": SCHEMA_VERSION,
            "contract": {
                "representation_name": "Z_t^{PI,L_g}", "world_model_latent": False,
                "source_tensor_schema_version": tensor.get("schema_version"), "source_graph_schema_version": graph.get("schema_version"),
                "history_only": True, "target_namespace_consumed": False, "action_consumed": False,
                "logical_flow_learned_numeric_fields": ["logical.total_data", "logical.e2e_remaining"],
                "carrying_role": "side_state_fused_before_single_flow_GRU_not_second_relation",
                "message_passing": "family_specific_directed_masked_mean_residual_layernorm",
                "physical_topology": graph.get("contract", {}).get("physical_topology"),
                "config": asdict(self.config),
            },
            "physical": {"node_latent": physical_hidden, "relation_latent": phy_rel_latent},
            "information": {"agent_latent": agent_hidden, "task_latent": task_hidden, "comm_relation_latent": comm_latent, "flow_relation_latent": flow_latent, "task_agent_relation_latent": ta_latent, "dag_relation_latent": dag_latent},
            "structural": {
                "physical_nodes": {k: _t(v).to(device) for k, v in blocks["physical_nodes"].items()},
                "agent_nodes": {k: _t(v).to(device) for k, v in blocks["agent_nodes"].items()},
                "task_nodes": {k: _t(v).to(device) for k, v in blocks["task_nodes"].items()},
                "physical_relations": {k: _t(v).to(device) for k, v in physical_rel.items()},
                "comm_relations": {k: _t(v).to(device) for k, v in comm.items()},
                "flow_relations": {k: _t(v).to(device) for k, v in flow_block.items()},
                "flow_carrying_state": {k: _t(v).to(device) for k, v in blocks["flow_carrying_state"].items()},
                "task_agent_relations": {k: _t(v).to(device) for k, v in ta.items()},
                "dag_relations": {k: _t(v).to(device) for k, v in dag.items()},
                "align_relations": {k: _t(v).to(device) for k, v in align.items()},
                "geo_comm_relations": {k: _t(v).to(device) for k, v in geo.items()},
            },
            "diagnostics": {"p2a_gate": p2a_gate, "p2a_value_message": p2a_value_message, "p2a_message": p2a_edge_message, "p2c_gate": p2c_gate, "p2c_value_message": p2c_value_message, "p2c_message": p2c_message, "p2c_initial_message": initial_p2c_message, "p2a_valid": align_valid, "p2c_valid": geo_valid},
        }
        return output


def output_semantic_digest(output: Mapping[str, Any]) -> str:
    payload: dict[str, Any] = {"schema_version": output.get("schema_version"), "contract": output.get("contract")}
    for section in ("physical", "information", "structural", "diagnostics"):
        payload[section] = {key: _jsonable(value) for key, value in output.get(section, {}).items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def save_encoder_package(model: PIJointGraphEncoder, path: str | Path) -> None:
    torch.save({"schema_version": SCHEMA_VERSION, "config": asdict(model.config), "normalization_stats": model.normalization_stats, "state_dict": model.state_dict()}, Path(path))


def load_encoder_package(path: str | Path, tensor_contract: Mapping[str, Any], graph_contract: Mapping[str, Any]) -> PIJointGraphEncoder:
    package = torch.load(Path(path), map_location="cpu", weights_only=False)
    if package.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("encoder package schema mismatch")
    model = PIJointGraphEncoder(DualGraphEncoderConfig(**package["config"]), tensor_contract, graph_contract, package["normalization_stats"])
    model.load_state_dict(package["state_dict"])
    return model


def validate_encoder_architecture_checks(model: PIJointGraphEncoder) -> dict[str, Any]:
    encoders = [model.physical_node_encoder, model.agent_encoder, model.task_encoder, model.physical_relation_encoder, model.comm_relation_encoder, model.logical_flow_encoder, model.carrying_encoder, model.flow_fuse_encoder]
    grus = [model.physical_gru, model.agent_gru, model.task_gru, model.flow_gru]
    processors = list(model.relation_processors.values())
    modules = dict(model.named_modules())
    checks = {
        "type_specific_encoders": len({id(next(module.parameters())) for module in encoders}) == len(encoders) and all(isinstance(module, TypeMLP) for module in encoders),
        "independent_grus": len({id(next(module.parameters())) for module in grus}) == 4,
        "no_relation_temporal_grus": "physical_relation_gru" not in modules and "comm_gru" not in modules and not model.config.csi_history_encoder,
        "logical_carrying_separate_then_fuse": model.logical_flow_encoder is not model.carrying_encoder and model.flow_fuse_encoder not in (model.logical_flow_encoder, model.carrying_encoder),
        "family_specific_processors": len(processors) == 5 and len({id(next(module.parameters())) for module in processors}) == 5,
        "independent_node_updates": len({id(next(module.parameters())) for module in (model.physical_node_update, model.agent_node_update, model.task_node_update)}) == 3,
        "direction_embedding": isinstance(model.direction_embedding, nn.Embedding),
        "direction_policy": model.config.enable_reverse_comm and model.config.enable_reverse_flow and model.config.enable_reverse_task_agent and not model.config.enable_reverse_dag,
        "cross_policy": model.config.enable_p2a and model.config.enable_p2c,
        "aggregation_update_policy": model.config.aggregator == "masked_mean" and model.config.update_rule == "residual_layernorm",
        "no_forbidden_shortcut_modules": not any(name in modules for name in ("agent_to_physical", "task_to_physical", "flow_to_physical", "comm_to_physical", "task_physical", "flow_physical_edge")),
        "development_config_not_research_frozen": model.config.development_only and not model.config.research_frozen,
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def validate_encoder_contract_checks(model: PIJointGraphEncoder, output: Mapping[str, Any], tensor: Mapping[str, Any], graph: Mapping[str, Any]) -> dict[str, Any]:
    blocks = graph["blocks"]
    structural_names = ("physical_nodes", "agent_nodes", "task_nodes", "physical_relations", "comm_relations", "flow_relations", "task_agent_relations", "dag_relations", "flow_carrying_state", "align_relations", "geo_comm_relations")
    def equal_value(left: Any, right: Any) -> bool:
        if isinstance(left, torch.Tensor):
            left = left.detach().cpu().numpy()
        if isinstance(right, torch.Tensor):
            right = right.detach().cpu().numpy()
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            try:
                return bool(np.array_equal(np.asarray(left), np.asarray(right)))
            except (TypeError, ValueError):
                return False
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return set(left) == set(right) and all(equal_value(left[k], right[k]) for k in left)
        return left == right
    structural = output.get("structural", {})
    structural_equal = all(name in structural and equal_value(structural[name], blocks[name]) for name in structural_names)
    node_alignment = all(equal_value(structural.get(name, {}).get(index_key), blocks[name].get(index_key)) for name, index_key in (("physical_nodes", "entity_index"), ("agent_nodes", "entity_index"), ("task_nodes", "task_index")))
    relation_alignment = all(equal_value(structural.get(name, {}).get(index_key), blocks[name].get(index_key)) for name, index_key in (("physical_relations", "source_index"), ("comm_relations", "relation_index"), ("flow_relations", "flow_index"), ("task_agent_relations", "relation_index"), ("dag_relations", "relation_index")))
    cross_alignment = all(equal_value(structural.get(name), blocks[name]) for name in ("align_relations", "geo_comm_relations"))
    presence_validity = all(equal_value(structural.get(name, {}).get("presence"), blocks[name].get("presence")) and ("validity" not in blocks[name] or equal_value(structural.get(name, {}).get("validity"), blocks[name].get("validity"))) for name in ("physical_relations", "comm_relations", "flow_relations", "task_agent_relations", "dag_relations", "align_relations", "geo_comm_relations"))
    checks = {
        "schema": output.get("schema_version") == SCHEMA_VERSION,
        "tensor_lineage": output.get("contract", {}).get("source_tensor_schema_version") == tensor.get("schema_version"),
        "graph_lineage": output.get("contract", {}).get("source_graph_schema_version") == graph.get("schema_version"),
        "history_only": output.get("contract", {}).get("history_only") is True and output.get("contract", {}).get("target_namespace_consumed") is False,
        "z_pi_only": output.get("contract", {}).get("representation_name") == "Z_t^{PI,L_g}" and output.get("contract", {}).get("world_model_latent") is False and "world_model_latent" not in output and "global_latent" not in output,
        "physical_shape": tuple(output["physical"]["node_latent"].shape[:2]) == tuple(np.asarray(blocks["physical_nodes"]["presence"]).shape),
        "agent_shape": tuple(output["information"]["agent_latent"].shape[:2]) == tuple(np.asarray(blocks["agent_nodes"]["presence"]).shape),
        "task_shape": tuple(output["information"]["task_latent"].shape[:2]) == tuple(np.asarray(blocks["task_nodes"]["presence"]).shape),
        "flow_shape": tuple(output["information"]["flow_relation_latent"].shape[:2]) == tuple(np.asarray(blocks["flow_relations"]["presence"]).shape),
        "single_flow_relation": "carrying_relation_latent" not in output.get("information", {}) and "flow_carrying_state" in output.get("structural", {}),
        "masked_padding_zero": bool(torch.all(output["physical"]["node_latent"][~_t(blocks["physical_nodes"]["presence"], dtype=torch.bool)] == 0)) and bool(torch.all(output["information"]["task_latent"][~_t(blocks["task_nodes"]["presence"], dtype=torch.bool)] == 0)),
        "p2c_wireless_only": bool(torch.all(output["diagnostics"]["p2c_message"][_t(blocks["comm_relations"]["relation_type_index"], dtype=torch.long) != 2] == 0)),
        "development_scope": model.config.development_only and not model.config.research_frozen,
        "complete_structural_interface": structural_equal,
        "node_alignment_preserved": node_alignment,
        "relation_alignment_preserved": relation_alignment,
        "cross_alignment_preserved": cross_alignment,
        "structural_presence_validity_preserved": presence_validity,
        "comm_width_from_tensor_contract": model.n_comm_rb == int(tensor.get("contract", {}).get("n_comm_rb", -1)),
    }
    architecture = validate_encoder_architecture_checks(model)
    checks["architecture"] = architecture["passed"]
    return {"checks": checks, "architecture_checks": architecture, "passed": bool(all(checks.values()))}


def validate_encoder_acceptance(receipt: Mapping[str, Any]) -> dict[str, Any]:
    required = receipt.get("required_checks", {})
    scope = receipt.get("scope", {})
    forbidden = ("rssm", "world_model", "future_action", "prediction", "loss", "planner", "training", "optimizer", "gpu", "locked_test", "formal_dataset")
    checks = {
        "all_required_present": set(required) == set(REQUIRED_ACCEPTANCE_CHECKS),
        "all_required_true": set(required) == set(REQUIRED_ACCEPTANCE_CHECKS) and all(required.get(name) is True for name in REQUIRED_ACCEPTANCE_CHECKS),
        "scope_explicit_false": all(scope.get(name) is False for name in forbidden),
    }
    expected = bool(all(checks.values()))
    checks["reported_passed_matches_expected"] = receipt.get("passed") is expected
    return {"checks": checks, "expected_passed": expected, "passed": bool(all(checks.values()) and expected)}
