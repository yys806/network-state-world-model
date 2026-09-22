"""STEP 4.4 structured RSSM world-model contract.

This is an untrained CPU development implementation.  Learned modules model
only vehicle motion and wireless CSI.  Wireless outage is an explicit known
stochastic event; Flow/Task/CPU/UAV transitions remain rules.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import PhysicalTopologyConfig, _physical_edges
from pi_jwm.step3_3_model_input_tensor_v1 import LIFECYCLE_VOCAB
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import FLOW_STATUS_VOCAB, FLOW_TYPE_VOCAB


SCHEMA_VERSION = "PI-JWM-Structured-RSSM-World-Model-v1-step4.4"
ENTITY_VEHICLE = 2
ENTITY_UAV = 3
COMM_WIRELESS = 2
COMM_WIRED = 3
ORIGINAL_ACCEPTANCE_CHECKS = (
    "zpi_to_structured_latent_alignment", "no_global_pooling", "independent_deterministic_states", "only_phy_comm_stochastic_modules",
    "vehicle_stochastic_eligibility", "uav_no_stochastic_motion", "static_no_stochastic_motion", "diagonal_gaussian_prior", "diagonal_gaussian_posterior",
    "prior_mean_mode", "stochastic_sampling_api", "future_prior_target_isolation", "posterior_prior_separation", "four_action_routing", "action_not_global_broadcast",
    "invalid_action_index_rejected", "dynamics_interaction_not_history_gru", "future_no_history_encoder", "independent_dynamics_processors", "state_feedback",
    "vehicle_motion_head_boundary", "acceleration_derived", "uav_rule_driven", "csi_learned", "no_unconstrained_rate_head", "flow_no_stochastic_state",
    "flow_no_remaining_head", "task_no_stochastic_state", "task_rule_driven", "agent_no_stochastic_state", "learned_then_rule_execution",
    "dynamic_physical_graph", "dynamic_comm_graph", "task_agent_update", "flow_identity_preserved", "dag_static_preserved", "align_geocomm_update",
    "future_only_object_isolation", "fixed_current_object_support", "recursive_prior_rollout", "step2_consumes_step1", "no_future_observation_leakage",
    "no_target_leakage", "structural_alignment", "fixed_seed_deterministic", "serialize_load", "permutation_equivariance", "cpu_autograd",
    "input_immutable", "scope",
)
ADDITIONAL_SERVICE_CHECKS = (
    "outage_not_decision_input", "outage_not_future_target_input", "comm_z_decodes_csi_not_outage", "nominal_rate_exact", "outage_probability_exact",
    "sample_conditioned_on_sinr", "outage_true_zero_rate", "outage_false_nominal_rate", "expectation_exact", "expectation_marked_approximate",
    "seeded_sample_reproducible", "seed_change_outage_not_nominal", "no_learned_outage_head", "no_learned_rate_head", "no_learned_service_residual",
    "wired_capacity_causal_provenance", "wired_membership_equality", "wired_rule_matches_simulator", "flow_remaining_cap", "intermediate_hop_no_e2e_reduction",
    "complete_stochastic_event_trace",
)
ADDITIONAL_STRUCTURAL_CHECKS = (
    "no_raw_index_learned_feature", "categorical_embedding_semantics", "graph_layers_effective",
    "future_topology_matches_builder_policy", "no_unintended_self_physical_edges", "carrying_state_complete",
    "hop_service_capped_by_hop_remaining", "hop_advancement", "dynamic_flow_comm_mapping",
    "flow_completion_presence_sync", "flow_completion_status_sync", "route_revision_semantics", "task_lifecycle_rule_complete",
    "dag_dynamic_rule", "comm_endpoint_presence_validity", "task_agent_dynamic_validity",
    "strong_recursive_counterfactual", "existing_return_flow_typed_binding",
    "future_return_birth_unsupported", "computation_finished_not_final_without_return",
    "input_flow_not_return_substitute",
)
REQUIRED_ACCEPTANCE_CHECKS = frozenset(ORIGINAL_ACCEPTANCE_CHECKS + ADDITIONAL_SERVICE_CHECKS + ADDITIONAL_STRUCTURAL_CHECKS)


def bind_existing_return_flows(
    task_presence: torch.Tensor,
    flow_known: torch.Tensor,
    flow_task_index: torch.Tensor,
    flow_type_index: torch.Tensor,
) -> torch.Tensor:
    """Bind current-support Return Flows by typed structural identity."""
    if not (flow_known.shape == flow_task_index.shape == flow_type_index.shape):
        raise ValueError("Flow support tensors must have identical shapes")
    mapping = torch.full(task_presence.shape, -1, dtype=torch.long, device=task_presence.device)
    return_type = FLOW_TYPE_VOCAB.index("Return")
    for batch_index in range(task_presence.shape[0]):
        for task_index in range(task_presence.shape[1]):
            if not bool(task_presence[batch_index, task_index]):
                continue
            hits = torch.nonzero(
                flow_known[batch_index]
                & (flow_task_index[batch_index] == task_index)
                & (flow_type_index[batch_index] == return_type),
                as_tuple=False,
            ).flatten()
            if len(hits) > 1:
                raise ValueError(f"multiple current-support Return Flows for task {task_index}")
            if len(hits) == 1:
                mapping[batch_index, task_index] = hits[0]
    return mapping


@dataclass(frozen=True)
class StructuredRSSMConfig:
    d_encoder: int = 16
    d_h: int = 16
    d_z: int = 4
    mlp_width: int = 24
    graph_layers: int = 2
    n_comm_rb: int = 50
    rollout_horizon: int = 2
    slot_duration_s: float = 0.1
    rb_bandwidth_mhz: float = 2.0
    noise_power_mw: float = 3.9810717055349695e-12
    outage_snr_threshold: float = 10.0
    v2v_tx_power_dbm: float = 23.0
    vehicle_to_infra_tx_power_dbm: float = 26.0
    uav_or_rsu_tx_power_dbm: float = 29.0
    prior_log_std_min: float = -5.0
    prior_log_std_max: float = 2.0
    posterior_log_std_min: float = -5.0
    posterior_log_std_max: float = 2.0
    physical_topology_mode: str = "radius_knn"
    physical_radius_m: float = 1000.0
    physical_k: int = 2
    initialization_seed: int = 440
    development_only: bool = True
    research_frozen: bool = False

    def __post_init__(self) -> None:
        if min(self.d_encoder, self.d_h, self.d_z, self.mlp_width, self.graph_layers, self.n_comm_rb, self.rollout_horizon) <= 0:
            raise ValueError("all dimensions, layers and horizon must be positive")
        if self.slot_duration_s <= 0 or self.rb_bandwidth_mhz <= 0 or self.noise_power_mw <= 0:
            raise ValueError("physical constants must be positive")
        if not self.prior_log_std_min < self.prior_log_std_max or not self.posterior_log_std_min < self.posterior_log_std_max:
            raise ValueError("invalid log_std clamp")


class MLP(nn.Module):
    def __init__(self, n_in: int, width: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, width), nn.SiLU(), nn.Linear(width, n_out), nn.LayerNorm(n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiagonalGaussian(nn.Module):
    def __init__(self, n_in: int, width: int, d_z: int, low: float, high: float):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, width), nn.SiLU(), nn.Linear(width, 2 * d_z))
        self.low, self.high = float(low), float(high)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mean, log_std = self.body(x).chunk(2, dim=-1)
        return {"mean": mean, "log_std": log_std.clamp(self.low, self.high)}


def wireless_nominal_rate_mbps(sinr_db: torch.Tensor, rb_bandwidth_mhz: float) -> torch.Tensor:
    """AirFogSim rule: MHz * log2(1 + 10 ** (SINR_dB/10)) -> Mbps."""
    return float(rb_bandwidth_mhz) * torch.log2(1.0 + torch.pow(10.0, sinr_db / 10.0))


def rayleigh_outage_probability(sinr_db: torch.Tensor, snr_threshold: float) -> torch.Tensor:
    """Exact audited callback (not a learned classifier)."""
    return 1.0 - torch.exp(-float(snr_threshold) / sinr_db.clamp_min(1e-9))


def apply_known_stochastic_wireless_service(
    sinr_db: torch.Tensor,
    *,
    rb_bandwidth_mhz: float,
    snr_threshold: float,
    mode: str,
    generator: torch.Generator | None = None,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    if mode not in {"sample", "expectation"}:
        raise ValueError("service mode must be sample or expectation")
    valid = torch.ones_like(sinr_db, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    nominal = wireless_nominal_rate_mbps(sinr_db, rb_bandwidth_mhz) * valid
    probability = rayleigh_outage_probability(sinr_db, snr_threshold) * valid
    if mode == "sample":
        if generator is None:
            raise ValueError("sample mode requires an explicit torch.Generator")
        draw = torch.rand(sinr_db.shape, generator=generator, device="cpu", dtype=sinr_db.dtype).to(sinr_db.device)
        outage = (draw < probability) & valid
        actual = torch.where(outage, torch.zeros_like(nominal), nominal)
    else:
        outage = torch.zeros_like(valid)
        draw = torch.full_like(sinr_db, float("nan"))
        actual = (1.0 - probability) * nominal
    return {
        "nominal_rate_mbps": nominal,
        "outage_probability": probability,
        "outage_uniform_draw": draw,
        "outage": outage,
        "actual_rate_mbps": actual,
        "mode": mode,
        "expected_service_approximation": mode == "expectation",
    }


def resolve_wireless_sinr_db(
    csi_attenuation_db: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    source_type_index: torch.Tensor,
    target_type_index: torch.Tensor,
    rb_allocation: torch.Tensor,
    *,
    noise_power_mw: float,
    v2v_tx_power_dbm: float,
    vehicle_to_infra_tx_power_dbm: float,
    uav_or_rsu_tx_power_dbm: float,
) -> torch.Tensor:
    """Resolve every wireless relation/RB using the complete allocation.

    Interference at a receiver uses the audited CSI relation from every other
    active transmitter to that receiver. Missing cross-channel CSI contributes
    no invented power and therefore must be rejected by upstream validation for
    a formally complete allocation; the frozen development graph contains the
    directed V/U/I support required here.
    """
    bsz, n_rel, n_rb = csi_attenuation_db.shape
    def power(tx_type: int, rx_type: int) -> float:
        if tx_type == ENTITY_VEHICLE:
            return float(v2v_tx_power_dbm if rx_type == ENTITY_VEHICLE else vehicle_to_infra_tx_power_dbm)
        return float(uav_or_rsu_tx_power_dbm)
    batches: list[torch.Tensor] = []
    for b in range(bsz):
        lookup = {(int(source_index[b, r]), int(target_index[b, r])): r for r in range(n_rel) if int(source_index[b, r]) >= 0 and int(target_index[b, r]) >= 0}
        rows: list[torch.Tensor] = []
        for r in range(n_rel):
            src, dst = int(source_index[b, r]), int(target_index[b, r])
            power_dbm = power(int(source_type_index[b, r]), int(target_type_index[b, r]))
            signal = torch.pow(10.0, (power_dbm - csi_attenuation_db[b, r]) / 10.0) * rb_allocation[b, r]
            interference = torch.full((n_rb,), float(noise_power_mw), dtype=signal.dtype, device=signal.device)
            for other in range(n_rel):
                if other == r or not bool(rb_allocation[b, other].any()):
                    continue
                cross = lookup.get((int(source_index[b, other]), dst))
                if cross is None:
                    continue
                other_power = power(int(source_type_index[b, other]), int(target_type_index[b, cross]))
                interference = interference + torch.pow(10.0, (other_power - csi_attenuation_db[b, cross]) / 10.0) * rb_allocation[b, other]
            linear = (signal + 1e-10) / (interference + 1e-10)
            rows.append((10.0 * torch.log10(linear)).clamp_min(1e-9))
        batches.append(torch.stack(rows))
    return torch.stack(batches)


def derive_wired_active_membership(carrying: Mapping[str, torch.Tensor], wired_links: set[tuple[int, int]]) -> dict[tuple[int, int, int], tuple[int, ...]]:
    active = carrying["active"].bool()
    src, dst = carrying["hop_source_index"].long(), carrying["hop_destination_index"].long()
    flow_index = carrying.get("flow_index", torch.arange(active.shape[1], device=active.device)[None].expand_as(src)).long()
    result: dict[tuple[int, int, int], list[int]] = {}
    for b in range(active.shape[0]):
        for f in range(active.shape[1]):
            pair = (int(src[b, f]), int(dst[b, f]))
            if bool(active[b, f]) and pair in wired_links:
                result.setdefault((b, *pair), []).append(int(flow_index[b, f]))
    return {key: tuple(sorted(value)) for key, value in result.items()}


def wired_fair_share_service(remaining_bytes: torch.Tensor, *, capacity_mbps: float, slot_duration_s: float, active_count: int) -> torch.Tensor:
    if active_count <= 0:
        return torch.zeros_like(remaining_bytes)
    per_flow_bytes = float(capacity_mbps) * 1e6 / 8.0 * float(slot_duration_s) / int(active_count)
    return torch.minimum(remaining_bytes, torch.full_like(remaining_bytes, per_flow_bytes))


class StructuredRSSMWorldModel(nn.Module):
    """Entity/relation-aligned structured RSSM with explicit rule feedback."""

    FAMILIES = ("physical", "agent", "communication", "flow", "task")

    @property
    def dynamics_processors(self) -> nn.ModuleDict:
        """Compatibility view of the first graph-interaction layer."""
        return self.dynamics_processor_layers[0]

    def __init__(self, config: StructuredRSSMConfig):
        super().__init__()
        self.config = config
        prior_rng = torch.random.get_rng_state()
        torch.manual_seed(config.initialization_seed)
        d0, d, z, w = config.d_encoder, config.d_h, config.d_z, config.mlp_width
        self.phy_init, self.agent_init, self.comm_init, self.flow_init, self.task_init = [MLP(d0, w, d) for _ in range(5)]
        self.action_encoders = nn.ModuleDict({name: MLP(4, w, d) for name in self.FAMILIES})
        self.route_action_encoder = MLP(2 + 2 * d, w, d)
        self.state_encoders = nn.ModuleDict({
            "physical": MLP(7 + d, w, d), "agent": MLP(7 + d, w, d),
            "communication": MLP(6 + d, w, d), "flow": MLP(7 + 2 * d, w, d), "task": MLP(4 + d, w, d),
        })
        processor_names = ("physical", "communication", "flow", "task_agent", "dag", "p2a", "p2c")
        self.dynamics_processor_layers = nn.ModuleList([
            nn.ModuleDict({name: MLP(3 * d, w, d) for name in processor_names})
            for _ in range(config.graph_layers)
        ])
        self.entity_type_embedding = nn.Embedding(16, d)
        self.comm_type_embedding = nn.Embedding(16, d)
        self.lifecycle_embedding = nn.Embedding(16, d)
        self.flow_type_embedding = nn.Embedding(16, d)
        self.flow_status_embedding = nn.Embedding(16, d)
        self.task_agent_type_embedding = nn.Embedding(16, d)
        self.gru_cells = nn.ModuleDict({
            "physical": nn.GRUCell(4 * d + z, d),
            "agent": nn.GRUCell(4 * d, d),
            "communication": nn.GRUCell(4 * d + z, d),
            "flow": nn.GRUCell(4 * d, d),
            "task": nn.GRUCell(4 * d, d),
        })
        self.phy_prior = DiagonalGaussian(d, w, z, config.prior_log_std_min, config.prior_log_std_max)
        self.comm_prior = DiagonalGaussian(d, w, z, config.prior_log_std_min, config.prior_log_std_max)
        self.phy_posterior = DiagonalGaussian(d + d0, w, z, config.posterior_log_std_min, config.posterior_log_std_max)
        self.comm_posterior = DiagonalGaussian(d + d0, w, z, config.posterior_log_std_min, config.posterior_log_std_max)
        self.vehicle_decoder = nn.Sequential(nn.Linear(d + z, w), nn.SiLU(), nn.Linear(w, 4))
        self.csi_decoder = nn.Sequential(nn.Linear(d + z, w), nn.SiLU(), nn.Linear(w, config.n_comm_rb))
        torch.random.set_rng_state(prior_rng)
        self.contract = {
            "schema_version": SCHEMA_VERSION,
            "latent_layout": {"h": list(self.FAMILIES), "z": ["physical_vehicle", "communication"]},
            "stochastic_families": ["physical_vehicle", "communication"],
            "learned_heads": ["vehicle_delta_xyz_next_speed", "future_per_rb_csi"],
            "csi_decoder_output_space": "raw_db",
            "known_stochastic_transition": "wireless_per_valid_rb_outage",
            "learned_outage_head": False,
            "learned_rate_head": False,
            "learned_service_residual": False,
            "future_target_consumed_by_prior": False,
            "future_history_encoder_reused": False,
            "fixed_current_object_support": True,
            "future_return_birth_supported": False,
            "physical_topology": {"mode": config.physical_topology_mode, "radius_m": config.physical_radius_m, "k": config.physical_k, "development_only": True, "research_frozen": False},
            "graph_layers": config.graph_layers,
            "categorical_feature_policy": "typed_embedding_only_no_continuous_index_scalars",
            "evidence_class": "UNTRAINED_DEVELOPMENT_WORLD_MODEL_EVIDENCE",
        }

    @staticmethod
    def _mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x * mask[..., None]

    def initialize_latent(self, z_pi: Mapping[str, Any], state: Mapping[str, torch.Tensor], *, posterior_mode: str = "mean", generator: torch.Generator | None = None) -> dict[str, Any]:
        """Initialize the structured latent state.

        ``mean`` and ``sample`` use the current-observation posterior
        ``q(z_t | h_t, Z_t)`` and are the semantic initialization for the
        training-loop current time step.  ``prior`` remains an explicit
        diagnostic mode that derives the initial stochastic state from prior
        heads without evaluating a posterior; it is not a substitute for the
        current-observation posterior in Stage 2 or validation.  Future
        recursive steps are prior-only, while Future Target teachers live in
        the separate training-only branch.
        """
        p = z_pi["physical"]["node_latent"]
        i = z_pi["information"]
        h = {
            "physical": self._mask(self.phy_init(p), state["entity_presence"]),
            "agent": self._mask(self.agent_init(i["agent_latent"]), state["entity_presence"]),
            "communication": self._mask(self.comm_init(i["comm_relation_latent"]), state["comm_presence"]),
            "flow": self._mask(self.flow_init(i["flow_relation_latent"]), state["flow_presence"]),
            "task": self._mask(self.task_init(i["task_latent"]), state["task_presence"]),
        }
        if posterior_mode == "prior":
            phy_q = self.phy_prior(h["physical"])
            comm_q = self.comm_prior(h["communication"])
            selected_mode = "mean"
            posterior = None
        else:
            phy_q = self.phy_posterior(torch.cat((h["physical"], p), -1))
            comm_q = self.comm_posterior(torch.cat((h["communication"], i["comm_relation_latent"]), -1))
            selected_mode = posterior_mode
            posterior = {"physical": phy_q, "communication": comm_q}
        z = {
            "physical": self._select_z(phy_q, selected_mode, generator) * state["vehicle_mask"][..., None],
            "communication": self._select_z(comm_q, selected_mode, generator) * state["comm_presence"][..., None],
        }
        return {"h": h, "z": z, "posterior": posterior, "initial_prior": {"physical": phy_q, "communication": comm_q} if posterior_mode == "prior" else None}

    @staticmethod
    def _select_z(dist: Mapping[str, torch.Tensor], mode: str, generator: torch.Generator | None) -> torch.Tensor:
        if mode == "mean":
            return dist["mean"]
        if mode != "sample" or generator is None:
            raise ValueError("sample mode requires explicit generator")
        eps = torch.randn(dist["mean"].shape, generator=generator, device="cpu", dtype=dist["mean"].dtype).to(dist["mean"].device)
        return dist["mean"] + torch.exp(dist["log_std"]) * eps

    def route_actions(self, action: Mapping[str, torch.Tensor], state: Mapping[str, torch.Tensor], graph: Mapping[str, torch.Tensor], route_latent: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        shapes = {
            "physical": state["entity_presence"].shape,
            "agent": state["entity_presence"].shape,
            "communication": state["comm_presence"].shape,
            "flow": state["flow_presence"].shape,
            "task": state["task_presence"].shape,
        }
        routed = {name: torch.zeros((*shape, self.config.d_h), dtype=state["position"].dtype, device=state["position"].device) for name, shape in shapes.items()}

        def scatter(family: str, indices: torch.Tensor, values: torch.Tensor, allowed: torch.Tensor, encoded_values: torch.Tensor | None = None) -> None:
            for b in range(indices.shape[0]):
                for row in range(indices.shape[1]):
                    idx = int(indices[b, row])
                    if idx < 0:
                        continue
                    if idx >= routed[family].shape[1] or not bool(allowed[b, idx]):
                        raise ValueError(f"invalid or absent {family} action index {idx}")
                    routed[family][b, idx] += self.action_encoders[family]((values if encoded_values is None else encoded_values)[b, row])

        scatter("physical", action["mobility_entity_index"], action["mobility_values"], state["uav_mask"])
        scatter("communication", action["comm_relation_index"], action["comm_values"], state["comm_presence"])
        scatter("agent", action["comp_agent_index"], action["comp_values"], state["entity_presence"])
        scatter("task", action["comp_task_index"], action["comp_values"], state["task_presence"])
        route_values = action["route_values"]
        entity_route_latent = route_latent if route_latent is not None else torch.zeros((*state["entity_presence"].shape, self.config.d_h), dtype=route_values.dtype, device=route_values.device)
        route_encoded = []
        for b in range(route_values.shape[0]):
            rows = []
            for r in range(route_values.shape[1]):
                src, dst = route_values[b, r, :2].long()
                if src < 0 or dst < 0 or src >= state["entity_presence"].shape[1] or dst >= state["entity_presence"].shape[1]:
                    rows.append(torch.zeros((2 + 2 * self.config.d_h,), dtype=route_values.dtype, device=route_values.device))
                    continue
                rows.append(torch.cat((route_values[b, r, 2:], entity_route_latent[b, src], entity_route_latent[b, dst])))
            route_encoded.append(torch.stack(rows))
        route_encoded_t = torch.stack(route_encoded)
        for family, indices, allowed in (("task", action["route_task_index"], state["task_presence"]), ("flow", action["route_flow_index"], state["flow_presence"])):
            for b in range(indices.shape[0]):
                for row in range(indices.shape[1]):
                    idx = int(indices[b, row])
                    if idx < 0:
                        continue
                    if idx >= routed[family].shape[1] or not bool(allowed[b, idx]):
                        raise ValueError(f"invalid or absent {family} action index {idx}")
                    routed[family][b, idx] += self.route_action_encoder(route_encoded_t[b, row])
        return routed

    def _state_features(self, family: str, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if family in {"physical", "agent"}:
            typed = self.entity_type_embedding(state["entity_type_index"].clamp(0, 15))
            raw = torch.cat((state["position"], state["speed"][..., None], state["acceleration"][..., None], typed, state["entity_presence"][..., None].float(), state["uav_mask"][..., None].float()), -1)
        elif family == "communication":
            csi = state["csi"]
            typed = self.comm_type_embedding(state["comm_type_index"].clamp(0, 15))
            raw = torch.cat((csi.mean(-1, keepdim=True), csi.std(-1, keepdim=True), typed, state["comm_validity"][..., None].float(), state["comm_presence"][..., None].float(), state["csi_mask"].float().mean(-1, keepdim=True), torch.zeros((*csi.shape[:2], 1), device=csi.device)), -1)
        elif family == "flow":
            flow_type = self.flow_type_embedding(state["flow_type_index"].clamp(0, 15))
            flow_status = self.flow_status_embedding(state["flow_status_index"].clamp(0, 15))
            raw = torch.cat((state["flow_total"][..., None], state["flow_remaining"][..., None], state["carrying_active"][..., None].float(), state["flow_presence"][..., None].float(), state["hop_progress"][..., None], state["hop_remaining"][..., None], flow_type, flow_status, torch.zeros((*state["flow_total"].shape, 1), device=state["flow_total"].device)), -1)
        else:
            typed = self.lifecycle_embedding(state["task_lifecycle_index"].clamp(0, 15))
            raw = torch.cat((state["task_work_remaining"][..., None], state["task_progress"][..., None], typed, state["task_presence"][..., None].float(), torch.zeros((*state["task_progress"].shape, 1), device=state["task_progress"].device)), -1)
        return self.state_encoders[family](raw)

    def _dynamics_interaction_once(self, latent: Mapping[str, Any], graph: Mapping[str, torch.Tensor], processors: nn.ModuleDict) -> dict[str, torch.Tensor]:
        h = latent["h"]
        def gather(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            return torch.gather(values, 1, index.clamp_min(0)[..., None].expand(-1, -1, values.shape[-1]))

        def mean_to(messages: torch.Tensor, index: torch.Tensor, valid: torch.Tensor, n: int) -> torch.Tensor:
            out = torch.zeros((messages.shape[0], n, messages.shape[-1]), device=messages.device, dtype=messages.dtype)
            count = torch.zeros((messages.shape[0], n, 1), device=messages.device, dtype=messages.dtype)
            for b in range(messages.shape[0]):
                ids = index[b][valid[b]]
                if ids.numel():
                    out[b].index_add_(0, ids, messages[b][valid[b]])
                    count[b].index_add_(0, ids, torch.ones((ids.numel(), 1), device=messages.device, dtype=messages.dtype))
            return out / count.clamp_min(1)

        # Physical typed propagation uses the rebuilt dense validity matrix.
        pv = graph["physical_relation_validity"].bool()
        src_phy = h["physical"][:, :, None, :].expand(-1, -1, h["physical"].shape[1], -1)
        dst_phy = h["physical"][:, None, :, :].expand(-1, h["physical"].shape[1], -1, -1)
        phy_msg = processors["physical"](torch.cat((src_phy, dst_phy, dst_phy - src_phy), -1)) * pv[..., None]
        phy_context = phy_msg.sum(1) / pv.sum(1).clamp_min(1)[..., None]

        cs, cd = graph["comm_source_index"].long(), graph["comm_target_index"].long()
        comm_valid = graph["comm_validity"].bool()
        comm_msg = processors["communication"](torch.cat((gather(h["agent"], cs), gather(h["agent"], cd), h["communication"]), -1)) * comm_valid[..., None]
        comm_agent = mean_to(comm_msg, cd, comm_valid, h["agent"].shape[1]) + mean_to(comm_msg, cs, comm_valid, h["agent"].shape[1])

        fs, fd = graph["flow_source_index"].long(), graph["flow_destination_index"].long()
        flow_valid = graph["flow_presence"].bool()
        flow_msg = processors["flow"](torch.cat((gather(h["agent"], fs), gather(h["agent"], fd), h["flow"]), -1)) * flow_valid[..., None]
        flow_agent = mean_to(flow_msg, fd, flow_valid, h["agent"].shape[1]) + mean_to(flow_msg, fs, flow_valid, h["agent"].shape[1])

        ta_t, ta_a, ta_valid = graph["task_agent_task_index"].long(), graph["task_agent_agent_index"].long(), graph["task_agent_validity"].bool()
        ta_msg = processors["task_agent"](torch.cat((gather(h["task"], ta_t), gather(h["agent"], ta_a), gather(h["agent"], ta_a) - gather(h["task"], ta_t)), -1)) * ta_valid[..., None]
        ta_agent = mean_to(ta_msg, ta_a, ta_valid, h["agent"].shape[1])
        ta_task = mean_to(ta_msg, ta_t, ta_valid, h["task"].shape[1])

        dag_src, dag_dst = graph["dag_edges"][..., 0].long(), graph["dag_edges"][..., 1].long()
        dag_valid = graph["dag_validity"].bool()
        dag_msg = processors["dag"](torch.cat((gather(h["task"], dag_src), gather(h["task"], dag_dst), gather(h["task"], dag_dst) - gather(h["task"], dag_src)), -1)) * dag_valid[..., None]
        dag_task = mean_to(dag_msg, dag_dst, dag_valid, h["task"].shape[1])

        p2a = processors["p2a"](torch.cat((h["physical"], h["agent"], h["agent"] - h["physical"]), -1)) * graph["align_validity"][..., None]
        p2c = processors["p2c"](torch.cat((gather(h["physical"], cs), gather(h["physical"], cd), h["communication"]), -1)) * graph["geo_comm_validity"][..., None]
        result = {
            "physical": phy_context,
            "agent": comm_agent + flow_agent + ta_agent + p2a,
            "communication": comm_msg + p2c,
            "flow": flow_msg,
            "task": ta_task + dag_task,
        }
        return result

    def dynamics_interaction(self, latent: Mapping[str, Any], graph: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        current = {"h": dict(latent["h"])}
        result: dict[str, torch.Tensor] = {}
        for processors in self.dynamics_processor_layers:
            result = self._dynamics_interaction_once(current, graph, processors)
            current["h"] = {name: current["h"][name] + result[name] for name in self.FAMILIES}
        return result

    def _next_h(self, family: str, latent: Mapping[str, Any], message: torch.Tensor, action: torch.Tensor, state_feature: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        parts = [message, action, state_feature, latent["h"][family]]
        if family in {"physical", "communication"}:
            parts.append(latent["z"][family])
        x = torch.cat(parts, -1)
        flat = self.gru_cells[family](x.reshape(-1, x.shape[-1]), latent["h"][family].reshape(-1, self.config.d_h)).reshape_as(latent["h"][family])
        return self._mask(flat, presence)

    def one_step(self, latent: Mapping[str, Any], state: Mapping[str, torch.Tensor], graph: Mapping[str, torch.Tensor], action: Mapping[str, torch.Tensor], *, prior_mode: str, service_mode: str, generator: torch.Generator | None) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
        routed = self.route_actions(action, state, graph, latent["h"]["agent"])
        messages = self.dynamics_interaction(latent, graph)
        presence = {"physical": state["entity_presence"], "agent": state["entity_presence"], "communication": state["comm_presence"], "flow": state["flow_presence"], "task": state["task_presence"]}
        next_h = {name: self._next_h(name, latent, messages[name], routed[name], self._state_features(name, state), presence[name]) for name in self.FAMILIES}
        phy_p, comm_p = self.phy_prior(next_h["physical"]), self.comm_prior(next_h["communication"])
        next_z = {
            "physical": self._select_z(phy_p, prior_mode, generator) * state["vehicle_mask"][..., None],
            "communication": self._select_z(comm_p, prior_mode, generator) * state["comm_presence"][..., None],
        }
        learned = {
            "vehicle_motion": self.vehicle_decoder(torch.cat((next_h["physical"], next_z["physical"]), -1)) * state["vehicle_mask"][..., None],
            "csi": self.csi_decoder(torch.cat((next_h["communication"], next_z["communication"]), -1)) * state["comm_presence"][..., None],
        }
        next_state, rule_trace = self.deterministic_transition(state, action, learned, graph=graph, service_mode=service_mode, generator=generator)
        next_graph = self.rebuild_graph(next_state, graph)
        return {"h": next_h, "z": next_z, "prior": {"physical": phy_p, "communication": comm_p}}, next_state, next_graph, {"learned": learned, "rule": rule_trace, "routed": routed}

    def deterministic_transition(self, state: Mapping[str, torch.Tensor], action: Mapping[str, torch.Tensor], learned: Mapping[str, torch.Tensor], *, graph: Mapping[str, torch.Tensor] | None = None, service_mode: str, generator: torch.Generator | None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        nxt = {k: v.clone() for k, v in state.items()}
        old_speed = state["speed"]
        vehicle = state["vehicle_mask"]
        motion = learned["vehicle_motion"]
        nxt["position"] = torch.where(vehicle[..., None], state["position"] + motion[..., :3], state["position"])
        nxt["speed"] = torch.where(vehicle, motion[..., 3].relu(), state["speed"])
        # UAV motion is rule-driven only.
        for b in range(action["mobility_entity_index"].shape[0]):
            for r in range(action["mobility_entity_index"].shape[1]):
                idx = int(action["mobility_entity_index"][b, r])
                if idx >= 0:
                    azimuth, elevation, speed, _ = action["mobility_values"][b, r]
                    delta = torch.stack((speed * torch.cos(azimuth) * torch.cos(elevation), speed * torch.sin(azimuth) * torch.cos(elevation), speed * torch.sin(elevation))) * self.config.slot_duration_s
                    nxt["position"][b, idx] = state["position"][b, idx] + delta
                    nxt["speed"][b, idx] = speed
        nxt["acceleration"] = (nxt["speed"] - old_speed) / self.config.slot_duration_s
        predicted_csi_mask = state["comm_presence"][..., None] & state["comm_wireless_mask"][..., None] & state["comm_validity"][..., None]
        nxt["csi"] = torch.where(predicted_csi_mask, learned["csi"], state["csi"])
        nxt["csi_mask"] = predicted_csi_mask.expand_as(state["csi_mask"])
        allocation = action["comm_allocation_mask"].bool()
        if allocation.shape != state["rb_active_mask"].shape:
            raise ValueError("complete Comm RB allocation shape mismatch")
        if bool((allocation & state["comm_wired_mask"][..., None]).any()):
            raise ValueError("wired relation cannot carry wireless RB allocation")
        nxt["rb_active_mask"] = allocation

        # Development transition receives an explicit SINR state derived by the
        # audited complete-allocation resolver; no future target is consulted.
        sinr_db = resolve_wireless_sinr_db(
            nxt["csi"], state["comm_source_index"], state["comm_target_index"], state["comm_source_type_index"], state["comm_target_type_index"], nxt["rb_active_mask"],
            noise_power_mw=self.config.noise_power_mw, v2v_tx_power_dbm=self.config.v2v_tx_power_dbm,
            vehicle_to_infra_tx_power_dbm=self.config.vehicle_to_infra_tx_power_dbm, uav_or_rsu_tx_power_dbm=self.config.uav_or_rsu_tx_power_dbm,
        )
        wireless = apply_known_stochastic_wireless_service(
            sinr_db, rb_bandwidth_mhz=self.config.rb_bandwidth_mhz,
            snr_threshold=self.config.outage_snr_threshold, mode=service_mode,
            generator=generator, valid_mask=state["comm_wireless_mask"][..., None] & nxt["csi_mask"] & nxt["rb_active_mask"],
        )
        per_relation_bytes = wireless["actual_rate_mbps"].sum(-1) * 1e6 / 8.0 * self.config.slot_duration_s
        delivered = torch.zeros_like(state["flow_remaining"])
        for b in range(delivered.shape[0]):
            for f in range(delivered.shape[1]):
                if not bool(state["flow_presence"][b, f] & state["carrying_active"][b, f]):
                    continue
                relation = int(state["flow_comm_relation_index"][b, f])
                amount = per_relation_bytes[b, relation] if relation >= 0 and bool(state["comm_wireless_mask"][b, relation]) else torch.tensor(0.0, device=delivered.device)
                if relation >= 0 and bool(state["comm_wired_mask"][b, relation]):
                    same = state["carrying_active"][b] & (state["flow_comm_relation_index"][b] == relation)
                    count = int(same.sum())
                    amount = wired_fair_share_service(state["flow_remaining"][b, f:f+1], capacity_mbps=float(state["wired_capacity_mbps"][b, relation]), slot_duration_s=self.config.slot_duration_s, active_count=count)[0]
                delivered[b, f] = torch.minimum(torch.minimum(amount, state["flow_remaining"][b, f]), state["hop_remaining"][b, f])
        terminal_hop = state["carrying_hop_destination_index"] == state["flow_destination_index"]
        next_hop_remaining = (state["hop_remaining"] - delivered).clamp_min(0)
        hop_complete = state["carrying_active"] & (state["hop_remaining"] > 0) & (next_hop_remaining == 0)
        nxt["hop_progress"] = state["hop_progress"] + delivered
        nxt["hop_remaining"] = next_hop_remaining
        e2e = delivered * terminal_hop
        nxt["flow_remaining"] = (state["flow_remaining"] - e2e).clamp_min(0)
        nxt["flow_presence"] = state["flow_presence"] & (nxt["flow_remaining"] > 0)
        nxt["carrying_active"] = state["carrying_active"] & (nxt["flow_remaining"] > 0)
        terminal_flow_complete = state["flow_presence"] & terminal_hop & (state["flow_remaining"] > 0) & (nxt["flow_remaining"] == 0)
        completed_flow_status = FLOW_STATUS_VOCAB.index("COMPLETED")
        nxt["flow_status_index"] = torch.where(
            terminal_flow_complete,
            torch.full_like(state["flow_status_index"], completed_flow_status),
            state["flow_status_index"],
        )
        # A completed intermediate hop advances along the declared route while
        # preserving the end-to-end Flow remaining amount.  Route indices are
        # gathered from the current state, never treated as learned scalars.
        for b in range(hop_complete.shape[0]):
            for f in range(hop_complete.shape[1]):
                if not bool(hop_complete[b, f]) or bool(terminal_hop[b, f]):
                    continue
                current = int(state["current_hop_index"][b, f])
                route = state["route_node_indices"][b, f]
                route_mask = state["route_node_mask"][b, f]
                next_hop = current + 1
                if next_hop + 1 >= route.shape[0] or not bool(route_mask[next_hop]) or not bool(route_mask[next_hop + 1]):
                    continue
                nxt["current_hop_index"][b, f] = next_hop
                nxt["current_holder_index"][b, f] = route[next_hop]
                nxt["carrying_hop_source_index"][b, f] = route[next_hop]
                nxt["carrying_hop_destination_index"][b, f] = route[next_hop + 1]
                nxt["hop_progress"][b, f] = 0.0
                nxt["hop_remaining"][b, f] = state["flow_remaining"][b, f]
        # CPU/task rules, not heads.
        cpu_service = action["comp_values"][..., 0].clamp_min(0) * self.config.slot_duration_s
        for b in range(cpu_service.shape[0]):
            for r in range(cpu_service.shape[1]):
                ti = int(action["comp_task_index"][b, r])
                if ti >= 0:
                    nxt["task_work_remaining"][b, ti] = torch.clamp(state["task_work_remaining"][b, ti] - cpu_service[b, r], min=0)
        # Route is a structural rule: update Carrying/current Task-Agent host,
        # never create a new logical Flow identity.
        for b in range(action["route_flow_index"].shape[0]):
            for r in range(action["route_flow_index"].shape[1]):
                fi, ti = int(action["route_flow_index"][b, r]), int(action["route_task_index"][b, r])
                if fi < 0 or ti < 0:
                    continue
                src, dst = int(action["route_values"][b, r, 0]), int(action["route_values"][b, r, 1])
                changed = (int(state["carrying_hop_source_index"][b, fi]) != src or int(state["carrying_hop_destination_index"][b, fi]) != dst)
                nxt["carrying_hop_source_index"][b, fi] = src
                nxt["carrying_hop_destination_index"][b, fi] = dst
                if changed:
                    nxt["flow_route_revision"][b, fi] = state["flow_route_revision"][b, fi] + 1
                host = (state["task_agent_task_index"][b] == ti) & (state["task_agent_relation_type_index"][b] == 3) & state["task_agent_validity"][b]
                if bool(host.any()):
                    nxt["task_agent_agent_index"][b, torch.nonzero(host, as_tuple=False)[0, 0]] = dst
        total_work = state["task_work_total"].clamp_min(1e-9)
        nxt["task_progress"] = 1.0 - nxt["task_work_remaining"] / total_work
        completed_idx = LIFECYCLE_VOCAB.index("completed")
        return_idx = state.get("return_flow_index", torch.full_like(state["task_lifecycle_index"], -1))
        existing_return = return_idx >= 0
        requirement_known = state.get("task_return_requirement_known", existing_return) | existing_return
        requires_return = state.get("task_requires_return", existing_return) | existing_return
        mapped_return_done = (return_idx >= 0) & (~nxt["flow_presence"].gather(1, return_idx.clamp_min(0)))
        return_done = requirement_known & ((~requires_return) | mapped_return_done)
        computation_finished = nxt["task_work_remaining"] <= 0
        return_birth_required = requirement_known & requires_return & (return_idx < 0) & state["task_presence"]
        unresolved_return_requirement = (~requirement_known) & (return_idx < 0) & state["task_presence"]
        final_done = computation_finished & return_done & state["task_presence"]
        nxt["task_completed"] = final_done
        nxt["task_return_requirement_known"] = requirement_known
        nxt["task_requires_return"] = requires_return
        nxt["return_birth_required"] = return_birth_required
        nxt["final_completion_unresolved_by_return_requirement"] = computation_finished & unresolved_return_requirement
        nxt["final_completion_blocked_by_fixed_support"] = computation_finished & (return_birth_required | unresolved_return_requirement)
        nxt["task_released"] = state.get("task_released", state["task_presence"]).clone()
        nxt["task_lifecycle_index"] = torch.where(final_done, torch.full_like(state["task_lifecycle_index"], completed_idx), state["task_lifecycle_index"])
        if graph is not None and "dag_edges" in graph:
            for b in range(nxt["task_released"].shape[0]):
                for t in range(nxt["task_released"].shape[1]):
                    pred = graph["dag_validity"][b] & (graph["dag_edges"][b, :, 1] == t)
                    nxt["task_released"][b, t] = bool((~pred).all() or nxt["task_completed"][b, graph["dag_edges"][b, pred, 0].long()].all())
        # Rebind after every hop and route rule has settled; missing links stay -1.
        nxt["flow_comm_relation_index"].fill_(-1)
        for b in range(nxt["flow_comm_relation_index"].shape[0]):
            for f in range(nxt["flow_comm_relation_index"].shape[1]):
                if not bool(nxt["flow_presence"][b, f] & nxt["carrying_active"][b, f]):
                    continue
                src = int(nxt["carrying_hop_source_index"][b, f]); dst = int(nxt["carrying_hop_destination_index"][b, f])
                hits = torch.nonzero((state["comm_source_index"][b] == src) & (state["comm_target_index"][b] == dst) & state["comm_presence"][b] & state["comm_validity"][b], as_tuple=False)
                if len(hits): nxt["flow_comm_relation_index"][b, f] = hits[0, 0]
        trace = {"wireless": wireless, "delivered_bytes": delivered, "terminal_hop": terminal_hop, "e2e_reduction_bytes": e2e, "cpu_service": cpu_service}
        return nxt, trace

    def rebuild_graph(self, state: Mapping[str, torch.Tensor], previous: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        graph = {k: v.clone() for k, v in previous.items()}
        pos = state["position"]
        delta = pos[:, :, None, :] - pos[:, None, :, :]
        graph["physical_relation_features"] = torch.cat((delta, torch.linalg.vector_norm(delta, dim=-1, keepdim=True)), -1)
        validity = torch.zeros((pos.shape[0], pos.shape[1], pos.shape[1]), dtype=torch.bool, device=pos.device)
        topo = PhysicalTopologyConfig(mode=self.config.physical_topology_mode, radius_m=self.config.physical_radius_m, k=self.config.physical_k, self_loops=False)
        for b in range(pos.shape[0]):
            edges = _physical_edges(pos[b].detach().cpu().numpy(), (state["entity_presence"][b] & state.get("position_mask", torch.ones_like(pos[b], dtype=torch.bool)).all(-1)).detach().cpu().numpy(), topo)
            for src, dst in edges:
                validity[b, src, dst] = True
        graph["physical_relation_validity"] = validity
        graph["comm_csi"] = state["csi"]
        graph["comm_csi_mask"] = state["csi_mask"]
        endpoint_valid = state["entity_presence"].gather(1, state["comm_source_index"].clamp_min(0)) & state["entity_presence"].gather(1, state["comm_target_index"].clamp_min(0))
        graph["comm_validity"] = state["comm_validity"] & state["comm_presence"] & endpoint_valid
        graph["comm_source_index"] = state["comm_source_index"]
        graph["comm_target_index"] = state["comm_target_index"]
        graph["flow_presence"] = state["flow_presence"] & (state["flow_remaining"] > 0)
        graph["flow_source_index"] = state["flow_source_index"]
        graph["flow_destination_index"] = state["flow_destination_index"]
        graph["flow_status_index"] = state["flow_status_index"]
        graph["task_presence"] = state["task_presence"]
        graph["task_released"] = state.get("task_released", state["task_presence"])
        graph["task_completed"] = state.get("task_completed", torch.zeros_like(state["task_presence"]))
        graph["task_agent_task_index"] = state["task_agent_task_index"]
        graph["task_agent_agent_index"] = state["task_agent_agent_index"]
        task_valid = state["task_presence"].gather(1, state["task_agent_task_index"].clamp_min(0))
        agent_valid = state["entity_presence"].gather(1, state["task_agent_agent_index"].clamp_min(0))
        graph["task_agent_validity"] = state["task_agent_validity"] & task_valid & agent_valid
        graph["dag_edges"] = previous["dag_edges"]  # identity/static semantics preserved
        graph["dag_validity"] = previous["dag_validity"] & graph["task_presence"].gather(1, graph["dag_edges"][..., 0].clamp_min(0)) & graph["task_presence"].gather(1, graph["dag_edges"][..., 1].clamp_min(0))
        graph["dag_satisfied"] = graph["dag_validity"] & graph["task_completed"].gather(1, graph["dag_edges"][..., 0].clamp_min(0))
        graph["align_validity"] = state["entity_presence"]
        graph["geo_comm_validity"] = state["comm_presence"] & state["comm_wireless_mask"] & state["comm_validity"]
        return graph

    def rollout(self, z_pi: Mapping[str, Any], state: Mapping[str, torch.Tensor], graph: Mapping[str, torch.Tensor], actions: Sequence[Mapping[str, torch.Tensor]], *, future_target: Any = None, prior_mode: str = "mean", service_mode: str = "expectation", generator: torch.Generator | None = None) -> dict[str, Any]:
        if len(actions) == 0 or len(actions) > self.config.rollout_horizon:
            raise ValueError("actions must be non-empty and within rollout horizon")
        if (prior_mode == "sample" or service_mode == "sample") and generator is None:
            raise ValueError("stochastic rollout requires explicit generator")
        latent = self.initialize_latent(z_pi, state)
        states, graphs, latents, traces = [], [], [], []
        current_state, current_graph = state, graph
        for action in actions:
            latent, current_state, current_graph, trace = self.one_step(latent, current_state, current_graph, action, prior_mode=prior_mode, service_mode=service_mode, generator=generator)
            states.append(current_state); graphs.append(current_graph); latents.append(latent); traces.append(trace)
        recursive = len(states) < 2 or states[1]["position"].grad_fn is not None or not torch.equal(states[1]["position"], state["position"])
        return {"schema_version": SCHEMA_VERSION, "states": states, "graphs": graphs, "latents": latents, "traces": traces, "diagnostics": {"future_target_ignored": True, "prior_only_after_t": True, "fixed_current_object_support": True, "future_return_birth_supported": False, "recursive_state_feedback": bool(recursive), "expected_service_approximation": service_mode == "expectation", "stochastic_event_trace_included": all("wireless" in row["rule"] for row in traces)}}

    def synthetic_fixture(self, batch_size: int = 1) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        b, ne, nc, nf, nt, rb, de = batch_size, 4, 3, 2, 3, self.config.n_comm_rb, self.config.d_encoder
        entity_type = torch.tensor([[ENTITY_VEHICLE, ENTITY_UAV, 4, 4]]).expand(b, -1).clone()
        state = {
            "entity_presence": torch.ones((b, ne), dtype=torch.bool), "position_mask": torch.ones((b, ne, 3), dtype=torch.bool), "entity_type_index": entity_type,
            "vehicle_mask": entity_type == ENTITY_VEHICLE, "uav_mask": entity_type == ENTITY_UAV,
            "position": torch.arange(b * ne * 3, dtype=torch.float32).reshape(b, ne, 3), "speed": torch.ones((b, ne)), "acceleration": torch.zeros((b, ne)),
            "comm_presence": torch.ones((b, nc), dtype=torch.bool), "comm_base_validity": torch.ones((b, nc), dtype=torch.bool), "comm_validity": torch.ones((b, nc), dtype=torch.bool), "comm_type_index": torch.tensor([[COMM_WIRELESS, COMM_WIRELESS, COMM_WIRED]]).expand(b, -1).clone(),
            "comm_wireless_mask": torch.tensor([[True, True, False]]).expand(b, -1).clone(), "comm_wired_mask": torch.tensor([[False, False, True]]).expand(b, -1).clone(),
            "csi": torch.full((b, nc, rb), 80.0), "csi_mask": torch.tensor([[[True] * rb, [True] * rb, [False] * rb]]).expand(b, -1, -1).clone(), "rb_active_mask": torch.ones((b, nc, rb), dtype=torch.bool), "wired_capacity_mbps": torch.tensor([[0.0, 0.0, 100.0]]).expand(b, -1).clone(),
            "comm_source_index": torch.tensor([[0, 1, 2]]).expand(b, -1).clone(), "comm_target_index": torch.tensor([[1, 0, 3]]).expand(b, -1).clone(), "comm_source_type_index": torch.tensor([[ENTITY_VEHICLE, ENTITY_UAV, 4]]).expand(b, -1).clone(), "comm_target_type_index": torch.tensor([[ENTITY_UAV, ENTITY_VEHICLE, 4]]).expand(b, -1).clone(),
            "flow_presence": torch.ones((b, nf), dtype=torch.bool), "flow_known": torch.ones((b, nf), dtype=torch.bool), "flow_task_index": torch.tensor([[0, 1]]).expand(b, -1).clone(), "flow_total": torch.full((b, nf), 10000.0), "flow_remaining": torch.full((b, nf), 5000.0), "flow_source_index": torch.zeros((b, nf), dtype=torch.long), "flow_destination_index": torch.tensor([[2, 3]]).expand(b, -1).clone(), "flow_type_index": torch.ones((b, nf), dtype=torch.long), "flow_status_index": torch.full((b, nf), 2, dtype=torch.long),
            "flow_identity_index": torch.arange(nf)[None].expand(b, -1).clone(), "flow_route_revision": torch.zeros((b, nf), dtype=torch.long), "current_holder_index": torch.tensor([[0, 2]]).expand(b, -1).clone(), "current_hop_index": torch.zeros((b, nf), dtype=torch.long), "hop_progress": torch.zeros((b, nf)), "hop_remaining": torch.full((b, nf), 5000.0), "route_node_indices": torch.tensor([[[0, 1, 2, -1], [2, 3, -1, -1]]]).expand(b, -1, -1).clone(), "route_node_mask": torch.tensor([[[True, True, True, False], [True, True, False, False]]]).expand(b, -1, -1).clone(),
            "carrying_active": torch.ones((b, nf), dtype=torch.bool), "carrying_hop_source_index": torch.tensor([[0, 2]]).expand(b, -1).clone(), "carrying_hop_destination_index": torch.tensor([[1, 3]]).expand(b, -1).clone(), "flow_comm_relation_index": torch.tensor([[0, 2]]).expand(b, -1).clone(),
            "task_presence": torch.ones((b, nt), dtype=torch.bool), "task_work_total": torch.full((b, nt), 100.0), "task_work_remaining": torch.full((b, nt), 80.0), "task_progress": torch.full((b, nt), 0.2), "task_lifecycle_index": torch.ones((b, nt), dtype=torch.long),
            "task_completed": torch.zeros((b, nt), dtype=torch.bool), "task_released": torch.tensor([[True, False, False]]).expand(b, -1).clone(), "return_flow_index": torch.full((b, nt), -1, dtype=torch.long), "task_requires_return": torch.zeros((b, nt), dtype=torch.bool), "task_return_requirement_known": torch.ones((b, nt), dtype=torch.bool), "return_birth_required": torch.zeros((b, nt), dtype=torch.bool), "final_completion_unresolved_by_return_requirement": torch.zeros((b, nt), dtype=torch.bool), "final_completion_blocked_by_fixed_support": torch.zeros((b, nt), dtype=torch.bool),
            "task_agent_task_index": torch.tensor([[0, 1]]).expand(b, -1).clone(), "task_agent_agent_index": torch.tensor([[2, 3]]).expand(b, -1).clone(), "task_agent_relation_type_index": torch.tensor([[3, 4]]).expand(b, -1).clone(), "task_agent_validity": torch.ones((b, 2), dtype=torch.bool),
        }
        graph = {"physical_relation_features": torch.zeros((b, ne, ne, 4)), "physical_relation_validity": torch.ones((b, ne, ne), dtype=torch.bool), "comm_csi": state["csi"].clone(), "comm_csi_mask": state["csi_mask"].clone(), "comm_validity": state["comm_validity"].clone(), "comm_source_index": state["comm_source_index"].clone(), "comm_target_index": state["comm_target_index"].clone(), "comm_type_index": state["comm_type_index"].clone(), "flow_presence": state["flow_presence"].clone(), "flow_source_index": state["flow_source_index"].clone(), "flow_destination_index": state["flow_destination_index"].clone(), "flow_type_index": state["flow_type_index"].clone(), "flow_status_index": state["flow_status_index"].clone(), "task_presence": state["task_presence"].clone(), "task_released": state["task_released"].clone(), "task_completed": state["task_completed"].clone(), "task_agent_task_index": state["task_agent_task_index"].clone(), "task_agent_agent_index": state["task_agent_agent_index"].clone(), "task_agent_relation_type_index": state["task_agent_relation_type_index"].clone(), "task_agent_validity": state["task_agent_validity"].clone(), "dag_edges": torch.tensor([[[0, 1], [1, 2]]]).expand(b, -1, -1).clone(), "dag_validity": torch.ones((b, 2), dtype=torch.bool), "align_validity": state["entity_presence"].clone(), "geo_comm_validity": state["comm_wireless_mask"].clone()}
        zpi = {"physical": {"node_latent": torch.randn((b, ne, de))}, "information": {"agent_latent": torch.randn((b, ne, de)), "comm_relation_latent": torch.randn((b, nc, de)), "flow_relation_latent": torch.randn((b, nf, de)), "task_latent": torch.randn((b, nt, de))}}
        zeros4 = lambda n: torch.zeros((b, n, 4))
        allocation = torch.zeros((b, nc, rb), dtype=torch.bool); allocation[:, 0, 0] = True
        action = {"mobility_entity_index": torch.tensor([[1]]).expand(b, -1).clone(), "mobility_values": torch.tensor([[[0.0, 0.0, 2.0, 1.0]]]).expand(b, -1, -1).clone(), "comm_relation_index": torch.tensor([[0]]).expand(b, -1).clone(), "comm_values": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).expand(b, -1, -1).clone(), "comm_allocation_mask": allocation, "comp_agent_index": torch.tensor([[2]]).expand(b, -1).clone(), "comp_task_index": torch.tensor([[0]]).expand(b, -1).clone(), "comp_values": torch.tensor([[[10.0, 0.0, 0.0, 0.0]]]).expand(b, -1, -1).clone(), "route_task_index": torch.tensor([[0]]).expand(b, -1).clone(), "route_flow_index": torch.tensor([[0]]).expand(b, -1).clone(), "route_values": zeros4(1)}
        return zpi, state, graph, action


def world_model_digest(output: Mapping[str, Any]) -> str:
    def conv(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, Mapping):
            return {str(k): conv(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [conv(v) for v in value]
        return value
    return hashlib.sha256(json.dumps(conv(output), sort_keys=True, separators=(",", ":"), allow_nan=True).encode()).hexdigest()


def save_world_model_package(model: StructuredRSSMWorldModel, path: str | Path) -> None:
    torch.save({"schema_version": SCHEMA_VERSION, "config": asdict(model.config), "contract": model.contract, "state_dict": model.state_dict()}, Path(path))


def load_world_model_package(path: str | Path) -> StructuredRSSMWorldModel:
    package = torch.load(Path(path), map_location="cpu", weights_only=False)
    if package.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("world-model package schema mismatch")
    model = StructuredRSSMWorldModel(StructuredRSSMConfig(**package["config"]))
    model.load_state_dict(package["state_dict"])
    return model


def validate_world_model_acceptance(receipt: Mapping[str, Any]) -> dict[str, Any]:
    required = receipt.get("required_checks", {})
    scope = receipt.get("scope", {})
    forbidden = ("loss", "optimizer", "training", "gpu", "planner", "candidate_generation", "locked_test", "formal_dataset", "performance_claim")
    checks = {
        "all_required_present": set(required) == set(REQUIRED_ACCEPTANCE_CHECKS),
        "all_required_true": set(required) == set(REQUIRED_ACCEPTANCE_CHECKS) and all(required.get(name) is True for name in REQUIRED_ACCEPTANCE_CHECKS),
        "scope_explicit_false": all(scope.get(name) is False for name in forbidden),
        "untrained_evidence_class": receipt.get("evidence_class") == "UNTRAINED_DEVELOPMENT_WORLD_MODEL_EVIDENCE",
    }
    expected = bool(all(checks.values()))
    checks["reported_passed_matches_expected"] = receipt.get("passed") is expected
    return {"checks": checks, "expected_passed": expected, "passed": bool(all(checks.values()) and expected)}
