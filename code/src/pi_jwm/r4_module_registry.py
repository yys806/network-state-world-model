"""Frozen R4 candidate registry and controlled single-module configurations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

from .r3_world_model import REFERENCE_COMPONENTS


R4_REGISTRY_SCHEMA = "PIJWM-R4-Module-Registry-v1"
MODULE_FAMILIES = (
    "field_encoder",
    "graph_encoder",
    "coupling",
    "dynamics",
    "head",
    "dag",
    "presence",
)
KNOWN_STATUSES = {"reference", "planned", "executable", "reserve", "deferred"}


@dataclass(frozen=True)
class CandidateSpec:
    family: str
    name: str
    status: str
    evidence: str
    question: str

    def __post_init__(self) -> None:
        if self.family not in MODULE_FAMILIES:
            raise ValueError(f"unknown R4 module family: {self.family}")
        if self.status not in KNOWN_STATUSES:
            raise ValueError(f"unknown R4 candidate status: {self.status}")
        if not self.name or not self.evidence or not self.question:
            raise ValueError("R4 candidate name, evidence, and question must be non-empty")


@dataclass(frozen=True)
class R4ModuleConfig:
    hidden_dim: int = 16
    history_steps: int = 8
    field_encoder: str = REFERENCE_COMPONENTS["field_encoder"]
    graph_encoder: str = REFERENCE_COMPONENTS["graph_encoder"]
    coupling: str = REFERENCE_COMPONENTS["coupling"]
    dynamics: str = REFERENCE_COMPONENTS["dynamics"]
    head: str = REFERENCE_COMPONENTS["head"]
    dag: str = "dag_summary_v1"
    presence: str = "fixed_observed_presence_v1"
    information_rate_mean: float | None = None
    information_rate_scale: float | None = None

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.history_steps <= 0:
            raise ValueError("hidden_dim and history_steps must be positive")
        has_mean = self.information_rate_mean is not None
        has_scale = self.information_rate_scale is not None
        if has_mean != has_scale:
            raise ValueError("rate normalization mean and scale must be provided together")
        if has_scale and float(self.information_rate_scale) <= 0.0:
            raise ValueError("rate normalization scale must be positive")
        if self.head == "hurdle_active_rate_v1" and not (has_mean and has_scale):
            raise ValueError("hurdle head requires train-only rate normalization")

    def component_names(self) -> dict[str, str]:
        return {family: str(getattr(self, family)) for family in MODULE_FAMILIES}


_SPECS = (
    CandidateSpec(
        "field_encoder",
        REFERENCE_COMPONENTS["field_encoder"],
        "reference",
        "PI-JWM R3 executable reference",
        "Does the frozen masked field interface execute unchanged?",
    ),
    CandidateSpec(
        "field_encoder",
        "symlog_masked_mlp_v1",
        "executable",
        "DreamerV3, Nature 2025, doi:10.1038/s41586-025-08744-2",
        "Does symlog reduce scale imbalance without breaking missing-value masks?",
    ),
    CandidateSpec(
        "field_encoder",
        "simnorm_masked_mlp_v1",
        "executable",
        "TD-MPC2, ICLR 2024, arXiv:2310.16828",
        "Does SimNorm stabilize representations under the frozen field budget?",
    ),
    CandidateSpec(
        "graph_encoder",
        REFERENCE_COMPONENTS["graph_encoder"],
        "reference",
        "PI-JWM R3 directed relation-aware reference",
        "Does the frozen directed message-passing interface execute unchanged?",
    ),
    CandidateSpec(
        "graph_encoder",
        "rgcn_v1",
        "executable",
        "R-GCN, ESWC 2018, doi:10.1007/978-3-319-93417-4_38",
        "Do discrete relation transforms improve typed directed graph encoding?",
    ),
    CandidateSpec(
        "graph_encoder",
        "edge_conditioned_relation_mpnn_v1",
        "executable",
        "Edge-Conditioned Convolution, CVPR 2017",
        "Do continuous edge attributes improve relation-aware messages?",
    ),
    CandidateSpec(
        "graph_encoder",
        "gatv2_v1",
        "reserve",
        "GATv2, ICLR 2022",
        "Is dynamic attention needed after simpler relation encoders are tested?",
    ),
    CandidateSpec(
        "coupling",
        REFERENCE_COMPONENTS["coupling"],
        "reference",
        "PI-JWM R3 explicit CIP/CEP/CFL gated reference",
        "Does the frozen explicit relation coupler execute unchanged?",
    ),
    CandidateSpec(
        "coupling",
        "no_cross_graph_coupling_v1",
        "executable",
        "PI-JWM structural lower-bound control",
        "Does removing all cross-graph messages reduce rollout quality?",
    ),
    CandidateSpec(
        "coupling",
        "relation_constrained_cross_attention_v1",
        "executable",
        "Coupled JEPA resource planning, IEEE TWC 2026, doi:10.1109/TWC.2025.3644600",
        "Can attention improve fusion while remaining restricted to CIP/CEP/CFL?",
    ),
    CandidateSpec(
        "coupling",
        "directional_jepa_v1",
        "reserve",
        "Coupled JEPA resource planning, IEEE TWC 2026, doi:10.1109/TWC.2025.3644600",
        "Does directional latent prediction add value beyond explicit coupling?",
    ),
    CandidateSpec(
        "dynamics",
        REFERENCE_COMPONENTS["dynamics"],
        "reference",
        "PI-JWM R3 deterministic Graph-GRU reference",
        "Does the frozen deterministic action-conditioned rollout execute unchanged?",
    ),
    CandidateSpec(
        "dynamics",
        "graph_rssm_v1",
        "executable",
        "PlaNet, ICML 2019; DreamerV3, Nature 2025",
        "Does a prior/posterior stochastic state improve calibrated long rollout?",
    ),
    CandidateSpec(
        "dynamics",
        "transformer_dynamics_v1",
        "deferred",
        "TransDreamer withdrawn record; insufficient PI-JWM long-memory evidence",
        "Is a sequence transformer justified by a measured long-memory bottleneck?",
    ),
    CandidateSpec(
        "head",
        REFERENCE_COMPONENTS["head"],
        "reference",
        "PI-JWM R3 typed deterministic heads",
        "Do the frozen typed heads execute unchanged?",
    ),
    CandidateSpec(
        "head",
        "heteroscedastic_typed_v1",
        "executable",
        "Kendall and Gal, NeurIPS 2017",
        "Does learned aleatoric variance improve NLL and interval calibration?",
    ),
    CandidateSpec(
        "head",
        "hurdle_active_rate_v1",
        "executable",
        "PI-JWM R2 activity plus active-only rate protocol",
        "Does separating activity from positive rate prevent zero-dominated regression?",
    ),
    CandidateSpec(
        "head",
        "deep_ensemble_v1",
        "deferred",
        "Deep Ensembles, NeurIPS 2017",
        "Does between-model uncertainty help after R4 fixes a single architecture?",
    ),
    CandidateSpec(
        "dag",
        "dag_summary_v1",
        "reference",
        "PI-JWM R3 parent-count, unfinished-parent-count, release-ready summary",
        "Does the frozen three-dimensional DAG summary execute unchanged?",
    ),
    CandidateSpec(
        "dag",
        "explicit_dag_message_passing_v1",
        "executable",
        "DAG offloading, IEEE IoT-J 2021, doi:10.1109/JIOT.2020.3030926",
        "Do explicit acyclic dependency messages improve task evolution prediction?",
    ),
    CandidateSpec(
        "presence",
        "fixed_observed_presence_v1",
        "reference",
        "PI-JWM R3 last-observed-presence rollout reference",
        "Does the frozen observed-presence rollout execute unchanged?",
    ),
    CandidateSpec(
        "presence",
        "soft_predicted_presence_v1",
        "executable",
        "VGRNN, NeurIPS 2019",
        "Does recursively weighted soft presence improve dynamic-topology rollout?",
    ),
    CandidateSpec(
        "presence",
        "calibrated_hard_presence_v1",
        "deferred",
        "VGRNN precedent plus PI-JWM calibration-only threshold rule",
        "Does a calibrated hard topology help only after soft recursion is stable?",
    ),
)


def candidate_registry() -> dict[str, CandidateSpec]:
    registry = {spec.name: spec for spec in _SPECS}
    if len(registry) != len(_SPECS):
        raise RuntimeError("R4 candidate names must be globally unique")
    return registry


def candidate_matrix() -> list[dict[str, str]]:
    fields = ("family", "name", "status", "evidence", "question")
    return [
        {field: str(asdict(spec)[field]) for field in fields}
        for spec in _SPECS
    ]


def reference_r4_config(
    *,
    hidden_dim: int = 16,
    history_steps: int = 8,
    information_rate_mean: float | None = None,
    information_rate_scale: float | None = None,
) -> R4ModuleConfig:
    return R4ModuleConfig(
        hidden_dim=hidden_dim,
        history_steps=history_steps,
        information_rate_mean=information_rate_mean,
        information_rate_scale=information_rate_scale,
    )


def _spec_for(family: str, name: str) -> CandidateSpec:
    if family not in MODULE_FAMILIES:
        raise ValueError(f"unknown R4 module family: {family}")
    spec = candidate_registry().get(name)
    if spec is None or spec.family != family:
        raise ValueError(f"unknown R4 candidate for {family}: {name}")
    return spec


def make_single_module_config(
    family: str,
    name: str,
    *,
    hidden_dim: int = 16,
    history_steps: int = 8,
    information_rate_mean: float | None = None,
    information_rate_scale: float | None = None,
) -> R4ModuleConfig:
    _spec_for(family, name)
    return replace(
        reference_r4_config(
            hidden_dim=hidden_dim,
            history_steps=history_steps,
            information_rate_mean=information_rate_mean,
            information_rate_scale=information_rate_scale,
        ),
        **{family: name},
    )


def validate_controlled_config(
    config: R4ModuleConfig,
    *,
    allow_statuses: Iterable[str] = ("reference", "executable"),
) -> None:
    allowed = set(allow_statuses)
    unknown_statuses = allowed - KNOWN_STATUSES
    if unknown_statuses:
        raise ValueError(f"unknown allowed R4 statuses: {sorted(unknown_statuses)}")
    reference = reference_r4_config(
        hidden_dim=config.hidden_dim,
        history_steps=config.history_steps,
    )
    changed = []
    for family, name in config.component_names().items():
        spec = _spec_for(family, name)
        if spec.status not in allowed:
            raise ValueError(
                f"R4 candidate is not allowed in this phase: {family}={name} "
                f"has status {spec.status}"
            )
        if name != getattr(reference, family):
            changed.append(family)
    if len(changed) > 1:
        raise ValueError(
            "R4 controlled screening may change only one module family; changed: "
            + ", ".join(changed)
        )


def assert_executable_config(config: R4ModuleConfig) -> None:
    try:
        validate_controlled_config(
            config,
            allow_statuses={"reference", "executable"},
        )
    except ValueError as error:
        if "not allowed" in str(error):
            raise ValueError(f"R4 candidate is not executable: {error}") from error
        raise


__all__ = [
    "CandidateSpec",
    "KNOWN_STATUSES",
    "MODULE_FAMILIES",
    "R4ModuleConfig",
    "R4_REGISTRY_SCHEMA",
    "assert_executable_config",
    "candidate_matrix",
    "candidate_registry",
    "make_single_module_config",
    "reference_r4_config",
    "validate_controlled_config",
]
