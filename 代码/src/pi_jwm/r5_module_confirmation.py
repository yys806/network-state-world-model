"""Controlled multi-seed confirmation matrix for PI-JWM coupling and graph modules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .r3_world_model import R3ReferenceConfig, R3ReferenceWorldModel
from .r4_module_registry import R4ModuleConfig
from .r4_world_model import (
    _CandidateGraphBackend,
    _CrossAttentionBackend,
    _GraphRSSMBackend,
)
from .r5_protocol import r5_combination_matrix
from .r5_world_model import R5WorldModel
from .r5_legacy_control import (
    LEGACY_ADAPTED_DYNAMICS,
    LegacyDirectedResidualBackend,
)


TRAINING_SEEDS = (20260803, 20260804, 20260805)


@dataclass(frozen=True)
class ConfirmationCandidate:
    combination_id: str
    label: str
    question: str
    components: dict[str, str]
    role: str
    configuration: dict[str, object]


@dataclass(frozen=True)
class ConfirmationRunSpec:
    combination_id: str
    training_seed: int
    reuse_existing: bool

    @property
    def run_id(self) -> str:
        return f"{self.combination_id}__seed_{self.training_seed}"


def _confirmation_configs(
    *,
    hidden_dim: int = 16,
    history_steps: int = 8,
    information_rate_mean: float | None = None,
    information_rate_scale: float | None = None,
) -> dict[str, tuple[str, str, str, R4ModuleConfig]]:
    control = r5_combination_matrix(
        hidden_dim=hidden_dim,
        history_steps=history_steps,
        information_rate_mean=information_rate_mean,
        information_rate_scale=information_rate_scale,
    )["B"].config
    return {
        "B": (
            "graph_rssm_control",
            "Reuse the existing three-seed Graph-RSSM control.",
            "reused_control",
            control,
        ),
        "F": (
            "graph_rssm_no_coupling",
            "Does removing CIP/CEP/CFL messages reduce rollout quality?",
            "controlled_module_ablation",
            replace(control, coupling="no_cross_graph_coupling_v1"),
        ),
        "G": (
            "graph_rssm_relation_constrained_attention",
            "Does relation-constrained attention improve over gated CIP/CEP/CFL coupling?",
            "controlled_module_ablation",
            replace(control, coupling="relation_constrained_cross_attention_v1"),
        ),
        "H": (
            "graph_rssm_edge_conditioned_mpnn",
            "Does the strongest R4 graph-encoder alternative improve over directed relational mean?",
            "controlled_module_ablation",
            replace(control, graph_encoder="edge_conditioned_relation_mpnn_v1"),
        ),
        "J": (
            "legacy_directed_dynamic_residual_v2_adapted",
            "Does the previous directed residual architecture remain competitive under the current R1/R2 protocol?",
            "architecture_control",
            replace(
                control,
                dynamics=LEGACY_ADAPTED_DYNAMICS,
                presence="soft_predicted_presence_v1",
            ),
        ),
    }


def build_confirmation_matrix(
    *,
    hidden_dim: int = 16,
    history_steps: int = 8,
    information_rate_mean: float | None = None,
    information_rate_scale: float | None = None,
) -> dict[str, ConfirmationCandidate]:
    """Build controlled module ablations plus one adapted architecture control."""

    return {
        combination_id: ConfirmationCandidate(
            combination_id=combination_id,
            label=label,
            question=question,
            components=config.component_names(),
            role=role,
            configuration=asdict(config),
        )
        for combination_id, (label, question, role, config) in _confirmation_configs(
            hidden_dim=hidden_dim,
            history_steps=history_steps,
            information_rate_mean=information_rate_mean,
            information_rate_scale=information_rate_scale,
        ).items()
    }


def build_confirmation_model(
    combination_id: str,
    *,
    hidden_dim: int = 16,
    history_steps: int = 8,
    information_rate_mean: float | None = None,
    information_rate_scale: float | None = None,
) -> R5WorldModel:
    """Build one executable confirmation composition under the current contract."""

    config = get_confirmation_config(
        combination_id,
        hidden_dim=hidden_dim,
        history_steps=history_steps,
        information_rate_mean=information_rate_mean,
        information_rate_scale=information_rate_scale,
    )
    backend_config = R3ReferenceConfig(
        hidden_dim=config.hidden_dim,
        history_steps=config.history_steps,
        use_cross_graph_coupling=config.coupling != "no_cross_graph_coupling_v1",
    )
    if config.dynamics == LEGACY_ADAPTED_DYNAMICS:
        return R5WorldModel(
            str(combination_id),
            config,
            LegacyDirectedResidualBackend(backend_config),
        )
    if config.coupling == "relation_constrained_cross_attention_v1":
        base = _CrossAttentionBackend(backend_config)
    elif config.graph_encoder == "edge_conditioned_relation_mpnn_v1":
        base = _CandidateGraphBackend(backend_config, mode="ecc")
    else:
        base = R3ReferenceWorldModel(backend_config)
    backend = _GraphRSSMBackend(backend_config, base=base)
    return R5WorldModel(str(combination_id), config, backend)


def get_confirmation_config(
    combination_id: str,
    *,
    hidden_dim: int = 16,
    history_steps: int = 8,
    information_rate_mean: float | None = None,
    information_rate_scale: float | None = None,
) -> R4ModuleConfig:
    """Return the frozen executable configuration without allocating a model."""

    try:
        return _confirmation_configs(
            hidden_dim=hidden_dim,
            history_steps=history_steps,
            information_rate_mean=information_rate_mean,
            information_rate_scale=information_rate_scale,
        )[str(combination_id)][3]
    except KeyError as error:
        raise ValueError(f"unknown confirmation combination: {combination_id}") from error


def build_confirmation_run_specs() -> tuple[ConfirmationRunSpec, ...]:
    return tuple(
        ConfirmationRunSpec(
            combination_id=combination_id,
            training_seed=seed,
            reuse_existing=combination_id == "B",
        )
        for combination_id in build_confirmation_matrix()
        for seed in TRAINING_SEEDS
    )


def write_confirmation_bundle(
    output_dir: str | Path,
    *,
    existing_r5_manifest_sha256: str,
    update_existing: bool = False,
    information_rate_mean: float | None = None,
    information_rate_scale: float | None = None,
) -> None:
    output = Path(output_dir)
    expected_files = {"matrix.json", "run_specs.json", "summary.json", "README.md", "manifest.json"}
    if output.exists() and any(output.iterdir()):
        if not update_existing:
            raise FileExistsError(f"confirmation bundle directory must be empty: {output}")
        unexpected = {path.name for path in output.iterdir()} - expected_files
        if unexpected:
            raise ValueError(
                "confirmation bundle contains unexpected files: "
                + ", ".join(sorted(unexpected))
            )
    output.mkdir(parents=True, exist_ok=True)
    matrix = build_confirmation_matrix(
        information_rate_mean=information_rate_mean,
        information_rate_scale=information_rate_scale,
    )
    specs = build_confirmation_run_specs()
    matrix_payload = {
        key: {
            "combination_id": value.combination_id,
            "label": value.label,
            "question": value.question,
            "components": value.components,
            "role": value.role,
            "configuration": value.configuration,
        }
        for key, value in matrix.items()
    }
    spec_payload = [
        {
            "combination_id": spec.combination_id,
            "training_seed": spec.training_seed,
            "reuse_existing": spec.reuse_existing,
        }
        for spec in specs
    ]
    summary = {
        "schema_version": "PIJWM-R5-Module-Confirmation-Protocol-v1",
        "candidate_count": len(matrix),
        "training_seed_count": len(TRAINING_SEEDS),
        "total_run_count": len(specs),
        "reused_run_count": sum(spec.reuse_existing for spec in specs),
        "new_gpu_run_count": sum(not spec.reuse_existing for spec in specs),
        "locked_test_accessed": False,
        "selection_status": "planned_controlled_confirmation",
    }

    def write_json(name: str, payload: object) -> None:
        (output / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    write_json("matrix.json", matrix_payload)
    write_json("run_specs.json", spec_payload)
    write_json("summary.json", summary)
    (output / "README.md").write_text(
        "# PI-JWM controlled module confirmation\n\n"
        "Reuse B from R5 and train only F/G/H/J with the frozen three seeds. "
        "F removes cross-graph coupling, G uses relation-constrained cross-attention, "
        "H changes only the graph encoder to edge-conditioned MPNN, and J adapts "
        "the previous directed residual architecture to the current R1 graph semantics "
        "and R2 prediction interface. Locked-test remains sealed.\n",
        encoding="utf-8",
    )
    files = {}
    for path in sorted(output.iterdir()):
        if path.name == "manifest.json":
            continue
        content = path.read_bytes()
        files[path.name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    write_json(
        "manifest.json",
        {
            "schema_version": summary["schema_version"],
            "existing_r5_manifest_sha256": existing_r5_manifest_sha256,
            "manifest_entry_count": len(files),
            "files": files,
        },
    )


__all__ = [
    "ConfirmationCandidate",
    "ConfirmationRunSpec",
    "TRAINING_SEEDS",
    "build_confirmation_matrix",
    "build_confirmation_model",
    "build_confirmation_run_specs",
    "get_confirmation_config",
    "write_confirmation_bundle",
]
