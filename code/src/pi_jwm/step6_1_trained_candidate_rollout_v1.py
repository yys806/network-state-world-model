"""CPU candidate rollout on the frozen Step 5.6C world model.

This module deliberately has no score, objective, optimizer or action choice.
It consumes only current observation input and a Planner v1 domain candidate.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import torch

from pi_jwm.step5_2_training_loop_v1 import _stack_actions
from pi_jwm.step6_0a_candidate_generation_v1 import (
    CandidateActionSequence, PlannerCandidateContext, validate_step,
)
from pi_jwm.step6_0c_planner_action_domain_v1 import PlannerActionDomainContext, require_domain_admissible


def clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_tree(item) for item in value)
    return copy.deepcopy(value)


def fingerprint(value: Any) -> str:
    """Content hash with keys, shapes, dtypes and tensor bytes (CPU only)."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item):
                digest.update(f"key:{key}:".encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"sequence:{len(item)}:".encode())
            for child in item:
                visit(child)
        elif item is None:
            digest.update(b"none")
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def _batch_tree(values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=0)
    if isinstance(first, Mapping):
        return {key: _batch_tree([value[key] for value in values]) for key in first}
    if first is None:
        return None
    raise TypeError(f"cannot batch {type(first).__name__}")


def _one(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[index:index + 1] if value.ndim and value.shape[0] == batch_size else value
    if isinstance(value, Mapping):
        return {key: _one(item, index, batch_size) for key, item in value.items()}
    return value


def compile_candidate_step(sequence: CandidateActionSequence, context: PlannerCandidateContext,
                           state: Mapping[str, torch.Tensor], step_index: int, *,
                           planner_domain_context: PlannerActionDomainContext) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Compile on this step's causal state through the unchanged formal adapter."""
    if not 0 <= step_index < sequence.horizon:
        raise ValueError("candidate step outside horizon")
    require_domain_admissible(sequence, context, planner_domain_context)
    validate_step(sequence.steps[step_index], replace(context, current_state=state))
    from build_step5_1d_unified_model_chain_v1 import build_action
    sample = {"static": copy.deepcopy(context.static), "history": copy.deepcopy(list(context.history)),
              "future_action": [step.frame() for step in sequence.steps[:step_index + 1]]}
    tensor, mapping = build_action(sample, dict(state), step_index)
    for declared, row in zip(sequence.steps[step_index].comm, mapping["comm"]):
        ri = int(row["relation_index"])
        if ri < 0 or ri != int(declared["relation_index"]):
            raise ValueError("Comm relation identity unresolved")
        if not (bool(state["comm_presence"][0, ri]) and bool(state["comm_validity"][0, ri])):
            raise ValueError("Comm relation is not current and valid")
    if not (bool(torch.isfinite(tensor["mobility_values"]).all()) and
            bool(torch.isfinite(tensor["comp_values"]).all())):
        raise ValueError("nonfinite action tensor")
    return tensor, mapping


@dataclass(frozen=True)
class PreparedRolloutAnchor:
    sample_id: str
    context: PlannerCandidateContext
    domain: PlannerActionDomainContext
    state: Mapping[str, torch.Tensor]
    graph: Mapping[str, torch.Tensor]
    z_pi: Mapping[str, Any]
    latent: Mapping[str, Any]
    fingerprints: Mapping[str, str]


@dataclass(frozen=True)
class CandidateRolloutTrace:
    candidate_id: str
    initial_fingerprints: Mapping[str, str]
    input_fingerprints: tuple[Mapping[str, str], ...]
    output_fingerprints: tuple[Mapping[str, str], ...]
    actions: tuple[Mapping[str, torch.Tensor], ...]
    mappings: tuple[Mapping[str, Any], ...]
    latents: tuple[Mapping[str, Any], ...]
    states: tuple[Mapping[str, torch.Tensor], ...]
    graphs: tuple[Mapping[str, torch.Tensor], ...]
    model_traces: tuple[Mapping[str, Any], ...]


def prepare_anchor(model: torch.nn.Module, encoder: torch.nn.Module,
                   tensor: Mapping[str, Any], graph_input: Mapping[str, Any],
                   state: Mapping[str, torch.Tensor], graph: Mapping[str, torch.Tensor],
                   context: PlannerCandidateContext, domain: PlannerActionDomainContext,
                   sample_id: str) -> PreparedRolloutAnchor:
    if model.training or encoder.training:
        raise ValueError("frozen rollout requires model.eval() and encoder.eval()")
    if torch.is_grad_enabled():
        raise ValueError("frozen rollout requires torch.no_grad()")
    z_pi = encoder(tensor, graph_input)
    latent = model.initialize_latent(z_pi, state, posterior_mode="mean", generator=None)
    if latent["posterior"] is None:
        raise AssertionError("current observation posterior was not used")
    fingerprints = {"state": fingerprint(state), "graph": fingerprint(graph),
                    "z_pi": fingerprint(z_pi), "latent": fingerprint(latent)}
    fingerprints["anchor"] = fingerprint(fingerprints)
    return PreparedRolloutAnchor(sample_id, context, domain, clone_tree(state), clone_tree(graph),
                                 clone_tree(z_pi), clone_tree(latent), fingerprints)


def _run(model: torch.nn.Module, anchor: PreparedRolloutAnchor,
         candidates: Sequence[CandidateActionSequence], *, batched: bool,
         prior_mode: str, service_mode: str,
         generators: Sequence[torch.Generator | None]) -> tuple[CandidateRolloutTrace, ...]:
    if not candidates or len(generators) != len(candidates):
        raise ValueError("candidate and generator counts must agree")
    if any(candidate.horizon != candidates[0].horizon for candidate in candidates):
        raise ValueError("batched candidates need a common horizon")
    if model.training or torch.is_grad_enabled():
        raise ValueError("rollout requires model.eval() and torch.no_grad()")
    if batched and (prior_mode != "mean" or service_mode != "expectation"):
        raise ValueError("stochastic candidates use paired serial generators")
    if not batched and len(candidates) != 1:
        raise ValueError("serial rollout accepts one candidate")
    states = [clone_tree(anchor.state) for _ in candidates]
    graphs = [clone_tree(anchor.graph) for _ in candidates]
    latents = [clone_tree(anchor.latent) for _ in candidates]
    initial = [{"anchor": anchor.fingerprints["anchor"], "state": fingerprint(s),
                "graph": fingerprint(g), "latent": fingerprint(l)}
               for s, g, l in zip(states, graphs, latents)]
    histories: list[dict[str, list[Any]]] = [dict(inputs=[], outputs=[], actions=[], mappings=[],
                                                 latents=[], states=[], graphs=[], traces=[])
                                           for _ in candidates]
    for step_index in range(candidates[0].horizon):
        actions, mappings = [], []
        for i, candidate in enumerate(candidates):
            histories[i]["inputs"].append({"state": fingerprint(states[i]),
                                           "graph": fingerprint(graphs[i]),
                                           "latent": fingerprint(latents[i])})
            action, mapping = compile_candidate_step(candidate, anchor.context, states[i], step_index,
                                                      planner_domain_context=anchor.domain)
            actions.append(action); mappings.append(mapping)
        if batched:
            batch_action = _stack_actions(actions)
            batch_latent, batch_state, batch_graph, batch_trace = model.one_step(
                _batch_tree(latents), _batch_tree(states), _batch_tree(graphs), batch_action,
                prior_mode="mean", service_mode="expectation", generator=None)
            count = len(candidates)
            next_values = [(_one(batch_latent, i, count), _one(batch_state, i, count),
                            _one(batch_graph, i, count), _one(batch_trace, i, count))
                           for i in range(count)]
        else:
            next_values = [model.one_step(latents[0], states[0], graphs[0], actions[0],
                                          prior_mode=prior_mode, service_mode=service_mode,
                                          generator=generators[0])]
        for i, (latent, state, graph, trace) in enumerate(next_values):
            if batched:
                # _stack_actions pads variable-width Comp rows. The rule trace
                # reports service per action row, so remove only those padded
                # diagnostic rows when returning the per-candidate trace.
                trace["rule"]["cpu_service"] = trace["rule"]["cpu_service"][
                    :, :actions[i]["comp_values"].shape[1]]
            latents[i], states[i], graphs[i] = latent, state, graph
            history = histories[i]
            history["outputs"].append({"state": fingerprint(state), "graph": fingerprint(graph),
                                       "latent": fingerprint(latent)})
            history["actions"].append(actions[i]); history["mappings"].append(mappings[i])
            history["latents"].append(latent); history["states"].append(state)
            history["graphs"].append(graph); history["traces"].append(trace)
    return tuple(CandidateRolloutTrace(candidate.candidate_id, initial[i],
                 tuple(history["inputs"]), tuple(history["outputs"]), tuple(history["actions"]),
                 tuple(history["mappings"]), tuple(history["latents"]), tuple(history["states"]),
                 tuple(history["graphs"]), tuple(history["traces"]))
                 for i, (candidate, history) in enumerate(zip(candidates, histories)))


def rollout_candidate_sequential(model: torch.nn.Module, anchor: PreparedRolloutAnchor,
                                 candidate: CandidateActionSequence, *, prior_mode: str = "mean",
                                 service_mode: str = "expectation",
                                 generator: torch.Generator | None = None) -> CandidateRolloutTrace:
    return _run(model, anchor, [candidate], batched=False, prior_mode=prior_mode,
                service_mode=service_mode, generators=[generator])[0]


def rollout_candidate_batch(model: torch.nn.Module, anchor: PreparedRolloutAnchor,
                            candidates: Sequence[CandidateActionSequence]) -> tuple[CandidateRolloutTrace, ...]:
    return _run(model, anchor, candidates, batched=True, prior_mode="mean",
                service_mode="expectation", generators=[None] * len(candidates))


def max_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max()) if left.numel() else 0.0


def action_response_summary(control: CandidateRolloutTrace,
                            intervention: CandidateRolloutTrace) -> list[dict[str, Any]]:
    rows = []
    for horizon, (a, b) in enumerate(zip(control.latents, intervention.latents), 1):
        left, right = control.model_traces[horizon - 1], intervention.model_traces[horizon - 1]
        rows.append({"horizon": horizon,
                     "hidden_delta": {name: max_delta(a["h"][name], b["h"][name]) for name in a["h"]},
                     "prior_delta": {family: {field: max_delta(a["prior"][family][field], b["prior"][family][field])
                                                for field in ("mean", "log_std")}
                                     for family in ("physical", "communication")},
                     "decoder_delta": {name: max_delta(left["learned"][name], right["learned"][name])
                                       for name in ("vehicle_motion", "csi")},
                     "state_different": control.output_fingerprints[horizon - 1]["state"] != intervention.output_fingerprints[horizon - 1]["state"],
                     "graph_different": control.output_fingerprints[horizon - 1]["graph"] != intervention.output_fingerprints[horizon - 1]["graph"]})
    return rows


def audit_recursive_link(trace: CandidateRolloutTrace, horizon_index: int,
                         *, substituted_input: Mapping[str, str] | None = None) -> None:
    """Reject a horizon that consumes a different prior state, graph or latent."""
    if not 1 <= horizon_index < len(trace.input_fingerprints):
        raise ValueError("recursive link requires H2 or later")
    actual = trace.input_fingerprints[horizon_index] if substituted_input is None else substituted_input
    if actual != trace.output_fingerprints[horizon_index - 1]:
        raise ValueError("recursive input differs from previous predicted output")
