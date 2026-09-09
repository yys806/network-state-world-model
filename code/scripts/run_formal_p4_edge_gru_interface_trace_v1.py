"""Read-only CPU trace of the frozen P4 edge-GRU interface (no intervention)."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "src"))

from run_formal_dual_graph_cpu_smoke_v1 import _manifest, _sha256
from run_formal_dual_graph_gpu_train_v1 import _prediction
from run_formal_p4_link_recall_diagnosis_v1 import (
    CANONICAL_TENSOR_MANIFEST_SHA256, FROZEN_SENTINEL_CHECKPOINT_SHA256,
    FROZEN_SENTINEL_SAMPLE_IDS_SHA256, POS_WEIGHT, RAW_THRESHOLD,
    _prepare_frozen_link_validation, _previous_activity_and_mask,
)

SCHEMA_VERSION = "PI-JWM-p4-edge-gru-interface-trace-v1"
HORIZONS = tuple(range(1, 21))
FROZEN_TP_FP_FN = (1902, 557, 5855)
FROZEN_HOOK_MODULE_FULL_NAME = (
    "pi_jwm.formal_dual_graph_world_model_v1."
    "FormalDualGraphWorldModel.edge_transition"
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _fraction_at_least(value: Any, threshold: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= threshold
    )


def _reconstruct_gru_cell(cell: torch.nn.GRUCell, input: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return GRUCell's r/z/n reconstruction and its update gate, without calling it."""
    if not isinstance(cell, torch.nn.GRUCell):
        raise ValueError("cell must be torch.nn.GRUCell")
    i_r, i_z, i_n = F.linear(input, cell.weight_ih, cell.bias_ih).chunk(3, -1)
    h_r, h_z, h_n = F.linear(hidden, cell.weight_hh, cell.bias_hh).chunk(3, -1)
    reset, update = torch.sigmoid(i_r + h_r), torch.sigmoid(i_z + h_z)
    candidate = torch.tanh(i_n + reset * h_n)
    return candidate + update * (hidden - candidate), update


@contextlib.contextmanager
def edge_gru_readonly_trace(model: torch.nn.Module) -> Iterator[dict[str, Any]]:
    """Capture edge-GRU calls while preserving the model output exactly."""
    cell = getattr(model, "edge_transition", None)
    if not isinstance(cell, torch.nn.GRUCell):
        raise ValueError("model.edge_transition must be torch.nn.GRUCell")
    before = set(cell._forward_hooks)
    audit: dict[str, Any] = {
        "module": "edge_transition",
        "module_full_name": f"{type(model).__module__}.{type(model).__qualname__}.edge_transition",
        "actual_hook_calls": 0,
        "hook_removed": False,
        "records": [],
        "max_gru_reconstruction_error": 0.0,
    }
    def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if len(args) != 2 or not all(isinstance(x, torch.Tensor) for x in args) or not isinstance(output, torch.Tensor):
            raise RuntimeError("edge_transition GRUCell hook inputs are invalid")
        message, hidden = args
        reconstructed, update = _reconstruct_gru_cell(cell, message, hidden)
        error = float((reconstructed - output).abs().max().item())
        audit["max_gru_reconstruction_error"] = max(float(audit["max_gru_reconstruction_error"]), error)
        audit["records"].append({"input_message": message.detach().clone(), "hidden_before": hidden.detach().clone(), "hidden_after": output.detach().clone(), "update_gate": update.detach().clone(), "reconstruction_error": error})
        audit["actual_hook_calls"] += 1
        return None
    handle = cell.register_forward_hook(hook)
    try:
        yield audit
    finally:
        handle.remove()
        audit["hook_removed"] = True
        if set(cell._forward_hooks) != before:
            raise RuntimeError("edge GRU trace hook removal left residual hooks")


def _q(value: torch.Tensor) -> dict[str, float] | str:
    if value.numel() == 0:
        return "not_computable"
    value = value.detach().cpu().to(torch.float64)
    return {f"q{int(p * 100)}": float(torch.quantile(value, p).item()) for p in (0., .1, .5, .9, 1.)}


def _bucket(pre: torch.Tensor, post: torch.Tensor, labels: torch.Tensor, prior: torch.Tensor, gate: torch.Tensor, selected: torch.Tensor) -> dict[str, Any] | str:
    if not selected.any().item():
        return "not_computable"
    pre, post, labels, gate = pre[selected], post[selected], labels[selected].bool(), gate[selected]
    pre_decision, post_decision = torch.sigmoid(pre) >= RAW_THRESHOLD, torch.sigmoid(post) >= RAW_THRESHOLD
    positives = labels
    pp, pn = positives & pre_decision & post_decision, positives & pre_decision & ~post_decision
    np, nn = positives & ~pre_decision & post_decision, positives & ~pre_decision & ~post_decision
    fn = pn | nn
    fn_count = int(fn.sum().item())
    return {"positive_count": int(positives.sum().item()), "false_negative_count": fn_count,
            "pre_positive_post_positive": int(pp.sum().item()), "pre_positive_post_negative": int(pn.sum().item()),
            "pre_negative_post_positive": int(np.sum().item()), "pre_negative_post_negative": int(nn.sum().item()),
            "incoming_below_count": int(nn.sum().item()), "downcross_count": int(pn.sum().item()),
            "incoming_below_fraction": float(nn.sum().item() / fn_count) if fn_count else "not_computable",
            "downcross_fraction": float(pn.sum().item() / fn_count) if fn_count else "not_computable",
            "pre_logit": _q(pre[fn]), "post_logit": _q(post[fn]), "delta_logit": _q((post-pre)[fn]), "update_gate_edge_mean": _q(gate[fn])}


def _check_vectors(values: Sequence[torch.Tensor], name: str) -> list[torch.Tensor]:
    if not isinstance(values, Sequence) or len(values) != 20:
        raise ValueError(f"{name} must provide exactly 20 horizons")
    output = []
    for i, value in enumerate(values, 1):
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or not value.numel() or not torch.isfinite(value).all().item():
            raise ValueError(f"{name} h{i} must be finite non-empty 1D tensor")
        output.append(value.detach().cpu())
    return output


def build_edge_gru_interface_trace(pre_update_logits: Sequence[torch.Tensor], post_update_logits: Sequence[torch.Tensor], official_logits: Sequence[torch.Tensor], update_gate_edge_means: Sequence[torch.Tensor], labels: Sequence[torch.Tensor], previous_link_activity: Sequence[torch.Tensor], provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Build no-fit interface evidence from 20 aligned horizons."""
    collections = [_check_vectors(v, n) for v, n in zip((pre_update_logits, post_update_logits, official_logits, update_gate_edge_means, labels, previous_link_activity), ("pre_update_logits", "post_update_logits", "official_logits", "update_gate_edge_means", "labels", "previous_link_activity"))]
    pre_rows, post_rows, official_rows, gate_rows, label_rows, prior_rows = collections
    for i, rows in enumerate(zip(pre_rows, post_rows, official_rows, gate_rows, label_rows, prior_rows), 1):
        if len({x.shape for x in rows}) != 1: raise ValueError(f"trace inputs are not aligned at h{i}")
        if float((rows[1] - rows[2]).abs().max()) > 1e-6: raise ValueError("post_update_logits and official_logits drift")
        if not torch.logical_or(rows[4] == 0, rows[4] == 1).all().item() or not torch.logical_or(torch.logical_or(rows[5] == -1, rows[5] == 0), rows[5] == 1).all().item(): raise ValueError("labels or previous_link_activity are invalid")
    def one(pre: torch.Tensor, post: torch.Tensor, label: torch.Tensor, prior: torch.Tensor, gate: torch.Tensor) -> dict[str, Any]:
        positive = label == 1
        return {"all_positive": _bucket(pre, post, label, prior, gate, positive), "continued_active": _bucket(pre, post, label, prior, gate, positive & (prior == 1)), "newly_active": _bucket(pre, post, label, prior, gate, positive & (prior == 0)), "previous_unobserved": _bucket(pre, post, label, prior, gate, positive & (prior == -1))}
    per = {f"h{i}": one(*rows) for i, rows in enumerate(zip(pre_rows, post_rows, label_rows, prior_rows, gate_rows), 1)}
    overall = one(torch.cat(pre_rows), torch.cat(post_rows), torch.cat(label_rows), torch.cat(prior_rows), torch.cat(gate_rows))
    critical = [overall, per["h1"], per["h20"]]
    incoming = all(
        x["all_positive"] != "not_computable"
        and _fraction_at_least(x["all_positive"]["incoming_below_fraction"], .80)
        for x in critical
    )
    down = all(
        x["all_positive"] != "not_computable"
        and _fraction_at_least(x["all_positive"]["downcross_fraction"], .80)
        for x in critical
    )
    pattern = "incoming_readout_below_threshold_dominant" if incoming else "transition_downcrossing_dominant" if down else "mixed_no_single_dominant_pattern"
    decision, truth = torch.sigmoid(torch.cat(post_rows)) >= RAW_THRESHOLD, torch.cat(label_rows).bool()
    counts = {"tp": int((decision & truth).sum()), "fp": int((decision & ~truth).sum()), "fn": int((~decision & truth).sum())}
    if tuple(counts[k] for k in ("tp", "fp", "fn")) != FROZEN_TP_FP_FN:
        raise ValueError("frozen sentinel candidate TP/FP/FN drifted from 1902/557/5855")
    return {"schema_version": SCHEMA_VERSION, "raw_threshold": RAW_THRESHOLD, "pos_weight": POS_WEIGHT, "overall": overall, "by_horizon": per, "highlighted_horizons": {f"h{i}": per[f"h{i}"] for i in (1,5,10,20)}, "edge_gru_interface_trace": pattern, "candidate": counts, "provenance": dict(provenance), "gpu_execution": False, "locked_test_accessed": False, "formal_performance_claim_ready": False, "p4_status": "blocked"}


def _collect_edge_gru_trace(model: torch.nn.Module, loader: Any) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict[str, int], dict[str, str], int, float, float, str]:
    rows = [[[] for _ in HORIZONS] for _ in range(6)]
    fingerprints = [[] for _ in HORIZONS]
    calls = 0
    max_post = max_gru = 0.0
    hook_module_full_name: str | None = None
    model.eval()
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            with edge_gru_readonly_trace(model) as audit:
                official = _prediction("coupled_dual_gnn_residual", model, batch, {})["link_activity_logits"]
            if audit["module"] != "edge_transition":
                raise ValueError("edge GRU trace hook module identity drift")
            current_full_name = audit["module_full_name"]
            if hook_module_full_name is None:
                hook_module_full_name = current_full_name
            elif current_full_name != hook_module_full_name:
                raise ValueError("edge GRU trace hook module identity drift")
            if audit["actual_hook_calls"] != 20: raise ValueError("edge GRU trace must capture exactly 20 calls per batch")
            calls += audit["actual_hook_calls"]; max_gru = max(max_gru, float(audit["max_gru_reconstruction_error"]))
            target, history = batch["target"], batch["history"]
            labels, mask = target["aggregate_link_activity"], target["aggregate_link_activity_mask"].bool()
            prior, prior_mask = _previous_activity_and_mask(history["aggregate_link_activity"], history["aggregate_link_activity_mask"].bool(), labels, mask)
            valid = torch.all(batch["static"]["physical_edge_endpoint_index"] >= 0, dim=-1)[:, None, :].expand_as(labels) & mask
            if official.shape != labels.shape: raise ValueError("official link logits do not align with labels")
            for h, record in enumerate(audit["records"]):
                batch_size, edges = official.shape[0], official.shape[2]
                hidden_width = record["hidden_before"].shape[-1]
                if record["hidden_before"].shape != (batch_size * edges, hidden_width) or record["hidden_after"].shape != (batch_size * edges, hidden_width):
                    raise ValueError("edge GRU hook state cannot be restored to batch-by-edge shape")
                pre_hidden = record["hidden_before"].reshape(batch_size, edges, hidden_width)
                post_hidden = record["hidden_after"].reshape(batch_size, edges, hidden_width)
                pre = model.link_activity_head(pre_hidden).squeeze(-1); post = model.link_activity_head(post_hidden).squeeze(-1)
                max_post = max(max_post, float((post - official[:, h]).abs().max()))
                if max_post > 1e-6 or record["reconstruction_error"] > 1e-6: raise ValueError("edge GRU trace numerical drift")
                active = valid[:, h]; fingerprints[h].append(f"{batch_i}:{''.join('1' if x else '0' for x in active.cpu().reshape(-1).tolist())}")
                prior_value = torch.where(prior_mask[:, h], prior[:, h], torch.full_like(prior[:, h], -1))
                gate = record["update_gate"].reshape(batch_size, edges, hidden_width).mean(-1)
                for dst, item in zip(rows, (pre[active], post[active], official[:,h][active], gate[active], labels[:,h][active], prior_value[active])): dst[h].append(item.detach().cpu())
    result = [[torch.cat(row) for row in collection] for collection in rows]
    counts = {f"h{i}": int(result[0][i-1].numel()) for i in HORIZONS}; counts["total"] = sum(counts.values())
    if hook_module_full_name is None:
        raise ValueError("edge GRU trace hook module identity is missing")
    return (*result, counts, {f"h{i}": hashlib.sha256("|".join(fingerprints[i-1]).encode()).hexdigest() for i in HORIZONS}, calls, max_post, max_gru, hook_module_full_name)


def _publish(output_dir: Path, report: Mapping[str, Any], *, before_rename: Any | None = None) -> None:
    target = Path(output_dir).resolve()
    if target.exists(): raise FileExistsError(f"output directory already exists: {target}")
    parent, staging = target.parent.resolve(), target.parent.resolve() / f".{target.name}.staging-{uuid.uuid4().hex}"; parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.mkdir(); name = "edge_gru_interface_trace.json"; (staging / name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = {k: {"size_bytes": v["bytes"], "sha256": v["sha256"]} for k,v in _manifest(staging)["files"].items()}
        (staging / "manifest.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION + "-manifest", "files": files, "gpu_execution": False, "locked_test_accessed": False, "formal_performance_claim_ready": False, "p4_status": "blocked"}, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        if before_rename is not None: before_rename()
        if target.exists(): raise FileExistsError(f"output directory already exists: {target}")
        staging.rename(target)
    except BaseException:
        if staging.exists(): shutil.rmtree(staging)
        raise


def run_formal_p4_edge_gru_interface_trace(*, run_root: Path, tensor_root: Path, method: str, output_dir: Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    tensor_root = Path(tensor_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    model, loader, provenance = _prepare_frozen_link_validation(
        run_root,
        tensor_root,
        method,
        CANONICAL_TENSOR_MANIFEST_SHA256,
        FROZEN_SENTINEL_CHECKPOINT_SHA256,
        FROZEN_SENTINEL_SAMPLE_IDS_SHA256,
    )
    if method != "coupled_dual_gnn_residual":
        raise ValueError("frozen method drift")
    contract = json.loads((tensor_root / "tensor_contract.json").read_text(encoding="utf-8"))
    if (int(contract.get("history_steps", -1)), int(contract.get("horizon_steps", -1))) != (8, 20):
        raise ValueError("frozen tensor contract must be history=8 horizon=20")
    for key, expected in (
        ("checkpoint_sha256", FROZEN_SENTINEL_CHECKPOINT_SHA256),
        ("sample_ids_sha256", FROZEN_SENTINEL_SAMPLE_IDS_SHA256),
        ("tensor_manifest_sha256", CANONICAL_TENSOR_MANIFEST_SHA256),
    ):
        if str(provenance.get(key, "")).lower() != expected.lower():
            raise ValueError(f"frozen provenance {key} drift")
    if (
        any(provenance.get(key) is not False for key in ("gpu_execution", "locked_test_accessed", "formal_performance_claim_ready"))
        or provenance.get("checkpoint_reload") != {"strict": True, "missing_keys": 0, "unexpected_keys": 0}
        or provenance.get("class_weights_source_split") != "train"
        or float(provenance.get("link_activity_pos_weight", float("nan"))) != POS_WEIGHT
    ):
        raise ValueError("prepare provenance flags, weights, or strict reload drift")
    selected = provenance.get("selected_validation_tensor_inputs")
    if (
        not isinstance(selected, list)
        or not selected
        or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("relative_path"), str)
            or not item["relative_path"]
            or isinstance(item.get("size_bytes"), bool)
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] <= 0
            or not _is_sha256(item.get("sha256"))
            for item in selected
        )
    ):
        raise ValueError("selected validation tensor provenance drift")
    input_files = {
        "config": {"path": str((run_root / "config.json").resolve()), "sha256": provenance.get("config_sha256")},
        **{
            name: {"path": provenance.get(name + "_path"), "sha256": provenance.get(name + "_sha256")}
            for name in ("checkpoint", "sample_ids", "class_weights", "tensor_manifest")
        },
    }
    for name, entry in input_files.items():
        path = entry.get("path")
        if not isinstance(path, str) or not path or not Path(path).is_absolute() or not _is_sha256(entry.get("sha256")):
            raise ValueError(f"input_files {name} path or SHA-256 drift")
    if len(loader) != 64:
        raise ValueError("frozen edge GRU trace requires 64 batches and 1280 hook calls")
    pre, post, official, gate, labels, prior, counts, fingerprints, calls, post_error, gru_error, hook_module_full_name = _collect_edge_gru_trace(model, loader)
    if calls != 1280:
        raise ValueError("frozen edge GRU trace requires 64 batches and 1280 hook calls")
    if hook_module_full_name != FROZEN_HOOK_MODULE_FULL_NAME:
        raise ValueError("frozen edge GRU trace hook module identity drift")
    if not math.isfinite(post_error) or post_error > 1e-6 or not math.isfinite(gru_error) or gru_error > 1e-6:
        raise ValueError("edge GRU trace numerical drift")
    trace_provenance = {
        **provenance,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__)),
        "input_files": input_files,
        "mask_counts": counts,
        "mask_fingerprints": fingerprints,
        "hook_calls": {"actual": calls, "expected": 1280},
        "hook_module": "edge_transition",
        "hook_module_full_name": hook_module_full_name,
        "hook_removed": True,
        "max_post_official_error": post_error,
        "max_gru_reconstruction_error": gru_error,
    }
    report = build_edge_gru_interface_trace(pre, post, official, gate, labels, prior, trace_provenance)
    _publish(output_dir, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "tensor-root", "output-dir"): parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--method", required=True); a = parser.parse_args()
    print(json.dumps({"p4_status": run_formal_p4_edge_gru_interface_trace(run_root=a.run_root, tensor_root=a.tensor_root, method=a.method, output_dir=a.output_dir)["p4_status"]}))
if __name__ == "__main__": main()
