"""Build deterministic STEP 4.2C-C Flow Sample/Tensor evidence."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from pi_jwm.step4_2a_graph_input_extension_v1 import amend_raw_graph_inputs, build_extended_sample
from pi_jwm.step4_2c_b_causal_flow_ledger_raw_v1 import amend_raw_with_causal_flow_ledger
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import (
    SAMPLE_SCHEMA_VERSION,
    TENSOR_SCHEMA_VERSION,
    apply_flow_normalization,
    build_flow_extended_sample,
    build_flow_tensor_batch,
    fit_flow_normalization_stats,
    load_flow_tensor_batch,
    sample_tensor_semantic_equality,
    save_flow_tensor_batch,
    validate_flow_acceptance,
    validate_flow_sample_checks,
    validate_flow_tensor_checks,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step4_2c_c_flow_sample_tensor_v1_20260920"
MULTI_SOURCE = ROOT / "code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/real_communication_outcome_semantics.json"
CROSS_SOURCE = ROOT / "code/artifacts/protocols/pi_jwm_step4_2c_c_real_multihop_cross_slot_v1_20260920/real_communication_outcome_semantics.json"
DIRECT_SOURCE = ROOT / "code/artifacts/protocols/pi_jwm_step4_2c_b_real_flow_trace_source_v1_20260920/real_raw_contract_finalization.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _source(path: Path, *, structural_fixture: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    original = json.loads(path.read_text(encoding="utf-8"))
    if structural_fixture:
        # This old real Input/Return trace predates the wired-relation minimum.
        # The relation is a labeled contract fixture; Flow rows remain real.
        original.setdefault("environment", {})["wired_edges"] = [
            {"u": "RSU_0", "v": "cloudServer_4", "capacity_mbps": 100, "prop_ms": 1.0, "bidirectional": True}
        ]
    graph_raw = amend_raw_graph_inputs(original)
    ledger_raw, receipt = amend_raw_with_causal_flow_ledger(graph_raw)
    if not receipt["passed"]:
        raise RuntimeError(f"Raw Flow amendment failed for {path}")
    return graph_raw, ledger_raw


def _sample(path: Path, anchor: int, split: str, *, structural_fixture: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    graph_raw, ledger_raw = _source(path, structural_fixture=structural_fixture)
    base = build_extended_sample(graph_raw, anchor_step=anchor)
    base["metadata"]["split"] = split
    sample = build_flow_extended_sample(base, ledger_raw)
    sample["metadata"]["evidence_class"] = "REAL_FLOW_TRACE_WITH_STRUCTURAL_COMM_FIXTURE" if structural_fixture else "REAL_TRACE"
    sample["metadata"]["source_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
    sample["metadata"]["source_sha256"] = sha256(path)
    return ledger_raw, sample


def semantic_digest(samples: list[dict[str, Any]], stats: dict[str, Any], tensor: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for value in (samples, stats, tensor["contract"], tensor["sample_static"], tensor["sample_metadata"]):
        digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in sorted(key for key, value in tensor.items() if isinstance(value, np.ndarray)):
        value = tensor[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def build_all() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    multi_raw, multi = _sample(MULTI_SOURCE, 1, "dev_train")
    cross_raw, cross_1 = _sample(CROSS_SOURCE, 1, "dev_train")
    _, cross_2 = _sample(CROSS_SOURCE, 2, "dev_train")
    _, cross_3 = _sample(CROSS_SOURCE, 3, "dev_validation")
    direct_raw, direct = _sample(DIRECT_SOURCE, 4, "dev_validation", structural_fixture=True)
    samples = [multi, cross_1, cross_2, cross_3, direct]
    stats = fit_flow_normalization_stats(samples)
    normalized = apply_flow_normalization(samples, stats)
    tensor = build_flow_tensor_batch(normalized, stats=stats)
    raw_context = {"multi": multi_raw, "cross": cross_raw, "direct": direct_raw}
    return normalized, stats, tensor, raw_context


def fixture_checks(sample: dict[str, Any], tensor: dict[str, Any], stats: dict[str, Any]) -> dict[str, bool]:
    target_only = copy.deepcopy(sample)
    future = copy.deepcopy(target_only["target"][0]["logical_flows"][0])
    future.update({"flow_id": "flow::Task_1::Input::1", "epoch": 1, "flow_index": 1, "target_index": 1})
    target_only["target"][0]["logical_flows"].append(future)
    target_only["static"]["target_index"]["logical_flow"][future["flow_id"]] = 1
    target_only["static"]["target_only_objects"]["logical_flow"].append(future["flow_id"])
    leaked = copy.deepcopy(target_only)
    leaked["static"]["input_entity_index"]["logical_flow"][future["flow_id"]] = 1

    tampered_tensor = copy.deepcopy(tensor)
    known_position = np.argwhere(tampered_tensor["logical_flow_known_mask"])[0]
    bi, hi, fi = (int(value) for value in known_position)
    tampered_tensor["logical_flow_index"][bi, hi, fi] += 7
    overflow_rejected = False
    try:
        build_flow_tensor_batch(
            apply_flow_normalization([sample], stats),
            stats=stats,
            contract=type("Bound", (), {"max_logical_flow": 0, "max_target_logical_flow": 0, "max_route_nodes": 0, "to_dict": lambda self: {}})(),
        )
    except ValueError as exc:
        overflow_rejected = "capacity overflow" in str(exc)
    receipt = {
        "required_checks": {
            "raw_sample_semantic_equality": True, "sample_tensor_semantic_equality": True,
            "history_causal_flow_union": True, "stable_flow_identity": True,
            "target_only_future_flow_isolation": True, "epoch_isolation": False,
            "presence_mask_correctness": True, "train_only_preprocessing": True,
            "no_silent_truncation": True, "input_return_categories": True,
            "depdata_runtime_zero": True, "deterministic_rebuild": True,
            "serialize_load": True, "scope": True,
        },
        "passed": True,
        "scope": {name: False for name in ("graph_builder", "information_graph", "physical_topology", "training", "gpu", "locked_test", "formal_dataset")},
    }
    return {
        "future_epoch_leak_rejected": not validate_flow_sample_checks(leaked)["target_only_future_flow_isolation"],
        "flow_id_tensor_slot_tamper_rejected": not validate_flow_tensor_checks(tampered_tensor)["passed"],
        "capacity_overflow_rejected": overflow_rejected,
        "receipt_tamper_rejected": not validate_flow_acceptance(receipt)["passed"],
        "categorical_identity_not_normalized": not bool({"epoch", "flow_index", "flow_type", "status", "presence", "route_revision"} & set(stats["features"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.refresh_existing:
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=args.refresh_existing)

    samples, stats, tensor, raw_context = build_all()
    samples_2, stats_2, tensor_2, _ = build_all()
    digest = semantic_digest(samples, stats, tensor)
    deterministic = digest == semantic_digest(samples_2, stats_2, tensor_2)
    sample_checks = [validate_flow_sample_checks(sample) for sample in samples]
    tensor_checks = validate_flow_tensor_checks(tensor)

    cross_indices = [index for index, sample in enumerate(samples) if sample["metadata"]["source_path"].endswith("pi_jwm_step4_2c_c_real_multihop_cross_slot_v1_20260920/real_communication_outcome_semantics.json")]
    flow_id = "flow::Task_1::Input::0"
    cross_rows = [row for index in cross_indices for frame in samples[index]["history"] for row in frame["logical_flows"] if row.get("known") and row["flow_id"] == flow_id]
    cross_carry = [row for index in cross_indices for frame in samples[index]["history"] for row in frame["carrying_states"] if row.get("known") and row["flow_id"] == flow_id]
    multihop_transitions = [
        row for step in raw_context["cross"]["steps"]
        for row in step["outcome"].get("flow_ledger_transition_evidence", [])
        if row.get("flow_id") == flow_id and "error" not in row
    ]
    real_multihop = {
        "single_flow_id": {row["flow_id"] for row in cross_rows} == {flow_id},
        "single_epoch": {row["epoch"] for row in cross_rows} == {0},
        "constant_logical_destination": {row["logical_destination"] for row in cross_rows} == {"cloudServer_4"},
        "single_tensor_slot": len({samples[index]["static"]["input_entity_index"]["logical_flow"][flow_id] for index in cross_indices}) == 1,
        "distinct_cross_slot_carrying_progress": len({row["hop_progress"] for row in cross_carry if row["active"]}) >= 2,
        "intermediate_hop_service_not_e2e": any(row.get("target_id") != "cloudServer_4" and float(row.get("logical_delivery_delta", 0.0)) == 0.0 for row in multihop_transitions),
        "final_hop_advances_e2e": any(row.get("target_id") == "cloudServer_4" and float(row.get("logical_delivery_delta", 0.0)) > 0.0 for row in multihop_transitions),
        "no_duplicate_flow_row_per_hop": all(len({row["flow_id"] for row in frame["logical_flows"] if row.get("known")}) == sum(bool(row.get("known")) for row in frame["logical_flows"]) for sample in samples for frame in sample["history"]),
    }
    categories = {row["flow_type"] for sample in samples for frame in sample["history"] for row in frame["logical_flows"] if row.get("known")}
    known_inactive = any(row.get("known") and not row.get("presence") and row.get("status") in {"COMPLETED", "SUPERSEDED"} for sample in samples for frame in sample["history"] for row in frame["logical_flows"])
    fixtures = fixture_checks(samples[0], tensor, stats)

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "tensor.npz"
        save_flow_tensor_batch(tensor, path)
        loaded = load_flow_tensor_batch(path)
        round_trip = tensor["contract"] == loaded["contract"] and all(np.array_equal(value, loaded[key]) for key, value in tensor.items() if isinstance(value, np.ndarray))

    required = {
        "raw_sample_semantic_equality": all(row["raw_sample_semantic_digest_equality"] for row in sample_checks),
        "sample_tensor_semantic_equality": sample_tensor_semantic_equality(samples, tensor),
        "history_causal_flow_union": all(row["history_causal_flow_union"] for row in sample_checks),
        "stable_flow_identity": all(row["stable_flow_id_index"] and row["flow_id_epoch_identity"] for row in sample_checks),
        "multi_hop_single_flow_tensor_identity": all(real_multihop.values()),
        "target_only_future_flow_isolation": fixtures["future_epoch_leak_rejected"] and all(row["target_only_future_flow_isolation"] for row in sample_checks),
        "epoch_isolation": fixtures["future_epoch_leak_rejected"],
        "presence_mask_correctness": known_inactive and tensor_checks["known_presence_independent"] and tensor_checks["masked_placeholder_zero"],
        "train_only_preprocessing": stats["source_split"] == "dev_train" and all("dev_train only" in row["mask_policy"] for row in stats["features"].values()),
        "no_silent_truncation": tensor_checks["no_silent_truncation"] and fixtures["capacity_overflow_rejected"],
        "input_return_categories": {"Input", "Return"} <= categories,
        "depdata_runtime_zero": "DepData" not in categories,
        "deterministic_rebuild": deterministic,
        "serialize_load": round_trip,
        "scope": True,
    }
    scope = {name: False for name in ("graph_builder", "information_graph", "physical_topology", "training", "gpu", "locked_test", "formal_dataset")}
    acceptance = {
        "schema_version": "PI-JWM-Step-4.2C-C-Acceptance-v1",
        "required_checks": required,
        "passed": all(required.values()) and all(value is False for value in scope.values()),
        "scope": scope,
        "sample_checks": sample_checks,
        "tensor_checks": tensor_checks,
        "contract_fixture_checks": fixtures,
        "real_multihop_checks": real_multihop,
        "deterministic_semantic_digest": digest,
        "evidence_boundary": {
            "real_trace": "direct Input/Return Flow rows, real two-hop events, and real cross-slot partial final-hop Decision states",
            "contract_fixture": "wired structural relation only for the older direct Input/Return source; future Epoch and tamper counterfactuals",
            "not_claimed": ["real Return multi-hop", "real same-destination partial-hop reroute", "formal Dataset capacity"],
        },
    }
    acceptance["validation"] = validate_flow_acceptance(acceptance)
    # multi-hop is an extra required check beyond the shared validator set.
    acceptance["validation"]["passed"] = bool(acceptance["validation"]["passed"] and required["multi_hop_single_flow_tensor_identity"])
    if not acceptance["validation"]["passed"]:
        raise RuntimeError(json.dumps(acceptance, ensure_ascii=False, default=str))

    write_json(args.output_dir / "model_ready_flow_samples.json", samples)
    write_json(args.output_dir / "flow_train_normalization_stats.json", stats)
    write_json(args.output_dir / "tensor_schema.json", tensor["contract"])
    write_json(args.output_dir / "acceptance.json", acceptance)
    save_flow_tensor_batch(tensor, args.output_dir / "flow_tensor.npz")
    files = ["model_ready_flow_samples.json", "flow_train_normalization_stats.json", "tensor_schema.json", "acceptance.json", "flow_tensor.npz"]
    sources = [MULTI_SOURCE, CROSS_SOURCE, DIRECT_SOURCE, Path(__file__), ROOT / "code/src/pi_jwm/step4_2c_c_flow_sample_tensor_v1.py", ROOT / "code/tests/test_step4_2c_c_flow_sample_tensor_v1.py"]
    manifest = {
        "schema_version": "PI-JWM-Step-4.2C-C-Manifest-v1",
        "sample_schema_version": SAMPLE_SCHEMA_VERSION,
        "tensor_schema_version": TENSOR_SCHEMA_VERSION,
        "files": {name: sha256(args.output_dir / name) for name in files},
        "source_files": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in sources},
        "deterministic_semantic_digest": digest,
        "passed": acceptance["passed"],
        "scope": scope,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"output": str(args.output_dir), "passed": acceptance["passed"], "required_checks": required}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
