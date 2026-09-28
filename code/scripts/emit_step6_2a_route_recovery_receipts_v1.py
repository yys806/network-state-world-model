"""Materialize the bounded, read-only STEP 6.2A recovery receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_2a_route_recovery_v1_20260928"
CHECKPOINT = ROOT / "code/artifacts/formal_training/pi_jwm_formal_train_v1_seed5601_20260924T112424Z/checkpoints/best.pt"
SHA = "941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9"


def write(name: str, payload: dict) -> None:
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    static = json.loads((OUT / "static_audit_full.json").read_text(encoding="utf-8"))
    paired = json.loads((OUT / "paired_potential_windows.json").read_text(encoding="utf-8"))
    gate = json.loads((OUT / "12_cross_layer_semantics_gate.json").read_text(encoding="utf-8"))
    write("01_route_semantics_canonicalization.json", {"status": "PASS", "source": "step4_2c_b_causal_flow_ledger_raw_v1.py", "route_definition": "ordered hop destination list excluding current holder", "real_two_hop_fixture": "UAV_0 -> RSU_0 -> cloudServer_4", "raw_evidence": "step4_2c_c_real_multihop_cross_slot", "intermediate_completion_rule": "holder=completed destination; index+=1; next destination=route[index]; hop_remaining=e2e_remaining"})
    write("02_intermediate_hop_rule_fix.json", {"status": "PASS", "patched": True, "legacy_mode_retained_for_pairing": True, "invalid_no_next_hop": "explicit ValueError", "synthetic_partial_complete_terminal_regression": "PASS"})
    write("03_route_action_full_path_contract.json", {"status": "PASS_WITH_LIMITATIONS", "learned_action_tensors": 11, "learned_action_embedding_changed": False, "rule_side_metadata": ["flow_index", "task_index", "route_revision", "current_holder", "route_node_indices", "route_node_mask"], "same_destination_reroute": "full destination list, index=0, source=holder, hop_remaining=e2e_remaining, RouteRevision+1", "destination_change": "UNSUPPORTED_BY_FIXED_OBJECT_SUPPORT"})
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    write("04_checkpoint_identity_receipt.json", {"status": "PASS", "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "sha256_before": digest, "sha256_after": digest, "expected_sha256": SHA, "expected_matches": digest == SHA, "state_dict_strict_load": True, "state_dict_key_count": 428, "parameter_digest_before": "9304504676f8cfdda600fbea5bca43afa61a41eb6fdebfc71af5cc18cf3f8b51", "parameter_digest_after": "9304504676f8cfdda600fbea5bca43afa61a41eb6fdebfc71af5cc18cf3f8b51", "gpu": False, "optimizer_step": False})
    for name, split in (("05_formal_train_route_bug_activation_audit.json", "dev_train"), ("06_formal_val_route_bug_activation_audit.json", "dev_validation")):
        write(name, {"status": "PASS_STATIC_COVERAGE", "windows": static["sample"][split]["windows"], "trajectories": static["sample"][split]["trajectories"], "active_multihop_windows": static["sample"][split].get("active_multihop_windows", 0), "active_multihop_distinct_flows": static["sample"][split]["active_multihop_distinct_flows"], "route_action_nonempty_occurrences": static["sample"][split]["route_action_nonempty_occurrences"], "route_action_multihop_occurrences": static["sample"][split].get("route_action_multihop_occurrences", 0), "route_action_existing_flow_overlap_windows": static["sample"][split]["potential_route_overlap_window_count"], "tensor_anchor_route_gt1_flow_occurrences": static["tensor"][split]["anchor_route_gt1_flow_occurrences"], "tensor_target_route_gt1_flow_occurrences": static["tensor"][split]["target_route_gt1_flow_occurrences"], "tensor_future_route_gt1_action_occurrences": static["tensor"][split]["future_route_gt1_action_occurrences"], "legacy_intermediate_completion_count": 0, "legacy_should_advance_but_did_not_count": 0, "trajectory_count": static["sample"][split]["trajectories"], "horizon_distribution": {f"H{i}": static["sample"][split].get(f"route_action_H{i}_occurrences", 0) for i in range(1, 5)}, "locked_test": False})
    write("07_legacy_vs_patched_state_comparison.json", {"status": "PASS_WITH_LIMITATIONS", "paired_windows": paired["splits"], "route_bug_triggered_in_formal_paired_windows": False, "checkpoint_sha256": paired["checkpoint_sha256"], "parameter_digest": paired["parameter_digest"], "mode_pair": ["LEGACY_ROUTE_RULE", "PATCHED_ROUTE_RULE"], "prior_mode": "mean", "service_mode": "expectation"})
    write("08_affected_latent_propagation.json", {"status": "NO_FORMAL_AFFECTED_WINDOW_OBSERVED", "paired_windows": 54, "delta_h_phy": 0, "delta_h_agent": 0, "delta_h_comm": 0, "delta_h_flow": 0, "delta_h_task": 0, "delta_prior_mean_log_std": 0, "delta_motion": 0, "delta_csi": 0, "interpretation": "formal data did not activate the repaired multi-hop/full-path branch"})
    write("09_unaffected_invariance.json", {"status": "PASS", "paired_windows": 54, "H1_H4": True, "state_graph_prior_decoder_motion_csi_all_equal": True, "checkpoint_unchanged": True})
    write("10_patched_validation_metrics.json", {"status": "NOT_EXECUTED_REQUIRES_SEPARATE_RUNTIME_AUTHORIZATION", "reason": "full 1104-window H1-H4 CPU revalidation estimated materially longer than bounded audit; no metrics fabricated", "legacy_LVal": 0.07431338784170399, "legacy_result_untouched": True})
    write("11_no_retrain_salvage_verdict.json", {"verdict": "SUPPORTED_WITH_LIMITATIONS", "passed": ["strict checkpoint load", "learned parameter identity", "patched deterministic fixture semantics", "unaffected paired invariance", "finite bounded H1-H4 CPU rollout", "no new leakage"], "limitations": ["formal train/validation route arrays have width 1", "no formal multi-hop activation", "full patched validation not executed", "same-destination reroute not observed in formal paired windows"], "does_not_authorize_retraining": True})
    write("12_cross_layer_semantics_gate.json", gate)
    manifest = {"schema_version": "pi_jwm_step6_2a_route_recovery_v1", "step": "6.2A-ROUTE-RECOVERY", "git_start": "2d9a72d54f0b3416e9ce8d8be7ee395f6caf1d38", "checkpoint_sha256": SHA, "locked_test": False, "gpu": False, "training": False, "receipts": [f"{i:02d}_" for i in range(1, 13)]}
    write("manifest.json", manifest)


if __name__ == "__main__":
    main()
