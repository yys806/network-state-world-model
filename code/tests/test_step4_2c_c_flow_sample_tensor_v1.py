import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pi_jwm.step4_2a_graph_input_extension_v1 import (
    amend_raw_graph_inputs,
    build_extended_sample,
)
from pi_jwm.step4_2c_b_causal_flow_ledger_raw_v1 import (
    CausalFlowLedger,
    amend_raw_with_causal_flow_ledger,
)
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import (
    FLOW_NUMERIC_FEATURE_ORDER,
    FLOW_STATUS_VOCAB,
    FLOW_TYPE_VOCAB,
    FlowTensorContract,
    apply_flow_normalization,
    build_flow_extended_sample,
    build_flow_tensor_batch,
    fit_flow_normalization_stats,
    load_flow_tensor_batch,
    save_flow_tensor_batch,
    sample_tensor_semantic_equality,
    validate_flow_acceptance,
    validate_flow_sample_checks,
    validate_flow_tensor_checks,
)


ROOT = Path(__file__).resolve().parents[2]
REAL_MULTI = ROOT / "code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/real_communication_outcome_semantics.json"
REAL_DIRECT = ROOT / "code/artifacts/protocols/pi_jwm_step4_2c_b_real_flow_trace_source_v1_20260920/real_raw_contract_finalization.json"


class Step42CCFlowSampleTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.multi_original = json.loads(REAL_MULTI.read_text(encoding="utf-8"))
        cls.direct_original = json.loads(REAL_DIRECT.read_text(encoding="utf-8"))

    def _ledger_raw(self, original):
        source = copy.deepcopy(original)
        # The real direct Input/Return trace predates the frozen wired-relation
        # minimum. Add only a structural fixture relation so the unchanged
        # STEP 4.2A validator can build its base sample; Flow rows stay real.
        if source is not self.multi_original and not source.get("environment", {}).get("wired_edges"):
            source.setdefault("environment", {})["wired_edges"] = [
                {"u": "RSU_0", "v": "cloudServer_4", "capacity_mbps": 100, "prop_ms": 1.0, "bidirectional": True}
            ]
        graph_raw = amend_raw_graph_inputs(source)
        ledger_raw, receipt = amend_raw_with_causal_flow_ledger(graph_raw)
        self.assertTrue(receipt["passed"])
        return graph_raw, ledger_raw

    def _sample(self, original=None, anchor=1, split="dev_train"):
        graph_raw, ledger_raw = self._ledger_raw(original or self.multi_original)
        base = build_extended_sample(graph_raw, anchor_step=anchor)
        base["metadata"]["split"] = split
        return ledger_raw, build_flow_extended_sample(base, ledger_raw)

    def _tensor(self, samples):
        stats = fit_flow_normalization_stats(samples)
        normalized = apply_flow_normalization(samples, stats)
        return build_flow_tensor_batch(normalized, stats=stats), stats

    def test_real_multihop_is_one_history_flow_slot_with_frozen_destination(self):
        _, sample = self._sample(anchor=1)
        index = sample["static"]["input_entity_index"]["logical_flow"]
        self.assertEqual({"flow::Task_1::Input::0": 0}, index)
        rows = [frame["logical_flows"][0] for frame in sample["history"]]
        self.assertFalse(rows[0]["known"])
        self.assertEqual("cloudServer_4", rows[1]["logical_destination"])
        self.assertEqual(0, rows[1]["epoch"])
        self.assertEqual("COMPLETED", rows[1]["status"])
        self.assertFalse(rows[1]["presence"])
        self.assertTrue(validate_flow_sample_checks(sample)["passed"])

    def test_input_and_return_are_distinct_categories_and_keep_raw_destinations(self):
        _, sample = self._sample(self.direct_original, anchor=4)
        known = [row for frame in sample["history"] for row in frame["logical_flows"] if row["known"]]
        types = {row["flow_type"] for row in known}
        self.assertIn("Input", types)
        self.assertIn("Return", types)
        for row in known:
            if row["flow_type"] == "Return":
                self.assertEqual("decision.tasks[].return_destination_id", row["logical_destination_source"])

    def test_next_hop_as_destination_and_raw_sample_tamper_are_rejected(self):
        _, sample = self._sample(anchor=1)
        tampered = copy.deepcopy(sample)
        row = next(row for row in tampered["history"][1]["logical_flows"] if row["known"])
        row["logical_destination"] = "RSU_0"
        checks = validate_flow_sample_checks(tampered)
        self.assertFalse(checks["raw_sample_semantic_digest_equality"])
        self.assertFalse(checks["passed"])

    def test_normal_hop_epoch_route_revision_and_intermediate_e2e_tampers_fail(self):
        _, sample = self._sample(anchor=1)
        for field, value, check in (
            ("epoch", 1, "flow_id_epoch_identity"),
            ("route_revision", 1, "raw_sample_semantic_digest_equality"),
            ("e2e_delivered", 999.0, "flow_conservation"),
        ):
            tampered = copy.deepcopy(sample)
            if field == "route_revision":
                row = next(row for row in tampered["history"][1]["carrying_states"] if row["known"])
            else:
                row = next(row for row in tampered["history"][1]["logical_flows"] if row["known"])
            row[field] = value
            checks = validate_flow_sample_checks(tampered)
            self.assertFalse(checks[check])
            self.assertFalse(checks["passed"])

    def test_future_epoch_is_target_only_and_cannot_use_history_index(self):
        _, sample = self._sample(anchor=1)
        future = copy.deepcopy(sample["target"][0]["logical_flows"][0])
        future.update({"flow_id": "flow::Task_1::Input::1", "epoch": 1, "flow_index": 1, "target_index": 1, "status": "ACTIVE", "presence": True})
        sample["target"][0]["logical_flows"].append(future)
        sample["static"]["target_index"]["logical_flow"][future["flow_id"]] = 1
        sample["static"]["target_only_objects"]["logical_flow"].append(future["flow_id"])
        sample["metadata"]["target_flow_semantic_digests"][str(sample["target"][0]["frame_index"])] = "fixture-expanded"
        self.assertNotIn(future["flow_id"], sample["static"]["input_entity_index"]["logical_flow"])
        tampered = copy.deepcopy(sample)
        tampered["static"]["input_entity_index"]["logical_flow"][future["flow_id"]] = 1
        self.assertFalse(validate_flow_sample_checks(tampered)["target_only_future_flow_isolation"])

    def test_superseded_known_row_is_not_padding_and_indices_do_not_shift(self):
        _, sample = self._sample(anchor=1)
        old = copy.deepcopy(sample["history"][1]["logical_flows"][0])
        old.update({"status": "SUPERSEDED", "presence": False, "known": True})
        new = copy.deepcopy(old)
        new.update({"flow_id": "flow::Task_1::Input::1", "flow_index": 1, "epoch": 1, "status": "ACTIVE", "presence": True})
        sample["static"]["input_entity_index"]["logical_flow"] = {old["flow_id"]: 0, new["flow_id"]: 1}
        carry_template = copy.deepcopy(sample["history"][1]["carrying_states"][0])
        for frame in sample["history"]:
            frame["logical_flows"] = [copy.deepcopy(old), copy.deepcopy(new)]
            frame["logical_flows"][0]["flow_index"] = 0
            frame["logical_flows"][1]["flow_index"] = 1
            carry = copy.deepcopy(carry_template)
            carry_new = copy.deepcopy(carry)
            carry_new.update({"flow_id": new["flow_id"], "flow_index": 1})
            carry["flow_index"] = 0
            frame["carrying_states"] = [carry, carry_new]
        tensor, _ = self._tensor([sample])
        self.assertTrue(tensor["logical_flow_known_mask"][0, 1, 0])
        self.assertFalse(tensor["logical_flow_presence"][0, 1, 0])
        self.assertTrue(tensor["logical_flow_known_mask"][0, 1, 1])

    def test_tensor_round_trip_identity_masks_and_no_categorical_normalization(self):
        _, sample = self._sample(anchor=1)
        tensor, stats = self._tensor([sample])
        self.assertEqual(set(stats["features"]), set(FLOW_NUMERIC_FEATURE_ORDER))
        self.assertNotIn("epoch", stats["features"])
        self.assertNotIn("route_revision", stats["features"])
        self.assertIn("Input", FLOW_TYPE_VOCAB)
        self.assertIn("SUPERSEDED", FLOW_STATUS_VOCAB)
        self.assertTrue(validate_flow_tensor_checks(tensor)["passed"])
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "flow_tensor.npz"
            save_flow_tensor_batch(tensor, path)
            loaded = load_flow_tensor_batch(path)
        self.assertEqual(tensor["contract"], loaded["contract"])
        for key in ("logical_flow_raw_features", "logical_flow_type_index", "carrying_holder_index", "route_node_indices"):
            self.assertTrue(np.array_equal(tensor[key], loaded[key]))

    def test_flow_id_tensor_slot_round_trip_tamper_fails(self):
        _, sample = self._sample(anchor=1)
        tensor, _ = self._tensor([sample])
        tampered = copy.deepcopy(tensor)
        tampered["logical_flow_index"][0, 1, 0] = 7
        checks = validate_flow_tensor_checks(tampered)
        self.assertFalse(checks["flow_id_tensor_index_round_trip"])
        self.assertFalse(checks["passed"])

    def test_depdata_cannot_be_fabricated_from_dag(self):
        _, sample = self._sample(anchor=1)
        fabricated = copy.deepcopy(sample)
        row = next(row for row in fabricated["history"][1]["logical_flows"] if row["known"])
        row["flow_type"] = "DepData"
        checks = validate_flow_sample_checks(fabricated)
        self.assertFalse(checks["depdata_runtime_zero"])
        self.assertFalse(checks["passed"])

    def test_capacity_overflow_is_rejected_not_truncated(self):
        _, sample = self._sample(anchor=1)
        stats = fit_flow_normalization_stats([sample])
        normalized = apply_flow_normalization([sample], stats)
        with self.assertRaisesRegex(ValueError, "capacity overflow"):
            build_flow_tensor_batch(normalized, stats=stats, contract=FlowTensorContract(max_logical_flow=0, max_target_logical_flow=0, max_route_nodes=0))

    def test_target_only_future_flow_reference_cannot_enter_history_tensor(self):
        _, sample = self._sample(anchor=1)
        baseline, _ = self._tensor([sample])
        changed = copy.deepcopy(sample)
        target = changed["target"][0]["logical_flows"][0]
        target["e2e_remaining"] = 123456.0
        candidate, _ = self._tensor([changed])
        self.assertTrue(np.array_equal(baseline["logical_flow_features"], candidate["logical_flow_features"]))

    def test_receipt_tamper_forces_top_level_false(self):
        required = {
            "raw_sample_semantic_equality": True,
            "sample_tensor_semantic_equality": True,
            "history_causal_flow_union": True,
            "stable_flow_identity": True,
            "target_only_future_flow_isolation": True,
            "epoch_isolation": True,
            "presence_mask_correctness": True,
            "train_only_preprocessing": True,
            "no_silent_truncation": True,
            "input_return_categories": True,
            "depdata_runtime_zero": True,
            "deterministic_rebuild": True,
            "serialize_load": True,
            "scope": True,
        }
        required["epoch_isolation"] = False
        receipt = {"required_checks": required, "passed": True, "scope": {"graph_builder": False, "information_graph": False, "physical_topology": False, "training": False, "gpu": False, "locked_test": False, "formal_dataset": False}}
        result = validate_flow_acceptance(receipt)
        self.assertFalse(result["expected_passed"])
        self.assertFalse(result["passed"])

    def test_presence_aware_normalization_excludes_repeated_inactive_completed_lineage(self):
        _, active = self._sample(self.direct_original, anchor=4, split="dev_train")
        baseline = fit_flow_normalization_stats([active])
        inactive = copy.deepcopy(active)
        for frame in inactive["history"]:
            for row in frame["logical_flows"]:
                if row.get("known"):
                    row.update({"presence": False, "status": "COMPLETED", "total_data": 10_000.0, "e2e_delivered": 10_000.0, "e2e_remaining": 0.0})
                    row["feature_mask"] = {"total_data": True, "e2e_delivered": True, "e2e_remaining": True}
            for row in frame["carrying_states"]:
                if row.get("known"):
                    row.update({"hop_progress": 10_000.0, "hop_remaining": 10_000.0})
                    row["feature_mask"] = {"hop_progress": True, "hop_remaining": True}
        repeated = fit_flow_normalization_stats([active, inactive, copy.deepcopy(inactive)])
        self.assertEqual(baseline["features"], repeated["features"])
        self.assertGreater(baseline["features"]["logical.total_data"]["count"], 0)
        self.assertTrue(all(feature["mask_policy"] == "presence=true AND feature_mask=true AND value!=null; train split only" for feature in baseline["features"].values()))

    def test_validation_and_target_never_enter_presence_aware_fit(self):
        _, active = self._sample(self.direct_original, anchor=4, split="dev_train")
        baseline = fit_flow_normalization_stats([active])
        validation = copy.deepcopy(active)
        validation["metadata"]["split"] = "dev_validation"
        target_only = copy.deepcopy(active)
        target_only["metadata"]["split"] = "dev_train"
        for frame in target_only["history"]:
            for row in frame["logical_flows"]:
                row["presence"] = False
        for frame in target_only["target"]:
            for row in frame["logical_flows"]:
                if row.get("known"):
                    row["total_data"] = 999_999.0
                    row["e2e_delivered"] = 999_999.0
                    row["e2e_remaining"] = 0.0
            for row in frame["carrying_states"]:
                if row.get("known"):
                    row["hop_progress"] = 999_999.0
                    row["hop_remaining"] = 999_999.0
        fitted = fit_flow_normalization_stats([active, validation, target_only])
        self.assertEqual(baseline["features"], fitted["features"])

    def test_target_carrying_namespace_is_preserved_and_validated(self):
        _, sample = self._sample(anchor=1)
        tensor, _ = self._tensor([sample])
        for key in (
            "target_carrying_known_mask", "target_carrying_active", "target_carrying_route_revision",
            "target_carrying_current_hop_index", "target_carrying_holder_index",
            "target_carrying_hop_source_index", "target_carrying_hop_destination_index",
            "target_carrying_raw_features", "target_carrying_features", "target_carrying_feature_mask",
            "target_route_node_indices", "target_route_node_mask",
        ):
            self.assertIn(key, tensor)
        self.assertTrue(validate_flow_tensor_checks(tensor)["passed"])
        self.assertTrue(sample_tensor_semantic_equality([sample], tensor))

    def test_full_sample_tensor_equality_rejects_history_carrying_holder_and_route_tamper(self):
        _, sample = self._sample(anchor=1)
        tensor, _ = self._tensor([sample])
        tampered = copy.deepcopy(sample)
        carry = next(row for row in tampered["history"][1]["carrying_states"] if row["known"])
        carry["current_holder"] = "RSU_0"
        self.assertFalse(sample_tensor_semantic_equality([tampered], tensor))
        tampered = copy.deepcopy(sample)
        carry = next(row for row in tampered["history"][1]["carrying_states"] if row["known"])
        carry["route"] = list(reversed(carry["route"]))
        self.assertFalse(sample_tensor_semantic_equality([tampered], tensor))

    def test_full_target_logical_equality_rejects_destination_status_and_epoch_tamper(self):
        _, sample = self._sample(anchor=1)
        tensor, _ = self._tensor([sample])
        for field, value in (("logical_destination", "RSU_0"), ("status", "ACTIVE"), ("epoch", 9)):
            tampered = copy.deepcopy(sample)
            row = next(row for row in tampered["target"][0]["logical_flows"] if row["known"])
            row[field] = value
            self.assertFalse(sample_tensor_semantic_equality([tampered], tensor), field)

    def test_full_target_carrying_equality_rejects_holder_hop_and_route_mask_tamper(self):
        _, sample = self._sample(anchor=1)
        tensor, _ = self._tensor([sample])
        tampered = copy.deepcopy(sample)
        row = next(row for row in tampered["target"][0]["carrying_states"] if row["known"])
        row["current_holder"] = "RSU_0"
        self.assertFalse(sample_tensor_semantic_equality([tampered], tensor))
        tampered_tensor = copy.deepcopy(tensor)
        tampered_tensor["target_route_node_mask"][0, 0, 0, 0] = False
        self.assertFalse(sample_tensor_semantic_equality([sample], tampered_tensor))

    def test_target_only_flow_cannot_be_inserted_into_history_tensor_namespace(self):
        _, sample = self._sample(anchor=1)
        future = copy.deepcopy(sample["target"][0]["logical_flows"][0])
        future.update({"flow_id": "flow::Task_1::Input::9", "epoch": 9, "flow_index": 1, "target_index": 1, "status": "ACTIVE", "presence": True})
        sample["target"][0]["logical_flows"].append(future)
        sample["target"][0]["carrying_states"].append(copy.deepcopy(sample["target"][0]["carrying_states"][0]))
        sample["target"][0]["carrying_states"][-1].update({"flow_id": future["flow_id"], "flow_index": 1, "target_index": 1})
        sample["static"]["target_index"]["logical_flow"][future["flow_id"]] = 1
        sample["static"]["target_only_objects"]["logical_flow"].append(future["flow_id"])
        tensor, _ = self._tensor([sample])
        tampered = copy.deepcopy(tensor)
        tampered["sample_static"][0]["input_entity_index"]["logical_flow"][future["flow_id"]] = 1
        self.assertFalse(validate_flow_tensor_checks(tampered)["target_only_future_flow_isolation"])

    def test_missing_target_carrying_tensor_cannot_claim_semantic_equality(self):
        _, sample = self._sample(anchor=1)
        tensor, _ = self._tensor([sample])
        tampered = copy.deepcopy(tensor)
        tampered.pop("target_carrying_holder_index")
        self.assertFalse(sample_tensor_semantic_equality([sample], tampered))

    def test_target_tensor_destination_tamper_is_rejected_by_validator(self):
        _, sample = self._sample(anchor=1)
        tensor, _ = self._tensor([sample])
        tampered = copy.deepcopy(tensor)
        tampered["target_logical_flow_destination_index"][0, 0, 0] = 0
        self.assertFalse(validate_flow_tensor_checks(tampered)["sample_target_logical_equality"])

    def test_target_tensor_status_and_future_epoch_identity_tamper_are_rejected(self):
        _, sample = self._sample(anchor=1)
        tensor, _ = self._tensor([sample])
        for key, value in (("target_logical_flow_status_index", 1), ("target_logical_flow_epoch", 9)):
            tampered = copy.deepcopy(tensor)
            tampered[key][0, 0, 0] = value
            checks = validate_flow_tensor_checks(tampered)
            self.assertFalse(checks["sample_target_logical_equality"], key)

    def test_receipt_subcheck_and_normalization_policy_tamper_force_top_level_false(self):
        _, sample = self._sample(anchor=1)
        tensor, _ = self._tensor([sample])
        tampered = copy.deepcopy(tensor)
        tampered["flow_normalization_stats"]["features"]["logical.total_data"]["mask_policy"] = "known=true only"
        self.assertFalse(validate_flow_tensor_checks(tampered)["presence_aware_normalization_policy"])
        required = {name: True for name in ("raw_sample_semantic_equality", "sample_tensor_semantic_equality", "history_causal_flow_union", "stable_flow_identity", "target_only_future_flow_isolation", "epoch_isolation", "multi_hop_single_flow_tensor_identity", "presence_mask_correctness", "train_only_preprocessing", "presence_aware_normalization_policy", "no_silent_truncation", "input_return_categories", "depdata_runtime_zero", "deterministic_rebuild", "serialize_load", "scope")}
        required["sample_target_carrying_equality"] = False
        receipt = {"required_checks": required, "passed": True, "scope": {name: False for name in ("graph_builder", "information_graph", "physical_topology", "training", "gpu", "locked_test", "formal_dataset")}}
        result = validate_flow_acceptance(receipt)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
