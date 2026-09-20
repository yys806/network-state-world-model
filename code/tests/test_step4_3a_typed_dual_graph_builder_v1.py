import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import (
    REQUIRED_ACCEPTANCE_CHECKS,
    PhysicalTopologyConfig,
    build_typed_dual_graph_batch,
    load_typed_dual_graph_batch,
    save_typed_dual_graph_batch,
    validate_acceptance_receipt,
    validate_typed_dual_graph_checks,
)


ROOT = Path(__file__).resolve().parents[2]
TENSOR_PATH = ROOT / "code/artifacts/protocols/pi_jwm_step4_2c_c_flow_sample_tensor_v1_20260920/flow_tensor.npz"


class Step43ATypedDualGraphBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tensor = load_flow_tensor_batch(TENSOR_PATH)
        cls.config = PhysicalTopologyConfig(mode="radius_knn", radius_m=1000.0, k=2)
        cls.graph = build_typed_dual_graph_batch(cls.tensor, cls.config)

    def test_required_typed_blocks_and_validation(self):
        expected = {
            "physical_nodes", "physical_relations", "agent_nodes", "task_nodes",
            "comm_relations", "task_agent_relations", "flow_relations", "dag_relations",
            "flow_carrying_state", "align_relations", "geo_comm_relations",
        }
        self.assertTrue(expected <= set(self.graph["blocks"]))
        checks = validate_typed_dual_graph_checks(self.graph, self.tensor)
        self.assertTrue(checks["passed"], checks)

    def test_current_graph_uses_last_history_frame_not_target(self):
        changed = copy.deepcopy(self.tensor)
        changed["target_entity_features"][:] = 999999.0
        changed["target_logical_flow_epoch"][:] = 777
        rebuilt = build_typed_dual_graph_batch(changed, self.config)
        for name, block in self.graph["blocks"].items():
            for key, value in block.items():
                if isinstance(value, np.ndarray):
                    self.assertTrue(np.array_equal(value, rebuilt["blocks"][name][key]), (name, key))

    def test_physical_topology_is_independent_of_information_relations(self):
        changed = copy.deepcopy(self.tensor)
        for key in ("comm_relation_presence", "task_agent_validity_mask", "logical_flow_known_mask", "dag_mask"):
            changed[key][:] = False
        rebuilt = build_typed_dual_graph_batch(changed, self.config)
        for key, value in self.graph["blocks"]["physical_relations"].items():
            if isinstance(value, np.ndarray):
                self.assertTrue(np.array_equal(value, rebuilt["blocks"]["physical_relations"][key]), key)

    def test_physical_membership_uses_spatial_eligibility_not_entity_class(self):
        changed = copy.deepcopy(self.tensor)
        bi, ei = 0, 0
        changed["entity_presence"][bi, -1, ei] = True
        changed["entity_position_mask"][bi, -1, ei, :] = False
        graph = build_typed_dual_graph_batch(changed, self.config)
        self.assertFalse(graph["blocks"]["physical_nodes"]["presence"][bi, ei])
        self.assertTrue(graph["blocks"]["agent_nodes"]["presence"][bi, ei])
        self.assertEqual(graph["blocks"]["physical_nodes"]["membership_reason_index"][bi, ei], 3)

    def test_all_topology_modes_are_explicit_and_development_only(self):
        for mode in ("radius", "knn", "radius_knn"):
            graph = build_typed_dual_graph_batch(self.tensor, PhysicalTopologyConfig(mode=mode, radius_m=1000, k=2))
            config = graph["contract"]["physical_topology"]
            self.assertEqual(config["mode"], mode)
            self.assertTrue(config["development_only"])
            self.assertFalse(config["research_frozen"])

    def test_comm_validity_is_independent_of_csi_mask(self):
        graph = copy.deepcopy(self.graph)
        block = graph["blocks"]["comm_relations"]
        pos = np.argwhere(block["presence"] & block["validity"])[0]
        bi, ri = map(int, pos)
        block["csi_mask"][bi, ri, :] = False
        block["csi"][bi, ri, :] = 0.0
        checks = validate_typed_dual_graph_checks(graph, self.tensor, compare_source=False)
        self.assertTrue(checks["presence_validity_mask_correct"], checks)

    def test_flow_endpoints_are_logical_not_carrying_hop_endpoints(self):
        block = self.graph["blocks"]["flow_relations"]
        carrying = self.graph["blocks"]["flow_carrying_state"]
        candidates = np.argwhere(block["known"] & carrying["known"] & carrying["active"])
        self.assertGreater(len(candidates), 0)
        bi, fi = map(int, candidates[0])
        self.assertEqual(block["source_index"][bi, fi], self.tensor["logical_flow_source_index"][bi, -1, fi])
        self.assertEqual(block["destination_index"][bi, fi], self.tensor["logical_flow_destination_index"][bi, -1, fi])

    def test_parallel_flow_rows_are_not_deduplicated(self):
        changed = copy.deepcopy(self.tensor)
        changed["logical_flow_known_mask"][0, -1, 1] = True
        changed["logical_flow_presence"][0, -1, 1] = True
        changed["logical_flow_index"][0, -1, 1] = 1
        changed["logical_flow_task_index"][0, -1, 1] = changed["logical_flow_task_index"][0, -1, 0]
        changed["logical_flow_source_index"][0, -1, 1] = changed["logical_flow_source_index"][0, -1, 0]
        changed["logical_flow_destination_index"][0, -1, 1] = changed["logical_flow_destination_index"][0, -1, 0]
        changed["logical_flow_type_index"][0, -1, 1] = changed["logical_flow_type_index"][0, -1, 0]
        changed["logical_flow_epoch"][0, -1, 1] = changed["logical_flow_epoch"][0, -1, 0] + 1
        graph = build_typed_dual_graph_batch(changed, self.config)
        self.assertEqual(int(graph["blocks"]["flow_relations"]["known"][0].sum()), 2)

    def test_geo_comm_does_not_require_physical_edge(self):
        graph = copy.deepcopy(self.graph)
        graph["blocks"]["physical_relations"]["presence"][:] = False
        checks = validate_typed_dual_graph_checks(graph, self.tensor, compare_source=False)
        self.assertTrue(checks["geo_comm_does_not_require_physical_edge"], checks)

    def test_wired_comm_can_be_valid_without_geocomm_or_csi(self):
        comm = self.graph["blocks"]["comm_relations"]
        geo = self.graph["blocks"]["geo_comm_relations"]
        wired = np.argwhere(comm["presence"] & comm["validity"] & (comm["relation_type_index"] == 3))
        self.assertGreater(len(wired), 0)
        bi, ri = map(int, wired[0])
        self.assertFalse(comm["csi_mask"][bi, ri].any())
        self.assertFalse(geo["validity"][bi, ri])

    def test_tampered_semantics_are_rejected(self):
        cases = []
        for block_name, field in (
            ("physical_nodes", "features"), ("agent_nodes", "entity_index"),
            ("task_nodes", "task_index"), ("comm_relations", "source_index"),
            ("task_agent_relations", "agent_index"), ("flow_relations", "destination_index"),
            ("dag_relations", "source_task_index"), ("align_relations", "agent_index"),
            ("geo_comm_relations", "comm_relation_index"),
        ):
            value = copy.deepcopy(self.graph)
            arr = value["blocks"][block_name][field]
            arr.flat[0] = arr.flat[0] + 1
            cases.append(value)
        for value in cases:
            with self.subTest():
                self.assertFalse(validate_typed_dual_graph_checks(value, self.tensor)["passed"])

    def test_masked_placeholder_and_padding_tamper_are_rejected(self):
        for block_name, feature_name, mask_name in (
            ("physical_nodes", "features", "feature_mask"),
            ("comm_relations", "csi", "csi_mask"),
            ("task_nodes", "features", "feature_mask"),
        ):
            value = copy.deepcopy(self.graph)
            block = value["blocks"][block_name]
            candidates = np.argwhere(~block[mask_name])
            pos = candidates[0] if len(candidates) else np.zeros(block[mask_name].ndim, dtype=int)
            block[mask_name][tuple(pos)] = False
            block[feature_name][tuple(pos)] = 3.0
            self.assertFalse(validate_typed_dual_graph_checks(value, self.tensor, compare_source=False)["passed"])

    def test_capacity_overflow_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "capacity overflow"):
            build_typed_dual_graph_batch(self.tensor, PhysicalTopologyConfig(max_physical_relations=0))

    def test_receipt_required_check_false_forces_top_level_failure(self):
        scope = {name: False for name in ("encoder", "gnn", "message_passing", "world_model", "loss", "planner", "training", "gpu", "locked_test", "formal_dataset")}
        required = {name: True for name in REQUIRED_ACCEPTANCE_CHECKS}
        required["logical_flow_relation_mapping"] = False
        receipt = {"required_checks": required, "scope": scope, "passed": True}
        self.assertFalse(validate_acceptance_receipt(receipt)["passed"])

    def test_serialize_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.npz"
            save_typed_dual_graph_batch(self.graph, path)
            loaded = load_typed_dual_graph_batch(path)
            self.assertTrue(validate_typed_dual_graph_checks(loaded, self.tensor)["passed"])


if __name__ == "__main__":
    unittest.main()
