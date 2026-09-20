import copy
import json
import tempfile
import unittest
from pathlib import Path

from pi_jwm.step4_1_pi_graph_mapping_v1 import (
    SCHEMA_VERSION,
    build_mapping,
    load_mapping,
    save_mapping,
    validate_mapping_checks,
)


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "code/artifacts/protocols/pi_jwm_communication_outcome_semantics_v2_20260919/real_communication_outcome_semantics.json"
TENSOR_SCHEMA = ROOT / "code/artifacts/protocols/pi_jwm_step3_3_model_input_tensor_v1_20260919/tensor_schema.json"


class Step41PIGraphMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(RAW.read_text(encoding="utf-8"))
        cls.tensor_schema = json.loads(TENSOR_SCHEMA.read_text(encoding="utf-8"))

    def test_mapping_covers_required_objects_relations_and_identity(self):
        mapping = build_mapping()
        self.assertEqual(mapping["schema_version"], SCHEMA_VERSION)
        self.assertEqual(
            set(mapping["graph_contract"]["information"]["node_types"]),
            {"agent", "task"},
        )
        self.assertEqual(
            set(mapping["graph_contract"]["information"]["relation_types"]),
            {"comm", "task_agent_src", "task_agent_host", "task_agent_exec", "task_agent_ret", "flow", "dag"},
        )
        self.assertEqual(mapping["identity_contract"]["physical_index_source"], "static.input_entity_index.physical")
        self.assertEqual(mapping["identity_contract"]["agent_index_source"], "static.input_entity_index.physical")
        self.assertFalse(mapping["scope"]["graph_builder_implemented"])

    def test_current_evidence_and_minimum_gaps_are_machine_checked(self):
        mapping = build_mapping()
        checks = validate_mapping_checks(mapping, self.raw, self.tensor_schema)
        self.assertTrue(checks["passed"], checks)
        gaps = {row["item"]: row["availability"] for row in mapping["required_additive_data_extensions"]}
        self.assertEqual(gaps["physical.position_m"], "RAW_AVAILABLE_BUT_NOT_EXPOSED")
        self.assertEqual(gaps["comm.wireless_csi"], "RAW_AVAILABLE_BUT_NOT_EXPOSED")
        self.assertEqual(gaps["flow.current_state"], "RAW_INSUFFICIENT")
        self.assertEqual(gaps["comm.wired_relation"], "RAW_AVAILABLE_BUT_NOT_EXPOSED")
        self.assertEqual(gaps["comm.wired_optional_dynamic_state"], "RAW_INSUFFICIENT")
        optional = next(row for row in mapping["required_additive_data_extensions"] if row["item"] == "comm.wired_optional_dynamic_state")
        self.assertFalse(optional["minimum_for_03"])
        self.assertNotIn("comm.wired_optional_dynamic_state", mapping["graph_readiness"]["blocking_gaps"])
        self.assertFalse(mapping["graph_readiness"]["minimum_definition_supported_by_current_tensor"])

    def test_wired_relation_uses_type_and_mask_without_fabricated_csi(self):
        mapping = build_mapping()
        comm = next(row for row in mapping["relations"] if row["relation"] == "comm")
        wired = comm["type_contracts"]["wired"]
        self.assertEqual(wired["relation_source"], "environment.wired_edges / WiredNetworkManager.hasLink")
        self.assertEqual(wired["csi_feature_mask"], False)
        self.assertIsNone(wired["csi_value"])
        self.assertEqual(wired["relation_type"], "wired")
        self.assertTrue(wired["minimum_relation_fields_complete_without_csi"])

    def test_cpu_capacity_allocation_service_and_availability_are_distinct(self):
        mapping = build_mapping()
        contract = mapping["cpu_semantic_contract"]
        self.assertEqual(contract["capacity"]["graph_role"], "information_agent.static_capability")
        self.assertEqual(contract["allocation"]["graph_role"], "comp_action")
        self.assertEqual(contract["actual_service"]["graph_role"], "execution_outcome")
        self.assertEqual(contract["available_cpu"]["graph_role"], "information_agent.dynamic_resource")
        self.assertEqual(contract["available_cpu"]["availability"], "RAW_INSUFFICIENT")
        self.assertEqual(len({row["graph_role"] for row in contract.values()}), 4)

    def test_forbidden_placements_and_flow_event_distinction(self):
        mapping = build_mapping()
        forbidden = {(row["fact"], row["forbidden_role"]) for row in mapping["forbidden_wrong_placements"]}
        self.assertIn(("CSI", "physical_edge_feature"), forbidden)
        self.assertIn(("past_hop_service", "current_flow_remaining_state"), forbidden)
        self.assertIn(("position_or_speed", "information_agent_resource_feature"), forbidden)
        flow = next(row for row in mapping["relations"] if row["relation"] == "flow")
        self.assertEqual(flow["state_source_status"], "RAW_INSUFFICIENT")
        self.assertNotIn("past_outcome_flow_service", flow["allowed_current_state_sources"])

    def test_negative_semantic_tamper_fails_receipt(self):
        mapping = build_mapping()
        broken = copy.deepcopy(mapping)
        csi = next(row for row in broken["field_mappings"] if row["field_id"] == "comm.csi")
        csi["graph_role"] = "physical_edge.dynamic_feature"
        checks = validate_mapping_checks(broken, self.raw, self.tensor_schema)
        self.assertFalse(checks["forbidden_placement_absent"])
        self.assertFalse(checks["passed"])

        broken_cpu = copy.deepcopy(mapping)
        capacity = next(row for row in broken_cpu["field_mappings"] if row["field_id"] == "agent.cpu_capacity_per_s")
        capacity["graph_role"] = "information_agent.dynamic_resource"
        checks = validate_mapping_checks(broken_cpu, self.raw, self.tensor_schema)
        self.assertFalse(checks["cpu_semantics_distinct"])
        self.assertFalse(checks["passed"])

    def test_round_trip(self):
        mapping = build_mapping()
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "mapping.json"
            save_mapping(mapping, path)
            loaded = load_mapping(path)
        self.assertEqual(mapping, loaded)


if __name__ == "__main__":
    unittest.main()
