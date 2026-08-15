from __future__ import annotations

import importlib
import importlib.util
import unittest

import numpy as np

from test_airfogsim_teacher_graph_v3 import source_graph


def subject():
    return importlib.import_module("pi_jwm.airfogsim_teacher_tensor_v3")


def teacher_graph():
    graph_module = importlib.import_module("pi_jwm.airfogsim_teacher_graph_v3")
    return graph_module.remap_teacher_aligned_graph(source_graph())


class TeacherAlignedTensorV3DiscoveryTests(unittest.TestCase):
    def test_teacher_aligned_tensor_module_exists(self):
        self.assertIsNotNone(
            importlib.util.find_spec("pi_jwm.airfogsim_teacher_tensor_v3")
        )


class TeacherAlignedTensorV3BehaviorTests(unittest.TestCase):
    def test_tensor_keeps_physical_and_information_edge_features_separate(self):
        module = subject()
        self.assertTrue(hasattr(module, "infer_teacher_tensor_contract"))
        self.assertTrue(hasattr(module, "tensorize_teacher_aligned_graph"))

        graph = teacher_graph()
        contract = module.infer_teacher_tensor_contract(
            [graph], history_steps=1, horizon_steps=1
        )
        arrays, report = module.tensorize_teacher_aligned_graph(graph, contract)

        self.assertFalse(
            {"csi_mean", "allocated_rb_count", "active_task_count", "rate_sum"}
            & set(module.PHYSICAL_EDGE_FEATURES)
        )
        info_index = report["information_edge_vocab"].index(
            "information_edge::vehicle_0::RSU_0::V2I"
        )
        csi_index = module.INFORMATION_EDGE_FEATURES.index("pre.csi_mean")
        rb_index = module.INFORMATION_EDGE_FEATURES.index(
            "action.allocated_rb_count"
        )
        rate_index = module.INFORMATION_EDGE_FEATURES.index("outcome.rate_sum")
        self.assertEqual(12.2, arrays["information_edge_state"][1, info_index, csi_index])
        self.assertEqual(1.0, arrays["information_edge_state"][1, info_index, rb_index])
        self.assertEqual(5.0, arrays["information_edge_state"][1, info_index, rate_index])

    def test_missing_wireless_fields_use_false_mask_and_zero_value(self):
        module = subject()
        self.assertTrue(hasattr(module, "infer_teacher_tensor_contract"))
        self.assertTrue(hasattr(module, "tensorize_teacher_aligned_graph"))
        graph = teacher_graph()
        contract = module.infer_teacher_tensor_contract([graph])
        arrays, report = module.tensorize_teacher_aligned_graph(graph, contract)
        info_index = report["information_edge_vocab"].index(
            "information_edge::vehicle_0::RSU_0::V2I"
        )
        sinr_index = module.INFORMATION_EDGE_FEATURES.index(
            "outcome.actual_sinr"
        )

        self.assertTrue(
            np.all(arrays["information_edge_state"][:, info_index, sinr_index] == 0.0)
        )
        self.assertFalse(
            np.any(
                arrays["information_edge_feature_mask"][:, info_index, sinr_index]
            )
        )

    def test_cip_cep_and_pending_cfl_are_materialized_as_indices(self):
        module = subject()
        self.assertTrue(hasattr(module, "infer_teacher_tensor_contract"))
        self.assertTrue(hasattr(module, "tensorize_teacher_aligned_graph"))
        graph = teacher_graph()
        contract = module.infer_teacher_tensor_contract([graph])
        arrays, report = module.tensorize_teacher_aligned_graph(graph, contract)
        info_index = report["information_edge_vocab"].index(
            "information_edge::vehicle_0::RSU_0::V2I"
        )
        physical_index = report["physical_edge_vocab"].index(
            "physical_edge::vehicle_0::RSU_0"
        )

        self.assertEqual(
            physical_index,
            arrays["cep_information_to_physical_edge_index"][info_index],
        )
        self.assertEqual([0, 1], arrays["cip_agent_node_index"].tolist())
        self.assertEqual(info_index, arrays["cfl_information_edge_index"][0])
        self.assertEqual([True, True], arrays["cfl_mask"][:, 0, info_index].tolist())
        self.assertIn("cfl_relations", report["counts"])
        self.assertEqual(1, report["counts"]["cfl_relations"])

    def test_tensor_validator_rejects_nonzero_masked_missing_value(self):
        module = subject()
        self.assertTrue(hasattr(module, "validate_teacher_tensors"))
        self.assertTrue(hasattr(module, "infer_teacher_tensor_contract"))
        self.assertTrue(hasattr(module, "tensorize_teacher_aligned_graph"))
        graph = teacher_graph()
        contract = module.infer_teacher_tensor_contract([graph])
        arrays, _ = module.tensorize_teacher_aligned_graph(graph, contract)
        sinr_index = module.INFORMATION_EDGE_FEATURES.index(
            "outcome.actual_sinr"
        )
        arrays["information_edge_state"][0, 0, sinr_index] = 1.0

        with self.assertRaisesRegex(ValueError, "masked information-edge values"):
            module.validate_teacher_tensors(arrays, contract)


if __name__ == "__main__":
    unittest.main()
