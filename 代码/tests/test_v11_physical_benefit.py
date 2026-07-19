import sys
import unittest
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "src"))


class PhysicalAlignmentTest(unittest.TestCase):
    def test_aligns_only_exact_input_end_times(self):
        from pi_jwm.v11_physical_benefit import align_decision_points

        aligned, rejected = align_decision_points(
            [
                {"seed": 0, "decision_time": 0.8},
                {"seed": 0, "decision_time": 0.7},
            ],
            [{"sample_id": 10, "seed": 0, "input_end_time": 0.8}],
        )

        self.assertEqual(aligned, [{"seed": 0, "decision_time": 0.8, "sample_id": 10}])
        self.assertEqual(rejected[0]["reason"], "no_exact_input_end_time")

    def test_rejects_duplicate_sample_time_keys(self):
        from pi_jwm.v11_physical_benefit import align_decision_points

        with self.assertRaisesRegex(ValueError, "duplicate sample time"):
            align_decision_points(
                [{"seed": 0, "decision_time": 0.8}],
                [
                    {"sample_id": 10, "seed": 0, "input_end_time": 0.8},
                    {"sample_id": 11, "seed": 0, "input_end_time": 0.8},
                ],
            )

    def test_normalizes_only_supported_physical_families(self):
        from pi_jwm.v11_physical_benefit import normalize_physical_family

        self.assertEqual(normalize_physical_family("default"), "identity_control")
        self.assertEqual(normalize_physical_family("rb_count"), "rb_repair")
        self.assertEqual(normalize_physical_family("mixed_offload_rb"), "offload_rb")
        self.assertEqual(normalize_physical_family("cpu_scale"), "compute_cpu")
        self.assertEqual(normalize_physical_family("return_route"), "return_route")
        self.assertIsNone(normalize_physical_family("offload_target"))


class PhysicalDescriptorTest(unittest.TestCase):
    def test_physical_descriptor_has_fixed_leakage_safe_order(self):
        from pi_jwm.v11_physical_benefit import (
            COMMON_DESCRIPTOR_NAMES,
            physical_action_descriptor,
        )

        values, names = physical_action_descriptor(
            {
                "action_family": "mixed_offload_rb",
                "rb_total": 40.0,
                "num_rb_tasks": 2,
                "cpu_total": 5.0,
                "num_cpu_overrides": 1,
                "num_offload_overrides": 1,
                "num_return_route_overrides": 0,
                "intervention_start_step": 1,
                "temporal_pattern": "persistent",
            }
        )

        self.assertEqual(names, COMMON_DESCRIPTOR_NAMES)
        self.assertEqual(values.shape, (len(COMMON_DESCRIPTOR_NAMES),))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values[names.index("family_offload")], 1.0)
        self.assertEqual(values[names.index("pattern_persistent")], 1.0)
        self.assertFalse(
            any(
                token in name
                for name in names
                for token in ("actual", "future", "oracle", "outcome", "seed")
            )
        )

    def test_physical_descriptor_rejects_unsupported_family(self):
        from pi_jwm.v11_physical_benefit import physical_action_descriptor

        with self.assertRaisesRegex(ValueError, "unsupported physical action family"):
            physical_action_descriptor({"action_family": "offload_target"})

    def test_selector_descriptor_matches_common_contract(self):
        from pi_jwm.v11_physical_benefit import (
            COMMON_DESCRIPTOR_NAMES,
            selector_action_descriptors,
        )
        from pi_jwm.v11_selector import CandidateBatch

        names = (
            "rb_total_sum",
            "rb_action_count",
            "cpu_total_sum",
            "cpu_action_count",
            "offload_action_count",
            "return_action_count",
            "action_family_identity",
            "action_family_rb",
            "action_family_offload",
            "action_family_compute",
            "action_family_return",
            "action_family_historical",
        )
        features = np.zeros((1, 2, len(names)), dtype=np.float32)
        features[0, 0, names.index("action_family_identity")] = 1.0
        features[0, 1, names.index("rb_total_sum")] = 12.0
        features[0, 1, names.index("rb_action_count")] = 2.0
        features[0, 1, names.index("action_family_rb")] = 1.0
        batch = CandidateBatch(
            context=np.zeros((1, 1), dtype=np.float32),
            candidate_features=features,
            candidate_mask=np.ones((1, 2), dtype=bool),
            stage=np.asarray(["offload"]),
            feature_names=names,
            candidate_names=("identity", "rb_repair__k8__q50__decayed"),
            context_feature_names=("context",),
        )

        descriptors, descriptor_names = selector_action_descriptors(batch)

        self.assertEqual(descriptor_names, COMMON_DESCRIPTOR_NAMES)
        self.assertEqual(descriptors.shape, (1, 2, len(COMMON_DESCRIPTOR_NAMES)))
        self.assertEqual(descriptors[0, 0, descriptor_names.index("family_control")], 1.0)
        self.assertEqual(descriptors[0, 1, descriptor_names.index("family_rb")], 1.0)
        self.assertEqual(descriptors[0, 1, descriptor_names.index("pattern_decayed")], 1.0)
        self.assertEqual(descriptors[0, 1, descriptor_names.index("intervention_start_step")], 1.0)


if __name__ == "__main__":
    unittest.main()
