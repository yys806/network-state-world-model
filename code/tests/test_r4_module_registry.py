from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class R4ModuleRegistryTests(unittest.TestCase):
    def test_registry_covers_every_frozen_module_family(self):
        from pi_jwm.r4_module_registry import MODULE_FAMILIES, candidate_registry

        registry = candidate_registry()
        self.assertEqual(
            {
                "field_encoder",
                "graph_encoder",
                "coupling",
                "dynamics",
                "head",
                "dag",
                "presence",
            },
            set(MODULE_FAMILIES),
        )
        for family in MODULE_FAMILIES:
            family_specs = [spec for spec in registry.values() if spec.family == family]
            self.assertEqual(1, sum(spec.status == "reference" for spec in family_specs))
            self.assertGreaterEqual(len(family_specs), 2)

    def test_reference_config_reuses_r3_components_and_is_executable(self):
        from pi_jwm.r3_world_model import REFERENCE_COMPONENTS
        from pi_jwm.r4_module_registry import (
            assert_executable_config,
            reference_r4_config,
        )

        config = reference_r4_config(hidden_dim=8, history_steps=2)
        for family, name in REFERENCE_COMPONENTS.items():
            self.assertEqual(name, getattr(config, family))
        self.assertEqual("dag_summary_v1", config.dag)
        self.assertEqual("fixed_observed_presence_v1", config.presence)
        assert_executable_config(config)

    def test_single_module_candidate_changes_only_requested_family(self):
        from pi_jwm.r4_module_registry import (
            make_single_module_config,
            reference_r4_config,
            validate_controlled_config,
        )

        reference = reference_r4_config(hidden_dim=8, history_steps=2)
        candidate = make_single_module_config(
            "field_encoder",
            "symlog_masked_mlp_v1",
            hidden_dim=8,
            history_steps=2,
        )
        validate_controlled_config(
            candidate,
            allow_statuses={"reference", "planned", "executable"},
        )
        changed = {
            family
            for family in reference.component_names()
            if getattr(reference, family) != getattr(candidate, family)
        }
        self.assertEqual({"field_encoder"}, changed)

    def test_unknown_and_nonexecutable_candidates_fail_explicitly(self):
        from pi_jwm.r4_module_registry import (
            assert_executable_config,
            make_single_module_config,
        )

        with self.assertRaisesRegex(ValueError, "unknown R4 candidate"):
            make_single_module_config("field_encoder", "invented_encoder_v1")

        deferred = make_single_module_config(
            "dynamics",
            "transformer_dynamics_v1",
        )
        with self.assertRaisesRegex(ValueError, "not executable"):
            assert_executable_config(deferred)

    def test_controlled_config_rejects_multiple_changed_families(self):
        from pi_jwm.r4_module_registry import (
            reference_r4_config,
            validate_controlled_config,
        )

        invalid = replace(
            reference_r4_config(),
            field_encoder="symlog_masked_mlp_v1",
            graph_encoder="rgcn_v1",
        )
        with self.assertRaisesRegex(ValueError, "one module family"):
            validate_controlled_config(
                invalid,
                allow_statuses={"reference", "planned", "executable"},
            )

    def test_candidate_matrix_is_machine_readable_and_records_evidence(self):
        from pi_jwm.r4_module_registry import candidate_matrix

        matrix = candidate_matrix()
        json.dumps(matrix, ensure_ascii=False)
        self.assertTrue(matrix)
        for row in matrix:
            self.assertEqual(
                {
                    "family",
                    "name",
                    "status",
                    "evidence",
                    "question",
                },
                set(row),
            )
            self.assertTrue(row["evidence"])
            self.assertTrue(row["question"])


if __name__ == "__main__":
    unittest.main()
