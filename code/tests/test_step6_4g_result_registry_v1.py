"""The repaired-domain Stage A registry must match accepted raw-derived evidence."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_project_knowledge_index_v1",
    ROOT / "code" / "scripts" / "build_project_knowledge_index_v1.py",
)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class RepairedStageARegistryTests(unittest.TestCase):
    def test_accepted_values_and_drift_detection(self):
        registry = ROOT / "docs" / "registries"
        experiments = json.loads((registry / "experiment_registry.json").read_text(encoding="utf-8"))
        full = json.loads((registry / "results_registry.json").read_text(encoding="utf-8"))
        row = next(
            item for item in full["results"]
            if item["id"] == "RES-STEP-6.4G-REPAIRED-STAGE-A-20261008"
        )
        payload = {"schema_version": full["schema_version"], "results": [row]}
        self.assertEqual(
            builder.validate_result_registry_against_evidence(ROOT, payload, experiments)["mismatches"],
            [],
        )
        changed = copy.deepcopy(payload)
        changed["results"][0]["metrics"]["S-CEM_vs_HRS_win"] += 1
        self.assertTrue(
            builder.validate_result_registry_against_evidence(ROOT, changed, experiments)["mismatches"]
        )
        changed = copy.deepcopy(payload)
        changed["results"][0]["selected_search_method"] = "MH-CEM"
        self.assertTrue(
            builder.validate_result_registry_against_evidence(ROOT, changed, experiments)["mismatches"]
        )


if __name__ == "__main__":
    unittest.main()
