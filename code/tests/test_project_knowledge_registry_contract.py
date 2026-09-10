from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_ROOT = WORKSPACE_ROOT / "docs" / "registries"
SCRIPT_PATH = WORKSPACE_ROOT / "code" / "scripts" / "build_project_knowledge_index_v1.py"
SPEC = importlib.util.spec_from_file_location("build_project_knowledge_index_v1", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load project knowledge index builder: {SCRIPT_PATH}")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ProjectKnowledgeRegistryContractTests(unittest.TestCase):
    def test_stable_human_and_machine_entrypoints_exist(self) -> None:
        required = (
            "docs/PROJECT_INDEX.md",
            "docs/LEARNING_PATH.md",
            "docs/RETRIEVAL_GUIDE.md",
            "docs/CODE_INDEX.md",
            "docs/ARCHITECTURE.md",
            "docs/RESEARCH_STATUS.md",
            "docs/EXPERIMENT_INDEX.md",
            "docs/RESULTS_INDEX.md",
            "docs/KNOWN_CONFLICTS.md",
            "docs/ARCHIVE_CANDIDATES.md",
            "docs/COLLABORATION_GUIDE.md",
            "docs/RESTRUCTURE_ACCEPTANCE.md",
            "docs/registries/document_authority.json",
            "docs/registries/experiment_registry.json",
            "docs/registries/results_registry.json",
            "docs/registries/historical_method_registry.json",
            "docs/registries/question_routes.json",
            "docs/registries/deferred_work.json",
        )
        for relative in required:
            self.assertTrue((WORKSPACE_ROOT / relative).is_file(), relative)

    def test_experiments_results_and_paths_are_bidirectionally_linked(self) -> None:
        experiment_payload = _read_json(REGISTRY_ROOT / "experiment_registry.json")
        result_payload = _read_json(REGISTRY_ROOT / "results_registry.json")
        experiments = {row["id"]: row for row in experiment_payload["experiments"]}
        self.assertEqual(len(experiments), len(experiment_payload["experiments"]))

        for result in result_payload["results"]:
            self.assertIn(result["experiment_id"], experiments)
            self.assertTrue((WORKSPACE_ROOT / result["audit"]).is_file())
            self.assertFalse(result["locked_test_accessed"])

        for experiment in experiments.values():
            for field in ("code", "protocol", "data", "checkpoint", "result", "audit"):
                relative = experiment.get(field)
                if relative:
                    self.assertTrue((WORKSPACE_ROOT / relative).exists(), f"{experiment['id']}:{field}")

    def test_every_important_experiment_has_a_complete_explicit_schema(self) -> None:
        payload = _read_json(REGISTRY_ROOT / "experiment_registry.json")
        self.assertEqual("PI-JWM-experiment-registry-v2", payload["schema_version"])
        required = {
            "id",
            "name",
            "date",
            "research_question",
            "method",
            "code",
            "code_version",
            "configuration",
            "parameters",
            "protocol",
            "data",
            "split",
            "seed",
            "checkpoint",
            "result",
            "metrics",
            "audit",
            "status",
            "conclusion",
            "claim_boundary",
            "locked_test_accessed",
            "field_notes",
        }
        for experiment in payload["experiments"]:
            self.assertEqual(set(), required - set(experiment), experiment["id"])
            for field in required - {"field_notes"}:
                if experiment[field] is None:
                    self.assertIn(field, experiment["field_notes"], experiment["id"])
                    self.assertTrue(experiment["field_notes"][field], experiment["id"])

    def test_registered_result_values_match_original_acceptance_evidence(self) -> None:
        experiment_payload = _read_json(REGISTRY_ROOT / "experiment_registry.json")
        result_payload = _read_json(REGISTRY_ROOT / "results_registry.json")

        report = builder.validate_result_registry_against_evidence(
            WORKSPACE_ROOT,
            result_payload,
            experiment_payload,
        )

        self.assertEqual(2, report["validated_result_count"])
        self.assertEqual([], report["mismatches"])

    def test_historical_methods_record_why_they_are_not_current(self) -> None:
        payload = _read_json(REGISTRY_ROOT / "historical_method_registry.json")
        self.assertEqual("PI-JWM-historical-method-registry-v1", payload["schema_version"])
        self.assertGreaterEqual(len(payload["methods"]), 5)
        required = {
            "id",
            "name",
            "period",
            "status",
            "research_question",
            "why_tried",
            "outcome",
            "why_not_current",
            "superseded_by",
            "code_patterns",
            "experiment_paths",
            "evidence_paths",
            "reusable_lessons",
            "claim_boundary",
            "keywords",
        }
        for method in payload["methods"]:
            self.assertEqual(set(), required - set(method), method["id"])
            self.assertNotEqual("current", method["status"])
            self.assertTrue(method["why_not_current"])
            for relative in method["experiment_paths"] + method["evidence_paths"]:
                self.assertTrue((WORKSPACE_ROOT / relative).exists(), f"{method['id']}:{relative}")

    def test_collaboration_rules_are_permanent_and_machine_routable(self) -> None:
        agents = (WORKSPACE_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        for phrase in (
            "Human-Driven Research Decisions",
            "Independent AI Assessment",
            "Explanation Ladder",
            "Evidence-Bounded Language",
        ):
            self.assertIn(phrase, agents)

        authority = _read_json(REGISTRY_ROOT / "document_authority.json")
        paths = {row["path"] for row in authority["documents"]}
        self.assertIn("docs/COLLABORATION_GUIDE.md", paths)

        routes = _read_json(REGISTRY_ROOT / "question_routes.json")
        route_ids = {row["id"] for row in routes["routes"]}
        self.assertTrue(
            {
                "ROUTE-CURRENT-METHOD",
                "ROUTE-EXPERIMENT-HISTORY",
                "ROUTE-RESULT-PROVENANCE",
                "ROUTE-DEFERRED-WORK",
            }.issubset(route_ids)
        )

    def test_deferred_work_is_guarded_against_automatic_execution(self) -> None:
        payload = _read_json(REGISTRY_ROOT / "deferred_work.json")
        builder.validate_deferred_work(payload)
        seed_item = next(row for row in payload["items"] if row["id"] == "P4-SEED-20260832")
        self.assertEqual(20260832, seed_item["seed"])
        self.assertFalse(seed_item["auto_start"])
        self.assertTrue(seed_item["authorization_required"])

    def test_generated_dependency_and_artifact_maps_keep_boundaries(self) -> None:
        generated = REGISTRY_ROOT / "generated"
        summary = _read_json(generated / "registry_summary.json")
        dependency = _read_json(generated / "python_dependency_map.json")
        self.assertEqual(0, summary["python_parse_error_count"])
        self.assertEqual("sealed", summary["locked_test_boundary"])
        self.assertFalse(summary["formal_performance_claim_ready"])
        self.assertGreater(summary["artifact_record_count"], 0)
        self.assertEqual(2, summary["verified_result_count"])
        self.assertEqual(0, summary["result_evidence_mismatch_count"])
        self.assertGreaterEqual(summary["historical_method_count"], 5)
        self.assertGreaterEqual(summary["question_route_count"], 4)
        self.assertEqual([], dependency["parse_errors"])

        nodes = {row["module"]: row for row in dependency["nodes"]}
        current = nodes["pi_jwm.formal_entity_aligned_rssm_world_model_v1"]
        self.assertEqual("current", current["lifecycle_status"])
        for imported in current["imports"]:
            if imported in nodes:
                self.assertNotEqual("historical", nodes[imported]["lifecycle_status"])


if __name__ == "__main__":
    unittest.main()
