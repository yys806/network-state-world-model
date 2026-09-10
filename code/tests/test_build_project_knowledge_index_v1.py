from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_project_knowledge_index_v1.py"
SPEC = importlib.util.spec_from_file_location("build_project_knowledge_index_v1", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load project knowledge index builder: {SCRIPT_PATH}")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class ProjectKnowledgeIndexBuilderTests(unittest.TestCase):
    def test_classifies_current_support_prototype_and_historical_code(self) -> None:
        self.assertEqual(
            "current",
            builder.classify_tracked_path(
                "code/src/pi_jwm/formal_entity_aligned_rssm_world_model_v1.py"
            )["lifecycle_status"],
        )
        self.assertEqual(
            "support",
            builder.classify_tracked_path(
                "code/src/pi_jwm/formal_airfogsim_window_v1.py"
            )["lifecycle_status"],
        )
        self.assertEqual(
            "prototype",
            builder.classify_tracked_path(
                "code/src/pi_jwm/formal_candidate_rollout_planner_v1.py"
            )["lifecycle_status"],
        )
        self.assertEqual(
            "historical",
            builder.classify_tracked_path("code/src/pi_jwm/v11_strategy_selector.py")[
                "lifecycle_status"
            ],
        )

    def test_extracts_only_local_python_dependencies(self) -> None:
        source = """
from pi_jwm.formal_p4_gate_v1 import evaluate
from pi_jwm import formal_motion_state_v1
import pi_jwm.formal_world_model_loss_v1
import torch
"""
        self.assertEqual(
            {
                "pi_jwm.formal_motion_state_v1",
                "pi_jwm.formal_p4_gate_v1",
                "pi_jwm.formal_world_model_loss_v1",
            },
            builder.extract_local_imports(source),
        )

    def test_dependency_map_accepts_utf8_bom_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "code" / "scripts" / "legacy.py"
            source.parent.mkdir(parents=True)
            source.write_text("import pi_jwm.formal_p4_gate_v1\n", encoding="utf-8-sig")
            inventory = [
                {
                    "path": "code/scripts/legacy.py",
                    "category": "runnable_script",
                    "lifecycle_status": "support",
                    "evidence_role": "execution_entry",
                }
            ]

            result = builder.build_dependency_map(root, inventory)

            self.assertEqual([], result["parse_errors"])

    def test_deferred_seed_registry_requires_explicit_authorization(self) -> None:
        valid = {
            "schema_version": "PI-JWM-deferred-work-v1",
            "items": [
                {
                    "id": "P4-SEED-20260832",
                    "status": "deferred",
                    "authorization_required": True,
                    "auto_start": False,
                    "locked_test_accessed": False,
                    "entrypoint": "code/scripts/run_formal_p4_entity_rssm_gpu_v1.py",
                }
            ],
        }
        builder.validate_deferred_work(valid)

        invalid = json.loads(json.dumps(valid))
        invalid["items"][0]["auto_start"] = True
        with self.assertRaisesRegex(ValueError, "auto_start"):
            builder.validate_deferred_work(invalid)

    def test_artifact_record_reads_control_files_without_hashing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "experiments" / "run-a"
            artifact.mkdir(parents=True)
            (artifact / "run_summary.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "seed": 20260831,
                        "method": "entity_aligned_dual_graph_rssm_v1",
                        "locked_test_accessed": False,
                    }
                ),
                encoding="utf-8-sig",
            )
            (artifact / "manifest.json").write_text("{}", encoding="utf-8")
            checkpoint = artifact / "checkpoints" / "large.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint-placeholder")

            record = builder.build_artifact_record(root, artifact)

            self.assertEqual("experiments/run-a", record["artifact_id"])
            self.assertEqual("passed", record["status"])
            self.assertEqual("20260831", record["seed"])
            self.assertEqual("false", record["locked_test_accessed"])
            self.assertEqual("experiments/run-a/manifest.json", record["manifest_path"])
            self.assertNotIn("large.pt", record["hashed_control_files"])

    def test_archive_candidates_keep_hashes_and_current_reference_count(self) -> None:
        inventory = [
            {
                "path": "code/src/pi_jwm/v11_old.py",
                "size_bytes": 12,
                "sha256": "a" * 64,
                "lifecycle_status": "historical",
            }
        ]
        dependencies = {
            "nodes": [
                {
                    "module": "pi_jwm.v11_old",
                    "path": "code/src/pi_jwm/v11_old.py",
                    "lifecycle_status": "historical",
                    "imported_by": ["pi_jwm.formal_p4_gate_v1", "tests.test_v11_old"],
                    "test_files": ["code/tests/test_v11_old.py"],
                },
                {
                    "module": "pi_jwm.formal_p4_gate_v1",
                    "path": "code/src/pi_jwm/formal_p4_gate_v1.py",
                    "lifecycle_status": "current",
                    "imported_by": [],
                    "test_files": [],
                },
            ]
        }

        rows = builder.build_archive_candidates(inventory, dependencies)

        self.assertEqual(1, len(rows))
        self.assertEqual("a" * 64, rows[0]["sha256"])
        self.assertEqual(2, rows[0]["imported_by_count"])
        self.assertEqual(1, rows[0]["current_imported_by_count"])
        self.assertEqual("keep_in_place_logical_archive", rows[0]["proposed_action"])

    def test_result_validation_detects_metric_drift_from_original_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            gates = [
                {"name": gate_name, "value": float(index)}
                for index, gate_name in enumerate(builder.RESULT_METRIC_TO_GATE.values(), start=1)
            ]
            audit_payload = {
                "seed": 7,
                "status": "passed",
                "selection": {"best_epoch": 3},
                "gate_recompute": {"recomputed": {"gates": gates}},
                "locked_test_accessed": False,
                "formal_performance_claim_ready": False,
            }
            audit.write_text(json.dumps(audit_payload), encoding="utf-8")
            results = {
                "schema_version": "PI-JWM-results-registry-v2",
                "results": [
                    {
                        "id": "R1",
                        "experiment_id": "E1",
                        "seed": 7,
                        "status": "passed_single_seed",
                        "selected_epoch": 3,
                        "metrics": {
                            metric_name: float(index)
                            for index, metric_name in enumerate(
                                builder.RESULT_METRIC_TO_GATE, start=1
                            )
                        },
                        "audit": "audit.json",
                        "audit_sha256": builder._sha256(audit),
                        "locked_test_accessed": False,
                    }
                ],
            }
            experiments = {"experiments": [{"id": "E1", "seed": 7}]}

            clean = builder.validate_result_registry_against_evidence(root, results, experiments)
            self.assertEqual([], clean["mismatches"])

            results["results"][0]["metrics"]["node_x_h20_ratio"] = 999.0
            drifted = builder.validate_result_registry_against_evidence(root, results, experiments)
            self.assertIn("R1: metric mismatch node_x_h20_ratio", drifted["mismatches"])


if __name__ == "__main__":
    unittest.main()
