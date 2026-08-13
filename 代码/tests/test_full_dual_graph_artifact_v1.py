from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.full_dual_graph_artifact_v1 import (  # noqa: E402
    REQUIRED_ARTIFACT_FILES,
    build_full_collector_status_flags,
    compare_replays,
    publish_atomic_bundle,
    validate_trajectory_frames,
)


def frames(*, fixture=False):
    rows = []
    for frame_index in range(2):
        rows.append(
            {
                "frame_index": frame_index,
                "fixture": fixture,
                "quarantined": False,
                "decision_snapshot": {
                    "phase": "decision",
                    "physical_edge_presence": [True, False],
                    "tasks": [
                        {
                            "task_id": "task0",
                            "lifecycle": "waiting_to_offload",
                        }
                    ],
                },
                "execution_snapshot": {
                    "phase": "execution",
                    "physical_edge_presence": [True, False],
                },
                "outcome_snapshot": {
                    "phase": "outcome",
                    "physical_edge_presence": [True, False],
                },
                "action": {
                    "decisions": [{"task_id": "task0", "selected": False}],
                    "hops": [],
                    "rb_allocations": [],
                },
                "decision_input_source_phases": {"channel": "decision"},
                "e1_history": {
                    "value": None if frame_index == 0 else [0.0, 0.0, 0.0],
                    "valid_mask": frame_index > 0,
                    "missing_reason": "NO_HISTORY" if frame_index == 0 else None,
                },
            }
        )
    return rows


VOCABULARY = {
    "physical_edge_indices": {"physical::a": 0, "physical::b": 1},
    "node_indices": {"uav0": 0},
    "task_indices": {"task0": 0},
    "dag_edge_indices": {},
    "flow_indices": {},
    "hop_indices": {},
}


class TrajectoryValidationTests(unittest.TestCase):
    def test_validates_phases_actionable_rows_history_and_presence_width(self):
        self.assertEqual([], validate_trajectory_frames(frames(), vocabulary=VOCABULARY, fixture=False))

    def test_rejects_contiguity_leak_fixture_mix_and_quarantine(self):
        bad = frames()
        bad[1]["frame_index"] = 3
        bad[0]["decision_input_source_phases"] = {"rate": "outcome"}
        bad[1]["fixture"] = True
        bad[1]["quarantined"] = True
        bad[0]["action"]["decisions"] = []
        bad[0]["decision_snapshot"]["physical_edge_presence"] = [True]

        errors = validate_trajectory_frames(bad, vocabulary=VOCABULARY, fixture=False)

        self.assertTrue(any("contiguous" in error for error in errors))
        self.assertTrue(any("outcome source" in error for error in errors))
        self.assertTrue(any("fixture" in error for error in errors))
        self.assertTrue(any("quarantined" in error for error in errors))
        self.assertTrue(any("missing decision" in error for error in errors))
        self.assertTrue(any("presence width" in error for error in errors))

    def test_rejects_cep_endpoint_mismatch_and_invalid_history_semantics(self):
        bad = frames()
        bad[0]["action"]["hops"] = [
            {
                "hop_id": "hop0",
                "physical_edge_id": "physical::a",
                "source_id": "wrong",
                "target_id": "b",
            }
        ]
        bad[0]["decision_snapshot"]["physical_edges"] = [
            {"edge_id": "physical::a", "source_id": "a", "target_id": "b"}
        ]
        bad[0]["e1_history"] = {
            "value": [0.0, 0.0, 0.0],
            "valid_mask": True,
            "missing_reason": None,
        }
        bad[1]["e1_history"] = {
            "value": None,
            "valid_mask": True,
            "missing_reason": None,
        }

        errors = validate_trajectory_frames(bad, vocabulary=VOCABULARY, fixture=False)

        self.assertTrue(any("CEP endpoints" in error for error in errors))
        self.assertTrue(any("first-frame E1" in error for error in errors))
        self.assertTrue(any("valid E1" in error for error in errors))


class ReplayAndPublicationTests(unittest.TestCase):
    def test_compare_replays_requires_exact_identity_and_reports_float_delta(self):
        reference = [{"flow_id": "f0", "value": 1.0}]
        replay = [{"flow_id": "f0", "value": 1.0 + 1e-10}]
        close = compare_replays(reference, replay)
        self.assertTrue(close["passed"])
        self.assertEqual([], close["exact_mismatches"])
        self.assertEqual(1, len(close["numeric_differences"]))

        changed = compare_replays(reference, [{"flow_id": "f1", "value": 1.0}])
        self.assertFalse(changed["passed"])
        self.assertTrue(changed["exact_mismatches"])

    def test_status_flags_only_promote_collector_implementation(self):
        self.assertEqual(
            {
                "v4_collector_implemented": True,
                "v4_dataset_complete": False,
                "training_eligible": False,
                "model_training_started": False,
                "gpu_started": False,
                "locked_test_accessed": False,
                "candidate_rollout_planner_complete": False,
                "final_method_frozen": False,
            },
            build_full_collector_status_flags(passed=True),
        )

    def test_publishes_atomically_with_manifest_and_refuses_missing_source_or_overwrite(self):
        payloads = {
            "collector_config.json": {"schema": "v4"},
            "vocabularies.json": VOCABULARY,
            "frames.jsonl": frames(),
            "coverage_report.json": {"fixture": False},
            "validation_report.json": {"passed": True},
            "replay_report.json": {"passed": True},
            "status_flags.json": build_full_collector_status_flags(passed=True),
        }
        source = SRC_ROOT / "pi_jwm" / "full_dual_graph_artifact_v1.py"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            publish_atomic_bundle(output, payloads, [source])
            self.assertEqual(set(REQUIRED_ARTIFACT_FILES), {path.name for path in output.iterdir()})
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("frames.jsonl", manifest["artifact_hashes"])
            self.assertTrue(manifest["source_hashes"])
            with self.assertRaises(FileExistsError):
                publish_atomic_bundle(output, payloads, [source])
            with self.assertRaises(FileNotFoundError):
                publish_atomic_bundle(Path(temporary) / "missing", payloads, [Path(temporary) / "none.py"])


if __name__ == "__main__":
    unittest.main()
