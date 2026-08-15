from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_p2_multistep_collector_preflight_v1.py"
)


def load_runner():
    name = "run_p2_multistep_collector_preflight_v1"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class P2MultistepCollectorRunnerTests(unittest.TestCase):
    def test_real_payload_executes_three_steps_and_aligns_history(self):
        runner = load_runner()
        payloads = runner.build_real_payloads(seed=0)
        frames = payloads["trajectory_frames.json"]["frames"]

        self.assertEqual(3, len(frames))
        self.assertEqual([0, 1, 2], [row["frame_index"] for row in frames])
        self.assertEqual(1, len(frames[0]["action"]["offloads"]))
        self.assertGreater(len(frames[0]["action"]["assignment_coo"]), 0)
        self.assertEqual([], frames[1]["action"]["offloads"])
        self.assertEqual([], frames[1]["action"]["assignment_coo"])
        self.assertEqual([], frames[2]["action"]["offloads"])
        self.assertEqual([], frames[2]["action"]["assignment_coo"])
        self.assertGreater(frames[0]["outcome_link"][0]["served_data"], 0.0)
        self.assertEqual(
            frames[0]["outcome_link"], frames[1]["pre_link_history_source"]
        )
        self.assertEqual(
            frames[1]["outcome_link"], frames[2]["pre_link_history_source"]
        )
        self.assertEqual(
            [0.0, 0.0, 0.0], frames[2]["pre_link_history"][0]["values"]
        )
        self.assertTrue(frames[2]["pre_link_history"][0]["valid"])
        self.assertEqual([0, 0, 0], [row["edge_index"] for row in frames])
        self.assertTrue(
            all(
                row["decision_time_channel"]["capture_phase"]
                == "before_action_setters"
                for row in frames
            )
        )
        self.assertEqual(
            3, payloads["summary.json"]["real_airfogsim_step_count"]
        )
        self.assertTrue(
            payloads["summary.json"]["status_flags"][
                "multistep_real_airfogsim_executed"
            ]
        )

    def test_fixture_has_strict_three_frame_history_semantics(self):
        runner = load_runner()
        payloads = runner.fake_passing_payloads_for_test()
        frames = payloads["trajectory_frames.json"]["frames"]

        self.assertEqual([0, 1, 2], [row["frame_index"] for row in frames])
        self.assertFalse(frames[0]["pre_link_history"][0]["valid"])
        self.assertEqual("no_history", frames[0]["pre_link_history"][0]["missing_reason"])
        self.assertEqual(
            frames[0]["outcome_link"], frames[1]["pre_link_history_source"]
        )
        self.assertTrue(frames[2]["pre_link_history"][0]["valid"])
        self.assertEqual("none", frames[2]["pre_link_history"][0]["missing_reason"])
        self.assertEqual(
            [0.0, 0.0, 0.0], frames[2]["pre_link_history"][0]["values"]
        )

    def test_writer_and_verifier_recompute_multistep_contract(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            runner.write_preflight_bundle(output_dir, runner.fake_passing_payloads_for_test())
            self.assertTrue(runner.verify_preflight_bundle(output_dir)["passed"])

            frame_path = output_dir / "trajectory_frames.json"
            data = json.loads(frame_path.read_text(encoding="utf-8"))
            data["frames"][1]["pre_link_history"][0]["values"][1] += 1.0
            frame_path.write_text(json.dumps(data), encoding="utf-8")
            report = runner.verify_preflight_bundle(output_dir)
            self.assertFalse(report["passed"])
            self.assertTrue(any("history" in row for row in report["errors"]))

    def test_writer_rejects_reordered_trace_and_changed_edge_index(self):
        runner = load_runner()
        for mutation in ("trace", "edge_index"):
            payloads = runner.fake_passing_payloads_for_test()
            if mutation == "trace":
                payloads["trajectory_frames.json"]["frames"][1]["temporal_trace"] = [
                    "action_setters_called",
                    "decision_time_observation_captured",
                ]
            else:
                payloads["trajectory_frames.json"]["frames"][1]["edge_index"] = 1
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "multistep evidence"):
                    runner.write_preflight_bundle(Path(tmp), payloads)
                self.assertFalse((Path(tmp) / "manifest.json").exists())

    def test_writer_rejects_cross_frame_node_or_flow_reindexing(self):
        runner = load_runner()
        for vocabulary_name, identity in (("node_indices", "uav0"), ("flow_indices", "task0")):
            payloads = runner.fake_passing_payloads_for_test()
            payloads["vocabularies.json"]["frame_snapshots"][1][vocabulary_name] = {
                identity: 7
            }
            with self.subTest(vocabulary=vocabulary_name), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "vocabulary"):
                    runner.write_preflight_bundle(Path(tmp), payloads)

    def test_writer_rejects_action_coo_identity_mismatch(self):
        runner = load_runner()
        payloads = runner.fake_passing_payloads_for_test()
        payloads["trajectory_frames.json"]["frames"][0]["action"] = {
            "offloads": [{"task_id": "task0"}],
            "assignment_coo": [[0, 0, 1, 0]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "COO identity"):
                runner.write_preflight_bundle(Path(tmp), payloads)

    def test_writer_rejects_valid_zero_marked_missing(self):
        runner = load_runner()
        payloads = runner.fake_passing_payloads_for_test()
        row = payloads["trajectory_frames.json"]["frames"][2]["pre_link_history"][0]
        row["valid"] = False
        row["missing_reason"] = "source_absent"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "multistep evidence"):
                runner.write_preflight_bundle(Path(tmp), payloads)

    def test_writer_rejects_unsafe_status_flags(self):
        runner = load_runner()
        for name in ("gpu_started", "training_eligible", "v4_collector_implemented"):
            payloads = runner.fake_passing_payloads_for_test()
            payloads["summary.json"]["status_flags"][name] = True
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "multistep evidence"):
                    runner.write_preflight_bundle(Path(tmp), payloads)

    def test_writer_failure_is_atomic(self):
        runner = load_runner()
        payloads = runner.fake_passing_payloads_for_test()
        payloads["validation_report.json"]["passed"] = False
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "canonical"
            with self.assertRaisesRegex(ValueError, "validation"):
                runner.write_preflight_bundle(output_dir, payloads)
            self.assertFalse(output_dir.exists())
            self.assertEqual([], list(Path(tmp).glob(".*.tmp-*")))

    def test_manifest_source_keys_are_portable_project_relative_paths(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            runner.write_preflight_bundle(output_dir, runner.fake_passing_payloads_for_test())
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            keys = tuple(manifest["source_hashes"])
            self.assertTrue(any(key.startswith("code/") for key in keys))
            self.assertTrue(all("浠ｇ爜" not in key for key in keys))
            self.assertTrue(all((runner.PROJECT_ROOT / Path(key)).is_file() for key in keys))

    def test_manifest_binds_transitive_single_step_dependency_closure(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            runner.write_preflight_bundle(output_dir, runner.fake_passing_payloads_for_test())
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(
                {
                    "code/scripts/small_experiments/airfogsim_strict_dual_graph_preflight.py",
                    "code/src/pi_jwm/airfogsim_contract_adapter.py",
                    "code/src/pi_jwm/airfogsim_cpu_inner_rule_v1.py",
                    "code/src/pi_jwm/cpu_inner_rule_v1.py",
                    "code/src/pi_jwm/information_edge_contract_v4.py",
                    "code/src/pi_jwm/single_step_collector_contract_v1.py",
                    "code/tests/test_airfogsim_contract_adapter.py",
                    "code/tests/test_airfogsim_cpu_inner_rule_v1.py",
                    "code/tests/test_cpu_inner_rule_v1.py",
                    "code/tests/test_information_edge_contract_v4.py",
                    "code/tests/test_single_step_collector_contract_v1.py",
                    "code/tests/small_experiments/test_airfogsim_strict_dual_graph_preflight.py",
                }.issubset(manifest["source_hashes"])
            )


if __name__ == "__main__":
    unittest.main()
