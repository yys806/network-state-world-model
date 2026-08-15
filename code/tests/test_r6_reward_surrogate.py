import unittest
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from pi_jwm.r6_reward_protocol import RewardScale
from pi_jwm.r6_reward_surrogate import (
    ACTION_MAGNITUDE_FIELDS,
    audit_template_support,
    build_reward_surrogate_preflight,
    build_factual_reward_arrays,
    derive_failure_events,
    summarize_factual_actions,
)


class R6RewardSurrogateTest(unittest.TestCase):
    def test_failure_event_is_counted_only_when_task_first_enters_failed(self):
        lifecycle = np.asarray(
            [
                [0, -1, -1],
                [4, 4, -1],
                [4, 4, 0],
                [4, 4, 4],
            ],
            dtype=np.int16,
        )
        present = lifecycle >= 0

        events = derive_failure_events(lifecycle, present)

        np.testing.assert_array_equal(events, np.asarray([0, 2, 0, 1]))

    def test_factual_action_summary_matches_candidate_magnitude_semantics(self):
        action = np.zeros((2, 3, 8), dtype=np.float32)
        present = np.zeros((2, 3), dtype=bool)
        action[0, 0, [0, 1, 3, 5, 6]] = [1.0, 1.0, 4.0, 1.0, 2.5]
        action[0, 1, [1, 3, 5, 6]] = [1.0, 6.0, 1.0, 3.5]
        present[0, :2] = True

        summary = summarize_factual_actions(action, present)

        self.assertEqual(
            ACTION_MAGNITUDE_FIELDS,
            ("offload_count", "rb_task_count", "rb_total", "cpu_task_count", "cpu_total"),
        )
        np.testing.assert_allclose(summary[0], [1.0, 2.0, 10.0, 2.0, 6.0])
        np.testing.assert_allclose(summary[1], np.zeros(5))

    def test_reward_rows_use_direct_outcomes_and_lifecycle_failure_transitions(self):
        teacher = {
            "time": np.asarray([0.1, 0.2], dtype=np.float32),
            "task_lifecycle_index": np.asarray([[0, -1], [4, 3]], dtype=np.int16),
            "task_present": np.asarray([[True, False], [True, True]]),
            "task_action": np.zeros((2, 2, 8), dtype=np.float32),
            "task_action_present": np.zeros((2, 2), dtype=bool),
        }
        target = {
            "time": np.asarray([0.1, 0.2], dtype=np.float32),
            "task_on_time_completion_event": np.asarray(
                [[False, False], [False, True]]
            ),
            "completed_task_delay": np.asarray([[0.0, 0.0], [0.0, 2.0]]),
            "completed_task_delay_valid": np.asarray(
                [[False, False], [False, True]]
            ),
            "delivered_data_total": np.asarray([0.0, 5.0]),
            "uav_energy_delta": np.asarray([[0.0], [3.0]]),
            "uav_energy_valid": np.asarray([[True], [True]]),
        }
        scale = RewardScale(
            completed_delay_p95=2.0,
            delivered_data_p95=10.0,
            energy_delta_p95=6.0,
            train_trajectory_count=1,
            delay_step_count=1,
            throughput_step_count=1,
            energy_step_count=1,
            source_manifest_sha256="a" * 64,
        )

        rows = build_factual_reward_arrays(teacher, target, scale=scale)

        np.testing.assert_array_equal(rows["failure_count"], [0, 1])
        np.testing.assert_array_equal(rows["on_time_completion_count"], [0, 1])
        # 1 completion - 1 failure - .1 delay + .05 throughput - .05 energy
        self.assertAlmostEqual(float(rows["reward_total"][1]), -0.1, places=6)
        self.assertEqual(rows["action_magnitude"].shape, (2, 5))

    def test_default_only_behavior_cannot_pass_six_template_support_gate(self):
        report = audit_template_support(
            observed_template_ids=["default"] * 100,
            minimum_samples_per_template=1,
        )

        self.assertFalse(report["candidate_reward_surrogate_ready"])
        self.assertEqual(report["covered_template_count"], 1)
        self.assertEqual(report["required_template_count"], 6)
        self.assertEqual(
            report["missing_templates"],
            [
                "deadline_first",
                "priority_first",
                "load_balance",
                "rate_aware",
                "energy_conservative",
            ],
        )

    def test_preflight_materializes_nonlocked_factual_rows_and_blocks_imagined_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher_root = root / "teacher"
            system_root = root / "system"
            output_root = root / "output"
            teacher_root.mkdir()
            system_root.mkdir()
            rows = [
                {
                    "seed": "0",
                    "split": "train",
                    "trajectory_id": "scenario__r00",
                    "v3_status": "materialized",
                    "v3_seed_dir": "seed_000",
                    "cpu_policy": "equal_share",
                },
                {
                    "seed": "6",
                    "split": "validation",
                    "trajectory_id": "scenario__r06",
                    "v3_status": "materialized",
                    "v3_seed_dir": "seed_006",
                    "cpu_policy": "deadline_aware",
                },
                {
                    "seed": "9",
                    "split": "locked_test",
                    "trajectory_id": "scenario__r09",
                    "v3_status": "locked_integrity_only",
                    "v3_seed_dir": "",
                    "cpu_policy": "feasible_exploration",
                },
            ]
            with (teacher_root / "trajectory_index.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            for seed, split in ((0, "train"), (6, "validation")):
                teacher_seed = teacher_root / f"seed_{seed:03d}"
                system_seed = system_root / f"seed_{seed:03d}"
                teacher_seed.mkdir()
                system_seed.mkdir()
                teacher = {
                    "time": np.asarray([0.1, 0.2], dtype=np.float32),
                    "task_lifecycle_index": np.asarray([[0], [4]], dtype=np.int16),
                    "task_present": np.asarray([[True], [True]]),
                    "task_action": np.zeros((2, 1, 8), dtype=np.float32),
                    "task_action_present": np.zeros((2, 1), dtype=bool),
                }
                system = {
                    "time": np.asarray([0.1, 0.2], dtype=np.float32),
                    "task_on_time_completion_event": np.zeros((2, 1), dtype=bool),
                    "completed_task_delay": np.zeros((2, 1), dtype=np.float32),
                    "completed_task_delay_valid": np.zeros((2, 1), dtype=bool),
                    "delivered_data_total": np.asarray([0.0, 1.0]),
                    "uav_energy_delta": np.asarray([[0.0], [1.0]], dtype=np.float32),
                    "uav_energy_valid": np.asarray([[True], [True]]),
                }
                np.savez_compressed(teacher_seed / "trajectory_tensors.npz", **teacher)
                np.savez_compressed(system_seed / "system_targets.npz", **system)
                (system_seed / "system_target_report.json").write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "split": split,
                            "trajectory_id": f"scenario__r{seed:02d}",
                        }
                    ),
                    encoding="utf-8",
                )
            scale_path = root / "reward_scale.json"
            scale_path.write_text(
                json.dumps(
                    RewardScale(
                        completed_delay_p95=1.0,
                        delivered_data_p95=1.0,
                        energy_delta_p95=1.0,
                        train_trajectory_count=1,
                        delay_step_count=1,
                        throughput_step_count=1,
                        energy_step_count=1,
                        source_manifest_sha256="b" * 64,
                    ).to_dict()
                ),
                encoding="utf-8",
            )

            summary = build_reward_surrogate_preflight(
                teacher_root=teacher_root,
                system_root=system_root,
                reward_scale_path=scale_path,
                output_root=output_root,
            )

            self.assertTrue(summary["factual_reward_dataset_ready"])
            self.assertFalse(summary["candidate_reward_surrogate_ready"])
            self.assertFalse(summary["locked_test_accessed"])
            self.assertEqual(summary["trajectory_count"], 2)
            self.assertEqual(summary["step_count"], 4)
            self.assertEqual(summary["split_step_counts"], {"train": 2, "validation": 2})
            self.assertEqual(summary["template_support"]["coverage_ratio"], 1 / 6)
            self.assertTrue((output_root / "factual_reward_rows.npz").is_file())
            self.assertTrue((output_root / "summary.json").is_file())
            self.assertTrue((output_root / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
