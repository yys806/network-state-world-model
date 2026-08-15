from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_reward_protocol import (  # noqa: E402
    RewardScale,
    ServiceFirstRewardProtocol,
    TransitionFacts,
    freeze_train_reward_scale,
)


def _write_trajectory(
    root: Path,
    *,
    name: str,
    split: str,
    delays: list[float],
    delivered_total: list[float],
    energy: list[float],
) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    steps = len(delivered_total)
    delay_array = np.zeros((steps, 2), dtype=np.float32)
    delay_valid = np.zeros((steps, 2), dtype=bool)
    for step, value in enumerate(delays):
        if value > 0:
            delay_array[step, 0] = value
            delay_valid[step, 0] = True
    energy_array = np.asarray(energy, dtype=np.float32).reshape(steps, 1)
    np.savez_compressed(
        directory / "system_targets.npz",
        completed_task_delay=delay_array,
        completed_task_delay_valid=delay_valid,
        delivered_data_total=np.asarray(delivered_total, dtype=np.float64),
        uav_energy_delta=energy_array,
        uav_energy_valid=np.ones_like(energy_array, dtype=bool),
    )
    (directory / "system_target_report.json").write_text(
        json.dumps(
            {
                "schema_version": "PI-JWM-formal-system-targets-v1",
                "trajectory_id": name,
                "seed": int(name.split("_")[-1]),
                "split": split,
            }
        ),
        encoding="utf-8",
    )


class R6RewardProtocolTest(unittest.TestCase):
    def _dataset(self, root: Path) -> None:
        _write_trajectory(
            root,
            name="seed_000",
            split="train",
            delays=[2.0, 4.0],
            delivered_total=[1.0, 2.0],
            energy=[2.0, 4.0],
        )
        _write_trajectory(
            root,
            name="seed_100",
            split="validation",
            delays=[2000.0, 4000.0],
            delivered_total=[1000.0, 2000.0],
            energy=[2000.0, 4000.0],
        )
        _write_trajectory(
            root,
            name="seed_200",
            split="calibration",
            delays=[3000.0, 6000.0],
            delivered_total=[2000.0, 3000.0],
            energy=[3000.0, 6000.0],
        )
        manifest = root / "manifest.json"
        manifest.write_text('{"schema_version":"fixture"}\n', encoding="utf-8")
        (root / "dataset_summary.json").write_text(
            json.dumps(
                {
                    "system_targets_ready": True,
                    "locked_test_accessed": False,
                    "split_counts": {"train": 1, "validation": 1, "calibration": 1},
                }
            ),
            encoding="utf-8",
        )

    def test_scale_uses_train_only_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._dataset(root)
            scale = freeze_train_reward_scale(root)
            self.assertAlmostEqual(3.9, scale.completed_delay_p95, places=6)
            self.assertAlmostEqual(1.95, scale.delivered_data_p95, places=6)
            self.assertAlmostEqual(3.9, scale.energy_delta_p95, places=6)
            self.assertEqual(1, scale.train_trajectory_count)
            self.assertEqual(2, scale.delay_step_count)
            self.assertEqual(2, scale.throughput_step_count)
            self.assertEqual(2, scale.energy_step_count)
            self.assertEqual(
                hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
                scale.source_manifest_sha256,
            )

    def test_scale_rejects_locked_test_and_nonpositive_train_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._dataset(root)
            report = root / "seed_100" / "system_target_report.json"
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["split"] = "locked_test"
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "locked_test"):
                freeze_train_reward_scale(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_trajectory(
                root,
                name="seed_000",
                split="train",
                delays=[0.0, 0.0],
                delivered_total=[0.0, 0.0],
                energy=[0.0, 0.0],
            )
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            (root / "dataset_summary.json").write_text(
                json.dumps({"system_targets_ready": True, "locked_test_accessed": False}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "positive train"):
                freeze_train_reward_scale(root)

    def test_service_first_reward_preserves_components_and_primary_dominance(self) -> None:
        protocol = ServiceFirstRewardProtocol(
            RewardScale(
                completed_delay_p95=10.0,
                delivered_data_p95=20.0,
                energy_delta_p95=30.0,
                train_trajectory_count=36,
                delay_step_count=10,
                throughput_step_count=10,
                energy_step_count=10,
                source_manifest_sha256="a" * 64,
            )
        )
        best_secondary = protocol.score(
            TransitionFacts(
                on_time_completion_count=0,
                failure_count=1,
                completed_delay_sum=0.0,
                delivered_data_delta=100.0,
                energy_delta=0.0,
                hard_violation_count=0,
            )
        )
        worst_secondary = protocol.score(
            TransitionFacts(
                on_time_completion_count=1,
                failure_count=0,
                completed_delay_sum=100.0,
                delivered_data_delta=0.0,
                energy_delta=100.0,
                hard_violation_count=0,
            )
        )
        self.assertTrue(best_secondary.valid)
        self.assertTrue(worst_secondary.valid)
        self.assertLess(best_secondary.total_reward, worst_secondary.total_reward)
        self.assertEqual(
            {
                "on_time_completion",
                "failure",
                "delay",
                "throughput",
                "energy",
            },
            set(worst_secondary.weighted_components),
        )
        self.assertAlmostEqual(
            sum(worst_secondary.weighted_components.values()),
            float(worst_secondary.total_reward),
        )

    def test_hard_violation_invalidates_transition_instead_of_trading_reward(self) -> None:
        protocol = ServiceFirstRewardProtocol(
            RewardScale(10.0, 20.0, 30.0, 36, 10, 10, 10, "b" * 64)
        )
        breakdown = protocol.score(
            TransitionFacts(100, 0, 0.0, 1000.0, 0.0, hard_violation_count=1)
        )
        self.assertFalse(breakdown.valid)
        self.assertIsNone(breakdown.total_reward)
        self.assertEqual("hard_constraint_violation", breakdown.invalid_reason)


if __name__ == "__main__":
    unittest.main()
