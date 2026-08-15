from __future__ import annotations

import sys
import inspect
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import launch_r6_gpu_training_matrix as launcher  # noqa: E402
import run_r6_joint_policy_gpu_readiness as readiness  # noqa: E402


class R6GPUTrainingLauncherTest(unittest.TestCase):
    def test_formal_commands_assign_one_auditable_sumo_port_per_run(self) -> None:
        parameters = inspect.signature(launcher.build_formal_commands).parameters
        self.assertIn("sumo_port_base", parameters)
        commands = launcher.build_formal_commands(
            python_executable=sys.executable,
            output_dir=CODE_ROOT / "artifacts" / "tmp",
            device="cuda",
            target_environment_steps=10_000,
            sumo_port_base=18_813,
        )

        ports = [command.sumo_port for command in commands]
        self.assertEqual(list(range(18_813, 18_813 + 18)), ports)
        self.assertEqual(18, len(set(ports)))
        for command in commands:
            port_index = command.argv.index("--sumo-port") + 1
            self.assertEqual(str(command.sumo_port), command.argv[port_index])

    def test_formal_environment_override_changes_only_the_sumo_port(self) -> None:
        override = getattr(readiness, "_apply_sumo_port_override", None)
        self.assertIsNotNone(override)
        source = {
            "sumo": {"sumo_port": 8813, "sumo_config": "scenario.sumocfg"},
            "traffic": {"traffic_mode": "SUMO"},
        }

        updated = override(source, 18_813)

        self.assertEqual(8813, source["sumo"]["sumo_port"])
        self.assertEqual(18_813, updated["sumo"]["sumo_port"])
        self.assertEqual(source["sumo"]["sumo_config"], updated["sumo"]["sumo_config"])
        self.assertEqual(source["traffic"], updated["traffic"])

    def test_worker_environment_records_the_cpu_thread_limit(self) -> None:
        builder = getattr(launcher, "build_worker_environment", None)
        self.assertIsNotNone(builder)

        environment = builder(cpu_threads=4, base_environment={"PATH": "test"})

        self.assertEqual("test", environment["PATH"])
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.assertEqual("4", environment[name])


if __name__ == "__main__":
    unittest.main()
