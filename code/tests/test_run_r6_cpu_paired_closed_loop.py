from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPT_PATH = CODE_ROOT / "scripts" / "run_r6_cpu_paired_closed_loop.py"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_subject():
    spec = importlib.util.spec_from_file_location("run_r6_cpu_paired_closed_loop", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load R6 paired runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R6PairedRunnerTest(unittest.TestCase):
    def test_runtime_invocation_captures_non_ascii_simulator_logs(self) -> None:
        subject = load_subject()

        def runtime_runner(*args, **kwargs):
            print("🔧 AirFogSim")
            return {"ok": True}

        result, stdout, stderr = subject.invoke_runtime_quietly(runtime_runner, object(), max_time=1.0)
        self.assertEqual(result, {"ok": True})
        self.assertIn("AirFogSim", stdout)
        self.assertEqual(stderr, "")


if __name__ == "__main__":
    unittest.main()
