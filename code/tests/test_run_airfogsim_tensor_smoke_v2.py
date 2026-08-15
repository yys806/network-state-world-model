from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
RUNNER_PATH = CODE_ROOT / "scripts" / "run_airfogsim_tensor_smoke_v2.py"
BUILDER_TEST_PATH = CODE_ROOT / "tests" / "test_build_airfogsim_tensor_v2.py"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunAirFogSimTensorSmokeV2Tests(unittest.TestCase):
    def test_runs_one_epoch_and_writes_evidence(self):
        fixture = load_module(BUILDER_TEST_PATH, "tensor_builder_fixture")
        builder = fixture.load_subject()
        runner = load_module(RUNNER_PATH, "run_airfogsim_tensor_smoke_v2")
        with tempfile.TemporaryDirectory() as source_temporary, tempfile.TemporaryDirectory() as tensor_temporary, tempfile.TemporaryDirectory() as output_temporary:
            source = Path(source_temporary)
            tensor = Path(tensor_temporary)
            output = Path(output_temporary)
            fixture.write_source_fixture(source)
            builder.build_tensor_dataset(
                source_dir=source,
                output_dir=tensor,
                graph_loader=lambda seed: fixture.fake_graph(seed),
            )

            result = runner.run_smoke(
                dataset_dir=tensor,
                output_dir=output,
                epochs=1,
                batch_size=1,
                hidden_dim=8,
                eval_splits=("dev_validation",),
            )

            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
            self.assertTrue(result["smoke_ready"])
            self.assertTrue(summary["smoke_ready"])
            self.assertEqual(1, len(summary["training_history"]))
            self.assertIn("dev_validation", evaluation)
            self.assertIn("link_activity", evaluation["dev_validation"])
            self.assertIn("active_only_rate", evaluation["dev_validation"])
            self.assertTrue((output / "model.pt").is_file())
            self.assertTrue((output / "REPORT.md").is_file())
            self.assertTrue((output / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
