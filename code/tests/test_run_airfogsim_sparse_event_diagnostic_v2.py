from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
RUNNER_PATH = CODE_ROOT / "scripts" / "run_airfogsim_sparse_event_diagnostic_v2.py"
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


class RunSparseEventDiagnosticV2Tests(unittest.TestCase):
    def test_runs_four_arms_with_fair_learned_initialization(self):
        fixture = load_module(BUILDER_TEST_PATH, "sparse_diagnostic_builder_fixture")
        builder = fixture.load_subject()
        runner = load_module(RUNNER_PATH, "run_airfogsim_sparse_event_diagnostic_v2")
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as tensor_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source = Path(source_tmp)
            tensor = Path(tensor_tmp)
            output = Path(output_tmp)
            fixture.write_source_fixture(source)
            builder.build_tensor_dataset(
                source_dir=source,
                output_dir=tensor,
                graph_loader=lambda seed: fixture.fake_graph(seed),
            )

            result = runner.run_diagnostic(
                dataset_dir=tensor,
                output_dir=output,
                epochs=1,
                batch_size=1,
                hidden_dim=8,
                eval_splits=("dev_validation",),
            )

            self.assertEqual(
                {"zero_activity", "last_persistence", "learned_unweighted", "learned_balanced"},
                set(result["arms"]),
            )
            evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
            history = json.loads((output / "training_history.json").read_text(encoding="utf-8"))
            stats = json.loads((output / "class_stats.json").read_text(encoding="utf-8"))
            self.assertIn("link_activity", evaluation["learned_balanced"]["dev_validation"])
            self.assertEqual("dev_train", stats["source_split"])
            self.assertEqual(
                history["learned_unweighted"]["initialization_sha256"],
                history["learned_balanced"]["initialization_sha256"],
            )
            self.assertEqual(
                history["learned_unweighted"]["sample_order"],
                history["learned_balanced"]["sample_order"],
            )
            self.assertTrue((output / "learned_balanced_model.pt").is_file())
            self.assertTrue((output / "learned_unweighted_model.pt").is_file())
            self.assertTrue((output / "REPORT.md").is_file())
            self.assertTrue((output / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
