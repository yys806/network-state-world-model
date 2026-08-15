from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_formal_airfogsim_window_v1 import _write_formal_fixture


class RunFormalDualGraphCpuSmokeV1Tests(unittest.TestCase):
    def test_interface_smoke_writes_auditable_five_method_artifacts(self):
        from run_formal_dual_graph_cpu_smoke_v1 import run_cpu_smoke

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            output_dir = root / "output"
            tensor_root.mkdir()
            _write_formal_fixture(tensor_root)

            result = run_cpu_smoke(
                tensor_root=tensor_root,
                output_dir=output_dir,
                mode="interface_smoke",
                seed=31,
                device="cpu",
                train_limit=1,
                evaluation_limit=1,
                hidden_dim=4,
                epochs=1,
            )

            self.assertTrue(result["cpu_smoke_ready"])
            self.assertFalse(result["locked_test_accessed"])
            self.assertEqual(5, len(result["completed_cpu_methods"]))
            for relative in (
                "config.json",
                "method_registry.json",
                "sample_ids.json",
                "training_history.json",
                "comparison.csv",
                "runtime.json",
                "run_summary.json",
                "manifest.json",
            ):
                self.assertTrue((output_dir / relative).is_file(), relative)
            for method in result["completed_cpu_methods"]:
                self.assertTrue((output_dir / "metrics" / f"{method}__validation.json").is_file())
                self.assertTrue((output_dir / "metrics" / f"{method}__calibration.json").is_file())
            for method in ("pooled_gru", "independent_dual_gnn", "coupled_dual_gnn"):
                self.assertTrue((output_dir / "checkpoints" / f"{method}.pt").is_file())

            samples = json.loads((output_dir / "sample_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(["seed000::window000000"], samples["train"])
            self.assertEqual(["seed001::window000000"], samples["validation"])
            self.assertEqual([], samples["calibration"])
            self.assertEqual(set(samples["train"]) & set(samples["validation"]), set())

            with (output_dir / "comparison.csv").open("r", encoding="utf-8", newline="") as handle:
                comparison = list(csv.DictReader(handle))
            self.assertEqual(5, len(comparison))
            self.assertEqual(
                {
                    "zero_activity",
                    "last_persistence",
                    "pooled_gru",
                    "independent_dual_gnn",
                    "coupled_dual_gnn",
                },
                {row["method"] for row in comparison},
            )

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            for relative, metadata in manifest["files"].items():
                payload = (output_dir / relative).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), metadata["sha256"])
                self.assertEqual(len(payload), metadata["bytes"])

    def test_configuration_never_accepts_locked_test_as_a_cpu_split(self):
        from run_formal_dual_graph_cpu_smoke_v1 import validate_cpu_protocol

        validate_cpu_protocol(("train", "validation", "calibration"))
        with self.assertRaisesRegex(ValueError, "locked_test"):
            validate_cpu_protocol(("train", "locked_test"))


if __name__ == "__main__":
    unittest.main()
