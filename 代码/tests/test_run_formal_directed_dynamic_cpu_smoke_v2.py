from __future__ import annotations

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


class RunFormalDirectedDynamicCpuSmokeV2Tests(unittest.TestCase):
    def test_cpu_protocol_rejects_locked_test_and_non_cpu_device(self):
        from run_formal_directed_dynamic_cpu_smoke_v2 import validate_cpu_protocol_v2

        validate_cpu_protocol_v2(("train", "validation", "calibration"), "cpu")
        with self.assertRaisesRegex(ValueError, "locked_test"):
            validate_cpu_protocol_v2(("train", "locked_test"), "cpu")
        with self.assertRaisesRegex(ValueError, "CPU"):
            validate_cpu_protocol_v2(("train", "validation"), "cuda")

    def test_cpu_smoke_uses_nonlocked_fixture_and_writes_v2_checkpoint(self):
        from run_formal_directed_dynamic_cpu_smoke_v2 import run_cpu_smoke_v2

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_root = root / "tensor"
            output_dir = root / "output"
            tensor_root.mkdir()
            _write_formal_fixture(tensor_root)

            result = run_cpu_smoke_v2(
                tensor_root=tensor_root,
                output_dir=output_dir,
                seed=53,
                train_limit=1,
                evaluation_limit=1,
                hidden_dim=4,
                epochs=1,
                batch_size=1,
            )

            self.assertTrue(result["training_run_complete"])
            self.assertFalse(result["locked_test_accessed"])
            for method in (
                "coupled_directed_dynamic_v2",
                "coupled_directed_dynamic_residual_v2",
            ):
                self.assertTrue(
                    (output_dir / "checkpoints" / f"{method}__best.pt").is_file()
                )


if __name__ == "__main__":
    unittest.main()
