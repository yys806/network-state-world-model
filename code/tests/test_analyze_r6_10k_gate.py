from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from analyze_r6_10k_gate import build_input_binding  # noqa: E402


class AnalyzeR610kGateRunnerTest(unittest.TestCase):
    def test_input_binding_hashes_all_frozen_matrix_control_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = {}
            for name, content in (
                ("run_records.json", b"records"),
                ("matrix_summary.json", b"matrix"),
                ("launch_manifest.json", b"manifest"),
            ):
                (root / name).write_bytes(content)
                expected[f"{name}_sha256"] = hashlib.sha256(content).hexdigest()

            self.assertEqual(build_input_binding(root), expected)

    def test_input_binding_rejects_missing_control_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "run_records.json"):
                build_input_binding(Path(tmp))


if __name__ == "__main__":
    unittest.main()
