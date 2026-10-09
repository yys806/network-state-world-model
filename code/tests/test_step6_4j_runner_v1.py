import json, subprocess, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code/scripts/run_step6_4j_pilot_v1.py"
PROTO = ROOT / "code/artifacts/protocols/pi_jwm_step6_4j_s_cem_gpu_pilot_v1_20261009/00_protocol.json"

class TestPilotRunner(unittest.TestCase):
    def test_requires_explicit_mode(self):
        p = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
    def test_protocol_scope_is_frozen(self):
        c = json.loads(PROTO.read_text(encoding="utf-8"))
        self.assertEqual((c["method"], c["K"], c["rho"], c["B_WM"], c["H"]), ("S-CEM", 4, .2, 512, 4))
        self.assertFalse(c["locked_test"])
        self.assertEqual(c["max_searches"], 16)
    def test_execute_rejects_without_cuda_before_result_namespace(self):
        p = subprocess.run([sys.executable, str(SCRIPT), "--execute", "--protocol", str(PROTO)], cwd=ROOT, capture_output=True, text=True)
        # Local host is CPU-only; the gate must reject rather than create a formal result.
        self.assertNotEqual(p.returncode, 0)

if __name__ == "__main__":
    unittest.main()
