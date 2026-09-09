import importlib.util
import sys
import unittest
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = CODE_ROOT / "scripts" / "run_formal_p4_world_model_gate_v1.py"
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))


def load_subject():
    spec = importlib.util.spec_from_file_location(
        "run_formal_p4_world_model_gate_v1", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load P4 gate script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class P4WorldModelGateTests(unittest.TestCase):
    def test_probe_horizon_uses_checkpoint_horizon_with_longer_tensor_contract(self):
        subject = load_subject()
        self.assertEqual(3, subject._resolve_probe_horizon({"horizon_steps": 3}, 20))

    def test_probe_horizon_rejects_checkpoint_longer_than_tensor_contract(self):
        subject = load_subject()
        with self.assertRaisesRegex(ValueError, "checkpoint horizon exceeds tensor contract"):
            subject._resolve_probe_horizon({"horizon_steps": 20}, 3)

    def test_probe_batch_slices_future_namespaces_to_checkpoint_horizon(self):
        subject = load_subject()
        batch = {
            "history": {"node_state": torch.zeros(1, 8, 2, 3)},
            "future_action": {"task_action": torch.zeros(1, 20, 4, 8)},
            "target": {"task_state": torch.zeros(1, 20, 4, 8)},
            "static": {"slot_seconds": torch.ones(1)},
        }
        sliced = subject._slice_probe_batch(batch, 3)
        self.assertEqual(20, batch["future_action"]["task_action"].shape[1])
        self.assertEqual(3, sliced["future_action"]["task_action"].shape[1])
        self.assertEqual(3, sliced["target"]["task_state"].shape[1])
        self.assertEqual(8, sliced["history"]["node_state"].shape[1])


if __name__ == "__main__":
    unittest.main()
