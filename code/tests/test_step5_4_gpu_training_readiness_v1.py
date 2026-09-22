import json
import tempfile
import unittest
from pathlib import Path

from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface


class Step54InterfaceTests(unittest.TestCase):
    def test_manifest_interface_has_no_fixed_development_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = [{"metadata": {"split": "train", "trajectory_id": "a"}, "future_action": [{"route": {"entries": [{"x": 1}]}, "comm": {"entries": []}, "comp": {"entries": []}, "mobility": {"entries": [{"x": 1}]}}]}, {"metadata": {"split": "validation", "trajectory_id": "b"}, "future_action": [{"route": {"entries": []}, "comm": {"entries": [{"x": 2}]}, "comp": {"entries": [{"x": 3}]}, "mobility": {"entries": []}}]}]
            sample_path = root / "samples.json"; sample_path.write_text(json.dumps(samples), encoding="utf-8")
            manifest = root / "manifest.json"; manifest.write_text(json.dumps({"sample_path": str(sample_path), "contract": {}, "provenance": {"source": "fixture"}}), encoding="utf-8")
            interface = FormalTrainingInterface.from_manifest(manifest)
            self.assertEqual(interface.train_indices, (0,))
            self.assertEqual(interface.validation_indices, (1,))
            coverage = interface.action_coverage()
            self.assertEqual(coverage["route"]["non_empty_count"], 1)
            self.assertEqual(coverage["comp"]["non_empty_count"], 1)
            self.assertEqual(coverage["mobility"]["non_empty_count"], 1)


if __name__ == "__main__":
    unittest.main()
