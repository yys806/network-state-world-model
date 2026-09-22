import json
import tempfile
import unittest
from pathlib import Path

from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface, validate_readiness
from pi_jwm.step5_2_training_loop_v1 import CurriculumConfig


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

    def test_readiness_negative_check_cannot_pass(self):
        checks = {"package_load": True, "trainer_construct": False, "cpu_train_step": True, "prior_only_validation": True, "checkpoint_reload": True, "model_device": True, "data_device": True, "checkpoint_map_location": True, "cpu_generic_dry_run": True, "dataset_contract": True, "action_coverage": True, "split_isolation": True}
        result = validate_readiness(checks, formal_dataset=False, research_decisions_frozen=False)
        self.assertEqual(result["training_stack_readiness"], "FAIL")
        self.assertEqual(result["formal_training_readiness"], "BLOCKED")

    def test_horizon_four_configuration_fixture(self):
        schedule = CurriculumConfig(horizons=(1, 2, 4), start_steps=(0, 1, 2))
        self.assertEqual(schedule.horizon_for_step(2, 4), 4)


if __name__ == "__main__":
    unittest.main()
