from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_r6_online_gpu_readiness.py"


def _load_subject():
    spec = importlib.util.spec_from_file_location("build_r6_online_gpu_readiness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class R6OnlineGPUReadinessTest(unittest.TestCase):
    def test_assessment_requires_online_state_real_updates_and_no_locked_access(self) -> None:
        subject = _load_subject()
        summary = {
            "state_source": "online_airfogsim_strict_dual_graph",
            "update_count": 1,
            "reports": [{"parameter_changed": True, "gradient_norm": 0.1}],
            "hard_violation_count": 0,
            "locked_test_accessed": False,
            "world_model_updated": False,
            "checkpoint_reload_verified": True,
            "nondefault_selection_count": 1,
            "distinct_explicit_state_count": 2,
        }
        checks = subject.assess_six_mode_summaries([summary] * 6)
        self.assertTrue(all(checks.values()))
        bad = dict(summary, locked_test_accessed=True)
        self.assertFalse(subject.assess_six_mode_summaries([summary] * 5 + [bad])["no_locked_test_access"])


if __name__ == "__main__":
    unittest.main()
