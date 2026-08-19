from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from pi_jwm.formal_airfogsim_protocol_audit_v1 import audit_formal_protocol
except ImportError:  # RED phase: the new audit module does not exist yet.
    audit_formal_protocol = None

from pi_jwm.formal_airfogsim_dataset_v1 import build_formal_trajectory_specs  # noqa: E402


FREEZE_PATH = (
    CODE_ROOT
    / "artifacts"
    / "audit"
    / "pi_jwm_p2c_scenario_calibration_20260819"
    / "freeze_proposal_v1.json"
)


def _freeze_proposal() -> dict[str, object]:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


class FormalAirFogSimProtocolAuditTests(unittest.TestCase):
    def test_user_confirmed_protocol_matches_code_and_calibration_evidence(self):
        self.assertTrue(callable(audit_formal_protocol), "protocol audit API is missing")
        if not callable(audit_formal_protocol):
            return

        report = audit_formal_protocol(
            build_formal_trajectory_specs(),
            _freeze_proposal(),
            project_root=CODE_ROOT.parent,
        )

        self.assertTrue(report["audit_ready"], report["failed_checks"])
        self.assertTrue(report["checks"]["scenario_matrix_frozen"])
        self.assertTrue(report["checks"]["formal_scale_frozen"])
        self.assertTrue(report["checks"]["formal_split_frozen"])
        self.assertTrue(report["checks"]["calibration_evidence_verified"])
        self.assertFalse(report["formal_data_approved"])

    def test_tampered_formal_scale_is_rejected(self):
        self.assertTrue(callable(audit_formal_protocol), "protocol audit API is missing")
        if not callable(audit_formal_protocol):
            return

        proposal = copy.deepcopy(_freeze_proposal())
        proposal["formal_scale_proposal"]["total_trajectories"] = 61
        report = audit_formal_protocol(
            build_formal_trajectory_specs(),
            proposal,
            project_root=CODE_ROOT.parent,
        )

        self.assertFalse(report["audit_ready"])
        self.assertIn("formal_scale_frozen", report["failed_checks"])


if __name__ == "__main__":
    unittest.main()
