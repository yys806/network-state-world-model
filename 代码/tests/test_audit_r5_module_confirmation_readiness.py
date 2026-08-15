from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
for root in (CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


class AuditR5ModuleConfirmationReadinessTest(unittest.TestCase):
    def test_audit_rejects_historical_core_source_drift(self) -> None:
        from audit_r5_module_confirmation_readiness import audit_readiness

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit"
            with self.assertRaisesRegex(ValueError, "core B source hash mismatch"):
                audit_readiness(
                    dataset_root=CODE_ROOT / "artifacts" / "datasets" / "airfogsim_teacher_aligned_v3",
                    evaluation_root=CODE_ROOT / "artifacts" / "evaluation" / "pi_jwm_eval_protocol_v3",
                    r4_screening_root=CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r4_gpu_screening_v1",
                    existing_r5_root=CODE_ROOT / "artifacts" / "formal_training" / "pi_jwm_r5_gpu_training_v1",
                    frozen_bundle_root=CODE_ROOT / "artifacts" / "preflight" / "pi_jwm_r5_module_confirmation_v2",
                    output_dir=output,
                    hidden_dim=4,
                    combination_ids=("F",),
                )


if __name__ == "__main__":
    unittest.main()
