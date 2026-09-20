"""Collect a tiny real cross-slot two-hop trace for STEP 4.2C-C.

This is a configuration-only reuse of the frozen STEP 2.4 collector.  Lowering
the wired-link capacity makes the second carrying hop remain observable across
Decision snapshots without changing AirFogSim semantics.
"""
from __future__ import annotations

import run_step2_4_real_airfogsim_communication_outcome_semantics_v1 as base


base.OUTPUT = (
    base.CODE
    / "artifacts"
    / "protocols"
    / "pi_jwm_step4_2c_c_real_multihop_cross_slot_v1_20260920"
)
base.TRAJECTORY_ID = "step4.2c-c-real-multihop-cross-slot-seed0"
base.DECISION_STEPS = 8
base.WIRED_EDGES = [
    {
        "u": base.WIRED_RSU,
        "v": base.WIRED_CLOUD,
        "capacity_mbps": 0.00001,
        "prop_ms": 1.0,
        "bidirectional": True,
    }
]


if __name__ == "__main__":
    base.main()
