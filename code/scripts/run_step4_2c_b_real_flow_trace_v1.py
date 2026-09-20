"""Collect a non-locked real AirFogSim trace with Input/Return Flow sources.

This wrapper reuses the frozen Step 2.3 runner and only adds observer-backed
``return_size`` to the new Step 4.2C-B source artifact.  It does not alter the
simulator state machine or overwrite the frozen Step 2.3 artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "code" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_step2_3_real_airfogsim_raw_contract_finalization_v1 as base  # noqa: E402

OUTPUT = ROOT / "code/artifacts/protocols/pi_jwm_step4_2c_b_real_flow_trace_source_v1_20260920"


def main() -> None:
    original = base._task_rows

    def task_rows_with_return_size(env):
        observable, future, snapshot = original(env)
        by_id = {row.task_id: row for row in snapshot.tasks}
        for row in [*observable, *future]:
            source = by_id[str(row["task_id"])]
            row["return_size"] = float(source.return_size)
            row["return_size_source"] = "Task.getReturnedSize via TaskSnapshot.return_size"
        return observable, future, snapshot

    base._task_rows = task_rows_with_return_size
    base.OUTPUT = OUTPUT
    base.TRAJECTORY_ID = "step4.2c-b-real-flow-ledger-seed0"
    base.DECISION_STEPS = 12
    base.main()


if __name__ == "__main__":
    main()
