"""Statically audit whether PI-JWM has a candidate-action rollout planner mechanism."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.formal_candidate_rollout_planner_audit_v1 import (
    evaluate_candidate_rollout_planner_claims,
)


def inspect_candidate_planner_sources(
    *, policy_source: str, model_source: str, transition_source: str
) -> dict[str, Any]:
    """Conservatively distinguish direct scoring from candidate-wise model rollout."""

    direct_scorer_markers = (
        "candidate_descriptors",
        "self.candidate_encoder",
        "self.scorer",
    )
    model_action_markers = (
        "def forward",
        'future_action["task_action"]',
        "range(self.config.horizon_steps)",
    )
    has_direct_scorer = all(marker in policy_source for marker in direct_scorer_markers)
    model_action_conditioned = all(marker in model_source for marker in model_action_markers)
    policy_references_world_model = (
        "FormalDualGraphWorldModel" in policy_source or "world_model" in policy_source
    )
    transition_records_executed_actions = all(
        marker in transition_source for marker in ("JointTransition", "candidate_index")
    )

    # The audited R6 source supplies descriptors to a scorer; it exposes neither a
    # candidate sequence generator nor a world-model rollout/selection closure.
    observed = {
        "legal_candidate_actions_generated": False,
        "common_belief_reused_per_candidate": False,
        "world_model_invoked_per_candidate": False,
        "candidate_action_injected_into_rollout": False,
        "candidate_future_state_extracted": False,
        "candidate_task_cost_risk_extracted": False,
        "selection_uses_predicted_future_objective": False,
        "selected_first_action_feedback_replanning": False,
        "world_model_action_conditioned": model_action_conditioned,
    }
    return {
        "observed": observed,
        "source_observations": {
            "direct_candidate_descriptor_scorer_detected": has_direct_scorer,
            "policy_references_world_model": policy_references_world_model,
            "formal_model_action_conditioned": model_action_conditioned,
            "transition_records_executed_candidate_index": transition_records_executed_actions,
        },
    }


def run_audit(*, output_dir: str | Path) -> dict[str, Any]:
    """Write a code-only audit; never load data, checkpoints, GPU, or locked test."""

    output_path = Path(output_dir)
    if "locked_test" in str(output_path).lower():
        raise ValueError("locked_test path is forbidden")
    policy_path = SRC_ROOT / "pi_jwm" / "r6_joint_policy.py"
    model_path = SRC_ROOT / "pi_jwm" / "formal_dual_graph_world_model_v1.py"
    transition_path = SRC_ROOT / "pi_jwm" / "r6_rollout.py"
    inspection = inspect_candidate_planner_sources(
        policy_source=policy_path.read_text(encoding="utf-8"),
        model_source=model_path.read_text(encoding="utf-8"),
        transition_source=transition_path.read_text(encoding="utf-8"),
    )
    report = evaluate_candidate_rollout_planner_claims(inspection["observed"])
    report.update(
        {
            "audit_completed": True,
            "formal_performance_claim_ready": False,
            "observed": inspection["observed"],
            "source_observations": inspection["source_observations"],
            "evidence": {
                "policy_source": str(policy_path.resolve()),
                "world_model_source": str(model_path.resolve()),
                "transition_source": str(transition_path.resolve()),
                "scope": "code-only historical/current mechanism audit",
            },
            "execution_policy": {
                "gpu_started": False,
                "training_started": False,
                "locked_test_accessed": False,
                "locked_test_materialized": False,
            },
        }
    )
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "candidate_rollout_planner_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_audit(output_dir=args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
