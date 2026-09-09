from __future__ import annotations

import copy
import unittest


def _report(link=0.7, node=1.0, throughput=1.0, rb=1.0, delay=1.0):
    metrics = {
        "event.link_activity.f1": {"status": "computed", "value": link},
        "state.node.x.mae": {"status": "computed", "value": node},
        "system.communication_throughput.mae": {"status": "computed", "value": throughput},
        "resource.rb_occupancy.mae": {"status": "computed", "value": rb},
        "state.task.delay.mae": {"status": "computed", "value": delay},
    }
    return {
        "horizons": {
            "overall": {"metrics": copy.deepcopy(metrics)},
            "k=5": {"metrics": copy.deepcopy(metrics)},
            "k=10": {"metrics": copy.deepcopy(metrics)},
            "k=20": {"metrics": copy.deepcopy(metrics)},
        }
    }


class FormalP4GateV1Tests(unittest.TestCase):
    def test_gate_rank_is_lexicographic_and_uses_calibration_only_for_calibration_delta(self):
        from pi_jwm.formal_p4_gate_v1 import evaluate_p4_checkpoint_gates

        persistence_validation = _report(link=0.7)
        persistence_calibration = _report(link=0.6)
        candidate_validation = _report(link=0.66, node=1.2)
        for horizon in ("k=5", "k=10", "k=20"):
            candidate_validation["horizons"][horizon]["metrics"][
                "state.node.x.mae"
            ]["value"] = 1.0
        candidate_calibration = _report(link=0.66)
        result = evaluate_p4_checkpoint_gates(
            validation=candidate_validation,
            calibration=candidate_calibration,
            persistence_validation=persistence_validation,
            persistence_calibration=persistence_calibration,
            validation_state_nll=0.3,
        )

        self.assertTrue(result["all_numeric_gates_passed"])
        self.assertEqual([0, 0.0, 0.3], result["rank_key"])

    def test_failed_gate_count_precedes_state_nll(self):
        from pi_jwm.formal_p4_gate_v1 import evaluate_p4_checkpoint_gates

        baseline = _report(link=0.7)
        failed = evaluate_p4_checkpoint_gates(
            validation=_report(link=0.5, node=2.0),
            calibration=_report(link=0.5),
            persistence_validation=baseline,
            persistence_calibration=baseline,
            validation_state_nll=0.01,
        )
        passed = evaluate_p4_checkpoint_gates(
            validation=_report(link=0.7),
            calibration=_report(link=0.76),
            persistence_validation=baseline,
            persistence_calibration=baseline,
            validation_state_nll=100.0,
        )

        self.assertGreater(failed["failed_hard_gate_count"], 0)
        self.assertLess(tuple(passed["rank_key"]), tuple(failed["rank_key"]))


if __name__ == "__main__":
    unittest.main()
