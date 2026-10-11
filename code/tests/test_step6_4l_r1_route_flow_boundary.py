import json
import unittest
from pathlib import Path
from copy import deepcopy
from audit_step6_4l_r1_acceptance_v1 import validate_observed_identity


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v1"
WM = ROOT / "code/artifacts/audit/pi_jwm_step6_4l_r1_20261011_v2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class Step64LR1RouteFlowBoundaryTests(unittest.TestCase):
    def test_identity_guard_rejects_unqualified_and_broken_recursive_evidence(self):
        row = {"candidate": {"generation_metadata": {"planner_action_domain": "V1"}},
               "official_admission_passed": True,
               "parameter_before": "p", "parameter_after": "p",
               "initial": {"state": "s0", "graph": "g0", "latent": "l0"},
               "inputs": [{"state": "s0", "graph": "g0", "latent": "l0"},
                          {"state": "s1", "graph": "g1", "latent": "l1"}],
               "outputs": [{"state": "s1", "graph": "g1", "latent": "l1"},
                           {"state": "s2", "graph": "g2", "latent": "l2"}],
               "finite": [True, True]}
        validate_observed_identity(row)
        for mutate in (lambda x: x["candidate"]["generation_metadata"].clear(),
                       lambda x: x["inputs"][1].update(state="foreign"),
                       lambda x: x["finite"].__setitem__(1, False),
                       lambda x: x.update(parameter_after="changed")):
            bad = deepcopy(row)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_observed_identity(bad)

    def test_real_failure_is_tti_before_wireless(self):
        post = read(RUNTIME / "02_post_step.json")
        events = read(RUNTIME / "03_runtime_events.json")
        self.assertEqual(post["task"]["failure_reason_code"], 1)
        self.assertEqual(post["task"]["failure_reason"], "Task fails due to transmission timeout.")
        self.assertLess(float(post["simulation_time_s"]),
                        float(post["task"]["arrival_time"] + post["task"]["task_deadline"]))
        names = [row["event"] for row in events]
        self.assertLess(names.index("failOffloadingTask.after"), names.index("_updateWirelessCommunication.before"))
        self.assertEqual([row for row in events if row["event"] == "_allocate_communication_RBs"][-1]["requested"], {})

    def test_official_qualification_and_boundary(self):
        a = read(WM / "A_08_wm_scorer.json")
        self.assertEqual(a["stage"], "PASS")
        self.assertEqual(a["wm_scorer"]["horizons"], 4)
        self.assertEqual(a["wm_scorer"]["H_eff"], 0)
        self.assertIn("UNSUPPORTED_FUTURE_RETURN_BIRTH:Task_5:H1",
                      json.dumps(a["wm_scorer"]["diagnostics"]))

    def test_shadow_result_is_separate_from_formal_score(self):
        b = read(WM / "B_08_wm_scorer.json")
        self.assertEqual(b["stage"], "PASS")
        self.assertEqual(b["wm_scorer"]["H_eff"], 4)
        self.assertEqual(b["wm_scorer"]["scorer_status"], "SCOREABLE")
        self.assertFalse(b["wm_scorer"]["formal_score_promoted"])
        self.assertTrue(b["shadow_flow"]["formal_score_not_promoted"])

    def test_comm_matrix_is_currently_supported_but_unscheduled(self):
        matrix = read(WM / "04_route_comm_support_matrix.json")
        self.assertEqual(matrix["classification"]["primary"], "TEMPORARILY_NO_SERVICE")
        self.assertFalse(matrix["classification"]["physically_unsupported_hop"])
        self.assertFalse(matrix["classification"]["raw_or_graph_support_missing"])
        self.assertTrue(matrix["model"]["matching_relation_indices"])
        self.assertFalse(matrix["airfogsim"]["new_transfer_event_for_task5"])

    def test_r1_namespace_and_scope(self):
        receipt = read(WM / "05_independent_acceptance.json")
        self.assertEqual(receipt["status"], "PASS")
        self.assertFalse(receipt["GPU"])
        self.assertFalse(receipt["locked_test"])
        self.assertTrue(receipt["checks"]["same_formal_candidate_anchor_not_proven"])


if __name__ == "__main__":
    unittest.main()
