import unittest

from pi_jwm.step6_3d_method_selection_v1 import (
    paired_outcome, cluster_bootstrap_advantage, select_method, tune_cem_config,
)


class MethodSelectionTests(unittest.TestCase):
    def test_h4_success_precedes_lexicographic_objective(self):
        self.assertEqual(paired_outcome((1, 0., 0., 0., 0.), None), 1)
        self.assertEqual(paired_outcome(None, None), 0)
        self.assertEqual(paired_outcome((1, 0., 0., 0., 0.), (0, 1., 1., 1., 1.)), -1)
        self.assertEqual(paired_outcome((1, 0., 0., 0., 0.), (1, 0., 0., 0., 0.)), 0)

    def test_anchor_cluster_bootstrap_and_simplicity_rule(self):
        rows = {f"anchor{i}": [1] * 5 for i in range(64)}
        estimate = cluster_bootstrap_advantage(rows, seed=6316, replications=1000)
        self.assertEqual(estimate["mean_paired_advantage"], 1.0)
        self.assertEqual(estimate["ci95_lower"], 1.0)
        self.assertEqual(select_method({("S-CEM", "HRS"): estimate,
                                        ("MH-CEM", "HRS"): estimate,
                                        ("MH-CEM", "S-CEM"): {"ci95_lower": 0.0}}), "S-CEM")
        self.assertEqual(select_method({("S-CEM", "HRS"): {"ci95_lower": 0.0},
                                        ("MH-CEM", "HRS"): {"ci95_lower": 0.0}}), "HRS")

    def test_train_config_uses_success_then_paired_objective_then_simplicity(self):
        configs = [(3, .1), (3, .2), (4, .1), (4, .2)]
        results = {config: {("a", 6301): (1, 0., 0., 0., 0.),
                            ("b", 6301): None} for config in configs}
        results[(4, .2)][("b", 6301)] = (9, 0., 0., 0., 0.)
        self.assertEqual(tune_cem_config(results)["selected_config"], (4, .2))
        results[(4, .2)][("b", 6301)] = None
        results[(3, .2)][("a", 6301)] = (0, 0., 0., 0., 0.)
        self.assertEqual(tune_cem_config(results)["selected_config"], (3, .2))


if __name__ == "__main__":
    unittest.main()
