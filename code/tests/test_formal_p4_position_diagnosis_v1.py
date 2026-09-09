import unittest

import numpy as np

from run_formal_p4_position_diagnosis_v1 import summarize_position_errors


class FormalP4PositionDiagnosisTests(unittest.TestCase):
    def test_summary_uses_present_nodes_and_separates_coordinates(self):
        target = np.array(
            [
                [[1.0, 0.0, 0.0], [99.0, 99.0, 99.0]],
                [[2.0, 0.0, 0.0], [99.0, 99.0, 99.0]],
            ],
            dtype=np.float32,
        )
        predicted = np.array(
            [
                [[1.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[2.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        persistence = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        present = np.array([[True, False], [True, False]])
        node_types = np.array(["vehicle", "rsu"], dtype=object)

        result = summarize_position_errors(target, predicted, persistence, present, node_types)

        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["mae"]["x"], 0.5)
        self.assertAlmostEqual(result["persistence_mae"]["x"], 1.5)
        self.assertAlmostEqual(result["delta"]["x"], -1.0)
        self.assertEqual(result["by_node_type"]["vehicle"]["count"], 2)
        self.assertNotIn("rsu", result["by_node_type"])


if __name__ == "__main__":
    unittest.main()
