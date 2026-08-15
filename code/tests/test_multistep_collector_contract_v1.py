from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.information_edge_contract_v4 import MissingReason  # noqa: E402
from pi_jwm.multistep_collector_contract_v1 import (  # noqa: E402
    EdgeIdentity,
    LinkHistoryLedger,
    LinkOutcome,
    TrajectoryVocabulary,
)


class TrajectoryVocabularyTests(unittest.TestCase):
    def test_indices_are_append_only_across_disappearance_and_reappearance(self):
        vocabulary = TrajectoryVocabulary()
        edge = EdgeIdentity(
            "ie::uav0::rsu0", "uav0", "rsu0", "wireless:U2I"
        )
        first = vocabulary.observe(
            node_ids=("uav0", "rsu0"), edges=(edge,), flow_ids=("task0",)
        )
        second = vocabulary.observe(node_ids=("rsu0",), edges=(), flow_ids=())
        third = vocabulary.observe(
            node_ids=("uav0", "rsu0", "rsu1"),
            edges=(edge,),
            flow_ids=("task1",),
        )

        self.assertEqual(0, first.node_indices["rsu0"])
        self.assertEqual(1, first.node_indices["uav0"])
        self.assertFalse(second.node_presence[first.node_indices["uav0"]])
        self.assertEqual(first.node_indices["uav0"], third.node_indices["uav0"])
        self.assertEqual(2, third.node_indices["rsu1"])
        self.assertEqual(first.edge_indices[edge.edge_id], third.edge_indices[edge.edge_id])
        self.assertEqual(1, third.flow_indices["task1"])

    def test_first_observation_is_deterministic_and_rejects_duplicates(self):
        vocabulary = TrajectoryVocabulary()
        snapshot = vocabulary.observe(
            node_ids=("uav1", "rsu0", "uav0"), edges=(), flow_ids=("task1", "task0")
        )
        self.assertEqual({"rsu0": 0, "uav0": 1, "uav1": 2}, snapshot.node_indices)
        self.assertEqual({"task0": 0, "task1": 1}, snapshot.flow_indices)
        with self.assertRaisesRegex(ValueError, "duplicate node"):
            vocabulary.observe(node_ids=("rsu0", "rsu0"), edges=(), flow_ids=())
        with self.assertRaisesRegex(ValueError, "duplicate flow"):
            vocabulary.observe(node_ids=(), edges=(), flow_ids=("task2", "task2"))

    def test_edge_binding_conflict_is_rejected_without_partial_mutation(self):
        vocabulary = TrajectoryVocabulary()
        edge = EdgeIdentity("ie0", "uav0", "rsu0", "wireless:U2I")
        vocabulary.observe(
            node_ids=("uav0", "rsu0"), edges=(edge,), flow_ids=("task0",)
        )
        before = vocabulary.snapshot()
        with self.assertRaisesRegex(ValueError, "binding"):
            vocabulary.observe(
                node_ids=("uav0", "rsu0", "rsu1"),
                edges=(EdgeIdentity("ie0", "uav0", "rsu1", "wireless:U2I"),),
                flow_ids=("task1",),
            )
        self.assertEqual(before, vocabulary.snapshot())

    def test_dangling_edge_is_rejected_without_partial_mutation(self):
        vocabulary = TrajectoryVocabulary()
        vocabulary.observe(node_ids=("uav0",), edges=(), flow_ids=())
        before = vocabulary.snapshot()
        with self.assertRaisesRegex(ValueError, "endpoint"):
            vocabulary.observe(
                node_ids=("uav0",),
                edges=(EdgeIdentity("ie0", "uav0", "rsu0", "wireless:U2I"),),
                flow_ids=("task0",),
            )
        self.assertEqual(before, vocabulary.snapshot())


class LinkHistoryLedgerTests(unittest.TestCase):
    def test_first_projection_is_no_history(self):
        history = LinkHistoryLedger(edge_ids=("ie0",))
        first = history.project(edge_ids=("ie0",))[0]
        self.assertFalse(first.valid)
        self.assertEqual(MissingReason.NO_HISTORY, first.missing_reason)
        self.assertEqual((0.0, 0.0, 0.0), first.values)

    def test_positive_then_observed_zero_outcome_remains_valid(self):
        history = LinkHistoryLedger(edge_ids=("ie0",))
        history.commit(
            frame_index=0,
            outcomes={
                "ie0": LinkOutcome(
                    active_flow_count=1.0,
                    effective_rate_per_s=4.0,
                    served_data=0.4,
                )
            },
            frame_validated=True,
        )
        second = history.project(edge_ids=("ie0",))[0]
        self.assertEqual((1.0, 4.0, 0.4), second.values)
        self.assertTrue(second.valid)
        self.assertEqual(MissingReason.NONE, second.missing_reason)

        history.commit(
            frame_index=1,
            outcomes={"ie0": LinkOutcome(0.0, 0.0, 0.0)},
            frame_validated=True,
        )
        third = history.project(edge_ids=("ie0",))[0]
        self.assertEqual((0.0, 0.0, 0.0), third.values)
        self.assertTrue(third.valid)
        self.assertEqual(MissingReason.NONE, third.missing_reason)

    def test_failed_or_invalid_commit_does_not_mutate_history(self):
        history = LinkHistoryLedger(edge_ids=("ie0",))
        history.commit(
            frame_index=0,
            outcomes={"ie0": LinkOutcome(1.0, 4.0, 0.4)},
            frame_validated=True,
        )
        before = history.snapshot()
        with self.assertRaisesRegex(ValueError, "validated"):
            history.commit(
                frame_index=1,
                outcomes={"ie0": LinkOutcome(0.0, 0.0, 0.0)},
                frame_validated=False,
            )
        self.assertEqual(before, history.snapshot())
        with self.assertRaisesRegex(ValueError, "nonnegative finite"):
            history.commit(
                frame_index=1,
                outcomes={"ie0": LinkOutcome(1.0, float("nan"), 0.0)},
                frame_validated=True,
            )
        self.assertEqual(before, history.snapshot())

    def test_frame_indices_and_edge_ids_are_strict(self):
        history = LinkHistoryLedger(edge_ids=("ie0",))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            history.commit(
                frame_index=1,
                outcomes={"ie0": LinkOutcome(0.0, 0.0, 0.0)},
                frame_validated=True,
            )
        with self.assertRaisesRegex(ValueError, "unknown edge"):
            history.commit(
                frame_index=0,
                outcomes={"missing": LinkOutcome(0.0, 0.0, 0.0)},
                frame_validated=True,
            )


if __name__ == "__main__":
    unittest.main()
