from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r6_joint_action import (  # noqa: E402
    CPUAllocation,
    ComputeTaskContext,
    JointActionCandidate,
    JointActionCandidateSet,
    JointActionContext,
    OffloadBinding,
    OffloadTaskContext,
    RBAllocation,
    TargetContext,
    generate_joint_candidates,
    validate_joint_candidate,
)


def _context() -> JointActionContext:
    return JointActionContext.create(
        scenario_id="load_high__density_dense__r07",
        seed=507,
        slot=12,
        split="validation",
        offload_tasks=(
            OffloadTaskContext(
                task_id="Task_1",
                source_node_id="vehicle_0",
                legal_targets=(
                    TargetContext("UAV_0", distance=10.0, available_cpu=20.0, rate_proxy=4.0, energy_proxy=3.0),
                    TargetContext("RSU_0", distance=20.0, available_cpu=50.0, rate_proxy=8.0, energy_proxy=1.0),
                ),
                deadline_remaining=0.4,
                priority=3.0,
                input_mb=2.0,
            ),
            OffloadTaskContext(
                task_id="Task_2",
                source_node_id="vehicle_1",
                legal_targets=(
                    TargetContext("UAV_0", distance=8.0, available_cpu=20.0, rate_proxy=3.0, energy_proxy=2.0),
                    TargetContext("RSU_0", distance=18.0, available_cpu=50.0, rate_proxy=7.0, energy_proxy=1.0),
                ),
                deadline_remaining=1.5,
                priority=1.0,
                input_mb=1.0,
            ),
        ),
        compute_tasks=(
            ComputeTaskContext("Task_3", "UAV_0", node_capacity=12.0, deadline_remaining=0.3, priority=2.0),
            ComputeTaskContext("Task_4", "UAV_0", node_capacity=12.0, deadline_remaining=1.0, priority=1.0),
        ),
        default_rb_plan={"Task_1": [0, 1], "Task_2": [2, 3]},
        rb_capacity=6,
    )


class R6JointActionTest(unittest.TestCase):
    def test_generator_is_deterministic_and_has_one_default_first(self) -> None:
        first = generate_joint_candidates(_context(), max_candidates=6)
        second = generate_joint_candidates(_context(), max_candidates=6)
        self.assertEqual(first, second)
        self.assertEqual("airfogsim_default", first.candidates[0].candidate_id)
        self.assertEqual("default", first.candidates[0].template_id)
        self.assertEqual(1, sum(item.template_id == "default" for item in first.candidates))
        self.assertEqual(
            {"default", "deadline_first", "priority_first", "load_balance", "rate_aware", "energy_conservative"},
            {item.template_id for item in first.candidates},
        )
        self.assertTrue(all(validate_joint_candidate(_context(), item).valid for item in first.candidates))

    def test_every_nondefault_is_complete_and_descriptor_is_fixed(self) -> None:
        candidates = generate_joint_candidates(_context(), max_candidates=6)
        widths = {len(item.descriptor) for item in candidates.candidates}
        self.assertEqual({JointActionCandidate.DESCRIPTOR_DIM}, widths)
        for item in candidates.candidates[1:]:
            self.assertEqual({"Task_1", "Task_2"}, {row.task_id for row in item.offload})
            self.assertEqual({"Task_1", "Task_2"}, {row.task_id for row in item.rb})
            self.assertEqual({"Task_3", "Task_4"}, {row.task_id for row in item.cpu})
        descriptors, mask = candidates.padded_descriptors(max_candidates=8)
        self.assertEqual((8, JointActionCandidate.DESCRIPTOR_DIM), descriptors.shape)
        self.assertEqual([True] * 6 + [False, False], mask.tolist())

    def test_validation_rejects_dag_neighbor_rb_and_cpu_violations(self) -> None:
        context = _context()
        base = generate_joint_candidates(context, max_candidates=6).candidates[1]
        invalid_offload = JointActionCandidate.create(
            candidate_id="bad_target",
            template_id="deadline_first",
            offload=(OffloadBinding("Task_1", "vehicle_0", "cloud_0"),) + base.offload[1:],
            rb=base.rb,
            cpu=base.cpu,
        )
        self.assertIn("legal target", validate_joint_candidate(context, invalid_offload).reasons[0])
        duplicate_rb = JointActionCandidate.create(
            candidate_id="bad_rb",
            template_id="deadline_first",
            offload=base.offload,
            rb=(RBAllocation("Task_1", (0, 1)), RBAllocation("Task_2", (1, 2))),
            cpu=base.cpu,
        )
        self.assertTrue(any("RB" in reason for reason in validate_joint_candidate(context, duplicate_rb).reasons))
        excessive_cpu = JointActionCandidate.create(
            candidate_id="bad_cpu",
            template_id="deadline_first",
            offload=base.offload,
            rb=base.rb,
            cpu=(CPUAllocation("UAV_0", "Task_3", 10.0), CPUAllocation("UAV_0", "Task_4", 10.0)),
        )
        self.assertTrue(any("CPU capacity" in reason for reason in validate_joint_candidate(context, excessive_cpu).reasons))
        unreleased = JointActionCandidate.create(
            candidate_id="bad_task",
            template_id="deadline_first",
            offload=(OffloadBinding("Task_9", "vehicle_0", "UAV_0"),) + base.offload[1:],
            rb=base.rb,
            cpu=base.cpu,
        )
        self.assertTrue(any("DAG-released" in reason for reason in validate_joint_candidate(context, unreleased).reasons))

    def test_candidate_set_rejects_duplicate_default_and_duplicate_ids(self) -> None:
        default = JointActionCandidate.create(
            candidate_id="airfogsim_default",
            template_id="default",
            offload=(),
            rb=(RBAllocation("Task_1", (0, 1)), RBAllocation("Task_2", (2, 3))),
            cpu=(),
        )
        with self.assertRaisesRegex(ValueError, "exactly one default"):
            JointActionCandidateSet.create((default, default))
        other = JointActionCandidate.create(
            candidate_id="airfogsim_default",
            template_id="deadline_first",
            offload=(),
            rb=default.rb,
            cpu=(),
        )
        with self.assertRaisesRegex(ValueError, "candidate_id"):
            JointActionCandidateSet.create((default, other))

    def test_generator_supports_stage_sparse_compute_only_context(self) -> None:
        context = JointActionContext.create(
            scenario_id="compute_only",
            seed=507,
            slot=13,
            split="validation",
            offload_tasks=(),
            compute_tasks=(
                ComputeTaskContext("Task_3", "UAV_0", 12.0, 0.3, 2.0),
                ComputeTaskContext("Task_4", "UAV_0", 12.0, 1.0, 1.0),
            ),
            default_rb_plan={},
            rb_capacity=6,
        )
        candidates = generate_joint_candidates(context, max_candidates=6)
        self.assertEqual(6, len(candidates.candidates))
        self.assertTrue(all(not item.offload and not item.rb for item in candidates.candidates))
        self.assertTrue(all(item.cpu for item in candidates.candidates[1:]))


if __name__ == "__main__":
    unittest.main()
