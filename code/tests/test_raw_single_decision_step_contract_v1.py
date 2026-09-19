from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.raw_single_decision_step_contract_v1 import (  # noqa: E402
    ActionBundle,
    CommAction,
    CompAction,
    ContractError,
    DecisionSnapshot,
    EntityState,
    ExecutionReceipt,
    OutcomeSnapshot,
    RouteAction,
    SetterReceipt,
    SingleDecisionStep,
    TaskState,
    UavMobilityAction,
    apply_action_bundle_to_airfogsim,
    load_airfogsim_scheduler_classes_from_source,
    validate_single_decision_step,
)


def _entities(*, uav_x: float = 0.0):
    return (
        EntityState("uav-0", "uav", True, (uav_x, 0.0, 10.0)),
        EntityState("vehicle-0", "vehicle", True, (5.0, 0.0, 0.0)),
        EntityState("rsu-0", "rsu", True, (20.0, 0.0, 0.0)),
    )


def _tasks(*, routed: bool = False, computed: float = 0.0):
    return (
        TaskState(
            "task-route",
            "vehicle-0",
            "vehicle-0" if not routed else "rsu-0",
            "waiting_to_offload" if not routed else "offloading",
            () if not routed else ("rsu-0",),
            None,
            0.0,
            10.0,
            2.0,
            0.0,
        ),
        TaskState(
            "task-comp",
            "vehicle-0",
            "rsu-0",
            "computing",
            (),
            None,
            0.0,
            0.0,
            5.0,
            computed,
        ),
    )


def _decision(*, entities=None, tasks=None, source_phases=None):
    return DecisionSnapshot(
        trajectory_id="trajectory-0",
        frame_index=4,
        decision_time_s=1.0,
        slot_duration_s=0.1,
        entities=_entities() if entities is None else entities,
        tasks=_tasks() if tasks is None else tasks,
        n_rb=4,
        node_cpu_capacity_per_s={"rsu-0": 3.0},
        source_phases={
            "entities": "decision",
            "tasks": "decision",
            "channel": "decision",
            "resources": "decision",
        }
        if source_phases is None
        else source_phases,
    )


def _action(*, mobility=None):
    return ActionBundle(
        trajectory_id="trajectory-0",
        frame_index=4,
        decision_time_s=1.0,
        route=(
            RouteAction(
                "task-route",
                "vehicle-0",
                "offload",
                "rsu-0",
                ("rsu-0",),
            ),
        ),
        comm=(CommAction("task-route", (0, 1)),),
        comp=(CompAction("task-comp", "rsu-0", 2.0),),
        mobility=(
            UavMobilityAction("uav-0", 0.0, 0.0, 10.0),
        )
        if mobility is None
        else mobility,
    )


def _receipt():
    return ExecutionReceipt(
        execution_start_time_s=1.0,
        execution_end_time_s=1.1,
        setter_calls=(
            SetterReceipt("offload", "task-route", True),
            SetterReceipt("rb", "task-route", True),
            SetterReceipt("cpu_callback", None, True),
            SetterReceipt("uav_mobility", "uav-0", True),
        ),
        env_step_called=True,
        env_step_completed=True,
    )


def _valid_step():
    after_entities = _entities(uav_x=1.0)
    after_tasks = _tasks(routed=True, computed=0.2)
    outcome = OutcomeSnapshot(
        outcome_time_s=1.1,
        entities=after_entities,
        tasks=after_tasks,
        delivered_data_by_task={"task-route": 0.5},
        served_cpu_work_by_task={"task-comp": 0.2},
    )
    next_decision = _decision(entities=after_entities, tasks=after_tasks)
    next_decision = replace(next_decision, frame_index=5, decision_time_s=1.1)
    return SingleDecisionStep(
        decision=_decision(),
        action=_action(),
        execution=_receipt(),
        outcome=outcome,
        next_decision=next_decision,
    )


class ContractValidationTests(unittest.TestCase):
    def test_valid_four_family_step_closes_at_next_decision(self):
        step = _valid_step()
        self.assertIs(step, validate_single_decision_step(step))

    def test_same_slot_outcome_cannot_be_a_decision_source(self):
        step = _valid_step()
        bad_decision = replace(
            step.decision,
            source_phases={**step.decision.source_phases, "channel": "outcome"},
        )
        with self.assertRaisesRegex(ContractError, "same_slot_outcome_leak"):
            validate_single_decision_step(replace(step, decision=bad_decision))

    def test_future_task_cannot_enter_decision_observation(self):
        step = _valid_step()
        future_task = replace(step.decision.tasks[0], arrival_time_s=1.2)
        bad_decision = replace(
            step.decision,
            tasks=(future_task, *step.decision.tasks[1:]),
        )
        with self.assertRaisesRegex(ContractError, "future_task_observation_leak"):
            validate_single_decision_step(replace(step, decision=bad_decision))

    def test_mobility_is_uav_only_and_every_present_uav_has_an_explicit_action(self):
        vehicle_action = UavMobilityAction("vehicle-0", 0.0, 0.0, 1.0)
        with self.assertRaisesRegex(ContractError, "mobility_non_uav"):
            validate_single_decision_step(
                replace(_valid_step(), action=_action(mobility=(vehicle_action,)))
            )

        with self.assertRaisesRegex(ContractError, "missing_uav_mobility_action"):
            validate_single_decision_step(
                replace(_valid_step(), action=_action(mobility=()))
            )

    def test_comp_allocation_respects_decision_time_capacity(self):
        step = _valid_step()
        bad_action = replace(
            step.action,
            comp=(CompAction("task-comp", "rsu-0", 4.0),),
        )
        with self.assertRaisesRegex(ContractError, "cpu_capacity_exceeded"):
            validate_single_decision_step(replace(step, action=bad_action))

    def test_outcome_and_next_decision_must_align_in_time_and_identity(self):
        step = _valid_step()
        bad_next = replace(step.next_decision, decision_time_s=1.2)
        with self.assertRaisesRegex(ContractError, "next_decision_time_mismatch"):
            validate_single_decision_step(replace(step, next_decision=bad_next))

        missing_uav = replace(
            step.next_decision,
            entities=tuple(row for row in step.next_decision.entities if row.entity_id != "uav-0"),
        )
        with self.assertRaisesRegex(ContractError, "outcome_next_decision_mismatch"):
            validate_single_decision_step(replace(step, next_decision=missing_uav))


class _TaskManager:
    def __init__(self):
        self.route = None

    def offloadTask(self, task_node_id, task_id, target_node_id, current_time, route):
        self.route = (task_node_id, task_id, target_node_id, current_time, tuple(route))
        return True


class _ChannelManager:
    n_RB = 4


class _RuntimeTask:
    def __init__(self, task_id):
        self.task_id = task_id
        self.computed = 0.0

    def getTaskId(self):
        return self.task_id


class _MinimalEnv:
    def __init__(self):
        self.simulation_time = 1.0
        self.simulation_interval = 0.1
        self.task_manager = _TaskManager()
        self.channel_manager = _ChannelManager()
        self.activated_offloading_tasks_with_RB_Nos = {}
        self.task_return_routes = {}
        self.uav_mobility_patterns = {}
        self.alloc_cpu_callback = None
        self.uav_position = [0.0, 0.0, 10.0]
        self.compute_task = _RuntimeTask("task-comp")

    def step(self):
        pattern = self.uav_mobility_patterns["uav-0"]
        distance = pattern["speed"] * self.simulation_interval
        self.uav_position[0] += distance * math.cos(pattern["angle"]) * math.cos(pattern["phi"])
        self.uav_position[1] += distance * math.sin(pattern["angle"]) * math.cos(pattern["phi"])
        self.uav_position[2] += distance * math.sin(pattern["phi"])
        allocations = self.alloc_cpu_callback({"rsu-0": [self.compute_task]})
        self.compute_task.computed += allocations["task-comp"] * self.simulation_interval
        self.simulation_time += self.simulation_interval
        return False


class RealSchedulerBoundaryTests(unittest.TestCase):
    def test_repository_airfogsim_sources_execute_all_four_action_families(self):
        schedulers = load_airfogsim_scheduler_classes_from_source(
            CODE_ROOT / "reference" / "AirFogSim"
        )
        env = _MinimalEnv()

        receipt = apply_action_bundle_to_airfogsim(
            env,
            _action(),
            task_scheduler=schedulers.task,
            communication_scheduler=schedulers.communication,
            computation_scheduler=schedulers.computation,
            traffic_scheduler=schedulers.traffic,
        )

        self.assertTrue(receipt.env_step_completed)
        self.assertEqual((0, 1), tuple(env.activated_offloading_tasks_with_RB_Nos["task-route"]))
        self.assertEqual(1.0, env.uav_position[0])
        self.assertAlmostEqual(0.2, env.compute_task.computed)
        self.assertEqual(
            ["offload", "rb", "cpu_callback", "uav_mobility"],
            [row.setter_kind for row in receipt.setter_calls],
        )


if __name__ == "__main__":
    unittest.main()
