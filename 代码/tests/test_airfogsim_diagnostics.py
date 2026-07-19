import copy
import importlib.util
import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
AIRFOGSIM_ROOT = CODE_ROOT / "reference" / "AirFogSim"
for path in (CODE_ROOT / "src", AIRFOGSIM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class RewardDiagnosticsTest(unittest.TestCase):
    def test_reward_components_reconstruct_task_utility(self):
        from pi_jwm.airfogsim_diagnostics import reward_components

        result = reward_components(
            delta_done=2,
            delta_failed=1,
            throughput_delta=99.0,
            throughput_weight=0.01,
        )

        self.assertEqual(result["reward_done"], 2.0)
        self.assertEqual(result["reward_failed"], -1.0)
        self.assertGreater(result["reward_throughput"], 0.0)
        self.assertAlmostEqual(
            result["task_utility"],
            result["reward_done"] + result["reward_failed"] + result["reward_throughput"],
        )

    def test_reward_components_clip_negative_throughput_to_zero(self):
        from pi_jwm.airfogsim_diagnostics import reward_components

        result = reward_components(0, 0, -10.0)

        self.assertEqual(result["throughput_delta"], 0.0)
        self.assertEqual(result["task_utility"], 0.0)


class EnergyDiagnosticsTest(unittest.TestCase):
    def test_energy_snapshot_totals_include_active_and_removed_uavs(self):
        from pi_jwm.airfogsim_diagnostics import energy_snapshot_totals

        snapshot = {
            "uavs": {
                "UAV_1": {
                    "status": "active",
                    "remaining_energy": 80.0,
                    "last_consumption": {
                        "fly": 2.5,
                        "hover": 0.0,
                        "sensing": 0.5,
                        "receive": 0.2,
                        "send": 0.3,
                        "total": 3.5,
                    },
                },
                "UAV_2": {
                    "status": "removed",
                    "remaining_energy": -0.5,
                    "last_consumption": {
                        "fly": 0.0,
                        "hover": 1.2,
                        "sensing": 0.0,
                        "receive": 0.1,
                        "send": 0.2,
                        "total": 1.5,
                    },
                },
            }
        }

        result = energy_snapshot_totals(snapshot)

        self.assertEqual(result["energy_num_active"], 1)
        self.assertEqual(result["energy_num_removed"], 1)
        self.assertAlmostEqual(result["energy_remaining"], 79.5)
        self.assertAlmostEqual(result["energy_fly"], 2.5)
        self.assertAlmostEqual(result["energy_hover"], 1.2)
        self.assertAlmostEqual(result["energy_total"], 5.0)
        self.assertAlmostEqual(
            result["energy_total"],
            result["energy_fly"]
            + result["energy_hover"]
            + result["energy_sensing"]
            + result["energy_receive"]
            + result["energy_send"],
        )

    def test_energy_manager_snapshot_survives_uav_exhaustion_and_is_read_only(self):
        from pi_jwm.airfogsim_runtime import capture_energy_manager_snapshot

        module_path = AIRFOGSIM_ROOT / "airfogsim" / "manager" / "energy_manager.py"
        spec = importlib.util.spec_from_file_location("pi_jwm_test_energy_manager", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        EnergyManager = module.EnergyManager

        manager = EnergyManager(
            {
                "initial_energy_range": [1, 1],
                "fly_unit_cost": 2.5,
                "hover_unit_cost": 1.2,
                "sensing_unit_cost": 0.5,
                "receive_unit_cost": 0.1,
                "send_unit_cost": 0.1,
            },
            ["UAV_1"],
        )
        manager.updateEnergyPattern("UAV_1", True, 0, 0.0, 0.0)
        manager.updateEnergy()

        snapshot = capture_energy_manager_snapshot(manager)
        self.assertEqual(snapshot["uavs"]["UAV_1"]["status"], "removed")
        self.assertAlmostEqual(snapshot["uavs"]["UAV_1"]["remaining_energy"], -1.5)
        self.assertAlmostEqual(snapshot["uavs"]["UAV_1"]["last_consumption"]["total"], 2.5)

        mutated = copy.deepcopy(snapshot)
        mutated["uavs"]["UAV_1"]["remaining_energy"] = 999.0
        self.assertAlmostEqual(
            capture_energy_manager_snapshot(manager)["uavs"]["UAV_1"]["remaining_energy"],
            -1.5,
        )


class PairedAttributionTest(unittest.TestCase):
    def test_paired_candidate_effects_use_same_group_default(self):
        from pi_jwm.airfogsim_diagnostics import paired_candidate_effects

        rows = [
            {
                "seed": 0,
                "decision_time": 1.0,
                "candidate_id": "default",
                "action_family": "default",
                "task_utility": 1.0,
                "throughput_delta": 10.0,
                "energy_total": 5.0,
            },
            {
                "seed": 0,
                "decision_time": 1.0,
                "candidate_id": "rb_more",
                "action_family": "rb_count",
                "task_utility": 1.5,
                "throughput_delta": 12.0,
                "energy_total": 6.0,
            },
        ]

        result = paired_candidate_effects(
            rows,
            metric_fields=("task_utility", "throughput_delta", "energy_total"),
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["candidate_id"], "rb_more")
        self.assertAlmostEqual(result[0]["effect_task_utility"], 0.5)
        self.assertAlmostEqual(result[0]["effect_throughput_delta"], 2.0)
        self.assertAlmostEqual(result[0]["effect_energy_total"], 1.0)

    def test_paired_candidate_effects_reject_missing_default(self):
        from pi_jwm.airfogsim_diagnostics import paired_candidate_effects

        with self.assertRaisesRegex(ValueError, "exactly one default"):
            paired_candidate_effects(
                [
                    {
                        "seed": 0,
                        "decision_time": 1.0,
                        "candidate_id": "rb_more",
                        "action_family": "rb_count",
                        "task_utility": 1.0,
                    }
                ],
                metric_fields=("task_utility",),
            )


class StepAggregationTest(unittest.TestCase):
    def test_energy_step_metrics_use_before_after_remaining_and_after_components(self):
        from pi_jwm.airfogsim_diagnostics import energy_step_metrics

        before = {
            "uavs": {
                "UAV_1": {
                    "status": "active",
                    "remaining_energy": 100.0,
                    "last_consumption": {},
                }
            }
        }
        after = {
            "uavs": {
                "UAV_1": {
                    "status": "active",
                    "remaining_energy": 97.5,
                    "last_consumption": {
                        "fly": 2.5,
                        "hover": 0.0,
                        "sensing": 0.0,
                        "receive": 0.0,
                        "send": 0.0,
                        "total": 2.5,
                    },
                }
            }
        }

        result = energy_step_metrics(before, after)

        self.assertAlmostEqual(result["energy_before"], 100.0)
        self.assertAlmostEqual(result["energy_after"], 97.5)
        self.assertAlmostEqual(result["energy_total"], 2.5)
        self.assertAlmostEqual(result["energy_balance_error"], 0.0)

    def test_energy_step_metrics_do_not_repeat_stale_removed_uav_consumption(self):
        from pi_jwm.airfogsim_diagnostics import energy_step_metrics

        removed = {
            "uavs": {
                "UAV_1": {
                    "status": "removed",
                    "remaining_energy": -1.5,
                    "last_consumption": {
                        "fly": 2.5,
                        "hover": 0.0,
                        "sensing": 0.0,
                        "receive": 0.0,
                        "send": 0.0,
                        "total": 2.5,
                    },
                }
            }
        }

        result = energy_step_metrics(removed, removed)

        self.assertAlmostEqual(result["energy_total"], 0.0)
        self.assertAlmostEqual(result["energy_balance_error"], 0.0)

    def test_summarize_candidate_steps_reconstructs_totals(self):
        from pi_jwm.airfogsim_diagnostics import summarize_candidate_steps

        common = {
            "seed": 0,
            "decision_time": 1.0,
            "candidate_id": "default",
            "action_family": "default",
            "rb_total": 2.0,
            "cpu_total": 3.0,
            "action_applied": True,
        }
        rows = [
            {
                **common,
                "step": 0,
                "delta_done": 0.0,
                "delta_failed": 0.0,
                "throughput_delta": 10.0,
                "reward_done": 0.0,
                "reward_failed": 0.0,
                "reward_throughput": 0.1,
                "task_utility": 0.1,
                "energy_total": 2.5,
                "energy_after": 97.5,
            },
            {
                **common,
                "step": 1,
                "delta_done": 1.0,
                "delta_failed": 0.0,
                "throughput_delta": 20.0,
                "reward_done": 1.0,
                "reward_failed": 0.0,
                "reward_throughput": 0.2,
                "task_utility": 1.2,
                "energy_total": 1.2,
                "energy_after": 96.3,
            },
        ]

        result = summarize_candidate_steps(rows)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["num_steps"], 2)
        self.assertAlmostEqual(result[0]["delta_done"], 1.0)
        self.assertAlmostEqual(result[0]["throughput_delta"], 30.0)
        self.assertAlmostEqual(result[0]["task_utility"], 1.3)
        self.assertAlmostEqual(result[0]["energy_total"], 3.7)
        self.assertAlmostEqual(result[0]["energy_after"], 96.3)

    def test_audit_diagnostic_rows_flags_balance_and_reward_errors(self):
        from pi_jwm.airfogsim_diagnostics import audit_diagnostic_rows

        rows = [
            {
                "seed": 0,
                "decision_time": 1.0,
                "step": 0,
                "candidate_id": "default",
                "action_family": "default",
                "reward_done": 1.0,
                "reward_failed": 0.0,
                "reward_throughput": 0.1,
                "task_utility": 99.0,
                "energy_total": 2.5,
                "energy_balance_error": 0.5,
                "action_applied": True,
            }
        ]

        result = audit_diagnostic_rows(rows)

        self.assertFalse(result["passed"])
        self.assertEqual(result["reward_reconstruction_errors"], 1)
        self.assertEqual(result["energy_balance_errors"], 1)

    def test_audit_diagnostic_rows_fails_when_candidate_action_was_not_applied(self):
        from pi_jwm.airfogsim_diagnostics import audit_diagnostic_rows

        result = audit_diagnostic_rows(
            [
                {
                    "reward_done": 0.0,
                    "reward_failed": 0.0,
                    "reward_throughput": 0.0,
                    "task_utility": 0.0,
                    "energy_total": 0.0,
                    "energy_balance_error": 0.0,
                    "action_applied": False,
                }
            ]
        )

        self.assertEqual(result["invalid_action_rows"], 1)
        self.assertFalse(result["passed"])


class AirFogSimRuntimeTest(unittest.TestCase):
    def test_channel_throughput_uses_global_total_without_subtype_double_counting(self):
        module_path = CODE_ROOT / "scripts" / "run_airfogsim_counterfactual_action_smoke_v0.py"
        spec = importlib.util.spec_from_file_location("airfogsim_counterfactual_smoke", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class FakeEnv:
            channel = {"data_size": 100.0}
            V2I_channel = {"data_size": 25.0}
            V2U_channel = {"data_size": 20.0}
            U2I_channel = {"data_size": 15.0}

        self.assertEqual(module.channel_throughput(FakeEnv()), 100.0)

    def test_runtime_paths_point_to_reference_simulator(self):
        from pi_jwm.airfogsim_runtime import resolve_airfogsim_paths

        root, examples = resolve_airfogsim_paths(CODE_ROOT)

        self.assertEqual(root, CODE_ROOT / "reference" / "AirFogSim")
        self.assertEqual(examples, root / "examples")

    def test_diagnostic_config_is_copied_before_overrides(self):
        from pi_jwm.airfogsim_runtime import make_diagnostic_config

        source = {
            "simulation": {"max_simulation_time": 100.0},
            "traffic": {"max_n_vehicles": 1, "max_n_UAVs": 1, "RSU_positions": []},
            "task_profile": {"task_node_gen_poss": 0.1},
        }

        result = make_diagnostic_config(source, max_time=10.0)

        self.assertEqual(source["simulation"]["max_simulation_time"], 100.0)
        self.assertEqual(result["simulation"]["max_simulation_time"], 10.0)
        self.assertEqual(result["traffic"]["max_n_UAVs"], 2)
        self.assertEqual(result["task_profile"]["task_node_gen_poss"], 0.8)


class DiagnosticRunnerContractTest(unittest.TestCase):
    def _load_runner(self):
        module_path = CODE_ROOT / "scripts" / "run_pi_jwm_energy_reward_diagnostic.py"
        spec = importlib.util.spec_from_file_location("pi_jwm_energy_reward_runner", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_runner_defaults_to_canonical_artifact_directory(self):
        module = self._load_runner()

        self.assertEqual(
            module.DEFAULT_OUTPUT_DIR,
            CODE_ROOT / "artifacts" / "reports" / "pi_jwm_energy_reward_diagnostic_20260713",
        )

    def test_balanced_candidate_selection_preserves_action_family_coverage(self):
        module = self._load_runner()
        candidates = [
            {"candidate_id": "default", "action_family": "rb_count"},
            {"candidate_id": "rb_1", "action_family": "rb_count"},
            {"candidate_id": "rb_2", "action_family": "rb_count"},
            {"candidate_id": "cpu", "action_family": "cpu_scale"},
            {"candidate_id": "return", "action_family": "return_route"},
        ]

        selected = module.select_balanced_candidates(candidates, max_candidates=4)

        self.assertEqual(selected[0]["action_family"], "default")
        self.assertEqual(
            {item["action_family"] for item in selected},
            {"default", "rb_count", "cpu_scale", "return_route"},
        )

    def test_markdown_table_does_not_require_optional_tabulate(self):
        import pandas as pd

        module = self._load_runner()
        rendered = module.markdown_table(pd.DataFrame([{"family": "rb", "effect": 1.25}]))

        self.assertIn("| family | effect |", rendered)
        self.assertIn("| rb | 1.25 |", rendered)

    def test_candidate_action_metadata_preserves_ranking_features(self):
        module = self._load_runner()
        candidate = {
            "rb_scale": 1.5,
            "cpu_scale": 0.5,
            "rb_plan": {"task_1": [0, 1]},
            "total_rb": 2,
            "total_cpu": 3.0,
            "num_offload_overrides": 1,
            "num_cpu_overrides": 2,
            "num_return_route_overrides": 0,
        }

        result = module.candidate_action_metadata(candidate, horizon=3)

        self.assertEqual(result["horizon"], 3)
        self.assertEqual(result["num_rb_tasks"], 1)
        self.assertEqual(result["total_rb"], 2)
        self.assertEqual(result["total_cpu"], 3.0)
        self.assertEqual(result["num_offload_overrides"], 1)

    def test_selector_timed_intervention_skips_step_zero(self):
        module = self._load_runner()

        self.assertFalse(module.temporal_pattern_active(0, 1, "persistent"))
        self.assertTrue(module.temporal_pattern_active(1, 1, "persistent"))
        self.assertTrue(module.temporal_pattern_active(2, 1, "persistent"))
        self.assertTrue(module.temporal_pattern_active(1, 1, "decayed"))
        self.assertFalse(module.temporal_pattern_active(2, 1, "decayed"))

    def test_selector_timed_intervention_rejects_unknown_pattern(self):
        module = self._load_runner()

        with self.assertRaisesRegex(ValueError, "temporal pattern"):
            module.temporal_pattern_active(1, 1, "unknown")

    def test_aligned_point_rows_include_sample_id(self):
        module = self._load_runner()

        aligned, rejected = module.align_points_to_sample_index(
            [{"seed": 2, "decision_time": 1.0}],
            [{"sample_id": 782, "seed": 2, "input_end_time": 1.0}],
        )

        self.assertEqual(aligned[0]["sample_id"], 782)
        self.assertEqual(rejected, [])

    def test_stage_selection_prefers_decision_time_support_coverage(self):
        module = self._load_runner()
        selected = module.select_decision_points(
            [
                {
                    "seed": 0,
                    "decision_time": 0.3,
                    "decision_stage": "offload_rb",
                    "num_to_offload_tasks": 1,
                    "default_rb_plan": {"a": [0]},
                },
                {
                    "seed": 0,
                    "decision_time": 8.1,
                    "decision_stage": "offload_rb",
                    "num_to_offload_tasks": 8,
                    "default_rb_plan": {str(i): [i] for i in range(7)},
                },
                {
                    "seed": 0,
                    "decision_time": 4.4,
                    "decision_stage": "compute",
                    "num_computing_tasks": 1,
                },
                {
                    "seed": 0,
                    "decision_time": 6.1,
                    "decision_stage": "compute",
                    "num_computing_tasks": 2,
                },
            ],
            max_points=2,
        )

        self.assertEqual([row["decision_time"] for row in selected], [8.1, 6.1])

    def test_temporal_metadata_is_explicit(self):
        module = self._load_runner()

        metadata = module.candidate_action_metadata(
            {
                "rb_plan": {},
                "intervention_start_step": 1,
                "temporal_pattern": "decayed",
            },
            horizon=3,
        )

        self.assertEqual(metadata["intervention_start_step"], 1)
        self.assertEqual(metadata["temporal_pattern"], "decayed")

    def test_prepare_candidate_step_schedules_default_before_delayed_override(self):
        module = self._load_runner()
        events = []

        class FakeEnv:
            activated_offloading_tasks_with_RB_Nos = {}

        class FakeAlgorithm:
            def scheduleStep(self, env):
                events.append("schedule_default")
                env.activated_offloading_tasks_with_RB_Nos = {"task": [9]}

        env = FakeEnv()
        algorithm = FakeAlgorithm()
        candidate = {
            "action_family": "rb_count",
            "rb_plan": {"task": [0, 1]},
            "num_offload_overrides": 0,
            "num_cpu_overrides": 0,
            "num_return_route_overrides": 0,
        }

        inactive = module.prepare_candidate_step(env, algorithm, candidate, active=False)
        self.assertEqual(env.activated_offloading_tasks_with_RB_Nos, {"task": [9]})
        self.assertEqual(
            inactive,
            {
                "offload": 0,
                "cpu": 0,
                "return_route": 0,
                "rb": 0,
                "action_applicable": False,
                "action_supported": False,
                "action_changed": False,
            },
        )

        active = module.prepare_candidate_step(env, algorithm, candidate, active=True)
        self.assertEqual(env.activated_offloading_tasks_with_RB_Nos, {"task": [0, 1]})
        self.assertEqual(
            active,
            {
                "offload": 0,
                "cpu": 0,
                "return_route": 0,
                "rb": 1,
                "action_applicable": True,
                "action_supported": False,
                "action_changed": True,
            },
        )
        self.assertEqual(events, ["schedule_default", "schedule_default"])

    def test_causal_rb_policy_resolves_new_current_task_ids(self):
        module = self._load_runner()

        class FakeCommScheduler:
            def getNumberOfRB(self, env):
                return 4

        class FakeAlgorithm:
            commScheduler = FakeCommScheduler()

            def scheduleStep(self, env):
                env.activated_offloading_tasks_with_RB_Nos = {"new-task": [0, 1, 2, 3]}

        class FakeEnv:
            activated_offloading_tasks_with_RB_Nos = {}

        env = FakeEnv()
        result = module.prepare_candidate_step(
            env,
            FakeAlgorithm(),
            {
                "action_protocol": "causal_policy_v1",
                "action_family": "rb_count",
                "rb_scale": 0.5,
                "policy_coverage": 1,
                "policy_rank": 1,
                "rb_plan": {},
                "offload_overrides": {},
                "cpu_overrides": {},
                "return_route_overrides": {},
            },
            active=True,
        )

        self.assertEqual(env.activated_offloading_tasks_with_RB_Nos, {"new-task": [0, 1]})
        self.assertEqual(result["rb"], 1)
        self.assertTrue(result["action_applicable"])
        self.assertTrue(result["action_supported"])
        self.assertTrue(result["action_changed"])

    def test_causal_policy_without_current_support_is_safe_noop(self):
        module = self._load_runner()

        class FakeCommScheduler:
            def getNumberOfRB(self, env):
                return 4

        class FakeAlgorithm:
            commScheduler = FakeCommScheduler()

            def scheduleStep(self, env):
                env.activated_offloading_tasks_with_RB_Nos = {}

        class FakeEnv:
            activated_offloading_tasks_with_RB_Nos = {}

        result = module.prepare_candidate_step(
            FakeEnv(),
            FakeAlgorithm(),
            {
                "action_protocol": "causal_policy_v1",
                "action_family": "rb_count",
                "rb_scale": 0.5,
                "policy_coverage": 1,
                "policy_rank": 1,
                "rb_plan": {},
                "offload_overrides": {},
                "cpu_overrides": {},
                "return_route_overrides": {},
            },
            active=True,
        )

        self.assertTrue(result["action_applicable"])
        self.assertFalse(result["action_supported"])
        self.assertFalse(result["action_changed"])

    def test_causal_cpu_policy_scales_current_default_allocation(self):
        module = self._load_runner()

        class FakeTask:
            def getTaskId(self):
                return "new-task"

            def getAssignedTo(self):
                return "node"

            def getCurrentNodeId(self):
                return "node"

        class FakeManager:
            def getComputingTasks(self):
                return {"node": [FakeTask()]}

        class FakeAlgorithm:
            def scheduleStep(self, env):
                env.alloc_cpu_callback = lambda computing_tasks, **kwargs: {"new-task": 4.0}

        class FakeEnv:
            task_manager = FakeManager()
            alloc_cpu_callback = None

        env = FakeEnv()
        result = module.prepare_candidate_step(
            env,
            FakeAlgorithm(),
            {
                "action_protocol": "causal_policy_v1",
                "action_family": "cpu_scale",
                "cpu_scale": 0.5,
                "policy_coverage": 1,
                "policy_rank": 1,
                "rb_plan": {},
                "offload_overrides": {},
                "cpu_overrides": {},
                "return_route_overrides": {},
            },
            active=True,
        )

        self.assertEqual(env.alloc_cpu_callback(env.task_manager.getComputingTasks()), {"new-task": 2.0})
        self.assertEqual(result["cpu"], 1)
        self.assertTrue(result["action_applicable"])
        self.assertTrue(result["action_changed"])

    def test_causal_offload_policy_uses_fixed_alternative_rank(self):
        module = self._load_runner()
        targets = []

        class FakeTask:
            def isComputing(self):
                return False

            def isComputed(self):
                return False

            def changeOffloadTo(self, target_id, route, current_time):
                targets.append((target_id, route, current_time))

        class FakeManager:
            def getTaskByTaskId(self, task_id):
                return FakeTask()

        class FakeTaskScheduler:
            def getAllToOffloadTaskInfos(self, env):
                return [{"task_id": "new-task", "task_node_id": "source"}]

        class FakeEntityScheduler:
            def getNeighborNodeInfosById(self, env, source_id, sorted_by, max_num):
                return [{"id": "nearest"}, {"id": "alternative"}]

        class FakeAlgorithm:
            taskScheduler = FakeTaskScheduler()
            entityScheduler = FakeEntityScheduler()

            def scheduleStep(self, env):
                env.activated_offloading_tasks_with_RB_Nos = {}

        class FakeEnv:
            task_manager = FakeManager()
            activated_offloading_tasks_with_RB_Nos = {}
            simulation_time = 1.0

        result = module.prepare_candidate_step(
            FakeEnv(),
            FakeAlgorithm(),
            {
                "action_protocol": "causal_policy_v1",
                "action_family": "offload_target",
                "policy_coverage": 1,
                "policy_rank": 1,
                "rb_plan": {},
                "offload_overrides": {},
                "cpu_overrides": {},
                "return_route_overrides": {},
            },
            active=True,
        )

        self.assertEqual(targets, [("alternative", ["alternative"], 1.0)])
        self.assertEqual(result["offload"], 1)
        self.assertTrue(result["action_applicable"])
        self.assertTrue(result["action_changed"])

    def test_formal_temporal_candidates_are_task_id_free_causal_rules(self):
        module = self._load_runner()

        expanded = module.build_formal_temporal_candidates(
            [
                {
                    "candidate_id": "default",
                    "action_family": "default",
                    "rb_plan": {},
                },
                {
                    "candidate_id": "rb_task_old",
                    "action_family": "rb_count",
                    "rb_scale": 0.5,
                    "rb_plan": {"old-task": [0, 1]},
                },
            ],
            intervention_start_step=1,
            temporal_patterns=("persistent", "decayed"),
            max_candidates=8,
        )

        self.assertEqual(sum(row["action_family"] == "default" for row in expanded), 1)
        for candidate in expanded:
            self.assertEqual(candidate["action_protocol"], "causal_policy_v1")
            self.assertEqual(candidate.get("rb_plan", {}), {})
            self.assertEqual(candidate.get("offload_overrides", {}), {})
            self.assertEqual(candidate.get("cpu_overrides", {}), {})
            self.assertEqual(candidate.get("return_route_overrides", {}), {})
            self.assertNotIn("old-task", repr(candidate))

    def test_delayed_offload_skips_task_that_started_computing(self):
        module = self._load_runner()

        class FakeTask:
            def isComputing(self):
                return True

            def isComputed(self):
                return False

            def changeOffloadTo(self, *args):
                raise AssertionError("stale offload must not mutate a computing task")

        class FakeManager:
            def getTaskByTaskId(self, task_id):
                return FakeTask()

            def getComputingTasks(self):
                return {}

            def getWaitingToReturnTaskInfos(self):
                return {}

        class FakeEnv:
            task_manager = FakeManager()
            simulation_time = 1.0
            activated_offloading_tasks_with_RB_Nos = {}

        class FakeAlgorithm:
            def scheduleStep(self, env):
                env.activated_offloading_tasks_with_RB_Nos = {}

        counts = module.prepare_candidate_step(
            FakeEnv(),
            FakeAlgorithm(),
            {
                "action_family": "offload_target",
                "offload_overrides": {"task": "node_new"},
                "num_offload_overrides": 1,
                "cpu_overrides": {},
                "return_route_overrides": {},
                "rb_plan": {},
            },
            active=True,
        )

        self.assertEqual(counts["offload"], 0)

    def test_delayed_rb_plan_projects_only_onto_current_default_tasks(self):
        module = self._load_runner()

        class FakeManager:
            def getComputingTasks(self):
                return {}

            def getWaitingToReturnTaskInfos(self):
                return {}

        class FakeEnv:
            task_manager = FakeManager()
            activated_offloading_tasks_with_RB_Nos = {}

        class FakeAlgorithm:
            def scheduleStep(self, env):
                env.activated_offloading_tasks_with_RB_Nos = {
                    "current": [9],
                    "new": [8],
                }

        env = FakeEnv()
        counts = module.prepare_candidate_step(
            env,
            FakeAlgorithm(),
            {
                "action_family": "rb_count",
                "offload_overrides": {},
                "cpu_overrides": {},
                "return_route_overrides": {},
                "rb_plan": {"current": [0, 1], "stale": [2]},
            },
            active=True,
        )

        self.assertEqual(
            env.activated_offloading_tasks_with_RB_Nos,
            {"current": [0, 1], "new": [8]},
        )
        self.assertEqual(counts["rb"], 1)

    def test_delayed_cpu_filters_tasks_outside_current_computing_buckets(self):
        module = self._load_runner()

        class FakeTask:
            def __init__(self, task_id, node_id):
                self.task_id = task_id
                self.node_id = node_id

            def getTaskId(self):
                return self.task_id

            def getAssignedTo(self):
                return self.node_id

            def getCurrentNodeId(self):
                return self.node_id

        class FakeManager:
            def getComputingTasks(self):
                return {"node": [FakeTask("current", "node")]}

        class FakeEnv:
            task_manager = FakeManager()
            alloc_cpu_callback = None

        env = FakeEnv()
        applied = module.apply_cpu_overrides(
            env,
            {"cpu_overrides": {"current": 3.0, "stale": 9.0}},
        )

        self.assertEqual(applied, 1)
        self.assertEqual(env.alloc_cpu_callback(env.task_manager.getComputingTasks()), {"current": 3.0})

    def test_delayed_return_route_skips_task_no_longer_waiting(self):
        module = self._load_runner()
        events = []

        class FakeManager:
            def getTaskByTaskId(self, task_id):
                return object()

            def getWaitingToReturnTaskInfos(self):
                return {}

        class FakeEnv:
            task_manager = FakeManager()

        class FakeScheduler:
            def setTaskReturnRoute(self, *args):
                events.append(args)

        class FakeAlgorithm:
            taskScheduler = FakeScheduler()

        applied = module.apply_return_route_overrides(
            FakeEnv(),
            FakeAlgorithm(),
            {"return_route_overrides": {"stale": ["rsu"]}},
        )

        self.assertEqual(applied, 0)
        self.assertEqual(events, [])

    def test_group_audit_excludes_invalid_candidates_from_nontrivial_spread(self):
        module = self._load_runner()
        frame = module.pd.DataFrame(
            [
                {"seed": 0, "decision_time": 1.0, "task_utility": 1.0, "action_applied": True},
                {"seed": 0, "decision_time": 1.0, "task_utility": 1.0, "action_applied": True},
                {"seed": 0, "decision_time": 1.0, "task_utility": 99.0, "action_applied": False},
            ]
        )

        row = module.make_group_audit(frame).iloc[0]

        self.assertEqual(row["num_candidates"], 3)
        self.assertEqual(row["num_valid_candidates"], 2)
        self.assertEqual(row["num_invalid_actions"], 1)
        self.assertEqual(row["utility_spread"], 0.0)
        self.assertFalse(row["is_nontrivial"])

    def test_family_summary_keeps_invalid_coverage_but_uses_only_valid_effects(self):
        module = self._load_runner()
        frame = module.pd.DataFrame(
            [
                {
                    "action_family": "cpu_scale",
                    "effect_task_utility": 1.0,
                    "effect_energy_total": 2.0,
                    "action_applied": True,
                },
                {
                    "action_family": "cpu_scale",
                    "effect_task_utility": 99.0,
                    "effect_energy_total": 99.0,
                    "action_applied": False,
                },
            ]
        )

        row = module.make_family_summary(frame).iloc[0]

        self.assertEqual(row["num_candidates"], 2)
        self.assertEqual(row["num_valid_candidates"], 1)
        self.assertEqual(row["num_invalid_candidates"], 1)
        self.assertEqual(row["mean_effect_task_utility"], 1.0)
        self.assertEqual(row["mean_effect_energy_total"], 2.0)

    def test_temporal_candidate_expansion_keeps_one_default_and_unique_variants(self):
        module = self._load_runner()

        expanded = module.expand_temporal_candidates(
            [
                {"candidate_id": "default", "action_family": "default", "rb_plan": {}},
                {"candidate_id": "rb_2", "action_family": "rb_count", "rb_plan": {"t": [0, 1]}},
            ],
            intervention_start_step=1,
            temporal_patterns=("persistent", "decayed"),
            max_candidates=8,
        )

        self.assertEqual(sum(row["action_family"] == "default" for row in expanded), 1)
        self.assertEqual(
            {row["candidate_id"] for row in expanded},
            {"default", "rb_2__persistent", "rb_2__decayed"},
        )
        self.assertTrue(all(row["intervention_start_step"] == 1 for row in expanded))

    def test_formal_cli_accepts_alignment_and_temporal_protocol(self):
        module = self._load_runner()

        args = module.parse_args(
            [
                "--sample-index-csv",
                "sample_index.csv",
                "--intervention-start-step",
                "1",
                "--temporal-patterns",
                "persistent",
                "decayed",
            ]
        )

        self.assertEqual(args.sample_index_csv, Path("sample_index.csv"))
        self.assertEqual(args.intervention_start_step, 1)
        self.assertEqual(args.temporal_patterns, ["persistent", "decayed"])

    def test_alignment_precedes_stage_balancing(self):
        module = self._load_runner()

        selected, rejected = module.select_aligned_decision_points(
            [
                {"seed": 0, "decision_time": 0.3, "decision_stage": "offload_rb"},
                {"seed": 0, "decision_time": 0.8, "decision_stage": "offload_rb"},
                {"seed": 0, "decision_time": 1.0, "decision_stage": "compute"},
            ],
            [
                {"sample_id": 0, "seed": 0, "input_end_time": 0.8},
                {"sample_id": 2, "seed": 0, "input_end_time": 1.0},
            ],
            max_points=2,
        )

        self.assertEqual([row["sample_id"] for row in selected], [0, 2])
        self.assertEqual(rejected[0]["decision_time"], 0.3)

    def test_reproduction_command_includes_temporal_protocol(self):
        module = self._load_runner()
        args = module.parse_args(
            [
                "--sample-index-csv",
                "sample_index.csv",
                "--intervention-start-step",
                "1",
                "--temporal-patterns",
                "persistent",
                "decayed",
            ]
        )

        command = module.build_reproduction_command(args)

        self.assertIn("--sample-index-csv sample_index.csv", command)
        self.assertIn("--intervention-start-step 1", command)
        self.assertIn("--temporal-patterns persistent decayed", command)

    def test_metric_definitions_document_raw_and_composite_quantities(self):
        module = self._load_runner()

        result = module.metric_definitions_markdown()

        self.assertIn("task_utility", result)
        self.assertIn("energy_total", result)
        self.assertIn("energy_balance_error", result)
        self.assertIn("敏感性", result)

    def test_effect_axis_uses_compact_non_overlapping_tick_labels(self):
        import matplotlib.pyplot as plt

        module = self._load_runner()
        fig, ax = plt.subplots()
        ax.barh(["rb_count"], [-0.0151017])
        module.format_effect_axis(ax)
        fig.canvas.draw()
        labels = [item.get_text() for item in ax.get_xticklabels() if item.get_text()]
        plt.close(fig)

        self.assertLessEqual(len(labels), 6)
        self.assertTrue(all(len(label.partition(".")[2]) <= 3 for label in labels if "." in label))


class EnergyRankingDiagnosticTest(unittest.TestCase):
    def _load_ranking(self):
        module_path = CODE_ROOT / "scripts" / "run_pi_jwm_energy_utility_ranking_diagnostic.py"
        spec = importlib.util.spec_from_file_location("pi_jwm_energy_ranking", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_energy_targets_use_group_default_overhead(self):
        import pandas as pd

        module = self._load_ranking()
        frame = pd.DataFrame(
            [
                {"seed": 0, "decision_time": 1.0, "action_family": "default", "task_utility": 1.0, "energy_total": 10.0},
                {"seed": 0, "decision_time": 1.0, "action_family": "rb_count", "task_utility": 1.2, "energy_total": 12.0},
            ]
        )

        result = module.add_energy_targets(frame, lambdas=(0.0, 1.0))

        self.assertAlmostEqual(result.iloc[0]["energy_overhead_ratio"], 0.0)
        self.assertAlmostEqual(result.iloc[1]["energy_overhead_ratio"], 0.2)
        self.assertAlmostEqual(result.iloc[1]["target_lambda_0"], 1.2)
        self.assertAlmostEqual(result.iloc[1]["target_lambda_1"], 1.0)

    def test_action_features_do_not_encode_seed_identity(self):
        import pandas as pd

        module = self._load_ranking()
        rows = []
        for seed in (0, 4):
            rows.append(
                {
                    "seed": seed,
                    "rb_scale": 1.0,
                    "total_rb": 2,
                    "num_rb_tasks": 1,
                    "cpu_scale": 1.0,
                    "total_cpu": 0.0,
                    "num_offload_overrides": 0,
                    "num_cpu_overrides": 0,
                    "num_return_route_overrides": 0,
                    "context_num_to_offload_tasks": 1,
                    "context_num_computing_tasks": 0,
                    "context_num_waiting_return_tasks": 0,
                    "action_family": "default",
                }
            )

        features, _ = module.build_action_context_features(pd.DataFrame(rows))

        self.assertTrue((features[0] == features[1]).all())

    def test_combined_findings_separate_facts_interpretations_and_hypotheses(self):
        import pandas as pd

        module = self._load_ranking()
        main_summary = {
            "num_seeds": 5,
            "num_step_rows": 100,
            "num_candidates": 20,
            "num_decision_groups": 5,
            "num_nontrivial_groups": 2,
            "quality_audit": {"passed": True},
        }
        family = pd.DataFrame(
            [
                {
                    "action_family": "rb_count",
                    "num_candidates": 4,
                    "mean_effect_task_utility": -0.1,
                    "positive_utility_ratio": 0.25,
                    "mean_effect_energy_total": -1.0,
                }
            ]
        )
        metrics = pd.DataFrame(
            [
                {
                    "split": "test",
                    "lambda": 0.0,
                    "num_nontrivial_groups": 1,
                    "top1_hit_mean": 0.0,
                    "normalized_top1_regret_mean": 0.9,
                }
            ]
        )

        result = module.build_combined_findings(main_summary, family, metrics)

        self.assertIn("## 观测事实", result)
        self.assertIn("## 合理解释", result)
        self.assertIn("## 待验证假设", result)
        self.assertIn("## 下一步建议", result)

    def test_post_diagnostic_links_are_machine_readable(self):
        module = self._load_ranking()

        result = module.attach_post_diagnostics(
            {"framework": "PI-JWM"},
            ranking_summary="ranking/summary.json",
            combined_findings="findings.md",
        )

        self.assertEqual(result["framework"], "PI-JWM")
        self.assertEqual(result["post_diagnostics"]["ranking_summary"], "ranking/summary.json")
        self.assertEqual(result["post_diagnostics"]["combined_findings"], "findings.md")


if __name__ == "__main__":
    unittest.main()
