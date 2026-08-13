from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FakeTask:
    def __init__(self, task_id: str):
        self._task_id = task_id

    def getTaskId(self) -> str:
        return self._task_id


class FakeNode:
    def __init__(self, cpu: float):
        self._cpu = cpu

    def getFogProfile(self) -> dict[str, float]:
        return {"cpu": self._cpu}


class FakeEnv:
    def __init__(self, capacities: dict[str, float]):
        self._nodes = {node_id: FakeNode(cpu) for node_id, cpu in capacities.items()}

    def _getNodeById(self, node_id: str):
        return self._nodes.get(node_id)


class FakeChannelManager:
    def __init__(self):
        self.sending = None
        self.receiving = None

    def setThisTimeslotTransSize(self, sending, receiving) -> None:
        self.sending = dict(sending)
        self.receiving = dict(receiving)

    def getRateByChannelType(self, tx_idx, rx_idx, channel_type, rb_indices):
        self.last_rate_query = (tx_idx, rx_idx, channel_type, list(rb_indices))
        return [1.5, 2.5]


class FakeTransferTask(FakeTask):
    def __init__(self, task_id: str, source: str, route: list[str]):
        super().__init__(task_id)
        self.source = source
        self.route = route

    def getCurrentNodeId(self) -> str:
        return self.source

    def getToOffloadRoute(self) -> list[str]:
        return list(self.route)


class CapacitySafeCpuAllocationTests(unittest.TestCase):
    def test_each_node_uses_its_own_capacity(self):
        from pi_jwm.airfogsim_contract_adapter import capacity_safe_cpu_allocations

        env = FakeEnv({"vehicle": 2.0, "rsu": 9.0})
        computing_tasks = {
            "vehicle": [FakeTask("v0"), FakeTask("v1")],
            "rsu": [FakeTask("r0"), FakeTask("r1"), FakeTask("r2")],
        }

        result = capacity_safe_cpu_allocations(env, computing_tasks)

        self.assertEqual({"v0": 1.0, "v1": 1.0, "r0": 3.0, "r1": 3.0, "r2": 3.0}, result)
        self.assertAlmostEqual(2.0, result["v0"] + result["v1"])
        self.assertAlmostEqual(9.0, result["r0"] + result["r1"] + result["r2"])

    def test_declared_concurrency_limit_omits_excess_tasks(self):
        from pi_jwm.airfogsim_contract_adapter import capacity_safe_cpu_allocations

        env = FakeEnv({"p0": 3.0})
        tasks = [FakeTask(f"t{index}") for index in range(4)]

        result = capacity_safe_cpu_allocations(env, {"p0": tasks}, max_tasks_per_node=3)

        self.assertEqual({"t0": 1.0, "t1": 1.0, "t2": 1.0}, result)
        self.assertNotIn("t3", result)

    def test_missing_or_nonpositive_capacity_produces_no_allocation(self):
        from pi_jwm.airfogsim_contract_adapter import capacity_safe_cpu_allocations

        env = FakeEnv({"zero": 0.0})
        result = capacity_safe_cpu_allocations(
            env,
            {"missing": [FakeTask("a")], "zero": [FakeTask("b")]},
        )

        self.assertEqual({}, result)


class TransmissionAccountingTests(unittest.TestCase):
    def test_activated_profiles_become_direct_events_with_timeslot_capacity(self):
        from pi_jwm.airfogsim_contract_adapter import activated_transmission_events

        manager = FakeChannelManager()
        env = type("Env", (), {"channel_manager": manager, "simulation_interval": 0.25})()
        task = FakeTransferTask("t0", "UAV_0", ["RSU_0"])

        events = activated_transmission_events(
            env,
            {
                "t0": {
                    "task": task,
                    "tx_idx": 1,
                    "rx_idx": 2,
                    "channel_type": "U2I",
                    "RB_Nos": [3, 4],
                }
            },
        )

        self.assertEqual(
            [{"source": "UAV_0", "target": "RSU_0", "planned_capacity": 1.0}],
            events,
        )
        self.assertEqual((1, 2, "U2I", [3, 4]), manager.last_rate_query)

    def test_sending_accumulates_by_source_and_receiving_by_target(self):
        from pi_jwm.airfogsim_contract_adapter import direct_transmission_totals

        sending, receiving = direct_transmission_totals(
            [
                {"source": "UAV_0", "target": "RSU_0", "planned_capacity": 2.0},
                {"source": "UAV_0", "target": "vehicle_0", "planned_capacity": 3.0},
                {"source": "vehicle_1", "target": "UAV_0", "planned_capacity": 4.0},
            ]
        )

        self.assertEqual({"UAV_0": 5.0, "vehicle_1": 4.0}, sending)
        self.assertEqual({"RSU_0": 2.0, "vehicle_0": 3.0, "UAV_0": 4.0}, receiving)

    def test_totals_are_applied_to_the_reference_channel_boundary(self):
        from pi_jwm.airfogsim_contract_adapter import apply_transmission_totals

        manager = FakeChannelManager()
        apply_transmission_totals(manager, {"UAV_0": 5.0}, {"RSU_0": 5.0})

        self.assertEqual({"UAV_0": 5.0}, manager.sending)
        self.assertEqual({"RSU_0": 5.0}, manager.receiving)

    def test_negative_or_nonfinite_transfer_amount_is_rejected(self):
        from pi_jwm.airfogsim_contract_adapter import direct_transmission_totals

        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    direct_transmission_totals(
                        [{"source": "a", "target": "b", "planned_capacity": value}]
                    )


class MissingValueEncodingTests(unittest.TestCase):
    def test_observed_zero_is_distinct_from_unmodelled(self):
        from pi_jwm.airfogsim_contract_adapter import encode_optional_value

        observed = encode_optional_value(0.0, status="direct")
        missing = encode_optional_value(None, status="not_modeled")

        self.assertEqual({"value": 0.0, "observed_mask": 1, "status": "direct"}, observed)
        self.assertEqual(
            {"value": None, "observed_mask": 0, "status": "not_modeled"},
            missing,
        )

    def test_unmodelled_value_cannot_carry_a_numeric_placeholder(self):
        from pi_jwm.airfogsim_contract_adapter import encode_optional_value

        with self.assertRaises(ValueError):
            encode_optional_value(0.0, status="not_modelled")


if __name__ == "__main__":
    unittest.main()
