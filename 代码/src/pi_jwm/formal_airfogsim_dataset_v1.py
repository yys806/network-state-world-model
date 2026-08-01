"""Formal AirFogSim dataset protocol for PI-JWM.

The numeric scenario values are calibrated simulator inputs, not real-world
measurements. The protocol keeps complete trajectories isolated by seed.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


SCHEMA_VERSION = "PI-JWM-AirFogSim-formal-protocol-v1"
SPLIT_NAMES = ("train", "validation", "calibration", "locked_test")
CPU_POLICY_IDS = ("equal_share", "deadline_aware", "feasible_exploration")


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    load_level: str
    density_level: str
    task_lambda: float
    max_vehicles: int
    vehicle_arrival_lambda: float
    calibration_status: str = "candidate_pending_probe"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectorySpec:
    trajectory_id: str
    seed: int
    repetition: int
    split: str
    cpu_policy: str
    scenario: ScenarioSpec

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = SCHEMA_VERSION
        return value


DEFAULT_SCENARIOS = tuple(
    ScenarioSpec(
        scenario_id=f"load_{load_level}__density_{density_level}",
        load_level=load_level,
        density_level=density_level,
        task_lambda=task_lambda,
        max_vehicles=max_vehicles,
        vehicle_arrival_lambda=vehicle_arrival_lambda,
    )
    for load_level, task_lambda in (("low", 0.5), ("medium", 1.0), ("high", 2.0))
    for density_level, max_vehicles, vehicle_arrival_lambda in (
        ("sparse", 20, 0.5),
        ("dense", 40, 1.0),
    )
)


def build_formal_trajectory_specs(
    scenarios: Sequence[ScenarioSpec] = DEFAULT_SCENARIOS,
) -> list[TrajectorySpec]:
    """Build the deterministic six-scenario, ten-repetition protocol."""

    split_by_repetition = (
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "calibration",
        "locked_test",
    )
    specs: list[TrajectorySpec] = []
    for scenario_index, scenario in enumerate(scenarios):
        for repetition, split in enumerate(split_by_repetition):
            policy = CPU_POLICY_IDS[(scenario_index + repetition) % len(CPU_POLICY_IDS)]
            specs.append(
                TrajectorySpec(
                    trajectory_id=f"{scenario.scenario_id}__r{repetition:02d}",
                    seed=scenario_index * 100 + repetition,
                    repetition=repetition,
                    split=split,
                    cpu_policy=policy,
                    scenario=scenario,
                )
            )
    return specs


def validate_formal_protocol(specs: Sequence[TrajectorySpec]) -> dict[str, Any]:
    """Validate the exact formal-v1 balance and isolation contract."""

    rows = list(specs)
    checks = {
        "trajectory_count_60": len(rows) == 60,
        "scenario_count_6": len({row.scenario.scenario_id for row in rows}) == 6,
        "ten_trajectories_per_scenario": all(
            count == 10
            for count in Counter(row.scenario.scenario_id for row in rows).values()
        ),
        "split_counts_36_12_6_6": Counter(row.split for row in rows)
        == Counter({"train": 36, "validation": 12, "calibration": 6, "locked_test": 6}),
        "cpu_policy_counts_20_each": Counter(row.cpu_policy for row in rows)
        == Counter({policy: 20 for policy in CPU_POLICY_IDS}),
        "trajectory_ids_unique": len({row.trajectory_id for row in rows}) == len(rows),
        "trajectory_seeds_unique": len({row.seed for row in rows}) == len(rows),
        "split_names_valid": all(row.split in SPLIT_NAMES for row in rows),
        "cpu_policy_ids_valid": all(row.cpu_policy in CPU_POLICY_IDS for row in rows),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_valid": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


def require_split_access(split: str, *, allow_locked_test: bool = False) -> None:
    """Reject label access to the locked test split before explicit unlock."""

    normalized = str(split)
    if normalized not in SPLIT_NAMES:
        raise ValueError(f"unknown formal dataset split: {normalized}")
    if normalized == "locked_test" and not allow_locked_test:
        raise PermissionError("locked_test labels are unavailable before model freeze")


def apply_formal_scenario_overrides(
    config: dict[str, Any],
    scenario: ScenarioSpec,
) -> dict[str, Any]:
    """Apply only the declared load and density knobs to an AirFogSim config."""

    configured = copy.deepcopy(config)
    configured["traffic"]["max_n_vehicles"] = int(scenario.max_vehicles)
    configured["traffic"]["arrival_lambda"] = float(scenario.vehicle_arrival_lambda)
    configured["task"]["task_generation_kwargs"]["lambda"] = float(
        scenario.task_lambda
    )
    for node_kind in ("vehicle", "uav"):
        configured["task_profile"][node_kind]["lambda"] = float(scenario.task_lambda)
    configured["pi_jwm_formal_scenario"] = scenario.to_dict()
    return configured
