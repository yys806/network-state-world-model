"""Runtime configuration helpers for using AirFogSim as a PI-JWM data source."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def resolve_airfogsim_paths(code_root: str | Path) -> tuple[Path, Path]:
    code_root = Path(code_root).resolve()
    simulator_root = code_root / "reference" / "AirFogSim"
    examples_root = simulator_root / "examples"
    if not (simulator_root / "airfogsim").is_dir() or not examples_root.is_dir():
        raise FileNotFoundError(f"AirFogSim reference simulator not found under {simulator_root}")
    return simulator_root, examples_root


def make_diagnostic_config(config: dict[str, Any], max_time: float) -> dict[str, Any]:
    result = deepcopy(config)
    result["simulation"]["max_simulation_time"] = float(max_time)
    result["traffic"]["max_n_vehicles"] = 50
    result["traffic"]["max_n_UAVs"] = 2
    result["traffic"]["RSU_positions"] = [
        [100, 100, 0],
        [700, 100, 0],
        [100, 700, 0],
        [700, 700, 0],
    ]
    result["task_profile"]["task_node_gen_poss"] = 0.8
    return result


def capture_energy_manager_snapshot(manager) -> dict[str, dict]:
    """Adapt the inspected AirFogSim EnergyManager state into a stable read-only schema."""
    try:
        active = manager._UAVs_energy_info
        removed = manager._removed_UAVs_energy_info
    except AttributeError as error:
        raise TypeError("unsupported AirFogSim EnergyManager layout") from error
    costs = {
        "fly": float(manager.getConfig("fly_unit_cost")),
        "hover": float(manager.getConfig("hover_unit_cost")),
        "sensing": float(manager.getConfig("sensing_unit_cost")),
        "receive": float(manager.getConfig("receive_unit_cost")),
        "send": float(manager.getConfig("send_unit_cost")),
    }
    pattern_fields = {
        "fly": "is_flying",
        "hover": "is_hovering",
        "sensing": "using_sensor_num",
        "receive": "receiving_data_size",
        "send": "sending_data_size",
    }
    uavs = {}
    for status, source in (("removed", removed), ("active", active)):
        for uav_id, info in source.items():
            consumption = {
                name: float(info.get(pattern_fields[name], 0.0)) * cost
                for name, cost in costs.items()
            }
            consumption["total"] = sum(consumption.values())
            uavs[uav_id] = {
                "status": status,
                "remaining_energy": float(info["energy"]),
                "last_consumption": consumption,
            }
    return {"uavs": uavs}
