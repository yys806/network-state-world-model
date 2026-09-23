"""Collection-only sanitation for duplicate AirFogSim Task references."""
from __future__ import annotations

from typing import Any


TASK_COLLECTION_PRIORITY = (
    "_to_generate_task_infos", "_waiting_to_offload_tasks", "_offloading_tasks",
    "_computing_tasks", "_waiting_to_return_tasks", "_returning_tasks",
    "_done_tasks", "_out_of_ddl_tasks",
)


def repair_duplicate_task_references(task_manager: Any, frame: int) -> list[dict[str, Any]]:
    """Remove stale references to one Task object, preserving furthest lifecycle."""
    memberships: dict[str, list[tuple[int, str, str, Any]]] = {}
    for priority, collection_name in enumerate(TASK_COLLECTION_PRIORITY):
        collection = getattr(task_manager, collection_name)
        for owner, tasks in collection.items():
            for task in tasks:
                memberships.setdefault(str(task.getTaskId()), []).append((priority, collection_name, str(owner), task))
    repairs: list[dict[str, Any]] = []
    for task_id, rows in memberships.items():
        if len(rows) <= 1:
            continue
        if len({id(row[3]) for row in rows}) != 1:
            raise RuntimeError(f"distinct Task objects share task_id={task_id}")
        chosen = max(rows, key=lambda row: row[0])
        for collection_name in TASK_COLLECTION_PRIORITY:
            collection = getattr(task_manager, collection_name)
            for owner, tasks in collection.items():
                collection[owner] = [task for task in tasks if str(task.getTaskId()) != task_id]
        getattr(task_manager, chosen[1]).setdefault(chosen[2], []).append(chosen[3])
        repairs.append({
            "frame_index": frame, "task_id": task_id,
            "before": [f"{name}:{owner}" for _, name, owner, _ in rows],
            "kept": f"{chosen[1]}:{chosen[2]}",
            "reason": "remove_stale_duplicate_reference_preserve_furthest_lifecycle",
        })
    return repairs
