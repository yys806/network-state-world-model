"""Dependency-free helpers shared by PI-JWM simulator export scripts."""


def allocate_cpu_by_assigned_node(task_dicts, get_node_info, max_tasks_per_node=3):
    if max_tasks_per_node <= 0:
        raise ValueError("max_tasks_per_node must be positive")

    grouped = {}
    node_infos = {}
    accepted = []
    for task_dict in task_dicts:
        assigned_node_id = task_dict["assigned_to"]
        if assigned_node_id not in node_infos:
            node_infos[assigned_node_id] = get_node_info(assigned_node_id)
        if node_infos[assigned_node_id] is None:
            continue
        node_tasks = grouped.setdefault(assigned_node_id, [])
        if len(node_tasks) >= max_tasks_per_node:
            continue
        node_tasks.append(task_dict)
        accepted.append(task_dict)

    allocations = {}
    for assigned_node_id, node_tasks in grouped.items():
        cpu_capacity = float(node_infos[assigned_node_id].get("fog_profile", {}).get("cpu", 0.0))
        allocated_cpu = cpu_capacity / len(node_tasks)
        for task_dict in node_tasks:
            allocations[task_dict["task_id"]] = allocated_cpu
    return allocations, accepted
