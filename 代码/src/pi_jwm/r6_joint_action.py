"""Pure contracts and deterministic candidate generation for R6 joint actions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np


R6_JOINT_ACTION_PROTOCOL_VERSION = "PIJWM-R6-joint-action-candidate-v1"
TEMPLATE_ORDER = (
    "default",
    "deadline_first",
    "priority_first",
    "load_balance",
    "rate_aware",
    "energy_conservative",
)
NONLOCKED_SPLITS = frozenset({"train", "validation", "calibration"})


def _finite_nonnegative(value: float, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


def _nonempty(value: str, *, field: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field} cannot be empty")
    return result


@dataclass(frozen=True)
class TargetContext:
    node_id: str
    distance: float
    available_cpu: float
    rate_proxy: float
    energy_proxy: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _nonempty(self.node_id, field="target node_id"))
        for field in ("distance", "available_cpu", "rate_proxy", "energy_proxy"):
            object.__setattr__(self, field, _finite_nonnegative(getattr(self, field), field=field))


@dataclass(frozen=True)
class OffloadTaskContext:
    task_id: str
    source_node_id: str
    legal_targets: tuple[TargetContext, ...]
    deadline_remaining: float
    priority: float
    input_mb: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _nonempty(self.task_id, field="offload task_id"))
        object.__setattr__(
            self, "source_node_id", _nonempty(self.source_node_id, field="source_node_id")
        )
        if not self.legal_targets:
            raise ValueError("DAG-released offload task must have a legal target")
        target_ids = [target.node_id for target in self.legal_targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("offload legal target IDs must be unique")
        for field in ("deadline_remaining", "priority", "input_mb"):
            object.__setattr__(self, field, _finite_nonnegative(getattr(self, field), field=field))


@dataclass(frozen=True)
class ComputeTaskContext:
    task_id: str
    node_id: str
    node_capacity: float
    deadline_remaining: float
    priority: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _nonempty(self.task_id, field="compute task_id"))
        object.__setattr__(self, "node_id", _nonempty(self.node_id, field="compute node_id"))
        for field in ("node_capacity", "deadline_remaining", "priority"):
            object.__setattr__(self, field, _finite_nonnegative(getattr(self, field), field=field))


@dataclass(frozen=True)
class OffloadBinding:
    task_id: str
    source_node_id: str
    target_node_id: str


@dataclass(frozen=True)
class RBAllocation:
    task_id: str
    rb_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rb_ids", tuple(int(value) for value in self.rb_ids))


@dataclass(frozen=True)
class CPUAllocation:
    node_id: str
    task_id: str
    amount: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", _finite_nonnegative(self.amount, field="CPU amount"))


@dataclass(frozen=True)
class JointActionContext:
    scenario_id: str
    seed: int
    slot: int
    split: str
    offload_tasks: tuple[OffloadTaskContext, ...]
    compute_tasks: tuple[ComputeTaskContext, ...]
    default_rb_plan: tuple[RBAllocation, ...]
    rb_capacity: int

    @classmethod
    def create(
        cls,
        *,
        scenario_id: str,
        seed: int,
        slot: int,
        split: str,
        offload_tasks: Sequence[OffloadTaskContext],
        compute_tasks: Sequence[ComputeTaskContext],
        default_rb_plan: Mapping[str, Sequence[int]],
        rb_capacity: int,
    ) -> "JointActionContext":
        split_value = str(split)
        if split_value == "locked_test":
            raise ValueError("locked_test is sealed until R9")
        if split_value not in NONLOCKED_SPLITS:
            raise ValueError(f"unsupported action split: {split_value}")
        if int(slot) < 0:
            raise ValueError("slot must be nonnegative")
        if int(rb_capacity) <= 0:
            raise ValueError("rb_capacity must be positive")
        offload = tuple(sorted(offload_tasks, key=lambda item: item.task_id))
        compute = tuple(sorted(compute_tasks, key=lambda item: item.task_id))
        if len({item.task_id for item in offload}) != len(offload):
            raise ValueError("offload task IDs must be unique")
        if len({item.task_id for item in compute}) != len(compute):
            raise ValueError("compute task IDs must be unique")
        plan = tuple(
            RBAllocation(str(task_id), tuple(int(rb) for rb in values))
            for task_id, values in sorted(default_rb_plan.items())
        )
        context = cls(
            scenario_id=_nonempty(scenario_id, field="scenario_id"),
            seed=int(seed),
            slot=int(slot),
            split=split_value,
            offload_tasks=offload,
            compute_tasks=compute,
            default_rb_plan=plan,
            rb_capacity=int(rb_capacity),
        )
        _validate_rb_plan(context, plan)
        capacities: dict[str, float] = {}
        for task in compute:
            previous = capacities.setdefault(task.node_id, task.node_capacity)
            if not math.isclose(previous, task.node_capacity, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("compute tasks on one node disagree on CPU capacity")
        return context

    def protocol_fingerprint(self) -> str:
        payload = {
            "schema_version": R6_JOINT_ACTION_PROTOCOL_VERSION,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "slot": self.slot,
            "split": self.split,
            "offload_tasks": [asdict(item) for item in self.offload_tasks],
            "compute_tasks": [asdict(item) for item in self.compute_tasks],
            "default_rb_plan": [asdict(item) for item in self.default_rb_plan],
            "rb_capacity": self.rb_capacity,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateValidation:
    valid: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class JointActionCandidate:
    DESCRIPTOR_DIM: ClassVar[int] = 18

    candidate_id: str
    template_id: str
    offload: tuple[OffloadBinding, ...]
    rb: tuple[RBAllocation, ...]
    cpu: tuple[CPUAllocation, ...]
    descriptor: tuple[float, ...]
    base_schedule_id: str = "airfogsim_default_current_slot"
    protocol_version: str = R6_JOINT_ACTION_PROTOCOL_VERSION

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        template_id: str,
        offload: Sequence[OffloadBinding],
        rb: Sequence[RBAllocation],
        cpu: Sequence[CPUAllocation],
        descriptor: Sequence[float] | None = None,
    ) -> "JointActionCandidate":
        template = str(template_id)
        if template not in TEMPLATE_ORDER:
            raise ValueError(f"unsupported joint action template: {template}")
        offload_value = tuple(sorted(offload, key=lambda item: item.task_id))
        rb_value = tuple(sorted(rb, key=lambda item: item.task_id))
        cpu_value = tuple(sorted(cpu, key=lambda item: (item.node_id, item.task_id)))
        if descriptor is None:
            descriptor = _basic_descriptor(template, offload_value, rb_value, cpu_value)
        values = tuple(float(value) for value in descriptor)
        if len(values) != cls.DESCRIPTOR_DIM or any(not math.isfinite(value) for value in values):
            raise ValueError(f"candidate descriptor must contain {cls.DESCRIPTOR_DIM} finite values")
        return cls(
            candidate_id=_nonempty(candidate_id, field="candidate_id"),
            template_id=template,
            offload=offload_value,
            rb=rb_value,
            cpu=cpu_value,
            descriptor=values,
        )


@dataclass(frozen=True)
class JointActionCandidateSet:
    candidates: tuple[JointActionCandidate, ...]

    @classmethod
    def create(cls, candidates: Sequence[JointActionCandidate]) -> "JointActionCandidateSet":
        values = tuple(candidates)
        if not values:
            raise ValueError("joint candidate set cannot be empty")
        defaults = [item for item in values if item.template_id == "default"]
        if len(defaults) != 1 or values[0].template_id != "default":
            raise ValueError("candidate set must contain exactly one default at index 0")
        ids = [item.candidate_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_id values must be unique")
        return cls(values)

    def padded_descriptors(self, *, max_candidates: int) -> tuple[np.ndarray, np.ndarray]:
        limit = int(max_candidates)
        if limit < len(self.candidates):
            raise ValueError("max_candidates cannot truncate an already frozen candidate set")
        descriptors = np.zeros((limit, JointActionCandidate.DESCRIPTOR_DIM), dtype=np.float32)
        mask = np.zeros(limit, dtype=bool)
        for index, candidate in enumerate(self.candidates):
            descriptors[index] = np.asarray(candidate.descriptor, dtype=np.float32)
            mask[index] = True
        return descriptors, mask


def _validate_rb_plan(context: JointActionContext, rows: Sequence[RBAllocation]) -> list[str]:
    reasons: list[str] = []
    expected = {row.task_id for row in context.default_rb_plan}
    observed = {row.task_id for row in rows}
    if observed != expected:
        reasons.append("RB plan tasks must equal the current default communication task set")
    all_rb = [rb for row in rows for rb in row.rb_ids]
    if any(rb < 0 or rb >= context.rb_capacity for rb in all_rb):
        reasons.append("RB index is outside current capacity")
    if len(all_rb) != len(set(all_rb)):
        reasons.append("RB IDs must be unique across tasks")
    return reasons


def validate_joint_candidate(
    context: JointActionContext,
    candidate: JointActionCandidate,
) -> CandidateValidation:
    reasons: list[str] = []
    offload_by_id = {task.task_id: task for task in context.offload_tasks}
    compute_by_id = {task.task_id: task for task in context.compute_tasks}
    if candidate.template_id == "default":
        if candidate.offload or candidate.cpu:
            reasons.append("default candidate must preserve AirFogSim offload and CPU schedules")
    else:
        if {row.task_id for row in candidate.offload} != set(offload_by_id):
            reasons.append("offload bindings must equal DAG-released task set")
        if {row.task_id for row in candidate.cpu} != set(compute_by_id):
            reasons.append("CPU bindings must equal current computing task set")
    for row in candidate.offload:
        task = offload_by_id.get(row.task_id)
        if task is None:
            reasons.append(f"offload task {row.task_id} is not DAG-released")
            continue
        if row.source_node_id != task.source_node_id:
            reasons.append(f"offload source differs for task {row.task_id}")
        legal = {target.node_id for target in task.legal_targets}
        if row.target_node_id not in legal:
            reasons.append(f"offload target is not a legal target for task {row.task_id}")
    reasons.extend(_validate_rb_plan(context, candidate.rb))
    cpu_by_node: dict[str, float] = {}
    for row in candidate.cpu:
        task = compute_by_id.get(row.task_id)
        if task is None:
            reasons.append(f"CPU task {row.task_id} is not currently computing")
            continue
        if row.node_id != task.node_id:
            reasons.append(f"CPU node differs for task {row.task_id}")
        cpu_by_node[row.node_id] = cpu_by_node.get(row.node_id, 0.0) + row.amount
    capacity_by_node = {task.node_id: task.node_capacity for task in context.compute_tasks}
    for node_id, total in cpu_by_node.items():
        if total > capacity_by_node[node_id] + 1e-7:
            reasons.append(f"CPU capacity exceeded at node {node_id}")
    return CandidateValidation(valid=not reasons, reasons=tuple(reasons))


def _template_score(template: str, task: OffloadTaskContext) -> tuple[float, str]:
    if template == "deadline_first":
        return (-1.0 / max(task.deadline_remaining, 1e-6), task.task_id)
    if template == "priority_first":
        return (-task.priority, task.task_id)
    if template == "load_balance":
        return (-task.input_mb, task.task_id)
    if template == "rate_aware":
        return (-max(target.rate_proxy for target in task.legal_targets), task.task_id)
    if template == "energy_conservative":
        return (min(target.energy_proxy for target in task.legal_targets), task.task_id)
    return (0.0, task.task_id)


def _choose_target(template: str, task: OffloadTaskContext) -> TargetContext:
    if template in {"deadline_first", "priority_first"}:
        return min(task.legal_targets, key=lambda target: (target.distance, target.node_id))
    if template == "load_balance":
        return max(task.legal_targets, key=lambda target: (target.available_cpu, -target.distance, target.node_id))
    if template == "rate_aware":
        return max(task.legal_targets, key=lambda target: (target.rate_proxy, -target.distance, target.node_id))
    if template == "energy_conservative":
        return min(task.legal_targets, key=lambda target: (target.energy_proxy, target.distance, target.node_id))
    raise ValueError(f"target selection is undefined for template {template}")


def _allocate_rb(context: JointActionContext, template: str) -> tuple[RBAllocation, ...]:
    task_by_id = {task.task_id: task for task in context.offload_tasks}
    task_ids = [row.task_id for row in context.default_rb_plan]
    task_ids.sort(
        key=lambda task_id: _template_score(template, task_by_id[task_id])
        if task_id in task_by_id
        else (0.0, task_id)
    )
    if not task_ids:
        return ()
    counts = {task_id: 0 for task_id in task_ids}
    for index in range(context.rb_capacity):
        counts[task_ids[index % len(task_ids)]] += 1
    cursor = 0
    result = []
    for task_id in task_ids:
        count = counts[task_id]
        result.append(RBAllocation(task_id, tuple(range(cursor, cursor + count))))
        cursor += count
    return tuple(sorted(result, key=lambda row: row.task_id))


def _compute_weight(template: str, task: ComputeTaskContext) -> float:
    if template == "deadline_first":
        return 1.0 / max(task.deadline_remaining, 1e-6)
    if template == "priority_first":
        return max(task.priority, 1e-6)
    if template == "load_balance":
        return 1.0
    if template == "rate_aware":
        return max(task.priority, 1.0)
    if template == "energy_conservative":
        return 1.0 / max(task.priority, 1.0)
    raise ValueError(f"CPU weighting is undefined for template {template}")


def _allocate_cpu(context: JointActionContext, template: str) -> tuple[CPUAllocation, ...]:
    by_node: dict[str, list[ComputeTaskContext]] = {}
    for task in context.compute_tasks:
        by_node.setdefault(task.node_id, []).append(task)
    rows: list[CPUAllocation] = []
    for node_id, tasks in sorted(by_node.items()):
        weights = [_compute_weight(template, task) for task in tasks]
        denominator = sum(weights)
        for task, weight in zip(tasks, weights):
            rows.append(
                CPUAllocation(
                    node_id,
                    task.task_id,
                    task.node_capacity * weight / denominator,
                )
            )
    return tuple(rows)


def _basic_descriptor(
    template: str,
    offload: Sequence[OffloadBinding],
    rb: Sequence[RBAllocation],
    cpu: Sequence[CPUAllocation],
) -> tuple[float, ...]:
    one_hot = [float(template == name) for name in TEMPLATE_ORDER]
    values = [
        float(template == "default"),
        float(len(offload)),
        float(len(rb)),
        float(sum(len(item.rb_ids) for item in rb)),
        float(len(cpu)),
        float(sum(item.amount for item in cpu)),
        *one_hot,
    ]
    values.extend([0.0] * (JointActionCandidate.DESCRIPTOR_DIM - len(values)))
    return tuple(values)


def _context_descriptor(
    context: JointActionContext,
    template: str,
    offload: Sequence[OffloadBinding],
    rb: Sequence[RBAllocation],
    cpu: Sequence[CPUAllocation],
) -> tuple[float, ...]:
    base = list(_basic_descriptor(template, offload, rb, cpu))
    if context.offload_tasks:
        base[12] = min(task.deadline_remaining for task in context.offload_tasks)
        base[13] = float(np.mean([task.priority for task in context.offload_tasks]))
        base[14] = float(sum(task.input_mb for task in context.offload_tasks))
    if context.compute_tasks:
        base[15] = min(task.deadline_remaining for task in context.compute_tasks)
        base[16] = float(np.mean([task.priority for task in context.compute_tasks]))
        base[17] = float(len({task.node_id for task in context.compute_tasks}))
    return tuple(base)


def generate_joint_candidates(
    context: JointActionContext,
    *,
    max_candidates: int = 6,
) -> JointActionCandidateSet:
    limit = int(max_candidates)
    if limit < len(TEMPLATE_ORDER):
        raise ValueError(f"max_candidates must be at least {len(TEMPLATE_ORDER)}")
    default = JointActionCandidate.create(
        candidate_id="airfogsim_default",
        template_id="default",
        offload=(),
        rb=context.default_rb_plan,
        cpu=(),
        descriptor=_context_descriptor(context, "default", (), context.default_rb_plan, ()),
    )
    candidates = [default]
    for template in TEMPLATE_ORDER[1:]:
        offload = tuple(
            OffloadBinding(task.task_id, task.source_node_id, _choose_target(template, task).node_id)
            for task in context.offload_tasks
        )
        rb = _allocate_rb(context, template)
        cpu = _allocate_cpu(context, template)
        candidate = JointActionCandidate.create(
            candidate_id=template,
            template_id=template,
            offload=offload,
            rb=rb,
            cpu=cpu,
            descriptor=_context_descriptor(context, template, offload, rb, cpu),
        )
        validation = validate_joint_candidate(context, candidate)
        if not validation.valid:
            raise ValueError(f"generated candidate {template} is invalid: {validation.reasons}")
        candidates.append(candidate)
    return JointActionCandidateSet.create(candidates[:limit])
