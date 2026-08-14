# PI-JWM v4 Full Dual-Graph Collector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify the first complete non-training PI-JWM v4 collector that records truthful physical structure, task DAGs, logical communication flows, carrying hops, three causal snapshots, legal joint actions, real AirFogSim outcomes, and an auditable multi-seed preflight bundle.

**Architecture:** Add versioned v4 modules beside the existing P2 single-step and multistep prototypes so their evidence semantics remain unchanged. The implementation is split into pure contracts, append-only vocabularies, deterministic coverage policy, AirFogSim observer/executor adapters, trajectory validation, and atomic artifact publication. AirFogSim remains an external simulator/data source; no third-party source file is modified.

**Tech Stack:** Python 3.10, `dataclasses`, `enum`, `hashlib`, `json`, `pathlib`, NumPy, `unittest`, existing PI-JWM v4 field validators, AirFogSim CPU environment.

**Execution status (2026-08-14):** Tasks 1--9 are implemented. Task 10 self-review found that the first canonical manifest hashed only eight newly added sources and therefore did not satisfy the frozen direct/transitive source, test, design, and configuration closure. The remediation expands the canonical matrix to 116 files and adds a RED-then-GREEN regression. The old bundle must be archived and regenerated after this tracked plan/design state is committed; completion is determined by the regenerated canonical manifest and fresh verifier output. Integration into `main` remains pending explicit user direction. The detailed checkboxes below preserve the original execution recipe rather than serving as the authoritative status ledger.

---

## Fixed Scope and Safety Boundary

- Work only in `D:\shen\网络组\.worktrees\v4-full-dual-graph-collector-design` on branch `codex/v4-full-dual-graph-collector-design`.
- Reuse the external AirFogSim checkout through the ignored junction at `代码/reference/AirFogSim`; do not modify or commit it.
- Do not modify the historical semantics of `single_step_collector_contract_v1.py`, `multistep_collector_contract_v1.py`, or their canonical artifacts.
- Do not run GPU jobs, training, formal baselines, or locked tests.
- Natural and fixture trajectories remain `training_eligible=false`.
- The coverage strategy identifier is exactly `balanced_two_arm_v1`.
- The environment gate is exactly `traffic_interval / simulation_interval == 1`.
- The status boundary retains `candidate_rollout_planner_complete=false` and `locked_test_accessed=false`.
- Every implementation task follows RED -> GREEN -> focused regression -> commit.

## File Map

- Create `代码/src/pi_jwm/full_dual_graph_collector_contract_v1.py`: immutable identities, snapshots, decisions, actions, rejection codes, and pure joint validation.
- Create `代码/src/pi_jwm/full_dual_graph_vocabulary_v1.py`: append-only node/physical-edge/task/DAG-edge/flow/hop vocabularies and route-revision ledger.
- Create `代码/src/pi_jwm/full_dual_graph_coverage_v1.py`: stable target-family choice and balanced RB allocation arms.
- Create `代码/src/pi_jwm/airfogsim_full_dual_graph_observer_v1.py`: direct AirFogSim snapshot extraction and execution-time hook.
- Create `代码/src/pi_jwm/airfogsim_full_dual_graph_collector_v1.py`: validate, apply setters, execute one real step, quarantine partial failures, and record outcome.
- Create `代码/src/pi_jwm/full_dual_graph_artifact_v1.py`: trajectory validation, replay comparison, status flags, manifest closure, and atomic publication.
- Create `代码/scripts/run_p2_full_dual_graph_collector_preflight_v1.py`: natural multi-seed and isolated fixture runner.
- Create corresponding test files under `代码/tests/`.
- Generate ignored evidence under `代码/artifacts/preflight/pi_jwm_p2_full_dual_graph_collector_v1/` only after all tests pass.
- Modify `本地计划表.md` and `文档/研究进展/2026-08-13-PI-JWM-v4全双图采集器设计.md` only to reflect verified implementation facts.

---

### Task 1: Complete Identity, Snapshot, and Joint-Action Contract

**Files:**

- Create: `代码/tests/test_full_dual_graph_collector_contract_v1.py`
- Create: `代码/src/pi_jwm/full_dual_graph_collector_contract_v1.py`

- [ ] **Step 1: Write failing tests for immutable identities and valid wireless/local/wired actions**

Add tests using these public types and function names:

```python
from pi_jwm.full_dual_graph_collector_contract_v1 import (
    CarryingHop,
    DagEdge,
    DecisionRow,
    JointFrameAction,
    LogicalFlow,
    PhysicalEdge,
    PhysicalNode,
    RbAllocation,
    SnapshotPhase,
    TaskLifecycle,
    TaskSnapshot,
    validate_joint_frame_action,
)


def valid_wireless_action():
    nodes = (
        PhysicalNode("uav0", "U", True),
        PhysicalNode("rsu0", "I", True),
    )
    edges = (PhysicalEdge("pe::uav0::rsu0", "uav0", "rsu0", "U2I", True),)
    task = TaskSnapshot(
        task_id="task0",
        task_node_id="uav0",
        lifecycle=TaskLifecycle.WAITING_TO_OFFLOAD,
        current_node_id="uav0",
        route_nodes=(),
        return_destination_id="uav0",
        arrival_time=0.0,
    )
    flow = LogicalFlow("flow::traj0::task0::offload::0", "traj0", "task0", "offload", 0)
    hop = CarryingHop("hop::flow0::0", flow.flow_id, 0, "uav0", "rsu0", "pe::uav0::rsu0", "wireless")
    decision = DecisionRow(
        task_id="task0",
        lifecycle=TaskLifecycle.WAITING_TO_OFFLOAD,
        selected=True,
        reason="selected",
        target_node_id="rsu0",
        route_nodes=("rsu0",),
        flow_id=flow.flow_id,
        hop_id=hop.hop_id,
        requested_target_family="nearest_remote",
        executed_target_family="nearest_remote",
        target_family_fallback=False,
    )
    action = JointFrameAction(
        frame_index=0,
        decisions=(decision,),
        flows=(flow,),
        hops=(hop,),
        rb_allocations=(RbAllocation(flow.flow_id, hop.hop_id, 0),),
    )
    return nodes, edges, (task,), action


class ContractTests(unittest.TestCase):
    def test_valid_wireless_action_passes(self):
        nodes, edges, tasks, action = valid_wireless_action()
        self.assertIs(action, validate_joint_frame_action(
            action,
            phase=SnapshotPhase.DECISION,
            nodes=nodes,
            physical_edges=edges,
            tasks=tasks,
            dag_edges=(),
            n_rb=4,
        ))

    def test_local_action_has_no_flow_hop_or_rb(self):
        nodes, _, tasks, _ = valid_wireless_action()
        decision = DecisionRow(
            task_id="task0",
            lifecycle=TaskLifecycle.WAITING_TO_OFFLOAD,
            selected=True,
            reason="selected",
            target_node_id="uav0",
            route_nodes=("uav0",),
            flow_id=None,
            hop_id=None,
            requested_target_family="local",
            executed_target_family="local",
            target_family_fallback=False,
        )
        action = JointFrameAction(0, (decision,), (), (), ())
        self.assertIs(action, validate_joint_frame_action(
            action,
            phase=SnapshotPhase.DECISION,
            nodes=nodes,
            physical_edges=(),
            tasks=tasks,
            dag_edges=(),
            n_rb=4,
        ))

    def test_wired_hop_has_cep_but_no_rb(self):
        nodes = (
            PhysicalNode("rsu0", "I", True),
            PhysicalNode("cloud0", "C", True),
        )
        edges = (PhysicalEdge("pe::rsu0::cloud0", "rsu0", "cloud0", "wired", True),)
        tasks = (TaskSnapshot(
            task_id="task0",
            task_node_id="rsu0",
            lifecycle=TaskLifecycle.WAITING_TO_OFFLOAD,
            current_node_id="rsu0",
            route_nodes=(),
            return_destination_id="rsu0",
            arrival_time=0.0,
        ),)
        flow = LogicalFlow("flow::traj0::task0::offload::0", "traj0", "task0", "offload", 0)
        hop = CarryingHop("hop::flow0::0", flow.flow_id, 0, "rsu0", "cloud0", "pe::rsu0::cloud0", "wired")
        decision = DecisionRow(
            task_id="task0",
            lifecycle=TaskLifecycle.WAITING_TO_OFFLOAD,
            selected=True,
            reason="selected",
            target_node_id="cloud0",
            route_nodes=("cloud0",),
            flow_id=flow.flow_id,
            hop_id=hop.hop_id,
            requested_target_family="capacity_remote",
            executed_target_family="capacity_remote",
            target_family_fallback=False,
        )
        action = JointFrameAction(0, (decision,), (flow,), (hop,), ())
        self.assertIs(action, validate_joint_frame_action(
            action,
            phase=SnapshotPhase.DECISION,
            nodes=nodes,
            physical_edges=edges,
            tasks=tasks,
            dag_edges=(),
            n_rb=4,
        ))
```

- [ ] **Step 2: Add failing rejection tests for every setter-pre hard gate**

Use `subTest` and immutable replacements to cover these exact rejection codes:

```python
cases = {
    "rb_out_of_range": "rb_out_of_range",
    "duplicate_coo": "duplicate_rb_allocation",
    "route_first_hop_mismatch": "route_first_hop_mismatch",
    "cep_endpoint_mismatch": "cep_endpoint_mismatch",
    "node_absent": "node_absent_at_decision",
    "offload_wrong_lifecycle": "offload_wrong_lifecycle",
    "duplicate_task_decision": "duplicate_task_decision",
    "return_destination_mismatch": "return_destination_mismatch",
    "same_transmitter_rb_conflict": "same_transmitter_rb_conflict",
    "wireless_without_rb": "wireless_flow_without_rb",
    "wired_with_rb": "nonwireless_flow_has_rb",
    "dag_edge_used_as_hop": "dag_edge_used_as_communication_hop",
    "same_slot_outcome_leak": "same_slot_outcome_leak",
    "unknown_identity": "unknown_identity",
}
```

Each case must assert `CollectorContractError.code` exactly; do not assert only message text.

- [ ] **Step 3: Run the contract tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='D:\shen\网络组\.worktrees\v4-full-dual-graph-collector-design\代码\src'
cd D:\shen\网络组\.worktrees\v4-full-dual-graph-collector-design\代码\tests
D:\miniconda\envs\airfogsim\python.exe -m unittest test_full_dual_graph_collector_contract_v1.py -v
```

Expected: import failure for `pi_jwm.full_dual_graph_collector_contract_v1`.

- [ ] **Step 4: Implement the immutable contract and pure validator**

Create the module with these public declarations:

```python
COLLECTOR_CONTRACT_VERSION = "PIJWM-P2-Full-Dual-Graph-Collector-v1"

class SnapshotPhase(str, Enum):
    DECISION = "decision"
    EXECUTION = "execution"
    OUTCOME = "outcome"

class TaskLifecycle(str, Enum):
    WAITING_TO_OFFLOAD = "waiting_to_offload"
    OFFLOADING = "offloading"
    COMPUTING = "computing"
    WAITING_TO_RETURN = "waiting_to_return"
    RETURNING = "returning"
    DONE = "done"
    FAILED = "failed"
    TO_GENERATE = "to_generate"

class CollectorContractError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code

@dataclass(frozen=True)
class PhysicalNode:
    node_id: str
    node_type: str
    present: bool

@dataclass(frozen=True)
class PhysicalEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    present: bool

@dataclass(frozen=True)
class DagEdge:
    dag_edge_id: str
    source_task_id: str
    target_task_id: str
    communication_mapping: str = "not_modeled"

@dataclass(frozen=True)
class TaskSnapshot:
    task_id: str
    task_node_id: str
    lifecycle: TaskLifecycle
    current_node_id: str
    route_nodes: tuple[str, ...]
    return_destination_id: str | None
    arrival_time: float

@dataclass(frozen=True)
class LogicalFlow:
    flow_id: str
    trajectory_id: str
    task_id: str
    phase: str
    route_revision: int

@dataclass(frozen=True)
class CarryingHop:
    hop_id: str
    flow_id: str
    hop_index: int
    source_id: str
    target_id: str
    physical_edge_id: str
    transport: str

@dataclass(frozen=True)
class RbAllocation:
    flow_id: str
    hop_id: str
    rb_index: int

@dataclass(frozen=True)
class DecisionRow:
    task_id: str
    lifecycle: TaskLifecycle
    selected: bool
    reason: str
    target_node_id: str | None
    route_nodes: tuple[str, ...]
    flow_id: str | None
    hop_id: str | None
    requested_target_family: str | None
    executed_target_family: str | None
    target_family_fallback: bool

@dataclass(frozen=True)
class JointFrameAction:
    frame_index: int
    decisions: tuple[DecisionRow, ...]
    flows: tuple[LogicalFlow, ...]
    hops: tuple[CarryingHop, ...]
    rb_allocations: tuple[RbAllocation, ...]

def validate_joint_frame_action(
    action: JointFrameAction,
    *,
    phase: SnapshotPhase,
    nodes: Sequence[PhysicalNode],
    physical_edges: Sequence[PhysicalEdge],
    tasks: Sequence[TaskSnapshot],
    dag_edges: Sequence[DagEdge],
    n_rb: int,
    input_source_phases: Mapping[str, SnapshotPhase] | None = None,
) -> JointFrameAction:
```

Implement the body without mutating any input. Normalize the arguments into ID-indexed dictionaries only after rejecting duplicate IDs. Validate in this order: frame/phase and source timing; complete actionable-task decision coverage; lifecycle/action compatibility; flow-task-phase-revision identity; route first hop and return destination; hop-flow and physical endpoint/CEP identity; DAG/communication ID-space isolation; transport-specific RB rules; RB integer/range/duplicate checks; same-transmitter RB conflicts. Return the original `action` object only after all checks pass.

- [ ] **Step 5: Run GREEN and relevant existing contract regressions**

Run:

```powershell
D:\miniconda\envs\airfogsim\python.exe -m unittest test_full_dual_graph_collector_contract_v1.py test_information_edge_contract_v4.py test_single_step_collector_contract_v1.py test_multistep_collector_contract_v1.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

```powershell
git add 代码/src/pi_jwm/full_dual_graph_collector_contract_v1.py 代码/tests/test_full_dual_graph_collector_contract_v1.py
git commit -m "feat: add full dual-graph collector contract"
```

---

### Task 2: Append-Only Full Vocabulary and Route Revision Ledger

**Files:**

- Create: `代码/tests/test_full_dual_graph_vocabulary_v1.py`
- Create: `代码/src/pi_jwm/full_dual_graph_vocabulary_v1.py`

- [ ] **Step 1: Write RED tests for six isolated vocabularies**

Test append-only indices for nodes, physical edges, tasks, DAG edges, logical flows, and carrying hops across disappearance and reappearance. Assert that a physical edge ID cannot change endpoints, a DAG edge ID cannot enter the hop vocabulary, and a hop cannot bind to a different physical edge later.

Use this expected snapshot surface:

```python
snapshot = vocabulary.observe(
    nodes=nodes,
    physical_edges=physical_edges,
    tasks=tasks,
    dag_edges=dag_edges,
    flows=flows,
    hops=hops,
)
self.assertEqual(0, snapshot.node_indices["rsu0"])
self.assertFalse(snapshot.node_presence[snapshot.node_indices["uav0"]])
self.assertEqual(0, snapshot.flow_indices["flow::traj0::task0::offload::0"])
self.assertEqual(0, snapshot.hop_indices["hop::flow0::0"])
```

- [ ] **Step 2: Write RED tests for route revisions**

Require:

```python
ledger = RouteRevisionLedger()
self.assertEqual(0, ledger.assign("traj0", "task0", "offload", ("rsu0",)))
self.assertEqual(0, ledger.assign("traj0", "task0", "offload", ("rsu0",)))
self.assertEqual(1, ledger.assign("traj0", "task0", "offload", ("uav1", "rsu0")))
self.assertEqual(0, ledger.assign("traj0", "task0", "return", ("uav0",)))
```

Reject an empty route, unsupported phase, noncontiguous imported revision, and a mutation attempted after `snapshot()` validation failure.

- [ ] **Step 3: Run RED**

```powershell
D:\miniconda\envs\airfogsim\python.exe -m unittest test_full_dual_graph_vocabulary_v1.py -v
```

Expected: module import failure.

- [ ] **Step 4: Implement transactional vocabulary observation**

Create:

```python
@dataclass(frozen=True)
class FullVocabularySnapshot:
    node_indices: dict[str, int]
    physical_edge_indices: dict[str, int]
    task_indices: dict[str, int]
    dag_edge_indices: dict[str, int]
    flow_indices: dict[str, int]
    hop_indices: dict[str, int]
    node_presence: tuple[bool, ...]
    physical_edge_presence: tuple[bool, ...]
    task_presence: tuple[bool, ...]
    dag_edge_presence: tuple[bool, ...]
    flow_presence: tuple[bool, ...]
    hop_presence: tuple[bool, ...]

class FullTrajectoryVocabulary:
    def observe(
        self,
        *,
        nodes: Sequence[PhysicalNode],
        physical_edges: Sequence[PhysicalEdge],
        tasks: Sequence[TaskSnapshot],
        dag_edges: Sequence[DagEdge],
        flows: Sequence[LogicalFlow],
        hops: Sequence[CarryingHop],
    ) -> FullVocabularySnapshot:

class RouteRevisionLedger:
    def assign(self, trajectory_id: str, task_id: str, phase: str, route: tuple[str, ...]) -> int:
```

For `observe`, build all next mappings and immutable binding dictionaries locally, reject duplicate/cross-space IDs and changed bindings, compute six presence vectors in index order, then commit the new state in one assignment block. Stable new IDs are sorted lexicographically. Never delete or reindex an identity. For `assign`, key state by `(trajectory_id, task_id, phase)`; return the existing revision for an unchanged route and increment exactly once for a changed nonempty route.

- [ ] **Step 5: Run GREEN plus historical multistep vocabulary regressions**

```powershell
D:\miniconda\envs\airfogsim\python.exe -m unittest test_full_dual_graph_vocabulary_v1.py test_multistep_collector_contract_v1.py -v
```

- [ ] **Step 6: Commit Task 2**

```powershell
git add 代码/src/pi_jwm/full_dual_graph_vocabulary_v1.py 代码/tests/test_full_dual_graph_vocabulary_v1.py
git commit -m "feat: add full collector identity vocabularies"
```

---

### Task 3: Deterministic Target and Balanced RB Coverage Policy

**Files:**

- Create: `代码/tests/test_full_dual_graph_coverage_v1.py`
- Create: `代码/src/pi_jwm/full_dual_graph_coverage_v1.py`

- [ ] **Step 1: Write RED tests for stable target-family selection**

Define `TargetCandidate(node_id, is_local, distance, available_cpu)` and test:

- the requested family is stable for the same `(trajectory_id, task_id, route_revision)`;
- a batch of three deterministic task keys covers `local`, `nearest_remote`, and `capacity_remote` once each;
- nearest remote sorts by distance, then CPU descending, then node ID;
- capacity remote sorts by CPU descending, then distance, then node ID;
- missing requested family uses fixed fallback and records it;
- dictionary/input order does not change the result.

- [ ] **Step 2: Write RED tests for both RB arms**

Use three `WirelessFlowRequest` values: two with different transmitters and one sharing the first transmitter. Assert:

```python
orthogonal = allocate_rb_coverage(requests, n_rb=6, arm="orthogonal")
self.assertEqual((), overlapping_rbs(orthogonal))

reuse = allocate_rb_coverage(requests[:2], n_rb=2, arm="interference_reuse")
self.assertTrue(set(reuse[0].rb_indices) & set(reuse[1].rb_indices))

same_tx = allocate_rb_coverage(requests, n_rb=2, arm="interference_reuse")
self.assertFalse(has_same_transmitter_rb_conflict(same_tx))
```

Also assert every unselected flow receives `rb_budget_exhausted` and no selected wireless flow has zero RBs.

- [ ] **Step 3: Run RED**

```powershell
D:\miniconda\envs\airfogsim\python.exe -m unittest test_full_dual_graph_coverage_v1.py -v
```

- [ ] **Step 4: Implement stable hashing and policy output**

Create:

```python
COVERAGE_POLICY_VERSION = "PIJWM-Balanced-Coverage-v1"

@dataclass(frozen=True)
class TargetChoice:
    requested_family: str
    executed_family: str | None
    target_node_id: str | None
    fallback: bool
    reason: str

def choose_target_family(
    *,
    trajectory_id: str,
    task_id: str,
    route_revision: int,
    candidates: Sequence[TargetCandidate],
) -> TargetChoice:
    digest = hashlib.sha256(f"{trajectory_id}\0{task_id}\0{route_revision}".encode("utf-8")).digest()
    requested = ("local", "nearest_remote", "capacity_remote")[int.from_bytes(digest[:8], "big") % 3]

def choose_resource_arm(trajectory_id: str, seed: int) -> str:

def allocate_rb_coverage(
    requests: Sequence[WirelessFlowRequest],
    *,
    n_rb: int,
    arm: str,
) -> tuple[FlowResourceDecision, ...]:
```

For target choice, partition candidates into the three families, sort with the exact keys frozen in the design, and use fallback order `local`, `nearest_remote`, `capacity_remote`. For the resource arm, SHA-256 `(trajectory_id, seed)` and map the low bit to the two arm names. For RB allocation, sort requests by their frozen task/flow key; `orthogonal` consumes unused RBs in ascending order, while `interference_reuse` restarts RB selection for a new transmitter but advances past RBs already used by that same transmitter. Emit a nonselected decision with `rb_budget_exhausted` when no legal RB remains. Do not use Python's process-randomized `hash()`.

- [ ] **Step 5: Run GREEN and repeat the deterministic tests in a fresh process**

```powershell
D:\miniconda\envs\airfogsim\python.exe -m unittest test_full_dual_graph_coverage_v1.py -v
D:\miniconda\envs\airfogsim\python.exe -m unittest test_full_dual_graph_coverage_v1.py -v
```

Expected: identical pass results both times.

- [ ] **Step 6: Commit Task 3**

```powershell
git add 代码/src/pi_jwm/full_dual_graph_coverage_v1.py 代码/tests/test_full_dual_graph_coverage_v1.py
git commit -m "feat: add balanced collector coverage policy"
```

---

### Task 4: AirFogSim Decision and Execution Snapshot Observer

**Files:**

- Create: `代码/tests/test_airfogsim_full_dual_graph_observer_v1.py`
- Create: `代码/src/pi_jwm/airfogsim_full_dual_graph_observer_v1.py`

- [ ] **Step 1: Write RED tests for direct physical graph extraction**

Build a minimal real AirFogSim environment through the existing P2 `_build_environment` helper. Assert:

- all current V/U/I nodes appear;
- all directed non-self V/U/I endpoint pairs supported by the nine channel types appear as wireless physical structure edges;
- only `wired_manager.hasLink(src, dst)` pairs appear as wired physical edges;
- `edge_present` is independent of `channel_manager.*_active_links`;
- no distance threshold is used to drop wireless structure edges;
- cloud nodes do not receive invented wireless edges.

- [ ] **Step 2: Write RED tests for task/DAG/lifecycle extraction**

Assert waiting/offloading/computing/waiting-return/returning/done/failed tasks map to the exact `TaskLifecycle` enum. Verify task DAG edges remain precedence-only and use `communication_mapping="not_modeled"`. Verify the observer reads the original return destination without changing it.

- [ ] **Step 3: Write RED test for the execution hook order and restoration**

Wrap a minimal fake environment method sequence and assert trace order:

```python
(
    "decision_snapshot_captured",
    "traffic_update_started",
    "traffic_update_finished",
    "execution_snapshot_captured",
    "task_update_started",
)
```

After the context manager exits, assert `env._updateTraffics` is the original bound method. Also reject `traffic_interval / simulation_interval != 1` before installing a hook.

- [ ] **Step 4: Run RED**

```powershell
$env:PYTHONUTF8='1'
D:\miniconda\envs\airfogsim\python.exe -m unittest test_airfogsim_full_dual_graph_observer_v1.py -v
```

- [ ] **Step 5: Implement the direct observer and temporary hook**

Expose:

```python
OBSERVER_VERSION = "PIJWM-AirFogSim-Full-Observer-v1"

@dataclass(frozen=True)
class AirFogSimSnapshot:
    phase: SnapshotPhase
    simulation_time: float
    nodes: tuple[PhysicalNode, ...]
    physical_edges: tuple[PhysicalEdge, ...]
    tasks: tuple[TaskSnapshot, ...]
    dag_edges: tuple[DagEdge, ...]
    channel_rows: tuple[dict[str, object], ...]

def observe_airfogsim_snapshot(env, *, phase: SnapshotPhase) -> AirFogSimSnapshot:

@contextmanager
def capture_execution_snapshot(
    env,
    observer: Callable[[], AirFogSimSnapshot],
) -> Iterator[list[AirFogSimSnapshot]]:
```

For `observe_airfogsim_snapshot`, enumerate node collections in stable ID order, create all supported directed non-self V/U/I wireless pairs, add only directly proven I/C wired pairs, enumerate all task-manager lifecycle collections, translate task DAGs without communication payload, and read decision-time CSI only for present wireless edges. For `capture_execution_snapshot`, require a one-element mutable capture list, save the original bound `_updateTraffics`, replace it with a wrapper that calls the original method then appends exactly one observer result, yield the list, and restore the original method in `finally`, including when `env.step()` raises.

- [ ] **Step 6: Run GREEN and P2 timing regressions**

```powershell
D:\miniconda\envs\airfogsim\python.exe -m unittest test_airfogsim_full_dual_graph_observer_v1.py test_airfogsim_single_step_collector_v1.py test_run_p2_single_step_collector_preflight_v1.py -v
```

- [ ] **Step 7: Commit Task 4**

```powershell
git add 代码/src/pi_jwm/airfogsim_full_dual_graph_observer_v1.py 代码/tests/test_airfogsim_full_dual_graph_observer_v1.py
git commit -m "feat: add AirFogSim full snapshot observer"
```

---

### Task 5: Natural Frame Builder and Complete Decision Rows

**Files:**

- Create: `代码/tests/test_airfogsim_full_dual_graph_frame_builder_v1.py`
- Create: `代码/src/pi_jwm/airfogsim_full_dual_graph_frame_builder_v1.py`

- [ ] **Step 1: Write RED tests for all actionable lifecycle states**

Use small task fixtures or real Task objects to assert:

- every waiting-to-offload task receives a selected local/remote decision or an explicit nonselection reason;
- offloading and returning tasks keep their current route and do not emit a new offload setter action;
- waiting-to-return tasks receive a route ending at the frozen return destination or `return_destination_absent`;
- computing, done, failed, and to-generate tasks produce lifecycle rows but no communication decision;
- RB shortage yields `rb_budget_exhausted` without silently removing the task;
- local and wired decisions contain no RB records.

- [ ] **Step 2: Write RED tests for flow/hop identity construction**

Assert:

```python
self.assertEqual(
    "flow::traj0::task0::offload::0",
    build_logical_flow_id("traj0", "task0", "offload", 0),
)
self.assertEqual(
    "hop::flow::traj0::task0::offload::0::0::uav0::rsu0",
    build_carrying_hop_id(flow_id, 0, "uav0", "rsu0"),
)
```

Verify an offload flow and return flow for the same task receive different IDs, and route changes increment revisions through `RouteRevisionLedger`.

- [ ] **Step 3: Run RED**

```powershell
D:\miniconda\envs\airfogsim\python.exe -m unittest test_airfogsim_full_dual_graph_frame_builder_v1.py -v
```

- [ ] **Step 4: Implement the pure frame builder**

Expose:

```python
@dataclass(frozen=True)
class BuiltFrameDecision:
    action: JointFrameAction
    lifecycle_rows: tuple[dict[str, object], ...]
    resource_policy: str

def build_frame_decision(
    snapshot: AirFogSimSnapshot,
    *,
    trajectory_id: str,
    frame_index: int,
    seed: int,
    n_rb: int,
    vocabulary: FullTrajectoryVocabulary,
    route_revisions: RouteRevisionLedger,
    node_cpu: Mapping[str, float],
    node_distance: Mapping[tuple[str, str], float],
) -> BuiltFrameDecision:
```

The body must iterate actionable tasks in the frozen stable order, call the Task 3 target policy only for waiting-to-offload tasks, construct single-hop natural routes, preserve existing offload/return routes, validate return destinations, assign route revisions, create logical flows and carrying hops only when transport is required, pass wireless requests through the RB allocator, add explicit nonselection rows for every unserved actionable task, observe the resulting identities in the vocabulary, and finish by calling `validate_joint_frame_action`.

- [ ] **Step 5: Run GREEN and pure module regression set**

```powershell
D:\miniconda\envs\airfogsim\python.exe -m unittest test_airfogsim_full_dual_graph_frame_builder_v1.py test_full_dual_graph_collector_contract_v1.py test_full_dual_graph_vocabulary_v1.py test_full_dual_graph_coverage_v1.py -v
```

- [ ] **Step 6: Commit Task 5**

```powershell
git add 代码/src/pi_jwm/airfogsim_full_dual_graph_frame_builder_v1.py 代码/tests/test_airfogsim_full_dual_graph_frame_builder_v1.py
git commit -m "feat: build complete natural collector frame actions"
```

---

### Task 6: Real One-Step Executor, Outcome Recorder, and Quarantine

**Files:**

- Create: `代码/tests/test_airfogsim_full_dual_graph_collector_v1.py`
- Create: `代码/src/pi_jwm/airfogsim_full_dual_graph_collector_v1.py`

- [ ] **Step 1: Write RED tests proving invalid actions call no setters or step**

Create spy schedulers and a spy environment. For each Task 1 rejection case assert:

```python
self.assertEqual([], task_scheduler.calls)
self.assertEqual([], communication_scheduler.calls)
self.assertEqual([], return_scheduler.calls)
self.assertEqual(0, env.step_calls)
```

- [ ] **Step 2: Write RED tests for setter ordering and execution snapshot**

Expected trace:

```python
(
    "decision_snapshot_captured",
    "action_validated",
    "cpu_callback_installed",
    "offload_setters_called",
    "return_route_setters_called",
    "rb_setters_called",
    "env_step_started",
    "traffic_update_started",
    "traffic_update_finished",
    "execution_snapshot_captured",
    "env_step_finished",
    "outcome_snapshot_captured",
)
```

Only include setter phase names that had records, but preserve their relative order.

- [ ] **Step 3: Write RED tests for partial-setter quarantine**

Make the second setter raise after the first succeeds. Assert the result has:

```python
result.quarantined is True
result.quarantine_reason == "quarantined_after_partial_setter_failure"
result.stepped is False
result.training_eligible is False
```

Assert no retry occurs on the same environment and no validated frame payload is returned.

- [ ] **Step 4: Write RED real-AirFogSim tests**

Using the existing real environment builder, execute at least:

- one natural remote offload with real wireless transmission;
- one local execution with no fake flow/RB;
- one return-route action ending at the original destination;
- two different transmitters reusing an RB in a fixture and producing direct interference/outage/rate rows;
- one runtime failure or node-disappearance fixture retained as outcome rather than rejected.

Assert actual `task.getToOffloadRoute()[0]` matches every recorded carrying hop before execution.

- [ ] **Step 5: Run RED**

```powershell
$env:PYTHONUTF8='1'
D:\miniconda\envs\airfogsim\python.exe -m unittest test_airfogsim_full_dual_graph_collector_v1.py -v
```

- [ ] **Step 6: Implement the executor and outcome ledger**

Create:

```python
@dataclass(frozen=True)
class FullCollectorStepResult:
    trajectory_id: str
    frame_index: int
    decision_snapshot: AirFogSimSnapshot
    execution_snapshot: AirFogSimSnapshot | None
    outcome_snapshot: AirFogSimSnapshot | None
    action: JointFrameAction
    lifecycle_rows: tuple[dict[str, object], ...]
    transfer_rows: tuple[dict[str, object], ...]
    cpu_rows: tuple[dict[str, object], ...]
    energy_rows: tuple[dict[str, object], ...]
    temporal_trace: tuple[str, ...]
    quarantined: bool
    quarantine_reason: str | None
    stepped: bool
    training_eligible: bool

def execute_full_collector_step(
    env,
    built: BuiltFrameDecision,
    *,
    trajectory_id: str,
    task_scheduler,
    communication_scheduler,
    computation_scheduler,
    observer: Callable[..., AirFogSimSnapshot] = observe_airfogsim_snapshot,
) -> FullCollectorStepResult:
```

The body must capture decision state, revalidate the complete action, install the existing deterministic CPU callback, apply new offload setters, then return-route setters, then grouped task RB setters, and run exactly one real `env.step()` inside the execution-snapshot hook. Track how many setters succeeded; if a later setter raises, return a quarantined result without stepping or retrying. After a successful step, capture outcome state and build direct transfer, per-RB, CPU, and repaired energy ledgers. Group RB allocations by task only after proving every allocation maps to that task's current carrying hop. Never infer per-RB fields from aggregate rate.

- [ ] **Step 7: Run GREEN plus all P2 real-step regressions**

```powershell
D:\miniconda\envs\airfogsim\python.exe -m unittest test_airfogsim_full_dual_graph_collector_v1.py test_airfogsim_single_step_collector_v1.py test_run_p2_single_step_collector_preflight_v1.py test_run_p2_multistep_collector_preflight_v1.py -v
```

- [ ] **Step 8: Commit Task 6**

```powershell
git add 代码/src/pi_jwm/airfogsim_full_dual_graph_collector_v1.py 代码/tests/test_airfogsim_full_dual_graph_collector_v1.py
git commit -m "feat: execute and record full AirFogSim collector steps"
```

---

### Task 7: Trajectory Validation, E1 History, and Replay Contract

**Files:**

- Create: `代码/tests/test_full_dual_graph_artifact_v1.py`
- Create: `代码/src/pi_jwm/full_dual_graph_artifact_v1.py`

- [ ] **Step 1: Write RED tests for frame and trajectory validation**

Require validation of:

- contiguous frame indices;
- exactly one decision row for every actionable task;
- decision/execution/outcome phase labels;
- no same-frame outcome source in decision inputs;
- CEP endpoint identity;
- full physical-edge presence vectors matching vocabulary width;
- natural and fixture rows never mixed;
- quarantined frames excluded from validated trajectory frames;
- E1 first-frame `NO_HISTORY`, positive history, and valid observed zero.

- [ ] **Step 2: Write RED tests for replay comparison**

Require exact equality for identity dictionaries and action ledgers. Compare floats with named tolerances:

```python
REPLAY_ABS_TOL = 1e-9
REPLAY_REL_TOL = 1e-7
```

The report must list every non-bitwise float difference rather than silently rounding it away.

- [ ] **Step 3: Write RED tests for status flags and source closure**

Assert the passing collector flag set is exactly:

```python
{
    "v4_collector_implemented": True,
    "v4_dataset_complete": False,
    "training_eligible": False,
    "model_training_started": False,
    "gpu_started": False,
    "locked_test_accessed": False,
    "candidate_rollout_planner_complete": False,
    "final_method_frozen": False,
}
```

Test that a missing direct or transitive source file fails manifest construction.

- [ ] **Step 4: Run RED**

```powershell
D:\miniconda\envs\airfogsim\python.exe -m unittest test_full_dual_graph_artifact_v1.py -v
```

- [ ] **Step 5: Implement validators, replay comparison, and atomic publisher**

Expose:

```python
REQUIRED_ARTIFACT_FILES = (
    "collector_config.json",
    "vocabularies.json",
    "frames.jsonl",
    "coverage_report.json",
    "validation_report.json",
    "replay_report.json",
    "status_flags.json",
    "manifest.json",
)

def validate_trajectory_frames(
    frames: Sequence[Mapping[str, object]],
    *,
    vocabulary: Mapping[str, object],
    fixture: bool,
) -> list[str]:

def compare_replays(
    reference: Sequence[Mapping[str, object]],
    replay: Sequence[Mapping[str, object]],
    *,
    abs_tol: float = REPLAY_ABS_TOL,
    rel_tol: float = REPLAY_REL_TOL,
) -> dict[str, object]:

def build_full_collector_status_flags(*, passed: bool) -> dict[str, bool]:

def publish_atomic_bundle(
    output_dir: Path,
    payloads: Mapping[str, object],
    source_paths: Sequence[Path],
) -> None:
```

`validate_trajectory_frames` returns an ordered list of exact errors and never mutates frames. `compare_replays` recursively compares identity/action data exactly and floats with the frozen tolerances, returning `passed`, exact mismatches, and numeric-difference rows. `build_full_collector_status_flags` may set only `v4_collector_implemented` from `passed`; every other flag remains false. `publish_atomic_bundle` writes UTF-8 sorted JSON with newline termination in a sibling temporary directory, writes JSONL one frame per line, validates required payload names and source existence, computes hashes, writes the manifest last, and renames only after verification. Refuse to overwrite an existing canonical directory.

- [ ] **Step 6: Run GREEN plus P1/P2 artifact regressions**

```powershell
D:\miniconda\envs\airfogsim\python.exe -m unittest test_full_dual_graph_artifact_v1.py test_run_p2_single_step_collector_preflight_v1.py test_run_p2_multistep_collector_preflight_v1.py -v
```

- [ ] **Step 7: Commit Task 7**

```powershell
git add 代码/src/pi_jwm/full_dual_graph_artifact_v1.py 代码/tests/test_full_dual_graph_artifact_v1.py
git commit -m "feat: validate and publish full collector artifacts"
```

---

### Task 8: Natural Multi-Seed and Isolated Coverage-Fixture Runner

**Files:**

- Create: `代码/tests/test_run_p2_full_dual_graph_collector_preflight_v1.py`
- Create: `代码/scripts/run_p2_full_dual_graph_collector_preflight_v1.py`

- [ ] **Step 1: Write RED tests for configuration and safety gates**

Test that the runner:

- accepts only seeds `(0, 1, 2)` for the canonical preflight;
- runs both `orthogonal` and `interference_reuse` natural arms;
- requires at least 20 real steps or a naturally ended environment;
- rejects `traffic_interval / simulation_interval != 1`;
- has no CLI option for GPU, training, locked test, or formal dataset generation;
- separates natural episodes and fixtures in every report;
- refuses to publish if any required coverage fixture failed.

- [ ] **Step 2: Write RED tests for required fixture matrix**

The fixture report must contain these exact keys and each row must state `fixture=true` and `training_eligible=false`:

```python
REQUIRED_FIXTURES = (
    "multi_task_multi_flow",
    "cross_transmitter_rb_reuse",
    "wired_flow",
    "local_execution",
    "multihop_offload",
    "multihop_return",
    "node_disappearance_reappearance",
    "route_interruption",
    "deadline_failure",
    "tti_failure",
)
```

- [ ] **Step 3: Write RED verify-only and tamper tests**

Generate a temporary bundle through fixture payloads, verify it, mutate one frame and one source hash independently, and assert both tamper cases fail without changing the canonical directory.

- [ ] **Step 4: Run RED**

```powershell
$env:PYTHONUTF8='1'
D:\miniconda\envs\airfogsim\python.exe -m unittest test_run_p2_full_dual_graph_collector_preflight_v1.py -v
```

- [ ] **Step 5: Implement the runner**

Provide CLI:

```text
--output-dir PATH
--verify-only
--seeds 0 1 2
--steps 20
```

Implementation rules:

- Build each natural episode from a fresh AirFogSim environment.
- Derive the arm deterministically from `(trajectory_id, seed)` but ensure the canonical six-episode matrix contains both arms for every seed.
- Execute all actual steps through `execute_full_collector_step`.
- Build fixtures through real AirFogSim configs/objects/steps, never by writing expected outcomes directly.
- Run a second fresh replay for every natural episode before publication.
- Publish only after trajectory, coverage, replay, and source-closure validation pass.
- Keep all status flags conservative except `v4_collector_implemented=true` after the complete canonical bundle passes.

- [ ] **Step 6: Run GREEN in temporary directories only**

```powershell
D:\miniconda\envs\airfogsim\python.exe -m unittest test_run_p2_full_dual_graph_collector_preflight_v1.py -v
```

- [ ] **Step 7: Commit Task 8**

```powershell
git add 代码/scripts/run_p2_full_dual_graph_collector_preflight_v1.py 代码/tests/test_run_p2_full_dual_graph_collector_preflight_v1.py
git commit -m "feat: add full collector nontraining preflight runner"
```

---

### Task 9: Canonical CPU Preflight, Manifest Verification, and Truthful Documentation

**Files:**

- Generate: `代码/artifacts/preflight/pi_jwm_p2_full_dual_graph_collector_v1/`
- Modify: `本地计划表.md`
- Modify: `文档/研究进展/2026-08-13-PI-JWM-v4全双图采集器设计.md`
- Modify externally after repository verification: `D:\禹尧珅\人工智能知识库\北大科研\PIJWM\8.12之后推进.md`
- Modify externally only if a fixed statement is contradicted: `D:\禹尧珅\人工智能知识库\北大科研\PIJWM\PIJWM主文档.md`

- [ ] **Step 1: Run the full focused regression suite before generating canonical evidence**

```powershell
$env:PYTHONPATH='D:\shen\网络组\.worktrees\v4-full-dual-graph-collector-design\代码\src'
$env:PYTHONUTF8='1'
cd D:\shen\网络组\.worktrees\v4-full-dual-graph-collector-design\代码\tests
D:\miniconda\envs\airfogsim\python.exe -m unittest `
  test_full_dual_graph_collector_contract_v1.py `
  test_full_dual_graph_vocabulary_v1.py `
  test_full_dual_graph_coverage_v1.py `
  test_airfogsim_full_dual_graph_observer_v1.py `
  test_airfogsim_full_dual_graph_frame_builder_v1.py `
  test_airfogsim_full_dual_graph_collector_v1.py `
  test_full_dual_graph_artifact_v1.py `
  test_run_p2_full_dual_graph_collector_preflight_v1.py `
  test_information_edge_contract_v4.py `
  test_single_step_collector_contract_v1.py `
  test_airfogsim_single_step_collector_v1.py `
  test_run_p2_single_step_collector_preflight_v1.py `
  test_multistep_collector_contract_v1.py `
  test_run_p2_multistep_collector_preflight_v1.py `
  small_experiments/test_airfogsim_strict_dual_graph_preflight.py -v
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Generate the canonical non-training bundle once**

```powershell
cd D:\shen\网络组\.worktrees\v4-full-dual-graph-collector-design\代码
D:\miniconda\envs\airfogsim\python.exe scripts\run_p2_full_dual_graph_collector_preflight_v1.py --seeds 0 1 2 --steps 20
```

Expected: canonical directory created atomically; runner prints `v4_collector_implemented=true`, `v4_dataset_complete=false`, and `training_eligible=false`.

- [ ] **Step 3: Verify the canonical bundle independently**

```powershell
D:\miniconda\envs\airfogsim\python.exe scripts\run_p2_full_dual_graph_collector_preflight_v1.py --verify-only
```

Then independently recompute every manifest SHA-256 with a short read-only PowerShell loop and require zero mismatches. Do not regenerate on verification failure; diagnose the exact mismatch.

- [ ] **Step 4: Audit coverage and evidence boundaries**

Read the machine-readable reports and verify:

- 3 seeds x 2 natural arms are present;
- each natural episode has at least 20 real steps or a recorded natural termination;
- all ten fixtures are present and excluded from natural/training candidates;
- every actionable task has a decision row;
- illegal accepted actions, CEP mismatches, partial-setter training candidates, RB range violations, and same-transmitter conflicts are all zero;
- no field claims the old 13 missing dimensions were filled;
- no planner/training/final-method flag is true.

- [ ] **Step 5: Update repository and knowledge-base documentation with verified facts only**

In `本地计划表.md`, change P2 from “formal collector not implemented” to “v4 full dual-graph non-training collector preflight verified” only if Step 3 passes. Keep dataset, training, planner, GPU, locked-test, and final-method statuses false.

In the design document, add a short implementation-evidence section with the canonical artifact path, exact test count, commit IDs, and limitations.

In `8.12之后推进.md`, record the same facts and next serial gate: formal dataset scale/distribution audit. Do not describe fixture coverage as natural-data coverage.

- [ ] **Step 6: Run documentation truthfulness searches**

```powershell
rg -n "正式数据已完成|训练可用=true|规划器已完成|最终方法|18维全部有效|13维已补齐" 本地计划表.md 文档/研究进展/2026-08-13-PI-JWM-v4全双图采集器设计.md
```

Treat exit code 1 as “no forbidden wording found”. Review any match manually rather than replacing it mechanically.

- [ ] **Step 7: Commit Task 9**

The generated artifact is ignored and remains machine evidence on disk. Commit only code-tracked documentation changes:

```powershell
git add 本地计划表.md 文档/研究进展/2026-08-13-PI-JWM-v4全双图采集器设计.md
git commit -m "docs: record verified full collector preflight"
```

---

### Task 10: Final Verification and Integration Audit

**Files:**

- No new production files.
- Inspect every file and commit created by Tasks 1-9.

- [ ] **Step 1: Run syntax compilation for all new Python modules and runner**

```powershell
cd D:\shen\网络组\.worktrees\v4-full-dual-graph-collector-design\代码
D:\miniconda\envs\airfogsim\python.exe -m py_compile `
  src\pi_jwm\full_dual_graph_collector_contract_v1.py `
  src\pi_jwm\full_dual_graph_vocabulary_v1.py `
  src\pi_jwm\full_dual_graph_coverage_v1.py `
  src\pi_jwm\airfogsim_full_dual_graph_observer_v1.py `
  src\pi_jwm\airfogsim_full_dual_graph_frame_builder_v1.py `
  src\pi_jwm\airfogsim_full_dual_graph_collector_v1.py `
  src\pi_jwm\full_dual_graph_artifact_v1.py `
  scripts\run_p2_full_dual_graph_collector_preflight_v1.py
```

- [ ] **Step 2: Re-run the Task 9 focused regression command**

Expected: zero failures/errors. Record the exact number of tests from fresh output; do not reuse an earlier count.

- [ ] **Step 3: Verify clean tracked branch state and source closure**

```powershell
git status --short
git log --oneline --decorate -12
git diff main...HEAD --check
```

Expected: no tracked modifications; ignored AirFogSim junction and artifact do not appear. Confirm every runtime import from new modules points to a tracked file or the explicitly external AirFogSim checkout.

- [ ] **Step 4: Perform a theory-code-data-evidence matrix review**

For each design claim, identify:

- implementing module/function;
- direct test name;
- artifact/report field;
- current status flag.

Any claim without all four mappings remains `target definition` or `not implemented`; do not mark the collector complete until critical rows are closed.

- [ ] **Step 5: Request code review before integration**

Use the requesting-code-review workflow on the complete branch. Address only verified findings, rerun affected tests, and make a separate fix commit for each coherent issue.

- [ ] **Step 6: Present integration options without modifying main automatically**

Report branch path, commits, tests, artifact location, remaining limitations, and the fact that no GPU/locked test/training occurred. Do not merge, push, or delete worktrees without explicit user direction.

---

## Completion Definition

This plan is complete only when:

- all Task 1-10 checkboxes are satisfied;
- the focused fresh regression suite passes;
- the canonical bundle verifies with zero hash mismatches;
- the six natural episodes and ten non-training fixtures are separated and validated;
- the theory-code-data-evidence matrix has no unresolved critical collector mismatch;
- only `v4_collector_implemented=true` is upgraded;
- GPU, locked test, training, formal dataset, candidate-rollout planner, and final-method flags remain false.
