# PI-JWM P2 Joint-Action Attempt/Reject Ledger v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a machine-verifiable joint-action attempt ledger to a version-isolated P2-B v2 CPU preflight, then make P2-C v2 compute the natural-reference rejection rate from that ledger instead of an unobserved summary field.

**Architecture:** Preserve all P2-B v1 code and artifacts byte-for-byte. A pure PI-JWM ledger state machine owns attempt identity, transition validation, setter-call detail, terminal disposition, mutation/quarantine semantics, and role-separated summaries; a v2 adapter observes the existing v1 executor through instance-scoped scheduler, `env.step`, and observer wrappers that are restored in `finally`. A separate v2 artifact contract publishes either a nine-file success bundle or a three-file failure bundle atomically, and P2-C v2 independently reloads and validates `action_attempts.jsonl` before removing only the rejection-observation blocker.

**Tech Stack:** Python 3 standard library (`dataclasses`, `enum`, `hashlib`, `json`, `pathlib`, `tempfile`, `unittest`), existing PI-JWM P2 contracts, real AirFogSim only at the CPU boundary, Git worktree isolation.

**Execution choice:** Inline execution is frozen by the user. Use `superpowers:executing-plans`; do not dispatch subagents, start GPU work, access locked test, modify AirFogSim, generate formal trajectories, or overwrite existing artifacts.

---

## File responsibility map

- Create `代码/src/pi_jwm/action_attempt_ledger_v1.py`: pure identities, canonical digests, transition state machine, terminal-record validation, JSONL loading, role-separated summaries, and frame/attempt/replay alignment checks.
- Create `代码/src/pi_jwm/airfogsim_full_dual_graph_collector_v2.py`: instance-scoped runtime observation around the existing v1 executor; it does not duplicate the executor or modify v1 globals.
- Create `代码/src/pi_jwm/full_dual_graph_artifact_v2.py`: v2 success/failure file matrices, semantic validation, immutable atomic publication, and independent artifact/source verification.
- Create `代码/scripts/run_p2_full_dual_graph_collector_preflight_v2.py`: natural-reference, natural-replay, bootstrap, and fixture orchestration with one attempt per submitted joint action and no retry.
- Create `代码/src/pi_jwm/p2c_scale_distribution_audit_v2.py`: read-only P2-C v2 audit that recomputes rejection and quarantine facts from ledger rows.
- Create `代码/scripts/run_p2c_scale_distribution_audit_v2.py`: atomic P2-C v2 report publication and source/input closure verification.
- Create six matching test modules under `代码/tests/`.
- Modify `本地计划表.md`: record the actual gate state without promoting formal data or training.
- Create `文档/研究进展/2026-08-14-PI-JWM-P2联合动作Attempt-Reject-Ledger-v1实施与证据.md`: bind claims to commands, code paths, artifacts, and remaining blockers after fresh verification.

### Task 1: Pure attempt ledger state machine

**Files:**
- Create: `代码/tests/test_action_attempt_ledger_v1.py`
- Create: `代码/src/pi_jwm/action_attempt_ledger_v1.py`

- [ ] **Step 1: Write failing identity, transition, and terminal-validation tests**

Add focused `unittest` cases that import the wished-for API and assert these exact behaviors:

```python
from pi_jwm.action_attempt_ledger_v1 import (
    ActionAttemptLedger,
    AttemptIdentity,
    LedgerContractError,
    candidate_digest,
    summarize_attempts,
    validate_attempt_records,
)

identity = AttemptIdentity(
    run_role="natural_reference",
    episode_id="natural-seed-0-orthogonal",
    trajectory_id="natural-seed-0-orthogonal-trajectory",
    frame_index=0,
    candidate_ordinal=0,
)
ledger = ActionAttemptLedger()
attempt = ledger.begin(identity)
assert attempt.attempt_id == ledger.begin_id(identity)
assert attempt.attempt_id != ledger.begin_id(
    dataclasses.replace(identity, run_role="natural_replay")
)
```

Cover duplicate ID, duplicate terminal, non-contiguous transition, boolean-as-index rejection, empty identity, accepted-without-step, rejected-without-reason, mutation/quarantine contradictions, non-`natural_reference` ordinal handling, current natural-reference `candidate_ordinal != 0`, canonical candidate digest stability for dataclass/enum/plain containers, and per-role binary conservation.

- [ ] **Step 2: Run the ledger test and confirm RED**

Run:

```powershell
cd D:\shen\网络组\.worktrees\p2-action-attempt-ledger-v1\代码\tests
python -m unittest -v test_action_attempt_ledger_v1.py
```

Expected: import failure for `pi_jwm.action_attempt_ledger_v1`; the failure must be caused by the missing production module.

- [ ] **Step 3: Implement the minimal ledger API**

Implement these public types and functions:

```python
LEDGER_SCHEMA_VERSION = "PIJWM-Action-Attempt-Ledger-v1"
RUN_ROLES = ("natural_reference", "natural_replay", "fixture", "bootstrap")
MUTATION_STATES = ("none", "confirmed", "unknown_after_runtime_call")
SUCCESS_PATH = (
    "begun", "candidate_built", "contract_validated",
    "pre_setter_revalidated", "setters_applied",
    "env_step_started", "env_step_completed", "outcome_captured",
)

class LedgerContractError(ValueError): ...

@dataclass(frozen=True)
class AttemptIdentity:
    run_role: str
    episode_id: str
    trajectory_id: str
    frame_index: int
    candidate_ordinal: int

class ActionAttemptLedger:
    def begin(self, identity: AttemptIdentity) -> AttemptHandle: ...
    def begin_id(self, identity: AttemptIdentity) -> str: ...
    def terminal_records(self) -> list[dict[str, object]]: ...

def candidate_digest(value: object) -> str: ...
def validate_attempt_records(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]: ...
def summarize_attempts(rows: Sequence[Mapping[str, object]]) -> dict[str, object]: ...
```

`AttemptHandle` exposes only legal event methods: `candidate_built`, `contract_validated`, `pre_setter_revalidated`, `setter_started`, `setter_completed`, `setters_applied`, `env_step_started`, `env_step_completed`, `outcome_captured`, `accept`, and `reject`. `accept()` emits `terminal_stage="outcome_captured"`, no rejection fields, `environment_mutation_status="confirmed"`, `quarantined=false`, and `training_eligible=false`. `reject()` requires a stable rejection code/detail, derives no counts from caller-supplied summaries, and enforces quarantine whenever mutation is not `none`.

- [ ] **Step 4: Run the ledger test and confirm GREEN**

Run the command from Step 2. Expected: all ledger tests pass with zero errors.

- [ ] **Step 5: Run the nearest contract regression**

Run:

```powershell
python -m unittest -v test_full_dual_graph_collector_contract_v1.py test_full_dual_graph_artifact_v1.py
```

Expected: all existing v1 contract/artifact tests pass unchanged.

- [ ] **Step 6: Commit the ledger slice**

```powershell
git add -- 代码/src/pi_jwm/action_attempt_ledger_v1.py 代码/tests/test_action_attempt_ledger_v1.py
git commit -m "feat: add joint action attempt ledger"
```

### Task 2: Instance-scoped v2 runtime adapter

**Files:**
- Create: `代码/tests/test_airfogsim_full_dual_graph_collector_v2.py`
- Create: `代码/src/pi_jwm/airfogsim_full_dual_graph_collector_v2.py`
- Read only: `代码/src/pi_jwm/airfogsim_full_dual_graph_collector_v1.py`

- [ ] **Step 1: Write failing runtime-boundary tests**

Reuse the existing `FakeEnv`, `SpySchedulers`, action builder, and phase observer from `test_airfogsim_full_dual_graph_collector_v1.py` without changing v1 assertions. Test the API:

```python
result = execute_full_collector_step_v2(
    env,
    built,
    attempt=attempt,
    trajectory_id=identity.trajectory_id,
    task_scheduler=schedulers,
    communication_scheduler=schedulers,
    computation_scheduler=schedulers,
    observer=observer,
)
```

Required cases: pre-setter RB/contract failure has no setter and no step; first setter exception yields `unknown_after_runtime_call` plus quarantine; a completed setter followed by setter failure yields `confirmed` plus quarantine; `env.step` exception records called-but-not-completed; outcome observer exception records completed step and quarantine; accepted action records every setter in order, calls step exactly once, and restores scheduler methods, `env.step`, and observer behavior after return or exception.

- [ ] **Step 2: Run the v2 adapter test and confirm RED**

```powershell
python -m unittest -v test_airfogsim_full_dual_graph_collector_v2.py
```

Expected: import failure for the missing v2 adapter.

- [ ] **Step 3: Implement scheduler proxies and boundary wrappers**

Implement a small proxy instead of module-global monkeypatching:

```python
class _SchedulerProxy:
    def __init__(self, target: object, attempt: AttemptHandle): ...
    def setComputingCallBack(self, env, callback): ...
    def setTaskOffloading(self, env, task_node_id, task_id, target_node_id, route=None): ...
    def setTaskReturnRoute(self, env, task_id, route): ...
    def setCommunicationWithRB(self, env, task_id, rb_nos): ...

def execute_full_collector_step_v2(
    env,
    built: BuiltFrameDecision,
    *,
    attempt: AttemptHandle,
    trajectory_id: str,
    task_scheduler,
    communication_scheduler,
    computation_scheduler,
    observer=observe_airfogsim_snapshot,
) -> FullCollectorStepResult: ...
```

The adapter calls the real `execute_full_collector_step` exactly once. Reaching the first setter or wrapped `env.step` proves the executor's second validation returned; the wrapper then records `pre_setter_revalidated`. Immediately before delegating the real `env.step`, it records `setters_applied` and `env_step_started`; it records completion only after the real step returns. Observer phase calls distinguish decision, execution, and outcome capture. In a `finally`, restore every replaced instance attribute even if the v1 executor returns a quarantined result or raises.

- [ ] **Step 4: Run v2 and v1 executor tests and confirm GREEN**

```powershell
python -m unittest -v test_airfogsim_full_dual_graph_collector_v2.py test_airfogsim_full_dual_graph_collector_v1.py
```

Expected: all tests pass; v1 remains unchanged.

- [ ] **Step 5: Commit the adapter slice**

```powershell
git add -- 代码/src/pi_jwm/airfogsim_full_dual_graph_collector_v2.py 代码/tests/test_airfogsim_full_dual_graph_collector_v2.py
git commit -m "feat: observe v2 collector runtime attempts"
```

### Task 3: V2 success/failure artifact contract

**Files:**
- Create: `代码/tests/test_full_dual_graph_artifact_v2.py`
- Create: `代码/src/pi_jwm/full_dual_graph_artifact_v2.py`

- [ ] **Step 1: Write failing semantic and publication tests**

Test exact constants and APIs:

```python
ARTIFACT_CONTRACT_VERSION = "PIJWM-Full-Dual-Graph-Artifact-v2"
SUCCESS_REQUIRED_FILES = (
    "collector_config.json", "vocabularies.json", "frames.jsonl",
    "action_attempts.jsonl", "coverage_report.json", "validation_report.json",
    "replay_report.json", "status_flags.json", "manifest.json",
)
FAILURE_REQUIRED_FILES = (
    "action_attempts.jsonl", "failure_report.json", "manifest.json",
)
```

Cover accepted-reference/frame one-to-one mapping by trajectory/frame/candidate digest, rejected-with-frame rejection, replay digest/state alignment, fixture/bootstrap exclusion from the main denominator, no hidden ordinal/retry, success nine-file publication, failure three-file publication, refusal when either destination exists, tampering, portable source keys, temporary-directory cleanup, and fixed false status flags.

- [ ] **Step 2: Run artifact tests and confirm RED**

```powershell
python -m unittest -v test_full_dual_graph_artifact_v2.py
```

Expected: import failure for the missing artifact v2 module.

- [ ] **Step 3: Implement v2 validators and atomic publishers**

Implement:

```python
def validate_success_payloads(payloads: Mapping[str, object]) -> tuple[str, ...]: ...
def validate_bundle_alignment(
    frames: Sequence[Mapping[str, object]],
    attempts: Sequence[Mapping[str, object]],
) -> tuple[str, ...]: ...
def assert_publish_targets_absent(output_dir: Path) -> None: ...
def publish_success_bundle(output_dir: Path, payloads, source_paths) -> None: ...
def publish_failure_bundle(output_dir: Path, attempts, failure_report, source_paths) -> Path: ...
def verify_success_bundle(output_dir: Path, source_paths) -> dict[str, object]: ...
def verify_failure_bundle(failed_dir: Path, source_paths) -> dict[str, object]: ...
```

Write all JSON/JSONL into a sibling temporary directory, calculate hashes from the written bytes, write the manifest, recalculate every hash, and use `os.replace` only after semantic and hash checks pass. Never append to or backfill a v1 bundle.

- [ ] **Step 4: Run v2 artifact and v1 artifact tests and confirm GREEN**

```powershell
python -m unittest -v test_full_dual_graph_artifact_v2.py test_full_dual_graph_artifact_v1.py
```

Expected: all tests pass and v1 file-count semantics remain eight files.

- [ ] **Step 5: Commit the artifact slice**

```powershell
git add -- 代码/src/pi_jwm/full_dual_graph_artifact_v2.py 代码/tests/test_full_dual_graph_artifact_v2.py
git commit -m "feat: publish ledger-bound P2 artifacts"
```

### Task 4: CPU-only P2-B v2 runner and attempt roles

**Files:**
- Create: `代码/tests/test_run_p2_full_dual_graph_collector_preflight_v2.py`
- Create: `代码/scripts/run_p2_full_dual_graph_collector_preflight_v2.py`
- Read only: `代码/scripts/run_p2_full_dual_graph_collector_preflight_v1.py`

- [ ] **Step 1: Write failing request, role, and failure-path tests**

Test that the CLI exposes only `--seeds`, `--steps`, `--output-dir`, and `--verify-only`; canonical request remains seeds `(0, 1, 2)`, both resource arms, 20 steps, traffic/simulation interval 0.1, and CPU-only scope. Use injected builders/executors only at test seams to prove: an attempt begins after decision observation and before candidate build; builder contract errors terminalize at `contract_validation`; other builder errors terminalize at `candidate_build` with null digest; natural reference and replay use distinct IDs but matching digests; each fixture's first real collector step is `bootstrap` and controlled boundary step is `fixture`; a rejection stops the run with no retry or next candidate; collision checks happen before environment construction.

- [ ] **Step 2: Run the v2 runner test and confirm RED**

```powershell
python -m unittest -v test_run_p2_full_dual_graph_collector_preflight_v2.py
```

Expected: import failure for the missing runner.

- [ ] **Step 3: Implement attempt-aware episode execution**

Use v1 pure helpers and contracts, but create a new loop that owns the ledger:

```python
def execute_attempt_frame(
    env,
    schedulers,
    *,
    identity: AttemptIdentity,
    seed: int,
    vocabulary: FullTrajectoryVocabulary,
    route_revisions: RouteRevisionLedger,
) -> tuple[BuiltFrameDecision, FullCollectorStepResult]: ...

def run_natural_episode_v2(spec, *, steps: int, run_role: str) -> dict[str, object]: ...
def run_natural_replay_pair_v2(spec, *, steps: int) -> dict[str, object]: ...
def run_real_fixture_v2(name: str, *, seed: int) -> dict[str, object]: ...
def build_real_preflight_payloads_v2(seeds=CANONICAL_SEEDS, *, steps=20) -> dict[str, object]: ...
```

`execute_attempt_frame` captures the decision snapshot first, calls `ledger.begin`, calls the real v1 builder once, records the real candidate digest and first validation success only after the builder returns, then calls the v2 adapter once. It raises a typed run failure only after the attempt is terminal so the top-level can publish evidence.

- [ ] **Step 4: Implement the real fixture matrix without changing AirFogSim**

Reuse v1 environment/task setup helpers and reproduce only the fixture orchestration boundary needed to call `execute_attempt_frame`. Record the initial real step as `bootstrap` and the controlled step as `fixture`; do not count either in natural rejection rate. Restore transient fixture wrappers in `finally` and stop the complete run at the first rejected attempt.

- [ ] **Step 5: Implement top-level success/failure publication**

At process start call `assert_publish_targets_absent(args.output_dir)`. On success, include the terminal rows as `action_attempts.jsonl`, compute validation summaries from those rows, publish the nine-file bundle, and immediately verify it. On a catchable run-level or attempt failure, publish `<output-dir>_failed` with the actual terminal ledger rows and conservative failure fields; do not create the success directory or continue collecting.

- [ ] **Step 6: Run runner, adapter, and artifact tests and confirm GREEN**

```powershell
python -m unittest -v test_run_p2_full_dual_graph_collector_preflight_v2.py test_airfogsim_full_dual_graph_collector_v2.py test_full_dual_graph_artifact_v2.py
```

Expected: all tests pass with no GPU or locked-test access.

- [ ] **Step 7: Commit the runner slice**

```powershell
git add -- 代码/scripts/run_p2_full_dual_graph_collector_preflight_v2.py 代码/tests/test_run_p2_full_dual_graph_collector_preflight_v2.py
git commit -m "feat: collect role-separated P2 attempt evidence"
```

### Task 5: P2-C v2 ledger-derived audit

**Files:**
- Create: `代码/tests/test_p2c_scale_distribution_audit_v2.py`
- Create: `代码/tests/test_run_p2c_scale_distribution_audit_v2.py`
- Create: `代码/src/pi_jwm/p2c_scale_distribution_audit_v2.py`
- Create: `代码/scripts/run_p2c_scale_distribution_audit_v2.py`

- [ ] **Step 1: Write failing audit tests**

Build temporary bundles and cover missing ledger, duplicate/deleted attempt, invalid state path, handwritten `action_rejection_count` disagreement, frame mapping mismatch, role contamination, digest mismatch, mutation/quarantine contradiction, a valid ledger with zero natural-reference rejection, manifest/input/source tampering, and the retained blockers:

```python
assert report["rejection_quarantine"]["action_rejection_count"] == 0
assert report["rejection_quarantine"]["action_rejection_rate"] == 0.0
assert "action_rejection_rate_not_observed" not in report["blocking_reasons"]
assert set(report["blocking_reasons"]) >= {
    "scenario_matrix_not_frozen",
    "formal_scale_not_frozen",
    "formal_split_not_frozen",
}
assert report["audit_status"] == "blocked"
```

- [ ] **Step 2: Run P2-C v2 tests and confirm RED**

```powershell
python -m unittest -v test_p2c_scale_distribution_audit_v2.py test_run_p2c_scale_distribution_audit_v2.py
```

Expected: import failures for the missing audit and runner modules.

- [ ] **Step 3: Implement independent ledger recomputation**

Implement:

```python
AUDIT_SCHEMA_VERSION = "PIJWM-P2C-Scale-Distribution-Audit-v2"
FORMAL_CONFIG_SCHEMA_VERSION = "PIJWM-P2C-Formal-Data-Config-Candidate-v2"

def load_action_attempts(path: Path) -> list[Mapping[str, object]]: ...
def audit_bundle(bundle_dir: str | Path, *, project_root=None) -> dict[str, object]: ...
```

Reuse v1's read-only E1, snapshot, action, transfer, runtime-guard, and candidate-config helpers, but never reuse v1's unobserved rejection field as evidence. Re-run the ledger validator and frame/replay alignment, calculate counts separately for all four roles, derive the natural-reference rate as `rejected / attempts`, report mutation and quarantine counts, and remove only `action_rejection_rate_not_observed` after every ledger semantic gate passes.

- [ ] **Step 4: Implement P2-C v2 publication and verification**

Emit:

```text
p2c_scale_distribution_audit_v2.json
p2c_formal_data_config_candidate_v2.json
manifest.json
```

Bind all nine P2-B v2 input files and the v2 ledger, collector adapter, artifact, P2-B runner, P2-C code/tests, approved design, implementation plan, and research-progress document. Keep `formal_data_approved`, `training_eligible`, `gpu_started`, and `locked_test_accessed` false.

- [ ] **Step 5: Run P2-C v2 and v1 regression tests and confirm GREEN**

```powershell
python -m unittest -v test_p2c_scale_distribution_audit_v2.py test_run_p2c_scale_distribution_audit_v2.py test_p2c_scale_distribution_audit_v1.py test_run_p2c_scale_distribution_audit_v1.py
```

Expected: all tests pass; v1 still reports its historical blocker for a ledger-less v1 bundle.

- [ ] **Step 6: Commit the P2-C v2 slice**

```powershell
git add -- 代码/src/pi_jwm/p2c_scale_distribution_audit_v2.py 代码/scripts/run_p2c_scale_distribution_audit_v2.py 代码/tests/test_p2c_scale_distribution_audit_v2.py 代码/tests/test_run_p2c_scale_distribution_audit_v2.py
git commit -m "feat: audit P2 rejection rate from ledger"
```

### Task 6: Full CPU regression and real candidate gate

**Files:**
- Generate only: `代码/artifacts/preflight/pi_jwm_p2_full_dual_graph_collector_v2_candidate_20260814/` or its `_failed/` sibling
- Generate only after P2-B success: `代码/artifacts/audit/pi_jwm_p2c_scale_distribution_audit_v2_candidate_20260814/`

- [ ] **Step 1: Run the focused P1/P2 regression from the isolated worktree**

```powershell
cd D:\shen\网络组\.worktrees\p2-action-attempt-ledger-v1\代码\tests
python -m unittest -v test_action_attempt_ledger_v1.py test_airfogsim_full_dual_graph_collector_v2.py test_full_dual_graph_artifact_v2.py test_run_p2_full_dual_graph_collector_preflight_v2.py test_p2c_scale_distribution_audit_v2.py test_run_p2c_scale_distribution_audit_v2.py test_full_dual_graph_collector_contract_v1.py test_airfogsim_full_dual_graph_frame_builder_v1.py test_airfogsim_full_dual_graph_collector_v1.py test_full_dual_graph_artifact_v1.py test_p2c_scale_distribution_audit_v1.py test_run_p2c_scale_distribution_audit_v1.py
```

Expected: zero failures and zero errors. Record the exact test count from stdout rather than predicting it.

- [ ] **Step 2: Verify historical immutable bundles before a new run**

Run both existing `--verify-only` commands against the P2-B v1 and P2-C v1 canonical directories. Expected: both return exit code 0, proving v2 work did not invalidate historical source closure.

- [ ] **Step 3: Run the frozen real CPU candidate**

```powershell
python D:\shen\网络组\.worktrees\p2-action-attempt-ledger-v1\代码\scripts\run_p2_full_dual_graph_collector_preflight_v2.py --seeds 0 1 2 --steps 20 --output-dir D:\shen\网络组\代码\artifacts\preflight\pi_jwm_p2_full_dual_graph_collector_v2_candidate_20260814
```

Expected request shape: three seeds, two arms, and twenty reference frames per arm. Do not state 120 as observed until it is read from `action_attempts.jsonl`.

- [ ] **Step 4: Branch on actual candidate evidence**

If `_failed` exists or any real rejection is recorded: verify the failure bundle, preserve it byte-for-byte, report the real terminal stage/mutation/quarantine evidence, and stop before P2-C publication or canonical promotion. If the success directory exists: run v2 `--verify-only`, independently parse the ledger, assert natural-reference accepted/frame one-to-one alignment, binary conservation, reference/replay digest alignment, zero quarantine, and conservative status flags.

- [ ] **Step 5: Publish and verify the P2-C v2 candidate only after P2-B success**

```powershell
python D:\shen\网络组\.worktrees\p2-action-attempt-ledger-v1\代码\scripts\run_p2c_scale_distribution_audit_v2.py --bundle D:\shen\网络组\代码\artifacts\preflight\pi_jwm_p2_full_dual_graph_collector_v2_candidate_20260814 --output-dir D:\shen\网络组\代码\artifacts\audit\pi_jwm_p2c_scale_distribution_audit_v2_candidate_20260814
python D:\shen\网络组\.worktrees\p2-action-attempt-ledger-v1\代码\scripts\run_p2c_scale_distribution_audit_v2.py --verify-only --bundle D:\shen\网络组\代码\artifacts\preflight\pi_jwm_p2_full_dual_graph_collector_v2_candidate_20260814 --output-dir D:\shen\网络组\代码\artifacts\audit\pi_jwm_p2c_scale_distribution_audit_v2_candidate_20260814
```

Expected: rejection observation is ledger-derived, while scenario, formal-scale, and formal-split blockers remain and audit status remains `blocked`.

### Task 7: Evidence documentation, final verification, and integration-ready branch

**Files:**
- Modify: `本地计划表.md`
- Create: `文档/研究进展/2026-08-14-PI-JWM-P2联合动作Attempt-Reject-Ledger-v1实施与证据.md`

- [ ] **Step 1: Write evidence-backed status updates**

Record only facts read from the verified artifacts and fresh command output: exact counts by role, accepted/rejected conservation, rejection rate if observable, quarantine/mutation distribution, file/hash verification, P2-C blockers, and whether a success or failure bundle exists. Label the ledger as collection audit infrastructure, not a world model, policy, or candidate-rollout planner.

- [ ] **Step 2: Run documentation claim scans**

Search the two updated documents for claims that would imply formal dataset completion, training eligibility, GPU execution, locked-test access, completed rollout planning, or final-method freezing. Every such status must remain false or explicitly not implemented.

- [ ] **Step 3: Run fresh final verification**

Re-run the complete focused suite from Task 6, both v2 `--verify-only` commands if success artifacts exist, both v1 historical `--verify-only` commands, `git diff --check`, and a source scan proving no v1 production file or AirFogSim file changed on this branch.

- [ ] **Step 4: Commit documentation only after evidence is fresh**

```powershell
git add -- 本地计划表.md 文档/研究进展/2026-08-14-PI-JWM-P2联合动作Attempt-Reject-Ledger-v1实施与证据.md
git commit -m "docs: record P2 attempt ledger evidence"
```

- [ ] **Step 5: Review branch scope**

Use `git status --short`, `git diff --stat 647edd6...HEAD`, and `git log --oneline 647edd6..HEAD`. The branch must contain only the plan, new v2/ledger production files, their tests, and the two evidence documents; it must not contain the main worktree's unrelated dirty changes.

## Plan self-review record

- Spec coverage: Tasks 1–5 cover identity, roles, transitions, mutation, runtime observation, success/failure publication, v1/v2 isolation, P2-C recomputation, manifests, and all named negative tests; Tasks 6–7 cover the frozen CPU request, immutable failure handling, historical verification, and conservative documentation.
- Placeholder scan: implementation steps name concrete APIs, files, commands, expected RED/GREEN evidence, failure branches, and prohibited expansions.
- Type consistency: `AttemptIdentity`, `ActionAttemptLedger`, `AttemptHandle`, terminal JSON rows, `validate_attempt_records`, `summarize_attempts`, and `validate_bundle_alignment` retain the same names and roles across runner, artifact, and P2-C tasks.
- Existing-experiment impact: P2-B v1 source and artifacts remain immutable and retain historical preflight meaning; no trained checkpoint is invalidated. P2-B v1 still cannot support an observed rejection-rate claim, and P2-C v2 does not retroactively repair it.
