# PI-JWM P2 Multistep Temporal Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the P2 action-pre channel evidence, then verify a three-frame real AirFogSim CPU trajectory with strict previous-frame backfill, stable append-only identities, atomic publication, and conservative status claims.

**Architecture:** Keep AirFogSim unchanged. Extend the existing PI-JWM single-step adapter with an explicit callback that executes after action validation but before every scheduler setter, and use that callback to capture decision-time CSI. Add a simulator-independent multistep contract for stable node/edge/flow identities and committed previous outcomes, then compose three real single steps in a dedicated nontraining runner. Reuse v4 validators and the frozen CPU rule; preserve outcome-side channel data under explicit timing labels.

**Tech Stack:** Python 3, `dataclasses`, NumPy, `unittest`, AirFogSim public scheduler boundaries, JSON/SHA-256 manifests, PowerShell, Conda environment `airfogsim`.

---

## File Map

- Modify `代码/src/pi_jwm/airfogsim_single_step_collector_v1.py`: add the validated-before-setter observation hook and expose the observation/trace in the one-step result.
- Modify `代码/scripts/run_p2_single_step_collector_preflight_v1.py`: capture setter-before CSI, label outcome-side channel values, validate timing payloads, bind the correction design, and rebuild the canonical single-step bundle.
- Modify `代码/tests/test_airfogsim_single_step_collector_v1.py`: prove hook ordering against real adapter control flow.
- Modify `代码/tests/test_run_p2_single_step_collector_preflight_v1.py`: reject outcome-side attenuation used as action-pre and detect timing/provenance tampering.
- Create `代码/src/pi_jwm/multistep_collector_contract_v1.py`: append-only vocabularies, edge bindings, previous-outcome projection, and transactional history commit.
- Create `代码/tests/test_multistep_collector_contract_v1.py`: exhaustive pure-contract tests.
- Create `代码/scripts/run_p2_multistep_collector_preflight_v1.py`: run the fixed three-frame real fixture and atomically write/verify its bundle.
- Create `代码/tests/test_run_p2_multistep_collector_preflight_v1.py`: fixture-payload, publication, verifier, failure-atomicity, and status-boundary tests.
- Modify `本地计划表.md`, `task_plan.md`, `findings.md`, `progress.md`: record the corrected fact boundary and the exact verification result without upgrading P2 to a formal v4 dataset.
- Preserve `代码/artifacts/preflight/pi_jwm_p2_single_step_collector_v1/` by moving it to `代码/artifacts/preflight/pi_jwm_p2_single_step_collector_v1_pre_temporal_fix_20260813/` only after confirming the destination does not exist and the source manifest is readable.
- Create canonical `代码/artifacts/preflight/pi_jwm_p2_single_step_collector_v1/` and `代码/artifacts/preflight/pi_jwm_p2_multistep_collector_v1/` only from fully verified temporary candidates.

### Task 1: Lock the before-setter observation boundary

**Files:**
- Modify: `代码/tests/test_airfogsim_single_step_collector_v1.py`
- Modify: `代码/src/pi_jwm/airfogsim_single_step_collector_v1.py`

- [ ] **Step 1: Write the failing adapter-order tests**

Add tests using the existing fake environment/schedulers and a shared call ledger:

```python
def test_pre_action_observer_runs_after_validation_and_before_any_setter(self):
    calls = []
    result = execute_candidate(
        env,
        valid_action,
        task_ids=("task0",),
        node_ids=("uav0", "rsu0"),
        edge_count=1,
        flow_count=1,
        n_rb=1,
        task_scheduler=RecordingTaskScheduler(calls),
        communication_scheduler=RecordingCommunicationScheduler(calls),
        computation_scheduler=RecordingComputationScheduler(calls),
        pre_action_observer=lambda: calls.append("decision_time_csi_read") or {"value": [1.0]},
    )
    self.assertLess(calls.index("decision_time_csi_read"), calls.index("offload_setter"))
    self.assertLess(calls.index("decision_time_csi_read"), calls.index("rb_setter"))
    self.assertEqual({"value": [1.0]}, result.pre_action_observation)

def test_invalid_action_never_calls_pre_action_observer_or_setters(self):
    with self.assertRaises(ValueError):
        execute_candidate(...invalid duplicate-RB action..., pre_action_observer=observer)
    self.assertEqual([], calls)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m unittest 代码.tests.test_airfogsim_single_step_collector_v1 -v
```

Expected: FAIL because `execute_candidate` does not accept `pre_action_observer`, and the result has no `pre_action_observation` or `temporal_trace`.

- [ ] **Step 3: Implement the minimum hook and trace**

Extend `SingleStepExecutionResult` with:

```python
pre_action_observation: Any
temporal_trace: tuple[str, ...]
```

Extend `execute_candidate(..., pre_action_observer: Callable[[], Any] | None = None)`. After `validate_candidate_action(...)` succeeds and before `SingleStepRecorder.install_cpu_callback` or either scheduler setter, execute the observer exactly once. Record these ordered phases:

```python
(
    "action_validated",
    "decision_time_observation_captured",
    "cpu_callback_installed",
    "action_setters_called",
    "env_step_started",
    "env_step_finished",
)
```

If no observer is supplied, store `None` but still record the explicit `decision_time_observation_skipped` phase. Do not catch observer exceptions and do not call any setter after an observer failure.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the same `unittest` command. Expected: all adapter tests PASS with no warnings.

- [ ] **Step 5: Commit the adapter boundary**

```powershell
git add -- 代码/src/pi_jwm/airfogsim_single_step_collector_v1.py 代码/tests/test_airfogsim_single_step_collector_v1.py
git commit -m "fix: capture P2 observations before action setters"
```

### Task 2: Correct and republish the single-step temporal evidence

**Files:**
- Modify: `代码/tests/test_run_p2_single_step_collector_preflight_v1.py`
- Modify: `代码/scripts/run_p2_single_step_collector_preflight_v1.py`
- Input: `docs/superpowers/specs/2026-08-13-p2-multistep-temporal-contract-design.md`
- Preserve: `代码/artifacts/preflight/pi_jwm_p2_single_step_collector_v1/`
- Create: `代码/artifacts/preflight/pi_jwm_p2_single_step_collector_v1_pre_temporal_fix_20260813/`

- [ ] **Step 1: Write failing temporal-evidence tests**

Add fixture tests that require:

```python
audit = payloads["field_mask_audit.json"]["candidate_audits"][0]
pre = audit["decision_time_channel"]
event = payloads["transfer_events.json"]["candidates"][0]["events"][0]
self.assertEqual("before_action_setters", pre["capture_phase"])
self.assertEqual("after_fast_fading_before_transfer", event["capture_phase"])
self.assertEqual(
    "direct_decision_time_csi_before_setters",
    field_by_name(audit, "pre_rb_optional.channel_attenuation_db")["provenance"],
)
self.assertIsNot(
    event["outcome_channel_attenuation_db"],
    pre["channel_attenuation_db"],
)
```

Add verifier tampering tests that replace the pre field provenance with `direct_runtime_channel_event` or remove the order trace and expect `verify_preflight_bundle(...)["passed"] is False`. Also require `write_preflight_bundle` to reject a payload where pre values are sourced from the event object even when `validation_report.json["passed"]` is manually true.

- [ ] **Step 2: Run the runner tests and verify RED**

Run:

```powershell
python -m unittest 代码.tests.test_run_p2_single_step_collector_preflight_v1 -v
```

Expected: FAIL because the current runner has no setter-before CSI record and labels post-fast-fading `attenuation_db` as the pre field.

- [ ] **Step 3: Implement explicit decision-time and outcome-side capture**

Add `_capture_decision_time_channel(env, source, target, rb_indices)` that resolves node types and indices through the existing environment APIs and calls `channel_manager.getCSI(...)` before any setter. Return:

```python
{
    "capture_phase": "before_action_setters",
    "simulation_time": float(env.simulation_time),
    "source": source,
    "target": target,
    "rb_indices": [...],
    "channel_attenuation_db": [...],
    "source_method": "channel_manager.getCSI",
}
```

Pass this function as the adapter's `pre_action_observer`. Store its output and trace in every candidate row. Change `_event_from_profile` so the post-`computeRate` array is named `outcome_channel_attenuation_db` and carries `capture_phase="after_fast_fading_before_transfer"`; update all event consumers accordingly. `_field_audit` must derive E1 mean/std and E3 per-RB values only from `candidate["decision_time_channel"]`.

Add a pure `validate_temporal_payloads(payloads)` used by both writer and verifier. It must independently check capture phases, source method, source/target/RB identity, trace order, and that no pre field names a transfer-event source. The verifier must parse payload files and call this validator instead of trusting the saved validation boolean.

Bind the new design document in `source_hashes` and keep all conservative flags false.

- [ ] **Step 4: Run single-step unit and regression tests**

Run:

```powershell
python -m unittest 代码.tests.test_airfogsim_single_step_collector_v1 代码.tests.test_run_p2_single_step_collector_preflight_v1 代码.tests.test_single_step_collector_contract_v1 -v
```

Expected: PASS.

- [ ] **Step 5: Archive the defective canonical bundle recoverably**

Use PowerShell read-only checks first and verify both resolved targets stay under the intended preflight directory:

```powershell
$preflight = (Resolve-Path -LiteralPath '代码/artifacts/preflight').Path
$source = (Resolve-Path -LiteralPath (Join-Path $preflight 'pi_jwm_p2_single_step_collector_v1')).Path
$archive = Join-Path $preflight 'pi_jwm_p2_single_step_collector_v1_pre_temporal_fix_20260813'
if ((Split-Path -Parent $source) -ne $preflight -or (Split-Path -Parent $archive) -ne $preflight) {
    throw 'archive paths escaped the intended preflight directory'
}
if (Test-Path -LiteralPath $archive) { throw "archive already exists: $archive" }
$manifest = Get-Content -Raw -LiteralPath (Join-Path $source 'manifest.json') | ConvertFrom-Json
if (-not $manifest.schema_version) { throw 'source manifest is unreadable' }
Move-Item -LiteralPath $source -Destination $archive
```

This is a recoverable move within `代码/artifacts/preflight/`; do not delete any prior self-review bundle.

- [ ] **Step 6: Run and independently verify the corrected real bundle**

Run:

```powershell
conda run -n airfogsim python 代码/scripts/run_p2_single_step_collector_preflight_v1.py
conda run -n airfogsim python 代码/scripts/run_p2_single_step_collector_preflight_v1.py --verify-only
```

Expected: both commands exit 0; the canonical bundle reports explicit setter-before CSI and outcome-side capture, and all forbidden status flags remain false.

- [ ] **Step 7: Commit code and corrected evidence**

Stage only the runner, tests, and any tracked artifact files. Inspect `git diff --cached --stat` before committing.

```powershell
git add -- 代码/scripts/run_p2_single_step_collector_preflight_v1.py 代码/tests/test_run_p2_single_step_collector_preflight_v1.py
git add -f -- 代码/artifacts/preflight/pi_jwm_p2_single_step_collector_v1
git commit -m "fix: correct P2 single-step channel timing evidence"
```

If artifacts are intentionally ignored and the repository's established pattern keeps them untracked, do not force-add them; record the exact verified path and manifest hash in `progress.md` instead.

### Task 3: Implement append-only identity and committed-history contracts

**Files:**
- Create: `代码/tests/test_multistep_collector_contract_v1.py`
- Create: `代码/src/pi_jwm/multistep_collector_contract_v1.py`

- [ ] **Step 1: Write failing vocabulary tests**

Define the wished-for API in tests:

```python
vocabulary = TrajectoryVocabulary()
first = vocabulary.observe(
    node_ids=("uav0", "rsu0"),
    edges=(EdgeIdentity("ie::uav0::rsu0", "uav0", "rsu0", "wireless:U2I"),),
    flow_ids=("task0",),
)
second = vocabulary.observe(node_ids=("rsu0",), edges=(), flow_ids=())
third = vocabulary.observe(
    node_ids=("uav0", "rsu0", "rsu1"),
    edges=(EdgeIdentity("ie::uav0::rsu0", "uav0", "rsu0", "wireless:U2I"),),
    flow_ids=("task1",),
)
self.assertEqual(first.node_indices["uav0"], third.node_indices["uav0"])
self.assertEqual(2, third.node_indices["rsu1"])
self.assertFalse(second.node_presence[first.node_indices["uav0"]])
```

Add separate tests for deterministic first insertion, append-only indices, disappearance/reappearance, edge binding conflict, dangling endpoints, duplicate observations, and flow-index non-reuse.

- [ ] **Step 2: Run vocabulary tests and verify RED**

Run:

```powershell
python -m unittest 代码.tests.test_multistep_collector_contract_v1 -v
```

Expected: import failure because the new module does not exist.

- [ ] **Step 3: Implement the minimum vocabulary types**

Create:

```python
@dataclass(frozen=True)
class EdgeIdentity:
    edge_id: str
    source_id: str
    target_id: str
    edge_class: str

@dataclass(frozen=True)
class VocabularySnapshot:
    node_indices: dict[str, int]
    edge_indices: dict[str, int]
    flow_indices: dict[str, int]
    node_presence: tuple[bool, ...]
    edge_presence: tuple[bool, ...]
    flow_presence: tuple[bool, ...]

class TrajectoryVocabulary:
    def observe(self, *, node_ids, edges, flow_ids) -> VocabularySnapshot: ...
```

Validate identifiers before mutation, sort only identities first seen in the same call, append new identities, preserve all old indices, and reject edge-binding changes or dangling endpoints atomically.

- [ ] **Step 4: Run vocabulary tests and verify GREEN**

Run the focused test command. Expected: vocabulary tests PASS.

- [ ] **Step 5: Write failing history tests**

Add tests for:

```python
history = LinkHistoryLedger()
first = history.project(edge_ids=("ie0",))
self.assertEqual(False, first[0].valid)
self.assertEqual(MissingReason.NO_HISTORY, first[0].missing_reason)

history.commit(
    frame_index=0,
    outcomes={"ie0": LinkOutcome(active_flow_count=1.0, effective_rate_per_s=4.0, served_data=0.4)},
    frame_validated=True,
)
second = history.project(edge_ids=("ie0",))
self.assertEqual((1.0, 4.0, 0.4), second[0].values)

history.commit(
    frame_index=1,
    outcomes={"ie0": LinkOutcome(0.0, 0.0, 0.0)},
    frame_validated=True,
)
third = history.project(edge_ids=("ie0",))
self.assertTrue(third[0].valid)
self.assertEqual(MissingReason.NONE, third[0].missing_reason)
```

Also test that `frame_validated=False`, skipped frame indices, NaN/negative values, unknown edges, or partial mutation leave the previous committed history unchanged.

- [ ] **Step 6: Run history tests and verify RED**

Run the focused test command. Expected: FAIL because history types are absent.

- [ ] **Step 7: Implement transactional history commit**

Add immutable `LinkOutcome` and projected-field records plus `LinkHistoryLedger`. Validate all outcomes into a temporary normalized mapping, require contiguous frame indices, and replace committed state only after all validation passes. `project()` before any commit returns `NO_HISTORY`; after a successful zero-outcome commit it returns valid zeros with `MissingReason.NONE`.

- [ ] **Step 8: Run all pure-contract tests and commit**

```powershell
python -m unittest 代码.tests.test_multistep_collector_contract_v1 代码.tests.test_information_edge_contract_v4 -v
git add -- 代码/src/pi_jwm/multistep_collector_contract_v1.py 代码/tests/test_multistep_collector_contract_v1.py
git commit -m "feat: add P2 multistep identity and history contracts"
```

Expected: tests PASS and commit succeeds.

### Task 4: Compose the fixed three-frame real trajectory

**Files:**
- Create: `代码/tests/test_run_p2_multistep_collector_preflight_v1.py`
- Create: `代码/scripts/run_p2_multistep_collector_preflight_v1.py`

- [ ] **Step 1: Write failing payload and verifier tests**

Load the runner by path, as existing runner tests do. Require a `fake_passing_payloads_for_test()` with exactly the eight designed files, strict false status flags, three frames, first-frame `NO_HISTORY`, second-frame positive prior result, and third-frame valid zero prior result. Add tampering tests for:

- reordered temporal phases;
- changed edge index between frames;
- history values not equal to the preceding frame outcome;
- valid zero rewritten as missing;
- `training_eligible`, `v4_collector_implemented`, or `gpu_started` set true;
- source/artifact hash mismatch;
- validation failure leaving no canonical manifest.

- [ ] **Step 2: Run runner tests and verify RED**

```powershell
python -m unittest 代码.tests.test_run_p2_multistep_collector_preflight_v1 -v
```

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement in-memory validation and atomic bundle I/O**

Define:

```python
REQUIRED_FILES = (
    "trajectory_frames.json",
    "vocabularies.json",
    "temporal_trace.json",
    "history_alignment_audit.json",
    "resource_bundle.json",
    "validation_report.json",
    "summary.json",
    "manifest.json",
)
```

Implement `validate_multistep_payloads`, `write_preflight_bundle`, and `verify_preflight_bundle`. The validator must recompute frame contiguity, trace order, vocabulary stability, action/outcome/history identity, exact previous-frame values, first-frame `NO_HISTORY`, valid-zero semantics, and forbidden flags. The verifier must parse every payload and rerun this validator plus all hashes. Write only to a unique sibling temporary directory and publish with `os.replace`; remove the temporary directory on failure and never overwrite a nonempty canonical directory.

- [ ] **Step 4: Run I/O tests and verify GREEN**

Run the focused test command. Expected: all fixture, tampering, and atomicity tests PASS.

- [ ] **Step 5: Write the failing real-payload contract test**

Add a test guarded by AirFogSim availability that calls `build_real_payloads(seed=0)` and checks:

```python
self.assertEqual(3, len(frames))
self.assertEqual([0, 1, 2], [row["frame_index"] for row in frames])
self.assertEqual(1, len(frames[0]["action"]["offloads"]))
self.assertGreater(frames[0]["outcome_link"][0]["served_data"], 0.0)
self.assertEqual(frames[0]["outcome_link"], frames[1]["pre_link_history_source"])
self.assertTrue(all(row["valid"] for row in frames[2]["pre_link"]))
self.assertTrue(all(row["value"] == 0.0 for row in frames[2]["pre_link"]))
```

Require three real `env.step()` calls, setter-before CSI on every frame, exactly one frame-0 offload/RB action, no new actions on frames 1/2, stable indices, and per-frame CPU/energy conservation.

- [ ] **Step 6: Run the real-payload test and verify RED**

```powershell
conda run -n airfogsim python -m unittest 代码.tests.test_run_p2_multistep_collector_preflight_v1 -v
```

Expected: FAIL because `build_real_payloads` is not implemented.

- [ ] **Step 7: Implement the three-frame AirFogSim fixture**

Reuse the single-step environment builder, warm-up, target selection, scheduler adapter, CPU ledger, event recorder, energy accounting, and v4 validators rather than copying their semantics. For frame 0, use the first pre-action distance-ranked remote target and all legal RBs. For frames 1 and 2, pass empty `CandidateAction.offloads` and `rb_assignments` while executing one real step each.

Before every action, observe node/edge/flow identities and capture decision-time CSI. Slice transfer events per frame rather than reusing the cumulative list. Aggregate a `LinkOutcome` for every vocabulary edge; an observed frame with no event commits explicit zero outcome. Project the ledger before each frame, validate the complete frame, then commit only after success.

Do not relabel the observed-edge vocabulary as a complete strict dual graph. Set `scope="three_frame_nontraining_temporal_fixture"` and all method/training completion flags false.

- [ ] **Step 8: Run real fixture and regression tests**

```powershell
conda run -n airfogsim python -m unittest 代码.tests.test_run_p2_multistep_collector_preflight_v1 代码.tests.test_run_p2_single_step_collector_preflight_v1 代码.tests.test_airfogsim_single_step_collector_v1 代码.tests.test_multistep_collector_contract_v1 -v
```

Expected: PASS.

- [ ] **Step 9: Commit the multistep runner**

```powershell
git add -- 代码/scripts/run_p2_multistep_collector_preflight_v1.py 代码/tests/test_run_p2_multistep_collector_preflight_v1.py
git commit -m "feat: add P2 three-frame temporal smoke"
```

### Task 5: Publish, verify, and record the evidence boundary

**Files:**
- Create: `代码/artifacts/preflight/pi_jwm_p2_multistep_collector_v1/`
- Modify: `本地计划表.md`
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

- [ ] **Step 1: Generate the canonical multistep bundle**

Confirm the canonical directory does not already contain files, then run:

```powershell
conda run -n airfogsim python 代码/scripts/run_p2_multistep_collector_preflight_v1.py
conda run -n airfogsim python 代码/scripts/run_p2_multistep_collector_preflight_v1.py --verify-only
```

Expected: both exit 0. Record the manifest schema, artifact count, source count, frame count, and every acceptance gate from actual output.

- [ ] **Step 2: Run fresh focused and broader regressions**

```powershell
conda run -n airfogsim python -m unittest `
  代码.tests.test_single_step_collector_contract_v1 `
  代码.tests.test_airfogsim_single_step_collector_v1 `
  代码.tests.test_run_p2_single_step_collector_preflight_v1 `
  代码.tests.test_multistep_collector_contract_v1 `
  代码.tests.test_run_p2_multistep_collector_preflight_v1 `
  代码.tests.test_information_edge_contract_v4 `
  代码.tests.test_airfogsim_cpu_inner_rule_v1 `
  代码.tests.test_cpu_inner_rule_v1 -v

conda run -n airfogsim python -m py_compile `
  代码/src/pi_jwm/airfogsim_single_step_collector_v1.py `
  代码/src/pi_jwm/multistep_collector_contract_v1.py `
  代码/scripts/run_p2_single_step_collector_preflight_v1.py `
  代码/scripts/run_p2_multistep_collector_preflight_v1.py
```

Expected: all selected tests and compilation PASS. If a broader repository test fails, record the exact failing test and determine whether it predates this change; do not describe the whole repository as passing without evidence.

- [ ] **Step 3: Independently inspect machine-readable truth claims**

Use PowerShell JSON parsing to assert:

```powershell
$single = Get-Content -Raw '代码/artifacts/preflight/pi_jwm_p2_single_step_collector_v1/manifest.json' | ConvertFrom-Json
$multi = Get-Content -Raw '代码/artifacts/preflight/pi_jwm_p2_multistep_collector_v1/manifest.json' | ConvertFrom-Json
foreach ($name in 'gpu_started','locked_test_accessed','training_eligible','v4_collector_implemented','v4_dataset_complete','candidate_rollout_planner_complete','final_method_frozen') {
    if ($single.status_flags.$name -or $multi.status_flags.$name) { throw "unsafe flag: $name" }
}
```

Also recalculate SHA-256 for every manifest artifact and source entry independently of each runner's `--verify-only` path. Expected: zero mismatches.

- [ ] **Step 4: Update project status documents from measured facts**

Record:

- the old single-step bundle's action-pre claim was corrected, not silently retained;
- the fixed three-frame fixture passed or the exact gate that failed;
- stable observed-edge identities and E1 history alignment are locally verified only;
- complete strict dual graphs, formal v4 dataset, multi-seed coverage, model training, and candidate world-model rollout remain pending;
- GPU and locked test were not used.

Do not rewrite unrelated pre-existing status sections or clean the dirty worktree.

- [ ] **Step 5: Commit final evidence and scoped status updates**

Inspect the staged diff line by line. Stage only files changed by this plan and tracked artifact files consistent with repository practice.

```powershell
git diff --cached --check
git diff --cached --stat
git commit -m "test: verify P2 multistep temporal evidence"
```

- [ ] **Step 6: Final verification after the last commit**

Repeat both `--verify-only` commands and the focused test command from Step 2 after the commit. Report actual counts and limitations; do not claim formal v4 collector or dataset completion.

## Plan Self-Review

- **Spec coverage:** Every design requirement has a task: timing correction in Tasks 1-2, append-only identity/history in Task 3, real three-frame execution in Task 4, and atomic evidence/status audit in Task 5.
- **Truth boundary:** The plan explicitly preserves the real single-step closed loop while retracting only the invalid action-pre field evidence. It never upgrades the observed-edge vocabulary into a complete dual graph.
- **TDD order:** Every production change is preceded by a named failing test and an explicit RED command.
- **Failure behavior:** Observation, frame validation, history commit, temporary publication, canonical replacement, and manifest verification each have an atomic failure test.
- **No fabricated data:** The real fixture uses a predeclared seed, target rule, and action schedule. Zero communication in later frames is an observed result with valid provenance, not a padded feature.
- **Scope:** GPU, locked test, formal data generation, model training, baselines, and candidate world-model rollout are excluded.
- **Placeholder scan:** The plan contains no TBD/TODO steps, unspecified “appropriate” handling, or references to undefined later work.
