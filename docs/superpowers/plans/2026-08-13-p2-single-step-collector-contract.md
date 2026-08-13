# PI-JWM P2 Single-Step Collector Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a non-training, versioned PI-JWM single-step collector that executes two action-different candidates in real AirFogSim, invokes the frozen CPU inner rule on each candidate's actual post-communication compute set, and publishes only auditable evidence.

**Architecture:** Keep pure action/COO validation and bundle schemas in a PI-JWM source module. Keep AirFogSim-specific state-machine execution, direct wireless event extraction, callback wrapping, and candidate comparison in a separate adapter. A small canonical runner will execute the adapter in the dedicated `airfogsim` environment and write a preflight bundle whose manifest explicitly blocks dataset/training claims.

**Tech Stack:** Python 3, dataclasses, NumPy, PyYAML, AirFogSim reference source, `unittest`, existing v4 validators and CPU inner-rule adapter.

---

### Task 1: Freeze the pure collector contract and action validation

**Files:**
- Create: `代码/src/pi_jwm/single_step_collector_contract_v1.py`
- Test: `代码/tests/test_single_step_collector_contract_v1.py`

- [ ] **Step 1: Write failing tests for the exact public contract**

```python
from pi_jwm.single_step_collector_contract_v1 import (
    CandidateAction,
    OffloadAction,
    RbAssignment,
    validate_candidate_action,
)

def test_candidate_action_rejects_duplicate_and_out_of_range_rb_records():
    action = CandidateAction(
        candidate_id="local",
        offloads=(OffloadAction("veh0", "task0", "veh0", ("veh0",)),),
        rb_assignments=(RbAssignment(0, 0, 0, 0), RbAssignment(0, 0, 0, 0)),
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_candidate_action(action, task_ids=("task0",), edge_count=1,
                                  flow_count=1, n_rb=1)

def test_bundle_flags_keep_single_step_separate_from_training():
    flags = build_single_step_status_flags()
    assert flags["single_step_real_airfogsim_executed"] is False
    assert flags["v4_collector_implemented"] is False
    assert flags["training_eligible"] is False
```

- [ ] **Step 2: Run the focused test and verify the missing-contract failure**

Run (from `代码/tests`): `python -m unittest test_single_step_collector_contract_v1.py -v`

Expected: FAIL because the new module and symbols do not exist.

- [ ] **Step 3: Implement the minimal pure dataclasses and validators**

Implement frozen `OffloadAction`, `RbAssignment`, and `CandidateAction` dataclasses. `validate_candidate_action` must check non-empty IDs, task/source/target membership, route endpoint equality, integer COO dtype/shape/uniqueness, and all four capacities before any AirFogSim setter can be called. Reuse `information_edge_contract_v4.validate_assignment_coo`; never rely on AirFogSim's modulo behavior for invalid RBs. Add `build_single_step_status_flags()` with all non-training flags set explicitly to false.

- [ ] **Step 4: Run the focused test and verify it passes**

Run (from `代码/tests`): `python -m unittest test_single_step_collector_contract_v1.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the pure contract**

```powershell
git add 代码/src/pi_jwm/single_step_collector_contract_v1.py 代码/tests/test_single_step_collector_contract_v1.py
git commit -m "feat: add P2 single-step action contract"
```

### Task 2: Implement the AirFogSim single-step execution adapter

**Files:**
- Create: `代码/src/pi_jwm/airfogsim_single_step_collector_v1.py`
- Test: `代码/tests/test_airfogsim_single_step_collector_v1.py`

- [ ] **Step 1: Write failing tests for callback and event boundary behavior**

```python
def test_cpu_callback_receives_candidate_compute_set_and_records_ledger():
    env = FakeEnvWithTaskManager()
    recorder = SingleStepRecorder(env, candidate_id="local")
    recorder.install_cpu_callback()
    allocations = env.task_manager.invoke_compute_callback({"veh0": [FakeTask("task0", "veh0")]})
    assert allocations == {"task0": 2.0}
    assert recorder.cpu_rows[0]["rule_version"] == "PIJWM-CPU-Inner-Rule-v1"

def test_invalid_rb_is_rejected_before_airfogsim_setter():
    with pytest.raises(ValueError, match="resource index"):
        execute_candidate(fake_env(), invalid_action())
    assert fake_env().communication_setter_calls == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from `代码/tests`): `python -m unittest test_airfogsim_single_step_collector_v1.py -v`

Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Implement the adapter around real simulator boundaries**

Implement a `SingleStepRecorder`/`execute_candidate` path that: (a) validates the candidate first; (b) calls `TaskScheduler.setTaskOffloading` and `CommunicationScheduler.setCommunicationWithRB`; (c) wraps the existing `make_airfogsim_cpu_callback` and records the exact computing-task IDs, before/after CPU work, allocations, and rule version; (d) instruments the real `AirFogSimEnv._compute_communication_rate`/`_execute_communication` boundary to record direct per-RB CSI, rate, outage, remaining-before, and delivered data when available; (e) calls exactly one real `env.step()` and records the documented simulator order; (f) builds E0/E1/action/outcome/ledger records using `MissingReason.NO_HISTORY` or `NOT_COLLECTED` rather than zero-as-valid; and (g) raises on any required validation failure. A local target must use AirFogSim's own `offloadTask` behavior, so it enters the compute set immediately; a remote target remains a communication task if the simulator does not complete it in the slot. Do not modify `代码/reference/AirFogSim`.

- [ ] **Step 4: Add contract validation for the recorded bundle**

Call `validate_assignment_coo`, `validate_link_outcome`, `validate_rb_outcome`, `validate_masked_field`, and the existing CPU conservation validators against the in-memory records. E2/E3 are valid only when the direct matrix sources were read; otherwise emit zero-filled masked arrays with the exact `MissingReason` and a machine-readable source note.

- [ ] **Step 5: Run adapter tests and the existing regression tests**

Run (from `代码/tests`): `python -m unittest test_airfogsim_single_step_collector_v1.py test_cpu_inner_rule_v1.py test_airfogsim_cpu_inner_rule_v1.py -v`

Expected: PASS; no historical test changes are required.

- [ ] **Step 6: Commit the adapter**

```powershell
git add 代码/src/pi_jwm/airfogsim_single_step_collector_v1.py 代码/tests/test_airfogsim_single_step_collector_v1.py
git commit -m "feat: execute real AirFogSim single-step candidates"
```

### Task 3: Add the canonical CPU-only preflight runner and artifact manifest

**Files:**
- Create: `代码/scripts/run_p2_single_step_collector_preflight_v1.py`
- Create: `代码/tests/test_run_p2_single_step_collector_preflight_v1.py`
- Create at runtime: `代码/artifacts/preflight/pi_jwm_p2_single_step_collector_v1/`

- [ ] **Step 1: Write failing tests for output names and truth flags**

```python
def test_runner_writes_all_required_files_and_blocks_training_claims(tmp_path):
    summary = write_preflight_bundle(tmp_path, fake_candidate_comparison())
    assert set(summary["required_files"]) == {
        "candidate_comparison.json", "action_ledger.json", "transfer_events.json",
        "single_step_graph.json", "resource_bundle.json", "field_mask_audit.json",
        "validation_report.json", "summary.json", "manifest.json",
    }
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status_flags"]["gpu_started"] is False
    assert manifest["status_flags"]["training_eligible"] is False
```

- [ ] **Step 2: Run the focused runner test and verify it fails**

Run (from `代码/tests`): `python -m unittest test_run_p2_single_step_collector_preflight_v1.py -v`

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement deterministic two-candidate CPU preflight**

Use the real AirFogSim example configuration and one fixed seed per candidate. Construct two independent environments from the same seed/configuration, warm them identically until a valid task is in `_waiting_to_offload_tasks`, then run one local-target candidate and one remote-target/RB candidate. Require a common pre-action snapshot hash. Write the nine files in the design document, include source hashes and test command hashes in `manifest.json`, and set `single_step_real_airfogsim_executed` true only after both `env.step()` calls complete and comparison evidence is present. Never write a successful artifact if the candidate difference is not observable; write `validation_report.json` with failure and exit nonzero instead.

- [ ] **Step 4: Run the fake runner test**

Run (from `代码/tests`): `python -m unittest test_run_p2_single_step_collector_preflight_v1.py -v`

Expected: PASS.

- [ ] **Step 5: Run the real short preflight in the dedicated environment**

Run: `conda run -n airfogsim python scripts\run_p2_single_step_collector_preflight_v1.py`

Expected: one completed CPU-only bundle, no GPU process, no locked-test access, and a nonzero exit with a diagnostic report if the real candidate branch cannot produce an observable difference.

- [ ] **Step 6: Verify the generated manifest and hashes**

Run (from `代码`): `python scripts\run_p2_single_step_collector_preflight_v1.py --verify-only` and inspect `validation_report.json` plus `manifest.json`.

Expected: all required files exist, required fields are validated, candidate provenance is direct, and all training/final-method flags remain false.

- [ ] **Step 7: Commit the runner and tests**

```powershell
git add 代码/scripts/run_p2_single_step_collector_preflight_v1.py 代码/tests/test_run_p2_single_step_collector_preflight_v1.py
git commit -m "feat: add P2 single-step preflight bundle runner"
```

### Task 4: Update project evidence and stop boundary

**Files:**
- Modify: `本地计划表.md`
- Modify: `findings.md`
- Modify: `progress.md`
- Modify: `task_plan.md`

- [ ] **Step 1: Record only verified P2 results**

Add the exact artifact path, command, seed/configuration, validation counts, and candidate-difference observation. State explicitly that this is a single-step non-training evidence bundle, not a complete v4 dataset, world-model training run, candidate-rollout planner, or final method.

- [ ] **Step 2: Run the full relevant CPU regression gate**

Run (from `代码`): `python -m unittest discover tests -p "test_*.py" -v` and separately rerun the P1/P2 focused suites if discovery is too broad.

Expected: all touched-contract tests pass; any unrelated pre-existing failure is recorded with its exact traceback.

- [ ] **Step 3: Commit evidence-document updates**

```powershell
git add 本地计划表.md findings.md progress.md task_plan.md
git commit -m "docs: record P2 single-step preflight evidence"
```

## Self-review gates

- The plan never calls the collector a dataset generator or planner.
- Every action enters AirFogSim through its existing scheduler APIs, and invalid RBs are rejected before the simulator's modulo behavior.
- The local/remote candidate difference is measured from simulator state and events; no synthetic task size, rate, or CPU delta is introduced.
- GPU, long training, locked tests, and final-method freezing remain explicitly out of scope until a later theory-code-data consistency audit passes.
