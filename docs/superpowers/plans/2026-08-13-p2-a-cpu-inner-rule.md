# P2-A CPU Inner Rule Implementation Plan

> Scope: implement and verify the frozen P1-A CPU boundary only. This plan does not authorize v4 dataset reconstruction, long training, GPU use, locked-test access, or method freezing.

**Goal:** Implement a deterministic, work-conserving, capacity-capped equal-sharing CPU inner rule that is called after each candidate action's communication update, plus an AirFogSim callback adapter and machine-readable preflight evidence.

**Frozen boundary:** The learned/planned action contains offload and RB decisions only. CPU is neither an action dimension nor a candidate-independent constant. For each rollout candidate and each rollout step, the CPU rule consumes that candidate's post-communication compute-task set.

**Core rule:** For task `m` on node `i`, let `d_m = remaining_work_m / slot_seconds`. Choose a common water level `lambda_i` such that

`sum_m min(d_m, lambda_i) = min(node_capacity_i, sum_m d_m)`

and return `f_m = min(d_m, lambda_i)` in stable `task_id` order.

---

## Task 1: RED tests for the pure rule

**Files:**

- Create: `代码/tests/test_cpu_inner_rule_v1.py`
- Target: `代码/src/pi_jwm/cpu_inner_rule_v1.py`

Add tests for empty input, zero capacity, demand below capacity, demand above capacity, unequal demands, multiple nodes, stable ordering, deterministic repetition, invalid numeric values, duplicate task IDs, missing capacity, conservation, and candidate-sensitive task sets.

Run and confirm RED because the production module does not exist yet:

```powershell
cd D:\shen\网络组\代码
python -m unittest tests.test_cpu_inner_rule_v1 -v
```

## Task 2: Implement the pure rule

**Files:**

- Create: `代码/src/pi_jwm/cpu_inner_rule_v1.py`

Implement immutable input/output dataclasses and:

```python
def allocate_work_conserving_cpu(
    tasks: Sequence[CpuTaskDemand],
    node_capacities: Mapping[str, float],
    slot_seconds: float,
) -> CpuRuleDecision:
    ...
```

Requirements:

- Validate all identifiers and numeric values; fail fast on missing node capacity.
- Sort nodes and task IDs to guarantee stable output.
- Include zero-demand tasks in the auditable decision with zero allocation.
- Never allocate negative/non-finite values, exceed task demand, or exceed node capacity.
- If aggregate demand is at least capacity, use the full capacity within floating-point tolerance.
- Apply only bounded deterministic residual correction within task headroom.

Run the RED test command until GREEN.

## Task 3: RED tests and implementation for the AirFogSim adapter

**Files:**

- Create: `代码/tests/test_airfogsim_cpu_inner_rule_v1.py`
- Create: `代码/src/pi_jwm/airfogsim_cpu_inner_rule_v1.py`

The adapter must expose a callback compatible with `TaskManager.computeTasks`, extract task ID, total/computed CPU work and current compute node from AirFogSim task objects, read node CPU capacity from the environment, invoke the pure rule, and return `{task_id: allocated_cpu}`.

Test with AirFogSim's real `Task` class plus a minimal environment/node fixture. Verify that the callback output exactly equals the pure rule for the same extracted inputs. Also verify node-assignment mismatch and missing node capacity are rejected. This is a callback-interface integration test, not evidence of a full AirFogSim trajectory run.

Run:

```powershell
cd D:\shen\网络组\代码
python -m unittest tests.test_airfogsim_cpu_inner_rule_v1 -v
```

## Task 4: Machine-readable preflight bundle

**Files:**

- Create: `代码/scripts/run_cpu_inner_rule_preflight_v1.py`
- Create: `代码/tests/test_run_cpu_inner_rule_preflight_v1.py`
- Generate: `代码/artifacts/preflight/pi_jwm_cpu_inner_rule_v1/rule_contract.json`
- Generate: `代码/artifacts/preflight/pi_jwm_cpu_inner_rule_v1/sample_cases.csv`
- Generate: `代码/artifacts/preflight/pi_jwm_cpu_inner_rule_v1/rejected_records.csv`
- Generate: `代码/artifacts/preflight/pi_jwm_cpu_inner_rule_v1/summary.json`
- Generate: `代码/artifacts/preflight/pi_jwm_cpu_inner_rule_v1/manifest.json`

The preflight must:

- Execute deterministic contract fixtures for all boundary cases and candidate-sensitive task sets.
- Execute the real AirFogSim `Task` callback-interface parity fixture.
- Record intentionally invalid fixtures as rejected and never training-eligible.
- Refuse to overwrite an existing evidence directory.
- Publish atomically only after all checks pass.
- Hash the design input, implementation files, test files, and every output except the manifest itself.
- State explicitly that P2-A is complete only at the CPU-rule/callback preflight level; v4 collector, v4 dataset, training, GPU, locked test, and final method remain incomplete/unaccessed.

Run tests first, then generate the canonical bundle once:

```powershell
cd D:\shen\网络组\代码
python -m unittest tests.test_run_cpu_inner_rule_preflight_v1 -v
python scripts\run_cpu_inner_rule_preflight_v1.py
```

## Task 5: Regression and documentation truthfulness

**Files:**

- Modify: `本地计划表.md`
- Modify: `D:\禹尧珅\人工智能知识库\北大科研\PIJWM\8.12之后推进.md`
- Modify only if implementation facts change a fixed statement: `D:\禹尧珅\人工智能知识库\北大科研\PIJWM\PIJWM主文档.md`

Run focused and relevant regression tests without GPU or locked data:

```powershell
cd D:\shen\网络组\代码
python -m unittest tests.test_cpu_inner_rule_v1 tests.test_airfogsim_cpu_inner_rule_v1 tests.test_run_cpu_inner_rule_preflight_v1 -v
python -m unittest tests.test_information_edge_contract_v4 tests.test_information_edge_audit_v4 tests.test_information_edge_contract_v4_artifacts -v
```

Audit code, documents, and evidence for forbidden overclaims. The only permitted completion claim is: `P2-A CPU inner rule and callback preflight verified`. Do not describe P2 dataset reconstruction, candidate-rollout planning, model training, or the final method as complete.
