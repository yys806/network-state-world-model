# R6 CPU Paired Closed-Loop Baseline Implementation Plan

> **For agentic workers:** Execute the tasks in order with TDD and update `task_plan.md` after each phase.

**Goal:** Run reproducible same-scenario/same-seed CPU policy pairs and produce auditable closed-loop deltas before any GPU strategy training.

**Architecture:** Keep the formal AirFogSim runtime and R1/R2 protocol unchanged by adding an optional allocator factory. A new CPU policy module implements the three existing rules plus a deterministic local-search heuristic; a runner selects only non-locked specs, runs all policies on identical scenario/seed configurations, computes pair identities and metrics, and writes a hash-bound bundle.

**Tech Stack:** Python standard library, existing AirFogSim runtime, NumPy, existing PI-JWM metric and resource validators, `unittest`.

---

### Task 1: Freeze the paired protocol

**Files:**
- Create: `文档/研究进展/2026-08-07-PI-JWM-R6-CPU配对闭环设计.md`
- Modify: `task_plan.md`, `本地计划表.md`, `progress.md`, `findings.md`, `PIJWM推进.md`

- [ ] Add the four-arm policy list, non-locked split boundary, same scenario/seed/config identity, deterministic local-search neighborhood, metric semantics, and no-GPU/no-locked-test gate.
- [ ] Record all commands, seeds, budgets, and output file names in the plan before execution.

### Task 2: Add failing tests for policy and pairing contracts

**Files:**
- Create: `代码/tests/test_r6_cpu_paired_policy.py`
- Create: `代码/tests/test_run_r6_cpu_paired_closed_loop.py`

- [ ] Test equal-share/deadline weights, deterministic local-search candidate order, capacity projection, negative/NaN rejection, and unknown policy rejection.
- [ ] Test that pair keys require identical scenario/seed/config fingerprints, reject locked-test, and preserve `not_computable` when a counterfactual is absent.
- [ ] Run the two test files and observe the expected `ModuleNotFoundError` before production implementation.

### Task 3: Implement the CPU policy module

**Files:**
- Create: `代码/src/pi_jwm/r6_cpu_paired_policy.py`

- [ ] Implement `PairedCpuPolicyAllocator` with `policy_id`, `allocate`, deterministic weights, bounded local-search candidates, and allocation audit rows.
- [ ] Implement `project_cpu_allocations` so every returned allocation is finite, non-negative, and sums no higher than node capacity.

### Task 4: Make the existing runtime injectable without changing defaults

**Files:**
- Modify: `代码/scripts/formal_airfogsim_runtime_v1.py`
- Modify: `代码/tests/test_formal_airfogsim_runtime_v1.py`

- [ ] Add optional `allocator_factory`; default remains `CpuPolicyAllocator` and all existing formal-v1 behavior remains unchanged.
- [ ] Add a test proving a custom allocator is called and default policy behavior is unchanged.

### Task 5: Implement paired runner and bundle writer

**Files:**
- Create: `代码/scripts/run_r6_cpu_paired_closed_loop.py`
- Create: `代码/src/pi_jwm/r6_cpu_paired_analysis.py`

- [ ] Select the 54 non-locked specs from the frozen formal protocol; never construct a locked-test path.
- [ ] Run four policies per pair with identical scenario/seed, record input fingerprints, actions, runtime summaries, and canonical metric rows.
- [ ] Compute paired deltas only for complete pair groups; write summary, pair audit, metrics, failures, README, and SHA-256 manifest into a new output directory.

### Task 6: CPU smoke and formal validation

**Files:**
- Modify: `task_plan.md`, `progress.md`, `findings.md`, `本地计划表.md`, `PIJWM推进.md`

- [ ] Run one validation pair smoke and the focused tests first.
- [ ] Run the complete non-locked CPU pairing only after smoke passes; preserve failed runs in a separate failure file.
- [ ] Independently recompute output hashes and verify all pair keys, no hard violations, and no locked-test access.
