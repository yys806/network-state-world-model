# P2-C Advisor Document Manifest Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind the canonical P2-C research-progress document into the existing P2-C audit manifest so document changes invalidate `--verify-only`.

**Architecture:** Reuse `CANONICAL_SOURCE_PATHS`, `_source_hashes`, `_portable_source_key`, and `verify_audit_bundle`; add no second manifest and no document-specific verification branch. Prove the behavior with one canonical-key test and one temporary-copy tamper test, then rebuild the ignored canonical artifact without changing the audit report, formal-data candidate, or four remaining blockers.

**Tech Stack:** Python 3.10, `unittest`, SHA-256, JSON manifests, PowerShell, existing PI-JWM P2-B/P2-C runners.

---

### Task 1: Add RED tests for advisor-document binding

**Files:**
- Modify: `代码/tests/test_run_p2c_scale_distribution_audit_v1.py`
- Test: `代码/tests/test_run_p2c_scale_distribution_audit_v1.py`

- [x] **Step 1: Add the canonical manifest-key assertion**

Add this module constant after `P2B_BUNDLE`:

```python
P2C_ADVISOR_DOCUMENT = (
    runner.PROJECT_ROOT
    / "文档"
    / "研究进展"
    / "2026-08-14-PI-JWM-P2-C正式数据规模与分布审计.md"
)
P2C_ADVISOR_DOCUMENT_KEY = P2C_ADVISOR_DOCUMENT.relative_to(
    runner.PROJECT_ROOT
).as_posix()
```

Inside `test_publish_and_verify_bind_inputs_outputs_and_portable_sources`, immediately after loading `manifest`, add:

```python
self.assertIn(P2C_ADVISOR_DOCUMENT_KEY, manifest["source_hashes"])
self.assertEqual(
    runner._sha256(P2C_ADVISOR_DOCUMENT),
    manifest["source_hashes"][P2C_ADVISOR_DOCUMENT_KEY],
)
```

- [x] **Step 2: Add the temporary-copy tamper test**

Add this method to `P2CAuditRunnerTests`:

```python
@unittest.skipUnless(P2B_BUNDLE.is_dir(), "canonical P2-B bundle is not available")
def test_p2c_advisor_document_tampering_breaks_source_verification(self):
    temporary_parent = CODE_ROOT / "artifacts" / "tmp"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_parent) as temporary:
        root = Path(temporary)
        copied_document = root / "p2c_advisor_document.md"
        copied_document.write_bytes(P2C_ADVISOR_DOCUMENT.read_bytes())
        source_paths = tuple(
            copied_document if path == P2C_ADVISOR_DOCUMENT else path
            for path in runner.CANONICAL_SOURCE_PATHS
        )
        output = root / "audit"
        runner.publish_audit_bundle(P2B_BUNDLE, output, source_paths=source_paths)

        copied_document.write_text(
            copied_document.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        verification = runner.verify_audit_bundle(
            P2B_BUNDLE,
            output,
            source_paths=source_paths,
        )

        self.assertFalse(verification["passed"])
        self.assertIn("source hash mismatch", verification["errors"])
```

- [x] **Step 3: Run RED and confirm both failures are caused by the missing canonical path**

Run:

```powershell
$env:PYTHONPATH='代码\src;代码\scripts'
$env:PYTHONUTF8='1'
D:\miniconda\envs\airfogsim\python.exe -m unittest `
  代码.tests.test_run_p2c_scale_distribution_audit_v1.P2CAuditRunnerTests.test_publish_and_verify_bind_inputs_outputs_and_portable_sources `
  代码.tests.test_run_p2c_scale_distribution_audit_v1.P2CAuditRunnerTests.test_p2c_advisor_document_tampering_breaks_source_verification -v
```

Expected: two assertion failures; the first reports the missing portable document key, and the second reports `verification["passed"]` remained true because the copied document was not substituted into `CANONICAL_SOURCE_PATHS`.

### Task 2: Add the minimal canonical source dependency

**Files:**
- Modify: `代码/scripts/run_p2c_scale_distribution_audit_v1.py`
- Test: `代码/tests/test_run_p2c_scale_distribution_audit_v1.py`

- [x] **Step 1: Add exactly one source path**

Append this entry to `CANONICAL_SOURCE_PATHS` after the P2-B design document:

```python
PROJECT_ROOT
    / "文档"
    / "研究进展"
    / "2026-08-14-PI-JWM-P2-C正式数据规模与分布审计.md",
```

Do not change `_source_hashes`, `_portable_source_key`, publishing, verification, report generation, or status flags.

- [x] **Step 2: Run the two RED tests and the complete P2-C test pair**

Run:

```powershell
$env:PYTHONPATH='代码\src;代码\scripts'
$env:PYTHONUTF8='1'
D:\miniconda\envs\airfogsim\python.exe -m unittest `
  代码.tests.test_p2c_scale_distribution_audit_v1 `
  代码.tests.test_run_p2c_scale_distribution_audit_v1 -v
```

Expected: 9 tests pass, including canonical document-key equality and temporary-copy tamper rejection.

- [x] **Step 3: Run static verification**

Run:

```powershell
D:\miniconda\envs\airfogsim\python.exe -m py_compile `
  代码\scripts\run_p2c_scale_distribution_audit_v1.py `
  代码\tests\test_run_p2c_scale_distribution_audit_v1.py
python -m ruff check `
  代码\scripts\run_p2c_scale_distribution_audit_v1.py `
  代码\tests\test_run_p2c_scale_distribution_audit_v1.py
git diff --check -- `
  代码/scripts/run_p2c_scale_distribution_audit_v1.py `
  代码/tests/test_run_p2c_scale_distribution_audit_v1.py
```

Expected: compilation succeeds, Ruff reports `All checks passed!`, and diff check emits no error.

### Task 3: Rebuild and atomically promote the P2-C canonical artifact

**Files:**
- Generate: `代码/artifacts/audit/pi_jwm_p2c_scale_distribution_audit_v1_advisor_doc_binding_candidate_20260814/`
- Archive: `代码/artifacts/audit/pi_jwm_p2c_scale_distribution_audit_v1_pre_advisor_doc_binding_20260814/`
- Replace: `代码/artifacts/audit/pi_jwm_p2c_scale_distribution_audit_v1/`

- [x] **Step 1: Publish a fresh candidate and verify it**

Run from `代码/`:

```powershell
$env:PYTHONUTF8='1'
D:\miniconda\envs\airfogsim\python.exe scripts\run_p2c_scale_distribution_audit_v1.py `
  --bundle artifacts\preflight\pi_jwm_p2_full_dual_graph_collector_v1 `
  --output-dir artifacts\audit\pi_jwm_p2c_scale_distribution_audit_v1_advisor_doc_binding_candidate_20260814
D:\miniconda\envs\airfogsim\python.exe scripts\run_p2c_scale_distribution_audit_v1.py `
  --bundle artifacts\preflight\pi_jwm_p2_full_dual_graph_collector_v1 `
  --output-dir artifacts\audit\pi_jwm_p2c_scale_distribution_audit_v1_advisor_doc_binding_candidate_20260814 `
  --verify-only
```

Expected: publication reports `audit_status=blocked`, exactly four blocking reasons and `formal_data_approved=false`; verify reports `passed=true`.

- [x] **Step 2: Compare candidate and canonical semantics**

Run:

```powershell
git diff --no-index -- `
  artifacts\audit\pi_jwm_p2c_scale_distribution_audit_v1\p2c_scale_distribution_audit_v1.json `
  artifacts\audit\pi_jwm_p2c_scale_distribution_audit_v1_advisor_doc_binding_candidate_20260814\p2c_scale_distribution_audit_v1.json
git diff --no-index -- `
  artifacts\audit\pi_jwm_p2c_scale_distribution_audit_v1\p2c_formal_data_config_candidate_v1.json `
  artifacts\audit\pi_jwm_p2c_scale_distribution_audit_v1_advisor_doc_binding_candidate_20260814\p2c_formal_data_config_candidate_v1.json
```

Expected: both commands show no content difference. Inspect the manifest separately and require exactly one new portable source key plus source-hash changes caused by the modified runner/test/plan/document; no absolute or `.worktrees/` key is allowed.

- [x] **Step 3: Validate move targets and promote without deleting evidence**

Resolve the audit root, canonical, candidate, and archive paths. Require canonical and candidate parents to equal the audit root and require the archive path not to exist. Then move canonical to `pi_jwm_p2c_scale_distribution_audit_v1_pre_advisor_doc_binding_20260814` and candidate to the canonical name with PowerShell `Move-Item -LiteralPath`.

- [x] **Step 4: Verify the promoted canonical**

Run the P2-C `--verify-only` command against `artifacts/audit/pi_jwm_p2c_scale_distribution_audit_v1`.

Expected: `{"errors": [], "passed": true}`.

### Task 4: Run the complete evidence gate and persist status

**Files:**
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`
- Modify only if status wording requires synchronization: `本地计划表.md`
- Modify only if status wording requires synchronization: `D:/禹尧珅/人工智能知识库/北大科研/PIJWM/8.12之后推进.md`

- [x] **Step 1: Re-run P2-B and P2-C canonical verification**

Run both existing `--verify-only` commands with `PYTHONUTF8=1`.

Expected: both return exit code 0 and `passed=true`.

- [x] **Step 2: Recompute all 83 AirFogSim manifest hashes**

Read P2-B `manifest.json`, select keys beginning with `代码/reference/AirFogSim/`, and recompute SHA-256 from the project root.

Expected: expected=83, matched=83, mismatched=0.

- [x] **Step 3: Run the complete P1/P2 focused suite**

Run the same 17 unittest modules recorded in the P2-C source-recovery audit with `PYTHONUTF8=1`.

Expected: 158 tests run and all pass. Do not describe a GBK import failure as a model/test pass.

- [x] **Step 4: Record the exact result and commit only scoped tracked files**

Update planning/evidence notes with RED output, GREEN output, candidate/canonical paths, verify results, 83/83 hashes, focused test count, and unchanged four blockers. Stage only:

```text
docs/superpowers/plans/2026-08-14-p2c-advisor-document-manifest-binding.md
代码/scripts/run_p2c_scale_distribution_audit_v1.py
代码/tests/test_run_p2c_scale_distribution_audit_v1.py
```

Include planning files only if their existing dirty content can be preserved and the staged diff contains only this task's appended status. Never stage unrelated user changes.

Commit message:

```text
audit: bind P2-C progress document to manifest
```

Before committing, run `git diff --cached --check` and inspect `git diff --cached --name-status`.

## Execution Evidence

- RED: both targeted tests failed for the intended missing binding: the portable document key was absent and document tampering left verification incorrectly passing.
- GREEN: the two P2-C modules passed 9/9 after adding one canonical source path; `py_compile`, Ruff, and `git diff --check` passed.
- Candidate publication retained `audit_status=blocked`, the same four blocking reasons, and `formal_data_approved=false`; candidate `--verify-only` passed.
- The audit report and formal-data candidate were byte-identical to the previous canonical. The manifest added exactly the advisor-document key, changed only the runner/test hashes among common source keys, and contained no absolute or `.worktrees/` keys.
- The previous canonical is preserved as `pi_jwm_p2c_scale_distribution_audit_v1_pre_advisor_doc_binding_20260814`; the promoted canonical passed `--verify-only`.
- Final gate before documentation updates: P2-B/P2-C verification passed, AirFogSim dependencies matched 83/83, and the focused suite passed 159/159 under `PYTHONUTF8=1`. The plan's estimate of 158 increased by one because this change adds one tamper regression test.
