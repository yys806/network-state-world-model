# PI-JWM Repository Governance and GitHub Publish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the migrated repository, make its current authority and directory ownership easy to understand, publish the verified current source and evidence-governance state, and push `main` to the existing GitHub origin without changing experiment behavior.

**Architecture:** Work in the real main checkout because the task includes repairing its linked worktrees and classifying its dirty state. Use official Git worktree repair where possible, minimal PowerShell junction replacement for the stale third-party link, small documentation/hygiene commits, explicit publish-set audits, and fresh verification before a non-force push.

**Tech Stack:** Git worktrees, PowerShell, Python 3 in the `airfogsim` Conda environment, `unittest`, Markdown, PI-JWM manifest verifiers.

---

### Task 1: Freeze the publish boundary and baseline

**Files:**
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`
- Reference: `docs/superpowers/specs/2026-08-15-repository-governance-and-github-publish-design.md`

- [ ] **Step 1: Record this governance phase in the existing planning files**

Add a dated section that fixes the selected conservative scope, records the dirty-tree baseline, and states that existing deletions require classification before staging.

- [ ] **Step 2: Capture the Git and filesystem baseline**

Run:

```powershell
git status --short --branch
git remote -v
git log --oneline --decorate -20
git worktree list --porcelain
git diff --name-status
git ls-files -o --exclude-standard
```

Expected: `main` is ahead of `origin/main`; three migrated worktrees still reference the former root before Task 2; no command mutates files.

- [ ] **Step 3: Classify ignored roots and candidate publish files**

Run:

```powershell
git check-ignore -v .worktrees tmp .ruff_cache 代码/artifacts 代码/reference/AirFogSim 文档/文献 文档/组会
git ls-files -o --exclude-standard | Group-Object { Split-Path $_ -Parent } | Sort-Object Count -Descending
```

Expected: local worktrees, generated artifacts, third-party simulator contents, caches, literature, and meeting binaries are excluded; source, tests, governance documents, and continuation notes remain candidates.

### Task 2: Repair the migrated linked worktrees

**Files:**
- Modify outside tracked content: `.git/worktrees/*` administrative metadata
- Modify outside tracked content: `.worktrees/*/.git`
- Replace junction only: `.worktrees/p2-action-attempt-ledger-v1/代码/reference/AirFogSim`

- [ ] **Step 1: Verify exact repair targets**

Run:

```powershell
$root = (Resolve-Path '.').Path
$targets = @(
  (Resolve-Path '.worktrees/p2-action-attempt-ledger-v1').Path,
  (Resolve-Path '.worktrees/p2-c-scale-distribution-audit').Path,
  (Resolve-Path '.worktrees/sparse-event-diagnostic-v2').Path
)
$targets | ForEach-Object { if (-not $_.StartsWith($root)) { throw "worktree outside repository root: $_" } }
git worktree list --porcelain
```

Expected: all three copied directories resolve under `D:\shen\PKU\PIJWM\.worktrees` while Git still reports the former root.

- [ ] **Step 2: Repair Git administrative paths using Git's worktree repair command**

Run from the main checkout:

```powershell
git worktree repair `
  '.worktrees/p2-action-attempt-ledger-v1' `
  '.worktrees/p2-c-scale-distribution-audit' `
  '.worktrees/sparse-event-diagnostic-v2'
```

Expected: `git worktree list --porcelain` reports the current absolute paths and no migration-related `prunable` message. Each copied `.git` file points into `D:/shen/PKU/PIJWM/.git/worktrees/`.

- [ ] **Step 3: Replace only the stale AirFogSim junction**

Resolve and verify the link itself, its old target, and the intended new target. Then remove the junction entry without recursion and recreate it:

```powershell
$link = (Resolve-Path '.worktrees/p2-action-attempt-ledger-v1/代码/reference').Path + '\AirFogSim'
$target = (Resolve-Path '代码/reference/AirFogSim').Path
$item = Get-Item -LiteralPath $link -Force
if ($item.LinkType -ne 'Junction') { throw "not a junction: $link" }
if (-not $target.StartsWith((Resolve-Path '.').Path)) { throw "target outside repository: $target" }
Remove-Item -LiteralPath $link -Force
New-Item -ItemType Junction -Path $link -Target $target | Out-Null
```

Expected: the link target is `D:\shen\PKU\PIJWM\代码\reference\AirFogSim` and `examples/config.yaml` resolves from both main and ledger worktrees.

- [ ] **Step 4: Rerun the two migration-sensitive verifiers**

Run with `PYTHONUTF8=1` and the ledger worktree's `代码/src` on `PYTHONPATH`:

```powershell
python run_p2_full_dual_graph_collector_preflight_v2.py --output-dir '<main>/代码/artifacts/preflight/pi_jwm_p2_full_dual_graph_collector_v2_candidate_20260814' --verify-only
python run_p2c_scale_distribution_audit_v2.py --bundle '<main>/代码/artifacts/preflight/pi_jwm_p2_full_dual_graph_collector_v2_candidate_20260814' --output-dir '<main>/代码/artifacts/audit/pi_jwm_p2c_scale_distribution_audit_v2_pre_document_closure_20260814' --verify-only
```

Expected: both return `passed: true`. If P2-C remains different, compare stored and fresh source-root closure before changing any evidence file.

### Task 3: Improve repository navigation and hygiene

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`
- Create: `文档/README.md`
- Modify: `代码/artifacts/README.md`
- Reference: `AGENTS.md`
- Reference: `本地计划表.md`
- Reference: `新对话接续说明_20260815.md`

- [ ] **Step 1: Rewrite the root entry point around the current authority**

The root README must state PI-JWM identity, current P2-C/P6 boundaries, directory ownership, environment commands using the current root, safe verification commands, and links to the local overview and latest continuation note. Remove stale `D:\shen\网络组`, deleted `文档/项目说明`, old v11-as-current, and old 60-seed launch guidance.

- [ ] **Step 2: Restore the document index without restoring deleted historical content**

Create `文档/README.md` that classifies `研究进展`, `组会`, and `文献`; identifies current versus historical material; and points to root authority files. It must not claim deleted `项目说明` or `工程治理` directories still exist.

- [ ] **Step 3: Update artifact guidance to match the current evidence hierarchy**

Keep artifacts local by default. Document canonical `audit`, `preflight`, `formal_training`, `analysis`, `reports`, `literature`, and `tmp` roles; explicitly separate machine evidence from historical model results and temporary copies.

- [ ] **Step 4: Normalize ignore rules**

Preserve existing ignore semantics, add any discovered local-only planning/render/cache paths, normalize line endings, and ensure the tracked `代码/artifacts/README.md` and `代码/artifacts/manifests/**` exceptions remain effective.

- [ ] **Step 5: Verify navigation and hygiene changes**

Run:

```powershell
rg -n -S 'D:\\shen\\网络组|文档/项目说明|最终 v11|直接续训100k' README.md 文档/README.md 代码/artifacts/README.md
git check-ignore -v .worktrees tmp .ruff_cache 代码/artifacts/example.json 代码/artifacts/README.md
git diff --check -- README.md .gitignore 文档/README.md 代码/artifacts/README.md
```

Expected: no stale root or removed-directory references; local-only paths are ignored; artifact governance files remain publishable; whitespace check passes.

- [ ] **Step 6: Commit navigation and hygiene**

Stage only the four governance files and commit:

```powershell
git add -- README.md .gitignore 文档/README.md 代码/artifacts/README.md
git diff --cached --name-status
git commit -m "docs: clarify PI-JWM repository navigation"
```

### Task 4: Classify and publish the current source, tests, and research records

**Files:**
- Candidate source: `代码/src/pi_jwm/*.py`
- Candidate scripts: `代码/scripts/**`
- Candidate tests: `代码/tests/**`
- Candidate research documents: `文档/研究进展/**`
- Candidate plans/specs: `docs/superpowers/**`
- Candidate root records: `AGENTS.md`, `本地计划表.md`, `新对话接续说明_*.md`, `task_plan.md`, `findings.md`, `progress.md`

- [ ] **Step 1: Build an explicit candidate inventory**

Run `git status --short` and classify every path by ownership. Exclude ignored artifacts, third-party contents, binaries under meeting/literature roots, caches, and the nested empty `代码/代码/artifacts/experiments` path. Do not stage existing deletions until reference and supersession checks justify them.

- [ ] **Step 2: Audit deleted tracked documents**

For each deleted path, search current tracked and candidate files for references. Classify as superseded historical deletion only when its current authority is identified or all remaining references are removed. Otherwise leave it unstaged and report it rather than restoring or committing it implicitly.

- [ ] **Step 3: Scan candidate files for secrets and GitHub size hazards**

Run credential-pattern scans over candidate text files without printing matched secret values, and enumerate candidate file sizes. Reject `.env`, private keys, API tokens, files at or above 100 MiB, and unexplained binaries from the publish set.

- [ ] **Step 4: Verify Python candidates before staging**

Run:

```powershell
python -m compileall -q 代码/src 代码/scripts 代码/tests
$env:PYTHONUTF8='1'
$env:PYTHONPATH='<root>/代码/src'
python -m unittest discover -s 代码/tests -p 'test_*.py'
```

Expected: compilation succeeds. The full test result is recorded exactly; pre-existing failures are not hidden. At minimum, the focused P1/P2 and modified runtime tests must pass before their files are published.

- [ ] **Step 5: Stage by ownership and review the exact diff**

Use explicit `git add -- <paths>` groups. Review `git diff --cached --stat`, `git diff --cached --name-status`, `git diff --cached --check`, large blobs, and secret scan results before each commit. Never use `git add -A` for this task.

- [ ] **Step 6: Commit verified groups**

Use separate commits for source/tests, research documentation, and planning/handoff records so later review can distinguish executable changes from narrative updates.

### Task 5: Close the ledger evidence document only if all existing gates pass

**Files:**
- Modify in ledger branch: `文档/研究进展/2026-08-14-PI-JWM-P2联合动作Attempt-Reject-Ledger-v1实施与证据.md`
- Generate local ignored artifact: `代码/artifacts/audit/pi_jwm_p2c_scale_distribution_audit_v2_candidate_20260814/`
- Modify after merge: `本地计划表.md`

- [ ] **Step 1: Confirm the ledger worktree is clean except for the known whitespace-cleaned evidence document**

Run `git status --short`, `git diff --check`, and inspect the document diff. Do not include unrelated files.

- [ ] **Step 2: Update and commit the evidence closure**

Record the verified P2-C v2 counts, three remaining blockers, and the fact that the final candidate is generated only after this commit. Stage and commit only this document.

- [ ] **Step 3: Generate and verify the final candidate in a new directory**

Run the documented v2 audit command without `--verify-only`, then rerun it with `--verify-only`. Compare audit JSON and candidate config byte-for-byte with the pre-document candidate; only the manifest source binding may change.

- [ ] **Step 4: Run the focused 203-test and four-verifier acceptance gate**

Record exact counts. Keep `formal_data_approved=false`, `training_eligible=false`, `gpu_started=false`, and `locked_test_accessed=false`, with the same three formal-data blockers.

- [ ] **Step 5: Merge the ledger branch only after acceptance**

Merge non-destructively into `main`, rerun the focused verification on the merged result, and leave both historical and final candidate artifact directories intact locally.

### Task 6: Final audit and GitHub push

**Files:**
- Modify: `本地计划表.md`
- Modify: `新对话接续说明_20260815.md` only where verified migration or closure facts changed
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

- [ ] **Step 1: Synchronize current status documents**

Update only verified facts: repaired paths, exact verification outcomes, committed publish set, remaining blockers, and GitHub target. Do not rewrite historical sections.

- [ ] **Step 2: Run the final verification gate**

Run fresh:

```powershell
git diff --check
git status --short --branch
git worktree list --porcelain
python -m compileall -q 代码/src 代码/scripts 代码/tests
python -m unittest discover -s 代码/tests -p 'test_*.py'
git log --oneline --decorate origin/main..main
```

Also rerun the final focused P1/P2 verifiers and repeat the staged/tracked secret and file-size audit.

- [ ] **Step 3: Fetch and check push topology**

Run:

```powershell
git fetch origin
git status --short --branch
git rev-list --left-right --count origin/main...main
```

Expected: no unexpected remote-only commits. If remote diverged, stop and reconcile without force pushing.

- [ ] **Step 4: Push without rewriting history**

Run:

```powershell
git push origin main
```

Expected: exit code 0 and `origin/main` advances to the verified local `main` commit.

- [ ] **Step 5: Verify the remote reference**

Run `git ls-remote origin refs/heads/main` and compare it with `git rev-parse main`. Report the commit, tests, remaining local-only files, and remaining PI-JWM research blockers.
