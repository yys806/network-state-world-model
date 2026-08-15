# PI-JWM Repository Governance and GitHub Publish Design

Date: 2026-08-15

## Goal

Make the migrated PI-JWM repository easier to continue, read, verify, and maintain without changing research semantics, executable behavior, evidence bytes, or historical provenance. Publish the resulting verified repository state to the existing GitHub `origin`.

## Current Constraints

- PI-JWM is the framework; AirFogSim remains a reference simulator and data source.
- The working tree contains extensive pre-existing modifications, deletions, and untracked files. They must be classified and preserved rather than reset or overwritten.
- `main` is 167 commits ahead of `origin/main`.
- Three copied linked worktrees still contain pre-migration Git metadata. The P2 ledger worktree also contains an AirFogSim junction pointing to the old root.
- Generated artifacts, third-party AirFogSim contents, local literature, meeting files, caches, and temporary files must not be uploaded accidentally.
- Formal data generation, GPU training, locked-test access, model changes, and experiment-result reinterpretation are outside this task.

## Approaches Considered

### A. Conservative governance in place (selected)

Repair path metadata, improve repository indexes and ignore rules, classify existing changes, preserve historical files in place, verify the repository, and publish explicit commits. This has the lowest risk of breaking evidence links or invalidating historical references.

### B. Physical directory normalization

Move historical documents, scripts, and artifacts into a newly designed directory hierarchy. This may look cleaner but would invalidate paths in manifests, plans, reports, and reproducibility commands. It is rejected for this pass.

### C. Clean export repository

Create a new minimal repository containing only current source, tests, and selected documents. This would produce a small public tree but sever commit history and omit evidence needed for PI-JWM continuity. It is rejected unless requested as a separate future deliverable.

## Selected Scope

### 1. Migration repair

- Relocate registered Git worktree administrative metadata from the old `D:/shen/网络组` root to `D:/shen/PKU/PIJWM`.
- Repair copied worktree `.git` pointers.
- Repair the P2 ledger AirFogSim junction to the current reference checkout.
- Preserve worktree branch heads, worktree contents, and evidence bytes.
- Do not prune or delete the three existing worktrees.

### 2. Repository navigation

- Make the root `README.md` the concise entry point for identity, directory ownership, current stage, safe commands, and authoritative continuation documents.
- Restore or replace directory-level indexes only where they materially improve navigation.
- Clearly label current, historical, generated, local-only, and third-party material.
- Keep `本地计划表.md` as the sole local overview plan and the 2026-08-15 continuation note as the current handoff source.

### 3. Hygiene and ignore policy

- Normalize `.gitignore` formatting and document why each local-only category is ignored.
- Keep `.worktrees/`, caches, temporary directories, generated artifacts, third-party AirFogSim, local literature, and meeting binaries out of Git unless an existing tracked governance file explicitly belongs in the repository.
- Scan tracked and candidate files for obvious credentials, oversized files, caches, and generated duplicates before staging.
- Do not delete ignored local data.

### 4. Existing dirty-state classification

Classify every tracked deletion, tracked modification, and untracked candidate into one of:

- preserve and publish;
- preserve locally but ignore;
- historical deletion intentionally represented in Git;
- unresolved and therefore excluded from the governance commit.

No existing deletion is treated as intentional solely because it appears in `git status`.

### 5. Evidence and theory protection

- Do not edit machine evidence or historical experiment output to make verification pass.
- After migration repair, rerun P2-B/P2-C verification from their intended worktree.
- Keep the three P2-C formal-data blockers unchanged.
- Do not describe R6 as a candidate-rollout planner.
- Any documentation cleanup must retain the distinction between current authority and historical evidence.

## Commit Strategy

Use small, reviewable commits rather than one repository-wide snapshot:

1. governance design;
2. worktree migration repair and repository navigation/hygiene;
3. classified current source, tests, and research documents where verified;
4. P2 ledger evidence closure only if its existing acceptance gates pass;
5. final plan/handoff synchronization.

Only intended paths are staged for each commit. No force push is allowed. Push the current `main` history to the configured `origin` after fresh verification and a final staged-content/secret/size audit.

## Verification

The publish gate requires:

- `git status`, staged diff, and commit list reviewed;
- no unexpected tracked files under ignored local-only roots;
- no obvious credential matches in files being published;
- no new oversized blob that exceeds GitHub limits;
- repaired `git worktree list --porcelain` paths with no migration-related `prunable` entries;
- all copied worktree `.git` pointers resolve;
- the P2 ledger AirFogSim config path resolves;
- P2-B v2 and P2-C v2 verification rerun after relocation;
- focused CPU/unit tests selected from the affected governance and P1/P2 paths;
- Python compilation for published source and scripts;
- `git diff --check` succeeds;
- no GPU use, formal-data generation, or locked-test access.

If a pre-existing test or verification failure remains, it is reported and the affected claim is withheld. A push occurs only when the exact commit set and remaining limitations are known.

## Non-Goals

- No model, collector, policy, metric, or experiment behavior change.
- No formal v4 data generation or model training.
- No locked-test access.
- No repository history rewrite, force push, bulk deletion, or artifact recompression.
- No physical reorganization that changes paths embedded in manifests or research records.
