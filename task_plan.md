# PI-JWM v11 Selector GPU Iteration Plan

## Goal

Use train/calibration/validation only to diagnose and improve the candidate selector, freeze one defensible method, then evaluate the locked method once on external holdout seeds 60-69. Historical matched test seeds 18-19 remain locked.

## Acceptance Gates

- Validation must improve over ranked default RMSE 233.7162005 on a clear majority of seeds.
- Selected candidates must have positive realized-benefit precision and a controlled negative-selection rate.
- The method must retain task-energy safety and use only deployable features at inference.
- External holdout is opened only after configuration freeze.
- Any result reported to the advisor is labelled deployable, sample_oracle, diagnostic_only, or external_holdout.

## Phases

| Phase | Status | Deliverable |
|---|---|---|
| 1. Reconstruct the completed GPU run | complete | 36 checkpoints and formal validation result available locally |
| 2. Attribute failure to ranking, uncertainty, Pareto, or defer | complete | 12-config x 6-policy validation audit |
| 3. Form and test one root-cause hypothesis | complete | Opportunity is identifiable; candidate ranking needs token-level interactions |
| 4. Implement the selected method with TDD | complete | Tests, reusable selector code, runner changes |
| 5. Local smoke and full validation checks | complete | Formal phase-selector reproduction plus 722 main and 84 script tests |
| 6. Sync and run the necessary GPU experiment | complete | CUDA smoke plus no-phase/phase-aware 3-seed probes completed |
| 7. Freeze on validation and evaluate external holdout once | complete | A gate not met; external was correctly kept locked and unaccessed |
| 8. Update report artifacts and PPT data placeholders | complete | CSV/JSON/NPZ/figures, manifest, and six-page PPT planning text updated |

## 2026-08-14 P2-C Advisor-Document Manifest Binding

| Task | Status | Evidence |
|---|---|---|
| RED: prove the progress document is not bound | complete | 2/2 focused tests failed for the expected missing-key and tamper-not-detected reasons |
| GREEN: add one canonical source path | complete | P2-C test pair passed 9/9; Python compile, Ruff, and diff checks passed |
| Rebuild and promote canonical audit | complete | Core audit/config JSON stayed byte-identical; old canonical was archived; promoted `--verify-only` passed |
| Complete evidence gate | complete | P2-B/P2-C verify passed, AirFogSim 83/83 matched, focused suite passed 159/159 |

The four P2-C blockers remain unchanged: `action_rejection_rate_not_observed`, `scenario_matrix_not_frozen`, `formal_scale_not_frozen`, and `formal_split_not_frozen`. No GPU task, formal trajectory generation, or locked-test access is authorized by this closure.

## Outcome

- Best validation result: B-grade RMSE 207.5399 versus 233.7162 ranked baseline.
- All 10 validation seeds improved; positive execution precision was 93.85% with zero Pareto violations.
- The pre-registered A-grade RMSE <200 gate was not met. Configuration remains a v11 candidate, and external seeds 60-69 remain unaccessed.

## Fixed Protocol

- Train seeds: 0-15, 20-43.
- Calibration seeds: 44-49.
- Validation seeds: 50-59.
- Historical matched test seeds 18-19: never reopen in this iteration.
- External holdout seeds 60-69: one evaluation after validation freeze.
- Actual UAV energy is audit-only; online decisions use physical task LCB and deployable energy proxy.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Formal CandidateSet grid deferred 100% on validation | 1 | Root-cause attribution is in progress; no threshold change yet |
| Inline attribution could not import `pi_jwm` because the piped Python process did not use the requested working directory | 1 | Use explicit absolute source/script paths for the diagnostic process |
| Explicit Unicode source path was still not visible to the piped Python process | 2 | Stop retrying the attribution command; probe cwd/path/encoding and use an environment-level `PYTHONPATH` or ASCII launcher |
| Piped script body converted the Chinese `代码` path segment to `??` before any model load | 3 | Abandon Unicode literals in stdin; inject the complete code root through `PI_JWM_CODE` and keep the Python body ASCII-only |
| `git bundle create` rejected a bare commit range as an empty ref set | 1 | Export the named `main` ref while excluding the remote base commit |
| Candidate-expert prototype requested unsupported HGB regressor loss `huber` | 1 | Use the installed sklearn's supported robust `absolute_error` loss |
| Phase-table prototype could not import the shared script metric helper | 1 | Add the repository scripts directory to `PYTHONPATH`; no experiment rows were evaluated before failure |
| Dense phase-table search could not JSON-serialize NumPy scalar types after selecting calibration parameters | 1 | Keep the already fixed calibration parameters and rerun validation formatting with scalar conversion only |
