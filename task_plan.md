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
| 5. Local smoke and full validation checks | in_progress | CPU smoke and relevant/full tests |
| 6. Sync and run the necessary GPU experiment | pending | Reproducible remote run and logs |
| 7. Freeze on validation and evaluate external holdout once | pending | Frozen config, holdout report, manifest |
| 8. Update report artifacts and PPT data placeholders | pending | CSV/JSON/figures and updated PPT planning document |

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
