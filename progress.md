# PI-JWM v11 Selector Progress Log

## 2026-07-20

- Resumed from completed RTX 4090 formal run.
- Confirmed repository HEAD is `f7cb651` and the worktree started clean.
- Confirmed formal output directory and checkpoint loading/selection APIs exist locally.
- Started systematic attribution of the 100% defer failure before changing any selector behavior.
- Confirmed all 36 checkpoint files and the formal summary are present locally.
- Confirmed the candidate gate passed while the selector gate failed only on RMSE and improved-seed count.
- First attribution launch stopped before loading data because `pi_jwm` was not on the piped Python process path; no experiment result was produced.
- A second launch with a literal Unicode path failed at the same pre-load import boundary; attribution data and checkpoints remain untouched.
- The environment probe proved `PYTHONPATH` works. A third launch then exposed stdin code-page corruption of the Chinese `代码` literal; it stopped before model loading. The launcher is being changed to an ASCII-only body with a full code-root environment variable.
- Completed the formal checkpoint attribution on validation only and wrote CSV/JSON artifacts under `selector_attribution_task_bridge_h10_f7cb651`.
- Attribution shows rank failure, not merely conservative defer: rank-only RMSE is 292.90-312.33 and all validation seeds worsen.
- Ran the existing schema-v6 benefit identifiability audit with HGB on selected-edge, pooled-interaction, and full interaction feature groups.
- Opportunity detection generalized strongly, but candidate ordering did not; the next architecture will separate the opportunity gate from token-level candidate ranking.
- Added the token-level ranker, opportunity-masked loss, deterministic mini-batch fit/predict, safe selection, and a train/calibration/validation-only runner using TDD.
- Local gate passed: 85 relevant selector tests, Python compilation, and diff checks.
- The first incremental bundle command produced no file because it did not include a named ref; remote state was unchanged.
- Synced commit `c7f743b` to the RTX 4090 with a verified Git bundle; remote selector tests passed 85/85.
- Completed full-data CUDA smoke and a no-phase 3-seed x 20-epoch token probe; the training chain is healthy but calibration found no safe threshold.
- Demonstrated that observable within-episode phase improves the HGB validation diagnostic to RMSE 232.023 with 100% positive precision on six executions.
- Added episode-phase context to the formal token runner with a failing-then-passing test; related tests now pass 86/86.
- Candidate-specific HGB prototype selected a 15-candidate train-only shortlist with validation oracle 170.80, then stopped before fitting because this sklearn version rejects `loss='huber'`.
- Candidate-specific experts completed after the loss correction but reached only validation RMSE 233.164; this over-split architecture was rejected.
- Phase-table prototype first launch stopped at metric-helper import because `scripts` was absent from `PYTHONPATH`; no result was produced.
- Phase-conditioned train statistics reached validation RMSE 207.540 with 10/10 seed improvements under the deployable Pareto rule.
- Confirmed the result is unchanged when validation `action_applied` is removed from the decision mask; no actual-rollout outcome is used online.
- Rejected phase smoothing, estimator ensemble, candidate experts, learned residual routing, and phase-restricted kNN after controlled train/calibration/validation comparisons.
- Added the reusable phase selector module, formal runner, decision trace, statistics cache, and freeze report using TDD.
- Formal runner reproduced validation RMSE 207.5398777 and freeze digest `887331b2...454a2` without warnings.
- Final verification passed: 722/722 main tests, 84/84 script tests, compileall, and diff checks.

## 2026-08-14 P2-C Advisor-Document Manifest Binding

- Added two RED assertions: the canonical manifest must include the P2-C research-progress document, and changing a temporary project-local copy must produce `source hash mismatch`. Both failed for the intended missing-binding reason before implementation.
- Added exactly one production dependency path to `CANONICAL_SOURCE_PATHS`; the P2-C test pair then passed 9/9, with compile, Ruff, and diff checks clean.
- Published a candidate audit, verified it, confirmed the audit report and candidate config were byte-identical to the prior canonical, archived the prior canonical, and promoted the candidate.
- Fresh evidence gate: P2-B `--verify-only` passed, P2-C `--verify-only` passed, AirFogSim manifest dependencies matched 83/83, and the 17-module focused suite passed 159/159 under `PYTHONUTF8=1`.
- Status remains `blocked` with four unchanged reasons; no GPU, locked test, formal trajectory generation, or third-party source modification occurred.
