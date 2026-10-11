# STEP 6.4J — Live Root diagnostic record (2026-10-11)

## Step Goal
定位 r39 `NO_SCOREABLE_H4` 的可核验机制证据，先复核原始统计，再以正式 Live Planner 路径检查 root 身份；身份不通过时停止评分探针。

## Definition Basis
用户授权的 STEP 6.4J 诊断范围；r39 raw receipts and frozen protocol.

## Initial State
r39 audit reported 3744 complete, 3064 dead-end, 0 scoreable H4, but no per-candidate ledger. Existing diagnostic script used static `load_frozen_runtime()` prepared anchors and full B512, which is out of scope as the primary route for this task.

## Files Involved
- `code/artifacts/backups/step6_4j_r39/pilot_results/**/decision_*.json`
- `code/artifacts/audit/pi_jwm_step6_4j_live_root_h4_diagnostic_20261011/diagnostic_receipt.json`
- `code/artifacts/audit/pi_jwm_step6_4j_live_root_h4_diagnostic_20261011/diagnostic_report.md`

## Changes
Independently counted complete/dead-end/scoreable fields; checked the formal live reconstruction path; wrote an additive diagnostic receipt and report. No scientific source, scorer, objective, candidate domain, H4, fallback or freeze configuration was changed.

## Validation
- 16 raw decision receipts read successfully.
- anchor-0006: 8×(344 complete, 0 dead-end); anchor-0029: 8×(124 complete, 383 dead-end).
- Totals: 3744 complete, 3064 dead-end, 0 scoreable H4.
- Root equality receipt was not produced within the bounded CPU reconstruction; status remains `ROOT_IDENTITY_UNVERIFIED`.
- GPU not used; `locked_test=false`; new score calls/transitions = 0.

## Results / Expected vs Actual
Expected: proceed to a bounded probe only for an identity-verified live root. Actual: r39 evidence lacks the initial causal prefix/History and candidate/RNG ledger, so identity cannot be proven. The hard gate correctly stopped the probe.

## Known Issues
Cannot attribute r39's 0 scoreable H4 to a unique rejection mechanism. Future Return-birth remains a hypothesis only, not a new r39 observation.

## Git
Pending commit and push after index/check validation.

## Next Step
Researcher decision on an additive live-root capture/replay evidence protocol.
