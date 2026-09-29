# STEP 6.3B/6.3C-PATCH — Comm Task Selection & TRAIN Self-Replay Audit

Status: `COMPLETE_FOR_CPU_PATCH_AUDIT; RESEARCHER_REVIEW_PENDING`. The recovery observations below remain as predecision provenance; the final patch evidence is recorded below.

## Researcher Decision — 2026-09-29

The researcher explicitly removed all-eligible Comm coverage and froze `E_comm(t)` as Tasks uniquely bindable to an existing current/predicted wireless Flow. A selected subset `S_comm(t) ⊆ E_comm(t)` must use a TRAIN-observed unique selected-task count conditional on its Comm structural signature; every selected Task has a row. Historical Comm rows that need a same-decision Route-created Flow are outside Planner-v1 formal candidates and projected self-replay, while original TRAIN actions remain intact. All other named policy/model/protocol boundaries remain unchanged. This decision supersedes the conflict status below; the old conflict paragraphs are retained as recovery provenance.

## Predecision Recovery Audit — Historical Goal and Definition Basis

The resumed task aims to close the Comm Task selection rule and verify that the resulting Planner v1 candidate grammar can self-replay Formal TRAIN H1 actions. The frozen STEP 6.3B contract says every currently eligible wireless Task receives at least one Comm row. STEP 6.3C uses that grammar for symbolic domain counting and lazy enumeration. Formal TRAIN supplies observed structure; the frozen Route policy is explicit no-op. This audit does not authorize a new scientific support rule.

## Predecision Recovery Audit — Initial State and Files Involved

The recovery checkout is `main` with uncommitted edits to `step6_3b_candidate_grammar_v1.py`, `step6_3b_candidate_support_v1.py`, `step6_3c_candidate_domain_v1.py`, the 6.3B support audit script, and four JSON audit artifacts; `run_step6_3bc_train_self_replay_v1.py` is untracked. Pre-existing untracked `TASK/` and `code/scripts/plot_step5_3e_tiny_overfit.py` are left alone. No source, artifact or commit from the broken chat is assumed accepted merely because it exists locally.

## Predecision Recovery Audit — Documented Intent, Actual Implementation, Evidence, Affected Files, Conflict

- **Documented Intent:** `docs/contracts/PIJWM_STEP_06_3B_STRUCTURED_CANDIDATE_GRAMMAR_V1.md` says every currently eligible wireless Task gets at least one Comm row; an empty Comm family requires no eligible Task. The 6.3C record describes symbolic counts based on coverage of eligible Tasks.
- **Actual Implementation:** the uncommitted grammar admits a subset of current wireless Tasks when its *selected Task count* has been observed for that Comm width signature in TRAIN. The uncommitted 6.3C domain counts and enumerates all Task subsets of those observed sizes. The count catalog does not identify which selected Tasks are causally representable on a given current state.
- **Evidence:** the old accepted TRAIN static domain receipt has 1935/4416 empty anchors and maximum 313949952 concrete choices. The uncommitted recomputed receipt has 14/4416 empty anchors and maximum 61092600570 choices. The existing projected TRAIN H1 audit artifact reports Comm pass 3688/4416 and full projected pass 3404/4416, with failures grouped under current wireless Flow identity and Comp zero-base/alpha ambiguity. The 6.3B focused suite reports 3 failures among 11 tests because assertions still express the all-eligible rule; the 6.3C suite passes 8/8.
- **Affected Files:** the above three source modules, 6.3B support catalog/audit, 6.3C TRAIN/Validation domain receipts, the 6.3B/6.3C contracts, records, tests and AI_CONTEXT summaries.
- **Conflict:** the patch changes the formal candidate pool and its cardinality while the frozen definition and acceptance wording still describe full Task coverage. TRAIN self-replay also does not close under the current causal binding. Passing the symbolic counting tests does not resolve either mismatch.
- **Status at recovery time:** `AWAITING_RESEARCHER_DECISION`. This was resolved by the explicit decision above. No optimizer, candidate ranking, GPU, training, baseline, closed loop or `locked_test` was run.

## Predecision Recovery Audit — Reuse, Validation, Results, Expected Versus Actual

The existing 6.3B binding and 6.3C counting code, TRAIN support catalog, Formal Dataset manifest, and current-state projection are reused for this audit. `python -m unittest discover -s code/tests -p 'test_step6_3b*.py'` produced 2 failures and 1 error in 11 tests; `test_step6_3c*.py` passed 8/8. Two attempts to rerun the full TRAIN H1 script made no terminal progress for more than ten minutes and were interrupted; the existing `06_train_h1_projected_self_replay.json` was not independently reproduced in this recovery chat. Source `compileall` and `git diff --check` passed. No full repository test suite was run.

Expected closure would require (1) a researcher-approved Task selection rule, (2) matching contract/code/tests/counts, (3) a self-replay audit whose claims precisely state what is and is not reproduced, and (4) consistent records and receipts. Current evidence does not meet those gates. In particular, the script's `full_projected_pass` checks decoding, rebinding and support admission; it does not compare every original action field against the rebound action. It must not be called exact self-replay.

## Predecision Recovery Audit — Known Issues, Git, Next Step

The Comm causal mismatch may reflect collector actions on Tasks without a current active wireless Flow. The present audit only establishes the observed mismatch; it does not infer a new binding rule. Comp mismatches need their own causal-source analysis before any claim that full historical actions are reproducible. The 6.3C old `PASS` applies to the previously frozen full-coverage domain, not to the changed uncommitted domain. Git commit and push are intentionally deferred while tests fail and a scientific definition remains unresolved.

At this historical point, the decision had not yet arrived. The researcher subsequently approved the specified partial-selection policy and excluded Route-created Flow dependencies, as stated above.

## Final Patch Verification — 2026-09-29

- **Step goal / definition basis:** implement the researcher-approved `E_comm(t)` and conditional `S_comm(t)` boundary while preserving Route explicit NOOP, TRAIN support, 145 Comm `(start,width)` pairs, 251 joint signatures, Comp/Mob, temporal labels, CandidateDomain, interleaved rollout, `B_WM`, Objective, frozen model and checkpoint.
- **Files and evidence:** candidate grammar/support/domain sources and focused tests; `06_train_h1_projected_self_replay.json`; `08_comp_amount_residual_diagnostic.json`; `07_step6_3bc_patch_acceptance.json`; CandidateDomain receipts `01`/`02`; implementation contracts and synchronized context/index records.
- **Validation:** patch acceptance passed. TRAIN H1 anchors `4416/4416`; raw Comm rows `1517`, projected `789`, excluded `728` with reason `ROUTE_CREATED_FLOW_NOT_AVAILABLE_IN_CURRENT_STATE`; `COMM_TASK_SELECTION` rejection `0`; Comm projection admitted `4415/4416`; full structural admission `4045/4416`; semantic comparison pass `3920`, mismatch `125`.
- **Residuals:** one projected unseen Comm structure, 349 Comp alpha residuals, 15 positive requests with zero CPU base, 6 unseen joint structures, and 125 Comp amount mismatches. The amount diagnostic covers 127 rows with absolute delta min/median/max `1.002736e-7` / `1.447511e-7` / `2.979292e-7`; tolerance remains `1e-7`. These are implementation residuals, not silently accepted exact replay.
- **Domain audit:** TRAIN empty domains `14/4416`, Validation descriptive empty domains `6/1104`; before-patch historical values remain `1935/4416` and `550/1104`. Candidate count changes are a domain-definition consequence, not a performance result.
- **Boundaries:** original TRAIN unchanged; Validation did not define support; no optimizer, ranking, baseline, closed loop, GPU, training, locked-test access, Random Search or CEM. Stop after this patch; STEP 6.3D is not started.
- **Git:** final commit and push are recorded in the completion report after verification.
