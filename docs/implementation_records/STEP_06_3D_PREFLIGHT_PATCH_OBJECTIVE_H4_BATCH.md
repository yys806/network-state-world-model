# STEP 6.3D-PREFLIGHT-PATCH — Objective Eligibility, H4 Scoreability, Batch Closure

Status: `STEP_6_3D_PREFLIGHT_PATCH=PASS`;
`H4_SEARCH_COMPARISON_READINESS=PENDING_RESEARCHER_DECISION_ON_MODEL_OBJECTIVE_SUPPORT`.
STEP 6.3D formal method comparison has not run.

## Step goal and definition basis

The researcher authorized a bounded preflight on 2026-09-30: replace only
selected anchors whose frozen 6.2B Objective cohort is empty; diagnose H4
scoreability on the corrected 32 Formal TRAIN anchors with HRS seed 6391 and
`B_WM=64`; and close the serial/batched one-step transition path. HRS, S-CEM,
MH-CEM, Grammar, Objective, `H_sup`, `B_WM`, checkpoint and method-selection
rule remain frozen. Validation search, formal tuning/comparison and
`locked_test` are outside this Step.

## Initial state

`HEAD=origin/main=34f38183c21227f9a8e95d4f13a9513a4cf88506` at start.
Historical selected manifests 01/02 have 32 TRAIN and 64 Validation anchors.
Accepted side-state audit 09 found three TRAIN and five Validation anchors
with `cohort_count=0`, making them necessarily unscoreable. The earlier
single TRAIN B_WM=512 probe produced 130 complete H4 trajectories, all
blocked by `UNSUPPORTED_FUTURE_RETURN_BIRTH` at H1. The formal search loop
was serial per transition. Unrelated preexisting `TASK/` and the Step 5.3E
plot script were left untouched.

## Files, changes and reuse

- `step6_3d_objective_anchor_patch_v1.py` and the builder create manifests
  15/16 by the original split-local quartile × Comp-base stratum and sample-id
  hash order. They skip static empty domains, zero cohorts and all previously
  selected anchors. Selection reads static causal state only, never rollout
  or search outcome. Historical manifests 01/02 are preserved.
- The sidecar replay refactor retains the old entrypoint; the replacement
  entrypoint replayed the eight new anchors against unmodified Formal Raw.
  Receipts 17/18 show exact alignment, and sidecar 19 merges the 88 retained
  with eight replacements. The frozen scorer-side construction and current
  `CandidateDomain` were used for receipt 20; all 96 selected anchors now
  have nonempty static CandidateDomain and positive cohort count. The
  `sidecar_alignment_pending=true` field in manifests 15/16 is a
  selection-time snapshot; later receipts 18/19/20 close that pending check
  without rewriting the selection evidence.
- `rollout_one_step_batch` shares the existing action adapter, model
  `one_step`, deterministic rule, state and side-state preparation with the
  serial primitive. It reuses STEP 6.1 batch tree/action stacking and
  per-candidate unbatching. The optional batch search wavefront uses the
  existing `TransitionBudgetAccountant.evaluate_batch`; each unique
  candidate-step costs one `B_WM`, and duplicate transitions hit the cache.
  Formal runner input paths now reference 15/16/19. The formal matrix was not
  executed.
- Scripts 21/23 check real nonlocked TRAIN serial/batch equivalence and
  throughput; script 22 is a resumable TRAIN-only 32-anchor H4 diagnostic.

## Validation and results

Machine evidence directory:
`code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/`.
Receipt 24 is the byte/SHA-256 manifest for 15–23, 32 per-anchor receipts
and the implementation sources; receipt 25 verifies the preflight gate and
hashes 24.

Static eligibility: 32 TRAIN and 64 Validation retained; eight replacements
(3 TRAIN, 5 Validation); zero remaining selected empty domains or cohorts.
Replacement deadline alignment: 8/8 exact, across five trajectories. Original
Raw and previous receipts remain unchanged.

CPU one-step equivalence on a real TRAIN fixture passed at batch sizes
1/4/8/16 with the existing `1e-4` STEP 6.1 numerical tolerance; action
mapping, causal parent fingerprints and next-state CandidateDomain support
matched. Each 64-request throughput probe accounted for 64 unique
transitions and 64 cache hits on replay. Measured transitions/s were
0.8695/0.8911/0.9640/0.9218, respectively; peak process RSS diagnostic
was about 0.405/0.519/1.066/1.937 GB. Batch 8 is the provisional CPU
choice from this bounded probe; it is only 1.11× batch 1. A second real TRAIN
fixture with scoreable H4 paths passed serial/batch Grammar admission,
`H_sup`, Objective values within the same tolerance and lexicographic
ordering at sizes 1/4/8/16. H4 scoreability distribution across all 32
anchors: 16 found at least one scoreable candidate and 16 found none at
the frozen 64-transition diagnostic budget. All 32 consumed exactly 64 unique
transitions (2048 total). There were 406 complete H4 sequences: 131 scoreable,
275 unscoreable. All 275 unscoreable sequences had the
`UNSUPPORTED_FUTURE_RETURN_BIRTH` support boundary; 27 anchors had at least
one such event. Grammar dead ends numbered 240 branches across 14 anchors.
There were zero empty-cohort residuals and zero scorer inconsistency cases.
These counts are HRS/seed 6391/TRAIN-only diagnostics, not a method-comparison
success rate or model-quality estimate. Receipts 22 and 25 provide each
anchor's counts and the aggregated reason taxonomy.

The exact-oracle fixture remains a correctness test, not method performance
evidence. No checkpoint training, GPU, Validation search, baseline, simulator
closed loop or `locked_test` access occurred.

## Expected vs actual, known issues, Git and next step

The static Objective eligibility defect is closed. The 16/16 split does not
meet the researcher's "majority scoreable" readiness case, and the researcher
deliberately did not freeze a numerical "large unscoreable fraction" threshold.
Because Return birth is the common fixed support reason, formal 6.3D tuning
and comparison remain stopped pending the researcher's scientific assessment.
No anchor was removed or replaced based on search outcome. The batch path
gives a mechanism speed probe, not a new search method or evidence that CEM
beats HRS. The acceptance receipt 25 and SHA manifest 24 close this bounded
Patch; neither selects a search method nor changes the frozen scorer.
