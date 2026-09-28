# STEP 6.3A — Candidate Support and Search Architecture Audit

## Goal and boundary

Audit the frozen Formal TRAIN action records before a later researcher choice
of candidate search. This Step does not implement or run an optimizer, rank
candidates, call the World Model, train, use GPU, execute a baseline, enter a
closed loop, or access `locked_test`. The accepted checkpoint and Formal
Dataset remain unchanged. Planner v1 still uses Comm/Comp/Mob and explicit
Route no-op; the objective remains the five-part strict lexicographic tuple.

## Definition basis and initial state

Start was `HEAD=origin/main=f58f08087bf27b2ffc21c2e6c68ec1d153acc439`.
Formal Dataset split identity comes from its manifest and split manifest:
48 TRAIN trajectories/4416 windows and 12 validation trajectories/1104
windows. Candidate support definitions use TRAIN only. Validation is a
descriptive application of the resulting summaries; it did not change a
template, threshold or recommendation.

## Security preflight

Tracked text and tracked/untracked secret-like filenames were inspected by
pattern class without printing matching contents. No high-confidence credential
or private-key match was found. `.gitignore` now contains only the requested
credential filename patterns. Receipt: `01_security_preflight.json`.

## TRAIN observations

- Comm: over 4608 raw TRAIN action steps, 1178 eligible opportunities, 674 explicit no-op opportunities and 504
  interventions. Non-empty rows have widths 1/2/3 (1161/231/222), using RB IDs
  0–49. Every observed row is a contiguous cyclic RB block. There were 100
  slots with an RB reused across rows, over 2289 relation-RB assignments. Raw action rows identify `task_id` and
  `rb_indices`, not a Comm relation index; relation identity is not fabricated.
  Current channel/action data distinguishes wireless Comm allocation from
  wired service, which is not an RB action. A later candidate must rebind task
  identity to current Comm slots using the existing causal adapter.
- Comp: the frozen `allocate_work_conserving_cpu` rule reproduces all 1969
  non-empty TRAIN action entries from current Decision computing Tasks, their
  remaining CPU work, static node CPU capacity and the 0.1 s slot duration.
  Reconstructed base multiplied by one slot-global alpha reproduces the
  observed actions; alpha support is exactly `{0.5, 0.75, 1.0}` with counts
  248/242/1479, and no mixed-alpha slot. Comp action rows per raw decision
  slot are 0/1/2/3/4 with counts 3123/1104/290/79/12. No equal-share rule was
  invented; the existing deterministic implementation is called.
- Mobility: the six observed profiles are HOLD plus five control profiles.
  For two present UAVs, the exact joint TRAIN signatures are shared profiles
  only: HOLD/HOLD and PROFILE_i/PROFILE_i. Per-UAV marginal support does not
  establish independent joint support. Heading delta is reconstructed against
  current Decision heading and elevation remains at its observed unchanged
  setting.
- Joint actions: TRAIN includes global no-op, one-family and multi-family
  intervention slots; global no-op=922, one active family=1816, multiple
  active families=528. There are 251 structural Comm/Comp/Mob joint signatures.
  The measured family product is not the observed joint law; independent family
  factorization is `NOT_SUPPORTED`. The detailed
  per-family conditional signature distributions and structural triple support
  are in receipt `05`.
- Temporal support: H1–H4 use `future_action_frame_indices` from the Formal
  sample index. Exact H4 has 2842 unique sequences and 2719 structural
  sequences across 4416 overlapping windows; 2425 structural sequences appear
  only once. Structural H1/H2/H3 sequence counts are 244/1165/2070. Repetition
  and singleton counts are recorded in
  receipt `06`. Marginal per-step support can compose unseen temporal sequences,
  so exact support claims need temporal conditioning or a declared support
  policy.

## Search space and support tiers

Receipt `07` reports representative TRAIN anchors spanning low/high activity,
high Comm rows and high Comp demand.
Counts are explicitly labeled as upper-bound illustrations: the exact syntax
for simultaneous multi-row Comm candidates is not frozen, and future predicted
eligibility depends on candidate rollout. The naïve factorized count grows
exponentially with horizon; the audit therefore cannot claim that exact
enumeration is feasible for every anchor, nor that Random Search or CEM is
necessary. Independent factorized CEM conflicts with the observed joint
structure. Conditional/hierarchical proposals are more consistent with the
evidence, but remain research proposals.

Support tiers are proposals, not implementation rules: Tier 0 is an observed
structural joint signature; Tier 1 combines individually observed family
signatures into an unobserved triple; Tier 2 uses a family value outside TRAIN
marginals. Coverage and structural counts are machine-recorded. Tier 1
permission requires a future researcher decision.

## H_sup interaction

The frozen scorer continues to use `H_eff=min_k H_sup(k)`. The existing
fixed-support audit reports 8,828 unsupported future Return-birth events over
the combined 5,520 TRAIN+validation windows, including 2,901 affected windows;
these are overlapping window-horizon events, not unique physical births.
That historical receipt does not retain a TRAIN-only H1–H4 histogram, so this
Step does not fabricate one or use validation to reconstruct it. Candidate
pool support-horizon grouping is a diagnostic proposal only; scorer semantics
are unchanged.

## Method readiness and fairness

Evidence-based recommendation: before selecting any optimizer, define a
support-constrained structured candidate syntax and compare it with conditional
sampling proposals. Do not assume independent family sampling. This is not a
researcher-frozen method decision. Future search comparisons must use the same
Planner action domain, frozen World Model, objective scorer, H<=4, rollout
budget, causal information and support policy. No comparison was executed.

## Acceptance and limitations

`STEP_6_3A=PASS` means the data/source action supports have been measured and
the unresolved search-grammar limits are explicitly bounded in receipts. It
does not select an optimizer or establish candidate quality, planner efficacy,
closed-loop performance or a scientific performance claim. Validation was
descriptive only; Formal future Return support counts remain aggregated across
splits in the older receipt.

Verification commands: the support audit script completed with `PASS`;
`compileall` passed; the cross-layer gate passed 115 tests. The full repository
unittest discovery ran 1993 tests and ended with 34 errors and one failure.
Most errors came from the Windows GBK console failing to encode a third-party
Unicode print or from absent historical local artifacts/bundles; one historical
STEP 6.0B receipt-rebuild test differed from its stored receipt. During that
over-broad suite, synthetic CPU trainer tests also executed. This was outside
the requested audit-only scope and is recorded as an execution deviation. No
Formal model training, checkpoint update, GPU run, or Formal Dataset write was
performed. The generated STEP 6.2A cross-layer receipt was restored after the
gate run and is not part of this change.

Machine evidence is in
`code/artifacts/protocols/pi_jwm_step6_3a_candidate_support_audit_v1_20260928/`.

## Git and next step

Implementation/evidence commit `b2059c4dbe173669f296b9f20a1225c9695d459e`
was pushed to `origin/main`. The final context/closure note is committed
separately so the current project status is recorded after the verification
deviation was discovered. Worktree-only user files `TASK/` and
`code/scripts/plot_step5_3e_tiny_overfit.py` were not included. Next action is
researcher review; do not start a candidate optimizer or closed loop.
