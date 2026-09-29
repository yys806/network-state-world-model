# STEP 6.3B — Planner v1 Structured Candidate Grammar and Support Policy

Status: `STEP_6_3B=PASS` for the bounded CPU grammar and admission contract only. Researcher decisions were supplied in the STEP 6.3B conversation. Source facts come from Formal TRAIN receipts and current code. No optimizer or candidate quality result follows.

## One step and one sequence

At step `τ`, take the current causal or recursively predicted state `S_τ`, derive eligible identities and relations `E_τ`, choose a structural mode `Z_τ`, instantiate concrete parameters `Θ_τ`, and emit a complete `CandidateActionStep`: `S_τ → E_τ → Z_τ → Θ_τ → A_τ`. Route is always `entries=[]`. Each next step repeats binding and instantiation on its own predicted input state; the anchor's eligible Task set and CPU base are not copied to H2–H4. Horizon is 1–4. `CandidateActionSequence` remains the shared search-independent action schema. `StructuredCandidateAdmission` adds per-step signatures, causal bindings, multidimensional support labels, rejection reasons and `h_sup=None` until rollout. Any later search backend must use the same admission policy before the same adapter, rollout and scorer.

## Comm

Eligible Tasks have a known, present, active fixed-support Flow carrying on a current valid **wireless** Comm relation. The current Flow endpoints must match that relation. Wired service receives no RB row. A Task must have a unique current eligible relation; unresolved or ambiguous binding rejects the candidate. The raw TRAIN action stores Task ID and RB indices, not a reusable historical relation index. The grammar binds each row to the current relation and verifies the supplied candidate's Task slot and relation index.

When eligible wireless Tasks exist, every one receives at least one row. A Task may receive several rows; the total is at most four, the TRAIN-observed per-slot maximum. A row is `(task_id, start_rb, width)` with width in `{1,2,3}`, RB ID in `0..49`, and cyclic block `{(start_rb+i) mod 50 : 0≤i<width}`. The `(start_rb,width)` pair must be observed in Formal TRAIN (145 of 150 pairs). Rows for different Tasks may reuse RB IDs; Formal TRAIN contains such reuse. Rows are sorted canonically before compilation; supplied candidates with a different row ordering or concrete representation are rejected so their fingerprint is stable across backends. No eligible Task means empty Comm family. This is a support restriction, not a simulator physical legality claim. The existing adapter merges repeated rows at their current relation through its relation-RB mask; the 6.0A duplicate-Task rejection was corrected.

The generic 6.0A all-family-empty `RULE_FALLBACK` remains representable, but it does not automatically enter this formal support pool when eligible wireless, computing or UAV objects require explicit rows. A later fallback/safety protocol must decide how to use it; 6.3B does not silently override the TRAIN-backed admission policy.

## Comp

For each Task explicitly `computing` in the current state, require a unique current Task-Agent `Exec` relation and observed static CPU capacity of its node. Recompute `B_τ=allocate_work_conserving_cpu(current remaining work, current Exec host, static capacity, 0.1 s)` at **each** step. An eligible step has one row for every Task in `B_τ`, with one slot-global `α ∈ {0.5,0.75,1.0}` and allocation `αB_τ`. Mixed alpha, arbitrary per-Task CPU amounts, an invented allocator and an empty Comp family with eligible Tasks are rejected. Empty Comp is permitted only when `B_τ` has no Tasks. TRAIN policy `comp.no_op=true` still emitted nonempty alpha=1 rows when Tasks were eligible; it is distinct from an empty action. If predicted lifecycle/Exec/capacity cannot identify a causal base, admission fails with a causal-support reason rather than inventing it. Zero-only base rows do not identify alpha from numeric output and remain a support boundary.

## Mobility

The operational 6.0C interface still accepts six marginal profiles per UAV. The **formal Planner v1 pool** uses one shared profile for every currently present UAV: `(HOLD,HOLD)` or `(PROFILE_i,PROFILE_i)` on the two-UAV Formal TRAIN support. Every present UAV has an explicit row; HOLD is a zero-speed command with current heading/elevation, not an empty family. Current control side-state supplies heading/elevation and is updated after each command for the next step. If no UAV is present, the family is empty. An asymmetric profile may be operationally legal but is outside the formal Planner v1 support policy.

UAV presence changes are re-evaluated at each predicted step. An empty or one-UAV Mobility shape still needs its own TRAIN-observed family/joint signature to enter the formal pool; operational legality by itself does not grant learned support.

## Joint and temporal support

The Formal TRAIN H1–H4 action-frame union contains 251 observed **structural** Comm–Comp–Mob triples. Formal pool admission requires the bound step's triple to be among them, in addition to family and parameter checks. These 251 signatures are coarse structure descriptions, not 251 concrete actions: Task ID, current relation, RB start, node and UAV identities are instantiated from the current state. A family-marginally observed but joint-unseen composition is labeled and reserved for a separately authorized support-expansion ablation. An unseen family marginal is excluded. Independent family-product generation is not the formal default (`INDEPENDENT_FAMILY_FACTORIZATION=NOT_SUPPORTED` from 6.3A).

TRAIN H1–H4 structural prefixes and observed adjacent pairs are recorded as temporal labels. Neither an unseen prefix nor an unseen adjacent pair is a hard admission gate. Every step must nevertheless rebind from its predicted state. Exact sequence replay and independent state-blind H1–H4 sampling are not this grammar. This admission choice is a researcher decision; temporal generalization remains unverified.

## Multidimensional labels and admission

Each step records family marginal `TRAIN_OBSERVED/TRAIN_UNSEEN`, joint structure `TRAIN_OBSERVED_STRUCTURAL/TRAIN_UNSEEN_STRUCTURAL`, temporal prefix/adjacent observedness, causal binding, fixed-support status and eventual `H_sup`. Comm pair support is checked on concrete rows before a bound step is emitted. `formal_pool_admitted=true` requires valid causal binding, observed family marginals, observed joint structure and no known fixed-support violation. Temporal observedness is diagnostic. Malformed identity, invalid parameter, nonempty Route, missing required eligible family, or unsupported structure rejects before rollout. `H_sup` is **not** inferred from action signatures: a candidate rollout/scorer computes it using fixed object support, with common `H_eff=min_k H_sup(k)` unchanged.

The TRAIN-only H1/H2/H3/H4 future-Return `H_sup` histogram is not available. No mixed TRAIN+validation statistic is presented as TRAIN evidence. Grammar and admission do not depend on that histogram; it remains a later descriptive support audit. The model's current deterministic rule does not promote a Task from offloading to computing after Flow completion; therefore a future Comp row needs an explicitly predicted `computing` state and unique current Exec relation. This limits future action instantiation and is not silently repaired in 6.3B.

## Evidence and open work

Formal TRAIN is the sole template source. Validation may be descriptive only. The receipt catalog is `code/artifacts/protocols/pi_jwm_step6_3b_candidate_grammar_v1_20260929/02_train_structural_support_catalog.json`; its manifest hash and schema are checked by the loader. The implementation is `code/src/pi_jwm/step6_3b_candidate_grammar_v1.py` and `step6_3b_candidate_support_v1.py`. Bounded CPU tests verify binding, compilation and labels. No search method, budget, elite ratio, warm start, candidate ranking, World Model training, GPU, locked test, baseline or closed-loop performance is selected or measured here. STEP 6.3C needs separate authorization.
