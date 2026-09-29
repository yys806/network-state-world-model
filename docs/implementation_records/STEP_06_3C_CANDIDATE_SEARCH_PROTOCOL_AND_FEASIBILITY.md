# STEP 6.3C — Candidate Search Protocol and Domain Feasibility Closure v1

## Step Goal and Definition Basis

Implement a search-independent, state-conditioned CandidateDomain; a one-step recursive frozen World Model primitive; a shared prefix and `B_WM` accounting contract; and all-anchor static domain feasibility evidence. The researcher explicitly froze interleaved search, no automatic horizon backoff, temporal support as diagnostic only, and candidate one-step transitions as the primary fair compute unit. STEP 6.3B grammar, Route no-op and Objective are not changed. No optimizer, ranking, baseline, closed loop, GPU, training or locked test belongs to this Step.

## Initial State and Files Involved

Start `HEAD=origin/main=df029647c748c9a4be66173abd388b805f954618`. STEP 6.3B and STEP 6.2B receipts remain PASS. Formal TRAIN catalog has 251 observed joint structural signatures and 145 Comm `(start,width)` pairs; its data manifest hash is checked by the loader. Existing 6.1 `model.one_step`, formal adapter and frozen checkpoint are reused. Files: `step6_3c_candidate_domain_v1.py`, `step6_3c_search_protocol_v1.py`, the additive 6.1 single-step primitive, focused tests, two bounded CPU audit scripts and STEP 6.3C receipts. The historical 6.1 preflight script had an unused stale import of `compile_candidate_step` from 6.0A; that import was removed so its accepted anchor loader could be reused. No 4.4 numerical transition or model parameter changed.

## Changes and Reuse

`CandidateDomain.from_state` calls the existing 6.3B causal wireless/Comp binding helpers and uses the same TRAIN catalog. It filters structural modes by current eligibility, then exposes Comm row-width groups, current Task assignments, supported start positions, Comp alpha and shared Mob modes. It lazily yields concrete choices and binds each with the unchanged 6.3B `bind_structured_step`. The symbolic count uses inclusion-exclusion over Task coverage and combinations with replacement over `(Task,start,width)` rows, so permutations never double-count one canonical action. A bounded synthetic full-domain enumeration and a separate formula brute-force test check exactness.

`rollout_one_step` in the 6.1 module calls the existing `compile_candidate_step`, route metadata builder and exact same `model.one_step` with mean prior and expectation service. `SearchNode` stores prefixes, current latent/state/graph/control and fingerprints. `TransitionBudgetAccountant` charges one unit per unique candidate-step evaluation, including every member of a batch forward, and caches only identical parent fingerprints plus canonical action fingerprint and causal provenance. Cache outputs are copied before reuse.

An empty H1 domain is `NO_FORMAL_CANDIDATE`; a later empty domain is `GRAMMAR_DEAD_END`. The grammar is not widened and `RULE_FALLBACK` is not recast as a formal candidate. H1–H4 feasibility horizons are recorded separately; this Step does not compare Objective values across horizons or make backoff policy.

## Validation and Results

The fixed, previously accepted STEP 6.1 non-locked Validation anchor (`sample_index=4416`) produced a canonical mechanism path through H1–H4. The one-step primitive and existing 6.1 sequential rollout had exact matching input/output fingerprints and zero max absolute difference in action tensors, latent, state, graph and model trace. Checkpoint SHA matched the accepted `941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9`; parameter digest was unchanged. This is one deterministic mechanism fixture, not a measured H4 feasibility rate or candidate quality. The path charged four candidate one-step transitions and had no cache hit or dead end. Batch/cache synthetic tests verify two unique steps in one Python call charge two `B_WM` units, while identical parent/action reuse charges no new unit.

The all-anchor TRAIN/Validation static feasibility distribution and final acceptance verdict are recorded in machine receipts after the full audit; Formal Validation is descriptive only. The current-state projection used for the audit is checked for exact equality against `build_state` on the first anchor of every trajectory. No Future Target package is loaded. Full command outputs and Context Consistency Check are included in the final acceptance receipt.

## Expected Versus Actual, Known Issues, Git, Next Step

The one-step numerical equivalence and bounded counting checks match the expected contract. Structural and concrete candidate counts must be read separately. A canonical path cannot establish the population probability of dead ends or search performance; static H1 cardinality does not establish dynamic H2–H4 feasibility. The frozen 6.3B boundary on predicted Comp lifecycle remains. Final Git commit/push identity is reported after verification. Any later optimizer method comparison requires separate STEP 6.3D authorization and must use this same domain, support policy, World Model, Objective and `B_WM` budget.

## Final Acceptance

The complete CPU audit covered 4416 Formal TRAIN anchors and 1104 Formal Validation anchors. TRAIN exact unique concrete candidate count had median 6, P90 205701120, P95 313949952, and maximum 313949952; compatible structural modes had median 5 and maximum 77. 1935 TRAIN anchors were empty (`NO_TRAIN_OBSERVED_JOINT_STRUCTURE_COMPATIBLE=1921`, `COMP_ALPHA_UNIDENTIFIABLE_FROM_ZERO_BASE=14`). These are explicit no-formal-candidate observations under the frozen policy, not support-expansion decisions. Validation was descriptive only and reported 550 empty domains without changing the TRAIN catalog.

The fixed accepted STEP 6.1 anchor completed H1-H4 interleaved rebinding. Existing sequential rollout and `rollout_one_step` matched exactly for actions, latent, state, graph and model trace; checkpoint SHA and parameter digest were unchanged. The smoke charged four unique transitions with no cache hits or dead ends. Synthetic tests cross-checked symbolic cardinality against bounded brute force and verified batch/cache accounting. Therefore `STEP_6_3C=PASS`: CandidateDomain, shared search protocol, feasibility evidence and dead-end semantics are closed. This does not select or implement an optimizer and does not establish candidate quality, dynamic horizon feasibility or closed-loop performance.
