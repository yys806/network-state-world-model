# STEP 6.3C — Candidate Search Protocol and Feasibility v1

Status: researcher-frozen search protocol; CPU evidence and acceptance are recorded separately. This contract does not select or implement an optimizer.

## Sole candidate domain

`CandidateDomain.from_state(S_τ, prior_structural_signatures)` is the shared entrypoint for every future search backend. It reuses STEP 6.3B's current wireless Flow/relation binding, causal CPU allocator, shared UAV profile, TRAIN-only catalog and `bind_structured_step()` admission. It returns eligible Task/relation identities, causal CPU base, present UAV slots, compatible TRAIN-observed joint structural modes, supported RB starts per width, symbolic cardinalities and a lazy concrete-choice iterator. Structural modes are coarse templates, never fixed Task/node/RB actions. Every yielded concrete action is rebound with STEP 6.3B; no second legality definition is introduced.

The formally admitted one-step domain is `D(S_τ,c_τ)`, where `c_τ` is the structural-signature prefix used for diagnostic temporal labels. A candidate horizon is interleaved:

`(S_τ,z_τ,G_τ,c_τ) → D → A_τ → frozen model.one_step → (Ŝ_{τ+1},ẑ_{τ+1},Ĝ_{τ+1},c_{τ+1}) → rebuild D`.

H2–H4 concrete identity, relation and CPU base are never bound at H0. The `SearchNode` holds the action/signature/support prefixes, current latent/state/graph/control side-state, causal fingerprints, depth and transition accounting. The one-step primitive calls the existing formal adapter and the same 4.4 `model.one_step` with `model.eval()`, `torch.no_grad()`, mean prior and expectation service. No second World Model rule is written.

## Exact symbolic one-step cardinality

For a TRAIN-observed Comm width multiset with multiplicity `m_w`, current bindable wireless Task count `T`, and TRAIN-supported RB starts `P_w` for width `w`, let `K(signature)` be the TRAIN-observed unique selected-task counts for that Comm signature. For each `k∈K(signature)`, choose `k` of the `T` current Tasks, then count unique canonical row multisets covering exactly those `k` selected Tasks:

`C_Comm(T,{m_w},K) = Σ_{k∈K,1≤k≤min(T,r)} binom(T,k) Σ_{j=0}^k (-1)^j binom(k,j) Π_w binom((k-j)P_w + m_w - 1,m_w)`, where `r=Σ_w m_w` is the number of Comm rows.

The zero-alternative term is zero for positive `m_w`. With no Comm rows, the count is one when selected count zero is TRAIN-supported, even if `T>0`. Repeated identical rows are counted once as a multiset; row permutations are not new actions. The implementation cross-checks this count against brute-force enumeration on bounded synthetic domains. A compatible joint structural mode contributes its Comm count times its one Comp alpha and one shared Mob profile. Across modes, the exact one-step count is their sum after excluding states where numeric Comp alpha is unidentifiable. Counts are integer symbolic values; full concrete spaces are never materialized merely to count them. Lazy iteration binds only actions requested by a caller.

## Dead ends and horizon

An empty domain at H1 is `NO_FORMAL_CANDIDATE`; an empty predicted-step domain terminates that branch as `GRAMMAR_DEAD_END`. Neither opens a new support class or inserts a generic fallback. H1, H2, H3 and H4 are four separately recorded feasibility horizons. No automatic H4→H3→H2→H1 backoff or cross-horizon Objective comparison is authorized. The existing `RULE_FALLBACK` stays separate and has no TRAIN-supported or safety claim.

## Common compute budget

`B_WM` counts actually evaluated candidate one-step transitions. A batch forward with `n` unique candidate steps charges `n`, regardless of the number of Python calls. The accountant records proposals, admissions, rejections, unique evaluations, cache hits, completed prefixes, dead-end branches and wall-clock diagnostic time. A deterministic cache key contains causal provenance, the parent latent/state/graph fingerprints and the canonical action-step fingerprint. A cache hit never charges another `B_WM` unit. The budget API checks a batch's entire unique miss set before executing it.

Future Random/CEM/other comparisons must share this CandidateDomain, admission, frozen World Model, Objective, horizon, causal information and `B_WM`. Wall-clock is diagnostic only. This contract does not choose a final optimizer, rollout budget value, iteration count, elite ratio or warm start.

## Evidence boundary

Formal TRAIN defines support; Formal Validation is descriptive only. Static H1 domain counts are not candidate ranking or predicted performance. One fixed non-locked Validation anchor from the accepted STEP 6.1 receipt is used for a bounded canonical-path H1–H4 mechanism smoke, without choosing actions by the Objective. A canonical path's feasibility or dead end is not a population rate. No GPU, training, optimizer, baseline, locked test or closed-loop run belongs to this Step.
