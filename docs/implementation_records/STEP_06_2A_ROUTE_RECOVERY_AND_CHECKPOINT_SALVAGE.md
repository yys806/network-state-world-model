# STEP 6.2A-ROUTE-RECOVERY — Deterministic Route Repair and Frozen Checkpoint Audit

## 1. Goal and boundary

This was a no-retrain recovery Step. It did not run an optimizer step, GPU training, locked test, candidate ranking, closed loop, baseline, or checkpoint replacement. The reference start was `2d9a72d54f0b3416e9ce8d8be7ee395f6caf1d38`; the frozen `best.pt` SHA-256 was `941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9`.

## 2. Root cause

4.2C-B/C stores `route_node_indices` as ordered hop destinations and excludes the current holder. For `A -> B -> C`, the route is `[B,C]`. The old 4.4 rule treated the array as `[holder,next,next]`, so an intermediate completion could leave holder/index unchanged and leave an impossible active zero-hop state. Route actions also changed endpoints without replacing the complete route array; `hop_count` was not a path constructor.

## 3. Repair

The patched rule validates the current route index against the carrying destination, sets the new holder/source to the completed destination, advances the index, selects `route[new_index]`, preserves E2E remaining, and resets hop remaining. Missing next hops fail explicitly. Same-destination reroute uses deterministic `RouteRuleMetadata` side-state to replace the complete destination list, reset index/source/destination/progress, set hop remaining to E2E remaining, rebind the communication relation, and increment RouteRevision. The metadata is outside the 11 learned action tensors, encoder, GRU, latent and training target. Destination-change reroute raises `UNSUPPORTED_BY_FIXED_OBJECT_SUPPORT` rather than fabricating an Epoch/Flow.

## 4. Evidence

The regression suite includes partial and complete `A -> B -> C`, terminal completion, real two-hop Input Raw → Tensor → Graph/State → rule alignment, communication rebinding, same-destination full-path reroute, and destination-change rejection. The cross-layer semantics gate passed 113 tests. The original checkpoint strict-loaded with 428 state-dict tensors; the parameter digest before/after was identical, and the checkpoint bytes were identical.

The full static scan consumed all 4416 train and 1104 validation windows. Both splits had zero anchor/target/future route rows with length greater than one; route tensor width was one throughout. There were 38 train and 16 validation windows with an existing anchor Flow overlapping a later Route action. Legacy/patched paired CPU rollout on those 54 windows and H1–H4 produced exact equality for state, graph, prior mean/log_std, Motion and CSI. This is unaffected-window evidence, not multi-hop capability evidence.

Full patched 1104-window H1–H4 validation was not executed because the bounded paired audit required about 700 seconds for 54 windows. Receipt 10 therefore states `NOT_EXECUTED_REQUIRES_SEPARATE_RUNTIME_AUTHORIZATION`; no replacement metric was invented. The legacy accepted `LVal=0.07431338784170399` remains untouched.

## 5. Salvage verdict

`CHECKPOINT_NO_RETRAIN_SALVAGE=SUPPORTED_WITH_LIMITATIONS`. The checkpoint interface and no-retrain compatibility are demonstrated. The limitation is material: formal data does not exercise multi-hop or full-path reroute, so this Step cannot claim formal performance preservation for those branches. The next formal training must first pass `CROSS_LAYER_RULE_SEMANTICS_GATE=PASS`.

Machine receipts are under `code/artifacts/protocols/pi_jwm_step6_2a_route_recovery_v1_20260928/`, including canonical semantics, patch contract, checkpoint identity, train/validation activation counts, paired state comparison, propagation, invariance, validation status, verdict and gate. The planner objective contract now points to the repaired route convention while scorer implementation and 6.2B remain unopened.

## Post-Researcher Closure

The researcher accepted the existing frozen best checkpoint without retraining and froze Planner v1 to `FORMAL_DATASET_SINGLE_HOP_SUPPORT_V1`. Route remains enabled, but a legal v1 path has exactly one node: the current frozen logical destination. Repaired multi-hop and same-destination full-path reroute code remains available outside the Planner v1 formal action domain. Formal train/validation multi-hop coverage remains zero; no formal multi-hop performance claim is made. Patched full validation remains unexecuted, and the original `LVal` remains a legacy accepted observation. The separate closure record and receipts calculate scorer implementation readiness; this postscript does not rewrite historical recovery measurements.
