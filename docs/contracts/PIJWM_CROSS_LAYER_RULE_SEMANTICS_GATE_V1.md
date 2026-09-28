# PI-JWM Cross-Layer Deterministic Semantics Gate v1

Before any future formal training, `CROSS_LAYER_RULE_SEMANTICS_GATE=PASS` is a mandatory prerequisite. Run the CPU-only gate and verify its source fingerprint immediately before freezing a new training source/configuration:

```powershell
python code/scripts/run_cross_layer_rule_semantics_gate_v1.py
python code/scripts/run_cross_layer_rule_semantics_gate_v1.py --check
```

The gate reads one real two-hop Input chain across Raw Ledger evidence, model-ready Sample, Tensor, Graph/State and deterministic World Model transition. It also checks intermediate and terminal Flow progress, same-destination Route full-path update, Return identity, Task completion, fixed-support Return birth, communication relation rebinding, CPU service, mobility rules and the E2E useful throughput extractor. The receipt is `code/artifacts/protocols/pi_jwm_step6_2a_route_recovery_v1_20260928/12_cross_layer_semantics_gate.json`.

Any change to route representation, Flow representation, Task lifecycle, action semantics, their adapters, or the rule transition invalidates the old receipt. The gate script compares SHA-256 fingerprints of relevant source and test files; a failed or stale receipt blocks future formal training. A passing gate establishes only consistency on its tested fixtures. Formal Dataset coverage, checkpoint compatibility and validation are separate gates.

The frozen route convention is: `route_node_indices` lists hop destinations in order and excludes the current holder. At hop index `i`, `carrying_hop_source_index=current_holder_index` and `carrying_hop_destination_index=route_node_indices[i]`. Completion of a nonterminal hop sets holder to the completed destination, advances `i`, and resets hop remaining to the unchanged E2E remaining. Same-destination reroute replaces the full destination list through deterministic rule metadata, resets `i=0`, preserves FlowID/Epoch and increments RouteRevision. Destination-change Route requires a new Epoch and is unsupported by the fixed Flow support in this version.

Main Throughput remains E2E Useful Throughput. Network Service Throughput remains diagnostic. The gate checks the shared Flow/Outcome extraction rule; it does not change either metric definition or authorize a baseline run.
