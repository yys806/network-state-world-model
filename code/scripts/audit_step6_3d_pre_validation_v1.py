"""Read-only scientific-source audit; synthetic CPU and TRAIN evidence only.

Never calls the formal runner or loads a Validation outcome. A failed gate is
an accepted audit finding, not a reason to mutate the frozen search algorithms.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/src"), str(ROOT / "code/scripts"), str(ROOT / "code/tests")]
import torch
from test_step6_3b_candidate_grammar_v1 import fixture
from pi_jwm.step6_1_trained_candidate_rollout_v1 import OneStepRolloutResult, fingerprint
from pi_jwm.step6_2b_planner_objective_scorer_v1 import objective_sort_key
from pi_jwm.step6_3b_candidate_grammar_v1 import CommBlockChoice, StructuredStepChoice, bind_structured_step
from pi_jwm.step6_3b_candidate_support_v1 import TrainStructuralSupportCatalog
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_3c_search_protocol_v1 import SearchNode, TransitionBudgetAccountant
from pi_jwm.step6_3d_fixed_budget_search_v1 import quotas, solve_fixed_budget
from pi_jwm.step6_3d_structured_proposal_v1 import SparseCategorical, StructuredProposalDistribution
from pi_jwm.step6_3d_method_selection_v1 import paired_outcome, cluster_bootstrap_advantage, select_method
from run_step6_3d_one_cpu_solve_v1 import OUT as TRAIN, source_hashes
from run_step6_3d_formal_cpu_matrix_v1 import result_path, validate_resume_result

OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_audit_v1_20261001"
METHODS = ("HRS", "S-CEM", "MH-CEM")
BUDGETS = (256, 512, 1024)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_bytes((json.dumps(value, ensure_ascii=False, indent=2,
        sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def synthetic_anchor(provenance="TRAIN_ONLY_SYNTHETIC_A"):
    context, operational, state, control, _ = fixture()
    context = replace(context, causal_provenance=provenance)
    operational = replace(operational, causal_provenance=provenance)
    signature = ("rows:1", "tasks:1;alpha:0.5", "HOLD,HOLD")
    catalog = TrainStructuralSupportCatalog("TRAIN_ONLY_SYNTHETIC", frozenset((signature,)),
        tuple(frozenset((signature[i],)) for i in range(3)),
        (frozenset(((signature,),)), frozenset(), frozenset(), frozenset()),
        frozenset(), frozenset((i, 1) for i in range(50)), {"rows:1": frozenset({1})})
    latent = {"path": torch.zeros((1, 1), dtype=torch.float64)}
    graph = {"oracle": torch.zeros((1, 1))}
    fps = {"state": fingerprint(state), "graph": fingerprint(graph), "latent": fingerprint(latent)}
    anchor = SimpleNamespace(context=context, domain=operational, state=state,
        graph=graph, latent=latent, fingerprints=fps)

    def transition(node, bound):
        start = int(bound.action.comm[0]["rb_indices"][0])
        next_latent = {"path": node.latent["path"] * 51 + start + 1}
        output = {"latent": fingerprint(next_latent), "state": fingerprint(state), "graph": fingerprint(graph)}
        return OneStepRolloutResult(next_latent, state, graph, bound.next_mobility_control,
            {}, {}, {}, node.fingerprints, output)

    def score(candidate, trace):
        assert len(trace.states) == 4
        return SimpleNamespace(objective_tuple=(sum(int(s.comm[0]["rb_indices"][0])
            for s in candidate.steps), 0., 0., 0., 0.), candidate_fingerprint=candidate.fingerprint)
    return anchor, catalog, transition, score


def discrete(outcome):
    row = asdict(outcome)
    row["budget_receipt"].pop("wall_clock_seconds_diagnostic_only")
    return row


def solve_fixture(method, budget, provenance="TRAIN_ONLY_SYNTHETIC_A", seed=6301):
    anchor, catalog, transition, score = synthetic_anchor(provenance)
    return solve_fixed_budget(method=method, seed=seed, b_wm=budget, anchor=anchor,
        catalog=catalog, transition=transition, score_h4=score, batch_size=16,
        transition_batch=lambda requests: tuple(transition(n, b) for n, b in requests),
        iterations=1 if method == "HRS" else 4, elite_ratio=None if method == "HRS" else .1)


def math_and_layers():
    # Independent rational oracle: q0=(1/3,1/3,1/3); elite=(1/4,3/4,0).
    table = SparseCategorical()
    table.update({"a": .25, "b": .75})
    first = {x: table.prior_weight / 3 + table.sparse_weights.get(x, 0) for x in "abc"}
    expected = {"a": Fraction(47,160), "b": Fraction(85,160), "c": Fraction(28,160)}
    assert all(abs(first[x]-float(expected[x])) < 1e-12 for x in first)
    before = first.copy()
    table.update({"c": 1.})
    second = {x: table.prior_weight / 3 + table.sparse_weights.get(x, 0) for x in "abc"}
    oracle = {x: .95 * (.5 * before[x] + .5 * int(x == "c")) + .05 / 3 for x in "abc"}
    assert all(abs(second[x]-oracle[x]) < 1e-12 for x in second)
    assert abs(sum(second.values())-1)<1e-12 and min(second.values())>0
    layers = ("mode", "count", "subset", "assignment", "start")
    evidence = {}
    for method in METHODS:
        proposal = StructuredProposalDistribution(method)
        for layer in layers:
            proposal.table(layer, (layer, ("a", "b", "c")))
        elite = [{"draws": [((layer, ("a", "b", "c")), "a") for layer in layers]},
                 {"draws": [((layer, ("a", "b", "c")), "b") for layer in layers]}]
        proposal.update_from_elites(elite)
        changed = sorted(key[0] for key, t in proposal.tables.items() if t.prior_weight != 1.)
        expected_layers = [] if method == "HRS" else ["count", "mode"] if method == "S-CEM" else sorted(layers)
        assert changed == expected_layers
        for key, distribution in proposal.tables.items():
            if key[0] in expected_layers:
                assert abs(distribution.sparse_weights["a"]-.2375)<1e-12
                assert abs(distribution.sparse_weights["b"]-.2375)<1e-12
        assert StructuredProposalDistribution(method).tables == {}
        # A different legal mask has a distinct table initialized to q0.
        masked = proposal.table("mode", ("mode", ("c",)))
        assert masked.prior_weight == 1 and not masked.sparse_weights
        evidence[method] = {"updated_layers": changed, "fresh_state": True,
            "mask_identity_separate": True, "remaining_layers_uniform": True}
    a = (0, 999., 999., 999., 999.)
    b = (1, 0., 0., 0., 0.)
    assert objective_sort_key(a, "z") < objective_sort_key(b, "a")
    assert objective_sort_key(a, "a") < objective_sort_key(a, "z")
    return {"verdict": "PASS", "eta": .5, "epsilon": .05,
        "q0": [1/3]*3, "elite": [.25,.75,0], "first_expected_fraction": {x:str(y) for x,y in expected.items()},
        "first_actual": first, "second_actual": second, "second_oracle": oracle,
        "strict_objective_before_fingerprint": True, "method_layers": evidence,
        "mask_mechanism": "Exact legal-option mask is part of the table key; changed masks get fresh normalized q0. No illegal sparse mass transfers."}


def repeated_elite_counterexample():
    """Exercise actual solver with controlled proposals, not a changed algorithm.

    Two scoreable candidates recur from cache; other branches spend the quota
    on distinct unscoreable paths. No newly discovered scoreable H4 after row1.
    """
    output = {}
    for method in ("S-CEM", "MH-CEM"):
        anchor, catalog, transition, _ = synthetic_anchor()
        alternate = ("rows:1", "tasks:1;alpha:0.75", "HOLD,HOLD")
        signatures = catalog.joint | {alternate}
        catalog = replace(catalog, joint=frozenset(signatures),
            family_marginal=tuple(frozenset(s[i] for s in signatures) for i in range(3)))
        draws = 0
        scored = set()
        update_observations = []
        actual_update = StructuredProposalDistribution.update_from_elites

        def controlled(domain, proposal, rng):
            nonlocal draws
            # Batch has 16 branches; wave H1 slots are fixed, later depths vary
            # except for slots 0/1 which repeat their previously scoreable paths.
            wave, offset = divmod(draws, 64)
            depth, slot = divmod(offset, 16)
            start = slot if depth == 0 else 0 if slot < 2 else (wave + 1) % 50
            draws += 1
            bound = bind_structured_step(StructuredStepChoice((CommBlockChoice("input", start, 1),),
                "SCALE_0.5", "PROFILE_HOLD"), domain.context, domain.operational_domain,
                domain.state, domain.mobility_control, domain.catalog, domain.prior_signatures)
            return bound, {"draws": [
                (("mode", tuple(m.signature for m in domain.modes)), bound.structural_signature),
                (("count", bound.structural_signature, (1,)), 1),
                (("start", "controlled"), start)]}

        def score(candidate, trace):
            starts = [int(s.comm[0]["rb_indices"][0]) for s in candidate.steps]
            if starts[0] in (0,1) and starts[1:] == [0,0,0]:
                scored.add(candidate.fingerprint)
                return SimpleNamespace(objective_tuple=(starts[0],0.,0.,0.,0.), candidate_fingerprint=candidate.fingerprint)
            return None

        def spy(proposal, elites):
            prior = {repr(k):v.snapshot() for k,v in proposal.tables.items()}
            actual_update(proposal, elites)
            after = {repr(k):v.snapshot() for k,v in proposal.tables.items()}
            update_observations.append({"globally_distinct_scoreable_so_far": len(scored),
                "elite_count":len(elites), "proposal_tables_changed":prior!=after,
                "mode_q0_weight_after": [v.prior_weight for k,v in proposal.tables.items() if k[0]=="mode"]})

        with patch("pi_jwm.step6_3d_fixed_budget_search_v1.sample_structured_step", controlled), \
             patch.object(StructuredProposalDistribution, "update_from_elites", spy):
            result = solve_fixed_budget(method=method, seed=6301, b_wm=256, anchor=anchor,
                catalog=catalog, transition=transition, score_h4=score, iterations=4, elite_ratio=.1,
                batch_size=16, transition_batch=lambda req: tuple(transition(n,b) for n,b in req))
        assert len(scored) == 2 and len(update_observations) > 1
        assert all(row["proposal_tables_changed"] for row in update_observations)
        assert any(row["proposal_updated"] for row in result.iteration_rows[1:])
        output[method] = {"iteration_rows": result.iteration_rows, "update_calls": update_observations,
            "total_globally_new_scoreable": len(scored), "new_scoreable_after_first_iteration": 0,
            "cache_hits": result.budget_receipt["N_cache_hits"]}
    anchor, catalog, transition, scorer = synthetic_anchor()
    accepted = None
    def only_one(candidate, trace):
        nonlocal accepted
        if accepted is None:
            accepted = candidate.fingerprint
        return scorer(candidate, trace) if candidate.fingerprint == accepted else None
    guard = solve_fixed_budget(method="S-CEM", seed=6301, b_wm=256,
        anchor=anchor, catalog=catalog, transition=transition, score_h4=only_one,
        iterations=4, elite_ratio=.1, batch_size=16,
        transition_batch=lambda req:tuple(transition(n,b) for n,b in req))
    assert guard.h4_scoreable_count==1 and not any(r["proposal_updated"] for r in guard.iteration_rows)
    return {"verdict": "BLOCKED_PENDING_NEWLY_SCOREABLE_DEFINITION", "evidence": output,
        "fewer_than_two_current_distinct_guard":"PASS: sole scoreable fingerprint fixture never updates",
        "documented_intent": "<2 newly scoreable H4 => proposal must not update",
        "actual_implementation": "unique_current counts distinct scoreable candidates appearing this iteration, including previously scored/cache-replayed H4",
        "conflict": "If newly means newly discovered in this solve, the update gate violates the instruction. If it means newly completed this iteration, it can pass. No semantic choice is made by the audit.",
        "retained_cap": 5, "current_and_retained_pool_deduplicated": True,
        "status": "Awaiting Researcher Decision", "scientific_sources_modified": False}


def fairness():
    anchor, catalog, transition, _ = synthetic_anchor()
    node = SearchNode.from_anchor(anchor.latent, anchor.state, anchor.graph,
        anchor.domain.mobility_states, anchor.context.causal_provenance)
    domain = CandidateDomain.from_state(anchor.context, anchor.domain, anchor.state,
        anchor.domain.mobility_states, catalog)
    bounds = list(domain.iter_bound())[:3]
    receipts = {}
    for method in METHODS:
        account = TransitionBudgetAccountant(2)
        forwards = []
        def batch(req):
            forwards.append(len(req))
            return tuple(transition(n,b) for n,b in req)
        account.evaluate_batch([(node,bounds[0]),(node,bounds[0]),(node,bounds[1])], batch)
        account.evaluate_batch([(node,bounds[1]),(node,bounds[0])], batch)
        try:
            account.evaluate_batch([(node,bounds[2])], batch)
        except RuntimeError as error:
            assert str(error) == "B_WM_EXHAUSTED"
        else:
            raise AssertionError("budget was exceeded")
        receipt = account.receipt()
        receipt.pop("wall_clock_seconds_diagnostic_only")
        assert receipt["N_unique_transition_evals"] == 2 and receipt["N_cache_hits"] == 3
        assert forwards == [2]
        receipts[method] = {"receipt": receipt, "forward_batch_members": forwards}
    assert receipts["HRS"] == receipts["S-CEM"] == receipts["MH-CEM"]
    return {"verdict":"PASS", "paired_explicit_requests": receipts,
        "cache_key": ["causal_provenance", "parent_latent_state_graph_fingerprint", "canonical_action_fingerprint"],
        "source_path_shared": "solve_fixed_budget -> TransitionBudgetAccountant.evaluate_batch -> transition_batch",
        "scope": "Accounting equality for identical requests; method-specific proposals need not have identical cache hit rates."}


def rng_quota_order():
    before_global = random.getstate()
    old = {(m,b):discrete(solve_fixture(m,b)) for b in BUDGETS for m in METHODS}
    # Insert unrelated anchor, method and seed solves between schedules.
    solve_fixture("MH-CEM",256,"TRAIN_ONLY_SYNTHETIC_B",6302)
    new = {(m,b):discrete(solve_fixture(m,b)) for b in (1024,256,512) for m in reversed(METHODS)}
    assert old == new and before_global == random.getstate()
    rows = []
    for (m,b), case in old.items():
        assert case["complete_sequence_count"] > 0
        assert case["budget_receipt"]["N_unique_transition_evals"] == b
        assert all(r["retained_count"] <= 5 for r in case["iteration_rows"])
        allocation = quotas(b, 1 if m == "HRS" else 4)
        assert min(allocation) >= 16 * 4
        rows.append({"method":m,"budget":b,"quotas":allocation,"complete_h4":case["complete_sequence_count"],
            "scoreable_distinct":case["h4_scoreable_count"], "discrete_sha256":hashlib.sha256(
                json.dumps(case,sort_keys=True).encode()).hexdigest()})
    paths = [str(result_path("TRAIN_ONLY_SYNTHETIC", "fixture", m,6301,b,1 if m=="HRS" else 4,
                None if m=="HRS" else .1).relative_to(ROOT)) for b in BUDGETS for m in METHODS]
    assert len(set(paths))==9
    fields = {"sample_id":"fixture","method":"S-CEM","seed":6301,"budget":1024,"iterations":4,"elite_ratio":.1}
    validate_resume_result(fields,fields,{})
    try:
        validate_resume_result({**fields,"budget":256},fields,{})
    except ValueError:
        mismatch_rejected=True
    else:
        raise AssertionError("resume budget mismatch accepted")
    return {"verdict":"PASS", "rows":rows,"matched_fields":"all SearchOutcome fields excluding elapsed timer only",
        "orders":[[256,512,1024],[1024,256,512]], "independent_anchor_seed_interleaved":True,
        "global_python_rng_unchanged":True,"resume_namespace_unique":True,"resume_budget_mismatch_rejected":mismatch_rejected,
        "scope":"Unmodified solver and real Grammar/CandidateDomain on synthetic TRAIN-only transitions; no frozen-model/GPU replay or Validation outcome."}


def selection_oracle():
    left=(0,1.,2.,3.,4.)
    right=(1,0.,0.,0.,0.)
    assert [paired_outcome(*p) for p in ((left,None),(None,right),(left,right),(right,left),(left,left),(None,None))] == [1,-1,1,-1,0,0]
    scenarios=[(.1,-.1,0.,"S-CEM"),(-.1,.1,0.,"MH-CEM"),(.1,.1,.1,"MH-CEM"),(.1,.1,0.,"S-CEM"),(0.,-.1,.1,"HRS")]
    observed=[]
    for s,mh,ms,expected in scenarios:
        ci={pair:{"ci95_lower":low} for pair,low in zip((("S-CEM","HRS"),("MH-CEM","HRS"),("MH-CEM","S-CEM")),(s,mh,ms))}
        actual=select_method(ci)
        assert actual==expected
        observed.append({"lower_bounds":[s,mh,ms],"expected":expected,"actual":actual})
    clusters={f"anchor_{a:02d}":[(a+i)%3-1 for i in range(5)] for a in range(64)}
    actual=cluster_bootstrap_advantage(clusters)
    # Independent explicit cluster resampling oracle, preserving all five outcomes.
    rng=random.Random(6316)
    anchors=sorted(clusters)
    samples=[]
    for _ in range(10000):
        selected=[anchors[rng.randrange(64)] for _ in range(64)]
        samples.append(sum(sum(clusters[a])/5 for a in selected)/64)
    samples.sort()
    assert actual["ci95_lower"]==samples[250] and actual["ci95_upper"]==samples[9750]
    return {"verdict":"PASS","pairwise_six_categories_oracle":True,"scenarios":observed,
        "bootstrap_oracle":actual,"lower_bound_strict_positive":True,
        "superiority_language":"passes pre-registered superiority criterion; no family-wise claim"}


def train_sanity():
    paths=sorted((TRAIN/"solve_results/train").glob("*.json"))
    inventory=json.loads((TRAIN/"train_local_backup_inventory.json").read_text(encoding="utf-8"))
    # All current scientific-source hashes must still match the accepted TRAIN identity.
    frozen=json.loads((ROOT/"code/artifacts/protocols/pi_jwm_step6_3d_3080ti_migration_v1_20260930/05_selected_execution_config.json").read_text(encoding="utf-8"))
    assert source_hashes()==frozen["source_sha256"]
    assert len(paths)==768
    expected_shas={row["path"]:row["sha256"] for row in inventory["files"]}
    paired={}
    for path in paths:
        assert sha(path)==expected_shas[path.relative_to(TRAIN).as_posix()]
        case=json.loads(path.read_text(encoding="utf-8"))
        assert case["source_sha256"]==frozen["source_sha256"] and case["locked_test"] is False
        if case["iterations"]==4 and case["elite_ratio"]==.1:
            key=(case["sample_id"],case["seed"])
            paired.setdefault(key,{})[case["method"]]=case["outcome"]
    assert len(paired)==96 and all(set(p)=={"S-CEM","MH-CEM"} for p in paired.values())
    per_anchor={}
    totals=Counter()
    differences=0
    for (anchor,seed), pair in sorted(paired.items()):
        value=paired_outcome(pair["MH-CEM"]["best_objective"],pair["S-CEM"]["best_objective"])
        totals[{1:"win",0:"tie",-1:"loss"}[value]]+=1
        per_anchor.setdefault(anchor,[]).append({"seed":seed,"outcome":value})
        differences+=int(pair["MH-CEM"]["best_fingerprint"]!=pair["S-CEM"]["best_fingerprint"])
    return {"verdict":"PASS_NO_IDENTICAL_BEHAVIOR_EVIDENCE", "case_count":96,
        "win":totals["win"],"tie":totals["tie"],"loss":totals["loss"],"different_best_fingerprints":differences,
        "per_anchor":{a:{"paired_seed_rows":v,"mean_advantage":sum(x["outcome"] for x in v)/3} for a,v in per_anchor.items()},
        "raw_input_sha256":{str(p.relative_to(ROOT)):sha(p) for p in paths},
        "claim_boundary":"TRAIN diagnostic only; not method selection or Validation evidence"}


def schema_receipt():
    return {"verdict":"PREREGISTERED_RUNNER_NOT_CONFORMANT", "schema_version":"PIJWM-Validation-summary-v1",
        "primary_budget":1024,"methods":list(METHODS),"budgets":list(BUDGETS),"anchors":64,"seeds":[6311,6312,6313,6314,6315],
        "per_method_budget_required":{
            "case_count":"320", "h4_scoreable_success_count":"best_objective != null count", "h4_scoreable_success_rate":"count / 320",
            "total_complete_h4":"sum outcome.complete_sequence_count (completion attempts)",
            "total_distinct_scoreable_h4":"sum outcome.h4_scoreable_count (deduplicated within solve)",
            "total_unscoreable_h4":"sum outcome.h4_unscoreable_count (completion attempts)",
            "return_birth_support_boundary_count":"sum score_residuals counts whose key contains UNSUPPORTED_FUTURE_RETURN_BIRTH; repeated scorer events",
            "grammar_dead_end_count":"sum budget_receipt.N_dead_end_branches",
            "scorer_exception_count":"sum support_horizon_counts.SCORER_EXCEPTION; do not double count score_residuals",
            "N_unique_transition_evals":"sum budget_receipt.N_unique_transition_evals; separate nominal B_WM",
            "cache_hits":"sum budget_receipt.N_cache_hits", "wall_clock_diagnostic":"sum in-solve timers plus separately logged actual start/finish/elapsed"},
        "pairwise_partition":["only_left_scoreable","only_right_scoreable","both_scoreable_left_win","both_scoreable_right_win","both_scoreable_tie","both_unscoreable"],
        "partition_invariant":"sum six mutually exclusive bins = total_paired_cases = 320",
        "comparisons":[["S-CEM","HRS"],["MH-CEM","HRS"],["MH-CEM","S-CEM"]],
        "statistics":{"raw_win_tie_loss":True,"anchor_cluster_mean_advantage":True,"percentile_ci95":True,
            "cluster_unit":"anchor; retain five paired seed outcomes","replications":10000,"seed":6316},
        "sensitivity_only":{"optional_holm":"NOT_SELECTED: no preregistered bootstrap p-value definition; no automatic Holm test",
            "simultaneous_diagnostic_plan":"For the three primary comparisons, reuse 10000 draws of 64 anchor clusters with seed6316; retain five paired seeds and the same draw indices across comparisons. Report per-comparison Bonferroni percentile 98.333333% intervals (tail 0.05/(2*3), indices floor(tail*10000) and floor((1-tail)*10000)). Diagnostic only; no guaranteed finite-sample simultaneous coverage or change to primary selection.",
            "changes_primary_selection":False},
        "nonadditive_count_warning":"complete counts paths; scoreable counts distinct fingerprints; unscoreable counts paths. Do not claim complete=distinct_scoreable+unscoreable.",
        "current_runner_missing":["six-way pairwise partition and 320 invariant","Return-birth boundary counts","scorer exception counts","1024 Stage-A-only completion and stop boundary"],
        "validation_outcomes_read":False}


def main():
    frozen=json.loads((ROOT/"code/artifacts/protocols/pi_jwm_step6_3d_3080ti_migration_v1_20260930/05_selected_execution_config.json").read_text(encoding="utf-8"))
    assert source_hashes()==frozen["source_sha256"], "Current source differs from formally accepted TRAIN: stop"
    validation_files=list((TRAIN/"solve_results/validation").rglob("*.json"))
    assert not validation_files, "Validation filenames exist: audit stops without opening outcomes"
    assert not (TRAIN/"07_validation_paired_comparison_receipt.json").exists()
    assert not (TRAIN/"08_selected_method.json").exists()
    math=math_and_layers()
    write("01_cem_math_oracle_receipt.json",math)
    elite=repeated_elite_counterexample()
    write("02_elite_update_gate_receipt.json",elite)
    write("03_b_wm_fairness_receipt.json",fairness())
    order=rng_quota_order()
    write("04_rng_order_independence_receipt.json",order)
    write("05_budget_quota_receipt.json",{"verdict":"PASS","batch_size":16,"minimum_new_transitions_per_full_wave_upper_bound":64,
        "rows":order["rows"],"claim":"All iterations can accommodate one 16-branch H1-H4 wave if Grammar remains legal; this does not guarantee scoreability on a real anchor."})
    write("06_method_selection_oracle_receipt.json",selection_oracle())
    write("07_validation_output_schema_receipt.json",schema_receipt())
    sanity=train_sanity()
    write("08_train_mh_vs_s_diagnostic.json",sanity)
    write("09_pre_validation_audit_receipt.json",{
        "PRE_VALIDATION_AUDIT":"BLOCKED","VALIDATION_DECISION":"VALIDATION_NO_GO",
        "baseline_commit":"b9b597a654c7c07105271b1cbb1103dc9bc10dab",
        "HRS_implementation":"PASS","S_CEM_layer_implementation":"PASS","MH_CEM_layer_implementation":"PASS",
        "S_CEM_full_implementation":"BLOCKED_NEWLY_SCOREABLE_SEMANTICS","MH_CEM_full_implementation":"BLOCKED_NEWLY_SCOREABLE_SEMANTICS",
        "cem_update_math":"PASS","elite_update_gate":"AWAITING_RESEARCHER_DECISION","batch_quota_h4":"PASS",
        "B_WM_cache_fairness":"PASS","rng_independence":"PASS_BOUNDED_SYNTHETIC",
        "execution_order_independence":"PASS_BOUNDED_SYNTHETIC_AND_NAMESPACE",
        "recommended_order_if_blockers_resolved":[1024,256,512],"current_runner_order":[256,512,1024],
        "VALIDATION_EXECUTION_ORDER":"UNCHANGED_PENDING_NO_GO_RESOLUTION",
        "method_selection_oracle":"PASS","schema":"PREREGISTERED_RUNNER_NOT_CONFORMANT",
        "blockers":["Meaning of newly scoreable H4 is unresolved; cached previous candidates trigger later CEM updates in a deterministic counterexample.",
            "Formal runner does not implement the preregistered six-way paired partition, Return-birth/scorer exception summaries, or Stage-A-only stop boundary."],
        "VALIDATION_RESULT_COUNT":0,"validation_outcomes_read":False,"gpu_search_started":False,"locked_test":False,
        "TRAIN_TUNING_CLOSURE":"PASS_IDENTITY_AND_FROZEN_RULE_UNCHANGED; does not establish this new semantic gate",
        "SEARCH_METHOD":"NOT_SELECTED","VALIDATION_COMPARISON":"NOT_STARTED",
        "H4_SEARCH_COMPARISON_READINESS":"SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION",
        "TRAIN_MH_VS_S":{k:sanity[k] for k in ("win","tie","loss")},
        "source_sha256":source_hashes(),"scientific_sources_modified":False,
        "stage_plan_status":"NOT_ACTIVATED; after research clarification and authorized implementation repair, re-audit; no automatic run",
        "future_stage_A":{"budget":1024,"cases":960,"nominal_transitions":983040,"methods":list(METHODS),"cem_configs":{"S-CEM":[4,.1],"MH-CEM":[4,.1]},"requires_authorization":True,"stop_after_completion_for_researcher_review":True},
        "future_stage_B":{"budgets":[256,512],"requires_separate_authorization":True,"auto_start":False}})
    manifest={p.name:sha(p) for p in sorted(OUT.glob("*.json")) if p.name!="10_sha_manifest.json"}
    manifest["code/scripts/audit_step6_3d_pre_validation_v1.py"]=sha(Path(__file__))
    manifest["code/tests/test_step6_3d_pre_validation_audit_v1.py"]=sha(ROOT/"code/tests/test_step6_3d_pre_validation_audit_v1.py")
    write("10_sha_manifest.json",{"schema_version":"PIJWM-pre-validation-audit-sha-v1","files":manifest,"locked_test":False})
    print(json.dumps({"verdict":"VALIDATION_NO_GO","validation_results":0,"TRAIN_MH_VS_S":{k:sanity[k] for k in ("win","tie","loss")}},ensure_ascii=False))


if __name__=="__main__":
    main()
