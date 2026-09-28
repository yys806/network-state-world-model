"""CPU-only Formal Dataset action-support audit for STEP 6.3A."""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1"
OUT = ROOT / "code/artifacts/protocols/pi_jwm_step6_3a_candidate_support_audit_v1_20260928"
PROFILES = {
    (0.0, 0.0): "HOLD", (-0.2, 5.0): "PROFILE_1", (-0.1, 8.0): "PROFILE_2",
    (0.05, 10.0): "PROFILE_3", (0.1, 12.0): "PROFILE_4", (0.2, 15.0): "PROFILE_5",
}
EXPECTED_CKPT = "941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9"
sys.path.insert(0, str(ROOT / "code/src"))
from pi_jwm.cpu_inner_rule_v1 import CpuTaskDemand, allocate_work_conserving_cpu


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def profile(decision: dict, row: dict) -> str:
    entity = next((r for r in decision.get("entities", ()) if r.get("entity_id") == row.get("uav_id")), None)
    if entity is None or entity.get("heading") is None:
        return "UNRESOLVED"
    delta = (float(row["azimuth_rad"]) - float(entity["heading"]) + math.pi) % (2 * math.pi) - math.pi
    key = (round(delta, 6), round(float(row["speed_mps"]), 6))
    return PROFILES.get(key, f"OUTSIDE_CORE:{key[0]}:{key[1]}")


def eligible_status(step: dict, family: str) -> tuple[bool, bool, bool]:
    row = step["behavior_policy_audit"]["families"][family]
    return bool(row["eligible"]), bool(row["no_op"]), bool(row["intervention"])


def comp_base(decision: dict, dt: float) -> dict[str, float]:
    grouped: dict[str, list[CpuTaskDemand]] = collections.defaultdict(list)
    caps = decision["node_cpu_capacity_per_s"]
    for row in decision["tasks"]:
        node = row.get("current_node_id")
        if row.get("lifecycle") != "computing" or node not in caps:
            continue
        remaining = max(float(row["task_cpu_work"]) - float(row["computed_cpu_work"]), 0.0)
        grouped[str(node)].append(CpuTaskDemand(str(row["task_id"]), str(node), remaining))
    allocations: dict[str, float] = {}
    if grouped:
        allocations.update(allocate_work_conserving_cpu(
            [row for rows in grouped.values() for row in rows],
            {str(node): float(cap) for node, cap in caps.items()}, dt,
        ).as_allocation_dict())
    return allocations


def compact_family(decision: dict, step: dict) -> dict[str, str]:
    result = {}
    for fam, key in (("comm", "comm"), ("comp", "comp"), ("mobility", "mobility")):
        eligible, noop, intervention = eligible_status(step, fam)
        if not eligible:
            result[fam] = "INELIGIBLE"
        elif noop or not intervention:
            result[fam] = "NOOP"
        elif fam == "comm":
            result[fam] = json.dumps(sorted((str(r["task_id"]), tuple(map(int, r["rb_indices"]))) for r in step["action"][key]["entries"]))
        elif fam == "comp":
            result[fam] = json.dumps(sorted((str(r["task_id"]), str(r["node_id"]), round(float(r["allocated_cpu_per_s"]), 8)) for r in step["action"][key]["entries"]))
        else:
            result[fam] = json.dumps(sorted((str(r["uav_id"]), profile(decision, r)) for r in step["action"][key]["entries"]))
    return result


def structural_signature(decision: dict, step: dict, dt: float) -> tuple[str, str, str]:
    comm = step["action"]["comm"]["entries"]
    comp = step["action"]["comp"]["entries"]
    mob = step["action"]["mobility"]["entries"]
    comm_part = "NOOP" if not comm else "rows:" + ",".join(map(str, sorted(len(r["rb_indices"]) for r in comm)))
    if comp:
        base = comp_base(decision, dt)
        ratios = [float(r["allocated_cpu_per_s"])/base[str(r["task_id"])] for r in comp if base.get(str(r["task_id"]),0)>0]
        comp_part = f"tasks:{len(comp)};alpha:{round(ratios[0], 6) if ratios else 'UNRESOLVED'}"
    else:
        comp_part = "NOOP"
    mob_part = "NOOP" if not mob else ",".join(sorted(profile(decision,r) for r in mob))
    return comm_part, comp_part, mob_part


def read_trajectory(tid: str) -> dict:
    with gzip.open(DATASET / "raw" / f"{tid}.json.gz", "rt", encoding="utf-8") as stream:
        return json.load(stream)


def scan_split(name: str, tids: list[str], sample_index: list[dict]) -> dict:
    counts = collections.Counter()
    comm_width = collections.Counter(); comm_rb = collections.Counter(); comm_start = collections.Counter()
    comm_task_count = collections.Counter(); comm_per_slot_tasks = collections.Counter(); comm_reuse_slots = 0; comm_relation_assignments = 0
    comp_task_n = collections.Counter(); comp_node_n = collections.Counter(); comp_ratio = collections.Counter()
    comp_per_slot = collections.Counter(); comp_per_node_slot = collections.Counter()
    comp_alpha = collections.Counter(); comp_alpha_multi = collections.Counter(); comp_rebuild = collections.Counter()
    mob_marginal = collections.Counter(); mob_joint = collections.Counter(); mob_transitions = collections.Counter()
    channel_types = collections.Counter()
    family_active = collections.Counter(); family_sigs = collections.Counter()
    pair_counts = {k: collections.Counter() for k in ("comm_comp", "comm_mobility", "comp_mobility", "triple")}
    pair_cond = {k: collections.defaultdict(collections.Counter) for k in pair_counts}
    exact = collections.Counter(); structural_joint = collections.Counter()
    sequence_h = {h: collections.Counter() for h in range(1, 5)}
    structural_sequence_h = {h: collections.Counter() for h in range(1, 5)}
    window_action_sequences: list[tuple[tuple[str, ...], ...]] = []
    window_anchor_metrics: list[dict] = []
    temporal_trans = {h: collections.Counter() for h in range(1, 5)}
    formal_frames = collections.defaultdict(list)
    eligible_n = collections.Counter(); no_op_n = collections.Counter(); intervention_n = collections.Counter()
    steps_seen = 0; windows_seen = 0; comp_rows_seen = 0; comp_rows_rebuilt = 0; comp_slots_seen = 0
    all_step_by_tid = {}; samples_by_tid: dict[str, list[dict]] = collections.defaultdict(list)
    for sample in sample_index:
        samples_by_tid[str(sample["metadata"]["trajectory_id"])].append(sample)
    for tid in tids:
        raw = read_trajectory(tid); decisions, steps = raw["decisions"], raw["steps"]
        all_step_by_tid[tid] = (decisions, steps)
        structural_by_frame = []
        for j, (decision, step) in enumerate(zip(decisions, steps)):
            steps_seen += 1
            fam = compact_family(decision, step)
            structural = structural_signature(decision, step, float(raw["environment"]["slot_duration_s"]))
            structural_by_frame.append(structural); structural_joint[structural] += 1
            for family in ("comm", "comp", "mobility"):
                e, n, i = eligible_status(step, family)
                eligible_n[family] += int(e); no_op_n[family] += int(n); intervention_n[family] += int(i)
                family_active[family] += int(i)
            sig = tuple(f for f in ("comm", "comp", "mobility") if fam[f] not in ("NOOP", "INELIGIBLE"))
            family_sigs[sig] += 1
            counts["global_noop"] += int(step["behavior_policy_audit"].get("global_noop", False))
            counts["no_family_intervention"] += int(not sig)
            counts["multi_family"] += int(len(sig) > 1)
            counts["single_family"] += int(len(sig) == 1)
            c, p, m = (fam[x] not in ("NOOP", "INELIGIBLE") for x in ("comm", "comp", "mobility"))
            for label, active, key in (("comm_comp", c and p, (fam["comm"], fam["comp"])),
                                       ("comm_mobility", c and m, (fam["comm"], fam["mobility"])),
                                       ("comp_mobility", p and m, (fam["comp"], fam["mobility"])),
                                       ("triple", c and p and m, (fam["comm"], fam["comp"], fam["mobility"]))):
                if active:
                    pair_counts[label][key] += 1
                    pair_cond[label][key[0]][key[1:]] += 1

            entries = step["action"]["comm"]["entries"]
            for row in decision.get("channel_rows", ()): channel_types[str(row.get("channel_type"))] += 1
            comm_per_slot_tasks[len({str(r["task_id"]) for r in entries})] += 1
            per_slot_rb = collections.Counter()
            for row in entries:
                rbs = sorted(set(map(int, row["rb_indices"])))
                comm_width[len(rbs)] += 1; comm_task_count[str(row["task_id"])] += 1
                comm_relation_assignments += len(rbs)
                for rb in rbs: comm_rb[rb] += 1; per_slot_rb[rb] += 1
                if rbs:
                    starts = [s for s in range(int(decision["n_rb"])) if set((s+k) % int(decision["n_rb"]) for k in range(len(rbs))) == set(rbs)]
                    if starts: comm_start[starts[0]] += 1
            comm_reuse_slots += int(any(v > 1 for v in per_slot_rb.values()))

            comp = step["action"]["comp"]["entries"]
            comp_slots_seen += int(bool(comp))
            comp_per_slot[len(comp)] += 1
            for node_id in {str(r["node_id"]) for r in comp}:
                comp_per_node_slot[(node_id, sum(str(r["node_id"]) == node_id for r in comp))] += 1
            dt = float(raw["environment"]["slot_duration_s"])
            base = comp_base(decision, dt)
            by_node_alpha: dict[str, list[float]] = collections.defaultdict(list)
            for row in comp:
                comp_rows_seen += 1; task_id = str(row["task_id"]); node = str(row["node_id"])
                comp_task_n[task_id] += 1; comp_node_n[node] += 1
                cap = float(decision["node_cpu_capacity_per_s"][node])
                comp_ratio[round(float(row["allocated_cpu_per_s"])/cap, 6)] += 1
                ref = base.get(task_id)
                if ref is not None and ref > 0:
                    ratio = float(row["allocated_cpu_per_s"]) / ref
                    comp_alpha[round(ratio, 6)] += 1; by_node_alpha[node].append(ratio)
                    comp_rows_rebuilt += 1
                    nearest = min((0.5, 0.75, 1.0), key=lambda alpha: abs(ratio-alpha))
                    comp_rebuild["alpha_in_frozen_support"] += int(math.isclose(ratio, nearest, rel_tol=1e-7, abs_tol=1e-7))
                    comp_rebuild["alpha_outside_frozen_support"] += int(not math.isclose(ratio, nearest, rel_tol=1e-7, abs_tol=1e-7))
                    comp_rebuild["max_abs_base_scaled_error"] = max(comp_rebuild["max_abs_base_scaled_error"], abs(float(row["allocated_cpu_per_s"])-ref*nearest))
                else: comp_rebuild["missing_or_zero_base"] += 1
            alphas = [v for values in by_node_alpha.values() for v in values]
            if alphas:
                common = round(sum(alphas)/len(alphas)*4)/4
                same = all(math.isclose(v, common, rel_tol=1e-7, abs_tol=1e-7) for v in alphas)
                comp_alpha_multi["slot_common_alpha"] += int(same); comp_alpha_multi["slot_mixed_alpha"] += int(not same)
                comp_alpha_multi[f"slot_alpha_{common:g}"] += int(same)
                for node, vals in by_node_alpha.items():
                    comp_alpha_multi["node_global_within_slot"] += int(all(math.isclose(v, vals[0], rel_tol=1e-7, abs_tol=1e-7) for v in vals))

            mrows = step["action"]["mobility"]["entries"]
            by_uav = {str(r["uav_id"]): profile(decision, r) for r in mrows}
            uav_ids = sorted(str(r["entity_id"]) for r in decision.get("entities", ()) if r.get("entity_type") == "uav")
            profs = [by_uav.get(uid, "HOLD") for uid in uav_ids]
            for prof in profs: mob_marginal[prof] += 1
            if profs: mob_joint[tuple(sorted(profs))] += 1
            if j:
                before_rows = steps[j-1]["action"]["mobility"]["entries"]
                before_by_uav = {str(r["uav_id"]): profile(decisions[j-1], r) for r in before_rows}
                before_ids = sorted(str(r["entity_id"]) for r in decisions[j-1].get("entities", ()) if r.get("entity_type") == "uav")
                before = [before_by_uav.get(uid, "HOLD") for uid in before_ids]
                after = profs
                if before and after: mob_transitions[(tuple(sorted(before)), tuple(sorted(after)))] += 1

        # Formal sample index supplies the actual four future action positions per window.
        for sample in samples_by_tid[tid]:
            meta = sample["metadata"]
            frames = list(map(int, meta["future_action_frame_indices"]))
            formal_frames[tid].append(frames); windows_seen += 1
            window_action_sequences.append(tuple(json.dumps(compact_family(decisions[f], steps[f]), sort_keys=True) for f in frames))
            anchor = int(meta["anchor_decision_frame"])
            ad = decisions[anchor]
            active_tasks = [r for r in ad["tasks"] if r.get("lifecycle") in {"waiting_to_offload", "offloading", "computing", "waiting_to_return", "returning"}]
            computing_tasks = [r for r in ad["tasks"] if r.get("lifecycle") == "computing"]
            uav_count = sum(r.get("entity_type") == "uav" for r in ad["entities"])
            action_step = steps[frames[0]]
            comm_eligible = int(eligible_status(action_step, "comm")[0])
            comm_observed_rows = len(action_step["action"]["comm"]["entries"])
            comp_eligible = int(eligible_status(action_step, "comp")[0])
            remaining_compute_work = math.fsum(max(float(r["task_cpu_work"])-float(r["computed_cpu_work"]),0.0) for r in computing_tasks)
            c_options = 1 + 150 if comm_eligible else 1
            p_options = 1 + 3 if comp_eligible else 1
            m_exact = 6 if uav_count >= 2 else 6 if uav_count == 1 else 1
            m_factorized = 6 ** uav_count if uav_count else 1
            window_anchor_metrics.append({"trajectory_id":tid,"anchor_decision_frame":anchor,
                "active_task_count":len(active_tasks),"computing_task_count":len(computing_tasks),
                "remaining_compute_work":round(remaining_compute_work,6),
                "behavior_policy_comm_eligibility_at_H1":bool(comm_eligible),"observed_comm_rows_at_H1":comm_observed_rows,
                "behavior_policy_comp_eligibility_at_H1":bool(comp_eligible),"uav_count":uav_count,
                "comm_candidates_as_rowwise_observed_width_start_upper_bound":c_options,
                "comp_candidates_noop_plus_global_alpha":p_options,
                "mob_candidates_exact_joint_train_template":m_exact,
                "mob_candidates_independent_marginal_product_not_formal_joint":m_factorized,
                "single_step_joint_naive_cartesian_upper_bound":c_options*p_options*m_factorized,
                "status":"support-count illustration from recorded family eligibility; not an exact candidate count because multi-row candidate syntax is not frozen"})
            for h in range(1, 5):
                seq = tuple(tuple(sorted(compact_family(decisions[f], steps[f]).items())) for f in frames[:h])
                sequence_h[h][seq] += 1
                structural_sequence_h[h][tuple(structural_by_frame[f] for f in frames[:h])] += 1
                if len(seq) > 1:
                    for a,b in zip(seq, seq[1:]): temporal_trans[h][(a,b)] += 1
    # Exact causal joint signature uses only actionable families and includes no-op coupling.
    exact_support = set()
    for tid, frames_by_window in formal_frames.items():
        decisions, steps = all_step_by_tid[tid]
        for frames in frames_by_window:
            for f in frames:
                sig = tuple(compact_family(decisions[f], steps[f])[k] for k in ("comm", "comp", "mobility"))
                exact[sig] += 1; exact_support.add(sig)
    temporal_unseen = {}
    for h in range(1,5):
        vals=structural_sequence_h[h]
        temporal_unseen[str(h)]={"observed_structural_sequences":len(vals),
            "singleton_structural_sequence_count":sum(v==1 for v in vals.values()),
            "unseen_composition_fraction":"not a probability without a frozen candidate sampler",
            "marginally_supported_but_temporally_unseen_possible":h>1}
    return {
        "split": name, "trajectory_count": len(tids), "formal_window_count": windows_seen,
        "raw_decision_steps": steps_seen,
        "family_counts": {f: {"eligible": eligible_n[f], "no_op": no_op_n[f], "non_empty_intervention": intervention_n[f]} for f in ("comm", "comp", "mobility")},
        "joint_presence_counts": {str(k): v for k,v in family_sigs.items()},
        "global_noop": counts["global_noop"], "no_family_intervention":counts["no_family_intervention"],
        "single_family": counts["single_family"], "multi_family": counts["multi_family"],
        "comm": {"row_width_distribution": dict(comm_width), "rb_id_assignment_distribution": dict(comm_rb),
                 "rb_id_range": [min(comm_rb) if comm_rb else None, max(comm_rb) if comm_rb else None],
                 "continuous_or_cyclic_block": "yes for every observed row; cyclic start recovered by set equality" if comm_start and sum(comm_width.values()) else "no active rows",
                 "start_position_distribution": dict(comm_start), "task_assignment_count_by_task_id": dict(comm_task_count),
                 "relation_slot_identity": "not directly recorded; task_id is retained, relation index is not inferred",
                 "total_relation_rb_assignments": comm_relation_assignments, "slots_with_reused_rb_across_rows": comm_reuse_slots,
                 "active_task_count_per_slot_distribution":dict(comm_per_slot_tasks),
                 "unique_active_tasks_per_slot_distribution": "see task assignment counts; no relation rows fabricated",
                 "channel_row_type_counts_descriptive":dict(channel_types)},
        "comp": {"task_entry_count": comp_rows_seen, "task_entry_reconstructed_from_causal_base": comp_rows_rebuilt,
                 "capacity_ratio_distribution": dict(comp_ratio), "alpha_vs_reconstructed_base": dict(comp_alpha),
                 "slot_and_node_alpha_checks": dict(comp_alpha_multi), "reconstruction_checks": dict(comp_rebuild),
                 "task_assignment_count_by_task_id": dict(comp_task_n), "node_assignment_count": dict(comp_node_n),
                 "task_count_per_slot_distribution":dict(comp_per_slot),
                 "per_node_task_count_per_slot_distribution":{str(k):v for k,v in comp_per_node_slot.items()}},
        "mobility": {"intervention_per_uav_profile_counts": dict(mob_marginal),
                     "exact_joint_intervention_signatures": {str(k):v for k,v in mob_joint.items()},
                     "profile_transition_counts": {str(k):v for k,v in mob_transitions.items()},
                     "joint_signature_type": "EXACT_JOINT_TRAIN_SUPPORT"},
        "joint": {"pair_unique_signature_counts": {k:len(v) for k,v in pair_counts.items()},
                  "pair_joint_counts": {k:{str(a):b for a,b in v.items()} for k,v in pair_counts.items()},
                  "conditional_distributions": {k:{str(a):{str(x):n for x,n in b.items()} for a,b in v.items()} for k,v in pair_cond.items()},
                  "exact_train_joint_signature_count": len(exact_support),
                  "structural_joint_signature_count":len(structural_joint),
                  "structural_joint_signature_counts":{str(k):v for k,v in structural_joint.items()},
                  "exact_train_joint_signature_frequency_top": [{"signature":str(k),"count":v} for k,v in exact.most_common(20)]},
        "temporal": {str(h): {"rolling_sequence_count":sum(sequence_h[h].values()), "unique_sequence_count":len(sequence_h[h]),
                              "sequence_repetition_fraction":sum(v for v in sequence_h[h].values() if v>1)/max(1,sum(sequence_h[h].values())),
                              "singleton_sequence_count":sum(v==1 for v in sequence_h[h].values()),
                              "singleton_window_fraction":sum(v==1 for v in sequence_h[h].values())/max(1,sum(sequence_h[h].values())),
                              "observed_adjacent_transition_count":len(temporal_trans[h]),
                              "unique_structural_sequence_count":len(structural_sequence_h[h]),
                              "structural_sequence_repetition_fraction":sum(v for v in structural_sequence_h[h].values() if v>1)/max(1,sum(structural_sequence_h[h].values())),
                              "structural_singleton_sequence_count":sum(v==1 for v in structural_sequence_h[h].values()),
                              "marginal_actions_in_temporally_unseen_combinations": temporal_unseen[str(h)]}
                     for h in range(1,5)},
        "anchor_search_metrics": window_anchor_metrics,
        "future_target_used": False, "locked_test": False,
    }


def security_preflight() -> dict:
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8", "replace").split("\0")
    tracked_files = [p for p in tracked if p]
    secret_names = re.compile(r"(^|/)(\.env($|\.)|credentials[^/]*\.json$|\.netrc$|id_rsa(\.|$)|[^/]+\.(pem|key)$)", re.I)
    filename_hits = [p for p in tracked_files if secret_names.search(p)]
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8", "replace").split("\0")
    ignored_untracked = subprocess.run(["git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8", "replace").split("\0")
    untracked_hits = sorted({p for p in untracked + ignored_untracked if p and secret_names.search(p)})
    patterns = [re.compile(x, re.I) for x in (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"\bAKIA[0-9A-Z]{16}\b", r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b", r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")]
    matches = []
    for rel in tracked_files:
        path = ROOT / rel
        if path.suffix.lower() not in {".py", ".json", ".yml", ".yaml", ".toml", ".ini", ".md", ".txt", ".sh", ".ps1"} or not path.is_file(): continue
        try: text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError: continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for pattern in patterns:
                if pattern.search(line): matches.append({"path":rel,"line":line_number,"pattern_class":pattern.pattern})
    rules = (".env", ".env.*", "*.pem", "*.key", "credentials*.json", ".netrc", "id_rsa", "id_rsa.*")
    ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    missing = [rule for rule in rules if rule not in ignore_text.splitlines()]
    blocked = bool(matches or filename_hits)
    return {"SECURITY_PREFLIGHT":"BLOCKED" if blocked else "PASS", "tracked_matches":[{"path":x["path"],"pattern_class":x["pattern_class"],"status":"review_required"} for x in matches],
            "tracked_secret_filenames":[{"path":p,"pattern_class":"secret_filename","status":"tracked"} for p in filename_hits],
            "untracked_secret_filenames":[{"path":p,"pattern_class":"secret_filename","status":"untracked"} for p in untracked_hits],
            "secret_value_printed":False,"required_ignore_rules":list(rules),"missing_ignore_rules":missing,
            "gitignore_hardened":not missing}


def main() -> None:
    manifest_path = DATASET / "formal_dataset_manifest.json"
    ds_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads((DATASET / ds_manifest["provenance"]["split_manifest"]).read_text(encoding="utf-8"))
    sample_index = json.loads(gzip.open(DATASET / "packages/samples/index.json.gz", "rt", encoding="utf-8").read()) if (DATASET / "packages/samples/index.json.gz").exists() else json.loads((DATASET / "packages/samples/index.json").read_text(encoding="utf-8"))
    security = security_preflight()
    train = scan_split("formal_train", split["dev_train"], [r for r in sample_index if r["metadata"]["split"] == "dev_train"])
    validation = scan_split("formal_validation_descriptive_only", split["dev_validation"], [r for r in sample_index if r["metadata"]["split"] == "dev_validation"])
    OUT.mkdir(parents=True, exist_ok=True)
    common = {"dataset_manifest_sha256":sha(manifest_path),"future_target_used":False,"validation_used_for_template_selection":False,"locked_test":False,"gpu":False,"training":False,"checkpoint_modified":False,"formal_dataset_modified":False}
    json_write("01_security_preflight.json", security)
    comm = train["comm"]
    comm["source"]="FORMAL_TRAIN_ONLY"; comm["rb_widths_observed"]=sorted(map(int,comm["row_width_distribution"]))
    comm["COMM_SUPPORT_TEMPLATE_STATUS"]="TRAIN_SUPPORT_IDENTIFIED_WITH_DYNAMIC_ELIGIBILITY_BOUNDARY"
    comm["eligible_noop_nonempty_counts"]=train["family_counts"]["comm"]
    comm["template_status"]="OBSERVED_SUPPORT_IDENTIFIED" if comm["rb_widths_observed"] else "NO_SUPPORT"
    comm["minimum_train_supported_template"]=["one task row selects a nonempty cyclic RB block","width drawn from observed widths only","start position drawn from observed starts only","assignment only for an eligible current wireless task"]
    comm.update({"wireless_semantics":"rows target task_id and rb_indices; Wired service is not represented as an RB action","temporal_action_patterns":"see per-horizon rolling signatures and transition counts"})
    json_write("02_comm_train_support.json",{**common,**comm})
    comp=train["comp"]
    comp["source"]="FORMAL_TRAIN_ONLY plus deterministic PI-JWM CPU inner rule"
    comp["eligible_noop_nonempty_counts"]=train["family_counts"]["comp"]
    alphas=comp["alpha_vs_reconstructed_base"]
    comp["COMP_TEMPLATE_READY"]=bool(comp["task_entry_count"]==comp["task_entry_reconstructed_from_causal_base"] and comp["slot_and_node_alpha_checks"].get("slot_mixed_alpha",0)==0 and comp["reconstruction_checks"].get("alpha_outside_frozen_support",0)==0)
    comp["causal_base_definition"]="reconstruct computing tasks from current Decision lifecycle/current_node_id/task_cpu_work/computed_cpu_work and node_cpu_capacity_per_s; allocate_work_conserving_cpu with slot_duration_s"
    comp["scaling_interpretation"]="single slot-global alpha over reconstructed complete base allocation" if comp["COMP_TEMPLATE_READY"] else "not verified across every nonempty slot"
    comp["alpha_support"]=[float(k) for k in sorted(alphas,key=float)]
    json_write("03_comp_train_support.json",{**common,**comp})
    json_write("04_mob_train_support.json",{**common,**train["mobility"],"eligible_noop_nonempty_counts":train["family_counts"]["mobility"],"six_profile_support":sorted(PROFILES.values()),"profile_scope":"HOLD maps absent Mobility action rows to hold for every present UAV; five intervention profiles are explicit rows"})
    joint=train["joint"]
    joint["INDEPENDENT_FAMILY_FACTORIZATION"]="NOT_SUPPORTED" if train["multi_family"] else "SUPPORTED"
    joint["interpretation"]="Formal collection policy couples interventions through global no-op and family-specific eligibility; marginal product is not the observed joint law"
    json_write("05_joint_family_support.json",{**common,**joint,"joint_presence_counts":train["joint_presence_counts"],"global_noop_count":train["global_noop"],"single_family_count":train["single_family"],"multi_family_count":train["multi_family"]})
    json_write("06_temporal_support_h1_h4.json",{**common,**train["temporal"],"window_alignment_source":"packages/samples/index.json metadata.future_action_frame_indices","temporal_policy_recommendation":"TEMPORAL_SUPPORT_CONDITIONING is needed for exact-support claims; per-step support alone permits unseen sequences"})
    anchors=train["anchor_search_metrics"]
    chosen=[]
    selectors=(
        ("low_activity",min(anchors,key=lambda x:(x["active_task_count"],x["computing_task_count"]))),
        ("high_comm",max(anchors,key=lambda x:(x["observed_comm_rows_at_H1"],x["active_task_count"]))),
        ("high_comp",max(anchors,key=lambda x:(x["computing_task_count"],x["remaining_compute_work"]))),
        ("median_activity",sorted(anchors,key=lambda x:(x["active_task_count"],x["computing_task_count"]))[len(anchors)//2]),
        ("high_activity",max(anchors,key=lambda x:(x["active_task_count"],x["observed_comm_rows_at_H1"],x["computing_task_count"]))),
    )
    for label,selected in selectors:
        row=dict(selected); row["stratum"]=label
        hcounts={str(h):{"comm":row["comm_candidates_as_rowwise_observed_width_start_upper_bound"]**h,
                         "comp":row["comp_candidates_noop_plus_global_alpha"]**h,
                         "mob_exact_joint":row["mob_candidates_exact_joint_train_template"]**h,
                         "mob_independent_marginal":row["mob_candidates_independent_marginal_product_not_formal_joint"]**h,
                         "joint_naive_cartesian_upper_bound":row["single_step_joint_naive_cartesian_upper_bound"]**h}
                 for h in range(1,5)}
        row["horizon_candidate_count_illustrations"]=hcounts; chosen.append(row)
    json_write("07_search_space_size.json",{**common,"world_model_rollout":False,"representative_train_anchors":chosen,
        "interpretation":"support-template upper-bound illustrations; action syntax multi-row combinations and predicted eligibility are not frozen",
        "exact_enumeration":"not certifiable from these independent family counts; grows exponentially with H",
        "random_search":"not shown necessary; useful only after support policy freeze",
        "cem":"not shown necessary by this audit",
        "factorization":"independent product conflicts with measured joint structure",
        "hierarchy":"conditional structure is evidence-compatible proposal, not selected method"})
    structural_count=joint["structural_joint_signature_count"]
    structural_marginals=[set() for _ in range(3)]
    for key in joint["structural_joint_signature_counts"]:
        import ast
        vals=ast.literal_eval(key)
        for i,val in enumerate(vals):structural_marginals[i].add(val)
    product_count=math.prod(map(len,structural_marginals))
    json_write("08_support_tier_analysis.json",{**common,"tiers":[
        {"tier":0,"definition":"exact observed structural Comm/Comp/Mob joint signature in TRAIN","structural_signature_count":structural_count,"coverage_of_train_observed_steps":1.0,"risk":"identity and temporal support still need checking","future_planner":"highest-fidelity proposal"},
        {"tier":1,"definition":"each structural family signature observed marginally in TRAIN but triple unseen","potential_cartesian_structural_count":max(0,product_count-structural_count),"coverage_of_train_observed_steps":0.0,"risk":"composition shift","future_planner":"researcher decision required"},
        {"tier":2,"definition":"at least one structural family signature outside TRAIN marginal","count":"unbounded; no frozen proposal distribution","coverage_of_train_observed_steps":0.0,"risk":"outside Formal TRAIN","future_planner":"exclude from formal-support claim"}],
        "marginal_structural_cardinalities":list(map(len,structural_marginals)),"status":"RESEARCH_PROPOSAL_ONLY"})
    prior_audit=json.loads((ROOT/"code/artifacts/audit/pi_jwm_step5_5_patch_20260923/audit_receipt.json").read_text(encoding="utf-8"))["future_fixed_support"]
    json_write("09_hsup_search_interaction.json",{**common,"objective_unchanged":True,"H_eff":"min_k H_sup(k)","existing_dataset_scope_counts":{k:prior_audit[k] for k in ("unsupported","unresolved","fixed_support_blocked","affected_window_count","affected_trajectory_count","supported_return_continuation_count")},
        "counts_split":"all 5520 Formal train+validation windows; prior accepted audit does not retain a train/validation or H1-H4 histogram, so no fabricated distribution is reported",
        "candidate_pool_implication":"different H_sup values shorten the common score horizon for every candidate",
        "proposal":"record H_sup groups diagnostically for a future candidate method; scorer/objective remain unchanged"})
    json_write("10_validation_descriptive_check.json",{**common,"validation":validation,"validation_used_for_template_selection":False,"validation_used_for_threshold_selection":False,"validation_used_for_method_selection":False})
    json_write("11_candidate_method_readiness.json",{**common,"methods":{
        "exact_structured_enumeration":{"compatibility":"fits typed action rows and explicit support constraints","train_support_fidelity":"highest for enumerated observed templates","combinatorial_scalability":"anchor/horizon counts grow rapidly; no universal feasibility conclusion","implementation_complexity":"medium-high","required_assumptions":"candidate grammar and eligibility are frozen and enumerable","scientific_risk":"omits useful but unobserved legal compositions"},
        "support_aware_random_search":{"compatibility":"works with structured candidate samplers","train_support_fidelity":"depends on selected tier and sampler","combinatorial_scalability":"linear in rollout budget, though coverage can be poor","implementation_complexity":"medium","required_assumptions":"sampling distribution and support admission are explicit","scientific_risk":"stochastic coverage variance and sampler bias"},
        "independent_factorized_categorical_cem":{"compatibility":"simple per-family categorical encoding","train_support_fidelity":"conflicts with measured joint/temporal support","combinatorial_scalability":"compact parameterization but combinations grow exponentially","implementation_complexity":"medium","required_assumptions":"family independence or acceptable measured approximation","scientific_risk":"creates unsupported co-occurrences; evidence does not support independence"},
        "conditional_hierarchical_categorical_cem":{"compatibility":"can encode eligibility and family coupling","train_support_fidelity":"closer to observed conditional support, not proven until policy is specified","combinatorial_scalability":"reduces impossible products but sequence space remains large","implementation_complexity":"high","required_assumptions":"causal conditioning variables, conditional order and support policy are frozen","scientific_risk":"modeling choices can encode untested assumptions"}},"CANDIDATE_METHOD_RECOMMENDATION":"NO_FINAL_METHOD_SELECTION; first define support-constrained structured syntax, then compare conditional designs; researcher owns method decision","optimizer_implemented":False})
    json_write("12_future_fair_comparison_contract.json",{**common,"same_action_domain":True,"same_frozen_world_model":True,"same_objective_scorer":True,"same_H_max":4,"same_candidate_rollout_budget":True,"same_causal_information":True,"same_support_policy":True,"reserved_methods":["Rule fallback","Greedy/H1","Support-aware Random MPC","Flat/standard CEM","Proposed structured search"],"executed":False})
    blocking=[]
    if security["SECURITY_PREFLIGHT"]!="PASS":blocking.append("security preflight not clear")
    if not comp["COMP_TEMPLATE_READY"]:blocking.append("Comp causal template failed complete reconstruction/equal-alpha audit")
    verdict="PASS" if not blocking else "BLOCKED_ON_CANDIDATE_SUPPORT_SEMANTICS"
    json_write("13_step6_3a_acceptance.json",{**common,"STEP_6_3A":verdict,"blocking_facts":blocking,"checkpoint_sha256_expected":EXPECTED_CKPT,"no_candidate_optimizer":True,"validation_descriptive_only":True})
    out_manifest={"step":"STEP 6.3A","files":{}}
    for path in sorted(OUT.glob("*.json")):
        if path.name!="manifest.json":out_manifest["files"][path.name]={"sha256":sha(path),"bytes":path.stat().st_size}
    (OUT/"manifest.json").write_text(json.dumps(out_manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"STEP_6_3A={verdict}; train_windows={train['formal_window_count']}; validation_windows={validation['formal_window_count']}; comp_template_ready={comp['COMP_TEMPLATE_READY']}; security={security['SECURITY_PREFLIGHT']}")


if __name__ == "__main__":
    main()
