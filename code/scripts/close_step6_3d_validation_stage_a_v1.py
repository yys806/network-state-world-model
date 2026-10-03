"""CPU-only independent acceptance and local archive of frozen Stage A.

Never launches the formal runner, changes source/config, or selects using Stage B.
"""
from __future__ import annotations
import collections
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
from run_step6_3d_formal_cpu_matrix_v1 import (EXECUTION_KEYS,FROZEN_CONFIGS,PAIRS,
    VALIDATION_SEEDS,selected_ids,result_path,validate_resume_result,
    validate_validation_provenance,summarize_validation_budget,sensitivity_diagnostic)
from run_step6_3d_one_cpu_solve_v1 import OUT,CHECKPOINT,EXPECTED_SHA,source_hashes

CONTROL=ROOT/'code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001'
CONFIG=ROOT/'code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/03_validation_execution_config_3080ti.json'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,value): Path(p).write_bytes((json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        while block:=stream.read(1024*1024): h.update(block)
    return h.hexdigest()
def require(ok,message):
    if not ok: raise ValueError(message)
def finite(x):
    if isinstance(x,float): return math.isfinite(x)
    if isinstance(x,dict): return all(finite(v) for v in x.values())
    if isinstance(x,list): return all(finite(v) for v in x)
    return True

def validate_stage_a_cases(rows,anchors,execution):
    require(len(anchors)==64 and len(set(anchors))==64,'64 exact anchors required')
    expected={(a,s,m) for a in anchors for s in VALIDATION_SEEDS for m in FROZEN_CONFIGS}
    indexed={}
    for r in rows:
        identity=(r['sample_id'],r['seed'],r['method'])
        require(identity in expected,'unexpected anchor/seed/method identity')
        require((1024,*identity) not in indexed,'duplicate case identity')
        k,rho=FROZEN_CONFIGS[r['method']]
        validate_resume_result(r,{'budget':1024,'iterations':k,'elite_ratio':rho,
            'split':'dev_validation','training':False,'closed_loop':False},execution)
        require(finite(r),'NaN/Inf in raw solve result')
        indexed[(1024,*identity)]=r
    require(len(indexed)==960 and {key[1:] for key in indexed}==expected,'missing case: exact 960 required')
    return indexed

def paired_oracle(left,right):
    if left is None and right is None: return 'both_unscoreable',0
    if right is None: return 'only_left_scoreable',1
    if left is None: return 'only_right_scoreable',-1
    require(len(left)==len(right)==5,'five-part objective required')
    # Independent strict lexicographic comparison; no fingerprint enters.
    for a,b in zip(left,right):
        if a<b: return 'both_scoreable_left_win',1
        if a>b: return 'both_scoreable_right_win',-1
    return 'both_scoreable_tie',0

def bootstrap_oracle(rows):
    means=[sum(rows[a])/5 for a in sorted(rows)]
    require(len(means)==64 and all(len(v)==5 for v in rows.values()),'64 five-seed clusters required')
    rng=random.Random(6316)
    samples=sorted(sum(rng.choice(means) for _ in means)/64 for _ in range(10000))
    return {'mean':sum(means)/64,'lower':samples[250],'upper':samples[9750]}

def runtime_clock_facts(runtime):
    elapsed=runtime['invocation_elapsed_seconds']
    utc_elapsed=(datetime.fromisoformat(runtime['invocation_finish_utc'])-
                 datetime.fromisoformat(runtime['invocation_start_utc'])).total_seconds()
    require(math.isfinite(elapsed) and elapsed>0 and utc_elapsed>0,'invalid runtime clocks')
    return {'actual_elapsed_seconds':elapsed,'utc_timestamp_interval_seconds':utc_elapsed,
        'utc_minus_monotonic_seconds':utc_elapsed-elapsed,
        'elapsed_basis':'Frozen runner time.perf_counter duration in seconds',
        'clock_difference_origin':'Unverified; both original time bases preserved, no equality asserted.'}

def main():
    config=read(CONFIG); execution={k:config[k] for k in EXECUTION_KEYS}
    require(source_hashes()==config['source_sha256'],'frozen scientific source drift')
    require(sha(CHECKPOINT)==EXPECTED_SHA==config['checkpoint_sha256'],'checkpoint SHA mismatch')
    bridge=validate_validation_provenance(config,execution)
    train_inv=read(OUT/'train_local_backup_inventory.json')
    require(len(train_inv['files'])==771,'historical TRAIN inventory incomplete')
    require(all(sha(OUT/f['path'])==f['sha256'] for f in train_inv['files']),'historical TRAIN evidence altered')
    anchors=selected_ids('16_validation_anchor_manifest_objective_eligible.json','dev_validation',64)
    eligibility=read(OUT/'20_objective_anchor_eligibility_receipt.json')
    eligible={r['sample_id'] for r in eligibility['rows'] if r['split']=='dev_validation'}
    require(set(anchors)==eligible and eligibility['cohort_zero_count']==eligibility['static_empty_count']==0,
            'empty cohort/static domain or changed eligible anchors')
    paths=sorted((OUT/'solve_results/validation').glob('*.json'))
    rows=[read(p) for p in paths]
    indexed=validate_stage_a_cases(rows,anchors,execution)
    for p,r in zip(paths,rows):
        k,rho=FROZEN_CONFIGS[r['method']]
        require(p==result_path('validation',r['sample_id'],r['method'],r['seed'],1024,k,rho),'wrong result namespace')
        o=r['outcome']; b=o['budget_receipt']
        require(b['B_WM']==1024 and 0<=b['N_unique_transition_evals']<=1024,'wrong transition budget')
        require((o['best_objective'] is not None)==(o['h4_scoreable_count']>0),'scoreability/objective inconsistency')
        require(r['sidecar_alignment_passed'] is True,'sidecar alignment failed')
        require(r['support_horizon_counts'].get('SCORER_EXCEPTION',0)==0,'scorer exception found')
        require(not any(n for reason,n in r['score_residuals'].items()
            if reason.startswith(('ScorerStateInconsistency','BurdenSemanticsBlocked'))),'scorer inconsistency found')
    require(len({r['parameter_digest'] for r in rows})==1,'model parameter digest differs across solves')
    primary_path=OUT/'07_validation_stage_a_primary_comparison_receipt.json'
    selected_path=OUT/'08_selected_method.json'
    primary,selected=read(primary_path),read(selected_path)
    validate_resume_result(primary,{'stage':'primary','budgets':[1024],'case_count':960,
        'primary_budget':1024,'nominal_B_WM_total':983040,'hard_stop_after_stage':True,
        'next_stage_auto_start':False,'validation_retuning':False},execution)
    validate_resume_result(selected,{'primary_budget':1024,'validation_retuning':False},execution)
    require(selected['primary_comparison_sha256']==sha(primary_path),'selection parent SHA mismatch')
    summary=summarize_validation_budget(indexed,anchors,VALIDATION_SEEDS,1024)
    require(json.loads(json.dumps(summary))==primary['comparison']['1024'],'raw-rebuilt summary disagrees with official receipt')
    pair_checks={}
    for left,right in PAIRS:
        key=f'{left}_vs_{right}'; official=summary['paired_outcomes'][key]
        bins=collections.Counter(); per_anchor={}
        for a in anchors:
            per_anchor[a]=[]
            for seed in VALIDATION_SEEDS:
                category,value=paired_oracle(indexed[(1024,a,seed,left)]['outcome']['best_objective'],
                                            indexed[(1024,a,seed,right)]['outcome']['best_objective'])
                bins[category]+=1; per_anchor[a].append(value)
        require(all(bins[k]==v for k,v in official['six_category_counts'].items()) and sum(bins.values())==320,'six-category oracle mismatch')
        require(per_anchor==official['per_anchor_seed_outcomes'],'paired seed cluster mismatch')
        oracle=bootstrap_oracle(per_anchor); ci=official['cluster_bootstrap']
        require(oracle=={'mean':ci['mean_paired_advantage'],'lower':ci['ci95_lower'],'upper':ci['ci95_upper']},'independent bootstrap mismatch')
        require(selected['pairwise_ci'][key]==ci,'selected receipt CI mismatch')
        pair_checks[key]={'six_category_counts':dict(official['six_category_counts']),
            'win':official['win'],'tie':official['tie'],'loss':official['loss'],
            'cluster_bootstrap':ci,'oracle':'PASS'}
    # Independent application of the frozen simplicity/superiority decision tree.
    passed={key:value['cluster_bootstrap']['ci95_lower']>0 for key,value in pair_checks.items()}
    s=passed['S-CEM_vs_HRS']; mh=passed['MH-CEM_vs_HRS']
    winner=('HRS' if not s and not mh else 'S-CEM' if s and not mh else
            'MH-CEM' if mh and not s else 'MH-CEM' if passed['MH-CEM_vs_S-CEM'] else 'S-CEM')
    require(winner==primary['selected_method']==selected['selected_method'],'frozen selection rule mismatch')
    sensitivity=sensitivity_diagnostic(summary)
    require(sensitivity==primary['sensitivity_diagnostic'],'diagnostic-only sensitivity mismatch')
    status=read(CONTROL/'02_runtime_status.json')
    require(status['state']=='COMPLETED_PENDING_ACCEPTANCE' and status['completed_cases']==960 and status['blocker'] is None,
            'runner has not completed and stopped cleanly')
    log_path=CONTROL/'formal_validation_stage_a_3080ti.log'; log=log_path.read_text()
    require(sum(line.startswith('validation: ') for line in log.splitlines())==960,'formal log solve count differs')
    require('Stage A complete: STOP. Researcher review required; Stage B not started.' in log,'hard-stop log missing')
    require(not (OUT/'09_validation_stage_b_budget_diagnostic_receipt.json').exists(),'Stage B receipt exists')
    diag=summary['method_diagnostics']; actual=sum(d['N_unique_transition_evals'] for d in diag.values())
    require(actual==status['actual_unique_transitions_completed'],'monitor and raw transition totals differ')
    runtime=primary['runtime_diagnostic']; elapsed=runtime['invocation_elapsed_seconds']
    clocks=runtime_clock_facts(runtime)
    require(elapsed>=sum(d['wall_clock_diagnostic']['in_solve_seconds_sum'] for d in diag.values()),'matrix elapsed below summed in-solve clocks')
    runtime_receipt={'start_utc':runtime['invocation_start_utc'],'finish_utc':runtime['invocation_finish_utc'],
        **clocks,'completed_cases':960,'cases_per_hour':960*3600/elapsed,
        'actual_transition_throughput_per_second':actual/elapsed,'in_solve_seconds_sum':sum(d['wall_clock_diagnostic']['in_solve_seconds_sum'] for d in diag.values()),
        'basis':'Single uninterrupted official Stage-A invocation wall-clock; includes primary matrix and summary bootstrap, excludes launch preflight and final post-run closure.',
        'supervisor_elapsed_seconds':status['elapsed_seconds'],
        'supervisor_elapsed_note':'Includes process startup and up to 30 seconds of monitoring detection lag; attached monitor exit_code is inferred from receipt existence, not captured OS exit status.'}
    backup_path=sorted((CONTROL/'snapshots').glob('*_backup_inventory.json'))[-1]
    backup=read(backup_path)
    require(backup['file_count']==960,'full local persistent backup inventory not complete')
    require({f['path'] for f in backup['files']}=={p.relative_to(ROOT).as_posix() for p in paths},'backup inventory identity set differs')
    require(all(sha(ROOT/f['path'])==f['sha256'] for f in backup['files']),'persistent raw backup SHA mismatch')
    per_anchor={}
    for a in anchors:
        per_anchor[a]={m:{'scoreable_seeds':[s for s in VALIDATION_SEEDS if indexed[(1024,a,s,m)]['outcome']['best_objective'] is not None],
            'unscoreable_seeds':[s for s in VALIDATION_SEEDS if indexed[(1024,a,s,m)]['outcome']['best_objective'] is None]} for m in FROZEN_CONFIGS}
    write(CONTROL/'10_independent_stage_a_summary.json',{'method_diagnostics':diag,'paired_oracles':pair_checks,
        'selected_method':winner,'sensitivity_diagnostic':sensitivity,'per_anchor_scoreability':per_anchor,
        'claim_boundary':'Planner v1 structured-search backbone / pure-search baseline; not final hybrid PI-JWM planner freeze.'})
    write(CONTROL/'11_formal_stage_a_runtime_receipt.json',runtime_receipt)
    evidence=paths+[primary_path,selected_path,log_path,CONFIG,CONTROL/'01_launch_preflight_receipt.json',
        CONTROL/'02_runtime_status.json',CONTROL/'monitor_attachment_receipt.json',
        CONTROL/'10_independent_stage_a_summary.json',CONTROL/'11_formal_stage_a_runtime_receipt.json']
    inventory={'file_count':len(evidence),'raw_case_count':960,'files':[{'path':p.relative_to(ROOT).as_posix(),
        'sha256':sha(p),'bytes':p.stat().st_size} for p in evidence],'local_persistent_storage':str(ROOT),
        'remote_repo':'/root/autodl-tmp/pi-jwm-step6-3d','remote_originals_preserved':True,
        'sftp_backup_inventory_sha256':sha(backup_path)}
    write(CONTROL/'12_stage_a_local_file_inventory.json',inventory)
    acceptance={'verdict':'STEP_6_3D_VALIDATION_STAGE_A=PASS','case_count':960,'nominal_B_WM_total':983040,
        'actual_unique_transitions':actual,'selected_search_method':winner,'method_diagnostics':diag,
        'seed':list(VALIDATION_SEEDS),'primary_budget':1024,
        'paired_comparisons':pair_checks,'runtime':runtime_receipt,'execution_config_id':config['execution_config_id'],
        'gpu_model':config['gpu_model'],'precision':'FP32','batch_size':16,'checkpoint_sha256':EXPECTED_SHA,
        'provenance_bridge':bridge,'source_scientific_files_unchanged':True,'train_771_file_SHA_integrity':'PASS',
        'frozen_configs':FROZEN_CONFIGS,'stage_b':'NOT_STARTED','stage_b_result_count':0,'locked_test':False,
        'return_birth_support_limitation':'SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION',
        'empty_cohort_count':0,'static_empty_count':0,'scorer_exception_count':0,'scorer_inconsistency_count':0,
        'inventory_sha256':sha(CONTROL/'12_stage_a_local_file_inventory.json'),'remaining_blockers':[],
        'warnings':['Future Return-birth fixed-support limitation retained.','SEARCH_METHOD is a pure-search backbone, not final hybrid PI-JWM planner.','One read-only backup connection reset recovered without restarting search.','UTC timestamps and monotonic elapsed differ; both reported, clock-adjustment origin Unverified.'],
        'acceptance_scope':'Execution, independent statistics, runtime and archive integrity; researcher owns final scientific interpretation.'}
    acceptance_path=CONTROL/'13_stage_a_acceptance_receipt.json'
    pending=CONTROL/'13_stage_a_acceptance_receipt.json.pending'
    write(pending,acceptance)
    archive_path=CONTROL/'stage_a_local_persistent_archive.zip'
    members=evidence+[CONTROL/'12_stage_a_local_file_inventory.json',acceptance_path]
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in members: archive.write(pending if p==acceptance_path else p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None and len(archive.namelist())==len(members),'archive member/integrity mismatch')
        require(all(hashlib.sha256(archive.read(p.relative_to(ROOT).as_posix())).hexdigest()==sha(pending if p==acceptance_path else p) for p in members),'archive per-file SHA mismatch')
    os.replace(pending,acceptance_path)
    write(CONTROL/'14_stage_a_archive_sha_manifest.json',{'archive_path':archive_path.relative_to(ROOT).as_posix(),
        'archive_sha256':sha(archive_path),'bytes':archive_path.stat().st_size,'member_count':len(members),
        'archive_integrity':'PASS','per_member_sha_verdict':'PASS','raw_results_in_git':False,
        'local_persistent_storage':str(archive_path),'remote_originals_preserved':True,
        'inventory_sha256':sha(CONTROL/'12_stage_a_local_file_inventory.json'),
        'acceptance_sha256':sha(CONTROL/'13_stage_a_acceptance_receipt.json')})
    print(json.dumps({'verdict':acceptance['verdict'],'case_count':960,'selected_method':winner,
        'actual_unique':actual,'elapsed_hours':elapsed/3600,'cases_per_hour':runtime_receipt['cases_per_hour'],
        'method_diagnostics':diag,'paired_comparisons':pair_checks,'archive_sha256':sha(archive_path)}))

if __name__=='__main__': main()
