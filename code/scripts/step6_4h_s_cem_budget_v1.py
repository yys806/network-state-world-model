"""Independent S-CEM budget protocol, CPU freeze, strict runner and raw audit.

No 6.4G phase gate is changed or invoked. Scientific search is reused unchanged.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/src'), str(ROOT/'code/scripts')]
import run_step6_4g_requalification_v1 as parent_runner
from prepare_step6_4g_requalification_v1 import read, sha, digest, source_hashes
from run_step6_4g_requalification_v1 import atomic, validate_result
from audit_step6_4g_phase_v1 import group_summary, oracle_win, oracle_bootstrap
from pi_jwm.step6_4d_full_cohort_b512_v1 import full_pair_analysis, COMPONENTS

OUT = ROOT/'code/artifacts/protocols/pi_jwm_step6_4h_s_cem_budget_v1_20261008_r2'
PARENT = parent_runner.OUT
CHECKPOINT_SHA = '941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9'
PARENT_ID = 'a7c793ef7619e107fe7f3a5e3fdf16cffb8ae64af76b35422ee613fb69914db8'


def planned_cases(anchors):
    if len(anchors)!=64 or len(set(anchors))!=64:
        raise ValueError('exact frozen 64 anchors required')
    return [dict(sample_id=a,seed=s,method='S-CEM',iterations=4,
                 elite_ratio=.2,budget=512,split='dev_validation')
            for a in anchors for s in range(6311,6316)]


def qualification(s):
    checks = dict(paired_complete=s['paired_cases']==320,
                  provenance=s['provenance'] is True, budget=s['budget'] is True,
                  scorer_clean=s['scorer_errors']==0,
                  same_scoreability=s['only_left']==s['only_right']==0,
                  no_primary_loss=s['primary_loss']==0,
                  mean_lower=s['mean_512']<s['mean_1024'],
                  median_lower=s['median_512']<s['median_1024'],
                  valid_runtime=all(math.isfinite(s[k]) and s[k]>0 for k in
                    ('mean_512','mean_1024','median_512','median_1024')))
    passed=all(checks.values())
    return dict(verdict='PASS' if passed else 'FAIL',checks=checks,
                recommended_budget=512 if passed else None,
                final_budget='RESEARCHER_DECISION_PENDING',
                claim_boundary='Not equivalence, optimality or real closed-loop performance')


def parent_rows():
    config=read(PARENT/'A/execution_config.json')
    accepted=read(PARENT/'A/acceptance.json')
    local=read(PARENT/'A/local_backup_acceptance.json')
    if (config['execution_config_id']!=PARENT_ID or accepted['verdict']!='PASS'
        or local['verdict']!='PASS' or local['acceptance_SHA256']!=sha(PARENT/'A/acceptance.json')
        or read(PARENT/'A/selected_method.json')['selected_method']!='S-CEM'):
        raise ValueError('accepted repaired S-CEM parent unavailable')
    for n,h in accepted['artifact_SHA256'].items():
        if sha(PARENT/'A'/n)!=h: raise ValueError('parent artifact drift '+n)
    if sha(PARENT/'A/raw_results.zip')!=local['archive_SHA256']:
        raise ValueError('parent archive SHA drift')
    # Verify frozen scientific closure against current LF source, not new docs.
    if source_hashes()!=config['source_sha256']:
        raise ValueError('parent scientific source semantics cannot be reused')
    for n,h in config['input_sha256'].items():
        if sha(ROOT/n)!=h: raise ValueError('parent data/contract drift '+n)
    if config['checkpoint_sha256']!=CHECKPOINT_SHA or sha(ROOT/config['checkpoint_path'])!=CHECKPOINT_SHA:
        raise ValueError('checkpoint drift')
    inventory={r['path']:r for r in read(PARENT/'A/inventory.json')['files']}
    anchors=read(PARENT/'02_frozen_anchor_manifest.json')['anchors']['validation']
    rows=[];refs=[]
    with zipfile.ZipFile(PARENT/'A/raw_results.zip') as archive:
        for case in planned_cases(anchors):
            c={**case,'budget':1024};p=parent_runner.raw_path('A',c)
            key=p.relative_to(ROOT).as_posix()
            if key not in inventory or sha(p)!=inventory[key]['sha256']:
                raise ValueError('parent raw SHA/identity missing '+key)
            if hashlib.sha256(archive.read(p.name)).hexdigest()!=sha(p):
                raise ValueError('parent archived raw mismatch')
            row=read(p)
            validate_result(row,parent_runner.result_identity(config,c,accepted['phase_gate_commit']))
            if row['parameter_digest']!=accepted['parameter_digest']: raise ValueError('parent model drift')
            rows.append(row);refs.append(dict(path=key,sha256=sha(p),identity=c))
    return anchors,rows,refs


def freeze():
    if (OUT/'00_protocol.json').exists(): raise ValueError('protocol already frozen; no overwrite')
    anchors,rows,refs=parent_rows()
    if shutil.disk_usage(ROOT).free<5*1024**3: raise ValueError('D persistent capacity insufficient')
    parent_config=read(PARENT/'A/execution_config.json')
    names=('acceptance.json','inventory.json','archive.json','local_backup_acceptance.json','selected_method.json','execution_config.json')
    parents={(PARENT/'A'/n).relative_to(ROOT).as_posix():sha(PARENT/'A'/n) for n in names}
    sources=source_hashes()
    for n in ('step6_4h_s_cem_budget_v1.py',):
        rel='code/scripts/'+n
        sources[rel]=hashlib.sha256((ROOT/rel).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    atomic(OUT/'01_manifest.json',dict(anchors=anchors,seeds=list(range(6311,6316)),cases=planned_cases(anchors)))
    atomic(OUT/'02_parent_refs.json',dict(verdict='PASS',count=320,files=refs,
        raw_and_archive_SHA_verified=True,source_semantics_unchanged=True,parents=parents))
    config={k:parent_config[k] for k in ('input_sha256','checkpoint_path','checkpoint_sha256','gpu_model',
        'execution_device','precision','batch_size','state_storage','bucket_strategy','eligibility_contract_SHA256',
        'support_catalog_SHA256','amp','bf16','fp16','quantization','torch_compile','prior_mode','service_mode','locked_test')}
    config.update(source_sha256=sources,parents=parents,phase='H',evidence_class='repaired-domain S-CEM budget qualification',
        anchor_manifest_SHA256=sha(OUT/'01_manifest.json'),parent_refs_SHA256=sha(OUT/'02_parent_refs.json'),
        method='S-CEM',iterations=4,elite_ratio=.2,budget=512,
        namespace=OUT.relative_to(ROOT).as_posix())
    config['execution_config_id']=digest(config)
    atomic(OUT/'03_execution_config.json',config)
    atomic(OUT/'00_protocol.json',dict(name='STEP 6.4H S-CEM BUDGET QUALIFICATION',
        execution_config_SHA256=sha(OUT/'03_execution_config.json'),new_cases=320,nominal_transitions=163840,
        parent_cases=320,parent_budget=1024,bootstrap=dict(cluster='anchor',seeds_per_cluster=5,replications=10000,seed=6316),
        qualification_gates=['complete paired provenance/budget','scorer zero','identical scoreability sets',
            'zero primary first-difference loss','lower mean and median in-solve runtime'],
        components=list(COMPONENTS),atomic_resume='full identity before model forward; atomic os.replace',
        stop_on=['duplicate runner','identity mismatch','source/input/checkpoint drift','NaN/Inf','scorer error','budget mismatch'],
        six_categories=['only_left_scoreable','only_right_scoreable','both_scoreable_left_better',
            'both_scoreable_right_better','both_scoreable_tie','both_unscoreable'],
        runtime='in-solve and case setup and invocation elapsed reported separately',
        minimum_free_bytes=5*1024**3,GPU_capacity='Recheck on server at launch; not measured in CPU preparation',
        locked_test=False,formal_closed_loop=False,final_budget='RESEARCHER_DECISION_PENDING',
        historical_6_4G_B='NOT_STARTED_BY_CONDITIONAL_STOP; unchanged'))
    atomic(OUT/'04_cpu_preflight.json',dict(verdict='PASS',parent_count=320,
        new_result_count=0,READY_FOR_GPU_LAUNCH=True,GPU_used=False,
        local_free_bytes=shutil.disk_usage(ROOT).free,server_checks_pending=True,locked_test=False))


def binding(strict_bytes=False):
    config=read(OUT/'03_execution_config.json')
    fields={k:v for k,v in config.items() if k!='execution_config_id'}
    if digest(fields)!=config['execution_config_id']: raise ValueError('execution config identity drift')
    if sha(OUT/'03_execution_config.json')!=read(OUT/'00_protocol.json')['execution_config_SHA256']:
        raise ValueError('protocol/config drift')
    for n,h in config['source_sha256'].items():
        if hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h: raise ValueError('source drift '+n)
        if strict_bytes and sha(ROOT/n)!=h: raise ValueError('executed bytes drift '+n)
    for n,h in {**config['input_sha256'],**config['parents']}.items():
        if sha(ROOT/n)!=h: raise ValueError('input/parent drift '+n)
    if sha(ROOT/config['checkpoint_path'])!=CHECKPOINT_SHA: raise ValueError('checkpoint drift')
    if sha(OUT/'01_manifest.json')!=config['anchor_manifest_SHA256'] or sha(OUT/'02_parent_refs.json')!=config['parent_refs_SHA256']:
        raise ValueError('manifest drift')
    for r in read(OUT/'02_parent_refs.json')['files']:
        if sha(ROOT/r['path'])!=r['sha256']: raise ValueError('parent raw drift')
    return config


def identity(config,case,commit):
    return {**case,**{k:config[k] for k in ('execution_config_id','source_sha256','checkpoint_sha256',
        'execution_device','gpu_model','precision','batch_size','state_storage','bucket_strategy',
        'eligibility_contract_SHA256','support_catalog_SHA256','anchor_manifest_SHA256','phase','evidence_class')},
        'parents':config['parents'],'phase_gate_commit':commit,'locked_test':False,
        'training':False,'closed_loop':False,'gpu':True,'sidecar_alignment_passed':True}


def path_for(case): return OUT/'solve_results'/(digest(case)+'.json')


def statistics(rows,parent):
    values=lambda rs:{(r['sample_id'],r['seed']):r['outcome']['best_objective'] for r in rs}
    left,right=values(rows),values(parent);paired,components=full_pair_analysis(left,right)
    # Independent raw lexicographic and anchor bootstrap reconstruction.
    clusters={a:[oracle_win(left[(a,s)],right[(a,s)]) for s in range(6311,6316)] for a in read(OUT/'01_manifest.json')['anchors']}
    boot=oracle_bootstrap(clusters)
    if any(paired['cluster_bootstrap'][k]!=v for k,v in boot.items()): raise ValueError('bootstrap oracle mismatch')
    primary=sum(1 for k,x in left.items() if x is not None and right[k] is not None and
        next((i for i in range(5) if x[i]!=right[k][i]),5)<3 and oracle_win(x,right[k])<0)
    if primary!=components['PRIMARY_OBJECTIVE_DEGRADATION_COUNT']: raise ValueError('component oracle mismatch')
    from collections import Counter
    bins_oracle=Counter();component_oracle={n:{'B512_better':0,'B1024_better':0} for n in COMPONENTS};equal=0
    for key,x in left.items():
        y=right[key];w=oracle_win(x,y)
        cat=('both_unscoreable' if x is None and y is None else 'only_right_scoreable' if x is None else
             'only_left_scoreable' if y is None else 'both_scoreable_left_better' if w>0 else
             'both_scoreable_right_better' if w<0 else 'both_scoreable_tie')
        bins_oracle[cat]+=1
        if x is not None and y is not None:
            first=next((i for i in range(5) if x[i]!=y[i]),None)
            if first is None: equal+=1
            else: component_oracle[COMPONENTS[first]]['B512_better' if w>0 else 'B1024_better']+=1
    if any(v!=bins_oracle[k] for k,v in paired['six_categories'].items()) or components['components']!=component_oracle or equal!=components['all_equal']:
        raise ValueError('independent six-category/component accounting mismatch')
    l,r=group_summary(rows),group_summary(parent);bins=paired['six_categories']
    gate=qualification(dict(paired_cases=len(rows),provenance=True,
        budget=l['N_unique_transition_evals']==163840 and r['N_unique_transition_evals']==327680,
        scorer_errors=l['scorer_exceptions']+r['scorer_exceptions']+l['scorer_inconsistencies']+r['scorer_inconsistencies'],
        only_left=bins['only_left_scoreable'],only_right=bins['only_right_scoreable'],primary_loss=primary,
        mean_512=l['runtime_seconds']['mean'],mean_1024=r['runtime_seconds']['mean'],
        median_512=l['runtime_seconds']['median'],median_1024=r['runtime_seconds']['median']))
    artifacts={'summary.json':{'512':l,'1024':r},'paired.json':paired,'objective_components.json':components,
        'qualification.json':gate,
        'runtime_tradeoff.json':{k+'_reduction':1-l['runtime_seconds'][k]/r['runtime_seconds'][k] for k in ('mean','median')}}
    return artifacts


def audit(commit):
    config=binding();planned=read(OUT/'01_manifest.json')['cases']
    if (OUT/'acceptance.json').exists(): raise ValueError('accepted evidence immutable')
    if set((OUT/'solve_results').glob('*.json'))!={path_for(c) for c in planned}: raise ValueError('raw coverage mismatch')
    rows=[];inventory=[]
    for c in planned:
        p=path_for(c);r=read(p);validate_result(r,identity(config,c,commit));rows.append(r)
        inventory.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),identity=c))
    _,parent,_=parent_rows()
    if {r['parameter_digest'] for r in rows}!={r['parameter_digest'] for r in parent}: raise ValueError('model drift')
    artifacts=statistics(rows,parent)
    artifacts['inventory.json']={'count':320,'files':inventory}
    finishes=sorted(OUT.glob('finish_*.json'))
    if not finishes or read(finishes[-1])['complete']!=320: raise ValueError('completed matrix runtime receipt required')
    artifacts['matrix_runtime.json']={
        'invocation_elapsed_s_total':sum(read(p)['elapsed_s'] for p in finishes),
        'start_time_utc':read(finishes[0])['start_time_utc'],
        'finish_time_utc':read(finishes[-1])['finish_time_utc'],
        'finish_SHA256':{p.name:sha(p) for p in finishes},
        'in_solve_seconds':artifacts['summary.json']['512']['runtime_seconds']['sum_in_solve'],
        'case_setup_seconds':artifacts['summary.json']['512']['matrix_loading_and_setup_seconds'],
        'note':'Invocation elapsed includes setup, integrity checks and persistence; paused gaps excluded from elapsed sum. B1024 single-method runtime from original raw; original A matrix spans three methods.'}
    for n,v in artifacts.items(): atomic(OUT/n,v)
    archive=OUT/'raw_results.zip'
    if archive.exists(): raise ValueError('unaccepted archive exists; manual inspection required')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for c in planned: z.write(path_for(c),path_for(c).name)
    with zipfile.ZipFile(archive) as z:
        for c in planned:
            p=path_for(c)
            if hashlib.sha256(z.read(p.name)).hexdigest()!=sha(p): raise ValueError('archive verify failed')
    atomic(OUT/'archive.json',dict(sha256=sha(archive),count=320,raw_preserved=True))
    atomic(OUT/'acceptance.json',dict(verdict='PASS',qualification=artifacts['qualification.json']['verdict'],count=320,
        phase_gate_commit=commit,artifact_SHA256={n:sha(OUT/n) for n in [*artifacts,'archive.json']},
        locked_test=False,final_budget='RESEARCHER_DECISION_PENDING'))


def verify_local():
    config=binding();accepted=read(OUT/'acceptance.json')
    for n,h in accepted['artifact_SHA256'].items():
        if sha(OUT/n)!=h: raise ValueError('accepted artifact drift '+n)
    planned=read(OUT/'01_manifest.json')['cases']
    if set((OUT/'solve_results').glob('*.json'))!={path_for(c) for c in planned}: raise ValueError('local coverage')
    inventory={r['path']:r['sha256'] for r in read(OUT/'inventory.json')['files']}
    archive=OUT/'raw_results.zip';rows=[]
    if sha(archive)!=read(OUT/'archive.json')['sha256']: raise ValueError('archive SHA drift')
    with zipfile.ZipFile(archive) as z:
        for c in planned:
            p=path_for(c);h=sha(p)
            if inventory.get(p.relative_to(ROOT).as_posix())!=h or hashlib.sha256(z.read(p.name)).hexdigest()!=h:
                raise ValueError('local raw/archive drift')
            r=read(p);validate_result(r,identity(config,c,accepted['phase_gate_commit']));rows.append(r)
    _,parent,_=parent_rows()
    for n,v in statistics(rows,parent).items():
        if read(OUT/n)!=v: raise ValueError('independent local statistic drift '+n)
    atomic(OUT/'local_backup_acceptance.json',dict(verdict='PASS',count=320,
        local_persistent_storage=str(OUT),acceptance_SHA256=sha(OUT/'acceptance.json'),
        archive_SHA256=sha(archive),independent_statistics='PASS',locked_test=False))


def execute(commit,resume):
    config=binding()
    if len(commit or '')!=40: raise ValueError('exact protocol freeze commit required')
    for ref in ('HEAD','origin/main'):
        if subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip()!=commit: raise ValueError('Git gate mismatch')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip(): raise ValueError('tracked dirty')
    for n,h in config['source_sha256'].items():
        if hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT)).hexdigest()!=h or sha(ROOT/n)!=h:
            raise ValueError('frozen Git/execution source mismatch '+n)
    for n in ('00_protocol.json','01_manifest.json','02_parent_refs.json','03_execution_config.json'):
        rel=(OUT/n).relative_to(ROOT).as_posix()
        if subprocess.check_output(['git','show',commit+':'+rel],cwd=ROOT)!=(OUT/n).read_bytes(): raise ValueError('protocol not committed')
    if shutil.disk_usage(ROOT).free<read(OUT/'00_protocol.json')['minimum_free_bytes']: raise ValueError('capacity gate')
    planned=read(OUT/'01_manifest.json')['cases'];actual=set((OUT/'solve_results').glob('*.json'))
    if not actual.issubset({path_for(c) for c in planned}) or (actual and not resume): raise ValueError('namespace/resume gate')
    if list(OUT.glob('STOP_*.json')): raise ValueError('prior failure; no automatic restart')
    # Launch on Linux only: reject any other formal matrix, including old phases.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid(): continue
        try: argv=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        if any(Path(os.fsdecode(arg)).name.startswith(('run_step6_4','run_step6_3d_formal','step6_4h_s_cem_budget'))
               for arg in argv[1:]) and b'--execute' in argv:
            raise ValueError('duplicate/old formal runner present')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    import torch
    import fcntl
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0)!=config['gpu_model']: raise ValueError('GPU identity')
    if torch.get_default_dtype()!=torch.float32 or torch.is_autocast_enabled(): raise ValueError('precision gate')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for c in planned:
            if path_for(c).exists(): validate_result(read(path_for(c)),identity(config,c,commit))
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');start=datetime.now(timezone.utc).isoformat();began=time.perf_counter();new=0
        atomic(OUT/f'launch_{stamp}.json',dict(start_time_utc=start,phase_gate_commit=commit,initial_count=len(actual),resume=resume))
        from run_step6_3d_one_cpu_solve_v1 import run,source_hashes as solver_hashes,execution_identity
        for i,c in enumerate(planned):
            try:
                binding(strict_bytes=True)
                if not path_for(c).exists():
                    before=time.perf_counter()
                    row=run(c['sample_id'],'S-CEM',c['seed'],512,4,.2,batch_size=16,device='cuda')
                    solver=execution_identity(device='cuda',gpu_model=config['gpu_model'],batch_size=16,checkpoint_sha256=CHECKPOINT_SHA,source_sha256=solver_hashes())
                    if any(row.get(k)!=v for k,v in {**c,**solver}.items()): raise ValueError('solver execution identity mismatch')
                    row.update(solver_execution_config_id=row['execution_config_id'],solver_source_sha256=row['source_sha256'])
                    row.update(identity(config,c,commit));row['case_total_seconds']=time.perf_counter()-before
                    validate_result(row,identity(config,c,commit));binding(strict_bytes=True);atomic(path_for(c),row);new+=1
                atomic(OUT/'runtime_status.json',dict(completed=i+1,target=320,start_time_utc=start,elapsed_s=time.perf_counter()-began,new_cases=new,locked_test=False))
                print(json.dumps({'completed':i+1,'target':320}),flush=True)
            except Exception as exc:
                atomic(OUT/f'STOP_{stamp}.json',dict(case=c,error_type=type(exc).__name__,reason=str(exc),no_retry=True));raise
        atomic(OUT/f'finish_{stamp}.json',dict(start_time_utc=start,finish_time_utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.perf_counter()-began,new_cases=new,complete=320))
        audit(commit)
        print('HARD STOP: no closed-loop or other matrix',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true')
    p.add_argument('--verify-local',action='store_true');p.add_argument('--audit',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--gate-commit');a=p.parse_args()
    if sum((a.freeze,a.execute,a.audit,a.verify_local))>1: p.error('one explicit operation')
    if a.freeze: freeze()
    elif a.execute: execute(a.gate_commit,a.resume)
    elif a.audit: audit(a.gate_commit)
    elif a.verify_local: verify_local()
    else: binding()
    print(json.dumps(dict(CPU_PREFLIGHT='PASS',GPU_started=a.execute,locked_test=False)))

if __name__=='__main__': main()
