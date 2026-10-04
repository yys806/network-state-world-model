"""One explicit repaired-domain phase per invocation; strict GPU/Git/resume."""
import argparse,hashlib,json,math,os,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
from prepare_step6_4g_requalification_v1 import OUT,read,sha,digest,source_hashes,PROTOCOL_NAME
from pi_jwm.step6_4g_requalification_v1 import phase_cases,require_phase_parent,GRID

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    os.replace(tmp,path)
def phase_parents(phase):
    parents={};tuning=selection=None
    if phase!='T':
        tuning=read(OUT/'T/acceptance.json')
        for n in ('T/acceptance.json','T/selected_configs.json','T/archive.json','T/local_backup_acceptance.json'):
            parents[n]=sha(OUT/n)
    if phase=='B':
        selection=read(OUT/'A/selected_method.json')
        for n in ('A/acceptance.json','A/selected_method.json','A/archive.json','A/local_backup_acceptance.json'):
            parents[n]=sha(OUT/n)
        if read(OUT/'A/acceptance.json')['verdict']!='PASS':raise ValueError('Phase A not accepted')
    require_phase_parent(phase,tuning,selection)
    if tuning is not None:
        if tuning['evidence_class']!='repaired-domain formal evidence':raise ValueError('historical tuning cannot be reused')
        for n in ('selected_configs.json','inventory.json','archive.json'):
            if tuning['artifact_SHA256'][n]!=sha(OUT/'T'/n):raise ValueError('T acceptance parent drift')
        local=read(OUT/'T/local_backup_acceptance.json')
        if local['verdict']!='PASS' or local['acceptance_SHA256']!=sha(OUT/'T/acceptance.json'):raise ValueError('T local backup not verified')
    if selection is not None:
        a=read(OUT/'A/acceptance.json')
        if a['evidence_class']!='repaired-domain formal evidence' or a['artifact_SHA256']['selected_method.json']!=sha(OUT/'A/selected_method.json'):
            raise ValueError('A selection provenance mismatch')
        local=read(OUT/'A/local_backup_acceptance.json')
        if local['verdict']!='PASS' or local['acceptance_SHA256']!=sha(OUT/'A/acceptance.json'):raise ValueError('A local backup not verified')
    configs=None if phase=='T' else read(OUT/'T/selected_configs.json')['configs']
    return parents,configs

def phase_config(phase):
    protocol=read(OUT/PROTOCOL_NAME);parents,configs=phase_parents(phase)
    fields={k:protocol[k] for k in ('source_sha256','input_sha256','checkpoint_path','checkpoint_sha256',
      'gpu_model','execution_device','precision','batch_size','state_storage','bucket_strategy','eligibility_contract_SHA256',
      'support_catalog_SHA256','anchor_manifest_SHA256','amp','bf16','fp16','quantization','torch_compile','prior_mode','service_mode','locked_test')}
    fields.update(phase=phase,protocol_SHA256=sha(OUT/PROTOCOL_NAME),parents=parents,
      selected_configs=configs,tuning_grid=protocol['grid'] if phase=='T' else None,
      evidence_class='repaired-domain formal evidence')
    return {**fields,'execution_config_id':digest(fields)}

def validate_binding(config,strict_bytes=False):
    if phase_config(config['phase'])!=config:raise ValueError('phase config/parent receipt identity drift')
    if source_hashes()!=config['source_sha256']:raise ValueError('scientific source drift')
    if strict_bytes:
        for n,h in config['source_sha256'].items():
            if sha(ROOT/n)!=h:raise ValueError('executed bytes must equal frozen Git LF source: '+n)
    for n,h in config['input_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('input drift: '+n)
    if sha(ROOT/config['checkpoint_path'])!=config['checkpoint_sha256']:raise ValueError('checkpoint identity drift')

def cases(config):
    anchors=read(OUT/'02_frozen_anchor_manifest.json')['anchors']
    return phase_cases(config['phase'],anchors['train'],anchors['validation'],config['selected_configs'])
def raw_path(phase,case):
    ident=digest(case)
    return OUT/phase/'solve_results'/f'{ident}.json'
def result_identity(config,case,gate_commit):
    return {**case,**{k:config[k] for k in ('execution_config_id','source_sha256','checkpoint_sha256',
      'execution_device','gpu_model','precision','batch_size','state_storage','bucket_strategy',
      'eligibility_contract_SHA256','support_catalog_SHA256','anchor_manifest_SHA256','phase','protocol_SHA256','evidence_class')},
      'parents':config['parents'],'phase_gate_commit':gate_commit,'locked_test':False,
      'training':False,'closed_loop':False,'gpu':True,'sidecar_alignment_passed':True}

def validate_result(row,expected):
    for key,value in expected.items():
        if row.get(key)!=value:raise ValueError('resume/result identity mismatch: '+key)
    # Check all floats in the full receipt, including runtime and diagnostics.
    def finite(value):
        if isinstance(value,float) and not math.isfinite(value):raise ValueError('NaN/Inf result')
        if isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,(list,tuple)):
            for v in value:finite(v)
    finite(row)
    o=row['outcome'];b=o['budget_receipt']
    for k in ('method','seed','budget','iterations','elite_ratio','batch_size'):
        if o[k]!=row[k]:raise ValueError('nested outcome identity drift: '+k)
    if b['B_WM']!=row['budget'] or b['N_unique_transition_evals']!=row['budget']:
        raise ValueError('solver requires quota fill, actual budget mismatch')
    count_keys=('N_unique_transition_evals','N_cache_hits','N_complete_sequences','N_dead_end_branches','N_proposed_steps','N_admitted_steps','N_rejected_steps')
    if any(type(b[k])!=int or b[k]<0 for k in count_keys):raise ValueError('invalid budget counts')
    if b['N_proposed_steps']!=b['N_admitted_steps']+b['N_rejected_steps']:raise ValueError('proposal accounting')
    if b['N_complete_sequences']!=o['complete_sequence_count']:raise ValueError('H4 complete accounting')
    if o['h4_scoreable_count']+o['h4_unscoreable_count']>o['complete_sequence_count']:raise ValueError('H4 count inconsistency')
    objective=o['best_objective']
    if objective is not None and (len(objective)!=5 or not all(type(v) in (int,float) and math.isfinite(v) for v in objective)):
        raise ValueError('invalid objective')
    if (objective is not None)!=(o['h4_scoreable_count']>0) or (objective is not None)!=(o['best_fingerprint'] is not None):raise ValueError('winner scoreability inconsistency')
    if row['support_horizon_counts'].get('SCORER_EXCEPTION',0) or any(k.startswith(('ScorerStateInconsistency:','BurdenSemanticsBlocked:')) for k in row['score_residuals']):
        raise ValueError('scorer exception/inconsistency: stop')
    cumulative=0
    from pi_jwm.step6_3d_fixed_budget_search_v1 import quotas
    if len(o['iteration_rows'])!=row['iterations']:raise ValueError('iteration count')
    for i,(r,q) in enumerate(zip(o['iteration_rows'],quotas(row['budget'],row['iterations']))):
        cumulative+=q
        if r['iteration']!=i+1 or r['quota']!=q or r['unique_transition_evals']!=cumulative:raise ValueError('iteration quota accounting')

def git_gate(config,commit):
    if not commit or len(commit)!=40:raise ValueError('exact phase-gate Git commit mandatory')
    for ref in ('HEAD','origin/main'):
        if subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip()!=commit:raise ValueError('HEAD=origin/main exact phase gate required')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise ValueError('tracked tree dirty')
    for n,h in config['source_sha256'].items():
        if hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT)).hexdigest()!=h:raise ValueError('Git scientific blob mismatch '+n)
    for n in (PROTOCOL_NAME,'02_frozen_anchor_manifest.json'):
        if subprocess.check_output(['git','show',commit+':'+(OUT/n).relative_to(ROOT).as_posix()],cwd=ROOT)!=(OUT/n).read_bytes():raise ValueError('protocol not frozen in phase commit')
    for n,h in config['parents'].items():
        if hashlib.sha256(subprocess.check_output(['git','show',commit+':'+(OUT/n).relative_to(ROOT).as_posix()],cwd=ROOT)).hexdigest()!=h:raise ValueError('phase parent not committed '+n)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('T','A','B'),required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--phase-gate-commit')
    p.add_argument('--accept-only',action='store_true');args=p.parse_args()
    config=phase_config(args.phase);validate_binding(config)
    planned=cases(config);actual=set((OUT/args.phase/'solve_results').glob('*.json'))
    allowed={raw_path(args.phase,c) for c in planned}
    if not actual.issubset(allowed):raise ValueError('unexpected raw identity namespace')
    if args.accept_only:
        from audit_step6_4g_phase_v1 import accept_phase
        accept_phase(args.phase,args.phase_gate_commit);return
    if not args.execute:
        print(json.dumps({'CPU_PREFLIGHT':'PASS','phase':args.phase,'cases':len(planned),
          'execution_config_id':config['execution_config_id'],'results':len(actual),'GPU_started':False}));return
    if actual and not args.resume:raise ValueError('initial result count must zero; explicit resume required')
    git_gate(config,args.phase_gate_commit);validate_binding(config,strict_bytes=True)
    if shutil.disk_usage(ROOT).free<read(OUT/PROTOCOL_NAME)['storage']['minimum_free_bytes']:raise ValueError('insufficient disk')
    # Historical runtime import has a CPU preflight default; preserve the
    # explicit qualified device visibility before importing that runtime.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0)!=config['gpu_model']:raise ValueError('qualified RTX3080Ti unavailable')
    if torch.get_default_dtype()!=torch.float32 or torch.is_autocast_enabled():raise ValueError('FP32/no autocast required')
    import fcntl
    lockpath=OUT/args.phase/'runner.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        confpath=OUT/args.phase/'execution_config.json'
        if confpath.exists() and read(confpath)!=config:raise ValueError('stored phase execution config mismatch')
        if not confpath.exists():atomic(confpath,config)
        # Full resume validation BEFORE any model forward.
        for c in planned:
            path=raw_path(args.phase,c)
            if path.exists():validate_result(read(path),result_identity(config,c,args.phase_gate_commit))
        resource={'GPU':torch.cuda.get_device_name(0),'CUDA':torch.version.cuda,'PyTorch':torch.__version__,
          'free_VRAM_bytes':torch.cuda.mem_get_info()[0],'disk_free_bytes':shutil.disk_usage(ROOT).free,
          'phase_gate_commit':args.phase_gate_commit,'tracked_clean':True,'initial_result_count':len(actual)}
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        start=datetime.now(timezone.utc).isoformat();began=time.perf_counter();new=0
        atomic(OUT/args.phase/f'launch_{stamp}.json',{'start_time_utc':start,**resource,'resume':args.resume,'locked_test':False})
        from run_step6_3d_one_cpu_solve_v1 import run,source_hashes as solver_hashes,execution_identity
        for i,c in enumerate(planned):
            validate_binding(config,strict_bytes=True);path=raw_path(args.phase,c)
            if not path.exists():
                before=time.perf_counter()
                try:
                    row=run(c['sample_id'],c['method'],c['seed'],c['budget'],c['iterations'],c['elite_ratio'],batch_size=16,device='cuda')
                    solver=execution_identity(device='cuda',gpu_model=config['gpu_model'],batch_size=16,
                      checkpoint_sha256=config['checkpoint_sha256'],source_sha256=solver_hashes())
                    for k,v in {**c,**solver}.items():
                        if row.get(k)!=v:raise ValueError('solver execution bridge mismatch '+k)
                    solver_id=row['execution_config_id'];solver_sources=row['source_sha256']
                    row.update(result_identity(config,c,args.phase_gate_commit))
                    row.update(solver_execution_config_id=solver_id,solver_source_sha256=solver_sources,
                      case_total_seconds=time.perf_counter()-before)
                    validate_result(row,result_identity(config,c,args.phase_gate_commit));validate_binding(config,strict_bytes=True)
                    atomic(path,row);new+=1
                except Exception as exc:
                    atomic(OUT/args.phase/f'STOP_{stamp}.json',{'case':c,'error_type':type(exc).__name__,'reason':str(exc),
                      'no_retry':True,'GPU_search_stopped':True,'locked_test':False});raise
            elapsed=time.perf_counter()-began
            atomic(OUT/args.phase/'runtime_status.json',{'start_time_utc':start,'update_time_utc':datetime.now(timezone.utc).isoformat(),
              'completed':i+1,'target':len(planned),'new_cases_this_invocation':new,'elapsed_s':elapsed,
              'cases_per_hour_new':new*3600/elapsed,'GPU_diagnostic':subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip(),
              'phase':args.phase,'invocation':stamp,'locked_test':False})
            print(json.dumps({'phase':args.phase,'complete':i+1,'target':len(planned),'elapsed_s':elapsed}),flush=True)
        atomic(OUT/args.phase/f'finish_{stamp}.json',{'start_time_utc':start,'finish_time_utc':datetime.now(timezone.utc).isoformat(),
          'elapsed_s':time.perf_counter()-began,'new_cases':new,'complete':len(planned),'locked_test':False})
        from audit_step6_4g_phase_v1 import accept_phase
        accept_phase(args.phase,args.phase_gate_commit)
        print('PHASE COMPLETE; HARD STOP for independent local archive, commit/push and next explicit phase gate',flush=True)

if __name__=='__main__':main()
