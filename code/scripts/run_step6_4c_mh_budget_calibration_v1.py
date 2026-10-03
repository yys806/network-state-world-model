"""Independent 96-case MH-only calibration; never original Stage B or closed loop."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/src'),str(ROOT/'code/scripts')]
from prepare_step6_4c_budget_calibration_v1 import OUT, read, sha, reference_inputs
from pi_jwm.step6_4c_budget_calibration_v1 import validate_identity, summarize, compare_pairs

def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n',
                   encoding='utf-8', newline='\n')
    tmp.replace(path)

def validate_binding(config):
    fields = {k:v for k,v in config.items() if k != 'execution_config_id'}
    expected_id = hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(',',':')).encode()).hexdigest()
    if expected_id != config['execution_config_id']:
        raise ValueError('execution configuration content mismatch')
    for family in ('source_sha256','input_sha256'):
        for path, digest in config[family].items():
            if sha(ROOT/path) != digest:
                raise ValueError(f'{family} drift: {path}')
    if sha(ROOT/config['checkpoint_path']) != config['checkpoint_sha256']:
        raise ValueError('checkpoint mismatch')

def result_identity(config, sid, seed, budget):
    return {'sample_id':sid,'seed':seed,'budget':budget,'method':'MH-CEM',
            'iterations':4,'elite_ratio':.1,'split':'dev_validation',
            'execution_config_id':config['execution_config_id'],
            'source_sha256':config['source_sha256'],'execution_device':'cuda',
            'gpu_model':'NVIDIA GeForce RTX 3080 Ti','precision':'FP32','batch_size':16,
            'state_storage':'cpu_cache_and_prefix','bucket_strategy':'none',
            'checkpoint_sha256':config['checkpoint_sha256'],'locked_test':False,
            'training':False,'closed_loop':False,'sidecar_alignment_passed':True}

def raw_path(sid, seed, budget):
    return OUT/f'solve_results/calibration/{hashlib.sha256(sid.encode()).hexdigest()[:16]}_MH-CEM_{seed}_b{budget}_k4_rho0p1.json'

def validate_scientific_result(row):
    o = row['outcome']; b = o['budget_receipt']
    if b['B_WM'] != row['budget'] or not 0 <= b['N_unique_transition_evals'] <= row['budget']:
        raise ValueError('budget accounting mismatch')
    if o['method'] != 'MH-CEM' or o['seed'] != row['seed'] or o['budget'] != row['budget']:
        raise ValueError('nested scientific identity mismatch')
    objective = o['best_objective']
    if objective is not None and (len(objective) != 5 or not all(math.isfinite(v) for v in objective)):
        raise ValueError('invalid/NaN/Inf objective')
    if row['support_horizon_counts'].get('SCORER_EXCEPTION',0) or any(
        k.startswith(('ScorerStateInconsistency:','BurdenSemanticsBlocked:')) for k in row['score_residuals']):
        raise ValueError('scorer inconsistency/exception; stop without restart')

def cases(config):
    return [(r['sample_id'],s,b) for b in (256,512)
            for r in read(OUT/'01_calibration_anchor_manifest.json')['selected']
            for s in (6311,6312,6313)]

def accept(config):
    rows = {256:[],512:[],1024:reference_inputs()}; inventory=[]
    expected_paths = set()
    for sid,seed,budget in cases(config):
        path = raw_path(sid,seed,budget); expected_paths.add(path)
        row = read(path)
        validate_identity(row,result_identity(config,sid,seed,budget))
        validate_scientific_result(row)
        rows[budget].append(row)
        inventory.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size})
    actual_paths = set((OUT/'solve_results/calibration').glob('*.json'))
    if actual_paths != expected_paths:
        raise ValueError('missing/extra calibration results')
    summaries = {str(b):summarize(values) for b,values in rows.items()}
    objectives = {b:{(r['sample_id'],r['seed']):r['outcome']['best_objective'] for r in values}
                  for b,values in rows.items()}
    pairs = {f'{a}_vs_{b}':compare_pairs(objectives[a],objectives[b]) for a,b in
             ((256,1024),(512,1024),(256,512))}
    reference = summaries['1024']
    tradeoffs = {str(b):{
        'runtime_mean_reduction':1-summaries[str(b)]['runtime_seconds']['mean']/reference['runtime_seconds']['mean'],
        'actual_transition_reduction':1-summaries[str(b)]['N_unique_transition_evals']/reference['N_unique_transition_evals'],
        'scoreable_rate_change':summaries[str(b)]['scoreable_rate']-reference['scoreable_rate'],
        'distinct_scoreable_h4_change':summaries[str(b)]['distinct_scoreable_h4']-reference['distinct_scoreable_h4']}
        for b in (256,512)}
    atomic(OUT/'10_raw_inventory.json',{'count':96,'files':inventory})
    atomic(OUT/'11_per_budget_summary.json',summaries)
    atomic(OUT/'12_paired_budget_comparison.json',pairs)
    atomic(OUT/'13_runtime_tradeoff_receipt.json',tradeoffs)
    atomic(OUT/'14_final_acceptance_receipt.json',{'verdict':'PASS','STEP_6_4C_BUDGET_CALIBRATION':'PASS',
        'case_count':96,'reference_count':48,'nominal_new_budget':36864,
        'actual_new_unique_transitions':sum(summaries[str(b)]['N_unique_transition_evals'] for b in (256,512)),
        'execution_config_id':config['execution_config_id'], 'CLOSED_LOOP_BUDGET':'RESEARCHER_DECISION_PENDING',
        'FORMAL_CLOSED_LOOP_PERFORMANCE':'NOT_STARTED','Stage B':'NOT_STARTED','locked_test':False,
        'claim_boundary':'CALIBRATION SUBSET; no method reselection or formal non-inferiority',
        'inventory_sha256':sha(OUT/'10_raw_inventory.json')})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--accept-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    config=read(OUT/'07_calibration_execution_config.json')
    validate_binding(config)
    reference_inputs()
    if args.accept_only:
        accept(config); return
    if not args.execute:
        print(json.dumps({'CPU_PREFLIGHT':'PASS','cases':96,'execution_config_id':config['execution_config_id'],
                          'gpu_search_started':False})); return
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
    if head != remote or dirty.strip():
        raise ValueError('launch requires HEAD=origin/main and tracked clean')
    import torch
    from run_step6_3d_one_cpu_solve_v1 import run, source_hashes, execution_identity
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != config['gpu_model']:
        raise ValueError('qualified RTX3080Ti unavailable')
    existing=list((OUT/'solve_results/calibration').glob('*.json'))
    if existing and not args.resume:
        raise ValueError('initial launch result count must be zero; explicit --resume required')
    start=datetime.now(timezone.utc).isoformat(); began=time.perf_counter(); completed=0
    allowed={raw_path(*c) for c in cases(config)}
    if not set(existing).issubset(allowed):
        raise ValueError('unexpected result identity namespace')
    for sid,seed,budget in cases(config):
        validate_binding(config)
        path=raw_path(sid,seed,budget); expected=result_identity(config,sid,seed,budget)
        if path.exists():
            row=read(path); validate_identity(row,expected); validate_scientific_result(row)
        else:
            row=run(sid,'MH-CEM',seed,budget,4,.1,batch_size=16,device='cuda')
            solver_identity=execution_identity(device='cuda',gpu_model=config['gpu_model'],batch_size=16,
                checkpoint_sha256=config['checkpoint_sha256'],source_sha256=source_hashes())
            validate_identity(row,{**{k:v for k,v in expected.items() if k not in ('execution_config_id','source_sha256')},
                                   **solver_identity})
            validate_scientific_result(row); validate_binding(config)
            row['solver_execution_config_id']=row['execution_config_id']
            row['solver_source_sha256']=row['source_sha256']
            row.update(expected); atomic(path,row)
        completed+=1
        elapsed=time.perf_counter()-began
        atomic(OUT/'09_runtime_status.json',{'start_time_utc':start,'update_time_utc':datetime.now(timezone.utc).isoformat(),
            'completed_cases':completed,'total_cases':96,'elapsed_wall_clock_seconds':elapsed,
            'cases_per_hour_this_invocation':completed*3600/elapsed,'launch_git_commit':head,
            'execution_config_id':config['execution_config_id'],'Stage B':'NOT_STARTED','locked_test':False})
        print(json.dumps({'completed':completed,'total':96,'budget':budget,'elapsed':elapsed}),flush=True)
    accept(config)
    print('STEP_6_4C_BUDGET_CALIBRATION=PASS; STOP',flush=True)

if __name__=='__main__':
    main()
