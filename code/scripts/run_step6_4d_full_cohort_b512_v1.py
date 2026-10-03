"""Independent MH-only B512 expansion. No B256, Stage B or environment step."""
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from prepare_step6_4d_full_cohort_b512_v1 import ROOT, OUT, read, sha, frozen_inputs
from run_step6_4c_mh_budget_calibration_v1 import atomic, validate_binding, result_identity, validate_scientific_result
from pi_jwm.step6_4c_budget_calibration_v1 import validate_identity, summarize
from pi_jwm.step6_4d_full_cohort_b512_v1 import SEEDS, plan_cases, full_pair_analysis


def raw_path(a, s):
    return OUT/f'solve_results/expansion/{hashlib.sha256(a.encode()).hexdigest()[:16]}_MH-CEM_{s}_b512_k4_rho0p1.json'


def new_identity(config, a, s, commit):
    return {**result_identity(config, a, s, 512), 'protocol_freeze_commit': commit, 'gpu': True}


def validate_row(row, expected, parameter_digest):
    validate_identity(row, expected); validate_scientific_result(row)
    if row['outcome']['iterations'] != 4 or row['outcome']['elite_ratio'] != .1:
        raise ValueError('nested CEM configuration drift')
    if row['parameter_digest'] != parameter_digest: raise ValueError('model/encoder parameter digest drift')
    if (row['outcome']['best_objective'] is not None) != (row['outcome']['h4_scoreable_count'] > 0):
        raise ValueError('scoreability inconsistency')


def validate_git_source(config, commit, *, executing=False):
    # Bind immutable tracked Git blobs as well as current execution bytes.
    if not commit or len(commit) != 40: raise ValueError('exact protocol-freeze commit required')
    for n, digest in config['source_sha256'].items():
        data = subprocess.check_output(['git', 'show', commit+':'+n], cwd=ROOT)
        if hashlib.sha256(data).hexdigest() != digest: raise ValueError('protocol commit source mismatch: '+n)
    p = (OUT/'05_execution_config_3080ti.json').relative_to(ROOT).as_posix()
    if subprocess.check_output(['git', 'show', commit+':'+p], cwd=ROOT) != (ROOT/p).read_bytes():
        raise ValueError('configuration is not frozen in protocol commit')
    if executing:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        origin = subprocess.check_output(['git', 'rev-parse', 'origin/main'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True)
        if head != commit or origin != commit or dirty.strip():
            raise ValueError('launch requires exact protocol commit=HEAD=origin/main and tracked clean')


def accept(config, commit):
    validate_binding(config); anchors, reused, parent = frozen_inputs()
    left = dict(reused); files = []
    digest = next(iter(parent.values()))['parameter_digest']
    expected_paths = set()
    for a, s in plan_cases(anchors, reused):
        p = raw_path(a, s); expected_paths.add(p); row = read(p)
        validate_row(row, new_identity(config, a, s, commit), digest)
        left[(a, s)] = row
        files.append({'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'sample_id': a, 'seed': s, 'budget': 512})
    if expected_paths != set((OUT/'solve_results/expansion').glob('*.json')): raise ValueError('missing/extra new raw identity')
    if len(left) != 320 or set(left) != set(parent): raise ValueError('full paired coverage mismatch')
    values = {512: list(left.values()), 1024: list(parent.values())}
    summaries = {str(b): summarize(v) for b, v in values.items()}
    objectives = {b: {k: r['outcome']['best_objective'] for k, r in rows.items()} for b, rows in [(512, left), (1024, parent)]}
    paired, components = full_pair_analysis(objectives[512], objectives[1024])
    hard = {b: {a for a in anchors if all(objectives[b][(a, s)] is None for s in SEEDS)} for b in (512,1024)}
    difficulty = {'B512': sorted(hard[512]), 'B1024': sorted(hard[1024]), 'intersection': sorted(hard[512]&hard[1024]),
                  'only_B512_hard': sorted(hard[512]-hard[1024]), 'only_B1024_hard': sorted(hard[1024]-hard[512]),
                  'limitation': 'future Return-birth fixed-support boundary retained; no anchor removed'}
    a, b = summaries['512'], summaries['1024']
    tradeoff = {'mean_runtime_reduction': 1-a['runtime_seconds']['mean']/b['runtime_seconds']['mean'],
        'median_runtime_reduction': 1-a['runtime_seconds']['median']/b['runtime_seconds']['median'],
        'transition_reduction': 1-a['N_unique_transition_evals']/b['N_unique_transition_evals'],
        'candidate_count_reduction': 1-a['distinct_scoreable_h4']/b['distinct_scoreable_h4'],
        'scoreability_delta': a['scoreable_rate']-b['scoreable_rate'], 'runtime_scope': 'search internal; reused raw original timings retained'}
    atomic(OUT/'10_new_raw_inventory.json', {'count': len(files), 'files': files, 'reused_raw': '04_b512_reuse_receipt.json'})
    atomic(OUT/'11_per_budget_summary.json', summaries)
    atomic(OUT/'12_paired_budget_comparison.json', paired)
    atomic(OUT/'13_objective_first_difference.json', components)
    atomic(OUT/'14_runtime_tradeoff.json', tradeoff)
    atomic(OUT/'15_hard_anchor_and_residual_receipt.json', difficulty)
    actual = sum(r['outcome']['budget_receipt']['N_unique_transition_evals'] for k, r in left.items() if k not in reused)
    atomic(OUT/'16_final_acceptance_receipt.json', {'verdict': 'STEP_6_4D_FULL_COHORT_B512=PASS', 'coverage': 320,
        'reused_cases': len(reused), 'new_cases': len(files), 'parent_cases': 320, 'nominal_new_budget': len(files)*512,
        'actual_new_unique_transitions': actual, 'execution_config_id': config['execution_config_id'], 'protocol_freeze_commit': commit,
        'CLOSED_LOOP_BUDGET': 'RESEARCHER_DECISION_PENDING', 'FORMAL_CLOSED_LOOP_PERFORMANCE': 'NOT_STARTED',
        'Stage B': 'NOT_STARTED', 'locked_test': False, 'claim_boundary': 'budget quality/compute comparison, no equivalence or closed-loop performance claim'})


def main():
    p = argparse.ArgumentParser(); p.add_argument('--execute', action='store_true'); p.add_argument('--resume', action='store_true')
    p.add_argument('--accept-only', action='store_true'); p.add_argument('--protocol-commit')
    args = p.parse_args(); config = read(OUT/'05_execution_config_3080ti.json')
    validate_binding(config); anchors, reused, parent = frozen_inputs(); todo = plan_cases(anchors, reused)
    if args.protocol_commit: validate_git_source(config, args.protocol_commit, executing=args.execute)
    if args.accept_only: accept(config, args.protocol_commit); return
    if not args.execute:
        print(json.dumps({'CPU_PREFLIGHT': 'PASS', 'coverage': 320, 'reuse': len(reused), 'new_cases': len(todo),
                          'execution_config_id': config['execution_config_id'], 'gpu_started': False})); return
    if not args.protocol_commit: raise ValueError('--protocol-commit mandatory for GPU execution')
    import torch
    from run_step6_3d_one_cpu_solve_v1 import run, source_hashes, execution_identity
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != config['gpu_model']:
        raise ValueError('qualified RTX3080Ti unavailable')
    existing = set((OUT/'solve_results/expansion').glob('*.json'))
    if existing and not args.resume: raise ValueError('initial new-result count must be zero; explicit resume required')
    if not existing.issubset({raw_path(*k) for k in todo}): raise ValueError('unexpected raw namespace')
    parameter_digest = next(iter(parent.values()))['parameter_digest']
    # Validate every resume row before making any model forward.
    for a, s in todo:
        path = raw_path(a, s)
        if path.exists(): validate_row(read(path), new_identity(config, a, s, args.protocol_commit), parameter_digest)
    began = time.perf_counter(); start = datetime.now(timezone.utc).isoformat(); completed = 0; new_completed = 0
    for a, s in todo:
        validate_binding(config); path = raw_path(a, s); expected = new_identity(config, a, s, args.protocol_commit)
        if not path.exists():
            row = run(a, 'MH-CEM', s, 512, 4, .1, batch_size=16, device='cuda')
            solver = execution_identity(device='cuda', gpu_model=config['gpu_model'], batch_size=16,
                checkpoint_sha256=config['checkpoint_sha256'], source_sha256=source_hashes())
            base = {k:v for k,v in expected.items() if k not in ('execution_config_id','source_sha256','protocol_freeze_commit')}
            validate_row(row, {**base, **solver}, parameter_digest); validate_binding(config)
            row['solver_execution_config_id'] = row['execution_config_id']; row['solver_source_sha256'] = row['source_sha256']
            row.update(expected); atomic(path, row); new_completed += 1
        completed += 1; elapsed = time.perf_counter()-began
        atomic(OUT/'09_runtime_status.json', {'start_time_utc': start, 'update_time_utc': datetime.now(timezone.utc).isoformat(),
            'new_results_completed': completed, 'new_forward_cases_this_invocation': new_completed, 'new_target': len(todo),
            'coverage_completed': len(reused)+completed, 'coverage_target': 320, 'elapsed_wall_clock_seconds': elapsed,
            'cases_per_hour_this_invocation': new_completed*3600/elapsed, 'protocol_freeze_commit': args.protocol_commit,
            'execution_config_id': config['execution_config_id'], 'Stage B': 'NOT_STARTED', 'locked_test': False})
        print(json.dumps({'coverage': len(reused)+completed, 'total': 320, 'new_complete': completed, 'new_target': len(todo), 'elapsed': elapsed}), flush=True)
    accept(config, args.protocol_commit)
    print('STEP_6_4D_FULL_COHORT_B512=PASS; STOP', flush=True)


if __name__ == '__main__': main()
