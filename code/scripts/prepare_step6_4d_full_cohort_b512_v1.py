"""Freeze 64x5 coverage and authenticate unchanged parent/reuse evidence."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/src'), str(ROOT/'code/scripts')]
from prepare_step6_4c_budget_calibration_v1 import read, sha, write, PARENT, ARCHIVE, CONFIG
from run_step6_4c_mh_budget_calibration_v1 import result_identity, validate_scientific_result
from pi_jwm.step6_4c_budget_calibration_v1 import validate_identity
from pi_jwm.step6_4d_full_cohort_b512_v1 import SEEDS, COMPONENTS, plan_cases, require_same_sources

OUT = ROOT/'code/artifacts/protocols/pi_jwm_step6_4d_full_cohort_b512_v1_20261003'
CAL = ROOT/'code/artifacts/protocols/pi_jwm_step6_4c_mh_budget_calibration_v1_20261003'
NEW_FILES = ['code/src/pi_jwm/step6_4d_full_cohort_b512_v1.py',
             'code/scripts/prepare_step6_4d_full_cohort_b512_v1.py',
             'code/scripts/run_step6_4d_full_cohort_b512_v1.py',
             'code/scripts/manage_step6_4d_remote_v1.py',
             'code/tests/test_step6_4d_full_cohort_b512_v1.py']


def frozen_inputs():
    manifest = read(OUT/'01_full_cohort_manifest.json')
    anchors = [a['sample_id'] for a in manifest['selected']]
    old = read(CONFIG); calibration = read(CAL/'07b_calibration_execution_config_git_lf.json')
    groups = {}
    for name, config, budget, expected_count in [('03_b1024_parent_receipt.json', old, 1024, 320),
                                               ('04_b512_reuse_receipt.json', calibration, 512, None)]:
        receipt = read(OUT/name)
        for n, h in receipt['parents'].items():
            if sha(ROOT/n) != h: raise ValueError('parent provenance drift: '+n)
        rows = {}
        for item in receipt['files']:
            p = ROOT/item['path']
            if sha(p) != item['sha256']: raise ValueError('raw SHA drift: '+item['path'])
            row = read(p); a, s = row['sample_id'], row['seed']
            identity = result_identity(config, a, s, budget)
            validate_identity(row, identity); validate_scientific_result(row)
            if row['outcome']['iterations'] != 4 or row['outcome']['elite_ratio'] != .1:
                raise ValueError('nested frozen CEM configuration mismatch')
            if (a, s) in rows: raise ValueError('duplicate parent/reuse identity')
            rows[(a, s)] = row
        if expected_count is not None and set(rows) != {(a, s) for a in anchors for s in SEEDS}:
            raise ValueError('incomplete parent cohort')
        if len(rows) != receipt['count']: raise ValueError('receipt count mismatch')
        groups[budget] = rows
    plan_cases(anchors, groups[512])
    if any(groups[1024][k]['parameter_digest'] != row['parameter_digest'] for k, row in groups[512].items()):
        raise ValueError('parent/reuse model parameter digest mismatch')
    return anchors, groups[512], groups[1024]


def main():
    static = PARENT/'16_validation_anchor_manifest_objective_eligible.json'
    old = read(CONFIG); previous = read(CAL/'07b_calibration_execution_config_git_lf.json')
    if sha(static) != old['source_sha256'][static.relative_to(ROOT).as_posix()]:
        raise ValueError('frozen Stage A anchor manifest mismatch')
    anchors = read(static)['selected']
    plan_cases([a['sample_id'] for a in anchors], set())
    write(OUT/'01_full_cohort_manifest.json', {'selected': anchors, 'anchor_count': 64, 'seeds': list(SEEDS),
        'selection': 'all original frozen Stage A anchors, original order; no resampling/outcome selection',
        'source_manifest': static.relative_to(ROOT).as_posix(), 'source_manifest_sha256': sha(static), 'locked_test': False})
    inventory_path = ARCHIVE/'12_stage_a_local_file_inventory.json'
    accepted = read(ARCHIVE/'13_stage_a_acceptance_receipt.json')
    if accepted['inventory_sha256'] != sha(inventory_path): raise ValueError('Stage A inventory lineage mismatch')
    inventory = {r['path']: r['sha256'] for r in read(inventory_path)['files']}
    if read(PARENT/'08_selected_method.json')['selected_method'] != 'MH-CEM': raise ValueError('method changed')
    if read(PARENT/'06_frozen_selected_cem_configs.json')['configs']['MH-CEM'] != {'K': 4, 'elite_ratio': .1}:
        raise ValueError('frozen config changed')
    files = []
    for a in anchors:
        for s in SEEDS:
            p = PARENT/f"solve_results/validation/{hashlib.sha256(a['sample_id'].encode()).hexdigest()[:16]}_MH-CEM_{s}_b1024_k4_rho0p1.json"
            n = p.relative_to(ROOT).as_posix()
            if inventory.get(n) != sha(p): raise ValueError('Stage A raw/archive SHA mismatch')
            row = read(p); validate_identity(row, result_identity(old, a['sample_id'], s, 1024))
            validate_scientific_result(row)
            files.append({'path': n, 'sha256': sha(p), 'sample_id': a['sample_id'], 'seed': s})
    parent_paths = [CONFIG, inventory_path, ARCHIVE/'13_stage_a_acceptance_receipt.json',
                    PARENT/'08_selected_method.json', PARENT/'06_frozen_selected_cem_configs.json']
    write(OUT/'03_b1024_parent_receipt.json', {'verdict': 'PASS', 'count': 320, 'files': files,
        'parent_execution_config_id': old['execution_config_id'],
        'parents': {p.relative_to(ROOT).as_posix(): sha(p) for p in parent_paths}, 'rerun': False, 'locked_test': False})
    # Complete 6.4C source closure remains byte-for-byte unchanged. New orchestration
    # is added separately; none of the scientific solver/runtime files are patched.
    current = {n: sha(ROOT/n) for n in previous['source_sha256']}
    require_same_sources(previous['source_sha256'], current)
    cal_accept = read(CAL/'24_independent_raw_acceptance_receipt.json')
    if cal_accept['verdict'] != 'PASS' or cal_accept['case_count'] != 96: raise ValueError('6.4C not accepted')
    reuse_inventory = read(CAL/'25_144_file_inventory.json')
    if sha(CAL/'25_144_file_inventory.json') != read(CAL/'26_archive_acceptance_receipt.json')['inventory_sha256']:
        raise ValueError('6.4C inventory lineage mismatch')
    reusable = [r for r in reuse_inventory['files'] if r['budget'] == 512 and not r['reference_only']]
    if len(reusable) != 48: raise ValueError('expected original 48 B512 results')
    for item in reusable:
        p = ROOT/item['path']; row = read(p)
        if sha(p) != item['sha256']: raise ValueError('reuse raw SHA mismatch')
        validate_identity(row, result_identity(previous, row['sample_id'], row['seed'], 512))
        validate_scientific_result(row)
        if row['solver_source_sha256'] != {n: current[n] for n in row['solver_source_sha256']}:
            raise ValueError('scientific solver/source mismatch')
    reuse_parents = [CAL/'07b_calibration_execution_config_git_lf.json', CAL/'24_independent_raw_acceptance_receipt.json',
                     CAL/'25_144_file_inventory.json', CAL/'26_archive_acceptance_receipt.json']
    write(OUT/'04_b512_reuse_receipt.json', {'REUSE_VALID': True, 'count': 48, 'files': reusable,
        'verdict': 'PASS', 'copied_or_modified_or_rerun': False,
        'source_closure_sha256': current, 'scientific_semantics': 'all 6.4C frozen source bytes unchanged; same one-solve primitive',
        'parents': {p.relative_to(ROOT).as_posix(): sha(p) for p in reuse_parents}, 'locked_test': False})
    frozen_inputs()
    write(OUT/'02_expansion_protocol.json', {'method': 'MH-CEM', 'K': 4, 'rho': .1, 'H': 4,
        'budgets': [512], 'paired_reference_budget': 1024, 'coverage_cases': 320,
        'reused_cases': 48, 'new_cases': 272, 'nominal_new_budget': 139264,
        'seeds': list(SEEDS), 'execution_order': 'original frozen manifest order, then ascending five seeds; skip verified reused identities',
        'bootstrap': {'unit': 'anchor', 'clusters': 64, 'seeds_per_cluster': 5, 'replications': 10000, 'seed': 6316, 'ci': 'percentile 95%'},
        'pairwise': 'frozen strict lexicographic Objective; None stays unscoreable; six bins sum320',
        'objective_first_difference': list(COMPONENTS)+['all_equal'],
        'primary_degradation': 'both scoreable; first differing component favors B1024 and belongs to N_DDL/A_DDL/J_Delay',
        'burden_effort_only_degradation': 'both scoreable; first differing component favors B1024 and belongs to J_Burden/J_Effort',
        'runtime': 'original per-case in-solve timing; linear-interpolated quantiles; matrix/setup overhead separately',
        'hard_anchor': 'all five seeds unscoreable; intersection and both directional differences retained',
        'failure_policy': 'STOP on identity/source drift, NaN/Inf, scorer inconsistency or crash; no automatic restart/retuning',
        'claim_boundary': 'full Validation cohort budget quality/compute comparison; no equivalence, optimal-budget or real closed-loop claim',
        'CLOSED_LOOP_BUDGET': 'RESEARCHER_DECISION_PENDING', 'FORMAL_CLOSED_LOOP_PERFORMANCE': 'NOT_STARTED',
        'Stage B': 'NOT_STARTED', 'locked_test': False, 'environment_actions': False})
    sources = {**current, **{n: sha(ROOT/n) for n in NEW_FILES}}
    config = {k: previous[k] for k in ('execution_device','gpu_model','precision','batch_size','amp','bf16','fp16',
              'quantization','torch_compile','bucket_strategy','state_storage','prior','service','checkpoint_path','checkpoint_sha256')}
    config.update({'name': 'STEP_6_4D_FULL_COHORT_MH_CEM_B512_3080TI', 'source_sha256': sources,
        'input_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(OUT.glob('0[1-4]_*.json'))},
        'baseline_git_commit': 'cb19ca74064a76bb28b98b1fa126b444259676e3',
        'git_source_identity': 'exact owning protocol-freeze commit required by --protocol-commit; recorded at launch; avoids self-referential commit hash',
        'method': 'MH-CEM', 'K': 4, 'rho': .1, 'H': 4, 'budgets': [512], 'anchors': 64, 'seeds': list(SEEDS),
        'parent_stage_a_execution_config_id': old['execution_config_id'], 'reuse_execution_config_id': previous['execution_config_id'],
        'locked_test': False})
    config['execution_config_id'] = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    write(OUT/'05_execution_config_3080ti.json', config)
    print(json.dumps({'protocol': 'FROZEN', 'REUSE_VALID': True, 'reuse': 48, 'parent': 320, 'new_cases': 272,
                      'execution_config_id': config['execution_config_id'], 'gpu_started': False}))


if __name__ == '__main__': main()
