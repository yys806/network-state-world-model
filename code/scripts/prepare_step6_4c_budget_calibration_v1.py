"""Freeze static cohort first, then independently authenticate parent references."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/src'), str(ROOT/'code/scripts')]
from pi_jwm.step6_4c_budget_calibration_v1 import select_cohort, validate_identity, summarize

OUT = ROOT/'code/artifacts/protocols/pi_jwm_step6_4c_mh_budget_calibration_v1_20261003'
PARENT = ROOT/'code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929'
ARCHIVE = ROOT/'code/artifacts/protocols/pi_jwm_step6_3d_validation_stage_a_v1_20261001'
CONFIG = ROOT/'code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/03_validation_execution_config_3080ti.json'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    if path.exists() and path.read_bytes() != data:
        raise ValueError('refusing to overwrite frozen receipt: '+str(path))
    path.write_bytes(data)

def reference_inputs():
    manifest = OUT/'01_calibration_anchor_manifest.json'
    receipt = read(OUT/'03_b1024_paired_reference_receipt.json')
    for path, digest in receipt['parents'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('parent receipt drift')
    if receipt['manifest_sha256'] != sha(manifest):
        raise ValueError('cohort drift')
    rows = []
    historical = read(CONFIG)
    for item in receipt['files']:
        path = ROOT/item['path']
        if sha(path) != item['sha256']:
            raise ValueError('parent reference SHA drift')
        row = read(path)
        validate_identity(row, {**item['identity'], **{k: historical[k] for k in
            ('execution_config_id', 'execution_device', 'gpu_model', 'precision', 'batch_size',
             'checkpoint_sha256', 'source_sha256', 'state_storage', 'bucket_strategy', 'locked_test')}})
        rows.append(row)
    keys = {(r['sample_id'], r['seed']) for r in rows}
    expected = {(r['sample_id'], s) for r in read(manifest)['selected'] for s in (6311,6312,6313)}
    if keys != expected or len(rows) != 48:
        raise ValueError('paired references incomplete/duplicate')
    return rows

def main():
    source = PARENT/'16_validation_anchor_manifest_objective_eligible.json'
    original = read(source); config = read(CONFIG)
    relative = source.relative_to(ROOT).as_posix()
    if sha(source) != config['source_sha256'][relative]:
        raise ValueError('static manifest differs from pre-Stage-A frozen source')
    for path, digest in ((original['source_receipt'],original['source_sha256']),
                         (original['historical_manifest'],original['historical_manifest_sha256'])):
        if sha(ROOT/path) != digest:
            raise ValueError('static ancestry SHA mismatch')
    selected = select_cohort(original['selected'])
    manifest = {'source_manifest': relative, 'source_manifest_sha256': sha(source),
        'static_ancestry_verified_against_pre_stage_a_config': True,
        'selection_rule': 'sorted existing strata; floor(16/7)=2 each, first 2 labels +1; within stratum SHA256(UTF8 sample_id), sample_id tie-break',
        'selection_inputs': ['sample_id','stratum','concrete_count','eligible_compute_tasks'],
        'outcomes_used_for_selection': False, 'selected': selected, 'anchor_count': 16,
        'stratum_quotas': {s:sum(r['stratum']==s for r in selected) for s in sorted({r['stratum'] for r in selected})},
        'seeds': [6311,6312,6313], 'locked_test': False}
    # This immutable write precedes reading any selected B1024 objective.
    write(OUT/'01_calibration_anchor_manifest.json', manifest)
    protocol = {'scope':'CALIBRATION SUBSET; not full Validation; not original Stage B',
        'method':'MH-CEM','K':4,'rho':0.1,'budgets':[256,512], 'execution_order':[256,512],
        'anchors':16,'seeds':[6311,6312,6313],'new_cases':96,'nominal_new_budget':36864,
        'paired_reference_budget':1024,'paired_reference_cases':48,'rerun_b1024':False,
        'manifest_sha256':sha(OUT/'01_calibration_anchor_manifest.json'),
        'frozen_science':'unchanged CandidateDomain/Grammar/H4 Objective/Return/cache/CEM/prior mean/service expectation/checkpoint',
        'statistics':'paired descriptive six categories, sum48; no non-inferiority/equivalence claim; no bootstrap inference planned',
        'CLOSED_LOOP_BUDGET':'RESEARCHER_DECISION_PENDING','FORMAL_CLOSED_LOOP_PERFORMANCE':'NOT_STARTED',
        'Stage B':'NOT_STARTED','locked_test':False,'environment_actions':False,
        'failure_policy':'STOP on source/identity drift, NaN/Inf, scorer inconsistency/crash; never retune or automatically restart'}
    write(OUT/'02_calibration_protocol.json',protocol)
    inventory_path = ARCHIVE/'12_stage_a_local_file_inventory.json'
    inventory = {r['path']:r['sha256'] for r in read(inventory_path)['files']}
    acceptance = read(ARCHIVE/'13_stage_a_acceptance_receipt.json')
    if sha(inventory_path) != acceptance['inventory_sha256']:
        raise ValueError('parent archive inventory mismatch')
    selection = read(PARENT/'08_selected_method.json')
    if selection['selected_method'] != 'MH-CEM' or read(PARENT/'06_frozen_selected_cem_configs.json')['configs']['MH-CEM'] != {'K':4,'elite_ratio':.1}:
        raise ValueError('frozen method/config changed')
    files = []; rows = []
    for anchor in selected:
        for seed in manifest['seeds']:
            sid = anchor['sample_id']; short = hashlib.sha256(sid.encode()).hexdigest()[:16]
            path = PARENT/f'solve_results/validation/{short}_MH-CEM_{seed}_b1024_k4_rho0p1.json'
            rel = path.relative_to(ROOT).as_posix(); digest = sha(path)
            if inventory.get(rel) != digest:
                raise ValueError('raw does not match archived SHA')
            row = read(path)
            identity = {'sample_id':sid,'seed':seed,'method':'MH-CEM','budget':1024,
                        'iterations':4,'elite_ratio':.1,'split':'dev_validation',
                        'training':False,'closed_loop':False,'sidecar_alignment_passed':True}
            validate_identity(row,{**identity, **{k:config[k] for k in
                ('execution_config_id','execution_device','gpu_model','precision','batch_size',
                 'checkpoint_sha256','source_sha256','state_storage','bucket_strategy','locked_test')}})
            if row['outcome']['budget_receipt']['N_unique_transition_evals'] > 1024:
                raise ValueError('parent budget overrun')
            files.append({'path':rel,'sha256':digest,'identity':identity}); rows.append(row)
    write(OUT/'03_b1024_paired_reference_receipt.json',{'verdict':'PASS','count':48,'files':files,
        'manifest_sha256':sha(OUT/'01_calibration_anchor_manifest.json'),
        'parent_inventory_sha256':sha(inventory_path),'parent_execution_config_id':config['execution_config_id'],
        'parents':{p.relative_to(ROOT).as_posix():sha(p) for p in
            (CONFIG, ARCHIVE/'13_stage_a_acceptance_receipt.json',PARENT/'08_selected_method.json',
             PARENT/'07_validation_stage_a_primary_comparison_receipt.json')},'locked_test':False})
    write(OUT/'04_paired_b1024_reference_summary.json',{'budget':1024,'scope':'selected paired references only',**summarize(rows)})
    reference_inputs()
    print(json.dumps({'cohort':16,'strata':manifest['stratum_quotas'],'paired_reference_count':len(rows),'verdict':'PASS'}))

if __name__ == '__main__':
    main()
