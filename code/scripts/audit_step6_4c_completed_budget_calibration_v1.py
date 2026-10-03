"""Independent raw-result accounting; no model loading, search, or environment API."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'code/artifacts/protocols/pi_jwm_step6_4c_mh_budget_calibration_v1_20261003'
PARENT = ROOT/'code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, v):
    p.write_text(json.dumps(v, indent=2, sort_keys=True, ensure_ascii=False,
                           allow_nan=False)+'\n', encoding='utf-8', newline='\n')

def main():
    config=read(OUT/'07b_calibration_execution_config_git_lf.json')
    fields={k:v for k,v in config.items() if k!='execution_config_id'}
    assert hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':')).encode()).hexdigest()==config['execution_config_id']
    for family in ('source_sha256','input_sha256'):
        for n,h in config[family].items():assert digest(ROOT/n)==h,n
    assert digest(ROOT/config['checkpoint_path'])==config['checkpoint_sha256']
    manifest=read(OUT/'01_calibration_anchor_manifest.json')
    anchors=[r['sample_id'] for r in manifest['selected']]
    seeds=[6311,6312,6313]
    assert len(anchors)==len(set(anchors))==16 and manifest['seeds']==seeds
    expected={(a,s) for a in anchors for s in seeds}
    references=read(OUT/'03_b1024_paired_reference_receipt.json')
    for n,h in references['parents'].items():assert digest(ROOT/n)==h,n
    historical=read(ROOT/next(n for n in references['parents'] if n.endswith('03_validation_execution_config_3080ti.json')))
    rows={256:{},512:{},1024:{}}; inventory=[]; paths=set()
    for item in references['files']:
        p=ROOT/item['path'];assert digest(p)==item['sha256']
        r=read(p); key=(r['sample_id'],r['seed']); assert key in expected and key not in rows[1024]
        for k,v in item['identity'].items():assert r[k]==v,(p,k)
        for k in ('source_sha256','execution_config_id','checkpoint_sha256','execution_device',
                  'gpu_model','precision','batch_size','state_storage','bucket_strategy','locked_test'):
            assert r[k]==historical[k],(p,k)
        rows[1024][key]=r
        inventory.append({'path':item['path'],'sha256':digest(p),'bytes':p.stat().st_size,'budget':1024,'reference_only':True})
    for b in (256,512):
        for a,s in sorted(expected):
            p=OUT/f'solve_results/calibration/{hashlib.sha256(a.encode()).hexdigest()[:16]}_MH-CEM_{s}_b{b}_k4_rho0p1.json'
            paths.add(p);r=read(p)
            identity={'sample_id':a,'seed':s,'budget':b,'method':'MH-CEM','iterations':4,'elite_ratio':.1,
                      'split':'dev_validation','source_sha256':config['source_sha256'],
                      'execution_config_id':config['execution_config_id'],'gpu':True,
                      'execution_device':'cuda','gpu_model':'NVIDIA GeForce RTX 3080 Ti','precision':'FP32',
                      'batch_size':16,'state_storage':'cpu_cache_and_prefix','bucket_strategy':'none',
                      'checkpoint_sha256':config['checkpoint_sha256'],'training':False,
                      'closed_loop':False,'locked_test':False,'sidecar_alignment_passed':True}
            for k,v in identity.items():assert r[k]==v,(p,k)
            rows[b][(a,s)]=r
            inventory.append({'path':p.relative_to(ROOT).as_posix(),'sha256':digest(p),'bytes':p.stat().st_size,'budget':b,'reference_only':False})
    assert paths==set((OUT/'solve_results/calibration').glob('*.json')) and len(paths)==96
    reported=read(OUT/'11_per_budget_summary.json'); summary={}
    parameter_digests=set()
    for b,group in rows.items():
        assert set(group)==expected and len(group)==48
        values=list(group.values()); times=[]
        for r in values:
            o=r['outcome'];c=o['budget_receipt']
            assert o['method']=='MH-CEM' and o['iterations']==4 and o['elite_ratio']==.1
            assert o['budget']==r['budget']==c['B_WM']==b and o['seed']==r['seed']
            assert 0<=c['N_unique_transition_evals']<=b
            assert (o['best_objective'] is not None)==(o['h4_scoreable_count']>0)
            if o['best_objective'] is not None:
                assert len(o['best_objective'])==5 and all(math.isfinite(x) for x in o['best_objective'])
                assert o['best_fingerprint'] is not None
            t=c['wall_clock_seconds_diagnostic_only'];assert math.isfinite(t) and t>0;times.append(t)
            assert not r['support_horizon_counts'].get('SCORER_EXCEPTION',0)
            assert not any(k.startswith(('ScorerStateInconsistency:','BurdenSemanticsBlocked:')) for k in r['score_residuals'])
            parameter_digests.add(r['parameter_digest'])
        totals={'case_count':48,'scoreable_cases':sum(r['outcome']['best_objective'] is not None for r in values),
            'complete_h4':sum(r['outcome']['complete_sequence_count'] for r in values),
            'distinct_scoreable_h4':sum(r['outcome']['h4_scoreable_count'] for r in values),
            'unscoreable_h4':sum(r['outcome']['h4_unscoreable_count'] for r in values),
            'return_birth_boundary':sum(n for r in values for k,n in r['score_residuals'].items()
                if k.startswith('H4_SUPPORT_BOUNDARY:') and 'UNSUPPORTED_FUTURE_RETURN_BIRTH' in k),
            'scorer_exceptions':0}
        for k in ('N_unique_transition_evals','N_cache_hits','N_proposed_steps','N_admitted_steps','N_rejected_steps','N_dead_end_branches'):
            totals[k]=sum(r['outcome']['budget_receipt'][k] for r in values)
        for k,v in totals.items():assert reported[str(b)][k]==v,(b,k)
        q=statistics.quantiles(times,n=100,method='inclusive')
        runtime={'median':statistics.median(times),'P90':q[89],'P95':q[94],
                 'max':max(times),'mean':statistics.mean(times),'sum_in_solve':sum(times)}
        for k,v in runtime.items():assert math.isclose(reported[str(b)]['runtime_seconds'][k],v,abs_tol=1e-9),(b,k)
        summary[str(b)]={**totals,'scoreable_rate':totals['scoreable_cases']/48,'runtime_seconds':runtime,
            'actual_transitions_per_second':totals['N_unique_transition_evals']/sum(times)}
    assert len(parameter_digests)==1,'model/encoder parameter digest differs from parent'
    comparisons={}; old=read(OUT/'12_paired_budget_comparison.json')
    names=['only_left_scoreable','only_right_scoreable','both_scoreable_left_better',
           'both_scoreable_right_better','both_scoreable_tie','both_unscoreable']
    for a,b in ((256,1024),(512,1024),(256,512)):
        counts=dict.fromkeys(names,0);clusters={};first_difference={str(i):0 for i in range(5)}
        for anchor,seed in sorted(expected):
            x=rows[a][(anchor,seed)]['outcome']['best_objective'];y=rows[b][(anchor,seed)]['outcome']['best_objective']
            if x is None and y is None:kind=5;value=0
            elif y is None:kind=0;value=1
            elif x is None:kind=1;value=-1
            elif tuple(x)<tuple(y):kind=2;value=1
            elif tuple(x)>tuple(y):kind=3;value=-1
            else:kind=4;value=0
            if x is not None and y is not None and x!=y:
                first_difference[str(next(i for i in range(5) if x[i]!=y[i]))]+=1
            counts[names[kind]]+=1
            clusters.setdefault(anchor,[]).append({'seed':seed,'outcome':value})
        assert sum(counts.values())==48 and all(len(v)==3 for v in clusters.values())
        key=f'{a}_vs_{b}';wtl={'win':counts[names[0]]+counts[names[2]],
            'tie':counts[names[4]]+counts[names[5]],'loss':counts[names[1]]+counts[names[3]]}
        assert old[key]['six_categories']==counts
        for k,v in wtl.items():assert old[key][k]==v
        assert old[key]['per_anchor_seed_outcomes']==clusters
        comparisons[key]={'six_categories':counts,**wtl,'total_paired_cases':48,
            'per_anchor_seed_outcomes':clusters,'first_differing_component_exploratory_only':first_difference}
    stage_b=list((PARENT/'solve_results/validation').glob('*_b256_*'))+list((PARENT/'solve_results/validation').glob('*_b512_*'))
    assert not stage_b,'original Stage B was started'
    status=read(OUT/'09_runtime_status.json');assert status['completed_cases']==96
    in_solve=sum(summary[str(b)]['runtime_seconds']['sum_in_solve'] for b in (256,512))
    runtime={'start_time_utc':status['start_time_utc'],'finish_last_case_utc':status['update_time_utc'],
        'matrix_elapsed_wall_clock_seconds':status['elapsed_wall_clock_seconds'],
        'new_in_solve_wall_clock_sum':in_solve,'matrix_setup_and_orchestration_overhead_seconds':status['elapsed_wall_clock_seconds']-in_solve,
        'effective_cases_per_hour':96*3600/status['elapsed_wall_clock_seconds'],
        'in_solve_transition_throughput':sum(summary[str(b)]['N_unique_transition_evals'] for b in (256,512))/in_solve,
        'note':'Search-internal timing excludes checkpoint/data/root preparation; matrix overhead is recorded separately. B1024 references are not rerun.'}
    receipt={'verdict':'PASS','scope':'CALIBRATION SUBSET; no formal non-inferiority or online budget freeze',
        'case_count':96,'reference_count':48,'anchor_count':16,'seeds':seeds,
        'execution_config_id':config['execution_config_id'],'per_budget':summary,'paired_comparisons':comparisons,
        'nominal_new_budget':36864,'actual_new_unique_transition_evals':sum(summary[str(b)]['N_unique_transition_evals'] for b in (256,512)),
        'parameter_digest_unchanged_from_stage_a':True,'runtime':runtime,'Stage B':'NOT_STARTED',
        'CLOSED_LOOP_BUDGET':'RESEARCHER_DECISION_PENDING','FORMAL_CLOSED_LOOP_PERFORMANCE':'NOT_STARTED','locked_test':False}
    write(OUT/'24_independent_raw_acceptance_receipt.json',receipt)
    write(OUT/'25_144_file_inventory.json',{'count':144,'files':inventory})
    # Preserve all original directories; standalone ZIP contains the paired raw evidence.
    archive=OUT/'mh_budget_calibration_raw_and_receipts.zip'
    included={ROOT/item['path'] for item in inventory}
    # Closure receipts 26+ bind this archive afterwards; keep them outside to
    # avoid an archive/manifest checksum cycle on a later independent audit.
    included.update(p for p in OUT.glob('*.json') if int(p.name[:2]) <= 25)
    included.add(OUT/'runtime/formal_calibration.log')
    included.discard(OUT/'26_archive_acceptance_receipt.json')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(included):z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in included:
            assert hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()==digest(p)
    write(OUT/'26_archive_acceptance_receipt.json',{'verdict':'PASS','archive':archive.relative_to(ROOT).as_posix(),
        'archive_sha256':digest(archive),'inventory_sha256':digest(OUT/'25_144_file_inventory.json'),
        'archive_file_count':len(included),'original_raw_preserved':True,'local_persistent_storage':str(OUT),
        'raw_new96_and_parent48_individually_verified':True,'locked_test':False})
    print(json.dumps({'independent_acceptance':'PASS','archive':'PASS','cases':96,'references':48,
                     'runtime':runtime,'summaries':summary,'pairs':{k:{n:v[n] for n in ('six_categories','win','tie','loss')} for k,v in comparisons.items()}}))

if __name__=='__main__':main()
