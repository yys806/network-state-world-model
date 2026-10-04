"""Independent stdlib-only acceptance. Run only after all new cases finish.

Archive only new raw files. Existing 6.4C/Stage A raw stay at original paths,
referenced by SHA and persistent local inventories, never copied or rewritten.
"""
from pathlib import Path
import hashlib
import json
import math
import random
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'code/artifacts/protocols/pi_jwm_step6_4d_full_cohort_b512_v1_20261003'
SEEDS = [6311,6312,6313,6314,6315]
NAMES = ['N_DDL','A_DDL','J_Delay','J_Burden','J_Effort']


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v): (OUT/n).write_text(json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    config = read(OUT/'05_execution_config_3080ti.json'); fields = {k:v for k,v in config.items() if k!='execution_config_id'}
    assert hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':')).encode()).hexdigest()==config['execution_config_id']
    for family in ('source_sha256','input_sha256'):
        for n,h in config[family].items(): assert sha(ROOT/n)==h,n
    assert sha(ROOT/config['checkpoint_path'])==config['checkpoint_sha256']
    status = read(OUT/'09_runtime_status.json'); commit = status['protocol_freeze_commit']
    anchors = [r['sample_id'] for r in read(OUT/'01_full_cohort_manifest.json')['selected']]
    expected = {(a,s) for a in anchors for s in SEEDS}; assert len(expected)==320
    rows = {512:{},1024:{}}; inventory=[]; new_paths=set()
    for name,budget,reused in [('03_b1024_parent_receipt.json',1024,False),('04_b512_reuse_receipt.json',512,True)]:
        receipt=read(OUT/name)
        for n,h in receipt['parents'].items(): assert sha(ROOT/n)==h,n
        old_id=config['reuse_execution_config_id'] if reused else config['parent_stage_a_execution_config_id']
        old_config=read(ROOT/next(n for n in receipt['parents'] if ('07b_calibration_execution_config' if reused else '03_validation_execution_config_3080ti') in n))
        for item in receipt['files']:
            p=ROOT/item['path']; assert sha(p)==item['sha256'];r=read(p);key=(r['sample_id'],r['seed'])
            assert key in expected and key not in rows[budget]
            for k in ('source_sha256','execution_config_id','execution_device','gpu_model','precision','batch_size','checkpoint_sha256','state_storage','bucket_strategy','locked_test'):
                assert r[k]==old_config[k],(p,k)
            assert r['execution_config_id']==old_id
            rows[budget][key]=r
            inventory.append({'path':item['path'],'sha256':sha(p),'budget':budget,'reused':reused,'new':False,'bytes':p.stat().st_size})
    reused_keys=set(rows[512]);assert len(reused_keys)==48 and set(rows[1024])==expected
    for a,s in sorted(expected-reused_keys):
        p=OUT/f'solve_results/expansion/{hashlib.sha256(a.encode()).hexdigest()[:16]}_MH-CEM_{s}_b512_k4_rho0p1.json'
        r=read(p); new_paths.add(p)
        identity={'sample_id':a,'seed':s,'budget':512,'method':'MH-CEM','iterations':4,'elite_ratio':.1,
            'execution_config_id':config['execution_config_id'],'source_sha256':config['source_sha256'],
            'protocol_freeze_commit':commit,'gpu':True,'execution_device':'cuda','gpu_model':config['gpu_model'],
            'precision':'FP32','batch_size':16,'checkpoint_sha256':config['checkpoint_sha256'],
            'state_storage':'cpu_cache_and_prefix','bucket_strategy':'none','split':'dev_validation',
            'training':False,'closed_loop':False,'locked_test':False,'sidecar_alignment_passed':True}
        for k,v in identity.items():assert r[k]==v,(p,k)
        rows[512][(a,s)]=r
        inventory.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'budget':512,'reused':False,'new':True,'bytes':p.stat().st_size})
    assert new_paths==set((OUT/'solve_results/expansion').glob('*.json')) and len(new_paths)==272
    reported=read(OUT/'11_per_budget_summary.json');summaries={};digests=set()
    for b,group in rows.items():
        assert set(group)==expected;times=[]
        for r in group.values():
            o=r['outcome'];counter=o['budget_receipt'];digests.add(r['parameter_digest'])
            assert r['method']==o['method']=='MH-CEM' and r['iterations']==o['iterations']==4 and r['elite_ratio']==o['elite_ratio']==.1
            assert r['budget']==o['budget']==counter['B_WM']==b and r['seed']==o['seed']
            assert 0<=counter['N_unique_transition_evals']<=b
            assert (o['best_objective'] is not None)==(o['h4_scoreable_count']>0)
            if o['best_objective'] is not None:assert len(o['best_objective'])==5 and all(math.isfinite(v) for v in o['best_objective'])
            assert not r['support_horizon_counts'].get('SCORER_EXCEPTION',0)
            assert not any(k.startswith(('ScorerStateInconsistency:','BurdenSemanticsBlocked:')) for k in r['score_residuals'])
            t=counter['wall_clock_seconds_diagnostic_only'];assert math.isfinite(t) and t>0;times.append(t)
        summary={'case_count':320,'scoreable_cases':sum(r['outcome']['best_objective'] is not None for r in group.values()),
            'complete_h4':sum(r['outcome']['complete_sequence_count'] for r in group.values()),
            'distinct_scoreable_h4':sum(r['outcome']['h4_scoreable_count'] for r in group.values()),
            'unscoreable_h4':sum(r['outcome']['h4_unscoreable_count'] for r in group.values()),
            'return_birth_boundary':sum(n for r in group.values() for k,n in r['score_residuals'].items() if k.startswith('H4_SUPPORT_BOUNDARY:') and 'UNSUPPORTED_FUTURE_RETURN_BIRTH' in k),
            'scorer_exceptions':0}
        for k in ['N_unique_transition_evals','N_cache_hits','N_proposed_steps','N_admitted_steps','N_rejected_steps','N_dead_end_branches']:
            summary[k]=sum(r['outcome']['budget_receipt'][k] for r in group.values())
        for k,v in summary.items():assert reported[str(b)][k]==v,(b,k)
        q=statistics.quantiles(times,n=100,method='inclusive')
        runtime={'mean':statistics.mean(times),'median':statistics.median(times),'P90':q[89],'P95':q[94],'max':max(times),'sum_in_solve':sum(times)}
        for k,v in runtime.items():assert math.isclose(reported[str(b)]['runtime_seconds'][k],v,abs_tol=1e-8),(b,k)
        summary.update(scoreable_rate=summary['scoreable_cases']/320,runtime_seconds=runtime,actual_transitions_per_second=summary['N_unique_transition_evals']/sum(times))
        summaries[str(b)]=summary
    assert len(digests)==1
    bins=dict.fromkeys(['only_left_scoreable','only_right_scoreable','both_scoreable_left_better','both_scoreable_right_better','both_scoreable_tie','both_unscoreable'],0)
    components={n:{'B512_better':0,'B1024_better':0} for n in NAMES};equal=both=0;cluster={a:[] for a in anchors}
    hard={b:set() for b in (512,1024)}
    for a,s in sorted(expected):
        x,y=[rows[b][(a,s)]['outcome']['best_objective'] for b in (512,1024)]
        if x is None and y is None:kind=5;v=0
        elif x is None:kind=1;v=-1
        elif y is None:kind=0;v=1
        else:
            both+=1;v=(tuple(x)<tuple(y))-(tuple(x)>tuple(y));kind=2 if v>0 else 3 if v<0 else 4
            if not v:equal+=1
            else:components[NAMES[next(i for i in range(5) if x[i]!=y[i])]]['B512_better' if v>0 else 'B1024_better']+=1
        bins[list(bins)[kind]]+=1;cluster[a].append(v)
    for b in hard:
        hard[b]={a for a in anchors if all(rows[b][(a,s)]['outcome']['best_objective'] is None for s in SEEDS)}
    assert sum(bins.values())==320 and len(cluster)==64 and all(len(v)==5 for v in cluster.values())
    primary=sum(components[n]['B1024_better'] for n in NAMES[:3]);secondary=sum(components[n]['B1024_better'] for n in NAMES[3:])
    pair=read(OUT/'12_paired_budget_comparison.json');comp=read(OUT/'13_objective_first_difference.json')
    assert pair['six_categories']==bins and comp['components']==components and comp['all_equal']==equal and comp['both_scoreable_count']==both
    assert comp['PRIMARY_OBJECTIVE_DEGRADATION_COUNT']==primary and comp['BURDEN_EFFORT_ONLY_DEGRADATION_COUNT']==secondary
    wtl={'win':bins['only_left_scoreable']+bins['both_scoreable_left_better'],'tie':bins['both_scoreable_tie']+bins['both_unscoreable'],'loss':bins['only_right_scoreable']+bins['both_scoreable_right_better']}
    for k,v in wtl.items():assert pair[k]==v
    means=[sum(cluster[a])/5 for a in sorted(cluster)];rng=random.Random(6316)
    boot=sorted(sum(means[rng.randrange(64)] for _ in range(64))/64 for _ in range(10000))
    assert pair['cluster_bootstrap']['ci95_lower']==boot[250] and pair['cluster_bootstrap']['ci95_upper']==boot[9750]
    difficulty=read(OUT/'15_hard_anchor_and_residual_receipt.json')
    for k,v in {'B512':hard[512],'B1024':hard[1024],'intersection':hard[512]&hard[1024],'only_B512_hard':hard[512]-hard[1024],'only_B1024_hard':hard[1024]-hard[512]}.items():assert difficulty[k]==sorted(v)
    original=ROOT/'code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/solve_results/validation'
    assert not list(original.glob('*_b256_*')) and not list(original.glob('*_b512_*'))
    actual=sum(rows[512][k]['outcome']['budget_receipt']['N_unique_transition_evals'] for k in expected-reused_keys)
    write('17_independent_acceptance_receipt.json',{'verdict':'STEP_6_4D_FULL_COHORT_B512=PASS','coverage':320,'parent_cases':320,'reused_cases':48,'new_cases':272,
        'actual_new_unique_transitions':actual,'nominal_new_budget':139264,'per_budget':summaries,'six_categories':bins,**wtl,
        'objective_first_difference':comp,'cluster_bootstrap':pair['cluster_bootstrap'],'hard_anchors':difficulty,
        'protocol_freeze_commit':commit,'execution_config_id':config['execution_config_id'],'CLOSED_LOOP_BUDGET':'RESEARCHER_DECISION_PENDING',
        'FORMAL_CLOSED_LOOP_PERFORMANCE':'NOT_STARTED','Stage B':'NOT_STARTED','locked_test':False})
    write('18_640_raw_inventory.json',{'count':640,'new_count':272,'reuse_count':48,'parent_count':320,'files':inventory})
    included=set(new_paths)|{p for p in OUT.glob('*.json') if int(p.name[:2])<=18}|{OUT/'runtime/formal_expansion.log'}
    archive=OUT/'new_b512_expansion_raw_and_receipts.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(included):z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in included:assert hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()==sha(p)
    write('19_archive_acceptance_receipt.json',{'verdict':'PASS','archive':archive.relative_to(ROOT).as_posix(),'archive_sha256':sha(archive),
        'archive_file_count':len(included),'inventory_sha256':sha(OUT/'18_640_raw_inventory.json'),'local_persistent_storage':str(OUT),
        'existing_6_4c_and_stage_a_raw_not_copied_or_modified':True,'referenced_original_raw_sha_and_local_paths':True,'locked_test':False})
    print(json.dumps({'independent_acceptance':'PASS','archive':'PASS','coverage':320,'new_cases':272,'actual_new_unique_transitions':actual,
                      'six_categories':bins,**wtl,'components':comp,'summaries':summaries}))


if __name__=='__main__':main()
