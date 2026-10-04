"""Independent raw reconstruction, math oracles and per-phase archive acceptance."""
import argparse,json,random,zipfile,os,hashlib
from collections import Counter
from pathlib import Path
from prepare_step6_4g_requalification_v1 import ROOT,OUT,read,sha,GRID,TRAIN_SEEDS,VALIDATION_SEEDS,digest,PROTOCOL_NAME
from run_step6_4g_requalification_v1 import phase_config,cases,raw_path,result_identity,validate_result,validate_binding,atomic

def oracle_win(a,b):
    if a is None or b is None:return int(a is not None)-int(b is not None)
    for x,y in zip(a,b):
        if x!=y:return 1 if x<y else -1
    return 0
def oracle_tuning(values):
    ranking=[]
    for k,r in GRID:
        outcomes=values[(k,r)]
        ranking.append({'config':(k,r),'h4_success_count':sum(v is not None for v in outcomes.values()),
          'paired_objective_net_outcome':sum(oracle_win(v,values[other][key]) for key,v in outcomes.items() for other in GRID if other!=(k,r))})
    ranking.sort(key=lambda r:(-r['h4_success_count'],-r['paired_objective_net_outcome'],*r['config']))
    return ranking
def oracle_bootstrap(clusters):
    means=[sum(clusters[k])/5 for k in sorted(clusters)];rng=random.Random(6316);n=len(means)
    boot=sorted(sum(means[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
    return {'ci95_lower':boot[250],'ci95_upper':boot[9750]}
def oracle_select(ci):
    s=ci['S-CEM_vs_HRS']['ci95_lower']>0;mh=ci['MH-CEM_vs_HRS']['ci95_lower']>0
    if not s and not mh:return 'HRS'
    if s and not mh:return 'S-CEM'
    if mh and not s:return 'MH-CEM'
    return 'MH-CEM' if ci['MH-CEM_vs_S-CEM']['ci95_lower']>0 else 'S-CEM'

def group_summary(rows):
    from pi_jwm.step6_4c_budget_calibration_v1 import summarize
    result=summarize(rows)
    result['scorer_inconsistencies']=sum(n for r in rows for k,n in r['score_residuals'].items() if k.startswith('ScorerStateInconsistency:'))
    result['nominal_B_WM_total']=sum(r['budget'] for r in rows)
    result['matrix_loading_and_setup_seconds']=sum(r.get('case_total_seconds',0)-r['outcome']['budget_receipt']['wall_clock_seconds_diagnostic_only'] for r in rows)
    result['all_seed_unscoreable_anchors']=sorted({r['sample_id'] for r in rows if all(x['outcome']['best_objective'] is None for x in rows if x['sample_id']==r['sample_id'])})
    return result

def analyze(phase,rows):
    from pi_jwm.step6_3d_method_selection_v1 import tune_cem_config,select_method
    if phase=='T':
        grids={};configs={};summaries={}
        for method in ('S-CEM','MH-CEM'):
            values={c:{(r['sample_id'],r['seed']):r['outcome']['best_objective'] for r in rows if r['method']==method and (r['iterations'],r['elite_ratio'])==c} for c in GRID}
            assert all(len(v)==96 for v in values.values())
            tuned=tune_cem_config(values);oracle=oracle_tuning(values)
            assert tuned['ranking']==oracle
            grids[method]=tuned;k,r=tuned['selected_config'];configs[method]={'K':k,'elite_ratio':r}
            for c in GRID:summaries[method+'_'+str(c)]=group_summary([r for r in rows if r['method']==method and (r['iterations'],r['elite_ratio'])==c])
        return {'summary.json':summaries,'tuning.json':{'grids':grids,'independent_oracle':'PASS','validation_used':False},
          'selected_configs.json':{'configs':configs,'selected_from':'repaired-domain Formal TRAIN only','locked_test':False}}
    if phase=='A':
        from run_step6_3d_formal_cpu_matrix_v1 import summarize_validation_budget,sensitivity_diagnostic
        anchors=tuple(read(OUT/'02_frozen_anchor_manifest.json')['anchors']['validation'])
        keyed={(1024,r['sample_id'],r['seed'],r['method']):r for r in rows}
        summary=summarize_validation_budget(keyed,anchors,VALIDATION_SEEDS,1024);ci={}
        for key,pair in summary['paired_outcomes'].items():
            left,right=key.split('_vs_');independent={}
            bins=Counter()
            for a in anchors:
                vals=[]
                for seed in VALIDATION_SEEDS:
                    x=keyed[(1024,a,seed,left)]['outcome']['best_objective'];y=keyed[(1024,a,seed,right)]['outcome']['best_objective']
                    w=oracle_win(x,y);vals.append(w)
                    category=('both_unscoreable' if x is None and y is None else 'only_right_scoreable' if x is None else
                      'only_left_scoreable' if y is None else 'both_scoreable_left_win' if w==1 else 'both_scoreable_right_win' if w==-1 else 'both_scoreable_tie')
                    bins[category]+=1
                independent[a]=vals
            assert independent==pair['per_anchor_seed_outcomes']
            assert all(pair['six_category_counts'][k]==bins[k] for k in pair['six_category_counts'])
            boot=oracle_bootstrap(independent)
            assert all(pair['cluster_bootstrap'][k]==v for k,v in boot.items());ci[key]=pair['cluster_bootstrap']
        selected=select_method({tuple(k.split('_vs_')):v for k,v in ci.items()})
        assert selected==oracle_select(ci)
        return {'summary.json':{m:group_summary([r for r in rows if r['method']==m]) for m in ('HRS','S-CEM','MH-CEM')},
          'paired.json':summary['paired_outcomes'],'sensitivity.json':sensitivity_diagnostic(summary),
          'selected_method.json':{'verdict':'PASS','selected_method':selected,'independent_oracle':'PASS',
            'primary_budget':1024,'scope':'Planner v1 pure-search backbone, not final hybrid','locked_test':False}}
    if phase=='B':
        from pi_jwm.step6_4d_full_cohort_b512_v1 import full_pair_analysis
        from pi_jwm.step6_4g_requalification_v1 import budget_requalification_gate
        ac=phase_config('A');parent=[]
        for c in cases(ac):
            if c['method']!='MH-CEM':continue
            r=read(raw_path('A',c));validate_result(r,result_identity(ac,c,read(OUT/'A/acceptance.json')['phase_gate_commit']));parent.append(r)
        old_inventory={r['path']:r['sha256'] for r in read(OUT/'A/inventory.json')['files']}
        for c in cases(ac):
            if c['method']=='MH-CEM' and sha(raw_path('A',c))!=old_inventory[raw_path('A',c).relative_to(ROOT).as_posix()]:raise ValueError('fresh PhaseA parent raw drift')
        values={512:{(r['sample_id'],r['seed']):r['outcome']['best_objective'] for r in rows},
                1024:{(r['sample_id'],r['seed']):r['outcome']['best_objective'] for r in parent}}
        paired,components=full_pair_analysis(values[512],values[1024]);clusters={a:[r['outcome'] for r in v] for a,v in paired['per_anchor_seed_outcomes'].items()}
        assert all(paired['cluster_bootstrap'][k]==v for k,v in oracle_bootstrap(clusters).items())
        # Independent first differing objective component and winner accounting.
        primary=0;wins=Counter();independent_bins=Counter();independent_components={n:{'B512_better':0,'B1024_better':0} for n in ('N_DDL','A_DDL','J_Delay','J_Burden','J_Effort')};equal=0
        for key,x in values[512].items():
            y=values[1024][key];w=oracle_win(x,y);wins[w]+=1
            category=('both_unscoreable' if x is None and y is None else 'only_right_scoreable' if x is None else
              'only_left_scoreable' if y is None else 'both_scoreable_left_better' if w==1 else 'both_scoreable_right_better' if w==-1 else 'both_scoreable_tie')
            independent_bins[category]+=1
            if x is not None and y is not None:
                first=next((i for i,(a,b) in enumerate(zip(x,y)) if a!=b),None)
                primary+=int(first is not None and first<3 and x[first]>y[first])
                if first is None:equal+=1
                else:independent_components[tuple(independent_components)[first]]['B512_better' if w==1 else 'B1024_better']+=1
        assert primary==components['PRIMARY_OBJECTIVE_DEGRADATION_COUNT']
        assert paired['win']==wins[1] and paired['tie']==wins[0] and paired['loss']==wins[-1]
        assert all(paired['six_categories'][k]==independent_bins[k] for k in paired['six_categories'])
        assert components['components']==independent_components and components['all_equal']==equal
        left,right=group_summary(rows),group_summary(parent);bins=paired['six_categories']
        gate=budget_requalification_gate({'paired_cases':len(rows),'scorer_exceptions':left['scorer_exceptions']+right['scorer_exceptions'],
          'scorer_inconsistencies':left['scorer_inconsistencies']+right['scorer_inconsistencies'],
          'only_B512_scoreable':bins['only_left_scoreable'],'only_B1024_scoreable':bins['only_right_scoreable'],
          'PRIMARY_OBJECTIVE_DEGRADATION_COUNT':primary,'mean_512':left['runtime_seconds']['mean'],'mean_1024':right['runtime_seconds']['mean'],
          'median_512':left['runtime_seconds']['median'],'median_1024':right['runtime_seconds']['median'],
          'budget_accounting_passed':left['N_unique_transition_evals']==163840 and right['N_unique_transition_evals']==327680})
        return {'summary.json':{'512':left,'1024':right},'paired.json':paired,'objective_components.json':components,
          'budget_gate.json':gate,'fresh_A_parent_refs.json':{'count':320,'Phase_A_inventory_SHA256':sha(OUT/'A/inventory.json')},
          'runtime_tradeoff.json':{'mean_reduction':1-left['runtime_seconds']['mean']/right['runtime_seconds']['mean'],
            'median_reduction':1-left['runtime_seconds']['median']/right['runtime_seconds']['median'],
            'candidate_reduction':None if right['distinct_scoreable_h4']==0 else 1-left['distinct_scoreable_h4']/right['distinct_scoreable_h4']}}
    raise ValueError('phase')

def accept_phase(phase,gate_commit):
    if not gate_commit or len(gate_commit)!=40:raise ValueError('phase gate commit required')
    path=OUT/phase
    if (path/'acceptance.json').exists():raise ValueError('phase already accepted; never rewrite acceptance or raw')
    config=phase_config(phase);validate_binding(config)
    if read(path/'execution_config.json')!=config:raise ValueError('actual config mismatch')
    planned=cases(config);expected={raw_path(phase,c) for c in planned}
    if set((path/'solve_results').glob('*.json'))!=expected:raise ValueError('missing/extra/duplicate raw identities')
    rows=[];inventory=[]
    for c in planned:
        p=raw_path(phase,c);r=read(p);validate_result(r,result_identity(config,c,gate_commit));rows.append(r)
        inventory.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size,'identity':c})
    if len({r['parameter_digest'] for r in rows})!=1:raise ValueError('model parameter drift')
    parameter_digest=rows[0]['parameter_digest']
    if phase!='T' and parameter_digest!=read(OUT/'T/acceptance.json')['parameter_digest']:
        raise ValueError('cross-phase frozen model parameter digest differs')
    artifacts=analyze(phase,rows)
    atomic(path/'inventory.json',{'count':len(rows),'files':inventory,'actual_unique_transitions':sum(r['outcome']['budget_receipt']['N_unique_transition_evals'] for r in rows),
      'nominal_transitions':sum(c['budget'] for c in planned)})
    for name,value in artifacts.items():atomic(path/name,value)
    archive=path/'raw_results.zip'
    if archive.exists():raise ValueError('existing unaccepted archive: inspect manually, no automatic overwrite')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for item in inventory:z.write(ROOT/item['path'],arcname=Path(item['path']).name)
    with zipfile.ZipFile(archive) as z:
        import hashlib
        for item in inventory:assert hashlib.sha256(z.read(Path(item['path']).name)).hexdigest()==item['sha256']
    atomic(path/'archive.json',{'verdict':'PASS','path':archive.relative_to(ROOT).as_posix(),'sha256':sha(archive),
      'inventory_sha256':sha(path/'inventory.json'),'raw_preserved':True,
      'local_persistent_backup_required_before_next_phase':read(OUT/PROTOCOL_NAME)['storage']['local_backup_root']+'/'+phase,'archive_roundtrip_verified':True})
    names=['inventory.json','archive.json','execution_config.json',*artifacts]
    passed=phase!='B' or artifacts['budget_gate.json']['passed']
    atomic(path/'acceptance.json',{'verdict':'PASS' if passed else 'FAIL','phase':phase,'cases':len(rows),
      'phase_gate_commit':gate_commit,'execution_config_id':config['execution_config_id'],
      'parameter_digest':parameter_digest,
      'artifact_SHA256':{n:sha(path/n) for n in names},'independent_math_oracle':'PASS',
      'evidence_class':'repaired-domain formal evidence','locked_test':False,
      'hard_stop':'commit/push and separate explicit next phase gate; no direct fall-through',
      'budget_decision':artifacts.get('budget_gate.json',{}).get('closed_loop_budget','RESEARCHER_DECISION_PENDING')})
    print(json.dumps({'phase':phase,'independent_acceptance':'PASS' if passed else 'FAIL','cases':len(rows)}))

def verify_accepted_local(phase):
    if os.name!='nt' or ROOT.drive.upper()!='D:':raise ValueError('Local D: persistent backup verification required, not remote ephemeral storage')
    p=OUT/phase;accepted=read(p/'acceptance.json');config=phase_config(phase);validate_binding(config)
    for n,h in accepted['artifact_SHA256'].items():
        if sha(p/n)!=h:raise ValueError('accepted phase artifact changed '+n)
    inv=read(p/'inventory.json');archive=read(p/'archive.json')
    if sha(ROOT/archive['path'])!=archive['sha256']:raise ValueError('archive SHA mismatch')
    rows=[]
    for c in cases(config):
        row=read(raw_path(phase,c));validate_result(row,result_identity(config,c,accepted['phase_gate_commit']));rows.append(row)
    with zipfile.ZipFile(ROOT/archive['path']) as z:
        for item in inv['files']:
            if sha(ROOT/item['path'])!=item['sha256'] or hashlib.sha256(z.read(Path(item['path']).name)).hexdigest()!=item['sha256']:raise ValueError('local raw/archive mismatch')
    # Independently rebuild statistics again from local raw, not remote summary.
    for n,v in analyze(phase,rows).items():
        if digest(v)!=digest(read(p/n)):raise ValueError('local statistical reconstruction mismatch '+n)
    receipt={'verdict':'PASS','phase':phase,'local_persistent_storage':str(p),
      'acceptance_SHA256':sha(p/'acceptance.json'),'archive_SHA256':archive['sha256'],
      'raw_count':len(rows),'local_independent_statistical_reconstruction':'PASS','locked_test':False}
    dest=p/'local_backup_acceptance.json'
    if dest.exists() and read(dest)!=receipt:raise ValueError('immutable local acceptance differs')
    if not dest.exists():atomic(dest,receipt)
    print(json.dumps({'local_backup_verdict':'PASS','phase':phase,'raw_count':len(rows)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('T','A','B'),required=True);p.add_argument('--phase-gate-commit');p.add_argument('--verify-accepted-local',action='store_true')
    a=p.parse_args()
    if a.verify_accepted_local:verify_accepted_local(a.phase)
    else:accept_phase(a.phase,a.phase_gate_commit)
