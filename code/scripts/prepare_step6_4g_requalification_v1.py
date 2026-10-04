"""Freeze repaired-domain protocol; TRAIN behavior catalog audit, no search."""
import collections,gzip,hashlib,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
from pi_jwm.step6_4g_requalification_v1 import FINAL_FALLBACK_POLICY,GRID,TRAIN_SEEDS,VALIDATION_SEEDS
OUT=ROOT/'code/artifacts/protocols/pi_jwm_step6_4g_repaired_search_v1_20261004'
OLD=ROOT/'code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929'
CAT=ROOT/'code/artifacts/protocols/pi_jwm_step6_3b_candidate_grammar_v1_20260929/02_train_structural_support_catalog.json'
DATA=ROOT/'code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1'
CHECKPOINT=ROOT/'code/artifacts/formal_training/pi_jwm_formal_train_v1_seed5601_20260924T112424Z/checkpoints/best.pt'
CHECKPOINT_SHA='941ee94131d406de914a79aeda43e929c92727631263d85221615422146a32c9'
PROTOCOL_NAME='00_protocol_r2.json'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(name,value):
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/name
    if p.exists():
        if read(p)==json.loads(json.dumps(value)):return
        raise RuntimeError('Immutable freeze differs: '+name)
    p.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def source_paths():
    # Conservative transitive scientific closure, plus orchestrator/auditor.
    return sorted([p.relative_to(ROOT).as_posix() for p in (ROOT/'code/src/pi_jwm').rglob('*.py')]+[
      'code/scripts/'+n for n in ('run_step6_3d_one_cpu_solve_v1.py','run_step6_3d_formal_cpu_matrix_v1.py',
      'run_step6_1_trained_candidate_rollout_preflight_v1.py','build_step5_1d_unified_model_chain_v1.py',
      'prepare_step6_4g_requalification_v1.py','run_step6_4g_requalification_v1.py',
      'audit_step6_4g_phase_v1.py')])
def source_hashes():return {n:hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in source_paths()}
def catalog_audit():
    from run_step6_3a_candidate_support_audit_v1 import structural_signature
    cat=read(CAT);split=read(DATA/'split_manifest.json')
    idxpath=DATA/'packages/samples/index.json.gz'
    if idxpath.exists():index=json.loads(gzip.decompress(idxpath.read_bytes()))
    else:index=read(DATA/'packages/samples/index.json')
    by=collections.defaultdict(list)
    for sample in index:
        if sample['metadata']['split']=='dev_train':by[sample['metadata']['trajectory_id']].append(sample['metadata'])
    joint=collections.Counter();pairs=collections.Counter();selected=collections.defaultdict(set)
    temporal={h:collections.Counter() for h in range(1,5)};windows=0;rawsha={}
    for tid in split['dev_train']:
        p=DATA/'raw'/f'{tid}.json.gz';rawsha[p.relative_to(ROOT).as_posix()]=sha(p)
        raw=json.loads(gzip.decompress(p.read_bytes()));steps=raw['steps'];decisions=raw['decisions']
        structures=[structural_signature(d,s,float(raw['environment']['slot_duration_s'])) for d,s in zip(decisions,steps)]
        for meta in by[tid]:
            frames=list(map(int,meta['future_action_frame_indices']));assert len(frames)==4;windows+=1
            for h in range(1,5):temporal[h][tuple(structures[f] for f in frames[:h])]+=1
            for f in frames:
                signature=structures[f];joint[signature]+=1;comm=steps[f]['action']['comm']['entries']
                selected[signature[0]].add(len({r['task_id'] for r in comm}))
                for row in comm:
                    rb=set(row['rb_indices']);n=int(decisions[f]['n_rb'])
                    starts=[s for s in range(n) if {(s+i)%n for i in range(len(rb))}==rb]
                    assert len(starts)==1;pairs[(starts[0],len(rb))]+=1
    assert windows==4416
    rebuilt={'joint_structural_signatures':[{'signature':list(s),'count_over_formal_window_positions':n} for s,n in sorted(joint.items())],
      'comm_start_width_pairs':[{'start':s,'width':w,'count_over_formal_window_positions':n} for (s,w),n in sorted(pairs.items())],
      'comm_selected_task_counts':{k:sorted(v) for k,v in sorted(selected.items())},
      'temporal_structural_prefixes':{str(h):[{'sequence':[list(s) for s in seq],'count':n} for seq,n in sorted(temporal[h].items())] for h in range(1,5)}}
    assert all(cat[k]==v for k,v in rebuilt.items()), 'STOP support catalog reconstruction differs'
    assert cat['source']=='FORMAL_TRAIN_ONLY' and cat['validation_used'] is False and cat['dataset_manifest_sha256']==sha(DATA/'formal_dataset_manifest.json')
    return {'verdict':'PASS','TRAIN_windows':windows,'catalog_SHA256':sha(CAT),'reconstructed_fields_match':list(rebuilt),
      'scientific_definition':'observed TRAIN behavior structural support, independent of current Planner execution eligibility',
      'execution_domain_definition':'intersection of unchanged behavior support and repaired current legality; not a catalog built from legal Planner winners',
      'catalog_rebuild_required':False,'TRAIN_recorded_actions_used':True,'future_targets_used':False,
      'Validation_outcomes_used':False,'raw_TRAIN_SHA256':rawsha,'joint_signatures':len(joint),'start_width_pairs':len(pairs)}

def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    if (OUT/PROTOCOL_NAME).exists():raise RuntimeError('Protocol frozen; no overwrite')
    if any(list((OUT/p/'solve_results').glob('*.json')) for p in ('T','A','B')):
        raise RuntimeError('initial formal result count must zero')
    if shutil.disk_usage(ROOT).free<5*1024**3:raise RuntimeError('local persistent backup disk too small')
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    write('01_support_catalog_semantic_audit.json',catalog_audit())
    # Reuse verified static audit only after all its current inputs/source bytes match.
    prior=ROOT/'code/artifacts/protocols/pi_jwm_step6_4f_comm_eligibility_v1_20261004/03_post_patch_full_audit.json.gz'
    post=json.loads(gzip.decompress(prior.read_bytes()))
    for n,h in post['source_SHA256'].items():assert sha(ROOT/n)==h, '6.4F source drift '+n
    for n,h in post['input_file_SHA256'].items():assert sha(ROOT/n)==h, '6.4F audited data drift '+n
    train=read(OLD/'15_train_anchor_manifest_objective_eligible.json');val=read(OLD/'16_validation_anchor_manifest_objective_eligible.json')
    anchors={'train':[r['sample_id'] for r in train['selected']],'validation':[r['sample_id'] for r in val['selected']]}
    rows={r['sample_id']:r for r in post['rows']}
    assert len(anchors['train'])==32 and len(anchors['validation'])==64
    audit_rows=[{'sample_id':a,**rows[a]['after']} for a in anchors['train']+anchors['validation']]
    assert all(not r['is_empty'] and r['exact_count']>0 for r in audit_rows)
    write('02_frozen_anchor_manifest.json',{'anchors':anchors,'static_domain_current':audit_rows,'original_anchor_manifest_SHA256':{
      'train':sha(OLD/'15_train_anchor_manifest_objective_eligible.json'),'validation':sha(OLD/'16_validation_anchor_manifest_objective_eligible.json')},
      '6_4F_full_static_archive_SHA256':sha(prior),'all_inputs_and_sources_reverified':True,'anchor_replacement':False})
    sources=source_hashes()
    inputs=dict(post['input_file_SHA256'])
    inputs['code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929/19_formal_selected_deadline_sidecars.json']=sha(OLD/'19_formal_selected_deadline_sidecars.json')
    inputs['docs/contracts/PIJWM_STEP_06_4F_COMM_ELIGIBILITY_V1.md']=sha(ROOT/'docs/contracts/PIJWM_STEP_06_4F_COMM_ELIGIBILITY_V1.md')
    for p in (DATA/'packages/normalization').iterdir():
        if p.is_file():inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    for family in ('samples','tensor','graph'):
        p=DATA/'packages'/family/'index.json'
        if p.exists():inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    shard_index=read(DATA/'packages/tensor/index.json')
    for shard in shard_index['shards']:
        p=DATA/'packages/samples'/shard['files']['samples']['path']
        assert sha(p)==shard['files']['samples']['sha256']
        inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    protocol={'schema_version':'PI-JWM-STEP-6.4G-REPAIRED-SEARCH-V1','evidence_class':'repaired-domain formal evidence',
      'old_evidence_class':'historical-under-pre-6.4F-domain','old_raw_reuse':False,
      'train_seeds':TRAIN_SEEDS,'validation_seeds':VALIDATION_SEEDS,'grid':GRID,
      'phases':{'T':{'cases':768,'budget':512,'nominal_transitions':393216},
        'A':{'cases':960,'budget':1024,'nominal_transitions':983040,'requires':'T independent PASS and committed new configs'},
        'B':{'cases':320,'budget':512,'nominal_transitions':163840,'requires':'A independent PASS and selected_method==MH-CEM'}},
      'total_if_all_phases':{'cases':2048,'nominal_transitions':1540096},
      'source_sha256':sources,'source_encoding':'Git LF scientific blobs; strict actual bytes required for GPU execution',
      'input_sha256':inputs,'checkpoint_path':CHECKPOINT.relative_to(ROOT).as_posix(),'checkpoint_sha256':CHECKPOINT_SHA,
      'eligibility_contract_SHA256':sources['code/src/pi_jwm/step6_4f_comm_eligibility_v1.py'],
      'support_catalog_SHA256':sha(CAT),'anchor_manifest_SHA256':sha(OUT/'02_frozen_anchor_manifest.json'),
      'gpu_model':'NVIDIA GeForce RTX 3080 Ti','execution_device':'cuda','precision':'FP32','batch_size':16,
      'state_storage':'cpu_cache_and_prefix','bucket_strategy':'none',
      'amp':False,'bf16':False,'fp16':False,'quantization':False,'torch_compile':False,
      'prior_mode':'mean','service_mode':'expectation','H':4,'Route':'EXPLICIT_NOOP_ONLY',
      'selection_rules':{'TRAIN':'original tune_cem_config: success,paired net,smaller K,rho',
        'A':'original select_method; anchor cluster5 seeds,10000 reps seed6316,lower95>0',
        'B':'identity320,scorer0,same_scoreability_set,primary_degradation0,mean/median runtime lower,strict budget'},
      'statistics_schema':{'per_group':['case_count','scoreable_cases','scoreable_rate','complete_h4','distinct_scoreable_h4','unscoreable_h4','return_birth_boundary','N_dead_end_branches','scorer_exceptions','scorer_inconsistencies','N_proposed_steps','N_admitted_steps','N_rejected_steps','N_unique_transition_evals','N_cache_hits','runtime mean/median/P90/P95/max/throughput'],
        'A_pairs':['six categories total320','win/tie/loss','64 anchor5 seed outcomes','cluster CI','diagnostic simultaneous sensitivity'],
        'B_pairs':['six categories total320','win/tie/loss','cluster CI','first difference N_DDL/A_DDL/J_Delay/J_Burden/J_Effort/all_equal','primary degradation count'],
        'hard_anchors':'retain and list every all-seed-unscoreable anchor'},
      'FINAL_FALLBACK_POLICY':FINAL_FALLBACK_POLICY,'locked_test':False,
      'formal_closed_loop_performance':'NOT_STARTED','old_Stage_B':'NOT_STARTED / DEFERRED',
      'runtime_bookkeeping':'per-case internal timer plus load/setup total; invocation start/finish UTC, elapsed, throughput, GPU diagnostics, resume segments; no short qualification benchmark substitute',
      'storage':{'minimum_free_bytes':5*1024**3,'local_backup_root':str(OUT),'raw_keep_original':True,
        'per_phase_archive':'raw inventory SHA256 and ZIP archive SHA256, independently verified locally before next phase gate',
        'original_dataset_not_in_Git':True},
      'scientific_change_during_run':'STOP; do not patch and resume',
      'systemic_failure_stop_plan':'exceptions/source drift/identity mismatch/NaN/scorer stop immediately, no auto-restart; low scoreability/hard anchors alone do not alter matrix or selection; runtime anomalies are diagnostics, never alter batch/precision'}
    write(PROTOCOL_NAME,protocol)
    template={'protocol_SHA256':sha(OUT/PROTOCOL_NAME),'source_sha256':sources,
      'phase_identity_formula':'SHA256(canonical JSON protocol hash+phase+source+parent receipt SHA+selected configs+CUDA FP32 batch16+checkpoint+eligibility/catalog/objective hashes)',
      'phase_commit_gate':'HEAD=origin/main exact --phase-gate-commit; all frozen source blobs equal protocol source at every phase',
      'T_parent':None,'A_parent':'new T configs+independent acceptance SHA','B_parent':'new T+new A selected-method and independent acceptance SHA',
      'no_scientific_source_changes_between_phases':True}
    write('03_phase_execution_identity_contract_r2.json',template)
    write('04_CPU_preflight_receipt_r2.json',{'verdict':'PASS','READY_FOR_GPU_LAUNCH':True,
      'support_catalog_audit':'PASS','all32_64_domains_nonempty':True,'checkpoint_SHA':'PASS','old_raw_reuse':False,
      'formal_result_count':0,'phase_T':'NOT_STARTED','phase_A':'NOT_STARTED','phase_B':'NOT_STARTED',
      'GPU_started':False,'GPU_device_qualification':'must inspect actual RTX3080Ti on SSH target before launch; TCP gateway alone insufficient',
      'local_free_bytes':shutil.disk_usage(ROOT).free,'minimum_free_bytes':5*1024**3,
      'protocol_sha256':sha(OUT/PROTOCOL_NAME),'locked_test':False})
    write('05_protocol_revision_receipt.json',{'final_protocol':PROTOCOL_NAME,'verdict':'PASS',
      'earlier_00_protocol':'CPU draft, never committed/executed; retained immutable',
      'revision_reason':'Before protocol commit: bind explicit normalization/sample packages, strengthen local archive/phase provenance and finish runner tests. Scientific source/algorithms unchanged.',
      'formal_result_count':0,'scientific_algorithm_changed':False,'GPU_started':False})
    print(json.dumps({'CPU_preflight':'PASS','READY_FOR_GPU_LAUNCH':True,'cases_if_all_phases':2048,'nominal_total':1540096}))

if __name__=='__main__':main()
