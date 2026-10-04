"""All frozen current windows, before/after symbolic audit; no WM or search.

Before patch, counterfactual uses only researcher-defined lifecycle intersection
in this process. Save the complete immutable baseline before changing source.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,gzip,hashlib,json,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
import torch
from pi_jwm.step3_3_model_input_tensor_v1 import LIFECYCLE_VOCAB
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import load_typed_dual_graph_batch
from pi_jwm.step5_5_full_sharded_loader_v1 import FORMAL_V1_WIRED_EDGES
from pi_jwm.step6_0a_candidate_generation_v1 import PlannerCandidateContext
from pi_jwm.step6_0c_planner_action_domain_v1 import context_from_current_raw
from pi_jwm.step6_3b_candidate_support_v1 import TrainStructuralSupportCatalog
import pi_jwm.step6_3b_candidate_grammar_v1 as grammar
import pi_jwm.step6_3c_candidate_domain_v1 as domain_module
from build_step5_1d_unified_model_chain_v1 import build_state
DATA=ROOT/'code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1'
PARENT=ROOT/'code/artifacts/protocols/pi_jwm_step6_3d_fixed_budget_search_v1_20260929'
CAT=ROOT/'code/artifacts/protocols/pi_jwm_step6_3b_candidate_grammar_v1_20260929/02_train_structural_support_catalog.json'
OUT=ROOT/'code/artifacts/protocols/pi_jwm_step6_4f_comm_eligibility_v1_20261004'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/name
    if p.exists():raise RuntimeError('Immutable receipt exists: '+name)
    p.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def life(state,slot):
    i=int(state['task_lifecycle_index'][0,slot]);return LIFECYCLE_VOCAB[i] if 0<=i<len(LIFECYCLE_VOCAB) else 'INVALID'
def expected_task_eligible(state,slot):
    return bool(state['task_presence'][0,slot]) and life(state,slot) in ('offloading','transmitting') and not bool(state.get('task_completed',torch.zeros_like(state['task_presence']))[0,slot])
def summary(rows):
    return {'windows':len(rows),'windows_with_stale_wireless_binding':sum(bool(r['stale_valid_flow_rows']) for r in rows),
      'stale_valid_wireless_flow_bindings':sum(len(r['stale_valid_flow_rows']) for r in rows),
      'stale_lifecycle_distribution':dict(sum((Counter(v['lifecycle'] for v in r['stale_valid_flow_rows']) for r in rows),Counter())),
      'old_wireless_eligible_tasks':sum(r['before']['eligible_count'] for r in rows),'new_wireless_eligible_tasks':sum(r['after']['eligible_count'] for r in rows),
      'before_empty_domains':sum(r['before']['is_empty'] for r in rows),'after_empty_domains':sum(r['after']['is_empty'] for r in rows),
      'before_structural_modes':sum(r['before']['mode_count'] for r in rows),'after_structural_modes':sum(r['after']['mode_count'] for r in rows),
      'before_exact_candidate_sum':sum(r['before']['exact_count'] for r in rows),'after_exact_candidate_sum':sum(r['after']['exact_count'] for r in rows),
      'domain_changed_count':sum(r['domain_changed'] for r in rows),'eligible_changed_count':sum(r['before']['eligible_tasks']!=r['after']['eligible_tasks'] for r in rows),
      'nonempty_to_empty':sum(not r['before']['is_empty'] and r['after']['is_empty'] for r in rows),'empty_to_nonempty':sum(r['before']['is_empty'] and not r['after']['is_empty'] for r in rows),
      'changed_ids':[r['sample_id'] for r in rows if r['domain_changed']]}
def details(domain):
    return {'eligible_tasks':sorted(domain.wireless_task_to_relation),'eligible_count':len(domain.wireless_task_to_relation),
      'mode_count':len(domain.modes),'exact_count':domain.exact_unique_single_step_count,'is_empty':domain.is_empty,
      'signatures':[list(m.signature) for m in domain.modes]}
def scan(phase):
    baseline=None
    if phase=='post':baseline=json.loads((OUT/'01_pre_patch_full_audit.json').read_text(encoding='utf-8'))
    original=grammar._wireless_bindings
    def expected(state,ids):return {k:v for k,v in original(state,ids).items() if expected_task_eligible(state,int(ids[k]))}
    catalog=TrainStructuralSupportCatalog.from_json(CAT)
    train_manifest=PARENT/'15_train_anchor_manifest_objective_eligible.json';val_manifest=PARENT/'16_validation_anchor_manifest_objective_eligible.json'
    selected={}
    for split,path in [('train_anchors',train_manifest),('validation_anchors',val_manifest)]:
        selected[split]={row['sample_id'] for row in json.loads(path.read_text(encoding='utf-8'))['selected']}
    assert len(selected['train_anchors'])==32 and len(selected['validation_anchors'])==64
    splits=json.loads((DATA/'split_manifest.json').read_text(encoding='utf-8'))
    index=json.loads((DATA/'packages/tensor/index.json').read_text(encoding='utf-8'))
    source_paths=[ROOT/'code/src/pi_jwm/step6_3b_candidate_grammar_v1.py',ROOT/'code/src/pi_jwm/step6_3c_candidate_domain_v1.py',ROOT/'code/src/pi_jwm/step6_4b_live_bridge_v1.py']
    if phase=='pre':
        write('00_pre_patch_source_snapshot.json',{'baseline_commit':'a6e0230f87397136fd274116436f2a85d1f45270','sources':{p.relative_to(ROOT).as_posix():{'sha256':sha(p),'text':p.read_text(encoding='utf-8')} for p in source_paths},'GPU':'NOT_USED','locked_test':False})
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [DATA/'formal_dataset_manifest.json',DATA/'split_manifest.json',CAT,train_manifest,val_manifest]}
    rows=[];started=time.perf_counter()
    previous={r['sample_id']:r for r in baseline['rows']} if baseline else {}
    for shard in sorted(index['shards'],key=lambda v:v['trajectory_id']):
        tid=shard['trajectory_id'];split='dev_train' if tid in splits['dev_train'] else 'dev_validation'
        paths={key:DATA/'packages'/key/shard['files'][key]['path'] for key in ('tensor','graph')}
        rawpath=DATA/'raw'/f'{tid}.json.gz'
        for family,p in paths.items():
            assert sha(p)==shard['files'][family]['sha256']
            hashes[p.relative_to(ROOT).as_posix()]=sha(p)
        hashes[rawpath.relative_to(ROOT).as_posix()]=sha(rawpath)
        tensor=load_flow_tensor_batch(paths['tensor']);graph=load_typed_dual_graph_batch(paths['graph'])
        raw=json.loads(gzip.decompress(rawpath.read_bytes()))
        assert len(tensor['sample_metadata'])==92
        for i,meta in enumerate(tensor['sample_metadata']):
            state,_=build_state(tensor,graph,i,wired_edges=FORMAL_V1_WIRED_EDGES)
            decision=raw['decisions'][int(meta['anchor_decision_frame'])]
            static=tensor['sample_static'][i]
            context=PlannerCandidateContext(static,({'frame_index':decision['frame_index'],'simulation_time_s':decision['simulation_time_s']},),state,meta['sample_id'])
            operational=context_from_current_raw(context,decision)
            old=domain_module.CandidateDomain.from_state(context,operational,state,operational.mobility_states,catalog)
            if phase=='pre':
                # Counterfactual stays in this audit process, not source/model/data.
                grammar._wireless_bindings=expected;domain_module._wireless_bindings=expected
                try:new=domain_module.CandidateDomain.from_state(context,operational,state,operational.mobility_states,catalog)
                finally:grammar._wireless_bindings=original;domain_module._wireless_bindings=original
                before,after=details(old),details(new)
            else:
                before=previous[meta['sample_id']]['before'];after=details(old)
                assert after==previous[meta['sample_id']]['after'],'Post != frozen expected intersection: '+meta['sample_id']
            ids={int(v):k for k,v in static['input_entity_index']['task'].items()};stale=[]
            for f in range(state['flow_presence'].shape[1]):
                if not all(bool(state[k][0,f]) for k in ('flow_known','flow_presence','carrying_active')):continue
                slot=int(state['flow_task_index'][0,f]);relation=int(state['flow_comm_relation_index'][0,f])
                if slot not in ids or not bool(state['task_presence'][0,slot]) or not 0<=relation<state['comm_presence'].shape[1]:continue
                if not all(bool(state[k][0,relation]) for k in ('comm_presence','comm_validity','comm_wireless_mask')):continue
                if int(state['comm_source_index'][0,relation])!=int(state['carrying_hop_source_index'][0,f]) or int(state['comm_target_index'][0,relation])!=int(state['carrying_hop_destination_index'][0,f]):continue
                if not expected_task_eligible(state,slot):stale.append({'task_id':ids[slot],'flow_slot':f,'lifecycle':life(state,slot),'task_completed':bool(state['task_completed'][0,slot])})
            rows.append({'sample_id':meta['sample_id'],'split':split,'stale_valid_flow_rows':stale,'before':before,'after':after,'domain_changed':before!=after})
        print(json.dumps({'phase':phase,'trajectories_complete':len(rows)//92,'windows':len(rows),'elapsed_s':round(time.perf_counter()-started,2)}),flush=True)
    if baseline:assert hashes==baseline['input_file_SHA256'],'Post input provenance changed'
    groups={'dev_train':[r for r in rows if r['split']=='dev_train'],'dev_validation':[r for r in rows if r['split']=='dev_validation']}
    assert len(groups['dev_train'])==4416 and len(groups['dev_validation'])==1104
    groups.update({name:[r for r in rows if r['sample_id'] in ids] for name,ids in selected.items()})
    result={'phase':phase,'summary':{name:summary(rs) for name,rs in groups.items()},'rows':rows,'input_file_SHA256':hashes,
       'source_SHA256':{p.relative_to(ROOT).as_posix():sha(p) for p in source_paths},'future_targets_loaded':False,'model_forward_count':0,'GPU':'NOT_USED','locked_test':False}
    write('01_pre_patch_full_audit.json' if phase=='pre' else '03_post_patch_full_audit.json',result)
    print(json.dumps({'phase':phase,'summary':{k:{x:v for x,v in value.items() if x!='changed_ids'} for k,value in result['summary'].items()}}),flush=True)
if __name__=='__main__':
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('pre','post'),required=True)
    scan(parser.parse_args().phase)
