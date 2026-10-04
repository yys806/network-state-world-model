"""Read-only provenance and sufficient invariant audit; no search/model forward."""
import ast, gzip, json
from types import SimpleNamespace
from pathlib import Path
import audit_step6_4f_comm_eligibility_v1 as audit
import torch
from pi_jwm.step4_4_structured_rssm_world_model_v1 import StructuredRSSMWorldModel, StructuredRSSMConfig
from pi_jwm.step3_3_model_input_tensor_v1 import LIFECYCLE_VOCAB

def invariant(state):
    """Sufficient condition, NOT a root-only domain equality assertion.

    Route NOOP cannot create Flow; presence/active only decrease. Task lifecycle
    changes only to completed. Unknown return requirement with no return slot
    makes completion impossible under the unchanged fixed-support rule.
    """
    active=state['flow_known'] & state['flow_presence'] & state['carrying_active']
    slots=set(int(state['flow_task_index'][0,f]) for f in torch.nonzero(active[0]).flatten())
    proof=[]
    for slot in sorted(slots):
        if not 0<=slot<state['task_presence'].shape[1]:return None
        if not audit.expected_task_eligible(state,slot):return None
        if bool(state['task_return_requirement_known'][0,slot]) or int(state['return_flow_index'][0,slot])>=0:return None
        proof.append({'task_slot':slot,'lifecycle':audit.life(state,slot),
          'return_requirement_known':False,'return_flow_index':-1})
    return {'active_flow_count':int(active.sum()),'tasks':proof,
            'condition':'NO_ACTIVE_KNOWN_FLOW_OR_ALL_ACTIVE_TASKS_COMMUNICABLE_AND_RETURN_REQUIREMENT_UNRESOLVED'}

def rule_oracle():
    # Invoke the existing rule function without constructing an nn.Module,
    # loading checkpoint, encoder/latent inference or candidate search.
    stub=SimpleNamespace(config=StructuredRSSMConfig())
    _,state,graph,action=StructuredRSSMWorldModel.synthetic_fixture(stub)
    state['task_lifecycle_index'][0,0]=LIFECYCLE_VOCAB.index('offloading')
    state['task_work_remaining'][0,0]=0
    action['comm_allocation_mask'].zero_();action['comp_task_index'].fill_(-1)
    learned={'vehicle_motion':torch.zeros((1,4,4)), 'csi':state['csi'].clone()}
    nxt,_=StructuredRSSMWorldModel.deterministic_transition(stub,state,action,learned,
        graph=graph,service_mode='expectation',generator=None)
    assert bool(nxt['task_completed'][0,0]) and bool(nxt['flow_presence'][0,0]) and bool(nxt['carrying_active'][0,0])
    assert not audit.expected_task_eligible(nxt,0)
    return {'completed_task_with_active_flow_possible':True,
      'scope':'synthetic existing deterministic rule oracle; not observed formal-search frequency',
      'model_forward_count':0,'search_calls':0}

def main():
    pre=json.loads((audit.OUT/'01_pre_patch_full_audit.json').read_text(encoding='utf-8'))
    post=json.loads((audit.OUT/'03_post_patch_full_audit.json').read_text(encoding='utf-8'))
    assert pre['input_file_SHA256']==post['input_file_SHA256']
    changed=set(post['summary']['validation_anchors']['changed_ids'])
    manifest=json.loads((audit.PARENT/'16_validation_anchor_manifest_objective_eligible.json').read_text(encoding='utf-8'))
    unchanged={r['sample_id'] for r in manifest['selected']}-changed
    certificates={}
    index=json.loads((audit.DATA/'packages/tensor/index.json').read_text(encoding='utf-8'))
    for shard in index['shards']:
        if not any(s.startswith(shard['trajectory_id']+'::') for s in unchanged):continue
        tensor=audit.load_flow_tensor_batch(audit.DATA/'packages/tensor'/shard['files']['tensor']['path'])
        graph=audit.load_typed_dual_graph_batch(audit.DATA/'packages/graph'/shard['files']['graph']['path'])
        for i,meta in enumerate(tensor['sample_metadata']):
            if meta['sample_id'] not in unchanged:continue
            state,_=audit.build_state(tensor,graph,i,wired_edges=audit.FORMAL_V1_WIRED_EDGES)
            cert=invariant(state)
            assert cert is not None, 'Cannot certify unchanged root across predicted horizon'
            certificates[meta['sample_id']]=cert
    assert len(certificates)==7
    inventory=json.loads((audit.ROOT/'code/artifacts/protocols/pi_jwm_step6_4d_full_cohort_b512_v1_20261003/18_640_raw_inventory.json').read_text(encoding='utf-8'))
    verified=[]
    for row in inventory['files']:
        p=audit.ROOT/row['path'];assert audit.sha(p)==row['sha256']
        raw=json.loads(p.read_text(encoding='utf-8'))
        assert raw['method']=='MH-CEM' and raw['budget']==row['budget'] and raw['locked_test'] is False
        outcome=raw['outcome']
        assert 'winner_sequence' not in outcome and 'predicted_states' not in outcome
        verified.append({'path':row['path'],'sha256':row['sha256'],'sample_id':raw['sample_id'],
          'budget':raw['budget'],'affected':raw['sample_id'] in changed})
    stage=[]
    for p in sorted((audit.PARENT/'solve_results/validation').glob('*.json')):
        raw=json.loads(p.read_text(encoding='utf-8'))
        assert raw['budget']==1024 and raw['locked_test'] is False
        stage.append({'path':p.relative_to(audit.ROOT).as_posix(),'sha256':audit.sha(p),
          'affected':raw['sample_id'] in changed})
    assert len(stage)==960 and sum(r['affected'] for r in stage)==855
    assert sum(r['affected'] for r in verified if r['budget']==512)==285
    protected=['step6_3d_fixed_budget_search_v1.py','step6_3d_structured_proposal_v1.py',
      'step6_3d_method_selection_v1.py','step6_3c_candidate_domain_v1.py',
      'step4_4_structured_rssm_world_model_v1.py','step6_1_trained_candidate_rollout_v1.py']
    proof={'FORMAL_SEARCH_EVIDENCE_REUSE':'TARGETED_REQUALIFICATION_REQUIRED',
      'B512_EVIDENCE_REQUALIFICATION_REQUIRED':True,'root_changed_validation_anchors':sorted(changed),
      'root_unchanged_future_invariant_certificates':certificates,
      'rule_oracle':rule_oracle(),
      'proof_basis':['prepare_anchor clones state without changing return/lifecycle fields',
        'Route EXPLICIT_NOOP_ONLY; no new Flow objects or task/return mapping',
        'flow_known preserved; presence and carrying_active cannot increase',
        'only lifecycle update is completed, impossible for certified active tasks with unresolved return requirement',
        'same eligible bindings for every subsequent state on every legal branch H1-H4; all other scientific solver sources unchanged'],
      'raw_sufficient_for_affected_winner_invariance':False,
      'raw_limitation':'fingerprint/objective/counts only, no full winner/candidate predicted-state trace',
      'old_640_raw_SHA_verified':verified,'old_Stage_A_960_inventory':stage,
      'minimum_proposed_requalification':{'affected_anchors':57,'Stage_A_B1024_three_methods':855,
        'MH_B512':285,'total_new_cases':1140,'nominal_transitions':1021440,
        'certified_unaffected_anchors':7,'reuse_B512_B1024_pairs':35,
        'TRAIN_tuning':'28/32 root anchors changed; historical tuning evidence does not validate repaired domain. K4 rho0.1 held by explicit researcher instruction; no automatic retuning.',
        'B256':'not required for current mainline; historical subset claims stay within old domain'},
      'PLANNER_V1_CLOSED_LOOP_B_WM':512,'GPU':'NOT_USED','locked_test':False}
    audit.write('07_formal_search_evidence_reuse_assessment.json',proof)
    small={'summary':post['summary'],'all_5520_post_rows_match_pre_frozen_expectation':True,
      'input_SHA_identical':True,'eligible_stale_tasks_after':0,
      'stale_Flow_objects_preserved':True,'full_audit_sha256':{
        n:audit.sha(audit.OUT/n) for n in ('01_pre_patch_full_audit.json','03_post_patch_full_audit.json')}}
    for name in small['full_audit_sha256']:
        compressed=gzip.compress((audit.OUT/name).read_bytes(),mtime=0)
        (audit.OUT/(name+'.gz')).write_bytes(compressed)
    audit.write('06_full_static_audit_summary.json',small)
    print(json.dumps({'verdict':proof['FORMAL_SEARCH_EVIDENCE_REUSE'],'certificates':len(certificates),'raw_verified':len(verified)}))

if __name__=='__main__':
    torch.set_num_threads(1);main()
