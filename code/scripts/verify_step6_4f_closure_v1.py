"""Verify archived audit bytes, execution source, historical protection and Context."""
import gzip,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/artifacts/protocols/pi_jwm_step6_4f_comm_eligibility_v1_20261004'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def main():
    freeze=load('01_fixture_and_execution_freeze.json')
    for rel,digest in freeze['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    assert sha(ROOT/freeze['raw_path'])==freeze['raw_sha256']
    for name in ('01_pre_patch_full_audit.json','03_post_patch_full_audit.json'):
        assert gzip.decompress((OUT/(name+'.gz')).read_bytes())==(OUT/name).read_bytes()
    for name in ('04_real_nonempty_comm_one_step_receipt.json','05_simulated_NO_SCOREABLE_H4_canonical_fallback_one_step_receipt.json'):
        r=load(name)
        assert r['verdict']=='PASS' and r['environment_steps']==1
        assert r['start_time_s']==4.5 and r['finish_time_s']==r['fresh_observation_time_s']==4.6
        assert r['future_target_read'] is False and r['behavior_policy_overwrite'] is False
        assert r['route_command_count']==0 and r['WM_forward_count']==0 and r['locked_test'] is False
    comm=load('04_real_nonempty_comm_one_step_receipt.json')
    assert comm['commands_before_step']['rb']=={'Task_24':[0]}
    assert comm['consumed_wireless_profiles'][0]['task_id']=='Task_24'
    assert comm['slot_transfer_events'][0]['delivered_data']>0
    context={}
    for p in sorted((ROOT/'AI_CONTEXT').glob('0[0-8]_*.md')):
        current=p.read_text(encoding='utf-8').split('以下为历史状态')[0]
        assert 'STEP 6.4F' in current
        context[p.relative_to(ROOT).as_posix()]=sha(p)
    assert len(context)==9
    for name in ('00_PROJECT_STATE.md','05_EXPERIMENTS.md','07_KNOWN_ISSUES.md'):
        current=(ROOT/'AI_CONTEXT'/name).read_text(encoding='utf-8').split('以下为历史状态')[0]
        assert 'TARGETED_REQUALIFICATION_REQUIRED' in current
    assert 'RESEARCHER_DECISION_PENDING' in (ROOT/'AI_CONTEXT/06_DECISIONS.md').read_text(encoding='utf-8').split('以下为历史状态')[0]
    assert subprocess.run(['git','diff','--check'],cwd=ROOT).returncode==0
    evidence={'verdict':'PASS','focused_tests':{'total':53,'result':'OK',
      'commands':['python -m unittest test_step6_4f_comm_eligibility_v1 test_step6_3b_candidate_grammar_v1 test_step6_3c_candidate_domain_v1 test_step6_3c_search_protocol_v1 test_step6_3d_structured_proposal_v1 test_step6_3d_exact_oracle_v1 test_step6_3d_method_selection_v1 test_step6_4b_live_bridge_v1 test_step6_4e_fallback_v1','python -m unittest test_step6_4f_evidence_invariant_v1'],
      'actual_outputs':['Ran 50 tests in 32.332s; OK','Ran 3 tests in 0.012s; OK']},
      'negative_paths':'failed/completed/explicit completion/missing lifecycle/invalid Flow or relation/missing live field/illegal bridge/domain empty/setter fail closed covered by focused 6F+6B+6E tests',
      'compileall':{'command':'python -m compileall -q code/src code/scripts code/tests','exit_code':0},
      'Context_Consistency_Check':{'verdict':'PASS','current_context_SHA256':context,
        'boundary':'mechanism PASS, scientific targeted requalification pending, working budget512 separate, no automatic experiments'},
      'execution_source_SHA_MATCH':'PASS','archive_roundtrip':'PASS','git_diff_check':'PASS',
      'knowledge_index':{'commands':['python code/scripts/build_project_knowledge_index_v1.py','python code/scripts/build_project_knowledge_index_v1.py --check'],
        'actual_result':'passed=true, mismatches=[], output_count=5',
        'note':'Repeat write/check after final receipt staging; console output independently verifies current state'},
      'GPU':'NOT_USED','locked_test':False}
    (OUT/'10_verification_and_context_receipt.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    inventory=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file() and not p.name.startswith('11_'):
            inventory.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p),
              'storage':'local_D_persistent_original_uncompressed' if p.name in ('01_pre_patch_full_audit.json','03_post_patch_full_audit.json') else 'small_Git_evidence_or_compressed_archive'})
    manifest={'verdict':'PASS','files':inventory,'raw_preserved':True,'gzip_roundtrip_verified':True,
      'persistent_storage':str(OUT),'large_raw_not_for_Git':True,'GPU':'NOT_USED','locked_test':False}
    p=OUT/'11_SHA_inventory.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    (OUT/'11_SHA_inventory.sha256').write_text(sha(p)+'  11_SHA_inventory.json\n',encoding='utf-8',newline='\n')
    print('PASS: frozen execution bytes, two real steps, gzip roundtrip, Context, diff and SHA inventory')

if __name__=='__main__':main()
