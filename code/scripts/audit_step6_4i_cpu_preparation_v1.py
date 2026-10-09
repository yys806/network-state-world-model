"""Read-only independent acceptance checks; no simulator or model execution."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'code/artifacts/protocols/pi_jwm_step6_4i_closed_loop_readiness_v1_20261009_r2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):return json.loads((E/n).read_text(encoding='utf-8'))
def verify():
 registry=json.loads((ROOT/'docs/registries/results_registry.json').read_text(encoding='utf-8'))
 entry=next(r for r in registry['mechanism_readiness_acceptance_sources'] if r['id']=='RES-STEP-6.4I-CPU-PREPARATION-20261009')
 assert sha(ROOT/entry['audit'])==entry['audit_sha256'] and entry['status']==read('15_acceptance.json')['verdict']=='PASS'
 assert entry['locked_test_accessed'] is False and read('15_acceptance.json')['locked_test'] is False
 config=read('11_accepted_working_config.json');normal=read('05_two_cycle_real_feedback.json');fallback=read('09_real_fallback_and_metrics.json');correction=read('10_metric_correction_receipt.json')
 assert (config['method'],config['K'],config['rho'],config['B_WM'],config['H'])==('S-CEM',4,.2,512,4)
 for n in ['01_fixture_and_execution_identity.json','06_fallback_fixture.json']:
  d=read(n);assert not d['locked_test'];assert sha(ROOT/d['raw_path'])==d['raw_sha256']
  for path,expected in d['source_sha256'].items():
   source=E/'executed_metric_source.py.txt' if path.endswith('step6_4i_real_metrics_v1.py') else ROOT/path
   assert sha(source)==expected,(path,'executed source mismatch')
 for path,expected in config['protected_sources'].items():assert sha(ROOT/path)==expected,path
 assert normal['verdict']=='PASS' and normal['environment_steps']==2 and normal['fresh_root_not_predicted']
 assert normal['parameters_unchanged'] and normal['normalization_unchanged'] and normal['no_behavior_overwrite']
 assert all(c['budget']['N_unique_transition_evals']==64 and c['distinct_scoreable_h4']>0 for c in normal['cycles'])
 assert fallback['verdict']=='PASS' and fallback['environment_steps']==1 and fallback['WM_forward_count']==0
 assert not fallback['future_target_read'] and not fallback['behavior_overwrite']
 assert sha(E/'09_real_fallback_and_metrics.json')==correction['real_fallback_receipt_sha256']
 assert sha(ROOT/'code/src/pi_jwm/step6_4i_real_metrics_v1.py')==correction['current_metric_source_sha256']
 assert correction['corrected_observed_new_task_count']==len(set(correction['fresh_observed_ids'])-set(correction['initial_observed_ids']))==2
 assert not correction['action_or_search_repeated']
 for row in read('12_interface_audit.json')['chain']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
 if (E/'16_inventory.json').exists():
  for row in read('16_inventory.json')['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
 return {'verdict':'PASS','normal_real_steps':2,'fallback_real_steps':1,'GPU':'NOT_USED','locked_test':False,'claim':'CPU mechanism and evidence only; formal protocol/GPU qualification pending'}
if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False))
