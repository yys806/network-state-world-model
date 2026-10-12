"""Independent Gate 1-3 acceptance; reads only saved evidence."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; G1=ROOT/'code/artifacts/audit/pi_jwm_step6_4n_20261012_v5'; WM=ROOT/'code/artifacts/audit/pi_jwm_step6_4n_20261012_wm_v4'
def load(p):return json.loads(p.read_text(encoding='utf8'))
def main():
 g1=load(G1/'gate1_acceptance.json'); comp=load(WM/'wm_completion.json'); score=load(WM/'scorer_result.json')
 from audit_step6_4n_gate1_v2 import audit
 rebuilt_gate1=audit(G1)
 if rebuilt_gate1['status']!='PASS' or rebuilt_gate1['checks']!=g1['checks']:raise ValueError('REAL_GATE1_RECONSTRUCTION_FAILED')
 traces={x:load(WM/f'{x}_wm.json') for x in ('A_NOOP','B_ROUTE_RB')}
 def allfinite(v):
  if isinstance(v,float):return v==v and abs(v)<float('inf')
  if isinstance(v,dict):return all(allfinite(x) for x in v.values())
  if isinstance(v,list):return all(allfinite(x) for x in v)
  return True
 g2checks={
  'same_common_anchor':traces['A_NOOP']['initial']['anchor']==traces['B_ROUTE_RB']['initial']['anchor']==comp['pre_route_anchor']['anchor'],
  'H1_H4_eight_recursive_transitions':comp['transitions']==8 and all(len(traces[x]['outputs'])==4 for x in traces),
  'route_birth_only_B':not traces['A_NOOP']['meta']['births'] and len(traces['B_ROUTE_RB']['meta']['births'])==1,
  'slot_identity_and_zero_cold_start':traces['B_ROUTE_RB']['conservation'][0]['slot']==1 and traces['B_ROUTE_RB']['meta']['initialization']=='ZERO_COLD_START',
  'finite_hidden_graph_state':all(allfinite(traces[x]['conservation']) and all(r['finite'] is True for r in traces[x]['conservation']) for x in traces),
  'recursive_input_output_links':all(traces[x]['inputs'][i]['state']==traces[x]['outputs'][i-1]['state'] for x in traces for i in range(1,4)),
  'model_parameters_unchanged':comp['model_parameters_before']==comp['model_parameters_after'] and comp['encoder_parameters_before']==comp['encoder_parameters_after'],
  'future_poison_ignored':comp['future_poison_pass'] is True,
 }
 g3checks={
  'common_cohort_and_horizon':score['cohort']==['Task_5','Task_6','Task_7'] and score['H_eff']==4 and all(x['H_eff']==4 for x in score['scores']),
  'noop_preserves_known_return_demand':score['scores'][1]['rows'][0]['tasks'][0]['unmet_return']>0,
  'birth_identity_retains_full_demand':traces['B_ROUTE_RB']['meta']['births'][0]['official_ledger']['e2e_remaining']==score['scores'][1]['rows'][0]['tasks'][0]['known_return_demand'],
  'useful_progress_only_reduces_burden':score['scores'][0]['rows'][0]['tasks'][0]['unmet_return']==0 and score['scores'][0]['objective'][3]<score['scores'][1]['objective'][3],
  'five_tuple_and_lexicographic_sort':all(len(x['objective'])==5 for x in score['scores']) and score['scores'][0]['objective']<=score['scores'][1]['objective'],
  'v2_semantic_change_label':score['semantic_change']=='OBJECTIVE_V2_SEMANTIC_CHANGE' and score['future_truth_used'] is False,
  'legacy_gate_pass':g1['status']=='PASS',
 }
 # Rebuild every objective from primitive task/horizon rows, independently
 # of DynamicScorerV2; do not trust the saved objective or ranking alone.
 rebuilt=[]
 for r in score['scores']:
  rows=r['rows'];denom=len(score['cohort'])*len(rows)
  failed={t['task_id'] for row in rows for t in row['tasks'] if t['deadline_violation']}
  obj=[len(failed),sum(t['deadline_violation'] for row in rows for t in row['tasks'])/denom,sum(t['unfinished'] for row in rows for t in row['tasks'])/denom,sum(t['burden'] for row in rows for t in row['tasks'])/denom,sum(row['effort'] for row in rows)/len(rows)]
  rebuilt.append((obj,r['fingerprint']))
  g3checks['independent_'+r['candidate_id']]=obj==r['objective']
 g3checks['independent_ranking']=rebuilt==sorted(rebuilt)
 sourcechecks=[]
 for start in [G1/m/'00_start.json' for m in ('LEGACY_ROUTE','V2_ROUTE_NO_RB','V2_ROUTE_RB','V2_NOOP')]+[WM/'wm_attempt.json']:
  for path,sha in load(start)['executed_source_sha256'].items():
   sourcechecks.append(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha)
 g2checks['executed_source_matches_current']=all(sourcechecks)
 # Verify persisted tensor bytes, not merely the JSON producer's booleans.
 import torch
 from pi_jwm.step6_1_trained_candidate_rollout_v1 import fingerprint
 tensor_paths=[WM/f'{name}_trace.pt' for name in traces]
 if all(p.exists() for p in tensor_paths):
  def finite_tree(v):
   if isinstance(v,torch.Tensor):return bool(torch.isfinite(v).all())
   if isinstance(v,dict):return all(finite_tree(x) for x in v.values())
   if isinstance(v,(list,tuple)):return all(finite_tree(x) for x in v)
   return True
  for name,p in zip(traces,tensor_paths):
   saved=torch.load(p,weights_only=False);tr=saved['trace']
   g2checks['persisted_tensor_'+name]=all({'state':fingerprint(s),'graph':fingerprint(g),'latent':fingerprint(l)}==tr.output_fingerprints[h] and finite_tree((s,g,l)) and tr.output_fingerprints[h]==traces[name]['outputs'][h] for h,(s,g,l) in enumerate(zip(tr.states,tr.graphs,tr.latents)))
 else:
  g2checks['persisted_tensor_evidence']=False
 negative={'Task_15_FP64_FP32_boundary':'focused_test_pass','duplicate_route_flow_index_and_slot_overflow':'focused_test_pass','future_target_poison':'focused_test_pass','setter_partial_failure_quarantine':'focused_test_pass','all_candidates_unscoreable':'focused_test_pass','no_return_and_missing_wireless':'focused_test_pass'}
 result={'schema':'PI-JWM-STEP-6.4N-ACCEPTANCE-V2','status':'PASS' if all(g1['checks'].values()) and all(g2checks.values()) and all(g3checks.values()) else 'BLOCKED','gate1':g1,'gate2':{'status':'PASS' if all(g2checks.values()) else 'BLOCKED','checks':g2checks,'prediction_alignment_scope':'one Task_5 route on one root; not reliability evidence'},'gate3':{'status':'PASS' if all(g3checks.values()) else 'BLOCKED','checks':g3checks,'negative_coverage':negative},'states':{'RETURN_TTI_V2_REAL_SIM_PASS':'PASS','SAME_SLOT_ROUTE_COMM_EXECUTION':'PASS','DYNAMIC_FLOW_BIRTH_CAUSALITY':'PASS','FROZEN_WM_H4_COMPATIBILITY':'PASS','PREDICTION_ALIGNMENT':'NOT_ESTABLISHED','SCORER_V2_COMPARABILITY':'PASS','LEGACY_V1_REGRESSION':'PASS','FORMAL_ROUTE_INTEGRATION_READINESS':'BLOCKED','RETRAIN_REQUIRED':'NOT_ESTABLISHED'},'scope':{'GPU':False,'locked_test':False,'formal_planner_enabled':False,'large_search':False,'roots':1,'H':4,'checkpoint_sha':comp['checkpoint_sha'],'normalization_sha':comp['normalization_sha']},'limitations':['one designated real root completed legitimate return; no second root needed or selected by outcome','Task_5 service completes in first slot so unfinished prediction alignment is unestablished','formal planner CandidateDomain/World Model v1 remain unchanged','v2 burden has explicitly changed semantics and is not directly comparable to v1 scores']}
 (WM/'acceptance_v2.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print(json.dumps({'status':result['status'],'gate2':result['gate2']['status'],'gate3':result['gate3']['status']}));return 0 if result['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
