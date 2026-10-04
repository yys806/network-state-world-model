"""No real matrix files; synthetic receipts exercise resume and summaries."""
import copy,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
from run_step6_4g_requalification_v1 import validate_result
from audit_step6_4g_phase_v1 import analyze,oracle_select

def synthetic_row(method='S-CEM',seed=6301,budget=512,k=3,rho=.2):
    fields=dict(method=method,seed=seed,budget=budget,iterations=k,elite_ratio=rho,batch_size=16,
      phase='T',execution_device='cuda',precision='FP32',gpu_model='NVIDIA GeForce RTX 3080 Ti',
      checkpoint_sha256='checkpoint',source_sha256={'source.py':'source'},execution_config_id='config',
      eligibility_contract_SHA256='eligibility',support_catalog_SHA256='support',sample_id='synthetic',split='dev_train')
    from pi_jwm.step6_3d_fixed_budget_search_v1 import quotas
    cumulative=0;iters=[]
    for i,q in enumerate(quotas(budget,k)):
        cumulative+=q;iters.append(dict(iteration=i+1,quota=q,unique_transition_evals=cumulative))
    out={key:fields[key] for key in ('method','seed','budget','iterations','elite_ratio','batch_size')}
    out.update(best_objective=None,best_fingerprint=None,h4_scoreable_count=0,h4_unscoreable_count=8,complete_sequence_count=8,
      iteration_rows=iters,budget_receipt=dict(B_WM=budget,N_unique_transition_evals=budget,N_cache_hits=3,
        N_complete_sequences=8,N_dead_end_branches=0,N_proposed_steps=budget+3,N_admitted_steps=budget+3,N_rejected_steps=0,wall_clock_seconds_diagnostic_only=1.))
    return {**fields,'outcome':out,'support_horizon_counts':{},'score_residuals':{},'case_total_seconds':2.}

class RunnerTests(unittest.TestCase):
    def test_resume_rejects_every_phase_scientific_identity_change(self):
        row=synthetic_row();expected={k:v for k,v in row.items() if k not in ('outcome','score_residuals','support_horizon_counts','case_total_seconds')}
        validate_result(row,expected)
        for key in expected:
            changed=copy.deepcopy(row);changed[key]='wrong'
            with self.assertRaises(ValueError,msg=key):validate_result(changed,expected)
    def test_budget_nan_scorer_and_iteration_fail_stop(self):
        row=synthetic_row()
        for mutate in (
          lambda r:r['outcome']['budget_receipt'].__setitem__('N_unique_transition_evals',511),
          lambda r:r['outcome']['iteration_rows'][0].__setitem__('quota',169),
          lambda r:r['outcome']['budget_receipt'].__setitem__('wall_clock_seconds_diagnostic_only',float('nan')),
          lambda r:r['support_horizon_counts'].__setitem__('SCORER_EXCEPTION',1),
          lambda r:r['score_residuals'].__setitem__('ScorerStateInconsistency:injected',1)):
            changed=copy.deepcopy(row);mutate(changed)
            with self.assertRaises(ValueError):validate_result(changed,{})
    def test_synthetic_six_categories_total320_independent_selection(self):
        anchors=['synthetic-'+str(i) for i in range(64)];rows=[]
        for method in ('HRS','S-CEM','MH-CEM'):
            for i,a in enumerate(anchors):
                for j,s in enumerate(range(6311,6316)):
                    r=synthetic_row(method,s,1024,1 if method=='HRS' else 3,None if method=='HRS' else .2)
                    r['sample_id']=a;index=(i*5+j)%6;obj=[1,0.,1.,0.,0.]
                    if index==5 or (method=='HRS' and index==0) or (method!='HRS' and index==1):obj=None
                    elif method!='HRS' and index==2:obj[0]=0
                    elif method!='HRS' and index==3:obj[0]=2
                    r['outcome']['best_objective']=obj
                    r['outcome']['best_fingerprint']=None if obj is None else 'synthetic'
                    r['outcome']['h4_scoreable_count']=int(obj is not None)
                    r['outcome']['h4_unscoreable_count']=8-int(obj is not None)
                    rows.append(r)
        with patch('audit_step6_4g_phase_v1.read',return_value={'anchors':{'validation':anchors}}):result=analyze('A',rows)
        pair=result['paired.json']['S-CEM_vs_HRS']
        self.assertEqual(sum(pair['six_category_counts'].values()),320)
        self.assertTrue(all(v>0 for v in pair['six_category_counts'].values()))
        self.assertEqual(result['selected_method.json']['selected_method'],oracle_select({k:v['cluster_bootstrap'] for k,v in result['paired.json'].items()}))

if __name__=='__main__':unittest.main()
