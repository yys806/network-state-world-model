import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import build_project_knowledge_index_v1 as builder


class StageAResultRegistryTests(unittest.TestCase):
    def test_stage_a_numeric_evidence_and_identity_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            audit={'verdict':'STEP_6_3D_VALIDATION_STAGE_A=PASS','seed':list(range(6311,6316)),
                'primary_budget':1024,'case_count':960,'nominal_B_WM_total':983040,
                'actual_unique_transitions':983040,'selected_search_method':'MH-CEM',
                'locked_test':False,'stage_b':'NOT_STARTED','runtime':{'actual_elapsed_seconds':100.},
                'method_diagnostics':{},'paired_comparisons':{}}
            metrics={'case_count':960,'nominal_B_WM_total':983040,'actual_unique_transitions':983040,'elapsed_seconds':100.}
            for method in ('HRS','S-CEM','MH-CEM'):
                audit['method_diagnostics'][method]={'h4_scoreable_success_count':200,'h4_scoreable_success_rate':.625}
                metrics[method+'_scoreable_count']=200; metrics[method+'_scoreable_rate']=.625
            for pair in ('S-CEM_vs_HRS','MH-CEM_vs_HRS','MH-CEM_vs_S-CEM'):
                audit['paired_comparisons'][pair]={'win':200,'tie':100,'loss':20,
                    'cluster_bootstrap':{'mean_paired_advantage':.5625,'ci95_lower':.1,'ci95_upper':.7}}
                for suffix,value in (('win',200),('tie',100),('loss',20),('mean',.5625),('ci95_lower',.1),('ci95_upper',.7)):
                    metrics[pair+'_'+suffix]=value
            path=root/'acceptance.json'; path.write_text(json.dumps(audit))
            result={'id':'res','experiment_id':'exp','result_type':'planner_search_stage_a',
                'audit':'acceptance.json','audit_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'seed':audit['seed'],'status':'passed_stage_a','selected_search_method':'MH-CEM',
                'primary_budget':1024,'locked_test_accessed':False,'metrics':metrics}
            payload={'schema_version':'PI-JWM-results-registry-v2','results':[result]}
            experiments={'experiments':[{'id':'exp','seed':audit['seed']}]}
            self.assertEqual(builder.validate_result_registry_against_evidence(root,payload,experiments)['mismatches'],[])
            changed=copy.deepcopy(payload); changed['results'][0]['metrics']['MH-CEM_scoreable_count']=201
            self.assertTrue(builder.validate_result_registry_against_evidence(root,changed,experiments)['mismatches'])
            changed=copy.deepcopy(payload); changed['results'][0]['selected_search_method']='HRS'
            self.assertTrue(builder.validate_result_registry_against_evidence(root,changed,experiments)['mismatches'])
            changed=copy.deepcopy(payload); changed['results'][0]['locked_test_accessed']=True
            self.assertTrue(builder.validate_result_registry_against_evidence(root,changed,experiments)['mismatches'])
            changed=copy.deepcopy(payload); changed['results'][0]['metrics']['elapsed_seconds']=float('nan')
            self.assertTrue(builder.validate_result_registry_against_evidence(root,changed,experiments)['mismatches'])

if __name__=='__main__': unittest.main()
