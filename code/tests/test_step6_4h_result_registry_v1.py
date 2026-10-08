import copy
import importlib.util
import json
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('index_builder',ROOT/'code/scripts/build_project_knowledge_index_v1.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)

class BudgetRegistryTests(unittest.TestCase):
    def test_accepted_raw_derived_numbers_and_drift(self):
        directory=ROOT/'docs/registries'
        experiments=json.loads((directory/'experiment_registry.json').read_text(encoding='utf-8'))
        full=json.loads((directory/'results_registry.json').read_text(encoding='utf-8'))
        row=next(r for r in full['results'] if r['id']=='RES-STEP-6.4H-S-CEM-BUDGET-20261008')
        payload={'schema_version':full['schema_version'],'results':[row]}
        self.assertEqual(builder.validate_result_registry_against_evidence(ROOT,payload,experiments)['mismatches'],[])
        for mutate in (lambda r:r['metrics'].__setitem__('primary_degradation',1),
                       lambda r:r.__setitem__('selected_search_method','MH-CEM'),
                       lambda r:r.__setitem__('final_budget',512),
                       lambda r:r.__setitem__('qualification_sha256','wrong')):
            changed=copy.deepcopy(payload);mutate(changed['results'][0])
            self.assertTrue(builder.validate_result_registry_against_evidence(ROOT,changed,experiments)['mismatches'])

if __name__=='__main__':unittest.main()
