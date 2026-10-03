"""CPU-only checks of percentile convention and readiness stop guards."""
import json
from pathlib import Path
import statistics
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/scripts'))
from audit_step6_4a_closed_loop_readiness_v1 import quantile

class ReadinessAuditTests(unittest.TestCase):
    def test_inclusive_percentiles_agree_with_standard_library(self):
        values=[2,4,5,11,13,35,77]
        cuts=statistics.quantiles(values,n=100,method='inclusive')
        self.assertEqual(quantile(values,.5),statistics.median(values))
        self.assertAlmostEqual(quantile(values,.9),cuts[89])
        self.assertAlmostEqual(quantile(values,.95),cuts[94])
        for bad in ([],[float('nan')],[float('inf')]):
            with self.assertRaises(ValueError):quantile(bad,.5)

    def test_audit_keeps_readiness_separate_from_search_selection(self):
        path=ROOT/'code/artifacts/protocols/pi_jwm_step6_4a_readiness_v1_20261003'
        a=json.loads((path/'06_go_no_go_receipt.json').read_text(encoding='utf-8'))
        p=json.loads((path/'04_stage_b_purpose_cost_comparison.json').read_text(encoding='utf-8'))
        self.assertEqual(a['closed_loop_readiness'],'BLOCKED')
        self.assertEqual(a['search_method'],'MH-CEM')
        self.assertEqual(a['stage_b'],'NOT_STARTED')
        self.assertEqual(a['gpu'],'NOT_USED')
        self.assertFalse(a['locked_test'])
        self.assertFalse(p['method_selection_changes'])
        self.assertEqual([x['cases'] for x in p['plans']],[1920,640,96])
        self.assertEqual([x['nominal_transitions'] for x in p['plans']],[737280,245760,36864])
        self.assertTrue(all('NEW_RESEARCH_PROPOSAL' in x['status'] for x in p['plans'][1:]))

if __name__=='__main__':unittest.main()
