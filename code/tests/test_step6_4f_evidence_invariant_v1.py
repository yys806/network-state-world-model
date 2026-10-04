"""Reject root-only reuse assertions when later Task completion is possible."""
import sys, unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
from assess_step6_4f_search_evidence_v1 import invariant, rule_oracle
from pi_jwm.step4_4_structured_rssm_world_model_v1 import StructuredRSSMWorldModel, StructuredRSSMConfig
from pi_jwm.step3_3_model_input_tensor_v1 import LIFECYCLE_VOCAB

class FutureDomainInvariantTests(unittest.TestCase):
    def state(self):
        _,s,_,_=StructuredRSSMWorldModel.synthetic_fixture(SimpleNamespace(config=StructuredRSSMConfig()))
        s['task_lifecycle_index'].fill_(LIFECYCLE_VOCAB.index('offloading'))
        s['task_return_requirement_known'].zero_()
        return s
    def test_no_active_flow_certificate(self):
        s=self.state();s['carrying_active'].zero_()
        self.assertEqual(invariant(s)['active_flow_count'],0)
    def test_unknown_return_requirement_certificate_and_rejections(self):
        s=self.state();self.assertIsNotNone(invariant(s))
        s['task_return_requirement_known'][0,0]=True
        self.assertIsNone(invariant(s))
        s=self.state();s['return_flow_index'][0,0]=0
        self.assertIsNone(invariant(s))
        s=self.state();s['task_lifecycle_index'][0,0]=LIFECYCLE_VOCAB.index('failed')
        self.assertIsNone(invariant(s))
    def test_rule_can_leave_completed_task_active_flow(self):
        self.assertTrue(rule_oracle()['completed_task_with_active_flow_possible'])

if __name__=='__main__':unittest.main()
