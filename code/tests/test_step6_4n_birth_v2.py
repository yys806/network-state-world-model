import unittest
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import torch
from pi_jwm.step6_4n_dynamic_v2 import append_validated_return,compact_anchor_support,FLOW_KEYS
from pi_jwm.step6_4n_rollout_v2 import restore_ledger,rollout_dynamic_v2,advance_stage_clock
from pi_jwm.step6_1_trained_candidate_rollout_v1 import fingerprint,clone_tree
from pi_jwm.step4_4_structured_rssm_world_model_v1 import StructuredRSSMWorldModel,StructuredRSSMConfig
from pi_jwm.step6_0a_candidate_generation_v1 import CandidateActionStep,CandidateActionSequence,Backend,PlannerCandidateContext

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'code/artifacts/audit/pi_jwm_step6_4n_20261012_wm_v4'

class DynamicBirthTests(unittest.TestCase):
    def setUp(self):
        self.a=torch.load(ART/'common_anchor.pt',weights_only=False)
        self.model=StructuredRSSMWorldModel(StructuredRSSMConfig()).eval()
        self.s,self.g,self.l=(self.a[k] for k in ('state','graph','latent'))
        self.kw=dict(task_index=0,source_index=4,destination_index=5,route=(5,),size=.2,flow_id='flow::Task_5::Return::0')

    def test_append_no_overwrite_identity_zero_hidden_and_no_mutation(self):
        fp=fingerprint((self.s,self.g,self.l))
        with torch.no_grad():r=append_validated_return(self.model,self.s,self.g,self.l,**self.kw)
        self.assertEqual(r.slot,1)
        for k in FLOW_KEYS:self.assertTrue(torch.equal(r.state[k][:,:1],self.s[k]),k)
        self.assertTrue(torch.equal(r.latent['h']['flow'][:,:1],self.l['h']['flow']))
        self.assertEqual(int(torch.count_nonzero(r.latent['h']['flow'][:,1])),0)
        self.assertEqual(fp,fingerprint((self.s,self.g,self.l)))
        with self.assertRaises(ValueError):append_validated_return(self.model,r.state,r.graph,r.latent,**self.kw)

    def test_invalid_indexes_capacity_multihop_no_relation(self):
        for kw in ({'source_index':-1},{'source_index':1000},{'task_index':1000},{'route':(3,5)},{'size':0.},{'max_flow_slots':1},{'flow_id':'flow::Task_5::Return::1'}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):append_validated_return(self.model,self.s,self.g,self.l,**{**self.kw,**kw})
        s=clone_tree(self.s);s['comm_validity'].zero_()
        with self.assertRaises(ValueError):append_validated_return(self.model,s,self.g,self.l,**self.kw)

    def test_duplicate_flow_id_and_invalid_ledger_slot(self):
        ss=clone_tree(self.a['snapshot']);ss['logical_flow_rows']*=2
        with self.assertRaises(ValueError):restore_ledger(ss)
        ss=clone_tree(self.a['snapshot']);ss['logical_flow_rows'][0]['flow_index']=3
        with self.assertRaises(ValueError):restore_ledger(ss)

    def test_clock_pre_and_post_service_boundary_no_reset(self):
        self.assertEqual(advance_stage_clock(clock=1.,tick_time=1.5,useful_progress=0.,unfinished=True,next_check_time=1.5,threshold=.5),(1.,None))
        self.assertEqual(advance_stage_clock(clock=1.,tick_time=1.5,useful_progress=0.,unfinished=True,next_check_time=1.6,threshold=.5),(1.,'TASK_FAIL_OUT_OF_TTI'))
        self.assertEqual(advance_stage_clock(clock=1.,tick_time=1.5,useful_progress=.01,unfinished=True,next_check_time=1.6,threshold=.5),(1.5,None))
        self.assertEqual(advance_stage_clock(clock=1.,tick_time=1.6,useful_progress=.01,unfinished=True,next_check_time=1.7,threshold=.5),(1.,'TASK_FAIL_OUT_OF_TTI'))

    def test_tti_terminal_is_returned_and_hard_rejected_without_future_truth(self):
        from pi_jwm.step6_4n_dynamic_v2 import DynamicScorerV2
        c=torch.load(ART/'B_ROUTE_RB_trace.pt',weights_only=False)['candidate']
        first=replace(c.steps[0],comm=())
        c=replace(c,steps=(first,)+c.steps[1:],candidate_id='NO_SERVICE_TTI')
        ctx=PlannerCandidateContext.from_sample(self.a['sample'],self.s,'test')
        anchor=SimpleNamespace(state=self.s,graph=self.g,latent=self.l,context=ctx,fingerprints=self.a['fingerprints'])
        with torch.no_grad():
            tr,meta=rollout_dynamic_v2(self.model,anchor,c,current_raw=self.a['current'],ledger_snapshot=self.a['snapshot'],
                catalog=SimpleNamespace(comm_start_width_pairs={(0,1)}),tti_threshold=.05)
        self.assertEqual(len(tr.states),1)
        self.assertEqual(meta['tti_failure']['reason'],'TASK_FAIL_OUT_OF_TTI')
        result=DynamicScorerV2(self.s,self.a['side'],self.a['current'],slot_duration_s=.1).score_set([(c,tr,meta)])
        self.assertEqual(result['status'],'OBJECTIVE_UNSCOREABLE_ALL_REJECTED')
        self.assertEqual(result['rejections']['NO_SERVICE_TTI']['terminal'],'TTI_FAILED')

    def test_unsupported_route_and_poison_stop_before_neural_transition(self):
        c=torch.load(ART/'B_ROUTE_RB_trace.pt',weights_only=False)['candidate']
        ctx=PlannerCandidateContext.from_sample(self.a['sample'],self.s,'test')
        a=SimpleNamespace(state=self.s,graph=self.g,latent=self.l,context=ctx,fingerprints=self.a['fingerprints'])
        kw=dict(current_raw=self.a['current'],ledger_snapshot=self.a['snapshot'],catalog=SimpleNamespace(comm_start_width_pairs={(0,1)}))
        with torch.no_grad():
            with self.assertRaisesRegex(ValueError,'FUTURE'):rollout_dynamic_v2(self.model,a,c,**{**kw,'current_raw':{**kw['current_raw'],'future_target':object()}})
            with self.assertRaisesRegex(ValueError,'DUPLICATE'):rollout_dynamic_v2(self.model,a,replace(c,steps=(replace(c.steps[0],route=c.steps[0].route*2),)),**kw)
            raw=clone_tree(kw['current_raw']);next(r for r in raw['tasks'] if r['task_id']=='Task_5')['required_returned_size']=0.
            with self.assertRaises(ValueError):rollout_dynamic_v2(self.model,a,replace(c,steps=(c.steps[0],)),**{**kw,'current_raw':raw})

if __name__=='__main__':unittest.main()
