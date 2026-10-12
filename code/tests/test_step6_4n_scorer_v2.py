import unittest
from dataclasses import replace
from pathlib import Path
import torch
from pi_jwm.step6_4n_dynamic_v2 import DynamicScorerV2,FLOW_KEYS
from pi_jwm.step6_1_trained_candidate_rollout_v1 import clone_tree,fingerprint,CandidateRolloutTrace
from pi_jwm.step6_0a_candidate_generation_v1 import CandidateActionStep,CandidateActionSequence,Backend
from pi_jwm.step6_2a_planner_objective_side_state_v1 import PlannerObjectiveCausalSideState,PlannerTaskCausalSideState
from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import FLOW_TYPE_VOCAB,FLOW_STATUS_VOCAB

ROOT=Path(__file__).resolve().parents[2]

class DynamicScorerTests(unittest.TestCase):
    def setUp(self):
        # Real saved causal state; deterministic rule alterations remain tests,
        # never simulator or prediction-performance evidence.
        self.anchor=torch.load(ROOT/'code/artifacts/audit/pi_jwm_step6_4n_20261012_wm_v4/common_anchor.pt',weights_only=False)['state']
        for k in FLOW_KEYS:
            x=self.anchor[k];self.anchor[k]=torch.cat((x,torch.zeros_like(x)),dim=1)
        self.ti=int(self.anchor['flow_task_index'][0,0])
        self.raw={'tasks':[{'task_id':'Task_5','lifecycle':'waiting_to_return','task_cpu_work':1.,'computed_cpu_work':1.,'required_returned_size':.2,'current_node_id':'UAV_0','return_destination_id':'UAV_1'}]}
        self.task=PlannerTaskCausalSideState('Task_5',self.ti,0.,1.,10.,10.,.2,5,True,False,1.)
        self.side=PlannerObjectiveCausalSideState(1.,1.,(self.task,),(),(True,False,False),(50.,None,None),'test causal current')
        self.scorer=DynamicScorerV2(self.anchor,self.side,self.raw,slot_duration_s=.1)

    def pair(self,identifier='A',remaining=None,tti=None):
        s=clone_tree(self.anchor)
        if remaining is not None:
            slot=1
            s['return_flow_index'][0,self.ti]=slot
            vals={'flow_known':True,'flow_presence':remaining>0,'flow_task_index':self.ti,'flow_type_index':FLOW_TYPE_VOCAB.index('Return'),'flow_total':.2,'flow_remaining':remaining,'hop_remaining':remaining,'carrying_active':remaining>0,'flow_status_index':FLOW_STATUS_VOCAB.index('ACTIVE' if remaining>0 else 'COMPLETED'),'flow_destination_index':5,'current_hop_index':0,'flow_comm_relation_index':38}
            for k,v in vals.items():s[k][0,slot]=v
            s['route_node_mask'][0,slot].fill_(False);s['route_node_mask'][0,slot,0]=True;s['route_node_indices'][0,slot,0]=5
        c=CandidateActionSequence((CandidateActionStep(),),'test_'+identifier,Backend.RULE_FALLBACK,None,'same')
        if identifier=='B':c=replace(c,steps=(CandidateActionStep(comp=({'task_id':'Task_5','node_id':'UAV_0','allocated_cpu_per_s':0.},)),))
        if remaining is not None and remaining<.2:c=replace(c,steps=(CandidateActionStep(comm=({'task_id':'Task_5','task_index':self.ti,'relation_index':38,'rb_indices':[0]},)),))
        tr=CandidateRolloutTrace(c.candidate_id,{'anchor':'same'},(),(),(),(),(),(s,),(),())
        return c,tr,{'source':'V2_CAUSAL_PREDICTED_TRANSITION','pre_route_state_fingerprint':fingerprint(self.anchor),'tti_failure':tti,'events':[{'task_id':'Task_5','horizon':1,'useful_progress':.2-remaining,'relation_index_before_service':38}] if remaining is not None else []}

    def test_noop_demand_and_birth_no_reward(self):
        a=self.scorer.score_set((self.pair(),))
        b=self.scorer.score_set((self.pair(remaining=.2),))
        self.assertEqual(a['scores'][0]['objective'],b['scores'][0]['objective'])
        self.assertEqual(a['scores'][0]['rows'][0]['tasks'][0]['unmet_return'],.2)

    def test_useful_progress_reduces_burden(self):
        a=self.scorer.score_set((self.pair(),))
        b=self.scorer.score_set((self.pair(remaining=.1),))
        self.assertLess(b['scores'][0]['objective'][3],a['scores'][0]['objective'][3])

    def test_remaining_drop_without_service_rejected(self):
        c,tr,m=self.pair(remaining=.1);c=replace(c,steps=(CandidateActionStep(),))
        with self.assertRaisesRegex(ValueError,'WITHOUT_COMM_SERVICE'):self.scorer.score_set(((c,tr,m),))

    def test_tti_is_hard_rejection_not_deadline(self):
        result=self.scorer.score_set((self.pair(tti={'reason':'TASK_FAIL_OUT_OF_TTI'}),))
        self.assertEqual(result['status'],'OBJECTIVE_UNSCOREABLE_ALL_REJECTED')
        self.assertEqual(result['rejections']['test_A']['terminal'],'TTI_FAILED')
        self.assertEqual(result['scores'],[])

    def test_deadline_is_separate_and_unique_task_count(self):
        scorer=DynamicScorerV2(self.anchor,replace(self.side,tasks=(replace(self.task,absolute_deadline_s=.9),)),self.raw,slot_duration_s=.1)
        pair=self.pair();c,tr,m=pair
        c=replace(c,steps=(CandidateActionStep(),)*4);tr=replace(tr,states=tr.states*4)
        result=scorer.score_set(((c,tr,m),))
        self.assertEqual(result['scores'][0]['objective'][:3],(1,1.,1.))

    def test_same_cohort_denominator_horizon(self):
        result=self.scorer.score_set((self.pair('A'),self.pair('B',remaining=.2)))
        self.assertEqual(result['H_eff'],1)
        self.assertEqual(result['cohort'],['Task_5'])
        self.assertEqual(result['effort_denominators'],(50.,None,None))

    def test_future_poison_rejected(self):
        with self.assertRaisesRegex(ValueError,'FUTURE'):DynamicScorerV2(self.anchor,self.side,{**self.raw,'future_target':object()},slot_duration_s=.1)

    def test_false_completion_invalid_index_missing_return_rejected(self):
        c,tr,m=self.pair(remaining=.2);s=tr.states[0];s['task_completed'][0,self.ti]=True
        with self.assertRaisesRegex(ValueError,'UNMET'):self.scorer.score_set(((c,tr,m),))
        c,tr,m=self.pair();tr.states[0]['return_flow_index'][0,self.ti]=10000
        with self.assertRaises(ValueError):self.scorer.score_set(((c,tr,m),))

    def test_empty_all_unscoreable_and_anchor_mismatch(self):
        self.assertEqual(self.scorer.score_set(())['H_eff'],0)
        c,tr,m=self.pair();m['pre_route_state_fingerprint']='wrong'
        with self.assertRaisesRegex(ValueError,'COMMON_ANCHOR'):self.scorer.score_set(((c,tr,m),))

    def test_fp32_does_not_establish_fp64_return_demand(self):
        raw=clone_tree(self.raw);raw['tasks'][0].update(lifecycle='computing',computed_cpu_work=1.-1e-16)
        self.assertEqual(float(torch.tensor(1.-1e-16,dtype=torch.float32)),1.)
        scorer=DynamicScorerV2(self.anchor,self.side,raw,slot_duration_s=.1)
        self.assertEqual(scorer.known,{})

if __name__=='__main__':unittest.main()
