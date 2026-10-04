"""Researcher-defined current communication eligibility; CPU only."""
import copy,sys,unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/tests')]
from test_step6_3b_candidate_grammar_v1 import fixture
from pi_jwm.step3_3_model_input_tensor_v1 import LIFECYCLE_VOCAB
from pi_jwm.step6_3b_candidate_grammar_v1 import _wireless_bindings,bind_structured_step,StructuredStepChoice,CommBlockChoice,CandidateGrammarViolation
from pi_jwm.step6_3c_candidate_domain_v1 import CandidateDomain
from pi_jwm.step6_4b_live_bridge_v1 import validate_command,SmokeFailure

class CommEligibilityTests(unittest.TestCase):
    def inputs(self,lifecycle='offloading'):
        context,operational,state,control,catalog=fixture()
        state['task_lifecycle_index'][0,0]=LIFECYCLE_VOCAB.index(lifecycle)
        decision={'n_rb':50,'entities':[{'entity_id':k,'entity_type':'uav' if k.startswith('UAV') else 'rsu'} for k in context.static['input_entity_index']['physical']],
          'tasks':[{'task_id':'input','lifecycle':lifecycle},{'task_id':'compute','lifecycle':'computing'}],
          'node_cpu_capacity_observation_rows':[{'node_id':'RSU_0','observed_mask':True,'capacity_per_s':10}]}
        runtime={'input':SimpleNamespace(),'compute':SimpleNamespace(getAssignedTo=lambda:'RSU_0')}
        return context,operational,state,control,catalog,decision,runtime
    def test_offloading_and_transmitting_eligible_and_bridge_accept(self):
        for lifecycle in ('offloading','transmitting'):
            context,op,state,control,cat,decision,runtime=self.inputs(lifecycle)
            self.assertEqual(_wireless_bindings(state,context.static['input_entity_index']['task']),{'input':0})
            bound=bind_structured_step(StructuredStepChoice((CommBlockChoice('input',0,1),),'SCALE_0.5','PROFILE_HOLD'),context,op,state,control,cat)
            commands=validate_command(bound.action,context,decision,runtime)
            self.assertEqual(commands['rb'],{'input':[0]});self.assertEqual(bound.action.route,())
    def test_failed_completed_or_explicit_completed_are_not_eligible(self):
        for lifecycle,completed in [('failed',False),('completed',False),('offloading',True),('transmitting',True)]:
            context,op,state,control,cat,_,_=self.inputs(lifecycle)
            state['task_completed'][0,0]=completed
            self.assertEqual(_wireless_bindings(state,context.static['input_entity_index']['task']),{})
            domain=CandidateDomain.from_state(context,op,state,control,cat)
            self.assertNotIn('input',domain.wireless_task_to_relation)
            with self.assertRaises(CandidateGrammarViolation):
                bind_structured_step(StructuredStepChoice((CommBlockChoice('input',0,1),),'SCALE_0.5','PROFILE_HOLD'),context,op,state,control,cat)
    def test_missing_lifecycle_fails_closed(self):
        context,_,state,*_=self.inputs();state.pop('task_lifecycle_index')
        self.assertEqual(_wireless_bindings(state,context.static['input_entity_index']['task']),{})
    def test_absent_invalid_or_non_unique_binding_not_eligible(self):
        for field in ('task_presence','flow_presence','flow_known','carrying_active','comm_presence','comm_validity','comm_wireless_mask'):
            context,_,state,*_=self.inputs();state[field][0,0]=False
            self.assertEqual(_wireless_bindings(state,context.static['input_entity_index']['task']),{})
        context,_,state,*_=self.inputs();state['carrying_hop_destination_index'][0,0]=3
        self.assertEqual(_wireless_bindings(state,context.static['input_entity_index']['task']),{})
    def test_bridge_rejects_stale_raw_and_completed_and_invalid_flow(self):
        context,op,state,control,cat,decision,runtime=self.inputs()
        action=bind_structured_step(StructuredStepChoice((CommBlockChoice('input',0,1),),'SCALE_0.5','PROFILE_HOLD'),context,op,state,control,cat).action
        for field,value in [('lifecycle','failed'),('task_completed',True)]:
            bad=copy.deepcopy(decision);bad['tasks'][0][field]=value
            with self.assertRaises(SmokeFailure):validate_command(action,context,bad,runtime)
        state['carrying_active'][0,0]=False
        with self.assertRaises(SmokeFailure):validate_command(action,context,decision,runtime)
    def test_comp_mob_route_and_future_poison_unchanged(self):
        context,op,state,control,cat,decision,runtime=self.inputs()
        choice=StructuredStepChoice((),'SCALE_0.5','PROFILE_HOLD')
        before=bind_structured_step(choice,context,op,state,control,cat).action
        poison=copy.deepcopy(state);poison.update(future_target=object(),future_schedule=object(),predicted_target=object())
        self.assertEqual(_wireless_bindings(poison,context.static['input_entity_index']['task']),{'input':0})
        state['task_lifecycle_index'][0,0]=LIFECYCLE_VOCAB.index('failed')
        after=bind_structured_step(choice,context,op,state,control,cat).action
        self.assertEqual(before.comp,after.comp);self.assertEqual(before.mob,after.mob);self.assertEqual(after.route,())
        decision.update(target=object(),future_schedule=object())
        self.assertEqual(validate_command(before,context,decision,runtime)['rb'],{})
if __name__=='__main__':unittest.main()
