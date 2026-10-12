import os
os.environ.setdefault('PYTHONUTF8', '1')
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/reference/AirFogSim'))
from airfogsim.entities.task import Task
from airfogsim.manager.task_manager import TaskManager
from pi_jwm.step6_4n_runtime_v2 import stage_aware_return_tti, validate_joint_return, commit_joint_return


def fixture():
    manager = TaskManager({})
    t = Task('T', 'UAV_1', 1., 1., 9., 1., .6,
             required_returned_size=.2, to_return_node_id='UAV_1')
    t._computed_size = 1.; t._assigned_to = 'UAV_0'
    t._routes = ['UAV_1', 'UAV_0']; t._last_transmission_time = 1.2
    t._start_to_transmit_time=1.0
    manager._waiting_to_return_tasks = {'UAV_0': [t]}
    manager.checkTaskDependency = lambda *args: True
    return manager, t


class RuntimeV2Tests(unittest.TestCase):
    def test_legacy_fails_but_return_clock_does_not_inherit_input(self):
        m,t=fixture(); m.setTaskReturnRouteAndStartReturn('T',['UAV_1'],1.8); m.checkTasks(1.8)
        self.assertEqual(t._failure_reason_code, 1)
        m,t=fixture(); original=m.checkTasks.__func__
        with stage_aware_return_tti(m) as evidence:
            m.setTaskReturnRouteAndStartReturn('T',['UAV_1'],1.8); m.checkTasks(1.8)
            self.assertEqual(t._failure_reason_code,-1)
            self.assertEqual(t.getLastTransmissionTime(),1.2)
            self.assertEqual(evidence['return_clocks']['T'],1.8)
            m.checkTasks(2.31); self.assertEqual(t._failure_reason_code,1)
        self.assertEqual(m.checkTasks.__func__,original)

    def test_zero_activity_does_not_refresh_return_clock(self):
        m,t=fixture()
        with stage_aware_return_tti(m) as evidence:
            m.setTaskReturnRouteAndStartReturn('T',['UAV_1'],1.8)
            t.transmit_to_Node('UAV_1',0.,2.0)
            self.assertEqual(evidence['return_clocks']['T'],1.8)
            t.transmit_to_Node('UAV_1',.01,2.1)
            self.assertEqual(evidence['return_clocks']['T'],2.1)
            with self.assertRaises(ValueError):
                m.setTaskReturnRouteAndStartReturn('T',['UAV_1'],2.2)

    def test_deadline_is_independent(self):
        m,t=fixture(); t._task_deadline=.1
        with stage_aware_return_tti(m):
            m.setTaskReturnRouteAndStartReturn('T',['UAV_1'],1.8); m.checkTasks(1.8)
        self.assertEqual(t._failure_reason_code,0)

    def test_prevalidate_no_mutation_partial_failure_no_step(self):
        from types import SimpleNamespace
        m,t=fixture()
        env=SimpleNamespace(task_manager=m, channel_manager=SimpleNamespace(n_RB=50),
                            task_return_routes={}, activated_offloading_tasks_with_RB_Nos={})
        env._getNodeTypeById=lambda n:'U' if n in {'UAV_0','UAV_1'} else None
        env._getNodeIdxById=lambda n:0
        row={'task_id':'T','lifecycle':'waiting_to_return','task_cpu_work':1.,'computed_cpu_work':1.,
             'current_node_id':'UAV_0','return_destination_id':'UAV_1','required_returned_size':.2}
        with self.assertRaises(ValueError):
            validate_joint_return(env,row,['UAV_1'],[50],{('UAV_0','UAV_1')})
        self.assertEqual(env.task_return_routes,{})
        self.assertEqual(validate_joint_return(env,row,['UAV_1'],[0],{('UAV_0','UAV_1')})['channel_type'],'U2U')
        def route_setter(e,tid,route): e.task_return_routes[tid]=route
        def rb_fail(*args): raise RuntimeError('partial')
        env._pi_jwm_v2_wireless_relations={('UAV_0','UAV_1')}
        env.step=lambda:None
        with self.assertRaisesRegex(RuntimeError,'PARTIAL_SETTER_FAILURE'):
            commit_joint_return(env,'T',['UAV_1'],[0],route_setter,rb_fail)
        self.assertEqual(env.task_return_routes,{'T':['UAV_1']})
        self.assertTrue(env._pi_jwm_v2_quarantined)
        with self.assertRaisesRegex(RuntimeError,'QUARANTINED'):env.step()

    def test_fp64_boundary_and_no_return_are_rejected(self):
        from types import SimpleNamespace
        m,t=fixture(); env=SimpleNamespace(task_manager=m,channel_manager=SimpleNamespace(n_RB=50))
        row={'task_id':'T','lifecycle':'computing','task_cpu_work':1.,'computed_cpu_work':1.-1e-16,
             'current_node_id':'UAV_0','return_destination_id':'UAV_1','required_returned_size':.2}
        with self.assertRaises(ValueError): validate_joint_return(env,row,['UAV_1'],[],{('UAV_0','UAV_1')})

    def test_fractional_rb_unknown_node_duplicate_pending_and_quarantine(self):
        from types import SimpleNamespace
        m,t=fixture()
        env=SimpleNamespace(task_manager=m,channel_manager=SimpleNamespace(n_RB=50),task_return_routes={},activated_offloading_tasks_with_RB_Nos={})
        env._getNodeTypeById=lambda n:'U' if n in {'UAV_0','UAV_1'} else None
        env._getNodeIdxById=lambda n:0 if n=='UAV_0' else 1 if n=='UAV_1' else -1
        env._pi_jwm_v2_wireless_relations={('UAV_0','UAV_1')}
        with self.assertRaises(ValueError):validate_joint_return(env,t,['UAV_1'],[.5],env._pi_jwm_v2_wireless_relations)
        env.task_return_routes['T']=['UAV_1']
        with self.assertRaises(ValueError):commit_joint_return(env,'T',['UAV_1'],[0],lambda *x:None,lambda *x:None)
        env.task_return_routes.clear();env._pi_jwm_v2_quarantined=True
        with self.assertRaises(ValueError):commit_joint_return(env,'T',['UAV_1'],[0],lambda *x:None,lambda *x:None)

    def test_input_zero_activity_clock_native_runtime_and_restore(self):
        from pi_jwm.step6_4n_joint_bridge_v2 import stage_aware_input_return_tti_v2
        m,t=fixture();m._waiting_to_return_tasks={};m._offloading_tasks={'UAV_1':[t]}
        t._computed_size=0.;t._to_offload_route=['UAV_0'];t._executed_locally=False
        original=m.checkTasks.__func__
        with stage_aware_input_return_tti_v2(m) as evidence:
            t.transmit_to_Node('UAV_0',0.,1.3)
            self.assertEqual(t.getLastTransmissionTime(),1.2)
            t.transmit_to_Node('UAV_0',.01,1.4)
            self.assertEqual(evidence['input_clocks']['T'],1.4)
            m.checkTasks(1.91);self.assertEqual(t._failure_reason_code,1)
        self.assertEqual(m.checkTasks.__func__,original)

    def test_joint_transaction_commits_all_families_after_validation(self):
        from types import SimpleNamespace
        from pi_jwm.step6_4n_joint_bridge_v2 import apply_joint_commands_v2
        calls=[]
        env=SimpleNamespace(task_return_routes={},activated_offloading_tasks_with_RB_Nos={},step=lambda: calls.append('step'))
        class Route:
            def setTaskReturnRoute(self,e,t,p): e.task_return_routes[t]=list(p); calls.append('route')
        class Comm:
            def setCommunicationWithRB(self,e,t,ids): e.activated_offloading_tasks_with_RB_Nos[t]=list(ids); calls.append('comm')
        class Comp:
            def setComputingCallBack(self,e,cb): e.alloc_cpu_callback=cb; calls.append('comp')
        class Traffic:
            def setUAVMobilityPatterns(self,e,p): e.uav_mobility_patterns=p; calls.append('mob')
        out=apply_joint_commands_v2(env,{'mode':'DYNAMIC_RETURN_V2','joint_validation_before_any_setter':True,
            'routes':{'T':['UAV_1']},'rb':{'T':[0]},'cpu':{},'mobility':{}},
            route_scheduler=Route(),communication=Comm(),computation=Comp(),traffic=Traffic())
        self.assertTrue(out['all_setters_committed']);self.assertEqual(calls,['route','comm','comp','mob'])

    def test_joint_transaction_quarantines_after_partial_setter_and_blocks_step(self):
        from types import SimpleNamespace
        from pi_jwm.step6_4n_joint_bridge_v2 import apply_joint_commands_v2
        env=SimpleNamespace(task_return_routes={},activated_offloading_tasks_with_RB_Nos={},step=lambda: None)
        class Route:
            def setTaskReturnRoute(self,e,t,p): e.task_return_routes[t]=list(p)
        class Comm:
            def setCommunicationWithRB(self,e,t,ids): raise RuntimeError('setter failed')
        class Comp:
            def setComputingCallBack(self,e,cb): pass
        class Traffic:
            def setUAVMobilityPatterns(self,e,p): pass
        with self.assertRaisesRegex(RuntimeError,'JOINT_PARTIAL_SETTER_FAILURE'):
            apply_joint_commands_v2(env,{'mode':'DYNAMIC_RETURN_V2','joint_validation_before_any_setter':True,
                'routes':{'T':['UAV_1']},'rb':{'T':[0]},'cpu':{},'mobility':{}},
                route_scheduler=Route(),communication=Comm(),computation=Comp(),traffic=Traffic())
        self.assertTrue(env._pi_jwm_v2_quarantined)
        with self.assertRaisesRegex(RuntimeError,'QUARANTINED'):
            env.step()


if __name__=='__main__': unittest.main()
