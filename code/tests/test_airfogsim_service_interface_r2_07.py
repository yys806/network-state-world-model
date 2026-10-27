"""Targeted contract failures and unit checks; no dataset or GPU access."""
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))

from pi_jwm.v2u4real_interface_v1 import InterfaceBlocked, make_absolute_replay_manager
from pi_jwm.airfogsim_service_interface_v1 import observe_post_step, read_post_step_link_service, slot_service


class _Base:
    _traffic_interval=.1
    _x_range=(0,600)
    _y_range=(0,600)
    _UAV_z_range=(45,60)
    def _update_map_by_grid(self): pass
    def reset(self, _=None): pass


class ServiceTests(unittest.TestCase):
    def test_mask_and_decimal_units(self):
        r=slot_service([2.,4.,8.],np.array([False,True,False]),.1)
        self.assertEqual(r['theoretical_bytes'],50000.)
        self.assertEqual(r['inactive_positive_count'],2)
        self.assertEqual(r['delivery_status'],'NOT_RUN')
        self.assertEqual(slot_service([2.],np.array([False]),.1)['theoretical_bytes'],0)
        self.assertEqual(slot_service([0.],np.array([True]),.1)['theoretical_bytes'],0)
        self.assertAlmostEqual(slot_service([1e-6],np.array([True]),.5)['theoretical_bytes'],.0625)

    def test_invalid_rate_mask_duration_and_overflow(self):
        for rate,mask,dt in (([1],[True,False],.1),([1],[1],.1),([np.nan],[True],.1),
                             ([-1],[True],.1),([np.inf],[True],.1),([1],[True],0),
                             ([1],[True],-1),([1],[True],float('inf')),([1],[True],True),
                             ([1e308],[True],1e308)):
            with self.subTest(rate=rate,dt=dt), self.assertRaises(InterfaceBlocked):
                slot_service(rate,np.asarray(mask),dt)

    def make_env(self):
        entity=SimpleNamespace(getPosition=lambda:[1,2,50],_last_updated_time=0.)
        tm=SimpleNamespace(getCurrentTime=lambda:.1,getVehicleTrafficInfos=lambda:{},
                           getUAVTrafficInfos=lambda:{'UAV_0':{'position':[1,2,50]}},
                           getMapIndexByNodeId=lambda _:(0,0))
        cm=SimpleNamespace(n_RB=3,U2I_Rate=np.array([[[2.,8.,4.]]]),
                           U2I_active_links=np.array([[[False,True,False]]]))
        return SimpleNamespace(traffic_manager=tm,traffic_interval=.1,simulation_interval=.1,simulation_time=.1,
                               vehicles={},UAVs={'UAV_0':entity},vehicle_ids_as_index=[],
                               uav_ids_as_index=['UAV_0'],rsu_ids_as_index=['RSU_0'],channel_manager=cm)

    def test_observation_preserves_native_lag(self):
        env=self.make_env(); r=observe_post_step(env)
        self.assertEqual(r['observation_time_s'],.1)
        self.assertEqual(env.UAVs['UAV_0']._last_updated_time,0.)
        self.assertFalse(r['native_timestamp_repaired'])
        env.simulation_time=0.
        with self.assertRaises(InterfaceBlocked):observe_post_step(env)

    def test_observation_rejects_different_contract(self):
        for change in ('interval','position','native','membership','grid'):
            env=self.make_env()
            if change=='interval':env.simulation_interval=.05
            if change=='position':env.UAVs['UAV_0'].getPosition=lambda:[2,2,50]
            if change=='native':env.UAVs['UAV_0']._last_updated_time=.1
            if change=='membership':env.UAVs={}
            if change=='grid':env.traffic_manager.getMapIndexByNodeId=lambda _:(None,None)
            with self.subTest(change=change), self.assertRaises(InterfaceBlocked):observe_post_step(env)

    def test_link_index_mask_and_shape_guards(self):
        env=self.make_env()
        r=read_post_step_link_service(env,'U2I','UAV_0','RSU_0',[1])
        self.assertEqual(r['theoretical_bytes'],100000.)
        for rbs in ([],[0],[1,1],[-1],[3],[True]):
            with self.subTest(rbs=rbs), self.assertRaises(InterfaceBlocked):
                read_post_step_link_service(env,'U2I','UAV_0','RSU_0',rbs)
        with self.assertRaises(InterfaceBlocked):read_post_step_link_service(env,'U2I','missing','RSU_0',[1])
        with self.assertRaises(InterfaceBlocked):read_post_step_link_service(env,'V2V','UAV_0','RSU_0',[1])
        env.channel_manager.U2I_Rate=np.zeros((2,1,3))
        with self.assertRaises(InterfaceBlocked):read_post_step_link_service(env,'U2I','UAV_0','RSU_0',[1])


class ReplayTests(unittest.TestCase):
    def rows(self):
        a=dict(time_s=0.,vehicles={'vehicle_0':{'position':[1,2,0]}},uavs={'UAV_0':{'position':[1,2,50]}})
        b=deepcopy(a);b['time_s']=.1;b['vehicles']={'vehicle_1':{'id':'vehicle_1','position':[3,2,0]}}
        c=deepcopy(b);c['time_s']=.2;c['vehicles']={}
        return [a,b,c]

    def test_default_still_refuses_dynamic_ids(self):
        with self.assertRaises(InterfaceBlocked):make_absolute_replay_manager(_Base)().bind_snapshots(self.rows())

    def test_dynamic_copy_absolute_reset_exhaustion_control(self):
        rows=self.rows(); tm=make_absolute_replay_manager(_Base,dynamic_vehicle_ids=True)()
        tm.bind_snapshots(rows);rows[1]['vehicles'].clear();tm.stepSimulation()
        self.assertEqual(set(tm._vehicle_infos),{'vehicle_1'})
        self.assertEqual(tm._vehicle_infos['vehicle_1']['position'],[3,2,0])
        tm.stepSimulation();self.assertEqual(tm._vehicle_infos,{})
        with self.assertRaises(InterfaceBlocked):tm.stepSimulation()
        with self.assertRaises(InterfaceBlocked):tm.updateUAVMobilityPatterns({'UAV_0':{'position':[2,2,50]}})
        tm.reset();self.assertEqual(tm._current_time,0)

    def test_dynamic_invalid_time_position_identity(self):
        for change in ('time','position','identity'):
            rows=self.rows()
            if change=='time':rows[1]['time_s']=.12
            if change=='position':rows[1]['vehicles']['vehicle_1']['position'][0]=600
            if change=='identity':rows[1]['vehicles']['vehicle_1']['id']='vehicle_2'
            with self.subTest(change=change), self.assertRaises(InterfaceBlocked):
                make_absolute_replay_manager(_Base,dynamic_vehicle_ids=True)().bind_snapshots(rows)


if __name__=='__main__':unittest.main()
