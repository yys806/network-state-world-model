"""Contract and counterexample tests. All generated data here are synthetic."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/src'))
sys.path.insert(0, str(ROOT/'code/scripts'))
from pi_jwm.v2u4real_interface_v1 import (
    InterfaceBlocked, UniqueKeyLoader, audit_dataset, common_to_rsu,
    make_absolute_replay_manager, read_pcd_xyz, rigid_matrix, transform_points,
)
import yaml


def write_ascii(path, mode='ascii'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA '+mode+'\n1 2 3 5\n4 5 6 7\n', encoding='ascii')


class GeometryTests(unittest.TestCase):
    def test_known_axis_and_inverse(self):
        t = np.array([[0,-1,0,10],[1,0,0,20],[0,0,1,30],[0,0,0,1]], float)
        p = np.array([[1,2,3]])
        q = transform_points(p,t)
        np.testing.assert_allclose(q, [[8,21,33]])
        np.testing.assert_allclose(transform_points(q,np.linalg.inv(t)),p)

    def test_rsu_is_explicit_setting(self):
        t=np.eye(4);t[:3,3]=[10,20,30]
        np.testing.assert_allclose(common_to_rsu([[12,22,32]],t),[[2,2,2]])

    def test_reflection_rejected(self):
        t=np.eye(4);t[0,0]=-1
        with self.assertRaises(InterfaceBlocked): rigid_matrix(t)

    def test_nonrigid_and_nonfinite_rejected(self):
        for value in (np.zeros((4,4)), np.eye(3), np.full((4,4),np.nan)):
            with self.assertRaises(InterfaceBlocked): rigid_matrix(value)


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def frame(self, scene, agent, key, extra=''):
        p=self.root/'train'/scene/agent
        y=p/'yaml/ouster'/(key+'.yaml');y.parent.mkdir(parents=True,exist_ok=True)
        y.write_text('lidar_pose: [1, 2, 3, 0, 0, 0]\nvehicles: {}\n'+extra,encoding='utf8')
        write_ascii(p/'ouster'/(key+'.pcd'))

    def test_missing_data_is_blocked(self):
        a=audit_dataset(self.root/'missing')
        self.assertEqual(a['status'],'BLOCKED');self.assertIsNone(a['actual_interval_s'])

    def test_reader_ascii_and_binary(self):
        p=self.root/'a.pcd';write_ascii(p)
        xyz,h=read_pcd_xyz(p);np.testing.assert_equal(xyz,[[1,2,3],[4,5,6]])
        raw=p.read_bytes().split(b'DATA ascii')[0]+b'DATA binary\n'
        p.write_bytes(raw+np.array([[1,2,3,5],[4,5,6,7]],dtype='<f4').tobytes())
        np.testing.assert_equal(read_pcd_xyz(p)[0],xyz)

    def test_empty_corrupt_compressed_nonfinite_rejected(self):
        p=self.root/'x.pcd';write_ascii(p,'binary_compressed')
        with self.assertRaises(InterfaceBlocked):read_pcd_xyz(p)
        write_ascii(p);p.write_text(p.read_text().replace('1 2 3 5','nan 2 3 5'))
        with self.assertRaises(InterfaceBlocked):read_pcd_xyz(p)
        write_ascii(p);p.write_text(p.read_text().replace('POINTS 2','POINTS 3'))
        with self.assertRaises(InterfaceBlocked):read_pcd_xyz(p)

    def test_duplicate_yaml_keys_rejected(self):
        with self.assertRaises(InterfaceBlocked):yaml.load('time: 1\ntime: 2\n',Loader=UniqueKeyLoader)

    def test_audit_missing_keys_duplicates_and_no_clock_inference(self):
        self.frame('small','1','000001','timestamp: 7\n')
        self.frame('small','1','000003','timestamp: 7\n')
        self.frame('small','2','000001','timestamp: 7\n')
        a=audit_dataset(self.root)
        self.assertEqual(a['frame_counts'],{'1':2,'2':1})
        self.assertEqual(a['missing_keys']['2'],['000003'])
        self.assertEqual(a['duplicate_raw_time_values']['1.timestamp'],['7'])
        self.assertEqual(a['frame_number_gaps']['1'],[[1,3]])
        self.assertEqual(a['status'],'BLOCKED');self.assertIsNone(a['actual_interval_s'])

    def test_select_smallest_train_and_cap(self):
        for scene, count in [('large',4),('small',2)]:
            for agent in ('1','2'):
                for i in range(count):self.frame(scene,agent,str(i))
        a=audit_dataset(self.root,max_frames=1)
        self.assertEqual(a['scene'],'small');self.assertTrue(a['truncated']);self.assertEqual(len(a['frames']),2)

    def test_test_split_not_accepted(self):
        (self.root/'test').mkdir()
        a=audit_dataset(self.root)
        self.assertEqual(a['status'],'BLOCKED');self.assertEqual(a['frames'],[])

    def test_field_types_include_all_objects(self):
        from pi_jwm.v2u4real_interface_v1 import field_inventory
        fields=field_inventory({'vehicles':{'1':{'confidence':1},'2':{'confidence':0.5}}})
        self.assertEqual({x['type'] for x in fields['vehicles.*.confidence']},{'int','float'})


class AirFogSimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['useCUPY']='False'
        sys.path.insert(0,str(ROOT/'code/reference/AirFogSim'))
        # Fail if the real simulator cannot import; no silent fake/skip PASS.
        from airfogsim.manager.traffic_manager import TrafficManager
        cls.tm_class=TrafficManager
        import run_v2u4real_airfogsim_interface_v1 as probe
        cls.probe=probe
        cls.config=json.loads((ROOT/'code/scripts/configs/v2u4real_airfogsim_interface_v1.json').read_text(encoding='utf8'))

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'.tmp')
        self.out=Path(self.tmp.name)
        self.c=self.probe.prepare_config(self.config,self.out)
        random.seed(20261025);np.random.seed(20261025)
    def tearDown(self):self.tmp.cleanup()

    def test_native_newborn_and_velocity_regression(self):
        result=self.probe.native_probe(self.c,self.out)
        self.assertEqual(result['status'],'BLOCKED')
        self.assertFalse(result['position_key_survived_setter'])

    def test_full_real_env_replay_and_channels_seeded(self):
        a=self.probe.environment_probe(self.c,20261025)
        random.seed(20261025);np.random.seed(20261025)
        b=self.probe.environment_probe(self.c,20261025)
        self.assertEqual(a,b);self.assertEqual(len(a),3)
        self.assertNotEqual(a[0]['grid']['vehicle_0'],a[1]['grid']['vehicle_0'])
        for row in a:
            for link, idx in [('V2I',0),('U2I',1)]:
                v=row['channel'][link]
                self.assertEqual(v['shape'],[1,1,50]);self.assertEqual(v['active_rb_indices'],[idx])
                # This diagnostic captures a defect, not a service PASS.
                self.assertGreater(row['diagnostic_outage_threshold_0']['inactive_rb_max_mbps'][link],1.99)

    def test_propagation_counterexample(self):
        a=self.probe.distance_probe()
        self.assertEqual(a['status'],'PASS')
        self.assertEqual(a['uma_geometry']['status'],'BLOCKED')

    def test_replay_contract_refuses_irregular_time_and_outside_bounds(self):
        cls=make_absolute_replay_manager(self.tm_class)
        tm=cls(deepcopy(self.c['traffic']),None,self.c['sumo']['sumo_net'])
        rows=self.probe.snapshots(.1);rows[1]['time_s']=.12
        with self.assertRaises(InterfaceBlocked):tm.bind_snapshots(rows)
        rows=self.probe.snapshots(.1);rows[1]['uavs']['UAV_0']['position'][0]=600
        with self.assertRaises(InterfaceBlocked):tm.bind_snapshots(rows)

    def test_replay_reset_exhaustion_and_control_rejection(self):
        cls=make_absolute_replay_manager(self.tm_class)
        tm=cls(deepcopy(self.c['traffic']),None,self.c['sumo']['sumo_net'])
        tm.bind_snapshots(self.probe.snapshots(.1));tm.stepSimulation();tm.reset()
        self.assertEqual(tm.getCurrentTime(),0)
        with self.assertRaises(InterfaceBlocked):tm.updateUAVMobilityPatterns({'UAV_0':{'speed':1}})
        for _ in range(3):tm.stepSimulation()
        with self.assertRaises(InterfaceBlocked):tm.stepSimulation()


if __name__=='__main__':unittest.main()
