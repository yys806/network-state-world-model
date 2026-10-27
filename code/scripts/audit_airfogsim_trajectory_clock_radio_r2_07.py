"""CPU-only R2-07 interface audit; synthetic inputs, no packet lifecycle.

Exit 0: software checks passed with explicit physical-model blocks.
Exit 1: unexpected failure. The result is never a transmission success receipt.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

import numpy as np

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='strict')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
sys.path.insert(0, str(ROOT / 'code/scripts'))
os.environ['useCUPY'] = 'False'
sys.path.insert(0, str(ROOT / 'code/reference/AirFogSim'))

from pi_jwm.v2u4real_interface_v1 import InterfaceBlocked, make_absolute_replay_manager, sha256
from pi_jwm.airfogsim_service_interface_v1 import observe_post_step, read_post_step_link_service, slot_service
from run_v2u4real_airfogsim_interface_v1 import prepare_config

OUT = ROOT / 'code/artifacts/audit/v2u4real_airfogsim_r2_07_20261027'
CONFIG = ROOT / 'code/scripts/configs/v2u4real_airfogsim_interface_v1.json'
SEED = 20261027


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def synthetic_snapshots(dt=.1):
    membership = [(0,), (0,), (0, 1, 2), (1, 2), (2,), (2, 3), (), (4,), (4,)]
    xs = (348., 349., 350., 349., 351., 350., 349., 350., 351.)
    zs = (50., 50.5, 51., 50., 50.5, 51., 50., 50.5, 51.)
    rows = []
    for i, ids in enumerate(membership):
        vehicles = {f'vehicle_{k}': dict(id=f'vehicle_{k}', position=[xs[i]+k, 300.+k, 0.],
                                       speed=0., acceleration=0., angle=0., routeId='synthetic') for k in ids}
        uavs = {'UAV_0': dict(position=[xs[i], 300., zs[i]], speed=0., acceleration=0., angle=0., phi=0.)}
        for group_name, group in (('vehicles', vehicles), ('uavs', uavs)):
            for key, node in group.items():
                if i and key in rows[-1][group_name]:
                    d = np.asarray(node['position']) - rows[-1][group_name][key]['position']
                    node['speed'] = float(np.linalg.norm(d)/dt)
                    node['angle'] = float(np.arctan2(d[1], d[0]))
                    if group_name == 'uavs':
                        node['phi'] = float(np.arctan2(d[2], np.linalg.norm(d[:2])))
        rows.append(dict(time_s=i*dt, vehicles=vehicles, uavs=uavs))
    return rows


def raw_state(env):
    return dict(env_time_s=float(env.simulation_time), traffic_time_s=float(env.traffic_manager.getCurrentTime()),
                vehicles={k: list(v.getPosition()) for k, v in env.vehicles.items()},
                uavs={k: list(v.getPosition()) for k, v in env.UAVs.items()},
                native_times={k: float(v._last_updated_time) for group in (env.vehicles, env.UAVs) for k, v in group.items()})


def environment_probe(c, rows):
    from airfogsim import AirFogSimEnv
    from airfogsim.manager.traffic_manager import TrafficManager
    random.seed(SEED)
    np.random.seed(SEED)
    env = AirFogSimEnv(deepcopy(c), interactive_mode=None)
    trace, during = [], []
    try:
        replay_cls = make_absolute_replay_manager(TrafficManager, dynamic_vehicle_ids=True)
        replay = replay_cls(deepcopy(c['traffic']), None, c['sumo']['sumo_net'])
        replay.bind_snapshots(rows)
        env.traffic_manager = replay
        original_wireless = env._updateWirelessCommunication
        def instrumented_wireless():
            during.append(raw_state(env))
            return original_wireless()
        env._updateWirelessCommunication = instrumented_wireless
        for row in rows[1:]:
            before = raw_state(env)
            env.alloc_cpu_callback = lambda *_a, **_kw: {}
            env.step()
            observation = observe_post_step(env)
            assert set(env.vehicles) == set(row['vehicles'])
            assert math.isclose(env.simulation_time, row['time_s'], abs_tol=1e-10)
            assert math.isclose(during[-1]['env_time_s'] + env.simulation_interval, row['time_s'], abs_tol=1e-10)
            assert math.isclose(during[-1]['traffic_time_s'], row['time_s'], abs_tol=1e-10)
            for kind in ('vehicles', 'uavs'):
                for key, node in row[kind].items():
                    entity = getattr(env, 'UAVs' if kind == 'uavs' else kind)[key]
                    np.testing.assert_allclose(entity.getPosition(), node['position'], rtol=0, atol=1e-10)
                    assert math.isclose(entity.getSpeed(), node['speed'], abs_tol=1e-10)
                    assert math.isclose(entity.getAcceleration(), node['acceleration'], abs_tol=1e-10)
                    assert math.isclose(entity._angle, node['angle'], abs_tol=1e-10)
                    if kind == 'uavs':
                        assert math.isclose(entity._phi, node['phi'], abs_tol=1e-10)
                    expected_grid = (int((node['position'][1]-replay._y_range[0])/replay._grid_width),
                                     int((node['position'][0]-replay._x_range[0])/replay._grid_width))
                    assert tuple(observation['entities'][key]['grid']) == expected_grid
            map_ids = [k for cell in replay._map_by_grid.flat for k in cell]
            assert set(map_ids) == set(row['vehicles']) | {'UAV_0', 'RSU_0'}
            cm = env.channel_manager
            assert not cm.V2I_active_links.any() and not cm.U2I_active_links.any()
            profiles = {key: dict(tx_idx=idx, rx_idx=0, channel_type='V2I', RB_Nos=[2*idx])
                        for idx, key in enumerate(env.vehicle_ids_as_index)}
            profiles['UAV_0'] = dict(tx_idx=0, rx_idx=0, channel_type='U2I', RB_Nos=[1, 7])
            cm.resetActiveLinks()
            for p in profiles.values():
                cm.activateLink(p['tx_idx'], p['rx_idx'], p['RB_Nos'], p['channel_type'])
            cm.computeRate(profiles)
            services = []
            for key, p in profiles.items():
                record = read_post_step_link_service(env, p['channel_type'], key, 'RSU_0', p['RB_Nos'])
                raw = cm.getRateByChannelType(p['tx_idx'], p['rx_idx'], p['channel_type'])
                np.testing.assert_array_equal(record['raw_rate_mbps'], raw)
                sinr = getattr(cm, p['channel_type']+'_SINR')[p['tx_idx'], p['rx_idx']]
                outage = getattr(cm, 'is_'+p['channel_type']+'_outage')[p['tx_idx'], p['rx_idx']]
                independently_scaled = np.where(outage, 0., cm.RB_bandwidth*np.log2(1+10**(sinr/10)))
                np.testing.assert_allclose(raw, independently_scaled, rtol=0, atol=1e-10)
                channel = getattr(cm, p['channel_type']+'Channel')
                positions = np.asarray([env.vehicles[k].getPosition() for k in env.vehicle_ids_as_index] if p['channel_type']=='V2I' else [env.UAVs[k].getPosition() for k in env.uav_ids_as_index])
                rsu_positions = np.asarray([env.RSUs[k].getPosition() for k in env.rsu_ids_as_index])
                direct_geometry = channel.pathloss_callback(positions, rsu_positions, np.asarray(cm.RB_frequencies))
                np.testing.assert_allclose(channel.PathLoss, direct_geometry, rtol=0, atol=1e-10)
                record['native_sinr_db'] = np.asarray(sinr).tolist()
                record['native_outage_mask'] = np.asarray(outage).tolist()
                record['native_pathloss_db'] = channel.PathLoss[p['tx_idx'], p['rx_idx']].tolist()
                record['unit_formula_crosscheck'] = 'PASS'
                record['current_position_to_native_pathloss_dispatch'] = 'PASS'
                selected = np.asarray(raw)[p['RB_Nos']]
                assert math.isclose(record['theoretical_bytes'], float(selected.sum())*1e6*.1/8, abs_tol=1e-8)
                services.append(record)
            # Same sample, empty allocation: no actual task or data packet submitted.
            cm.resetActiveLinks()
            cm.computeRate({})
            zero = read_post_step_link_service(env, 'U2I', 'UAV_0', 'RSU_0', [])
            assert zero['theoretical_bytes'] == 0
            assert cm.V2I_Rate.shape == (len(row['vehicles']), 1, 50)
            trace.append(dict(before_step=before, during_native_wireless=during[-1],
                              after_step=observation, vehicle_index=list(env.vehicle_ids_as_index),
                              v2i_shape=list(cm.V2I_Rate.shape), u2i_shape=list(cm.U2I_Rate.shape),
                              services=services, no_allocation_service=zero,
                              vendor_outage_threshold_unchanged=float(cm.snr_threshold)))
        # Adapter stops explicitly instead of silently holding/extrapolating the final frame.
        try:
            replay.stepSimulation()
        except InterfaceBlocked:
            exhausted = True
        else:
            raise AssertionError('exhaustion was not rejected')
        return dict(status='PASS', trace=trace, exhausted_rejected=exhausted,
                    bootstrap='time-zero TrafficManager snapshot is installed; native env entities first synchronize on first step',
                    rb_bandwidth_mhz=float(cm.RB_bandwidth), n_rb=int(cm.n_RB),
                    rb_frequencies_ghz=np.asarray(cm.RB_frequencies).tolist())
    finally:
        env.close()


def formula_probe():
    from airfogsim.channel_callback.pathloss_callback import UMa_LOS_pathloss, FreeSpacePathLoss
    from airfogsim.channel_callback.outage_callback import rayleigh_outage_prob
    from airfogsim.channel_callback.fading_callback import Rayleigh_fastfading
    freq = np.array([2.4])
    cases = []
    for horizontal in (50., 100., 200., 400.):
        tx = np.array([[horizontal, 0., 0.]])
        rx = np.array([[0., 0., 25.]])
        pl = float(UMa_LOS_pathloss(tx, rx, freq)[0,0,0])
        d_input = float(np.linalg.norm(tx-rx))
        d_effective = math.hypot(d_input, 23.5)
        # Reproduce both native branches as an algebra check, not adoption
        # of a new physical height convention (400 m exceeds the breakpoint).
        breakpoint = 4*(25-1)*(1.5-1)*2.4e9/3e8
        expected = 28+22*math.log10(1e-9+d_effective)+20*math.log10(1e-9+2.4)
        if d_input > breakpoint:
            expected += 18*math.log10(1e-9+d_effective)-9*math.log10(1e-9+breakpoint**2+23.5**2)
        assert math.isclose(pl, expected, abs_tol=1e-10)
        u = np.array([[horizontal, 0., 60.]])
        upl = float(FreeSpacePathLoss(u, rx, freq)[0,0,0])
        d3 = float(np.linalg.norm(u-rx))
        assert math.isclose(upl, 20*math.log10(1e-9+d3)+20*math.log10(1e-9+2.4)+32.44, abs_tol=1e-10)
        shifted = float(FreeSpacePathLoss(u+[0,0,100], rx+[0,0,100], freq)[0,0,0])
        assert math.isclose(upl, shifted, abs_tol=1e-10)
        cases.append(dict(horizontal_m=horizontal, v2i_input_norm_m=d_input, v2i_effective_norm_m=d_effective,
                          v2i_native_pl_db=pl, u2i_distance_3d_m=d3, u2i_native_pl_db=upl))
    # Direct source-formula mismatch; no channel parameters or callback replaced.
    db = np.array([-10., 0., 10., 20.])
    native = rayleigh_outage_prob(db, 10.)
    linear_candidate = rayleigh_outage_prob(10**(db/10), 10.)
    assert abs(native[-1]-linear_candidate[-1]) > .2
    np.random.seed(SEED)
    fading = Rayleigh_fastfading(2, 1, 50)
    np.random.seed(SEED)
    g1 = np.random.normal(0,1,(2,1)); g2 = np.random.normal(0,1,(2,1))
    r = np.sqrt(g1*g1+g2*g2); omega = np.mean(r*r)
    expected = 10*np.log10(1e-9+(2*r/omega)*np.exp(-r*r/omega))
    np.testing.assert_allclose(fading, np.repeat(expected[:,:,None], 50, axis=2), rtol=0, atol=1e-12)
    artificial = slot_service([2., 2., 8.], np.array([False, False, True]), .1)
    assert artificial['theoretical_bytes'] == 100000.
    assert slot_service([2., 2.], np.array([False, False]), .1)['theoretical_bytes'] == 0
    return dict(source_behavior_status='PASS', physical_model_acceptance='BLOCKED', distance_cases=cases,
                outage=dict(sinr_db=db.tolist(), threshold_as_configured=10., native_probability=native.tolist(),
                            linear_input_candidate_probability=linear_candidate.tolist(),
                            candidate_is_not_adopted=True, issue='dB is passed into a ratio-based exponential; threshold units and use of instantaneous vs mean SINR await decision'),
                fastfading=dict(shape=list(fading.shape), identical_across_rbs=bool(np.all(fading==fading[:,:,:1])),
                                reproduced_from_sample_pdf=True, physical_acceptance='BLOCKED'),
                inactive_rb_fixture=dict(scope='synthetic arithmetic guard, not a native RF measurement', result=artificial),
                sinr_floor=dict(native_floor_db=1e-9, implied_min_linear=10**(1e-9/10),
                                no_outage_floor_rate_per_2mhz_rb=2*math.log2(1+10**(1e-9/10)),
                                native_threshold_modified=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protected = json.loads((ROOT/'code/artifacts/audit/v2u4real_airfogsim_round1_20261025/source_identity.json').read_text(encoding='utf-8'))['files']
    protected = {p: h for p,h in protected.items() if p.startswith('code/reference/AirFogSim/')}
    for name in ('vehicle.py', 'uav.py', 'abstract/simple_node.py'):
        p = 'code/reference/AirFogSim/airfogsim/entities/' + name
        protected[p] = sha256(ROOT/p)
    evidence_roots = ('v2u4real_airfogsim_round1_20261025', 'v2u4real_r2_05b_20261027', 'v2u4real_r2_06_20261027')
    for name in evidence_roots:
        for p in (ROOT/'code/artifacts/audit'/name).rglob('*'):
            if p.is_file(): protected[p.relative_to(ROOT).as_posix()] = sha256(p)
    before = {p: sha256(ROOT/p) for p in protected}
    assert all(before[p] == h for p,h in protected.items()), 'inherited vendor hash differs'
    result = dict(schema_version=1, step='R2-07', status='BLOCKED',
                  claim_boundary='synthetic software interfaces only; no real physical replay, validated radio model or wireless delivery',
                  seed=SEED, source_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                  python=sys.version, executable=sys.executable, command=sys.argv,
                  config_path=CONFIG.relative_to(ROOT).as_posix(), config_sha256=sha256(CONFIG),
                  restrictions=dict(gpu_used=False, training_started=False, dataset_read=False,
                                    validation_test_locked_test_accessed=False, downloaded=False,
                                    propagation_model_modified=False, packet_lifecycle_implemented=False))
    try:
        cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
        c = prepare_config(cfg, args.output)
        c['traffic']['max_n_vehicles'] = 6
        c['simulation']['max_simulation_time'] = 2.
        write_json(args.output/'runtime_config.json', c)
        rows = synthetic_snapshots()
        write_json(args.output/'synthetic_snapshots.json', rows)
        a = environment_probe(c, rows)
        b = environment_probe(c, rows)
        assert a == b, 'seeded repeat differs'
        write_json(args.output/'trajectory_radio_trace.json', a)
        result.update(status='COMPLETED_WITH_BLOCKERS', software_status='PASS', repeated_runs_equal=True, steps_per_run=len(a['trace']),
                      trajectory_clock='PASS', rb_index_mask_units_service='PASS', native_timestamp='BLOCKED',
                      native_internal_clock_position_alignment='BLOCKED', physical_radio='BLOCKED',
                      real_dataset_time_replay='BLOCKED', packet_delivery='NOT_RUN',
                      task3_prerequisites='synthetic entity/time/grid interface PASS; task semantics and real time mapping pending',
                      task4_prerequisites='masked snapshot rate and unit conversion PASS; physical model and packet lifecycle pending',
                      formula_diagnostics=formula_probe())
    except Exception:
        result['software_status'] = 'BLOCKED'
        result['unexpected_failure'] = traceback.format_exc()
    result['protected_inputs'] = {p: dict(before=before[p], after=sha256(ROOT/p), unchanged=before[p]==sha256(ROOT/p)) for p in before}
    result['protection_status'] = 'PASS' if all(x['unchanged'] for x in result['protected_inputs'].values()) else 'BLOCKED'
    result['source_files'] = {p: sha256(ROOT/p) for p in (
        'code/src/pi_jwm/v2u4real_interface_v1.py', 'code/src/pi_jwm/airfogsim_service_interface_v1.py',
        'code/scripts/audit_airfogsim_trajectory_clock_radio_r2_07.py',
        'code/tests/test_airfogsim_service_interface_r2_07.py')}
    write_json(args.output/'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('protected_inputs','source_files','formula_diagnostics')}, ensure_ascii=False, indent=2))
    return 0 if result.get('software_status') == 'PASS' and result['protection_status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
