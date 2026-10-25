"""CPU feasibility evidence. Synthetic probes never stand in for real data.

Exit 0: all requested experiments pass; 2: documented BLOCKED/NOT_RUN;
1: unexpected execution/acceptance failure. Run with UTF-8 console encoding.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from pi_jwm.v2u4real_interface_v1 import (
    audit_dataset, common_to_rsu, make_absolute_replay_manager, sha256,
    transform_points,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def source_audit():
    m = json.loads((ROOT / 'code/reference/V2U4Real.SOURCE.json').read_text(encoding='utf-8'))
    verified = {}
    for rel, expected in m['files'].items():
        p = ROOT / 'code/reference/V2U4Real' / rel
        verified[rel] = p.exists() and sha256(p) == expected
    return {'status': 'PASS' if all(verified.values()) else 'BLOCKED',
            'commit': m['commit'], 'partial_snapshot': True, 'verified_files': verified}


def geometry_probe():
    # Explicit artificial right-handed test frame, metres, Z-up. These are NOT
    # inferred V2U4Real axes or metadata poses. Box corners are synthetic too.
    points = np.array([[0, 0, 0], [1, 2, 3], [-2, 1, 4]], float)
    t = np.array([[0, -1, 0, 20], [1, 0, 0, 30], [0, 0, 1, 4], [0, 0, 0, 1]], float)
    public = transform_points(points, t)
    inverse = transform_points(public, np.linalg.inv(t))
    rsu = np.eye(4); rsu[:3, 3] = [300, 300, 25]
    rsu_points = common_to_rsu(public, rsu)
    err = float(np.max(np.abs(inverse - points)))
    rsu_err = float(np.max(np.abs(transform_points(rsu_points, rsu) - public)))
    check(err < 1e-10 and rsu_err < 1e-10, 'rigid round-trip')
    # Independent known-axis check, rather than only a self-inverse check.
    np.testing.assert_allclose(public[1], [18, 31, 7], atol=1e-10)
    return {'status': 'PASS', 'scope': 'synthetic_geometry_only', 'roundtrip_max_error': err,
            'rsu_roundtrip_max_error': rsu_err, 'rsu_to_common': rsu.tolist(),
            'rsu_provenance': 'simulation setting', 'dataset_alignment': 'BLOCKED',
            'reason': 'Real cloud/annotation frame, calibration and axis directions unavailable'}


def snapshots(interval):
    rows = []
    for i, x in enumerate((348., 349., 350., 351.)):
        rows.append({'time_s': i * interval,
                     'vehicles': {'vehicle_0': {'id': 'vehicle_0', 'position': [x, 300, 0], 'speed': 10., 'acceleration': 0., 'angle': 0., 'routeId': 'synthetic'}},
                     'uavs': {'UAV_0': {'position': [x, 300, 50 + i * .5], 'speed': math.sqrt(125), 'acceleration': 0., 'angle': 0., 'phi': math.atan2(.5, 1.)}}})
    # Speeds drive shadow correlation, so derive them from the recorded steps.
    for i in range(1, len(rows)):
        for group in ('vehicles', 'uavs'):
            for key, node in rows[i][group].items():
                delta = np.array(node['position']) - rows[i-1][group][key]['position']
                node['speed'] = float(np.linalg.norm(delta) / interval)
                node['angle'] = float(np.arctan2(delta[1], delta[0]))
                if group == 'uavs':
                    node['phi'] = float(np.arctan2(delta[2], np.linalg.norm(delta[:2])))
    return rows


def prepare_config(cfg, output):
    c = deepcopy(cfg['airfogsim'])
    c['sumo']['sumo_net'] = str(ROOT / c['sumo']['sumo_net'])
    trip = output / 'synthetic_tripinfo.csv'
    # Includes time zero. A separate one-row file isolates a newborn at t=.1.
    trip.write_text('vehicle_id;data_timestep;vehicle_x;vehicle_y;vehicle_speed;vehicle_angle;vehicle_route\n'
                    'vehicle_0;0.0;348;300;10;0;synthetic\n'
                    'vehicle_0;0.1;349;300;10;0;synthetic\n'
                    'vehicle_0;0.2;350;300;10;0;synthetic\n'
                    'vehicle_0;0.3;351;300;10;0;synthetic\n', encoding='utf-8')
    c['traffic']['tripinfo'] = str(trip)
    return c


def native_probe(c, output):
    from airfogsim.manager.traffic_manager import TrafficManager
    from airfogsim.scheduler.traffic_sched import TrafficScheduler
    from types import SimpleNamespace
    tm = TrafficManager(deepcopy(c['traffic']), None, c['sumo']['sumo_net'])
    start = np.array(tm.getUAVTrafficInfos()['UAV_0']['position'])
    holder = SimpleNamespace(uav_mobility_patterns={})
    TrafficScheduler.setUAVMobilityPatterns(holder, {'UAV_0': {'speed': 2., 'angle': 0., 'phi': 0., 'position': [1, 1, 1]}})
    tm.updateUAVMobilityPatterns(holder.uav_mobility_patterns)
    tm.stepSimulation()
    end = np.array(tm.getUAVTrafficInfos()['UAV_0']['position'])
    np.testing.assert_allclose(end - start, [.2, 0, 0], atol=1e-10)
    rows = [{'time': tm.getCurrentTime(), 'vehicles': deepcopy(tm.getVehicleTrafficInfos()), 'uav_position': end.tolist()}]
    for _ in range(2):
        tm.stepSimulation()
        rows.append({'time': tm.getCurrentTime(), 'vehicles': deepcopy(tm.getVehicleTrafficInfos())})
    # Isolate the real-mode time-window ordering problem with one newborn.
    newborn = output / 'synthetic_newborn.csv'
    newborn.write_text('vehicle_id;data_timestep;vehicle_x;vehicle_y;vehicle_speed;vehicle_angle;vehicle_route\nnewborn;0.1;400;300;0;0;synthetic\n', encoding='utf-8')
    nc = deepcopy(c['traffic']); nc['tripinfo'] = str(newborn)
    nt = TrafficManager(nc, None, c['sumo']['sumo_net'])
    nt.stepSimulation()
    first = deepcopy(nt.getVehicleTrafficInfos())
    nt.stepSimulation()
    second = deepcopy(nt.getVehicleTrafficInfos())
    check(not first and not second, 'native newborn regression changed; inspect vendor source')
    return {'status': 'BLOCKED', 'scope': 'native_real_mode_synthetic_probe', 'trace': rows,
            'uav_displacement': (end-start).tolist(), 'position_key_survived_setter': 'position' in holder.uav_mobility_patterns['UAV_0'],
            'newborn_at_0_1': {'at_0_1': first, 'at_0_2': second},
            'reason': 'Native real mode drops newborn: IDs selected before clock advance; UAV setter is velocity integration, not absolute replay; vehicle z forced zero'}


def channel_record(cm):
    result = {}
    for name in ('V2I', 'U2I'):
        channel = getattr(cm, name + 'Channel')
        final = np.asarray(getattr(cm, name + 'Channel_with_fastfading'))
        rate = np.asarray(getattr(cm, name + '_Rate'))
        for array in (channel.PathLoss, channel.Shadow, channel.FastFading, final, rate):
            check(np.isfinite(array).all(), name + ' nonfinite value')
        check(final.shape == (1, 1, cm.n_RB) and rate.shape == final.shape, name + ' dimension')
        check(channel.PathLoss.shape == final.shape and channel.FastFading.shape == final.shape and channel.Shadow.shape == (1, 1), name + ' component dimensions')
        check((rate >= 0).all(), name + ' negative rate')
        np.testing.assert_allclose(final, channel.PathLoss + channel.Shadow[:, :, None] - channel.FastFading, atol=1e-10)
        result[name] = {'shape': list(final.shape), 'pathloss_db': channel.PathLoss.tolist(),
                        'shadow_db': channel.Shadow.tolist(), 'fast_fading_db': channel.FastFading.tolist(),
                        'final_attenuation_db': final.tolist(), 'rate_mbps_per_rb': rate.tolist(),
                        'active_shape': list(getattr(cm, name+'_active_links').shape),
                        'active_rb_indices': np.flatnonzero(getattr(cm, name+'_active_links')[0, 0]).tolist(),
                        'outage_mask': getattr(cm, 'is_'+name+'_outage').tolist(),
                        'sinr_db': getattr(cm, name+'_SINR').tolist()}
    return result


def environment_probe(c, seed):
    random.seed(seed); np.random.seed(seed)
    from airfogsim import AirFogSimEnv
    from airfogsim.manager.traffic_manager import TrafficManager
    env = AirFogSimEnv(deepcopy(c), interactive_mode=None)
    replay_cls = make_absolute_replay_manager(TrafficManager)
    replay = replay_cls(deepcopy(c['traffic']), None, c['sumo']['sumo_net'])
    rows = snapshots(c['traffic']['traffic_interval'])
    replay.bind_snapshots(rows)
    # Explicit instance-local adapter, installed before env.step; no patching of
    # AirFogSim classes, GUI stubs, task overrides, or approximate velocity path.
    env.traffic_manager = replay
    trace = []
    for row in rows[1:]:
        # Required native callback even when no computing tasks exist. Empty
        # allocation is an explicit no-op, not a replacement computation path.
        env.alloc_cpu_callback = lambda *_args, **_kwargs: {}
        env.step()
        check(abs(env.simulation_time-row['time_s']) < 1e-10, 'clock')
        for key, node in row['vehicles'].items():
            np.testing.assert_allclose(env.vehicles[key].getPosition(), node['position'])
        for key, node in row['uavs'].items():
            np.testing.assert_allclose(env.UAVs[key].getPosition(), node['position'])
        np.testing.assert_allclose(env.RSUs['RSU_0'].getPosition(), [300, 300, 25])
        for key in ('vehicle_0', 'UAV_0', 'RSU_0'):
            check(replay.getMapIndexByNodeId(key) != (None, None), 'grid membership')
        # Raw RF calculation accepts link profiles without AirFogSim Task
        # objects. This does not run a feature-packet service or offload task.
        cm = env.channel_manager
        cm.resetActiveLinks()
        profiles = {'v': {'tx_idx': 0, 'rx_idx': 0, 'channel_type': 'V2I', 'RB_Nos': [0]},
                    'u': {'tx_idx': 0, 'rx_idx': 0, 'channel_type': 'U2I', 'RB_Nos': [1]}}
        for p in profiles.values():
            cm.activateLink(p['tx_idx'], p['rx_idx'], p['RB_Nos'], p['channel_type'])
        cm.computeRate(profiles)
        # Outage-free diagnostic (local only) reveals whether inactive RBs
        # acquire phantom rate through the vendor SINR floor. Not a protocol.
        normal_threshold = cm.snr_threshold
        cm.snr_threshold = 0.
        cm.computeRate(profiles)
        inactive_rates = {name: float(getattr(cm, name+'_Rate')[0, 0, 2:].max()) for name in ('V2I', 'U2I')}
        cm.snr_threshold = normal_threshold
        cm.computeRate(profiles)
        trace.append({'time_s': env.simulation_time, 'vehicle': env.vehicles['vehicle_0'].getPosition(),
                      'uav': env.UAVs['UAV_0'].getPosition(), 'rsu': env.RSUs['RSU_0'].getPosition(),
                      'grid': {k: list(replay.getMapIndexByNodeId(k)) for k in ('vehicle_0', 'UAV_0', 'RSU_0')},
                      'native_entity_last_updated_time_s': {'vehicle': env.vehicles['vehicle_0']._last_updated_time, 'uav': env.UAVs['UAV_0']._last_updated_time},
                      'channel': channel_record(cm),
                      'diagnostic_outage_threshold_0': {'inactive_rb_max_mbps': inactive_rates, 'scope': 'diagnostic only; default restored'}})
    env.close()
    for name in ('V2I', 'U2I'):
        check(trace[-1]['channel'][name]['pathloss_db'] != trace[0]['channel'][name]['pathloss_db'], 'motion not reaching '+name)
    return trace


def distance_probe():
    from airfogsim.channel_callback.pathloss_callback import UMa_LOS_pathloss, FreeSpacePathLoss
    distances = [50., 100., 200., 400.]
    curves = {}
    f = np.array([2.4, 2.402])
    for name, model, z in [('V2I', UMa_LOS_pathloss, 0.), ('U2I', FreeSpacePathLoss, 50.)]:
        values = [model(np.array([[d, 0, z]]), np.array([[0., 0, 25.]]), f)[0, 0].tolist() for d in distances]
        check((np.diff(values, axis=0) > 0).all(), name+' distance monotonicity')
        curves[name] = values
    # Free-space exact formula; the vendor variable called d_2d is a 3D norm.
    expected = 20*np.log10(np.sqrt(np.array(distances)**2+25**2)+1e-9)+20*np.log10(2.4+1e-9)+32.44
    np.testing.assert_allclose(np.array(curves['U2I'])[:, 0], expected)
    # UMa's documented horizontal d_2d currently includes z, then adds default
    # antenna height again. Report the mismatch; do not silently repair it.
    d = 100.
    actual = UMa_LOS_pathloss(np.array([[d, 0, 0.]]), np.array([[0., 0, 25.]]), f)[0, 0, 0]
    intended = 28 + 22*np.log10(np.sqrt(d*d+23.5**2)+1e-9) + 20*np.log10(2.4+1e-9)
    check(abs(actual-intended) > 1e-3, 'UMa source changed: re-audit geometry')
    return {'status': 'PASS', 'distances_horizontal_m': distances, 'pathloss_db_by_rb': curves,
            'models': {'V2I': 'UMa_LOS_tr38901 (current vendor implementation)', 'U2I': 'free_space'},
            'uma_geometry': {'status': 'BLOCKED', 'actual_db_at_100m': float(actual),
                             'documented_horizontal_formula_db': float(intended), 'difference_db': float(actual-intended),
                             'reason': '3D norm used as horizontal distance; default h_BS=25/h_UT=1.5 are not read from node z'},
            'claim': 'Monotonicity/exact U2I arithmetic only; no real radio calibration or CSI measurement'}


def run(config, output, dataset_root=None):
    output.mkdir(parents=True, exist_ok=True)
    data = audit_dataset(dataset_root or ROOT / config['dataset_root'], config['max_frames'])
    write_json(output / 'data_audit.json', data)
    geometry = geometry_probe()
    write_json(output / 'geometry.json', geometry)
    result = {'schema': 'PIJWM-V2U4Real-AirFogSim-interface-round1-v1', 'device': 'cpu', 'gpu_used': False,
              'locked_test_accessed': False, 'training_run': False, 'data_downloaded': False,
              'A': {'status': 'BLOCKED', 'evidence': 'data_audit.json', 'reason': data['reason']},
              'B': {'status': 'BLOCKED', 'evidence': 'geometry.json', 'reason': geometry['reason']},
              'C': {'status': 'BLOCKED', 'reason': 'Real dataset pose/time contract unavailable'},
              'D': {'status': 'BLOCKED', 'reason': 'Not yet executed'},
              'E': {'status': 'NOT_RUN', 'reason': 'C/D real-data and propagation gates not both PASS; no feature service implemented'}}
    result['official_source'] = source_audit()
    os.environ['useCUPY'] = 'False'
    sys.path.insert(0, str(ROOT / 'code/reference/AirFogSim'))
    try:
        seed = config['seed']
        random.seed(seed); np.random.seed(seed)
        c = prepare_config(config, output)
        write_json(output / 'runtime_config.json', c)
        native = native_probe(c, output)
        write_json(output / 'native_replay.json', native)
        random.seed(seed); np.random.seed(seed)
        trace = environment_probe(c, seed)
        random.seed(seed); np.random.seed(seed)
        repeated = environment_probe(c, seed)
        check(trace == repeated, 'seed reproducibility')
        distance = distance_probe()
        write_json(output / 'adapted_environment.json', {'status': 'PASS', 'scope': 'synthetic_only', 'real_env_step_count': len(trace),
                    'seed': seed, 'repeat_equal': True, 'trace': trace})
        write_json(output / 'distance_probe.json', distance)
        result['C'].update(native_status='BLOCKED', adapted_synthetic_status='PASS', evidence=['native_replay.json', 'adapted_environment.json'])
        result['C']['timestamp_semantics'] = 'BLOCKED: native entity last_updated_time uses previous env time while position and TrafficManager use advanced frame; not rewritten'
        result['D'].update(status='BLOCKED', interface_status='PASS', reason=distance['uma_geometry']['reason'],
                           evidence=['adapted_environment.json', 'distance_probe.json'], simulation_csi_only=True)
        result['D']['additional_semantic_blockers'] = [
            'Outage callback consumes dB SINR although its exponential formula expects a linear power ratio (historically documented)',
            'With outage disabled for diagnosis, inactive RBs have about 2 Mbps through the SINR floor; sum allocated RBs only',
            'Fast-fading function broadcasts one PDF-derived value across RBs; not independent per-RB Rayleigh power draws']
    except Exception as exc:
        result['runtime_failure'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        write_json(output / 'acceptance.json', result)
        raise
    identity = {}
    for base in ('code/src/pi_jwm/v2u4real_interface_v1.py', 'code/scripts/run_v2u4real_airfogsim_interface_v1.py',
                 'code/scripts/probe_v2u4real_official_transform_v1.py', 'code/scripts/fetch_v2u4real_audit_sources_v1.py',
                 'code/tests/test_v2u4real_interface_v1.py',
                 'code/scripts/configs/v2u4real_airfogsim_interface_v1.json', 'code/scripts/configs/v2u4real_tiny.net.xml',
                 'code/reference/V2U4Real.SOURCE.json', 'code/reference/AirFogSim/airfogsim/airfogsim_env.py',
                 'code/reference/AirFogSim/airfogsim/manager/traffic_manager.py', 'code/reference/AirFogSim/airfogsim/scheduler/traffic_sched.py',
                 'code/reference/AirFogSim/airfogsim/manager/channel_manager_cp.py', 'code/reference/AirFogSim/airfogsim/all_channels.py'):
        identity[base] = sha256(ROOT / base)
    for p in (ROOT / 'code/reference/AirFogSim/airfogsim/channel_callback').glob('*.py'):
        identity[p.relative_to(ROOT).as_posix()] = sha256(p)
    write_json(output / 'source_identity.json', {'files': identity, 'python': sys.version, 'executable': sys.executable})
    write_json(output / 'acceptance.json', result)
    print(json.dumps({key: result[key] for key in ('A', 'B', 'C', 'D', 'E')}, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'code/scripts/configs/v2u4real_airfogsim_interface_v1.json')
    parser.add_argument('--output', type=Path, default=ROOT/'code/artifacts/audit/v2u4real_airfogsim_round1_20261025')
    parser.add_argument('--dataset-root', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    result = run(config, args.output.resolve(), args.dataset_root)
    return 0 if all(result[k]['status'] == 'PASS' for k in 'ABCDE') else 2


if __name__ == '__main__':
    raise SystemExit(main())
