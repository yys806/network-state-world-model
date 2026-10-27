"""First-round interface tools. No detector, scheduler, or dataset pose inference.

Recorded poses must already have an independently verified common metric frame.
Synthetic fixtures exercise software only; they cannot establish dataset semantics.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import math
import re

import numpy as np
import yaml


class InterfaceBlocked(ValueError):
    """A necessary contract is missing or cannot be verified."""


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise InterfaceBlocked(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rigid_matrix(matrix):
    """Check a supplied local-to-common rigid transform, without inferring axes."""
    t = np.asarray(matrix, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise InterfaceBlocked("finite 4x4 transform required")
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-10, rtol=0):
        raise InterfaceBlocked("invalid homogeneous bottom row")
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-10, rtol=0) or not math.isclose(np.linalg.det(r), 1, abs_tol=1e-10):
        raise InterfaceBlocked("rotation must be orthonormal with determinant +1")
    return t


def transform_points(points, local_to_common):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise InterfaceBlocked("finite Nx3 points required")
    t = rigid_matrix(local_to_common)
    return p @ t[:3, :3].T + t[:3, 3]


def common_to_rsu(points, rsu_to_common):
    """The RSU frame is an explicit simulation setting, never a measured sensor."""
    return transform_points(points, np.linalg.inv(rigid_matrix(rsu_to_common)))


def read_pcd_xyz(path):
    """Minimal strict PCD reader; unsupported compression is BLOCKED, not empty."""
    with Path(path).open('rb') as f:
        header = {}
        for _ in range(64):
            line = f.readline()
            if not line:
                raise InterfaceBlocked("PCD missing DATA header")
            parts = line.decode('ascii').strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == 'DATA':
                break
        else:
            raise InterfaceBlocked("PCD header too long")
        payload = f.read()
    fields = header.get('FIELDS', [])
    sizes = [int(x) for x in header.get('SIZE', [])]
    types = header.get('TYPE', [])
    counts = [int(x) for x in header.get('COUNT', ['1'] * len(fields))]
    n = int(header['POINTS'][0])
    if n <= 0 or len(set(fields)) != len(fields) or not set('xyz') <= set(fields):
        raise InterfaceBlocked("PCD requires nonempty unique fields and xyz")
    if not (len(fields) == len(sizes) == len(types) == len(counts)) or any(x <= 0 for x in counts):
        raise InterfaceBlocked("inconsistent PCD field schema")
    if n != int(header['WIDTH'][0]) * int(header['HEIGHT'][0]):
        raise InterfaceBlocked("PCD POINTS != WIDTH*HEIGHT")
    if any(counts[fields.index(a)] != 1 for a in 'xyz'):
        raise InterfaceBlocked("xyz must be scalar")
    mode = header['DATA'][0]
    if mode == 'ascii':
        # np.fromstring tolerates trailing junk, so use loadtxt with strict columns.
        import io
        data = np.loadtxt(io.BytesIO(payload), ndmin=2)
        if data.shape != (n, sum(counts)):
            raise InterfaceBlocked("PCD ASCII point count/width mismatch")
        offsets = np.cumsum([0] + counts[:-1])
        xyz = data[:, [offsets[fields.index(a)] for a in 'xyz']]
    elif mode == 'binary':
        kind = {'F': 'f', 'U': 'u', 'I': 'i'}
        dtype = np.dtype([(name, '<' + kind[typ] + str(size), (count,))
                          for name, typ, size, count in zip(fields, types, sizes, counts)])
        if len(payload) != n * dtype.itemsize:
            raise InterfaceBlocked("PCD binary byte count mismatch")
        data = np.frombuffer(payload, dtype=dtype, count=n)
        xyz = np.column_stack([data[a][:, 0] for a in 'xyz'])
    else:
        raise InterfaceBlocked(f"PCD DATA={mode} unsupported; use official Open3D reader")
    if not np.isfinite(xyz).all():
        raise InterfaceBlocked("nonfinite PCD xyz")
    return xyz, header


def field_inventory(value, prefix=''):
    """Observe types/shapes only; never assign units from a key name."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            # Object IDs vary by frame; group their field types.
            name = '*' if prefix == 'vehicles' else str(key)
            for field, specs in field_inventory(item, f'{prefix}.{name}'.lstrip('.')).items():
                for spec in specs:
                    bucket = result.setdefault(field, [])
                    if spec not in bucket:
                        bucket.append(spec)
        return result
    if isinstance(value, list):
        return {prefix: [{'type': 'list', 'length': len(value), 'element_types': sorted({type(v).__name__ for v in value}), 'unit': 'UNVERIFIED'}]}
    return {prefix: [{'type': type(value).__name__, 'unit': 'UNVERIFIED'}]}


def audit_dataset(root, max_frames=16):
    """Inspect one smallest train scene. Physical clock remains unverified.

    Filename stems are frame keys, not timestamps in seconds. Even if all files
    parse, a separate timestamp/coordinate contract is required for real replay.
    """
    result = {'status': 'BLOCKED', 'scope': 'real_train_data', 'root': str(root),
              'clock_unit': None, 'actual_interval_s': None, 'time_range_s': None,
              'physical_clock_verified': False, 'coordinates_verified': False,
              'anomalies': [], 'observed_fields': {}, 'frames': []}
    if max_frames < 1:
        raise ValueError('max_frames must be positive')
    root = Path(root)
    if not root.exists():
        result['reason'] = 'No local V2U4Real subset at configured root; nothing downloaded'
        return result
    train = root / 'train'
    if not train.is_dir() or train.resolve().parent != root.resolve():
        result['reason'] = 'root/train required; validation/test and symlink escapes refused'
        return result
    candidates = []
    for scene in train.iterdir():
        if not scene.is_dir() or not scene.resolve().is_relative_to(train.resolve()):
            continue
        paths = [scene / a / 'yaml' / 'ouster' for a in ('1', '2')]
        counts = [len([y for y in p.glob('*.yaml') if 'additional' not in y.name]) for p in paths]
        if min(counts) > 0:
            candidates.append((sum(counts), scene.name, scene))
    if not candidates:
        result['reason'] = 'No paired train scene with 1/2 yaml/ouster found'
        return result
    scene = min(candidates)[2]
    keys = {a: {p.stem for p in (scene / a / 'yaml' / 'ouster').glob('*.yaml')
                if 'additional' not in p.name} for a in ('1', '2')}
    all_keys = sorted(keys['1'] | keys['2'])
    result.update(scene=scene.name, frame_counts={a: len(v) for a, v in keys.items()},
                  matched_key_count=len(keys['1'] & keys['2']),
                  missing_keys={a: sorted(set(all_keys) - v) for a, v in keys.items()},
                  frame_key_range=[all_keys[0], all_keys[-1]], frame_key_unit='identifier_only',
                  inspected_keys=all_keys[:max_frames], truncated=len(all_keys) > max_frames)
    timestamps = {}
    for a in ('1', '2'):
        numeric = sorted(int(x) for x in keys[a] if x.isdigit())
        result.setdefault('frame_number_gaps', {})[a] = [[x, y] for x, y in zip(numeric, numeric[1:]) if y != x+1]
    for key in all_keys[:max_frames]:
        for agent in ('1', '2'):
            y = scene / agent / 'yaml' / 'ouster' / (key + '.yaml')
            p = scene / agent / 'ouster' / (key + '.pcd')
            row = {'agent': agent, 'frame_key': key}
            try:
                if not y.resolve().is_relative_to(scene.resolve()) or not p.resolve().is_relative_to(scene.resolve()):
                    raise InterfaceBlocked('frame path escapes scene')
                value = yaml.load(y.read_text(encoding='utf-8-sig'), Loader=UniqueKeyLoader)
                if not isinstance(value, dict):
                    raise InterfaceBlocked('YAML root must be a mapping')
                xyz, header = read_pcd_xyz(p)
                row.update(yaml_sha256=sha256(y), pcd_sha256=sha256(p), point_count=len(xyz),
                           pcd_header=header, xyz_min=xyz.min(0).tolist(), xyz_max=xyz.max(0).tolist(),
                           annotation_count=len(value.get('vehicles', {})),
                           label_association='same YAML frame key; physical-time association UNVERIFIED')
                fields = field_inventory(value)
                for name, specs in fields.items():
                    bucket = result['observed_fields'].setdefault(name, [])
                    for spec in specs:
                        if spec not in bucket:
                            bucket.append(spec)
                for name, v in value.items():
                    if re.search(r'time|stamp', str(name), re.I):
                        row.setdefault('raw_time_fields', {})[str(name)] = repr(v)
                        timestamps.setdefault((agent, str(name)), []).append((key, repr(v)))
                row['read_status'] = 'PASS'
            except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
                row.update(read_status='BLOCKED', error=str(exc))
                result['anomalies'].append({'agent': agent, 'frame_key': key, 'error': str(exc)})
            result['frames'].append(row)
    result['duplicate_raw_time_values'] = {f'{a}.{name}': [v for v, n in Counter(v for _, v in vals).items() if n > 1]
                                           for (a, name), vals in timestamps.items()}
    result['reason'] = 'File audit only: measured timestamp units, axes, extrinsics and annotation frame require independent verification'
    return result


def make_absolute_replay_manager(base_class, *, dynamic_vehicle_ids=False):
    """An opt-in subclass; no vendored source or global monkeypatch changes.

    Input is a synthetic/verified common-frame contract, NOT raw dataset YAML.
    Strict, uniform snapshots are required; no interpolation or frame holding.
    R2-07 opt-in dynamic_vehicle_ids admits synthetic vehicle entry/exit only;
    the original one-vehicle/one-UAV default contract is unchanged.
    """
    class AbsoluteReplayManager(base_class):
        def bind_snapshots(self, snapshots):
            from copy import deepcopy
            rows = deepcopy(snapshots)
            if not rows:
                raise InterfaceBlocked('empty replay')
            for i, row in enumerate(rows):
                if not math.isclose(row['time_s'], i * self._traffic_interval, abs_tol=1e-10, rel_tol=0):
                    raise InterfaceBlocked('replay requires zero origin and uniform explicit seconds')
                if set(row['uavs']) != {'UAV_0'} or (not dynamic_vehicle_ids and set(row['vehicles']) != {'vehicle_0'}):
                    raise InterfaceBlocked('round-one contract requires one stable vehicle and UAV')
                if any(not isinstance(k, str) or not k or v.get('id', k) != k
                       for k, v in row['vehicles'].items()):
                    raise InterfaceBlocked('vehicle identity must match snapshot key')
                for nodes in (row['vehicles'], row['uavs']):
                    for v in nodes.values():
                        p = np.asarray(v['position'], dtype=float)
                        if p.shape != (3,) or not np.isfinite(p).all():
                            raise InterfaceBlocked('invalid position')
                        if not (self._x_range[0] <= p[0] < self._x_range[1] and self._y_range[0] <= p[1] < self._y_range[1]):
                            raise InterfaceBlocked('position outside grid (upper bound excluded)')
                        if nodes is row['uavs'] and not self._UAV_z_range[0] <= p[2] <= self._UAV_z_range[1]:
                            raise InterfaceBlocked('UAV altitude outside configured range')
                        for name in ('speed', 'acceleration', 'angle', 'phi'):
                            if name in v and not math.isfinite(v[name]):
                                raise InterfaceBlocked('nonfinite motion field')
                        if v.get('speed', 0) < 0:
                            raise InterfaceBlocked('negative speed')
            self._snapshots = rows
            self._replay_index = 0
            self._install_snapshot(rows[0])

        def _install_snapshot(self, row):
            self._current_time = row['time_s']
            self._vehicle_infos = row['vehicles']
            self._UAV_infos = row['uavs']
            self._update_map_by_grid()

        def stepSimulation(self):
            if self._replay_index + 1 >= len(self._snapshots):
                raise InterfaceBlocked('replay exhausted; no extrapolation permitted')
            self._replay_index += 1
            self._install_snapshot(self._snapshots[self._replay_index])

        def reset(self, traci_connection=None):
            super().reset(traci_connection)
            if hasattr(self, '_snapshots'):
                self._replay_index = 0
                self._install_snapshot(self._snapshots[0])

        def updateVehicleMobilityPatterns(self, patterns):
            if patterns:
                raise InterfaceBlocked('recorded replay refuses mobility control')

        def updateUAVMobilityPatterns(self, patterns):
            if patterns:
                raise InterfaceBlocked('recorded replay refuses mobility control')

    return AbsoluteReplayManager
