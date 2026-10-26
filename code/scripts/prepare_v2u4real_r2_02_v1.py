"""R2-02: pinned single-train-scene acquisition and read-only sample audit.

No clock conversion, point transformation, simulator execution or raw repair.
Reuses R2-01 identities and R1's strict PCD/YAML readers. Raw data stay ignored.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import urllib.parse
import urllib.request

import zstandard
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/src'))
from pi_jwm.v2u4real_interface_v1 import InterfaceBlocked, audit_dataset, UniqueKeyLoader
import yaml
from prepare_v2u4real_r2_01_v1 import archive_url, write_json

DATA = ROOT/'code/artifacts/data/v2u4real_r2_01'
OUT = ROOT/'code/artifacts/manifests/v2u4real_r2_02_20261026'
SCENE = '2025-07-18-12-53_2'
REVISION = '36f64423a639db7c5bbc8ac8575b1ee74d7b8e9a'
IDENTITIES = {
    '1': (1070684502, '34e03eb7a34c896aab55b44f32797a1e66a62386a5a7c4e58074abfb6741fa75'),
    '2': (172308593, '64ae659bca67144c8713d5a689ce310474328a3c4ffd7a6f35948358b72b61f3'),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def config():
    c = json.loads((ROOT/'code/scripts/configs/v2u4real_r2_01.json').read_text(encoding='utf-8'))
    if c['dataset_revision'] != REVISION or c['scene_id'] != SCENE or c['split'] != 'train':
        raise InterfaceBlocked('only the explicitly authorized immutable train scene is allowed')
    for agent, (size, sha) in IDENTITIES.items():
        spec = c['archives'][agent]
        if (spec['bytes'], spec['sha256_expected']) != (size, sha) or spec['path'] != f'train/{SCENE}__agent-{agent}.tar.zst':
            raise InterfaceBlocked('archive identity changed')
        archive_url(c, agent)
    return c


def archive_path(c, agent):
    return DATA/'full_archives'/PurePosixPath(c['archives'][agent]['path']).name


def verify(path, spec):
    size = path.stat().st_size
    if size != spec['bytes']:
        raise InterfaceBlocked(f'archive size mismatch: {size} != {spec["bytes"]}')
    actual = digest(path)
    if actual != spec['sha256_expected']:
        raise InterfaceBlocked(f'archive SHA-256 mismatch: {actual}; do not use or extract')
    return {'status': 'PASS', 'path': path.relative_to(ROOT).as_posix(), 'bytes': size,
            'sha256': actual, 'sha256_expected': spec['sha256_expected']}


def download(c, agent):
    """Resume only this fixed archive; exact ranges checked before body writes."""
    target = archive_path(c, agent)
    target.parent.mkdir(parents=True, exist_ok=True)
    spec = c['archives'][agent]
    url = archive_url(c, agent)
    receipt_path = OUT/f'archive_agent_{agent}.json'
    receipt = {'dataset_revision': REVISION, 'official_url': url, 'expected_bytes': spec['bytes'],
               'sha256_expected': spec['sha256_expected'], 'requests': [], 'status': 'BLOCKED'}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt['official_url'] != url:
            raise InterfaceBlocked('receipt URL mismatch')
    if target.exists():
        receipt.update(verify(target, spec))
        if 'error' in receipt:
            receipt.setdefault('prior_errors', []).append(receipt.pop('error'))
        write_json(receipt_path, receipt)
        return receipt
    partial = target.with_suffix(target.suffix+'.part')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > spec['bytes']:
        raise InterfaceBlocked('oversized partial archive')
    if shutil.disk_usage(target.parent).free < spec['bytes']-offset+256*1024*1024:
        raise InterfaceBlocked('insufficient free space for archive plus reserve')
    # Reuse R2-01 prefix only after matching its original HTTPS receipt.
    if offset == 0:
        prefix = DATA/'prefixes'/f'agent-{agent}.first1m.zst.part'
        old_receipt = ROOT/f'code/artifacts/manifests/v2u4real_r2_01_20261026/official_sources/archive_prefix_agent_{agent}.json'
        if prefix.exists() and old_receipt.exists():
            old = json.loads(old_receipt.read_text(encoding='utf-8'))
            if old.get('official_url', old.get('url')) != url or old['sha256'] != digest(prefix):
                raise InterfaceBlocked('existing prefix does not match official receipt')
            with partial.open('xb') as f:
                f.write(prefix.read_bytes())
            offset = partial.stat().st_size
            receipt['reused_prefix_bytes'] = offset
    try:
        while offset < spec['bytes']:
            end = min(offset+16*1024*1024, spec['bytes'])-1
            # Query identifies the range to avoid CDN cache collisions; immutable revision unchanged.
            request = urllib.request.Request(url+f'?download=true&r2range={offset}-{end}', headers={
                'Range': f'bytes={offset}-{end}', 'User-Agent': 'PIJWM-R2-02-pinned-single-scene'})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{spec["bytes"]}':
                    raise InterfaceBlocked('server did not honor exact range; body not read')
                length = response.headers.get('Content-Length')
                if length is not None and int(length) != end-offset+1:
                    raise InterfaceBlocked('range length mismatch')
                start = offset
                with partial.open('ab') as f:
                    remaining = end-offset+1
                    while remaining:
                        block = response.read(min(1024*1024, remaining))
                        if not block:
                            raise InterfaceBlocked('truncated HTTP range; partial preserved')
                        f.write(block)
                        remaining -= len(block)
                        offset += len(block)
                receipt['requests'].append({'start': start, 'end': end, 'status': response.status,
                    'content_range': response.headers.get('Content-Range'),
                    'final_host': urllib.parse.urlsplit(response.url).hostname,
                    'completed_utc': datetime.now(timezone.utc).isoformat()})
            receipt['downloaded_bytes_on_disk'] = offset
            write_json(receipt_path, receipt)
            print(f'agent {agent}: {offset}/{spec["bytes"]} bytes', flush=True)
        result = verify(partial, spec)
        partial.rename(target)
        result['path'] = target.relative_to(ROOT).as_posix()
        receipt.update(result)
        if 'error' in receipt:
            receipt.setdefault('prior_errors', []).append(receipt.pop('error'))
        receipt['verified_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(receipt_path, receipt)
        return receipt
    except Exception as exc:
        receipt.update(status='BLOCKED', error=f'{type(exc).__name__}: {exc}',
                       downloaded_bytes_on_disk=partial.stat().st_size if partial.exists() else 0)
        write_json(receipt_path, receipt)
        raise


def safe_member(member, agent):
    p = PurePosixPath(member.name)
    if p.is_absolute() or '..' in p.parts or '\\' in member.name or p.parts[:3] != ('train', SCENE, agent):
        raise InterfaceBlocked(f'unsafe or out-of-scope member: {member.name}')
    if not (member.isdir() or member.isfile()):
        raise InterfaceBlocked(f'unsupported tar link/special member: {member.name}')
    return p


def inventory(c, agent):
    path = archive_path(c, agent)
    identity = verify(path, c['archives'][agent])
    rows = []
    with path.open('rb') as f, zstandard.ZstdDecompressor().stream_reader(f) as stream:
        with tarfile.open(fileobj=stream, mode='r|') as archive:
            for member in archive:
                safe_member(member, agent)
                rows.append({'name': member.name, 'bytes': member.size,
                             'type': 'directory' if member.isdir() else 'file'})
    counts = Counter('/'.join(PurePosixPath(r['name']).parts[3:-1]) for r in rows if r['type'] == 'file')
    result = {'archive': identity, 'agent': agent, 'members': rows,
              'member_count': len(rows), 'file_counts_by_subdirectory': dict(sorted(counts.items())),
              'duplicate_names': [k for k,n in Counter(r['name'] for r in rows).items() if n>1],
              'stream_inventory': 'PASS'}
    write_json(OUT/f'members_agent_{agent}.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'members'}, ensure_ascii=False), flush=True)
    return result


def choose_samples(inventories, count=16):
    """Choose published common file keys, without claiming physical synchronization."""
    if not 3 <= count <= 16:
        raise InterfaceBlocked('extract between 3 and 16 candidate file keys only')
    sequences = {}
    for agent in ('1','2'):
        inv = inventories[agent]
        if inv['duplicate_names']:
            raise InterfaceBlocked('duplicate archive members')
        for kind, directory, suffix in [('pcd', 'ouster', '.pcd'), ('yaml', 'yaml/ouster', '.yaml')]:
            prefix = f'train/{SCENE}/{agent}/{directory}/'
            sequences[f'{agent}.{kind}'] = sorted(
                PurePosixPath(r['name']).stem for r in inv['members']
                if r['type'] == 'file' and r['name'].startswith(prefix)
                and '/' not in r['name'][len(prefix):] and r['name'].endswith(suffix)
                and 'additional' not in r['name'])
    common = sorted(set.intersection(*(set(v) for v in sequences.values())))
    selected = common[:count]
    if len(selected) < 3:
        raise InterfaceBlocked('fewer than three complete common published file keys')
    ranks = {k: [v.index(x) for x in selected] for k,v in sequences.items()}
    continuous = all(all(y == x+1 for x,y in zip(v,v[1:])) for v in ranks.values())
    return {'keys': selected, 'sequences': sequences, 'selected_sequence_indices': ranks,
            'continuous_in_each_published_sequence': continuous,
            'common_file_key_count': len(common), 'physical_pairing_verified': False,
            'selection_basis': 'Earliest lexically sorted common PCD/YAML file keys; physical clock not inferred'}


def extract_selected(c, selection):
    output = DATA/'selected_samples'
    extracted = []
    for agent in ('1','2'):
        path = archive_path(c, agent)
        verify(path, c['archives'][agent])
        wanted = {f'train/{SCENE}/{agent}/{directory}/{key}{suffix}'
                  for key in selection['keys'] for directory,suffix in [('ouster','.pcd'),('yaml/ouster','.yaml')]}
        seen = set()
        with path.open('rb') as f, zstandard.ZstdDecompressor().stream_reader(f) as stream:
            with tarfile.open(fileobj=stream, mode='r|') as archive:
                for member in archive:
                    p = safe_member(member, agent)
                    if member.name not in wanted:
                        continue
                    if member.name in seen or not member.isfile() or not 0 < member.size <= 16*1024*1024:
                        raise InterfaceBlocked('duplicate, non-file or oversized selected member')
                    seen.add(member.name)
                    target = output.joinpath(*p.parts)
                    if not target.resolve().is_relative_to(output.resolve()):
                        raise InterfaceBlocked('sample path escape')
                    payload = archive.extractfile(member).read(member.size+1)
                    if len(payload) != member.size:
                        raise InterfaceBlocked('incomplete selected member')
                    sha = hashlib.sha256(payload).hexdigest()
                    if target.exists():
                        if target.stat().st_size != member.size or digest(target) != sha:
                            raise InterfaceBlocked('existing raw sample differs; no overwrite')
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open('xb') as dest:
                            dest.write(payload)
                    extracted.append({'agent': agent, 'member': member.name, 'bytes': member.size,
                                      'sha256': sha, 'path': target.relative_to(ROOT).as_posix(),
                                      'raw_bytes_unchanged': True})
        if seen != wanted:
            raise InterfaceBlocked(f'missing selected members: {sorted(wanted-seen)}')
    write_json(OUT/'extraction.json', {'selection': selection, 'files': extracted,
               'file_count': len(extracted), 'total_bytes': sum(x['bytes'] for x in extracted),
               'other_sensor_or_full_scene_extraction': False})
    print(f'extracted {len(extracted)} unchanged files for {len(selection["keys"])} candidate keys', flush=True)
    return extracted


def numeric_field(value, shape, name):
    """Structural checks only; no axis, unit or matrix direction inference."""
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in 'ifu' or not np.isfinite(array).all():
        raise InterfaceBlocked(f'{name}: finite numeric shape {shape} required')
    return {'shape': list(array.shape), 'numpy_dtype': str(array.dtype),
            'python_type': type(value).__name__, 'values': value}


def raw_time_fields(value, prefix=''):
    fields = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f'{prefix}.{key}'.lstrip('.')
            if re.search(r'time|stamp', str(key), re.I):
                fields[path] = {'value': item, 'type': type(item).__name__, 'unit': None, 'origin': None}
            fields.update(raw_time_fields(item, path))
    return fields


def inspect_metadata(path, agent):
    value = yaml.load(path.read_text(encoding='utf-8-sig'), Loader=UniqueKeyLoader)
    if not isinstance(value, dict):
        raise InterfaceBlocked('YAML root must be a mapping')
    pose = numeric_field(value['lidar_pose'], (6,), 'lidar_pose')
    speed = numeric_field(value['ego_speed'], (), 'ego_speed')
    cameras = {}
    for key, item in value.items():
        if re.fullmatch(r'camera\d+', str(key)):
            cameras[key] = {field: numeric_field(item[field], shape, f'{key}.{field}')
                            for field, shape in [('extrinsic', (4,4)), ('intrinsic', (3,3))]}
            matrix = np.asarray(item['extrinsic'])
            cameras[key]['extrinsic_structure_diagnostics'] = {
                'bottom_row': matrix[3].tolist(),
                'rotation_determinant': float(np.linalg.det(matrix[:3,:3])),
                'orthogonality_max_abs_error': float(np.max(np.abs(matrix[:3,:3].T @ matrix[:3,:3]-np.eye(3)))),
                'source_frame': None, 'target_frame': None, 'direction_verified': False}
    if not cameras or not isinstance(value['vehicles'], dict):
        raise InterfaceBlocked('camera calibration or vehicles mapping missing')
    centers = set()
    examples = []
    for object_id, obj in value['vehicles'].items():
        for name in ('angle','center','extent','location'):
            numeric_field(obj[name], (3,), f'vehicles.{object_id}.{name}')
        if any(x <= 0 for x in obj['extent']):
            raise InterfaceBlocked('nonpositive annotation extent')
        numeric_field(obj['speed'], (), f'vehicles.{object_id}.speed')
        if not isinstance(obj['obj_type'], str):
            raise InterfaceBlocked('annotation class must be a string')
        centers.add(tuple(obj['center']))
        if len(examples) < 1:
            examples.append({'id': object_id, **obj})
    return {'agent': agent, 'status': 'PASS', 'root_keys': list(value),
            'raw_time_fields': raw_time_fields(value), 'lidar_pose': pose, 'ego_speed': speed,
            'cameras': cameras, 'annotation_count': len(value['vehicles']),
            'annotation_center_values': [list(x) for x in sorted(centers)],
            'annotation_examples': examples,
            'dedicated_body_extrinsic_or_registration_matrix': None,
            'physical_semantics_status': 'BLOCKED'}


def source_identity():
    rows = []
    for manifest in ('V2U4Real.SOURCE.json','V2U4Real.R2_SOURCES.json'):
        source = json.loads((ROOT/'code/reference'/manifest).read_text(encoding='utf-8'))
        for name, spec in source['files'].items():
            expected = spec if isinstance(spec, str) else spec['sha256']
            actual = digest(ROOT/'code/reference/V2U4Real'/name)
            if actual != expected:
                raise InterfaceBlocked(f'official source snapshot changed: {name}')
            rows.append({'path': name, 'sha256': actual, 'status': 'PASS', 'commit': source['commit']})
    result = {'official_source_files': rows,
              'executed_sources': {name: digest(ROOT/name) for name in [
                  'code/scripts/prepare_v2u4real_r2_02_v1.py',
                  'code/scripts/prepare_v2u4real_r2_01_v1.py',
                  'code/scripts/configs/v2u4real_r2_01.json',
                  'code/tests/test_v2u4real_full_sample_v1.py']},
              'runtime': {'python': sys.version, 'numpy': np.__version__,
                          'pyyaml': yaml.__version__, 'zstandard': zstandard.__version__},
              'pi_jwm_reader_sha256': digest(ROOT/'code/src/pi_jwm/v2u4real_interface_v1.py'),
              'r2_01_manifest_sha256': digest(ROOT/'code/artifacts/manifests/v2u4real_r2_01_20261026/sample_manifest.json')}
    write_json(OUT/'source_identity.json', result)
    return result


def acceptance_counts(rows):
    # Readable files and adjacent published keys cannot qualify physical pairs.
    qualified = sum(row['read_status'] == 'PASS' and row['continuous_in_published_sequences']
                    and not row.get('duplicate_payload_paths', [])
                    and row['physical_pairing_status'] == 'PASS'
                    and row['calibration_semantics_status'] == 'PASS' for row in rows)
    return {'candidate_file_pairs': len(rows),
            'readable_file_pairs': sum(row['read_status'] == 'PASS' for row in rows),
            'qualified_complete_pairs': qualified,
            'status': 'PASS' if qualified >= 3 else 'BLOCKED'}


def audit_selected(c):
    archives = {a:verify(archive_path(c,a), c['archives'][a]) for a in ('1','2')}
    source_identity()
    inventories = {a:json.loads((OUT/f'members_agent_{a}.json').read_text(encoding='utf-8')) for a in ('1','2')}
    selection = choose_samples(inventories)
    extraction = json.loads((OUT/'extraction.json').read_text(encoding='utf-8'))
    if extraction['selection'] != selection:
        raise InterfaceBlocked('extraction selection changed')
    records = {x['member']:x for x in extraction['files']}
    expected = {f'train/{SCENE}/{a}/{directory}/{key}{suffix}'
                for a in ('1','2') for key in selection['keys']
                for directory,suffix in [('ouster','.pcd'),('yaml/ouster','.yaml')]}
    if len(records) != len(extraction['files']) or set(records) != expected:
        raise InterfaceBlocked('extraction manifest missing/duplicate/unexpected files')
    for member, record in records.items():
        path = DATA/'selected_samples'/member
        if path.relative_to(ROOT).as_posix() != record['path'] or path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise InterfaceBlocked(f'raw sample changed: {member}')
    audit = audit_dataset(DATA/'selected_samples', max_frames=16)
    write_json(OUT/'r1_strict_data_audit.json', audit)
    reads = {(row['agent'],row['frame_key']): row for row in audit['frames']}
    metadata_rows, sample_rows = [], []
    for key in selection['keys']:
        row = {'dataset_version': REVISION, 'split': 'train', 'scene_id': SCENE,
               'vehicle_frame_id': key, 'uav_frame_id': key, 'sensor': 'ouster',
               'timestamp_raw': {'vehicle': {}, 'uav': {}}, 'timestamp_unit': None,
               'timestamp_reference_origin': None, 'file_key': key,
               'pairing_evidence': {
                   'file_association': 'Official BaseDataset uses the same YAML stem to retrieve each agent PCD/YAML',
                   'collection_rule': 'Author preprint C.3: nearest-time pairing; reported offset within 20 ms',
                   'sample_acquisition_time_offsets': None,
                   'limitation': 'Same published key is insufficient to verify physical acquisition pairing'},
               'calibration_reference': 'Original YAML lidar_pose and camera*.extrinsic/intrinsic; official source fixed e099a3a',
               'continuous_in_published_sequences': selection['continuous_in_each_published_sequence'],
               'physical_continuity_status': 'BLOCKED', 'physical_pairing_status': 'BLOCKED',
               'calibration_semantics_status': 'BLOCKED', 'read_results': {},
               'missing_fields': ['physical acquisition times with unit/origin and per-agent offset',
                                  'scene world origin/geographic reference',
                                  'explicit LiDAR/body/antenna extrinsics and registration correction provenance',
                                  'sample-bound camera extrinsic direction and annotation axis contract'],
               'sample_status': 'BLOCKED', 'qualified_complete_pair': False}
        for agent, role in [('1','vehicle'),('2','uav')]:
            for directory,suffix,label in [('ouster','.pcd','lidar'),('yaml/ouster','.yaml','metadata')]:
                record = records[f'train/{SCENE}/{agent}/{directory}/{key}{suffix}']
                row[f'{role}_{label}_path'] = record['path']
                row[f'{role}_{label}_sha256'] = record['sha256']
            read = reads[(agent,key)]
            detail = {'agent': agent, 'frame_key': key}
            try:
                detail.update(inspect_metadata(ROOT/row[f'{role}_metadata_path'], agent))
            except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
                detail.update(status='BLOCKED', error=str(exc))
            metadata_rows.append(detail)
            row['timestamp_raw'][role] = detail.get('raw_time_fields', {})
            row['read_results'][role] = {'pcd_yaml_reader_status': read['read_status'],
                                         'metadata_structure_status': detail['status'],
                                         'point_count': read.get('point_count')}
        row['read_status'] = 'PASS' if all(v['pcd_yaml_reader_status'] == v['metadata_structure_status'] == 'PASS'
                                         for v in row['read_results'].values()) else 'BLOCKED'
        sample_rows.append(row)
    groups = {}
    for x in extraction['files']:
        groups.setdefault(x['sha256'], []).append(x['path'])
    duplicates = [v for v in groups.values() if len(v)>1]
    for row in sample_rows:
        paths = {row[f'{role}_{kind}_path'] for role in ('vehicle','uav') for kind in ('lidar','metadata')}
        row['duplicate_payload_paths'] = [group for group in duplicates if paths.intersection(group)]
    time_audit = {'status': 'BLOCKED', 'raw_time_fields_by_file': [
        {'agent': x['agent'], 'frame_key': x['frame_key'], 'fields': x.get('raw_time_fields')} for x in metadata_rows],
        'actual_interval_s': None, 'actual_time_range_s': None, 'timestamp_unit': None, 'reference_origin': None,
        'file_key_difference': {'unit': 'identifier_numeric_only',
            'selected_values': [str(Decimal(y)-Decimal(x)) for x,y in zip(selection['keys'],selection['keys'][1:])],
            'full_sequences': {k:dict(Counter(str(Decimal(y)-Decimal(x)) for x,y in zip(v,v[1:])))
                               for k,v in selection['sequences'].items()}},
        'published_key_continuity': selection['continuous_in_each_published_sequence'],
        'physical_missing_or_duplicate_frames': 'UNVERIFIED without acquisition clocks',
        'simulation_time_mapping': 'NOT_RUN'}
    write_json(OUT/'metadata_field_audit.json', {'files': metadata_rows, 'point_transform_executed': False})
    write_json(OUT/'time_field_audit.json', time_audit)
    counts = acceptance_counts(sample_rows)
    result = {'schema_version': 'V2U4Real-R2-02-v1', 'dataset': c['dataset'], 'dataset_version': REVISION,
              'split': 'train', 'scene_id': SCENE, 'archives': archives, **counts,
              'r2_01_original_manifest': 'code/artifacts/manifests/v2u4real_r2_01_20261026/sample_manifest.json',
              'extracted_file_count': len(records), 'extracted_total_bytes': extraction['total_bytes'],
              'duplicate_payload_groups': duplicates, 'samples': sample_rows,
              'blockers': sample_rows[0]['missing_fields'],
              'anomalies': ['Two groups of byte-identical vehicle Ouster PCD payloads at distinct published keys; acquisition cause unknown'] if duplicates else [],
              'later_experiments': {k:'NOT_RUN' for k in ['clock_mapping','coordinate_transform','AirFogSim','radio','feature_service','training']},
              'gpu_used': False, 'locked_test_accessed': False}
    write_json(OUT/'sample_manifest.json', result)
    write_json(OUT/'acceptance.json', {
        'step': 'R2-02', 'overall_status': counts['status'],
        'archive_integrity': 'PASS', 'published_file_continuity': 'PASS' if selection['continuous_in_each_published_sequence'] else 'BLOCKED',
        'pcd_yaml_read_and_structure': 'PASS' if counts['readable_file_pairs'] == len(sample_rows) else 'BLOCKED',
        'numeric_pose_and_camera_calibration_acquired': 'PASS' if all(x['status'] == 'PASS' for x in metadata_rows) else 'BLOCKED',
        'physical_clock_and_pairing': 'BLOCKED', 'calibration_coordinate_semantics': 'BLOCKED',
        **counts, 'duplicate_payload_group_count': len(duplicates),
        'sample_manifest': 'code/artifacts/manifests/v2u4real_r2_02_20261026/sample_manifest.json',
        'later_experiments': result['later_experiments'], 'gpu_used': False, 'locked_test_accessed': False})
    print(json.dumps({k:result[k] for k in ['status','candidate_file_pairs','readable_file_pairs',
          'qualified_complete_pairs','extracted_file_count','extracted_total_bytes','duplicate_payload_groups']},ensure_ascii=False))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['download', 'inventory', 'extract', 'audit'])
    args = p.parse_args()
    c = config()
    if args.action == 'audit':
        result = audit_selected(c)
        return 0 if result['status'] == 'PASS' else 2
    elif args.action == 'extract':
        inventories = {a:json.loads((OUT/f'members_agent_{a}.json').read_text(encoding='utf-8')) for a in ('1','2')}
        selection = choose_samples(inventories)
        write_json(OUT/'selection.json', selection)
        extract_selected(c, selection)
    else:
        for agent in ('1','2'):
            if args.action == 'download': download(c, agent)
            else: inventory(c, agent)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
