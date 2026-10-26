"""Read-only R2-01 preparation audit, reusing R1's strict PCD/data readers.

Default: offline inspection of already acquired official prefixes. --fetch
requests only pinned train metadata and one 1-MiB prefix per archive. It cannot
download whole archives, validation/test samples, models, or run transforms.
Exit 2 denotes a documented incomplete preparation; 1 is an unexpected error.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import urllib.parse
import urllib.request

import zstandard

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/src'))
from pi_jwm.v2u4real_interface_v1 import InterfaceBlocked, audit_dataset, read_pcd_xyz, sha256


def validate_config(config):
    if not 0 < config['prefix_bytes_per_archive'] <= 1048576:
        raise InterfaceBlocked('hard 1-MiB prefix limit; full archive download is not supported')
    if not 0 < config['maximum_extracted_member_bytes'] <= 8388608:
        raise InterfaceBlocked('hard 8-MiB member limit')
    for key in ('data_root', 'audit_root'):
        if not (ROOT/config[key]).resolve().is_relative_to((ROOT/'code/artifacts').resolve()):
            raise InterfaceBlocked('all task data/output must stay inside project artifacts')
    for agent in ('1', '2'):
        archive_url(config, agent)
        if config['archives'][agent]['bytes'] <= config['prefix_bytes_per_archive']:
            raise InterfaceBlocked('refuse a request that would acquire a whole archive')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def archive_url(config, agent):
    name = config['archives'][agent]['path']
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or '?' in name or '#' in name:
        raise InterfaceBlocked('unsafe archive path')
    if len(p.parts) != 2 or p.parts[0] != 'train' or not name.endswith(f'__agent-{agent}.tar.zst'):
        raise InterfaceBlocked('only the pinned train archives are allowed')
    if config['dataset'] != 'VJiaLi/V2U4Real' or not re.fullmatch(r'[0-9a-f]{40}', config['dataset_revision']):
        raise InterfaceBlocked('official dataset and immutable revision required')
    return f"https://huggingface.co/datasets/{config['dataset']}/resolve/{config['dataset_revision']}/{name}"


def fetch_bounded(url, target, limit, expected_archive_size=None):
    """Refuse non-206 archive responses before reading. No auth or mirrors."""
    if urllib.parse.urlsplit(url).hostname != 'huggingface.co':
        raise InterfaceBlocked('initial URL must be the official Hugging Face host')
    headers = {'User-Agent': 'PIJWM-R2-01-bounded-preparation'}
    if expected_archive_size is not None:
        headers['Range'] = f'bytes=0-{limit-1}'
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=45) as response:
        if expected_archive_size is not None:
            expected = f'bytes 0-{limit-1}/{expected_archive_size}'
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                raise InterfaceBlocked('server did not honor the exact bounded Range; no archive body read')
        length = response.headers.get('Content-Length')
        if length is None or int(length) > limit:
            raise InterfaceBlocked('missing/oversized Content-Length; response not read')
        payload = response.read(int(length))
        if len(payload) != int(length):
            raise InterfaceBlocked('truncated HTTP response')
        receipt = {'url': url, 'http_status': response.status, 'content_range': response.headers.get('Content-Range'),
                   'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
                   'final_host': urllib.parse.urlsplit(response.url).hostname,
                   'accessed_at_utc': datetime.now(timezone.utc).isoformat()}
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != payload:
        raise InterfaceBlocked('existing file differs; refusing overwrite')
    if not target.exists():
        target.write_bytes(payload)
    return receipt


def fetch_inputs(config):
    validate_config(config)
    out = ROOT/config['audit_root']/'official_sources'
    catalog = out/'hf_train_pinned.body'
    if not catalog.exists():
        url = f"https://huggingface.co/api/datasets/{config['dataset']}/tree/{config['dataset_revision']}/train?limit=100"
        write_json(out/'hf_train_pinned.receipt.json', fetch_bounded(url, catalog, 131072))
    rows = json.loads(catalog.read_text(encoding='utf-8'))
    for agent, spec in config['archives'].items():
        row = next(x for x in rows if x['path'] == spec['path'])
        if row['size'] != spec['bytes'] or row['lfs']['oid'] != spec['sha256_expected']:
            raise InterfaceBlocked('pinned catalog and archive identity differ')
        target = ROOT/config['data_root']/'prefixes'/f'agent-{agent}.first1m.zst.part'
        if target.exists():
            if target.stat().st_size != config['prefix_bytes_per_archive']:
                raise InterfaceBlocked('existing prefix has unexpected size')
            continue
        receipt = fetch_bounded(archive_url(config, agent), target, config['prefix_bytes_per_archive'], spec['bytes'])
        receipt.update(whole_archive_downloaded=False, whole_archive_sha256_verified=False)
        write_json(out/f'archive_prefix_agent_{agent}.json', receipt)


def extract_first_pcd(prefix, output_root, config, agent):
    """Extract one complete member; no extractall, path traversal, or repair."""
    cap = config['prefix_bytes_per_archive']
    if prefix.stat().st_size > cap:
        raise InterfaceBlocked('prefix exceeds download cap')
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(prefix.read_bytes())) as stream:
        with tarfile.open(fileobj=stream, mode='r|') as archive:
            for number, member in enumerate(archive):
                if number > 16:
                    raise InterfaceBlocked('no first PCD found within bounded header inspection')
                p = PurePosixPath(member.name)
                if p.is_absolute() or '..' in p.parts or '\\' in member.name:
                    raise InterfaceBlocked('unsafe archive member')
                if member.issym() or member.islnk():
                    raise InterfaceBlocked('archive links not accepted')
                if member.isdir():
                    continue
                if not member.isfile() or member.size > config['maximum_extracted_member_bytes']:
                    raise InterfaceBlocked('unsupported/oversized archive member')
                if p.parts[:3] != ('train', config['scene_id'], agent) or p.suffix != '.pcd':
                    raise InterfaceBlocked('unexpected first member; do not infer an alternative sample')
                target = output_root.joinpath(*p.parts)
                if not target.resolve().is_relative_to(output_root.resolve()):
                    raise InterfaceBlocked('extracted path escapes data root')
                payload = archive.extractfile(member).read(member.size+1)
                if len(payload) != member.size:
                    raise InterfaceBlocked('incomplete PCD member in prefix')
                if target.exists() and target.read_bytes() != payload:
                    raise InterfaceBlocked('existing raw sample differs; refusing overwrite')
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(payload)
                return target, member.name
    raise InterfaceBlocked('no complete PCD member')


def inspect_real_pcd(path, member, agent):
    # Header-only inventory is separate from the existing XYZ parser. Do not
    # add a second PCD decoder or filter invalid observations into PASS.
    header = {}
    with path.open('rb') as f:
        for _ in range(64):
            line = f.readline(4096).decode('ascii').strip().split()
            if not line or line[0].startswith('#'): continue
            header[line[0]] = line[1:]
            if line[0] == 'DATA': break
        raw_body = f.read() if header.get('DATA') == ['ascii'] else b''
    row = {'agent': agent, 'archive_member': member, 'local_path': path.relative_to(ROOT).as_posix(),
           'bytes': path.stat().st_size, 'sha256': sha256(path), 'pcd_header': header,
           'metadata_available': False, 'sensor': PurePosixPath(member).parts[3],
           'frame_id': path.stem, 'whole_archive_sha256_verified': False}
    try:
        xyz, _ = read_pcd_xyz(path)
        row.update(status='PASS', point_count=len(xyz), xyz_min=xyz.min(0).tolist(), xyz_max=xyz.max(0).tolist())
    except (InterfaceBlocked, ValueError, KeyError, OSError) as exc:
        row.update(status='BLOCKED', error=str(exc))
        if raw_body:
            row['nonfinite_ascii_token_count'] = len(re.findall(rb'(?i)(?<!\S)[+-]?(?:nan|inf)(?!\S)', raw_body))
    return row


def run(config, output=None):
    validate_config(config)
    out = output or ROOT/config['audit_root']
    records = []
    missing_inputs = []
    receipts_verified = 0
    for agent in ('1', '2'):
        prefix = ROOT/config['data_root']/'prefixes'/f'agent-{agent}.first1m.zst.part'
        receipt_path = ROOT/config['audit_root']/'official_sources'/f'archive_prefix_agent_{agent}.json'
        if not prefix.is_file() or not receipt_path.is_file():
            missing_inputs.append(f'agent {agent}: bounded official prefix/retrieval receipt missing; use --fetch')
            continue
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('official_url', receipt.get('url')) != archive_url(config, agent) or receipt.get('sha256') != sha256(prefix):
            raise InterfaceBlocked('prefix and official retrieval receipt do not agree')
        receipts_verified += 1
        path, member = extract_first_pcd(prefix, ROOT/config['data_root']/'partial_real', config, agent)
        records.append(inspect_real_pcd(path, member, agent))
    write_json(out/'partial_real_pcd_audit.json', records)
    reused = audit_dataset(ROOT/config['data_root']/'partial_real', max_frames=config['preferred_complete_pairs'])
    write_json(out/'r1_strict_data_audit.json', reused)
    keys = sorted({r['frame_id'] for r in records})
    entries = []
    for key in keys:
        sides = {r['agent']: r for r in records if r['frame_id'] == key}
        entries.append({'dataset_version': config['dataset_revision'], 'split': 'train', 'scene_id': config['scene_id'],
                        'frame_id': key, 'timestamp_raw': None, 'timestamp_unit': None,
                        'vehicle_lidar_path': sides.get('1', {}).get('local_path'), 'vehicle_metadata_path': None,
                        'uav_lidar_path': sides.get('2', {}).get('local_path'), 'uav_metadata_path': None,
                        'vehicle_sensor': sides.get('1', {}).get('sensor'), 'uav_sensor': sides.get('2', {}).get('sensor'),
                        'pairing_evidence': 'Same filename key in official train archives; mixed sensors; not a verified synchronized pair',
                        'calibration_reference': 'Paper v1 sections 3.1/3.2 and owner issue #1; numerical sample calibration not obtained',
                        'sample_status': 'BLOCKED', 'qualified_complete_pair': False,
                        'missing': ['vehicle ouster PCD', 'vehicle YAML', 'UAV YAML', 'verified physical clock and calibration contract']})
    manifest = {'schema': 'PIJWM-V2U4Real-real-sample-preparation-v1', 'source_scope': 'complete members from two bounded prefixes only', 'records': entries,
                'requested_common_lidar': config['requested_common_lidar'], 'minimum_complete_pairs': config['minimum_complete_pairs'],
                'preferred_complete_pairs': config['preferred_complete_pairs'], 'qualified_complete_pairs': 0,
                'whole_archive_sha256_verification': 'NOT_RUN', 'raw_samples_committed': False}
    write_json(out/'sample_manifest.json', manifest)
    status = {'step': 'R2-01', 'overall_status': 'BLOCKED', 'official_train_archive_access': 'PASS' if receipts_verified == 2 else 'BLOCKED',
              'complete_raw_pcd_members_obtained': len(records), 'pcd_reads': {r['agent']: r['status'] for r in records},
              'minimum_complete_training_subset': 'BLOCKED', 'yaml_availability': 'BLOCKED', 'yaml_parse_execution': 'NOT_RUN',
              'physical_timestamp_and_calibration_contract': 'BLOCKED', 'r1_reader_reused': True,
              'coordinate_transform_acceptance': 'NOT_RUN', 'AirFogSim_replay': 'NOT_RUN', 'wireless_changes': 'NOT_RUN',
              'feature_or_compute_service': 'NOT_RUN', 'training': 'NOT_RUN', 'gpu_used': False, 'locked_test_accessed': False,
              'downloaded_prefix_bytes': sum((ROOT/config['data_root']/'prefixes'/f'agent-{a}.first1m.zst.part').stat().st_size for a in ('1','2') if (ROOT/config['data_root']/'prefixes'/f'agent-{a}.first1m.zst.part').is_file()),
              'missing_inputs': missing_inputs,
              'whole_archive_downloaded': False,
              'reason': ('No local official prefixes/receipts; use the bounded --fetch option' if missing_inputs else
                         'Only bounded archive prefixes were acquired: no 3 consecutive common-Ouster PCD/YAML pairs; no full archive/index or numerical metadata')}
    write_json(out/'acceptance.json', status)
    print(json.dumps(status, ensure_ascii=False))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'code/scripts/configs/v2u4real_r2_01.json')
    parser.add_argument('--fetch', action='store_true', help='bounded pinned train prefix/metadata fetch only')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.fetch: fetch_inputs(config)
    status = run(config)
    return 0 if status['overall_status'] == 'PASS' else 2


if __name__ == '__main__': raise SystemExit(main())
