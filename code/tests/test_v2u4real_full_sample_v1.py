"""R2-02 acquisition safety/acceptance tests; fixtures are not real-data evidence."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_v2u4real_r2_02_v1 as prep


def fake_inventory(agent, keys):
    return {'duplicate_names': [], 'members': [
        {'name': f'train/{prep.SCENE}/{agent}/{directory}/{key}{suffix}', 'type': 'file', 'bytes': 1}
        for key in keys for directory,suffix in [('ouster','.pcd'),('yaml/ouster','.yaml')]]}


class FullSampleTests(unittest.TestCase):
    def test_structural_success_never_qualifies_physical_pair(self):
        rows = [{'read_status': 'PASS', 'continuous_in_published_sequences': True,
                 'physical_pairing_status': 'BLOCKED', 'calibration_semantics_status': 'BLOCKED'} for _ in range(16)]
        result = prep.acceptance_counts(rows)
        self.assertEqual(result['readable_file_pairs'], 16)
        self.assertEqual(result['qualified_complete_pairs'], 0)
        self.assertEqual(result['status'], 'BLOCKED')
        for row in rows:
            row.update(physical_pairing_status='PASS', calibration_semantics_status='PASS',
                       duplicate_payload_paths=['duplicate'])
        self.assertEqual(prep.acceptance_counts(rows)['qualified_complete_pairs'], 0)

    def test_numeric_metadata_rejects_nonfinite_wrong_shapes_and_strings(self):
        for value in ([1,2], [1,2,float('nan')], [1,2,float('inf')], ['1','2','3']):
            with self.assertRaises(prep.InterfaceBlocked): prep.numeric_field(value, (3,), 'pose')
        self.assertEqual(prep.numeric_field([1,2,3], (3,), 'pose')['shape'], [3])

    def test_nested_clock_fields_preserve_values_without_inventing_units(self):
        value = {'sensor': {'timestamp': 12345}, 'lidar_pose': [0]*6}
        fields = prep.raw_time_fields(value)
        self.assertEqual(fields['sensor.timestamp']['value'], 12345)
        self.assertIsNone(fields['sensor.timestamp']['unit'])
        self.assertIsNone(fields['sensor.timestamp']['origin'])

    def test_fixed_scope_and_digest_before_use(self):
        c = prep.config()
        self.assertEqual(c['dataset_revision'], prep.REVISION)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'bad.zst'
            p.write_bytes(b'bad')
            spec = {'bytes': 3, 'sha256_expected': hashlib.sha256(b'yes').hexdigest()}
            with self.assertRaisesRegex(prep.InterfaceBlocked, 'SHA-256 mismatch'):
                prep.verify(p, spec)
            spec['bytes'] = 4
            with self.assertRaisesRegex(prep.InterfaceBlocked, 'size mismatch'):
                prep.verify(p, spec)

    def test_file_pairing_never_establishes_physical_pairing(self):
        inv = {a:fake_inventory(a, ['10.0','10.1','10.2','10.3']) for a in ('1','2')}
        selected = prep.choose_samples(inv, 3)
        self.assertTrue(selected['continuous_in_each_published_sequence'])
        self.assertFalse(selected['physical_pairing_verified'])
        self.assertEqual(selected['keys'], ['10.0','10.1','10.2'])

    def test_missing_frame_not_hidden_by_intersection(self):
        inv = {'1':fake_inventory('1', ['10.0','10.1','10.2','10.3']),
               '2':fake_inventory('2', ['10.0','10.2','10.3'])}
        selected = prep.choose_samples(inv, 3)
        self.assertFalse(selected['continuous_in_each_published_sequence'])

    def test_duplicate_members_and_unsafe_paths_rejected(self):
        inv = {a:fake_inventory(a, ['1','2','3']) for a in ('1','2')}
        inv['1']['duplicate_names'] = ['duplicate']
        with self.assertRaises(prep.InterfaceBlocked): prep.choose_samples(inv)
        for name in ('../escape', '/absolute', f'test/{prep.SCENE}/1/x', f'train/{prep.SCENE}/2/x'):
            with self.assertRaises(prep.InterfaceBlocked): prep.safe_member(tarfile.TarInfo(name), '1')
        member = tarfile.TarInfo(f'train/{prep.SCENE}/1/link')
        member.type = tarfile.SYMTYPE
        with self.assertRaises(prep.InterfaceBlocked): prep.safe_member(member, '1')

    def test_ignored_range_never_reads_body_or_writes_archive(self):
        class Response:
            status = 200
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, *args): raise AssertionError('body must not be read')
        with tempfile.TemporaryDirectory(dir=prep.ROOT/'.tmp') as d:
            base = Path(d)
            with patch.object(prep, 'DATA', base/'data'), patch.object(prep, 'OUT', base/'out'), \
                 patch.object(prep.urllib.request, 'urlopen', return_value=Response()):
                with self.assertRaisesRegex(prep.InterfaceBlocked, 'exact range'):
                    prep.download(prep.config(), '1')
                receipt = json.loads((base/'out/archive_agent_1.json').read_text(encoding='utf-8'))
                self.assertEqual(receipt['downloaded_bytes_on_disk'], 0)
                self.assertEqual(receipt['status'], 'BLOCKED')
                self.assertFalse(prep.archive_path(prep.config(), '1').exists())

    def test_extract_only_selected_raw_bytes_and_no_overwrite(self):
        c = prep.config()
        selection = {'keys': ['1','2','3']}
        with tempfile.TemporaryDirectory(dir=prep.ROOT/'.tmp') as d:
            base = Path(d)
            with patch.object(prep, 'DATA', base/'data'), patch.object(prep, 'OUT', base/'out'):
                for agent in ('1','2'):
                    tar = io.BytesIO()
                    with tarfile.open(fileobj=tar, mode='w') as f:
                        for directory,suffix in [('ouster','.pcd'), ('yaml/ouster','.yaml'), ('m1','.pcd')]:
                            for key in ['1','2','3','4']:
                                member = tarfile.TarInfo(f'train/{prep.SCENE}/{agent}/{directory}/{key}{suffix}')
                                member.size = 4
                                f.addfile(member, io.BytesIO(b'raw\n'))
                    path = prep.archive_path(c, agent)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(zstandard.ZstdCompressor().compress(tar.getvalue()))
                    c['archives'][agent].update(bytes=path.stat().st_size, sha256_expected=prep.digest(path))
                records = prep.extract_selected(c, selection)
                self.assertEqual(len(records), 12)
                self.assertEqual(len(list((base/'data/selected_samples').rglob('*.*'))), 12)
                (prep.ROOT/records[0]['path']).write_bytes(b'changed')
                with self.assertRaisesRegex(prep.InterfaceBlocked, 'no overwrite'):
                    prep.extract_selected(c, selection)


if __name__ == '__main__': unittest.main()
