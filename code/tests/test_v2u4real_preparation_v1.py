"""Safety/reproducibility tests; fixtures here are explicitly synthetic."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import zstandard

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/scripts'))
sys.path.insert(0, str(ROOT/'code/src'))
import prepare_v2u4real_r2_01_v1 as prep
from pi_jwm.v2u4real_interface_v1 import InterfaceBlocked


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT/'.tmp')
        self.root = Path(self.tmp.name)
        self.config = json.loads((ROOT/'code/scripts/configs/v2u4real_r2_01.json').read_text(encoding='utf-8'))

    def tearDown(self): self.tmp.cleanup()

    def make_prefix(self, name, payload=b'PCD bytes', link=False, size=None):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w') as tar:
            member = tarfile.TarInfo(name)
            if link:
                member.type = tarfile.SYMTYPE; member.linkname = '/outside'
                tar.addfile(member)
            else:
                member.size = len(payload) if size is None else size
                tar.addfile(member, io.BytesIO(payload))
        p = self.root/'prefix.zst'
        p.write_bytes(zstandard.ZstdCompressor().compress(buf.getvalue()))
        return p

    def test_only_train_and_immutable_revision(self):
        self.assertIn('/train/', prep.archive_url(self.config, '1'))
        for path in ('test/x__agent-1.tar.zst', '../train/x__agent-1.tar.zst', '/train/x__agent-1.tar.zst'):
            c=deepcopy(self.config);c['archives']['1']['path']=path
            with self.assertRaises(InterfaceBlocked): prep.archive_url(c,'1')
        c=deepcopy(self.config);c['dataset_revision']='main'
        with self.assertRaises(InterfaceBlocked): prep.archive_url(c,'1')

    def test_hard_download_and_output_boundaries(self):
        prep.validate_config(self.config)
        c=deepcopy(self.config);c['prefix_bytes_per_archive']=2**30
        with self.assertRaises(InterfaceBlocked): prep.validate_config(c)
        c=deepcopy(self.config);c['data_root']='../outside'
        with self.assertRaises(InterfaceBlocked): prep.validate_config(c)
        c=deepcopy(self.config);c['archives']['1']['bytes']=100
        with self.assertRaises(InterfaceBlocked): prep.validate_config(c)

    def test_complete_member_is_idempotent_and_not_overwritten(self):
        name=f"train/{self.config['scene_id']}/1/m1/1752814590.2.pcd"
        p=self.make_prefix(name)
        target,member=prep.extract_first_pcd(p,self.root/'data',self.config,'1')
        self.assertEqual(target.read_bytes(),b'PCD bytes');self.assertEqual(member,name)
        prep.extract_first_pcd(p,self.root/'data',self.config,'1')
        target.write_bytes(b'existing observation')
        with self.assertRaises(InterfaceBlocked): prep.extract_first_pcd(p,self.root/'data',self.config,'1')

    def test_path_traversal_links_and_oversize_are_blocked(self):
        for name,link in [('../escape.pcd',False),(f"train/{self.config['scene_id']}/1/m1/a.pcd",True)]:
            p=self.make_prefix(name,link=link)
            with self.assertRaises(InterfaceBlocked):prep.extract_first_pcd(p,self.root/'data',self.config,'1')
        p=self.make_prefix(f"train/{self.config['scene_id']}/1/m1/a.pcd",payload=b'x'*2048)
        c=deepcopy(self.config);c['maximum_extracted_member_bytes']=1024
        with self.assertRaises(InterfaceBlocked):prep.extract_first_pcd(p,self.root/'data',c,'1')

    def test_ignored_range_does_not_read_response_body(self):
        class Response:
            status=200;headers={'Content-Length':'1070684502'};url='https://huggingface.co/file'
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def read(self,*args):raise AssertionError('body must not be read')
        with patch.object(prep.urllib.request,'urlopen',return_value=Response()):
            with self.assertRaises(InterfaceBlocked):
                prep.fetch_bounded(prep.archive_url(self.config,'1'),self.root/'raw',1048576,1070684502)

    def test_wrong_range_does_not_read_response_body(self):
        class Response:
            status=206;headers={'Content-Range':'bytes 0-1070684501/1070684502'};url='https://huggingface.co/file'
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def read(self,*args):raise AssertionError('body must not be read')
        with patch.object(prep.urllib.request,'urlopen',return_value=Response()):
            with self.assertRaises(InterfaceBlocked):
                prep.fetch_bounded(prep.archive_url(self.config,'1'),self.root/'raw',1048576,1070684502)

    def test_partial_compressed_member_is_not_a_successful_sample(self):
        name=f"train/{self.config['scene_id']}/1/m1/a.pcd"
        p=self.make_prefix(name,payload=bytes(range(256))*100)
        p.write_bytes(p.read_bytes()[:25])
        with self.assertRaises((InterfaceBlocked,tarfile.ReadError,zstandard.ZstdError)):
            prep.extract_first_pcd(p,self.root/'data',self.config,'1')

    def test_missing_raw_assets_stay_blocked_without_fake_pass(self):
        with patch.object(prep,'ROOT',self.root):
            result=prep.run(self.config)
        self.assertEqual(result['overall_status'],'BLOCKED')
        self.assertEqual(result['official_train_archive_access'],'BLOCKED')
        self.assertEqual(result['complete_raw_pcd_members_obtained'],0)
        self.assertEqual(result['coordinate_transform_acceptance'],'NOT_RUN')
        manifest=json.loads((self.root/self.config['audit_root']/'sample_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['records'],[])
        self.assertEqual(manifest['qualified_complete_pairs'],0)


if __name__=='__main__':unittest.main()
