"""Tamper tests on copies; never modify accepted raw evidence."""
import contextlib,io,json,tempfile,shutil,unittest
from pathlib import Path
from unittest.mock import patch
import audit_step6_4n_acceptance_v2 as audit

class IndependentAcceptanceTests(unittest.TestCase):
    def test_modified_score_and_recursive_fingerprint_are_blocked(self):
        for kind in ('score','recursive'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                target=Path(directory)/'wm';target.mkdir()
                for p in audit.WM.iterdir():
                    if p.is_file() and p.suffix in ('.json','.pt'):shutil.copy2(p,target/p.name)
                path=target/('scorer_result.json' if kind=='score' else 'B_ROUTE_RB_wm.json')
                j=json.loads(path.read_text(encoding='utf8'))
                if kind=='score':j['scores'][0]['objective'][3]=99.
                else:j['inputs'][2]['state']='tampered'
                path.write_text(json.dumps(j),encoding='utf8')
                with patch.object(audit,'WM',target),contextlib.redirect_stdout(io.StringIO()):self.assertEqual(audit.main(),1)
                self.assertEqual(json.loads((target/'acceptance_v2.json').read_text(encoding='utf8'))['status'],'BLOCKED')

    def test_missing_primitive_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(audit,'WM',Path(directory)):
            with self.assertRaises(FileNotFoundError):audit.main()

if __name__=='__main__':unittest.main()
