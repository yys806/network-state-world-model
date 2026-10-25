"""Exercise the pinned official transform functions on artificial CPU inputs.

Uses NumPy/SciPy only, not OpenCOOD training/inference imports or GPU code.
This does not determine the coordinate axes of any real sample.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/src'))
from pi_jwm.v2u4real_interface_v1 import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'code/artifacts/audit/v2u4real_airfogsim_round1_20261025/official_transform_probe.json')
    args=parser.parse_args()
    path=ROOT/'code/reference/V2U4Real/opencood/utils/transformation_utils.py'
    manifest=json.loads((ROOT/'code/reference/V2U4Real.SOURCE.json').read_text(encoding='utf8'))
    if sha256(path)!=manifest['files']['opencood/utils/transformation_utils.py']:
        raise ValueError('official source fingerprint mismatch')
    spec=importlib.util.spec_from_file_location('_pijwm_official_v2u_transform_probe',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pose=[10,20,30,0,90,0]
    t=module.x_to_world(pose)
    expected=np.array([[0,-1,0,10],[1,0,0,20],[0,0,1,30],[0,0,0,1]],float)
    np.testing.assert_allclose(t,expected,atol=1e-12)
    np.testing.assert_allclose(module.x1_to_x2(pose,[0,0,0,0,0,0]),t)
    np.testing.assert_allclose(module.x1_to_x2([0,0,0,0,0,0],pose)@t,np.eye(4),atol=1e-12)
    matrix=np.eye(4);matrix[:3,3]=[4,5,6]
    actual=module.x1_to_x2([0,0,0,0,0,0],matrix)
    err=float(np.max(np.abs(actual-np.linalg.inv(matrix))))
    if err!=12.:
        raise AssertionError('mixed branch changed; re-audit source semantics')
    result={'scope':'official_function_synthetic_inputs_only','source_sha256':sha256(path),
            'pose_order':'x,y,z,roll,yaw,pitch','degree_default':True,
            'rotation':'intrinsic ZYX; comment NED, physical dataset axes UNVERIFIED',
            'known_rotation_matrix':t.tolist(),'pose_pose_branch':'PASS',
            'mixed_list_matrix_branch':{'status':'BLOCKED','max_error':err,
              'reason':'multiplies x2 directly where docstring says x2->world; verify call-specific annotation convention'},
            'dataset_transform_status':'BLOCKED','python':sys.version}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({'pose_pose_branch':'PASS','mixed_branch_error':err,'dataset_transform_status':'BLOCKED'}))
    return 0


if __name__=='__main__':raise SystemExit(main())
