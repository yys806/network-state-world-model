import sys,json,gzip,copy,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'code/scripts'),str(ROOT/'code/src')]
from step6_4i_live_planner_v1 import LiveSCEMPlanner,history_tensor
class ProviderTests(unittest.TestCase):
 def test_gpu_or_unapproved_formal_denied_before_loading_model(self):
  args=dict(model=None,encoder=None,stats=None,slot_template=None,catalog=None,protocol=None)
  with self.assertRaises(PermissionError):LiveSCEMPlanner(**args,device='cuda',engineering_only=True)
  with self.assertRaises(PermissionError):LiveSCEMPlanner(**args,device='cuda',engineering_only=False)
 def test_history_slot_parity_and_future_poison(self):
  import numpy as np
  from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
  from pi_jwm.step5_5_full_sharded_loader_v1 import _one
  p=ROOT/'code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1'
  tid='formal-v1-sim-2026092325-policy-2026092425'
  raw=json.loads(gzip.decompress((p/f'raw/{tid}.json.gz').read_bytes()));raw['decisions']=raw['decisions'][:4];raw['steps']=raw['steps'][:3]
  frozen=_one(load_flow_tensor_batch(p/f'packages/tensor/{tid}.npz'),2,92);stats=json.loads((p/'packages/normalization/stats.json').read_text())
  _,actual,_=history_tensor(raw,stats,frozen)
  poison=copy.deepcopy(raw);poison['future_action']=object();poison['target']=object()
  for d in poison['decisions']:d['internal_metadata']=object()
  _,other,_=history_tensor(poison,stats,frozen)
  for k,v in actual.items():
   if isinstance(v,np.ndarray) and not k.startswith(('target_','future_')):
    np.testing.assert_array_equal(v,frozen[k]);np.testing.assert_array_equal(v,other[k])
if __name__=='__main__':unittest.main()
