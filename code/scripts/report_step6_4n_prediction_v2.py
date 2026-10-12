"""Compare supplied CPU predictions to independent native execution evidence."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
REAL=ROOT/'code/artifacts/audit/pi_jwm_step6_4n_20261012_v10'
WM=ROOT/'code/artifacts/audit/pi_jwm_step6_4n_20261012_wm_v5'
def main():
 import torch
 results={};task_errors=[]
 for name,mode in (('A_NOOP','V2_NOOP'),('B_ROUTE_RB','V2_ROUTE_RB')):
  tr=torch.load(WM/f'{name}_trace.pt',weights_only=False)['trace']
  observed=json.loads((WM/f'{name}_wm.json').read_text(encoding='utf8'))
  errors=[]
  for h,s in enumerate(tr.states,1):
   real=json.loads((REAL/mode/f'04_h{h}.json').read_text(encoding='utf8'))
   ri=38
   ch=next(r for r in real['raw']['channel_rows'] if r['source_id']=='UAV_0' and r['target_id']=='UAV_1')
   pred=float(s['csi'][0,ri,0]);truth=float(ch['channel_attenuation_db'][0])
   rate={}
   if name=='B_ROUTE_RB' and h==1:
    ev=next(r for r in real['transfers'] if r['task_id']=='Task_5')
    pr=float(tr.model_traces[h-1]['rule']['wireless']['actual_rate_mbps'][0,ri,0]);rr=sum(ev['rate_per_s'])
    rate={'predicted_expected_rate':pr,'real_sampled_rate':rr,'absolute_error':abs(pr-rr),'relative_error':abs(pr-rr)/rr,'comparison':'expected service vs one stochastic realization; no reliability threshold approved'}
   errors.append({'h':h,'task_id':'Task_5','relation':ri,'rb':0,'predicted_attenuation_db':pred,'real_attenuation_db':truth,'absolute_attenuation_error_db':abs(pred-truth),'rate':rate,'task_rule_alignment':observed['alignment'][h-1]})
  results[name]=errors
 report={'schema':'PI-JWM-STEP-6.4N-PREDICTION-ERROR-v2','cases':results,'PREDICTION_ALIGNMENT':'NOT_ESTABLISHED','RETRAIN_REQUIRED':'NOT_ESTABLISHED','scope':'one real root; Task_5 finishes in first slot, caps hide rate error; not accepted accuracy evidence','training_support':{'RB_start_width':[0,1],'RB_marginal':'TRAIN_OBSERVED','dynamic_route_comm_joint':'NEW_OPT_IN_SUPPORT_NOT_VALIDATED_TRAIN_DISTRIBUTION','zero_cold_start_dynamic_hidden':'NO_RETRAIN_STRUCTURE_COMPATIBLE_BUT_PREDICTION_RELIABILITY_UNESTABLISHED'},'rule_alignment':'Task_5 remaining/completion align over all eight candidate-horizons; not learned-rate reliability','future_truth_consumed_by_prediction':False,'next_minimal_validation':'researcher reviews unsaturated-service prediction audit proposal; no additional root selected by future success'}
 (WM/'prediction_error.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 print(json.dumps({'status':report['PREDICTION_ALIGNMENT'],'B_H1':results['B_ROUTE_RB'][0]},ensure_ascii=False))
if __name__=='__main__':main()
