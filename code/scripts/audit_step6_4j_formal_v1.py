"""Independent on-disk acceptance for a formal 6.4J Pilot attempt."""
import argparse, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code/scripts')]
from pi_jwm.step6_4j_pilot_v1 import audit_formal_result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--episodes',nargs=2,required=True)
    ap.add_argument('--execution-config-id',required=True)
    a=ap.parse_args()
    try:
        result=audit_formal_result(a.root,a.episodes,a.execution_config_id)
    except Exception as exc:
        result={'status':'BLOCKED','exit_code':1,'reason':type(exc).__name__+':'+str(exc)}
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return int(result.get('exit_code',1))

if __name__=='__main__':raise SystemExit(main())
