"""Independent audit for additive STEP 6.4J live-root diagnostics."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
NS=ROOT/"code/artifacts/audit/pi_jwm_step6_4j_live_root_capture_20261011_v4"
def main():
 rows=[]; errors=[]
 for p in sorted(NS.glob("*_final.json")):
  d=json.loads(p.read_text()); stages=d.get("stages",{}); root=d.get("root",{}); cs=d.get("candidates",[])
  if stages.get("root_capture")!="PASS" or stages.get("root_identity")!="LIVE_ROOT_CAPTURE_VALID": errors.append(f"{p.name}:root gate")
  if not root.get("fingerprints",{}).get("anchor"): errors.append(f"{p.name}:missing anchor fingerprint")
  transitions=0
  for c in cs:
   transitions += int(c.get("completed_horizons") or 0)
   if c.get("scoreable_h4") and not (c.get("scorer_status")=="SCOREABLE" and c.get("H_eff")==4): errors.append(f"{p.name}:score mismatch")
   if c.get("scorer_status")=="OBJECTIVE_UNSCOREABLE" and c.get("H_eff")==4: errors.append(f"{p.name}:H_eff mismatch")
  if transitions>32: errors.append(f"{p.name}:budget {transitions}")
  rows.append({"sample_id":d["sample_id"],"root_status":stages.get("root_identity"),"domain_count":root.get("domain_count"),"candidates":len(cs),"completed_h4":sum(int(c.get("completed_horizons") or 0)==4 for c in cs),"scoreable_h4":sum(bool(c.get("scoreable_h4")) for c in cs),"scorer_exceptions":sum("scorer_exception" in c for c in cs),"rollout_exceptions":sum("rollout_exception" in c for c in cs),"wm_transitions":transitions,"future_return_birth":sum("UNSUPPORTED_FUTURE_RETURN_BIRTH" in str(c.get("diagnostics",{})) for c in cs)})
 out={"schema":"PI-JWM-STEP-6.4J-LIVE-ROOT-AUDIT-v1","passed":not errors,"errors":errors,"rows":rows,"totals":{"roots":len(rows),"completed_h4":sum(r["completed_h4"] for r in rows),"scoreable_h4":sum(r["scoreable_h4"] for r in rows),"scorer_exceptions":sum(r["scorer_exceptions"] for r in rows),"rollout_exceptions":sum(r["rollout_exceptions"] for r in rows),"wm_transitions":sum(r["wm_transitions"] for r in rows)},"r39_exact_equivalence":"UNPROVEN","locked_test":False,"GPU":"NOT_USED"}
 (NS/"independent_audit.json").write_text(json.dumps(out,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8"); print(json.dumps(out,ensure_ascii=False,sort_keys=True)); return 0 if not errors else 1
if __name__=="__main__": raise SystemExit(main())
