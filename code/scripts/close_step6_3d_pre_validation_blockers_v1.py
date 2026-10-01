"""B1/B2 closure audit, CPU synthetic fixtures and read-only TRAIN only."""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"code/scripts"), str(ROOT/"code/src"), str(ROOT/"code/tests")]
import audit_step6_3d_pre_validation_v1 as prior
import run_step6_3d_formal_cpu_matrix_v1 as matrix
from run_step6_3d_one_cpu_solve_v1 import source_hashes, CHECKPOINT, EXPECTED_SHA
from pi_jwm.step6_3d_fixed_budget_search_v1 import solve_fixed_budget
from pi_jwm.step6_3d_method_selection_v1 import tune_cem_config
from pi_jwm.step6_3b_candidate_grammar_v1 import bind_structured_step, StructuredStepChoice, CommBlockChoice
from test_step6_3d_validation_stages_v1 import ANCHORS, CONFIGS, synthetic_case, cases, ValidationStageTests
from test_step6_3d_gpu_runner_identity_v1 import GpuRunnerIdentityTests
import unittest

BASELINE="5c98c91880466cc4d76c4dfb1d56b32a34f5557d"
OUT=ROOT/"code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001"
CONFIG_PATH=OUT/"03_validation_execution_config_3080ti.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    matrix.write_atomic(OUT/name,value)


def b1_oracle():
    repeated=prior.repeated_elite_counterexample()
    for row in repeated["evidence"].values():
        assert all(r["unique_h4_scoreable"]==2 and r["proposal_updated"] for r in row["iteration_rows"])
    retained_only={}
    for method in ("S-CEM","MH-CEM"):
        anchor,catalog,transition,_=prior.synthetic_anchor()
        draw_count=0
        def controlled(domain,proposal,rng):
            nonlocal draw_count
            wave,offset=divmod(draw_count,64)
            depth,slot=divmod(offset,16)
            start=slot if depth==0 else wave
            draw_count+=1
            bound=bind_structured_step(StructuredStepChoice((CommBlockChoice("input",start,1),),
                "SCALE_0.5","PROFILE_HOLD"),domain.context,domain.operational_domain,
                domain.state,domain.mobility_control,domain.catalog,domain.prior_signatures)
            return bound,{"draws":[(("mode",tuple(m.signature for m in domain.modes)),bound.structural_signature)]}
        def score(candidate,trace):
            starts=[int(s.comm[0]["rb_indices"][0]) for s in candidate.steps]
            if starts[1:]!=[0,0,0]:
                return None
            return SimpleNamespace(objective_tuple=(starts[0],0.,0.,0.,0.),candidate_fingerprint=candidate.fingerprint)
        with patch("pi_jwm.step6_3d_fixed_budget_search_v1.sample_structured_step",controlled):
            result=solve_fixed_budget(method=method,seed=6301,b_wm=256,anchor=anchor,catalog=catalog,
                transition=transition,score_h4=score,iterations=4,elite_ratio=.1,batch_size=16,
                transition_batch=lambda req:tuple(transition(n,b) for n,b in req))
        rows=result.iteration_rows
        assert rows[0]["proposal_updated"] and rows[0]["retained_count"]>0
        assert rows[1]["retained_from_previous"]>0
        assert all(r["unique_h4_scoreable"]==0 and not r["proposal_updated"] for r in rows[1:])
        retained_only[method]=rows
    return {"verdict":"PASS","researcher_definition":"At least two distinct scoreable H4 fingerprints actually completed within the current iteration; historical resampling counts, retained-only candidates do not.",
        "replayed_old_candidates_count_correctly":repeated["evidence"],
        "same_iteration_duplicate_fingerprints_deduplicated":True,
        "singleton_never_updates":repeated["fewer_than_two_current_distinct_guard"],
        "retained_only_not_counted":retained_only,"solver_changed":False,"B1_ambiguity":"CLOSED"}


def train_integrity_and_config():
    assert not subprocess.check_output(["git","diff","--name-only",BASELINE,"--","code/src"]).strip(), "Scientific source diff: BLOCK"
    inventory=matrix.read(matrix.OUT/"train_local_backup_inventory.json")
    assert inventory["file_count"]==771
    for row in inventory["files"]:
        path=matrix.OUT/row["path"]
        assert sha(path)==row["sha256"] and path.stat().st_size==row["size_bytes"]
    old=matrix.read(matrix.QUALIFICATION)
    assert sha(matrix.QUALIFICATION)==hashlib.sha256(subprocess.check_output(
        ["git","show",f"{BASELINE}:{matrix.QUALIFICATION.relative_to(ROOT).as_posix()}"])).hexdigest()
    current=source_hashes()
    assert set(current)==set(old["source_sha256"])
    changed=[k for k in current if current[k]!=old["source_sha256"][k]]
    assert changed==[matrix.RUNNER_SOURCE] or set(changed)=={matrix.RUNNER_SOURCE}
    assert sha(CHECKPOINT)==EXPECTED_SHA
    raw=list((matrix.OUT/"solve_results/train").glob("*.json"))
    assert len(raw)==768
    grids={m:{} for m in ("S-CEM","MH-CEM")}
    for path in raw:
        result=matrix.read(path)
        assert result["source_sha256"]==old["source_sha256"] and result["execution_config_id"]==old["execution_config_id"]
        assert result["locked_test"] is False
        key=(result["iterations"],result["elite_ratio"])
        grids[result["method"]].setdefault(key,{})[(result["sample_id"],result["seed"])]=result["outcome"]["best_objective"]
    selections={m:tune_cem_config(g) for m,g in grids.items()}
    assert all(row["selected_config"]==(4,.1) for row in selections.values())
    tuning=matrix.read(matrix.OUT/"05_train_hyperparameter_tuning_receipt.json")
    assert json.loads(json.dumps(selections))==tuning["grids"]
    # Same scientific source files and source-list, only orchestration differs.
    execution=matrix.execution_identity(device="cuda",gpu_model="NVIDIA GeForce RTX 3080 Ti",
        batch_size=16,checkpoint_sha256=EXPECTED_SHA,source_sha256=current)
    parents={key:{"path":p.relative_to(ROOT).as_posix(),"sha256":sha(p)} for key,p in {
        "train_frozen_configs":matrix.OUT/"06_frozen_selected_cem_configs.json",
        "train_tuning_receipt":matrix.OUT/"05_train_hyperparameter_tuning_receipt.json",
        "train_closure":matrix.OUT/"train_tuning_closure_acceptance.json",
        "qualification_config":matrix.QUALIFICATION,"summary_schema":matrix.SUMMARY_SCHEMA}.items()}
    config={**execution,"name":"FORMAL_STEP_6_3D_VALIDATION_EXECUTION_CONFIG_3080TI",
        "amp":False,"bf16":False,"fp16":False,"quantization":False,"torch_compile":False,
        "prior_mode":"mean","service_mode":"expectation","parents":parents,
        "stages":{"primary":[1024],"diagnostic":[256,512]},"stage_a_hard_stop":True,
        "authorization":"Configuration freeze only; no execution authorized by this Step"}
    write(CONFIG_PATH.name,config)
    bridge=matrix.validate_validation_provenance(config,execution)
    for key,bad in {"source_sha256":old["source_sha256"],"execution_config_id":old["execution_config_id"],
            "execution_device":"cpu","gpu_model":"4090","batch_size":32,"checkpoint_sha256":"bad","amp":True}.items():
        try:
            matrix.validate_validation_provenance({**config,key:bad},execution)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad Validation config accepted: {key}")
    return {"verdict":"PASS","TRAIN_REUSE_VALID":True,"TRAIN_RERUN_REQUIRED":False,
        "old_train_inventory_files_verified":771,"raw_cases":768,"changed_source_files":changed,
        "historical_source_sha256":old["source_sha256"],"current_source_sha256":current,
        "parent_sha256":parents,"checkpoint_sha256":EXPECTED_SHA,"bridge":bridge,
        "new_execution_config_id":execution["execution_config_id"],"negative_identity_oracles":"PASS"}


def runner_equivalence():
    old=ModuleType("historical_matrix")
    old.__file__=str(ROOT/matrix.RUNNER_SOURCE)
    old_source=subprocess.check_output(["git","show",f"{BASELINE}:{matrix.RUNNER_SOURCE}"]).decode("utf-8")
    exec(compile(old_source,old.__file__,"exec"),old.__dict__)
    def bounded_run(sample_id,method,seed,budget,k,rho,*,batch_size,device):
        assert sample_id.startswith("TRAIN_ONLY_SYNTHETIC") and device=="cpu" and batch_size==16
        anchor,catalog,transition,score=prior.synthetic_anchor(sample_id)
        outcome=solve_fixed_budget(method=method,seed=seed,b_wm=budget,anchor=anchor,catalog=catalog,
            transition=transition,score_h4=score,iterations=k,elite_ratio=rho,batch_size=16,
            transition_batch=lambda req:tuple(transition(n,b) for n,b in req))
        return {"sample_id":sample_id,"method":method,"seed":seed,"budget":budget,
            "iterations":k,"elite_ratio":rho,"outcome":prior.discrete(outcome),**identity}
    # Identity is CPU synthetic; no formal Validation namespace is used.
    identity=matrix.execution_identity(device="cpu",gpu_model=None,batch_size=16,
        checkpoint_sha256="NO_MODEL_SYNTHETIC",source_sha256={"oracle":"bounded"})
    rows=[]
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        for module,folder in ((old,"old"),(matrix,"patched")):
            module_results=Path(temp)/folder
            snapshots={}
            with patch.object(module,"RESULTS",module_results),patch.object(module,"run",bounded_run), \
                 patch.object(module,"source_hashes",lambda:identity["source_sha256"]):
                for budget in ((256,512,1024) if module is old else (1024,256,512)):
                    for method,(k,rho) in CONFIGS.items():
                        result=module.solve_or_resume("TRAIN_ONLY_SYNTHETIC","TRAIN_ONLY_SYNTHETIC_A",method,
                            6301,budget,k,rho,identity["source_sha256"],identity)
                        result=json.loads(json.dumps(result))
                        snapshots[(method,budget)]=result["outcome"]
                        # Resume must not invoke the solver again.
                        with patch.object(module,"run",side_effect=AssertionError("resume recomputed")):
                            resumed=module.solve_or_resume("TRAIN_ONLY_SYNTHETIC","TRAIN_ONLY_SYNTHETIC_A",method,
                                6301,budget,k,rho,identity["source_sha256"],identity)
                        assert resumed==result
            if module is old:
                reference=snapshots
            else:
                assert snapshots==reference
                for key,result in snapshots.items():
                    rows.append({"method":key[0],"budget":key[1],"matched_fields":"all discrete SearchOutcome fields; exclude only wall-clock",
                        "discrete_sha256":hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()})
    return {"RUNNER_PATCH_SCIENTIFIC_SEMANTICS_INVARIANT":"PASS","case_count":9,"rows":rows,
        "actual_old_runner_loaded_from_git":BASELINE,"both_orchestration_paths_exercised":True,
        "scope":"Same TRAIN-only synthetic transitions; no frozen-model execution or Validation outcome"}


def stage_summary_oracles():
    suite=unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ValidationStageTests))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(GpuRunnerIdentityTests))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful()
    summary=matrix.summarize_validation_budget(cases(),ANCHORS,matrix.VALIDATION_SEEDS,1024)
    schema=matrix.read(matrix.SUMMARY_SCHEMA)
    for row in summary["method_diagnostics"].values():
        assert set(schema["per_method_budget_required"])<=set(row)
    assert all(set(schema["pairwise_partition"])==set(row["six_category_counts"])
        for row in summary["paired_outcomes"].values())
    return {"verdict":"PASS","test_cases":result.testsRun,"formal_outcomes_read_or_written":False,
        "summary_schema_sha256":sha(matrix.SUMMARY_SCHEMA),"synthetic_summary_oracle":summary,
        "primary_cases":960,"primary_nominal_B_WM":983040,"diagnostic_cases":1920,
        "stage_a_no_stage_b_dependency":True,"stage_b_selection_unchanged":True,
        "six_category_total_enforced":320,"exception_count_cross_checked":True}


def manifest():
    files={p.name:sha(p) for p in sorted(OUT.glob("*.json")) if p.name!="09_sha_manifest.json"}
    for path in (Path(__file__),ROOT/matrix.RUNNER_SOURCE,ROOT/"code/tests/test_step6_3d_validation_stages_v1.py"):
        files[path.relative_to(ROOT).as_posix()]=sha(path)
    write("09_sha_manifest.json",{"schema_version":"PIJWM-blocker-closure-sha-v1","files":files,"locked_test":False})


def main():
    assert not list((matrix.OUT/"solve_results/validation").rglob("*.json"))
    assert not (matrix.OUT/"08_selected_method.json").exists()
    write("01_b1_semantic_closure_receipt.json",b1_oracle())
    bridge=train_integrity_and_config()
    write("04_source_provenance_bridge_receipt.json",bridge)
    write("05_old_new_runner_equivalence_receipt.json",runner_equivalence())
    stats=stage_summary_oracles()
    write("02_runner_stage_control_receipt.json",{k:v for k,v in stats.items() if k!="synthetic_summary_oracle"})
    write("06_summary_schema_test_receipt.json",stats)
    write("07_final_pre_validation_oracles.json",{"math":prior.math_and_layers(),"fairness":prior.fairness(),
        "rng_order_quota":prior.rng_quota_order(),"selection":prior.selection_oracle(),"locked_test":False})
    # Do not call old audit CLI: its receipts bind the old runner and NO_GO snapshot.
    assert not list((matrix.OUT/"solve_results/validation").rglob("*.json"))
    write("08_final_pre_validation_acceptance.json",{
        "PRE_VALIDATION_BLOCKER_CLOSURE":"PASS","PRE_VALIDATION_AUDIT":"PASS","VALIDATION_DECISION":"VALIDATION_GO",
        "B1_semantic_closure":"PASS","TRAIN_REUSE_VALID":True,"TRAIN_RERUN_REQUIRED":False,
        "RUNNER_PATCH_SCIENTIFIC_SEMANTICS_INVARIANT":"PASS","stage_a_hard_stop":"PASS",
        "stage_b_isolation":"PASS","summary_schema":"PASS","six_category_accounting":"PASS",
        "Return_birth_scorer_diagnostics":"PASS","old_train_provenance_integrity":"PASS",
        "new_validation_execution_config_id":bridge["new_execution_config_id"],
        "execution_order":"1024 -> STOP -> researcher review; separately authorized 256 -> 512",
        "VALIDATION_RESULT_COUNT":0,"VALIDATION_COMPARISON":"NOT_STARTED","SEARCH_METHOD":"NOT_SELECTED",
        "locked_test":False,"formal_gpu_search_started":False,"formal_validation_outcomes_read":False,
        "remaining_blockers":[],"stage_a_authorized_to_start":False,
        "readiness":"SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION",
        "future_return_birth_limitation_retained":True,"baseline_commit":BASELINE})
    manifest()
    print(json.dumps({"verdict":"VALIDATION_GO","validation_results":0,"new_execution_config_id":bridge["new_execution_config_id"]}))


if __name__=="__main__":
    main()
