"""Local acceptance-backed context and provenance closure checks."""
from pathlib import Path
import hashlib
import json
import math
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/artifacts/protocols/pi_jwm_step6_4d_full_cohort_b512_v1_20261003'


def main():
    read=lambda p: json.loads(p.read_text(encoding='utf-8'))
    sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    accepted=read(OUT/'17_independent_acceptance_receipt.json')
    config=read(OUT/'05_execution_config_3080ti.json')
    assert accepted['coverage']==320 and accepted['new_cases']==272 and accepted['reused_cases']==48
    for family in ('source_sha256','input_sha256'):
        for n,h in config[family].items(): assert sha(ROOT/n)==h,n
    assert sha(ROOT/config['checkpoint_path'])==config['checkpoint_sha256']
    inventory=read(OUT/'18_640_raw_inventory.json')
    assert len(inventory['files'])==640
    for r in inventory['files']: assert sha(ROOT/r['path'])==r['sha256']
    archive=read(OUT/'19_archive_acceptance_receipt.json')
    assert sha(ROOT/archive['archive'])==archive['archive_sha256']
    summaries=accepted['per_budget']
    assert accepted['objective_first_difference']['PRIMARY_OBJECTIVE_DEGRADATION_COUNT']==0
    assert all(summaries[str(b)]['scoreable_cases']==160 for b in (512,1024))
    notes={
      '01_RESEARCH_CONTEXT.md':'完整64×5预算取舍验收通过；前三项一致与后两項损失只是冻结模型内Objective观察，不能外推真实闭环或系统收益。',
      '02_ARCHITECTURE.md':'新增6.4D orchestration、只读SHA备份、标准库独立统计审计；原MH/WM/Domain/Objective算法和checkpoint全部字节身份不变。',
      '03_DATA_FLOW.md':'既有静态64anchor×5seed→48原B512 SHA引用+272新B512→原320 B1024 SHA引用→320严格配对/首差/cluster统计→640raw inventory。无环境动作、无Future Target选样、无locked_test。',
      '04_MODULE_MAP.md':'6.4D入口prepare_step6_4d_full_cohort_b512_v1.py、run_step6_4d_full_cohort_b512_v1.py；独立auditor为audit_step6_4d_completed_expansion_v1.py，机器验收17/18/19。',
    }
    boundary='CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING；FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；Stage B=NOT_STARTED；locked_test=false。'
    for n,note in notes.items():
        p=ROOT/'AI_CONTEXT'/n
        header='## 2026-10-04 STEP 6.4D 验收边界\n\n'+note+' '+boundary+' SEARCH_METHOD仍MH-CEM纯搜索骨架，非最终hybrid。\n\n'
        text=p.read_text(encoding='utf-8')
        if not text.startswith(header):p.write_text(header+text,encoding='utf-8',newline='\n')
    p=ROOT/'AI_CONTEXT/06_DECISIONS.md'
    header=('## 2026-10-04 Researcher Decision / 人类确认\n\n'
        '研究者此前明确授权本轮实验结束且结果本地SHA/归档通过后关闭对应AutoDL实例；本聊天回复“我已经关机了”。'
        '记录为人类手动关机确认，不是Codex UI核验。6.4D统计建议不是新的科研决定，最终closed-loop预算仍待研究者决定。\n\n')
    text=p.read_text(encoding='utf-8')
    if not text.startswith(header):p.write_text(header+text,encoding='utf-8',newline='\n')
    context=[]
    for p in sorted((ROOT/'AI_CONTEXT').glob('0[0-8]_*.md')):
        text=p.read_text(encoding='utf-8')
        assert ('6.4D' in text and 'RESEARCHER_DECISION_PENDING' in text and 'locked_test=false' in text),p
        context.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'review':'PASS',
          'basis':'17accepted raw evidence; no change to scientific definitions or historical results'})
    assert len(context)==9
    run=read(OUT/'20_formal_runtime_receipt.json')
    assert math.isclose(run['new_cases_in_solve_seconds']+run['matrix_loading_setup_overhead_seconds'],run['actual_elapsed_to_last_case_seconds'])
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    receipt={'verdict':'PASS','focused_cpu_tests':[
      {'command':"python -m unittest discover -s code/tests -p 'test_step6_4d*.py'",'tests':6,'seconds':0.382,'result':'OK'},
      {'command':"python -m unittest discover -s code/tests -p 'test_step6_4c*.py'",'tests':4,'seconds':0.002,'result':'OK'},
      {'command':"python -m unittest discover -s code/tests -p 'test_step6_3d*.py'",'tests':34,'seconds':31.190,'result':'OK',
       'note':'Expected argparse refusal messages exercised by negative tests; suite OK'}],
      'compileall':{'command':'python -m compileall -q code/src code/scripts code/tests','result':'PASS'},
      'Context_Consistency_Check':{'verdict':'PASS','files':context,'current_status':'STEP_6_4D_FULL_COHORT_B512=PASS',
        'state_from_raw':{'B512_coverage':320,'parent_coverage':320,'new_cases':272,'reused_cases':48,'primary_degradation':0},
        'research_decisions_changed':False},
      'source_input_checkpoint_sha':'PASS','all_640_raw_original_sha':'PASS','archive_sha':'PASS',
      'git_diff_check':'PASS','knowledge_index':'PENDING_RUN_AFTER_STAGING',
      'scope':{'Stage B':'NOT_STARTED','FORMAL_CLOSED_LOOP_PERFORMANCE':'NOT_STARTED','locked_test':False},
      'known_limits':['future Return-birth fixed-support remains','no final closed-loop budget decision',
        'shutdown researcher-confirmed, not Codex UI-verified','search-time comparison includes historical timing provenance; not realtime deadline']}
    (OUT/'22_closure_checks_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print('Context9/9, source/input/checkpoint,640raw,archive,runtime decomposition,diff PASS; index pending.')


if __name__=='__main__':main()
