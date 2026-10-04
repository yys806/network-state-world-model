"""Local documentation/receipt closure; no model forward or remote operations."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REL = 'code/artifacts/protocols/pi_jwm_step6_4d_full_cohort_b512_v1_20261003'
OUT = ROOT / REL


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, data):
    if p.parent == ROOT/'docs/registries':
        old = json.loads(subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()]).decode('utf-8'))
        data = retain_order(old, data)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def retain_order(old, new):
    if isinstance(old, dict) and isinstance(new, dict):
        return {k: retain_order(old.get(k), new[k]) for k in list(old)+[k for k in new if k not in old] if k in new}
    if isinstance(old, list) and isinstance(new, list):
        by_id = {v.get('id'): v for v in old if isinstance(v, dict) and 'id' in v}
        return [retain_order(by_id.get(v.get('id')), v) if isinstance(v, dict) else v for v in new]
    return new


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    accepted = read(OUT/'17_independent_acceptance_receipt.json')
    assert accepted['verdict'] == 'STEP_6_4D_FULL_COHORT_B512=PASS'
    archived = read(OUT/'19_archive_acceptance_receipt.json')
    assert sha(ROOT/archived['archive']) == archived['archive_sha256']
    runtime = read(OUT/'09_runtime_status.json')
    raw = [read(p) for p in (OUT/'solve_results/expansion').glob('*.json')]
    assert len(raw) == 272
    internal = sum(r['outcome']['budget_receipt']['wall_clock_seconds_diagnostic_only'] for r in raw)
    elapsed = runtime['elapsed_wall_clock_seconds']
    write(OUT/'20_formal_runtime_receipt.json', {
        'start_time_utc': runtime['start_time_utc'],
        'last_case_finish_time_utc': runtime['update_time_utc'],
        'actual_elapsed_to_last_case_seconds': elapsed,
        'new_cases': 272, 'new_cases_per_hour': 272/elapsed*3600,
        'new_cases_in_solve_seconds': internal,
        'matrix_loading_setup_overhead_seconds': elapsed-internal,
        'actual_new_transition_throughput_in_solve': 139264/internal,
        'actual_new_transition_throughput_end_to_end': 139264/elapsed,
        'reused_48_runtime_excluded_from_this_invocation_elapsed': True,
        'full_320_per_budget_in_solve_runtime_source': REL+'/11_per_budget_summary.json',
        'scope': 'synchronous paused simulation; no realtime-control or closed-loop-performance claim',
        'locked_test': False})
    write(OUT/'21_instance_shutdown_receipt.json', {
        'status': 'SHUT_DOWN_REPORTED_BY_RESEARCHER',
        'confirmation_source': 'Human reply in this chat: 我已经关机了',
        'ui_shutdown_verified_by_codex': False,
        'computer_use_status': 'Stopped by safety mechanism: current Edge URL could not be reliably identified',
        'new_raw_local_sha_backup_before_shutdown': 'PASS',
        'independent_acceptance_before_shutdown': 'PASS',
        'archive_before_shutdown': 'PASS',
        'archive_sha256': archived['archive_sha256'],
        'gpu_search_finished': True, 'Stage B': 'NOT_STARTED', 'locked_test': False})
    summary = ('STEP_6_4D_FULL_COHORT_B512=PASS。完整64 anchors×5 seeds，B512覆盖320/320（48原6.4C结果按SHA引用、272新增），'
        '原Stage A B1024父结果320/320身份与SHA通过且未重跑。新增名义/实际独立一步转移139264，完整B512实际163840，B1024参考327680。'
        '两预算均160/320可评分H4（50%），六类配对0/0/30/109/21/160合计320，B512对B1024为30胜/181平/109负；'
        '64anchor×5seed cluster bootstrap95% CI=[-0.3375,-0.15625]。PRIMARY_OBJECTIVE_DEGRADATION_COUNT=0，'
        '前三项N_DDL/A_DDL/J_Delay在双方可评分160对中均相同；首差J_Burden为B512更好1/B1024更好8，J_Effort为29/101，全等21。'
        '独特可评分候选15668/33505，B512减少53.24%；内部搜索mean57.641/119.746秒、median43.564/90.506秒，分别节省51.863%/51.866%。'
        '两预算困难anchor均32，交集32、差集0；future Return-birth固定支持限制保留，scorer exception/inconsistency=0。'
        '新增272总墙钟24692.314秒（6h51m32s），加载/准备开销单列。640raw独立核验、本地D:持久备份、新raw ZIP/SHA归档PASS；旧raw未复制/修改。'
        'GPU搜索已硬停止，研究者已在本聊天确认手动关机（Codex未取得UI关机核验）。SEARCH_METHOD仍MH-CEM纯搜索骨架，非最终hybrid冻结。'
        '证据支持将B512作为节省计算的预算候选，但完整词典序Objective有损失，不能声称等价、最优或真实闭环性能。'
        'CLOSED_LOOP_BUDGET=RESEARCHER_DECISION_PENDING，FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，Stage B=NOT_STARTED，locked_test=false。'
        '唯一下一动作研究者审阅完整预算取舍并决定预算，不自动执行后续实验。')
    files = ['AI_CONTEXT/00_PROJECT_STATE.md','AI_CONTEXT/05_EXPERIMENTS.md','AI_CONTEXT/07_KNOWN_ISSUES.md',
        'AI_CONTEXT/08_CHANGELOG.md','docs/CHANGELOG.md','docs/EXPERIMENT_INDEX.md','docs/RESEARCH_STATUS.md',
        'docs/RESULTS_INDEX.md','docs/PIJWM_IMPLEMENTATION_TRACKER.md','docs/implementation_records/README.md',
        'docs/implementation_records/STEP_06_4D_FULL_COHORT_MH_B512.md','task_plan.md','progress.md','findings.md',
        '记录/本地计划表.md','记录/PIJWM主文档.md','记录/8.12之后推进.md']
    header = ('## 2026-10-04 STEP 6.4D 全量B512最终验收（当前）\n\n'+summary+
        '\n\n证据：`'+REL+'/17_independent_acceptance_receipt.json`、`19_archive_acceptance_receipt.json`、'
        '`20_formal_runtime_receipt.json`、`21_instance_shutdown_receipt.json`。执行协议commit9b7f50e58d410e012d9b27ed04661e8c88191f2a；'
        '分析/文档收口属于后续独立commit，不覆盖历史运行source。\n\n以下均为历史gate；较早RUNNING/未启动表述不是当前状态。\n\n---\n\n')
    for n in files:
        p = ROOT/n
        text = p.read_text(encoding='utf-8')
        if not text.startswith(header):
            p.write_text(header+text, encoding='utf-8', newline='\n')
    p = ROOT/'docs/implementation_records/STEP_06_4D_FULL_COHORT_MH_B512.md'
    details = ('## Final Changes / Reuse / Validation / Expected vs Actual / Git\n\n'
        '实际GPU272已完成，48复用与320父结果原路径保留；新增只读监控、标准库独立auditor及文档收口工具，不进入GPU科学source closure。'
        '独立auditor从640raw重建全部320pair/首差/counters/runtime/困难anchors及seed6316、10000replication bootstrap，逐项与runner一致。'
        '新增272raw+运行收据和log归档，291个entry逐项SHA通过；640原文件清单保留引用血缘。'
        '符合本Step完整扩样与配对验收预期；科学观察是计算节省与后两项目标质量损失并存，预算仍待研究者决定。'
        '只读SSH两次失活，原GPU进程未重启；最终新连接SFTP补齐59文件、runner已退出且log为PASS; STOP。'
        'computer-use因不能识别Edge URL被安全机制停止；研究者随后确认手动关机，此事实仅人类确认，未冒充UI验收。'
        'CPU回归/compileall/index/diff/Context一致性最终命令和输出见22_closure_checks_receipt.json。'
        '协议9b已push main；最终分析收口commit以Git历史定位，提交推送后停止。\n\n')
    text=p.read_text(encoding='utf-8')
    if details not in text:
        p.write_text(header+details+text[len(header):], encoding='utf-8', newline='\n')
    registry = ROOT/'docs/registries/experiment_registry.json'
    obj=read(registry); e=obj['experiments'][0]
    assert e['id']=='STEP-6.4D-FULL-COHORT-MH-B512-20261003'
    e.update(result=REL+'/17_independent_acceptance_receipt.json', metrics=REL+'/11_per_budget_summary.json',
        audit=REL+'/17_independent_acceptance_receipt.json', status='passed_full_cohort_budget_calibration',
        conclusion='320paired coverage PASS;51.9%search-time reduction;0primary-component degradation;109burden/effort losses;budget pending',
        summary_zh=summary)
    e['field_notes']={'result':'Independently reconstructed from640original raw; no StageA/6.4C rewrite',
        'metrics':'Complete320per budget; original reference timings retained;20separates272invocation overhead',
        'code_version':'GPU owning protocol commit9b; final analysis tools have separate commit identity'}
    write(registry,obj)
    p=ROOT/'docs/registries/results_registry.json'; obj=read(p)
    # The frozen index builder handles historical training and Stage A numeric
    # registry types. Keep this new result as an evidence route, without inventing
    # an unsupported numeric result type or altering frozen builder source.
    obj['budget_calibration_acceptance_sources']=[{
        'id':'RES-STEP-6.4D-FULL-COHORT-MH-B512-20261004', 'experiment_id':e['id'],
        'audit':e['audit'],'audit_sha256':sha(ROOT/e['audit']),
        'summary':e['metrics'],'pairwise':REL+'/12_paired_budget_comparison.json',
        'objective_components':REL+'/13_objective_first_difference.json',
        'archive':REL+'/19_archive_acceptance_receipt.json',
        'numeric_validation':'stdlib-only raw independent auditor; this is a navigation source, not an unsupported numeric registry row',
        'claim_boundary':e['claim_boundary'],'locked_test_accessed':False}]
    write(p,obj)
    p=ROOT/'docs/registries/question_routes.json'; obj=read(p); r=obj['routes'][0]
    assert r['id']=='ROUTE-STEP6.4D-B512-EXPANSION'
    r['primary_sources']=['docs/implementation_records/STEP_06_4D_FULL_COHORT_MH_B512.md',e['audit']]
    r['verification_sources'] += [REL+'/'+n for n in ['12_paired_budget_comparison.json','13_objective_first_difference.json','19_archive_acceptance_receipt.json','20_formal_runtime_receipt.json'] if REL+'/'+n not in r['verification_sources']]
    write(p,obj)
    p=ROOT/'docs/registries/deferred_work.json'; obj=read(p); d=obj['items'][0]
    d['summary_zh']='STEP6.4D仅B512完整64×5扩样已验收。B256扩样/原StageB/正式闭环/最终预算仍未授权；研究者下一步审阅并决定，不自动运行。'
    write(p,obj)
    print('Context/authority/process/registry final result sync written; research decisions unchanged.')


if __name__ == '__main__':
    main()
