"""Additive/current Step6.4G documentation and registry gate; no experiments."""
import argparse,copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REL='code/artifacts/protocols/pi_jwm_step6_4g_repaired_search_v1_20261004'
REC='docs/implementation_records/STEP_06_4G_REPAIRED_DOMAIN_FORMAL_SEARCH_REQUALIFICATION.md'
CON='docs/contracts/PIJWM_STEP_06_4G_REPAIRED_SEARCH_REQUALIFICATION_V1.md'
def write(path,value):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value,encoding='utf-8',newline='\n')
def load(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def dump(path,value):write(path,json.dumps(value,ensure_ascii=False,indent=2)+'\n')
BEGIN='<!-- STEP6.4G CURRENT -->';END='<!-- END STEP6.4G CURRENT -->'
def current(path,body):
    p=ROOT/path;old=p.read_text(encoding='utf-8') if p.exists() else ''
    if old.startswith(BEGIN):old=old.split(END,1)[1].lstrip()
    write(path,BEGIN+'\n## 2026-10-04 STEP 6.4G（当前授权 gate）\n\n'+body+'\n\n'+END+'\n\n以下是历史记录；旧调参/选法/预算结果只属于historical-under-pre-6.4F-domain。\n\n---\n\n'+old)
def main():
    p=argparse.ArgumentParser();p.add_argument('--state',choices=('PROTOCOL_READY','T_RUNNING'),default='PROTOCOL_READY');a=p.parse_args()
    running=a.state=='T_RUNNING'
    summary=('STEP_6_4G=IN_PROGRESS；CPU preflight/协议冻结PASS，完整旧TRAIN行为支持catalog复算一致，无需重建；原32/64anchors在修复域均非空，未替换。'
      'Phase T '+('RUNNING（以新raw/runtime为准）' if running else 'NOT_STARTED（须精确Git/device/source gate才开GPU）')+
      '，目标768；Phase A NOT_STARTED（目标960），Phase B NOT_STARTED（仅新A选MH才320）。旧调参/选法/预算均historical-under-pre-6.4F-domain；不复用任何旧raw，包含先前7不变量anchors也完整重跑。'
      'FINAL_FALLBACK_POLICY=CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1，研究者本轮明确冻结：NO_SCOREABLE_H4且当前域/输入完整才一个canonical A，A任何失败及空域/缺观测/slot unsupported/bridge/setter失败均C终止，不级联、不换动作、不重试。'
      'SEARCH_METHOD_REQUALIFIED=PENDING，S/MH新K/rho=PENDING；旧MH/K4rho0.1/B512只为历史/working决定，修复域最终预算待T→A→条件B gate。B512必须scoreability set一致、前三项目标劣化0、scorer0、mean/median耗时更低和严格budget。'
      'FORMAL_SEARCH_REQUALIFICATION=PENDING；CLOSED_LOOP_PRE_FORMAL_READINESS=PENDING_SEARCH_REQUALIFICATION；READY_FOR_FORMAL_CLOSED_LOOP_PROTOCOL=false。'+
      ('GPU PhaseT已在原3080Ti启动，CUDA12.8/PyTorch2.8.0+cu128/FP32/batch16；实际gatecommit6e0f47dfb60591fb57d8cd35ce14e768ba2e38a8；启动前tracked clean、结果0，2026-10-04T05:12:19Z启动；未开机、不换4090。' if running else '启动前3080Ti只读认证核验空闲、11912MiB free、tracked clean，无其他搜索进程；未启动实例，不换4090。')+
      'FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED，HYBRID_PLANNER=NOT_FROZEN，Stage B=NOT_STARTED / DEFERRED，locked_test=false。'
      '唯一下一动作：'+('监控PhaseT新结果身份与D:SHA备份，完成后独立验收+Git gate，源码不改。' if running else 'commit/push协议后服务器fast-forward到精确commit，重复preflight后只启动PhaseT。'))
    evidence=f'证据：`{REL}/00_protocol_r2.json`、`01_support_catalog_semantic_audit.json`、`02_frozen_anchor_manifest.json`、`04_CPU_preflight_receipt_r2.json`；实现记录`{REC}`。新目标evidence_class=repaired-domain formal evidence，尚无新正式科研结论。'
    for path in ['task_plan.md','progress.md','findings.md','记录/本地计划表.md','记录/PIJWM主文档.md','记录/8.12之后推进.md',
      'docs/PIJWM_IMPLEMENTATION_TRACKER.md','docs/implementation_records/README.md','docs/CHANGELOG.md',
      'docs/PROJECT_INDEX.md','docs/ARCHITECTURE.md','docs/RESEARCH_STATUS.md','docs/EXPERIMENT_INDEX.md','docs/RESULTS_INDEX.md']:
        current(path,summary+'\n\n'+evidence)
    names=sorted((ROOT/'AI_CONTEXT').glob('0[0-8]_*.md'))
    for path in names:
        body=summary
        if path.name.startswith('06_'):
            body='Researcher Decision（本轮人类粘贴指令）：正式fallback A-else-C；完整TRAIN768新调参→完整A960新选法→仅新MH获选才完整B512320；不复用任何旧raw；B512六项PASS gate预注册，失效预算待研究者、不得自动切1024。Codex不得固定新K/rho或预设MH胜出。\n\n'+summary
        elif path.name.startswith('02_') or path.name.startswith('03_') or path.name.startswith('04_'):
            body='新增`step6_4g_requalification_v1`仅实现阶段门/正式fallback dispatcher；复用原solver/WM/Objective/CandidateDomain修复规则，科学源码不改。A fallback复用6E prepare/execute与winner同bridge，不新增setter；dispatch不env.step，不调用模型。新增runner/independent auditor/CPU freeze，64anchor5seed统计沿用原rule，原目录只读。\n\n'+summary
        current(path.relative_to(ROOT).as_posix(),body+'\n\n'+evidence)
    contract='''# STEP 6.4G — 修复域正式搜索重新资格协议

## Researcher Decision / Fixed scientific boundary

完整TRAIN32×3 seeds×2 CEM×4(K3/4,rho.1/.2)×B512=768；原tune_cem_config规则success/paired net/smaller K/rho。独立验收并冻结新S/MH config，commit/push后才A。

A完整64×5×3方法B1024=960，原anchor-cluster bootstrap10000/seed6316/95% lower>0选法；六类互斥计数每组320，独立oracle重算与simultaneous diagnostic不改变主规则。不得预设旧MH结论。新选择非MH立即STOP，B不运行，闭环预算待研究者。

仅新MH获选才B51264×5=320；配对本轮A320 fresh MH B1024，不复用任何旧case。B gate：320身份完整、scorer0、scoreability set完全一致、PRIMARY_OBJECTIVE_DEGRADATION_COUNT=0、mean/median搜索耗时均降低、严格预算。J_Burden/Effort可损失但须报告，失败budget PENDING，不自动改1024。

WM/checkpoint/Dataset/normalization/Objective/Return/Route NOOP/H4/cache/RNG/comparator/Comm修复规则不改。TRAIN support catalog定义是behavior observed structures，与current legality不同；只读全量4416行为独立复算一致，保持原catalog。

## Fallback formally frozen

CURRENT_DOMAIN_A_ELSE_FAIL_CLOSED_C_V1。仅NO_SCOREABLE_H4且输入完整/当前域非空允许A一次canonical lazy choice。其他列举reason直接C；A preparation/admission/bridge/setter失败直接C，不级联、不改候选、不H3、不behavior fallback。current real only，RouteNOOP，原winner bridge；setter不原子风险只终止不重试。未来闭环必须记录trigger count/rate、原因、A execution和C termination；本轮没有闭环性能。

## Provenance / execution / resume

最终协议00_protocol_r2.json；早00仅CPU draft从未执行，原样保留并明确superseded。每phase config由协议SHA、全部冻结源码、checkpoint、eligibility/support/objective hashes及新父receipt SHA确定。T config已知；A/B身份在新选参/选法后按预冻结公式构造，不能用旧configs冒充。case identity涵盖phase/anchor/seed/method/K/rho/budget/device/FP32/batch16/source/checkpoint/eligibility/catalog/parents；错误resume在model forward前拒绝。GPU执行原始源码bytes必须等于Git LF SHA，CPU canonical hash不表示GPU资格。

每phase启动HEAD=origin/main exact phase gate commit且tracked clean；commit源码blob逐项匹配协议。阶段间只改receipts/docs，不改科研源码。fcntl排他锁防止重复runner；逐case原子保存，停止异常写STOP receipt不重启。完整matrix和时间诊断保留；低scoreability不改变矩阵/选法。没有额外免费forward，不改batch。

## Persistent backup and acceptance

每phase原raw保留，inventory/per-file SHA、ZIP/SHA回读、independent raw/math recompute；SFTP本地D:备份，`audit_step6_4g_phase_v1.py --verify-accepted-local`重新从本地raw复算并出local_backup_acceptance。下一phase parent强制包含该SHA及已提交新selected receipt。remote临时盘不冒充D:备份。

## Scope

不跑B256/旧三方法StageB/正式closedloop/performance/baseline/ablation/retraining/hybrid/locked_test。GPU若不可达，CPU协议commit后STOP READY_FOR_GPU_LAUNCH=true。真实GPU资源与软件/disk/checkpoint/source结果数量须重复核验，不能只看SSH网关。
'''
    write(CON,contract)
    record=f'''# STEP 6.4G — Repaired-domain formal search requalification

## Step Goal / Definition Basis

研究者最新粘贴文本授权整个pure-search链：完整新TRAIN→完整新A→条件新B与fallback冻结。本轮替代6.4F的1140case targeted proposal，不复用7未变anchors的旧结果。{CON}为工程合同。

## Initial State

HEAD=origin/main=1f2b9fafeeb00251383ae4551a78b1a9ef5bc48a，tracked clean。6.4F机制PASS但旧搜索需重新资格；服务器此前用户确认关机，当前只读SSH认证实际发现RTX3080Ti空闲，remote仍旧6.4D协议commit9b7f50e58d410e012d9b27ed04661e8c88191f2a，无search进程。本轮未操作开机。

## Files Involved / Changes / Reuse

新增6G phase/fallback模块、CPU冻结脚本、单phase runner、独立raw统计/本地备份auditor、SSH交互probe/deploy/launch/snapshot、meaningful tests。复用6F repaired predicate/3B grammar/3C Domain/3D solver/proposal/method-selection/6E合法动作准备与winner执行桥；这些既有源码不修改。支持目录完整TRAIN4416复算全部joint/start-width/task counts/temporal counts等同原值，保持原定义不重建。

## Validation (actual commands / outputs)

- TDD先运行6G tests因缺模块失败，实施后通过；runner synthetic六类test揭示summary调用少budget参数，在GPU前修复新orchestration，既有科研实现不改。
- focused 6G/6F/3B/3C/3D/6B/6E：67 tests in42.196s，OK；其中旧FallbackTests被import，计数如实包含其重复执行。
- CPU protocol preflight：原32/64固定anchor所有current domain非空、原全量静态source/input SHA再次匹配；checkpoint SHA和normalization/sample packages明确冻结。早版CPU draft原样保留，r2在protocol commit前补充输入与本地备份父链，未写任何正式solve result。
- 新phase identity/resume测试拒绝phase/source/device/checkpoint/seed/K/rho/batch/eligibility/support等错误；NaN/scorer/quota故障在保存前STOP；synthetic六类sum320和独立bootstrap/selection一致。不是正式Validation结果。
- compile/index/diff/Context命令及摘要见06_cpu_checks_receipt；全部通过后才能protocol commit/push/remote ff/GPU launch。

## Results / Expected vs Actual

{summary}

{evidence}

## Known Issues / Stop gates

真正新K/rho、选法、预算结论PENDING；不能宣称67测试或source一致即正式requalification PASS。Return-birth fixed-support边界保留，hard anchors不得删除。B512 gate失败待决定，A非MH停B。长GPU期间源码异常、crash/scorer/身份异常停止不修算法不自动重启。

## Git / Execution identity

protocol commit通过`git log -- {REC}`与06/07 gate receipts查询；scientific hash绑定Git LF，阶段间只提交新的已验收receipt与docs。每phase实际config独立保存于T/A/B execution_config.json；case gate commit不冒充旧3B/6C/6D执行身份。

## Next Step

{('只读监控T、SHA备份至768；独立验收+Git后才新A，依序执行授权链。' if running else 'Protocol commit+push，服务器ff该精确commit并重复preflight，才T；不可达则STOP。')}
'''
    write(REC,record)
    registry=load('docs/registries/experiment_registry.json')
    eid='STEP-6.4G-REPAIRED-SEARCH-20261004';entries=[e for e in registry['experiments'] if e['id']!=eid]
    e=copy.deepcopy(entries[0]);e.update(id=eid,name='Repaired-domain full TRAIN/StageA/conditional B512 requalification',
      date='2026-10-04',research_question='Requalify tuning,method,budget under repaired current Comm legality',
      method='TRAIN CEM grid; fresh three-method B1024 selection; conditional fresh MH B512',
      code='code/scripts/run_step6_4g_requalification_v1.py',
      code_version={'identity_type':'frozen_Git_LF_source_SHA256','git_commit':None,'manifest':REL+'/00_protocol_r2.json','note':'protocol gate commit read via Git; exact per-phase gate+source checked before GPU'},
      configuration=REL+'/03_phase_execution_identity_contract_r2.json',protocol=CON,data=REL+'/02_frozen_anchor_manifest.json',
      parameters={'T_cases':768,'A_cases':960,'conditional_B_cases':320,'batch_size':16,'precision':'FP32','GPU':'RTX3080Ti'},
      split=['dev_train','dev_validation'],seed={'T':[6301,6302,6303],'A_B':[6311,6312,6313,6314,6315]},
      checkpoint='code/artifacts/formal_training/pi_jwm_formal_train_v1_seed5601_20260924T112424Z/checkpoints/best.pt',
      result=REL,audit=REL+'/04_CPU_preflight_receipt_r2.json',metrics=None,status='T_RUNNING' if running else 'protocol_ready_not_started',
      conclusion='No new scientific results yet; freeze complete researcher rule and run full fresh chain',summary_zh=summary,
      claim_boundary='pure-search requalification only; formal closedloop/hybrid not started',locked_test_accessed=False,
      field_notes={'metrics':'new formal results pending; CPU/source/fixture gates are not performance','code_version':'per-file SHA before commit; exact commit required for GPU'})
    registry['experiments']=[e]+entries
    for row in entries:
        if any(x in row['id'] for x in ('STEP-6.3D-VALIDATION-STAGE-A','STEP-6.4C-','STEP-6.4D-')):
            row['current_definition_reuse']='historical-under-pre-6.4F-domain'
            row['current_definition_scope_note']='6.4G full fresh repaired-domain matrix authorized; no old raw enters new statistics'
    dump('docs/registries/experiment_registry.json',registry)
    routes=load('docs/registries/question_routes.json');routes['routes']=[r for r in routes['routes'] if r['id']!='ROUTE-STEP6.4G-REQUALIFICATION']
    routes['routes'].insert(0,{'id':'ROUTE-STEP6.4G-REQUALIFICATION','keywords':['6.4G','repaired-domain','TRAIN retuning','fallback freeze'],
      'primary_sources':[REC,REL+'/00_protocol_r2.json'],'verification_sources':[CON,REL+'/01_support_catalog_semantic_audit.json',REL+'/04_CPU_preflight_receipt_r2.json']})
    dump('docs/registries/question_routes.json',routes)
    results=load('docs/registries/results_registry.json');results['current_repaired_domain_protocol']=REL+'/00_protocol_r2.json';results['old_search_claim_scope']='historical-under-pre-6.4F-domain'
    dump('docs/registries/results_registry.json',results)
    deferred=load('docs/registries/deferred_work.json')
    for item in deferred['items']:
        if item['id']=='STEP-6.4F-TARGETED-SEARCH-REQUALIFICATION':
            item['scientific_gate_status']='SUPERSEDED_BY_EXPLICIT_FULL_STEP_6_4G_AUTHORIZATION'
            item['reason']='Researcher authorizes full fresh T768/A960/conditional B320, no targeted raw reuse; future experiments remain stopped'
            item['summary_zh']='原targeted1140建议被研究者完整重验证2048条件链替代，未授权其它实验。'
    dump('docs/registries/deferred_work.json',deferred)
    print('Step6.4G current records synchronized: '+a.state)

if __name__=='__main__':main()
