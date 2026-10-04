"""Bounded Step6.4F evidence and documentation closure (no simulator/search)."""
import copy,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/artifacts/protocols/pi_jwm_step6_4f_comm_eligibility_v1_20261004'
PREFIX=OUT.relative_to(ROOT).as_posix()
RECORD='docs/implementation_records/STEP_06_4F_COMM_ELIGIBILITY_RECONCILIATION.md'
CONTRACT='docs/contracts/PIJWM_STEP_06_4F_COMM_ELIGIBILITY_V1.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def put(p,v):
    dest=ROOT/p;dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def text(p,v):
    (ROOT/p).parent.mkdir(parents=True,exist_ok=True)
    (ROOT/p).write_text(v,encoding='utf-8',newline='\n')
SUMMARY='''STEP_6_4F=PASS；Comm当前资格修复与真实一步机制通过。共同谓词：Task present、offloading/transmitting、未完成，再与唯一合法活跃无线Flow绑定相交。Flow对象不删除，Flow存在不等于可执行。TRAIN4416/Validation1104完整静态前后审计与输入SHA一致；32/64搜索anchors中28/57的根候选域改变。真实TRAIN anchor0041：Task_24（offloading），UAV_1→vehicle_7，RB0/start0/width1，setter与真实无线传输事件一致，4.5→4.6s一次决策步；候选A独立episode、模拟NO_SCOREABLE_H4、同一bridge、无Comm的canonical合法动作亦4.5→4.6s。Future Target poison不改变当前输入/动作。FORMAL_SEARCH_EVIDENCE_REUSE=TARGETED_REQUALIFICATION_REQUIRED：7个Validation起点有覆盖全部H1–H4分支的规则不变量证明，另外57个旧raw无完整候选/预测轨迹，不足以证明winner/ranking不变。最小建议重新验证57×5×3方法B1024=855及57×5 MH B512=285，共1140cases；仅建议，未授权、未执行。旧TRAIN调参28/32根域受影响，不声称历史调参验证新域；K4/rho0.1按本轮研究者指令保持，不自动retune。B512_EVIDENCE_REQUALIFICATION_REQUIRED=true；PLANNER_V1_CLOSED_LOOP_B_WM=512作为研究者working-budget保留。CLOSED_LOOP_PRE_FORMAL_READINESS=READY_FOR_FALLBACK_DECISION仅表示执行机制就绪，不解除搜索证据重新验证门。FINAL_FALLBACK_POLICY=RESEARCHER_DECISION_PENDING；SEARCH_METHOD=MH-CEM pure-search骨架，非最终hybrid。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；Stage B=NOT_STARTED / DEFERRED；GPU=NOT_USED；locked_test=false。唯一下一动作是研究者审阅旧证据重新验证范围及fallback决定，不自动开跑。'''

def main():
    if (OUT/'09_final_acceptance_receipt.json').exists():
        raise RuntimeError('Closure record already exists; do not duplicate Context or overwrite receipts')
    audit=read(PREFIX+'/06_full_static_audit_summary.json')
    reuse=read(PREFIX+'/07_formal_search_evidence_reuse_assessment.json')
    comm=read(PREFIX+'/04_real_nonempty_comm_one_step_receipt.json')
    fallback=read(PREFIX+'/05_simulated_NO_SCOREABLE_H4_canonical_fallback_one_step_receipt.json')
    poison=read(PREFIX+'/02_future_target_poison_receipt.json')
    assert comm['verdict']==fallback['verdict']==poison['verdict']=='PASS'
    assert comm['start_time_s']==fallback['start_time_s']==4.5
    assert reuse['FORMAL_SEARCH_EVIDENCE_REUSE']=='TARGETED_REQUALIFICATION_REQUIRED'
    diff=subprocess.check_output(['git','diff','--name-only','--','code/src'],cwd=ROOT,text=True).splitlines()
    allowed={'code/src/pi_jwm/step6_3b_candidate_grammar_v1.py','code/src/pi_jwm/step6_4b_live_bridge_v1.py'}
    assert set(diff)==allowed
    protected=['step6_3c_candidate_domain_v1.py','step6_3d_fixed_budget_search_v1.py',
      'step6_3d_structured_proposal_v1.py','step6_3d_method_selection_v1.py',
      'step4_4_structured_rssm_world_model_v1.py','step6_1_trained_candidate_rollout_v1.py']
    gate={}
    for name in protected:
        rel='code/src/pi_jwm/'+name
        old=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)
        current=(ROOT/rel).read_bytes().replace(b'\r\n',b'\n')
        assert old.replace(b'\r\n',b'\n')==current
        gate[rel]=sha(ROOT/rel)
    old_e=ROOT/'code/artifacts/protocols/pi_jwm_step6_4e_comm_fallback_v1_20261004'
    old_hashes={}
    for rel in subprocess.check_output(['git','ls-files',str(old_e.relative_to(ROOT)).replace('\\','/')],cwd=ROOT,text=True).splitlines():
        original=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)
        assert original==(ROOT/rel).read_bytes(),rel
        old_hashes[rel]=sha(ROOT/rel)
    put(PREFIX+'/08_source_and_historical_provenance_gate.json',{
      'verdict':'PASS','tracked_scientific_source_changes':sorted(diff),
      'protected_source_SHA256':gate,'old_6_4E_all_tracked_bytes_unchanged':old_hashes,
      'old_640_search_raw_SHA_verified':True,'formal_dataset_and_normalization_changed':False,
      'checkpoint_loaded':False,'model_forward_count':0,'search_calls':0,'GPU':'NOT_USED','locked_test':False})
    put(PREFIX+'/09_final_acceptance_receipt.json',{
      'STEP_6_4F':'PASS','COMM_ELIGIBILITY_CONSISTENCY':'PASS',
      'REAL_NONEMPTY_COMM_EXECUTION':'PASS','FALLBACK_A_REAL_EXECUTION':'PASS',
      'Future_Target_leakage':'PASS','FULL_STATIC_AUDIT':'PASS',
      'FORMAL_SEARCH_EVIDENCE_REUSE':reuse['FORMAL_SEARCH_EVIDENCE_REUSE'],
      'B512_EVIDENCE_REQUALIFICATION_REQUIRED':True,'PLANNER_V1_CLOSED_LOOP_B_WM':512,
      'CLOSED_LOOP_PRE_FORMAL_READINESS':'READY_FOR_FALLBACK_DECISION',
      'readiness_boundary':'mechanism readiness only; revised-domain scientific evidence still requires separately authorized targeted requalification',
      'FINAL_FALLBACK_POLICY':'RESEARCHER_DECISION_PENDING',
      'FORMAL_CLOSED_LOOP_PERFORMANCE':'NOT_STARTED','Stage_B':'NOT_STARTED / DEFERRED',
      'GPU':'NOT_USED','locked_test':False,'focused_CPU_tests':53,
      'real_authorized_steps':2,'smoke_episodes':2,'search_calls':0,'WM_forward_count':0})
    text(PREFIX+'/.gitattributes','*.json -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n*.sha256 -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n*.gz -text\n.gitattributes -text\n')
    contract='''# STEP 6.4F — 当前通信资格合同

## Researcher Decision / Definition Basis

本轮研究者明确冻结 E_comm(t) 为有效唯一活跃无线Flow绑定与当前可通信Task生命周期的交集，仅动作资格修复。Task必须存在，lifecycle只能是offloading/transmitting；若task_completed适用则必须为false。failed、completed、unknown、缺失lifecycle均不得产生新Comm。旧Flow可留在模型状态，不删除Ledger对象。

## Source / Inputs / Outputs

`step6_4f_comm_eligibility_v1.py::comm_current_task_eligible`为raw-live与tensor共享Task谓词；`tensor_comm_task_eligible`解析冻结vocab。`step6_3b_candidate_grammar_v1.py::_wireless_bindings`保留原unique Flow、known/presence、current carrying hop active、wireless relation validity及端点一致规则，再加共同Task谓词。`step6_4b_live_bridge_v1.py::validate_command`同时验证当前tensor绑定及raw lifecycle/completed，关系必须匹配，不允许旧行为覆盖。

只用当前state/raw observation与因果History，不使用Future Target。缺失/无效字段fail closed；不放宽grammar/support。Comp、Mob、Route=EXPLICIT_NOOP_ONLY保持原代码。Dataset/Tensor/normalization/FlowLedger/WorldModel/checkpoint/Objective/CEM/RNG/B_WM不改。B512是研究者working-budget，不因本修复声称旧预算实验已验证新域。

## Failure / Fallback

当前无资格Task的Comm不能生成，显式外来非法动作bridge拒绝。域空、缺字段、非法bridge和setter异常维持6.4E smoke-only fail closed；setter不原子时有partial mutation风险，停止episode、不静默重试。候选A对非空合法域采用层级canonical排序，不评分不物化全集，并走winner同一个bridge；候选B仍无独立current behavior offer provider；C为不执行动作终止。最终fallback由研究者决定。

## Scientific evidence boundary

根域未改不自动证明未来域未改。重新验证范围与7个充分不变量证书见07 receipt：不活跃Flow不能被Route NOOP重新生为active；活跃任务return requirement未知且return slot=-1时，原固定支持规则不可能令其completed，因此新的Task过滤不会影响任何H1–H4分支。其他57个Validation anchors需重新验证旧winner/ranking；raw没有完整trace。合成旧rule oracle证明completed+activeFlow确实可能存在，不能只审计根域。

## Validation

53 CPU tests、TRAIN4416/Validation1104及32/64anchor同输入pre/post静态审计、两个预冻结TRAIN真实一步执行、future poison、旧raw SHA/历史6.4E字节保护。PASS仅为资格/执行机制，不是Planner性能证明。本Step未GPU、搜索或locked_test。
'''
    text(CONTRACT,contract)
    rows=[]
    for group,s in audit['summary'].items():
        rows.append(f"| {group} | {s['windows']} | {s['stale_valid_wireless_flow_bindings']} | {s['old_wireless_eligible_tasks']}→{s['new_wireless_eligible_tasks']} | {s['domain_changed_count']} | {s['before_empty_domains']}→{s['after_empty_domains']} | {s['before_structural_modes']}→{s['after_structural_modes']} | {s['before_exact_candidate_sum']}→{s['after_exact_candidate_sum']} |")
    table='| 数据组 | windows | stale有效Flow绑定 | eligible Tasks | 改变域 | 空域 | modes总和 | exact候选总和 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)
    record=f'''# STEP 6.4F — Comm eligibility reconciliation

## Step Goal

统一Planner当前Comm资格与真实执行，复验通信和候选A真实一步，审计历史搜索证据影响；CPU-only。

## Definition Basis

本轮研究者明确授权的交集定义，见`{CONTRACT}`。仅改变Planner eligibility，非FlowLedger/WorldModel科学定义变化。

## Initial State

fetch后HEAD=origin/main=a6e0230f87397136fd274116436f2a85d1f45270；tracked clean。6.4E BLOCKED原receipt保留。先保存源码00/pre全量01，才修复代码；未开GPU或搜索。

## Files Involved / Changes / Reuse

新增共同Task谓词；grammar `_wireless_bindings`与live `validate_command`两处接入；原Flow绑定和Comp/Mob/Route代码保持。复用6.4B/6.4E真实capture/bridge/setter和固定TRAIN causal prefix，新6Fnamespace和独立执行SHA；6E原receipt字节未变。新增pre/post审计、未来不变量及rule oracle、6F回归测试；老3B测试把computing与offloading任务分开，6E回归明确读历史被冻结非法动作，不再依赖修复后first-Comm。

## Validation — actual commands / outputs

- `python code/scripts/audit_step6_4f_comm_eligibility_v1.py --phase pre`：5520windows，原源码snapshot；约230.93s。
- 同脚本`--phase post`：5520逐行完全匹配预冻结交集，全部输入SHA一致；约207.6s。
- `D:/miniconda/envs/airfogsim/python.exe code/scripts/run_step6_4f_real_comm_smoke_v1.py --freeze`及`--execute`：两个独立TRAIN episode PASS；PYTHONIOENCODING=utf-8；未自动retry。
- `python code/scripts/assess_step6_4f_search_evidence_v1.py`：640raw SHA核验、StageA960inventory、7未来不变量证书及synthetic现有rule oracle PASS。
- focused unittest：6F资格6、3B13、3C domain/protocol、3D proposal/exact/method选择、6.4B7、6.4E7等合计50通过；新增未来证明3通过，总53。compile/index/diff/Context输出见10 verification receipt。

## Results

{SUMMARY}

{table}

stale Flow仍留在原state，数量不是after零；after eligible stale Task为0。TRAIN stale lifecycle：failed17921/unknown283；Validation failed5073/unknown92；completed绑定在此数据为0。Flow计数与unique eligible Task计数分母不同，不相减冒充同一量。所有组nonempty→empty/empty→nonempty均0。

真实nonempty Comm：Task24 taskslot7/relation95，U2V UAV_1→vehicle_7，RB0 width1，原setter/allocateRB消费并直接记录transfer delivered_data=0.6604788158018815（模拟器data单位）；outcome-only通道记录不回流决策。4.5→4.6s，Route无动作。候选A独立episode canonical无Comm/Comp、UAV hold，同桥一次4.5→4.6s；NO_SCOREABLE_H4是显式模拟reason，不运行MH搜索。

## Expected vs Actual

符合研究者资格与机制预期。历史搜索证据并非整体PASS：57/64根域改变，动态域也可能改变；不是小修复就默认可复用，也不是全部自动重跑。7证书充分条件覆盖全部预测分支，未把短采样当证明。

## Known Issues / Scientific conclusion boundary

重新验证1140cases只是最小建议、未授权执行；新域最终方法选择和B512比较需研究者审阅。旧TRAIN调参仅历史定义，K/rho保持研究者指令不retune。final fallback pending，behavior候选B未实现独立provider，setter partial mutation仍须终止。future Return-birth限制未变。本轮没有性能结论。

## Archive / Provenance

全量pre/post JSON各约50MB保留本地D:原目录，并生成mtime=0 gzip可逆压缩Git归档；11 inventory记录每文件SHA以及原JSON本地位置，避免提交巨大未压缩raw。本轮2真实step新身份26d7d93a06933c4e554a987e15fe04b3e95debf6a60cb5e6d4c3290b3758357d，执行后源码按SHA核验，旧执行身份不覆盖。

## Git

本Step在main；最终提交为Comm资格修复、CPU机制审计与Context收口，不重写历史运行provenance。具体commit通过`git log -- {RECORD}`和最终报告读取。

## Next Step

仅等待研究者审阅重新验证范围与fallback决定；不启动GPU、Stage B或正式闭环。
'''
    text(RECORD,record)
    evidence=f'证据：`{RECORD}`；`{PREFIX}/09_final_acceptance_receipt.json`；`{PREFIX}/07_formal_search_evidence_reuse_assessment.json`。历史6.4E BLOCKED及旧搜索accepted状态仅在旧定义内保留。'
    sections={
      '00_PROJECT_STATE.md':SUMMARY,
      '01_RESEARCH_CONTEXT.md':'研究边界：Flow存在与Planner动作资格分离。研究者授权当前Comm生命周期交集；旧搜索结论仅在旧资格定义成立，修复后57个Validation anchors须重新验证。MH-CEM pure-search及工作预算512保留，最终hybrid/正式闭环性能未冻结。',
      '02_ARCHITECTURE.md':'共享`step6_4f_comm_eligibility_v1`连接tensor grammar和raw-live bridge；模型状态/Flow Ledger不删除stale Flow。CandidateDomain科学实现只加Task资格交集；当前执行机制READY，不等于新域搜索科学证据PASS。',
      '03_DATA_FLOW.md':'因果current state/raw observation →共同Task lifecycle/completed predicate→唯一活跃无线绑定→Domain/grammar→同winner桥validate_command→原Comm setter→真实env一步→fresh observation。非空Comm和fallbackA各独立4.5→4.6s。传输event仅outcome；Future Target poison当前输入/动作不变。',
      '04_MODULE_MAP.md':'新增Task predicate模块、全量pre/post静态audit、assess未来域证书/旧rawSHA、真实Comm runner和6F tests。仅grammar/live bridge接入predicate；CEM、Objective、WM、Dataset保持。候选A复用6E fallback dispatch，不新增setter。',
      '05_EXPERIMENTS.md':SUMMARY,
      '06_DECISIONS.md':'Researcher Decision：本轮显式冻结E_comm(t)=有效唯一active wireless Flow绑定∩Task present∩offloading/transmitting∩not completed；只改Planner动作资格。K4/rho0.1/B512、Route NOOP保持。Codex证据建议57anchor targeted requalification不是新实验授权或研究者决定。FINAL_FALLBACK_POLICY=RESEARCHER_DECISION_PENDING。',
      '07_KNOWN_ISSUES.md':'6.4E stale-Flow动作/bridge不一致由6.4F修复并真实复验PASS。新独立issue：FORMAL_SEARCH_EVIDENCE_REUSE=TARGETED_REQUALIFICATION_REQUIRED，B512_EVIDENCE_REQUALIFICATION_REQUIRED=true；57/64旧winner/ranking无trace证据，7有全未来域充分证书。工作budget512仍保留。Final fallback待决定；behavior B无独立provider；setter非原子风险停止；Return-birth限制不改。',
      '08_CHANGELOG.md':SUMMARY}
    for name,body in sections.items():
        path='AI_CONTEXT/'+name
        old=(ROOT/path).read_text(encoding='utf-8')
        text(path,'## 2026-10-04 STEP 6.4F（当前）\n\n'+body+'\n\n'+evidence+'\n\n以下为历史状态，旧PASS不代表新资格定义下已经重新验证。\n\n---\n\n'+old)
    paths=['task_plan.md','progress.md','findings.md','记录/本地计划表.md','记录/PIJWM主文档.md','记录/8.12之后推进.md',
      'docs/PIJWM_IMPLEMENTATION_TRACKER.md','docs/implementation_records/README.md','docs/CHANGELOG.md',
      'docs/PROJECT_INDEX.md','docs/ARCHITECTURE.md','docs/RESEARCH_STATUS.md','docs/EXPERIMENT_INDEX.md','docs/RESULTS_INDEX.md']
    for path in paths:
        old=(ROOT/path).read_text(encoding='utf-8')
        text(path,'## 2026-10-04 STEP 6.4F（已完成当前授权范围）\n\n'+SUMMARY+'\n\n'+evidence+'\n\n---\n\n'+old)
    reg=read('docs/registries/experiment_registry.json');entry=copy.deepcopy(reg['experiments'][0])
    entry.update(id='STEP-6.4F-COMM-ELIGIBILITY-20261004',name='Current Comm eligibility reconciliation and real execution audit',
      research_question='Current planner/simulator legality reconciliation and historical search evidence scope',
      method='CPU static full dataset and two independent real one-step mechanisms; no search',
      code='code/scripts/run_step6_4f_real_comm_smoke_v1.py',
      code_version={'identity_type':'per_file_sha256_at_execution','git_commit':None,'manifest':PREFIX+'/01_fixture_and_execution_freeze.json','note':'New6F identity; old6E source/receipts preserved'},
      configuration=PREFIX+'/01_fixture_and_execution_freeze.json',
      parameters={'working_budget':512,'search_calls':0,'WM_forward_count':0,'authorized_steps_completed':2,'static_train_windows':4416,'static_validation_windows':1104,'device':'cpu'},
      protocol=CONTRACT,data=PREFIX+'/01_fixture_and_execution_freeze.json',split=['dev_train','dev_validation'],
      result=PREFIX+'/09_final_acceptance_receipt.json',audit=PREFIX+'/07_formal_search_evidence_reuse_assessment.json',
      status='passed_mechanism_targeted_evidence_requalification_required',conclusion='Eligibility and real mechanisms pass; old scientific search needs targeted requalification.',
      summary_zh=SUMMARY,claim_boundary='Mechanism only, no revised-domain search/performance acceptance',
      field_notes={'seed':'No planner RNG/search; simulator replay seeds frozen in fixture receipt','checkpoint':'Not loaded; no neural model forward','metrics':'No planner performance metrics','code_version':'per-file execution identity; final Git closure separate'})
    reg['experiments'].insert(0,entry)
    for e in reg['experiments'][1:]:
        if any(tag in e['id'] for tag in ('STEP-6.3D-VALIDATION-STAGE-A','STEP-6.4C-','STEP-6.4D-')):
            e['current_definition_reuse']='TARGETED_REQUALIFICATION_REQUIRED'
            e['current_definition_scope_note']='Historical accepted old Comm eligibility only; see Step6.4F assessment. No historical metrics/receipts rewritten.'
    put('docs/registries/experiment_registry.json',reg)
    results=read('docs/registries/results_registry.json')
    results['mechanism_readiness_acceptance_sources'].insert(0,{'id':'RES-STEP6.4F-COMM-ELIGIBILITY',
      'experiment_id':entry['id'],'audit':entry['result'],'audit_sha256':sha(ROOT/entry['result']),
      'status':'PASS','claim_boundary':entry['claim_boundary'],'locked_test_accessed':False,
      'numeric_validation':'Literal mechanism receipt; full static sums verified independently, not formal performance'})
    results['current_definition_search_evidence_assessment']=PREFIX+'/07_formal_search_evidence_reuse_assessment.json'
    put('docs/registries/results_registry.json',results)
    routes=read('docs/registries/question_routes.json')
    routes['routes'].insert(0,{'id':'ROUTE-STEP6.4F-COMM-ELIGIBILITY','keywords':['6.4F','Comm eligibility','stale Flow','search requalification'],
      'primary_sources':[RECORD,entry['result']], 'verification_sources':[CONTRACT,entry['audit'],PREFIX+'/06_full_static_audit_summary.json',PREFIX+'/04_real_nonempty_comm_one_step_receipt.json',PREFIX+'/05_simulated_NO_SCOREABLE_H4_canonical_fallback_one_step_receipt.json']})
    put('docs/registries/question_routes.json',routes)
    deferred=read('docs/registries/deferred_work.json')
    for item in deferred['items']:
        if item['id']=='STEP-6.4E-REAL-COMM-CONSISTENCY-RETEST':
            item['scientific_gate_status']='RESOLVED_BY_EXPLICIT_STEP_6_4F_AUTHORIZATION'
            item['reason']='6.4F eligibility repair and both real steps PASS; no automatic additional retest'
            item['summary_zh']='6.4E旧阻塞由明确授权6.4F修复复验；原历史记录保留，不自动再跑。'
    deferred['items'].insert(0,{'id':'STEP-6.4F-TARGETED-SEARCH-REQUALIFICATION','type':'researcher_decision_required',
      'status':'deferred','reason':'New action eligibility affects57 Validation root domains; propose minimal855 B1024+285 MH B512, no GPU authorization',
      'summary_zh':'旧域搜索证据需最小57anchor重新验证；工作预算512保留，禁止自动GPU重跑。',
      'authorization_required':True,'auto_start':False,'locked_test_accessed':False,'entrypoint':entry['audit'],
      'required_predecessors':['researcher scope review','new protocol and execution identity','CPU scientific gate'],
      'forbidden_until_authorized':['GPU search','Stage A rerun','B512 rerun','formal closed-loop performance']})
    put('docs/registries/deferred_work.json',deferred)
    print('Records/Context/registries written; compile/index/diff gates pending.')

if __name__=='__main__':main()
