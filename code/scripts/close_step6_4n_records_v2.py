"""Additive engineering/context records for the bounded v2 evidence."""
import json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
WM='code/artifacts/audit/pi_jwm_step6_4n_20261012_wm_v4'
REAL='code/artifacts/audit/pi_jwm_step6_4n_20261012_v5'
RECORD='docs/implementation_records/STEP_06_4N_DYNAMIC_RETURN_V2.md'
SUMMARY='''STEP 6.4N 的隔离 opt-in v2 CPU 机制已实现：Return 独立活动时钟、真实同隙 Route+RB、首步动作触发官方 Ledger Flow birth、共同 anchor 动态 slot/Graph/zero hidden、冻结 WM H1–H4 和 DynamicScorerV2。正式 v1 Route=EXPLICIT_NOOP_ONLY、旧 Scorer/Grammar/Domain/bridge、checkpoint 与 normalization 保持不变。Task_5 同根四分支 Gate 1 和双候选 Gate 2/3 通过；只是一条真实 root 的机制证据。预测可靠性 NOT_ESTABLISHED，正式 Route 集成 BLOCKED，是否需要重训 NOT_ESTABLISHED。GPU/训练/B512/正式S-CEM/baseline/locked_test/6.4O 均未启动。'''

def write(path,text):
 p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf8')
def prepend(path,text):
 p=ROOT/path;old=p.read_text(encoding='utf8');tag='<!-- STEP6.4N V2 CURRENT -->'
 if tag in old:return
 write(path,tag+'\n## 2026-10-12 STEP 6.4N — 隔离动态 Return CPU 机制\n\n'+text+'\n\n以下内容为此前记录，不覆盖本轮边界。\n\n---\n\n'+old)

def main():
 receipt=json.loads((ROOT/WM/'acceptance_v2.json').read_text(encoding='utf8'))
 if receipt['status']!='PASS':raise ValueError('ACCEPTANCE_NOT_PASS')
 detailed=SUMMARY+f'\n\n原始证据：`{REAL}/`、`{WM}/`；合同：`docs/contracts/PIJWM_STEP_06_4N_DYNAMIC_RETURN_V2.md`；实施记录：`{RECORD}`。唯一建议下一动作：研究者审阅本 Step 的预测误差与正式集成缺口；不自动运行下一阶段。'
 record=f'''# STEP 6.4N — Route–Comm–Flow 动态机制集成与 CPU 验证

## Step Goal / Definition Basis

实现研究者本轮明确授权的隔离 v2，不改变正式 v1 默认行为。定义依据为研究者 STEP 6.4N 的 A–D 方案及合同 `docs/contracts/PIJWM_STEP_06_4N_DYNAMIC_RETURN_V2.md`。私有定义目录 `D:/shen/OB/科研/PIJWM` 在本次主机上不存在，未读取或修改，不虚构文件哈希。此前 6.4L/R1/6.4M 源码、合同和真实回执已读取。远端 `git fetch origin` 成功，起点 main/origin/main=`26a51fa4b4fd27300bcb460fcd306c5f94fdd7ad`；gh 未安装，Git 远端直接核对。

## Initial State

已有未提交 runtime adapter、四分支 harness 和5项测试；v1 namespace 在执行前因 RB 检查失败。动态 WM/Scorer 与最终验收尚无。本轮保留旧失败证据，没有覆盖历史 receipt。

## Files Involved / Changes / Reuse

- `code/src/pi_jwm/step6_4n_runtime_v2.py`：instance-scoped Return 时钟、native Task 传输旁路、joint setter 前置检查/写回核验/部分失败隔离。Input、deadline、TTI 阈值与真实 radio 规则不变。
- `code/src/pi_jwm/step6_4n_dynamic_v2.py`：共同 anchor 去尾部无主填充、append Flow、DynamicScorerV2。
- `code/src/pi_jwm/step6_4n_rollout_v2.py`：官方 Ledger snapshot/Route action birth、ID→index、Graph/hidden 同步、当步进入冻结 one_step、递归与阶段 clock。
- `code/scripts/run_step6_4n_real_cpu_v2.py`、`run_step6_4n_wm_cpu_v2.py`：真实/模型有界实验。
- `code/scripts/audit_step6_4n_gate1_v2.py`、`audit_step6_4n_acceptance_v2.py`：独立原始落盘重建和评分重算，无模型选择或搜索。
- `code/tests/test_step6_4n_*.py`：23项机制/负例测试。
- `code/artifacts/manifests/pi_jwm_step6_4n_v2_20261012/config.json`：版本显式 opt-in。

复用官方 CausalFlowLedger、真实 AirFogSim setters/step、冻结 live history builder、Graph rebuild、ZERO_COLD_START append helper、冻结 one_step、v1 effort 公式；未修改上述 v1 文件。

## Validation (actual commands / outputs)

1. `D:/miniconda/envs/airfogsim/python.exe code/scripts/run_step6_4n_real_cpu_v2.py --mode <mode>`：四模式均 COMPLETED/各4步，最后 source-bound namespace={REAL}。
2. `python code/scripts/audit_step6_4n_gate1_v2.py`：9项 PASS；相同 prefix/current/RNG/time。
3. `python code/scripts/run_step6_4n_wm_cpu_v2.py`：两个候选各 H4，8 transitions；原参数 SHA 不变、Future poison PASS；source-bound namespace={WM}。
4. `python code/scripts/audit_step6_4n_acceptance_v2.py`：Gate 1/2/3 独立重建 PASS，原始逐 Task/Horizon 评分独立重算一致。
5. `python -m unittest discover -s code/tests -p 'test_step6_4n_*.py' -v`：23项通过。真实 TaskManager 测试使用 airfogsim 环境及 PYTHONUTF8=1。
6. v1 focused regression：`test_step6_2b_planner_objective_scorer_v1 test_step6_4f_comm_eligibility_v1 test_step6_3b_candidate_grammar_v1 test_step6_1_trained_candidate_rollout_v1 test_step6_4m_route_comm_flow_contract_v1 test_step6_4k_return_shadow_v1 test_step6_4b_live_bridge_v1`：61项通过。
7. compileall、知识索引 build/--check、diff 与 Context Consistency Check 在收口时执行；实际结果见最终 verification manifest。

## Results / Expected vs Actual

{SUMMARY}

真实1.8秒：旧 Route 被 TTI code1 拒绝；v2 无RB不继承Input时钟、Return clock保持1.8；v2 Route+RB0 同隙真实完成 Task_5，有效Return交付=0.07610754782944204；NOOP仍 waiting。无线 Task 原始传输参数=2.2081797374222076，但旧 collector 事件 delivered=Input大小0.556973122843206，存在已证实的历史采集字段错误。v2 直接记录 native方法参数、阶段需求与完成返回值，保留旧事件并单独解释，不修改历史数据或替真实结果打补丁。

模型共同 anchor 保留一个官方 Input Flow index0；移除31个无主填充后，B新Return index1与Ledger一致，而旧Shadow index32不是正式动态slot证据。出生在首步Route内部、learned dynamics前；Route effect不重复执行。A H4保持未满足Return，B首步完成。评分共同cohort=Task_5/6/7，分母Comm=50/Comp=52/Mob=30，H_eff=4。A=(0,0,1,1,0)，B=(0,0,2/3,2/3,1/600)。这是v2目标下两个指定动作的机制排序，不是搜索或正式性能结论。

## Known Issues / Status

本轮保留修复失败证据，未覆盖namespace：v1执行前RB检查失败；v2/v3真实step后的Shadow/collector schema冲突；WM v2缺wired配置；WM v3完成后relation清除触发Scorer拒绝。最终source-bound v5=16真实诊断步、WM v4=8转移；本轮累计50真实诊断步及24模型转移，始终同一指定Root/H≤4/两个固定模型候选，无搜索扩容或按未来结果选样。两项独立验收负例能拒绝score/fingerprint篡改与receipt缺失。

delivery_manifest保存82份实际AirFogSim Python/许可/依赖源文件ZIP与逐SHA、SUMO静态输入SHA。reference目录无独立Git元数据，不虚称上游commit；v2来自版本管理内adapter，未改AirFogSim源文件，完整运行依赖仍需安装。

{json.dumps(receipt['states'],ensure_ascii=False,indent=2)}

预测剩余量/完成事件在本 Task_5 H1–H4 对齐，不能证明 learned CSI 准确；第一隙小需求已饱和，错误速率也可能给相同完成结果。误差报告单独记录CSI/速率差和任务规则误差。最多2 Root是上限而非必须数量；Task_5已完成合法传输，故未根据结果另挑Root。

当前 v2 harness 候选为Route NOOP与Task_5 Return+RB0，并非可替代完整正式 CandidateDomain 的搜索域。完整 Route+Comm+Comp+Mob 批量联合资源 prevalidation / pre-existing returning实时bridge / 输入阶段零服务活动clock / 长 horizon 终止封装 尚未建立正式验收；部分setter失败会隔离并禁止step/retry。H≤4本root不会触发新Return TTI，但规则测试覆盖阈值前后与服务刷新。正式集成 readiness=BLOCKED，不能给出“不需重训”。Input优化/future birth/多跳/reroute/Epoch变化仍fail closed。

## Minimum Formal Integration Changes / Historical Revalidation Scope

1. 把v2动态对象映射、当前returning资格和批量联合资源事务接入版本化 CandidateDomain/Grammar/live bridge；先做CPU零副作用与所有setter失败门。
2. 预注册最多第二Root/未饱和服务条件，核验冻结模型CSI→速率→有效进展误差，研究者据证据决定是否重训；不可按未来成功结果选Root。
3. 历史6.4F–J以Route NOOP/fixed-support/v1评分为界保留，不删除；新Domain/Scorer会改变支持和排序，不能沿用6.4G方法选法、6.4H预算资格或6.4J性能身份。若正式集成获授权，至少重新核验支持catalog/候选域、搜索选法/预算、fallback、closed-loop协议、日志验收与指标分母。

## Context Consistency Check / Git / Next Step

AI_CONTEXT 00/02/03/04/05/06/07/08 增量同步，仅06记录研究者在本轮明确授权的v2研究方案；主文档、计划表、推进、tracker、索引和registry同步。Git commit/push见最终完成报告；未纳入无关未跟踪文件。唯一建议下一动作：研究者审阅预测可靠性未建立与正式集成缺口。完成本Step后停止，不启用正式Route搜索/6.4O/训练。
'''
 write(RECORD,record)
 for path in ('task_plan.md','progress.md','findings.md','记录/本地计划表.md','记录/PIJWM主文档.md','记录/8.12之后推进.md','docs/PIJWM_IMPLEMENTATION_TRACKER.md','docs/implementation_records/README.md','AI_CONTEXT/00_PROJECT_STATE.md','AI_CONTEXT/02_ARCHITECTURE.md','AI_CONTEXT/03_DATA_FLOW.md','AI_CONTEXT/04_MODULE_MAP.md','AI_CONTEXT/05_EXPERIMENTS.md','AI_CONTEXT/07_KNOWN_ISSUES.md','AI_CONTEXT/08_CHANGELOG.md','docs/CHANGELOG.md'):
  prepend(path,detailed)
 prepend('AI_CONTEXT/06_DECISIONS.md','研究者本轮明确授权阶段独立Return TTI、同隙Route+Comm、冻结模型动态Flow adapter和共同pre-Route Return需求Scorer v2，仅隔离CPU实验。未批准正式v1 Route搜索、重训、GPU或6.4O。\n\n'+detailed)
 reg=ROOT/'docs/registries/experiment_registry.json';j=json.loads(reg.read_text(encoding='utf8'))
 eid='STEP-6.4N-DYNAMIC-RETURN-V2-20261012'
 entry={'id':eid,'name':'Opt-in dynamic Return CPU mechanism','date':'2026-10-12','research_question':'Does one real pre-Route root support causal Return birth, same-slot RB and comparable v2 scoring?','method':'stage-aware Return TTI + causal dynamic Flow + frozen WM + DynamicScorerV2','code':'code/scripts/run_step6_4n_wm_cpu_v2.py','code_version':{'identity_type':'executed_source_SHA256','git_commit':None,'manifest':WM+'/wm_attempt.json','note':'Exact source SHA recorded before execution; closure commit identified by git history'},'configuration':'code/artifacts/manifests/pi_jwm_step6_4n_v2_20261012/config.json','parameters':{'device':'cpu','H':4,'roots':1,'candidates':2},'protocol':'docs/contracts/PIJWM_STEP_06_4N_DYNAMIC_RETURN_V2.md','data':REAL+'/V2_NOOP/01_prefix.json','split':['dev_train_mechanism_smoke'],'seed':[2026092318,2026092418],'checkpoint':'code/artifacts/formal_training/pi_jwm_formal_train_v1_seed5601_20260924T112424Z/checkpoints/best.pt','result':WM+'/acceptance_v2.json','metrics':WM+'/prediction_error.json','audit':WM+'/acceptance_v2.json','status':'cpu_mechanism_pass_formal_integration_blocked','conclusion':SUMMARY,'summary_zh':SUMMARY,'claim_boundary':'one real root; no formal search/performance; prediction reliability and retraining need unestablished','locked_test_accessed':False,'field_notes':{'code_version.git_commit':'Closure hash in final Git history; source files bound by SHA before execution','metrics':'Small paired diagnostic; no general accuracy/performance inference'}}
 j['experiments']=[entry]+[x for x in j['experiments'] if x['id']!=eid];write(str(reg.relative_to(ROOT)),json.dumps(j,ensure_ascii=False,indent=2)+'\n')
 reg=ROOT/'docs/registries/results_registry.json';j=json.loads(reg.read_text(encoding='utf8'));j['step6_4n_dynamic_return_v2']={'experiment_id':eid,'acceptance':WM+'/acceptance_v2.json','states':receipt['states'],'summary_zh':SUMMARY};write(str(reg.relative_to(ROOT)),json.dumps(j,ensure_ascii=False,indent=2)+'\n')
 reg=ROOT/'docs/registries/deferred_work.json';j=json.loads(reg.read_text(encoding='utf8'));id='STEP-6.4N-FORMAL-ROUTE-INTEGRATION';j['items']=[{'id':id,'type':'researcher_decision_required','status':'deferred','reason':'prediction reliability not established and full joint domain transaction not accepted','summary_zh':'CPU机制通过不解禁正式Route；需研究者决定最小可靠性验证','authorization_required':True,'auto_start':False,'locked_test_accessed':False,'entrypoint':RECORD,'required_predecessors':['researcher reviews v2 diagnostic evidence','joint Domain/bridge validation','prediction reliability evidence'],'forbidden_until_authorized':['formal Route search','6.4O','training','GPU','locked_test'],'scientific_gate_status':'BLOCKED'}]+[x for x in j['items'] if x['id']!=id];write(str(reg.relative_to(ROOT)),json.dumps(j,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
