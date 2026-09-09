# P4 link低召回edge GRU单次因果核验实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` task-by-task. Terra implements only the frozen mechanics; Sol owns specification and acceptance.

**Goal:** 在不训练、不修改checkpoint和正式模型的前提下，只旁路`edge_transition`，判断edge GRU更新是否足以解释当前主要低召回。

**Architecture:** 复用现有recall diagnosis的输入冻结、mask、分组和发布逻辑。对同一validation loader先运行原始路径，再通过临时forward hook运行唯一反事实路径；两条路径逐元素核对输入一致后，按预注册六项门机械判定。正式模型源码、loss、阈值、数据和方案B均不修改。

**Tech Stack:** Python 3、PyTorch CPU、`unittest`、JSON、SHA-256。

---

## 文件边界

- Modify: `code/scripts/run_formal_p4_link_recall_diagnosis_v1.py`——只抽取可复用的冻结validation准备函数，不改变现有runner行为。
- Modify: `code/tests/test_run_formal_p4_link_recall_diagnosis_v1.py`——保护抽取前后的既有行为。
- Create: `code/scripts/run_formal_p4_edge_gru_bypass_intervention_v1.py`——唯一干预、比较、判定和原子发布入口。
- Create: `code/tests/test_run_formal_p4_edge_gru_bypass_intervention_v1.py`——TDD规格、失败边界和runner测试。
- Create after real run: `code/artifacts/audit/pi_jwm_p4_edge_gru_bypass_intervention_20260904/`——JSON与manifest。
- Update after acceptance: `task_plan.md`、`progress.md`、`findings.md`、`记录/本地计划表.md`、`记录/PIJWM主文档.md`、`记录/8.12之后推进.md`、`PROJECT_CONTEXT.md`。

不修改`code/src/pi_jwm/formal_dual_graph_world_model_v1.py`，不创建checkpoint，不提交Git。

### Task 1：冻结核心接口并得到RED

- [ ] 在新测试文件中导入以下尚不存在的接口：

```python
from run_formal_p4_edge_gru_bypass_intervention_v1 import (
    build_edge_gru_bypass_intervention_audit,
    edge_gru_bypass,
    run_formal_p4_edge_gru_bypass_intervention,
)
```

- [ ] 构造20步合成logit，测试原始与干预路径使用原始float32的`sigmoid(logit) >= 0.9`，包含`torch.nextafter`边界。
- [ ] 构造持续活跃、新激活和previous-unobserved正样本，测试两条路径都复用`build_link_recall_diagnosis`的完整分组与h1..h20结构。
- [ ] 明确断言六项预注册门字段及总判定；分别构造全部通过和每一项单独失败的输入。
- [ ] 运行：

```powershell
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src'
python -m unittest discover -s .\code\tests -p 'test_run_formal_p4_edge_gru_bypass_intervention_v1.py' -v
```

预期：因新模块不存在而RED。不得用跳过测试或空实现制造GREEN。

### Task 2：实现唯一hook与比较核心

- [ ] 实现context manager `edge_gru_bypass(model)`：

```python
handle = model.edge_transition.register_forward_hook(
    lambda module, args, output: args[1]
)
try:
    yield counter
finally:
    handle.remove()
```

必须验证模型具有`edge_transition`且为`torch.nn.GRUCell`；hook只绑定这个对象。counter记录真实调用次数，退出后再运行一次forward必须证明hook已移除。

- [ ] 实现：

```python
build_edge_gru_bypass_intervention_audit(
    baseline_logits,
    intervention_logits,
    labels,
    previous_link_activity,
    persistence_predictions,
    provenance,
)
```

两条路径分别调用现有`build_link_recall_diagnosis`。额外逐元素计算原始集合：

```text
continued & persistence_positive & baseline_candidate_negative
```

并统计其中`intervention_candidate_positive`的数量，以及干预前后raw logit的q0/q10/q50/q90/q100。

- [ ] 固定六项判定：

```text
recovered_continued >= 1396
intervention_continued_fn <= 3336
h1_continued_recall >= 0.20
h20_continued_recall >= 0.20
intervention_overall_f1 >= 0.4723570869
intervention_overall_fp <= 1114
```

原始TP/FP/FN不等于`1902/557/5855`必须抛错，不能输出“不支持”掩盖输入漂移。

- [ ] 运行Task 1测试并达到GREEN。

### Task 3：复用冻结输入准备，不复制runner

- [ ] 从`run_formal_p4_link_recall_diagnosis_v1.py`抽取私有函数：

```python
_prepare_frozen_link_validation(
    run_root,
    tensor_root,
    method,
    expected_tensor_manifest_sha256=None,
)
```

返回strict-reloaded model、固定validation DataLoader和provenance基础字段。它必须保留现有全部检查：seed/method、run manifest、checkpoint/class weights SHA、train-only `pos_weight=50`、canonical tensor manifest、公共tensor文件、全部selected validation trajectory、sample IDs、model contract及`locked_test=false`。

- [ ] 让原recall runner改为调用该函数；其真实输出结构、TP/FP/FN硬门和发布路径不变。
- [ ] 运行原诊断9项测试；预期`9/9 OK`。

### Task 4：实现CPU-only intervention runner

- [ ] 新runner只调用一次Task 3准备函数，随后：

```text
baseline = _collect_validation(model, loader)
with edge_gru_bypass(model):
    intervention = _collect_validation(model, loader)
```

- [ ] 逐horizon验证labels、previous activity、persistence和mask counts完全一致；任何漂移立即报错。
- [ ] hook预期调用次数为`20 * len(loader)`；实际不相等立即报错。报告记录模块全名、hook语义、actual/expected calls和成功移除证明。
- [ ] provenance新增干预脚本SHA、基线诊断SHA、intervention=`edge_transition_returns_previous_hidden`，并保持GPU/locked/formal claim全为false。
- [ ] 输出使用同级唯一staging；异常清理自身staging；拒绝覆盖已有目标；manifest字段为`size_bytes`和SHA-256。

### Task 5：补齐失败路径测试并审查

- [ ] 测试hook只影响`edge_transition`，不改变其他GRU对象；异常时也移除hook。
- [ ] 测试hook调用次数漂移、baseline/intervention标签或mask漂移、原始TP/FP/FN漂移、非CPU/locked/provenance漂移全部拒绝。
- [ ] 测试checkpoint missing/unexpected key、selected trajectory SHA/size漂移仍由共享准备函数拒绝。
- [ ] 测试发布成功的JSON/manifest可复核，并测试staging创建后失败会清理且不覆盖目标。
- [ ] Terra运行两组定向测试、相关模型和概率回归、compileall、`git diff --check`。
- [ ] Sol逐项规格审查；任何缺口交回同一实现者做最小TDD修复。
- [ ] fresh Sol可用时做代码质量审查；如模型额度不可用，由主Sol逐行审查并独立运行边界探针，不以worker自报代替验收。

### Task 6：主Sol真实CPU核验

- [ ] 主Sol独立运行：

```powershell
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src'
python .\code\scripts\run_formal_p4_edge_gru_bypass_intervention_v1.py `
  --run-root .\code\artifacts\experiments\pi_jwm_p4_edge_feedback_gru_input_gpu_20260831\seed_20260831 `
  --tensor-root .\code\artifacts\formal_tensor\pi_jwm_v4_tensor_v4_rule_contract_h20_rb_repair_unlocked_20260827 `
  --method coupled_dual_gnn_residual `
  --baseline-diagnosis .\code\artifacts\audit\pi_jwm_p4_link_recall_diagnosis_20260904\link_recall_diagnosis.json `
  --output-dir .\code\artifacts\audit\pi_jwm_p4_edge_gru_bypass_intervention_20260904
```

- [ ] 重新计算输出文件size/SHA，核对原始TP/FP/FN、六项判定、h1/h5/h10/h20、provenance、hook calls、GPU和locked flags。
- [ ] 若六项全部通过，只记录`sufficient_to_explain_dominant_low_recall`，再设计一个最小训练修复；不得把旁路当正式模型。
- [ ] 任一项失败，记录`not_sufficient_to_explain_dominant_low_recall`并停止；不得自动执行第二种干预。
- [ ] 更新过程和权威记录，唯一下一动作由真实结果决定。

## 实施停止门

- 任何代码触及正式模型、loss、数据、阈值或方案B：停止。
- 任何测试需要GPU或`locked_test`：停止。
- 任何输入哈希、样本、mask或原始TP/FP/FN漂移：停止。
- 任何实现同时关闭CFE、rule feedback或其他GRU：停止。
- 真实干预只运行一次；不通过时不调门、不换hook、不重跑。
