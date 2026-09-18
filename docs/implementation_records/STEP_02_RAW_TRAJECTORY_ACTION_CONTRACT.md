# STEP 2 — 冻结单决策步 Raw Trajectory + 四类 Action Contract

## Step Goal
冻结 `Decision_t -> Action_t -> Execution_t -> Outcome_t -> Decision_{t+1}` 的可核验工程合同，包含决策可见字段、来源/单位/ID/时间语义、Route/Comm/Comp/UAV Mobility 四类动作的 AirFogSim 入口、执行和结果字段，并完成低成本闭环验证。

## Definition Basis
依据 `D:\shen\OB\科研\PIJWM\01仿真系统与原始轨迹.md`、`02数据集构建与模型输入.md`、`06策略器与候选动作规划.md`，尤其 06 中研究者已明确的 `A_t=(A^Route,A^Comm,A^Comp,A^Mob)`；车辆由 SUMO 推进，Mob 仅代表 UAV。定义文件只读，未修改。

## Initial State
Step 1 审计确认既有 Route/Comm/Comp 接口和单步采集顺序可复用，但缺少正式四类合同、UAV mobility collector/adapter 和 Outcome→下一 Decision 对齐断言。Step 1 的旧“UAV 是否引入”判断按研究者本次明确决定纠正为已确定目标。

## Files Involved
新增 `code/src/pi_jwm/raw_single_decision_step_contract_v1.py`、`code/tests/test_raw_single_decision_step_contract_v1.py`、`code/scripts/run_raw_single_decision_step_contract_v1.py`、`docs/contracts_PIJWM_RAW_SINGLE_DECISION_STEP_CONTRACT_V1.md`；新增 `code/artifacts/protocols/pi_jwm_raw_single_decision_step_contract_v1_20260918/`；同步 tracker、authority records、AI_CONTEXT 和 registries。

## Changes
- 用 dataclass 固定单步对象、字段来源阶段、四类动作字段、setter receipt、时间闭环和 ID 对齐检查。
- 复用仓库 AirFogSim scheduler 源码，精确调用 `setTaskOffloading/setTaskReturnRoute/setCommunicationWithRB/setComputingCallBack/setUAVMobilityPatterns`。
- 明确车辆运动由 SUMO 外生推进，present UAV 必须给出一个 mobility action。
- 加入最小环境闭环脚本和既有非 locked 真实 action ledger 的结构化证据扫描。

## Reuse
复用稳定 ID、frame/trajectory 对齐、mask/presence、既有 action attempt ledger 和 AirFogSim 原 scheduler。未修改模型、双图、RSSM、Loss、Planner、数据集和 checkpoint。

## Validation
- `python -m unittest discover -s .\\code\\tests -p 'test_raw_single_decision_step_contract_v1.py' -v`：6 项通过。
- `python .\\code\\scripts\\run_raw_single_decision_step_contract_v1.py`：返回码 0；四类 setter、env step、Outcome/下一 Decision 对齐通过。
- `historical_real_evidence.json`：找到既有 accepted 非 locked `cpu_callback+offload+rb` 记录并确认 env step completed。
- 未启动训练，未使用 GPU，未访问 `locked_test`。

## Results
Step 2 合同已冻结，低成本最小闭环通过。完整真实 AirFogSim 四类动作轨迹尚未完成，机器证据明确标为 false；这不阻碍本 Step 的“最小可核验闭环”要求，但阻碍把完整真实四类采集声明为已实现。

## Expected vs Actual
预期是建立可执行的四类动作和单步时序契约，并用真实源接口验证。实际达成；UAV setter 来源真实，环境闭环为最小环境，真实历史轨迹仅覆盖三类动作。

## Known Issues
- 本机缺少 `shapely` 和完整 `traci`，因此未运行完整 AirFogSim 场景；加载器只对未调用的禁飞区几何辅助提供显式失败占位，不伪造其能力。
- Decision 字段已经冻结工程来源，但完整 collector 对 speed/acceleration/angles 的真实采集接线尚未实施。
- 真实四类 action ledger 和真实场景 Outcome 闭环留待后续独立授权。

## Git
本记录完成后应形成独立 Conventional Commit 并 push `origin/main`。不包含工作区原有 2026-09-16 `task_plan.md/progress.md/findings.md` 改动。

## Next Step
唯一建议：冻结一条真实 AirFogSim 轨迹的四类动作采集接线，先做单条非 locked 决策步的真实字段与 Outcome 对齐验收；不自动执行。
