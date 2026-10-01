# STEP 6.3D — Validation 前阻塞修复与最终 Go/No-Go

## Step Goal / Definition Basis

只关闭上轮 B1/B2。研究者于2026-10-01明确：newly指当前轮实际完成的 distinct scoreable H4，历史重采样可计入，retained-only不可计入；授权 runner 的阶段、统计和身份修补，禁止改算法/重跑TRAIN/启动Validation。正式合同见 `docs/contracts/PIJWM_STEP_06_3D_VALIDATION_STAGES_AND_CEM_UPDATE_SEMANTICS_V1.md`。Definition06及现有模型支持边界沿用TRAIN closure来源，本次不改私有定义。

## Initial State / Files Involved

fetch后 `HEAD=origin/main=5c98c91880466cc4d76c4dfb1d56b32a34f5557d`，tracked clean，原ZIP/TASK/绘图脚本保留。已读00状态、TRAIN closure、pre-Validation audit、三个方法模块、正式runner、权威计划/进展和治理入口；正式Validation文件名计数为0，不读outcome。旧审计的NO_GO保留历史，新收口收据覆盖当前状态。

修改 `code/scripts/run_step6_3d_formal_cpu_matrix_v1.py`；新增闭合审计脚本 `close_step6_3d_pre_validation_blockers_v1.py`、`test_step6_3d_validation_stages_v1.py`、合同和本记录。证据目录 `code/artifacts/protocols/pi_jwm_step6_3d_pre_validation_blocker_closure_v1_20261001/`：01语义、02阶段、03新配置、04身份桥、05旧新等价、06schema、07再审计oracles、08最终验收、09 SHA manifest。

## Changes / Reuse

1. B1使用原solver证明当前轮完成的旧候选可以触发更新、轮内重复只计两个不同指纹；另一个显式提议fixture首轮得到elite，后续轮current=0而retained>0，均不更新。单一指纹不更新。搜索/CEM源文件无改动。
2. runner新增明确primary/diagnostic阶段。Stage A只1024，960cases/983040名义预算，完成写主比较/选择收据后return。Stage B要求先有正确身份且SHA互相绑定的Stage A文件，只运行256/512并写独立诊断，不覆盖选择。CLI缺阶段/执行配置拒绝，无自动继续分支。
3. 每method预算汇总补全，六类配对互斥计数和320断言；保留五seed原cluster及冻结bootstrap。Return/scorer计数单列、异常双来源一致性验证，计时区分solve时间和当前调用含resume读的耗时，不将其冒充跨恢复完整elapsed。预注册敏感性区间只作诊断。
4. 原TRAIN source/execution/raw/05/06保持原样。新Validation execution config记录当前source SHA、独立ID，父级文件逐份SHA绑定。当前source map只有正式runner改变；所有框架源码、one-solve、域/语法/rollout/scorer/comparator/选型规则、anchors/seeds/checkpoint未改。

## Validation — Actual Commands and Outputs

使用 test-driven-development：新阶段接口测试先运行6个，因缺汇总/阶段函数出现预期5个error；实现后6/6 PASS。新增Stage A SHA失配拒绝用例先出现预期1个失败，再补齐拒绝门通过。测试仅构造合成字典/CPU transition和临时输出，不在正式目录创建Validation result。

`python code/scripts/close_step6_3d_pre_validation_blockers_v1.py`：全部oracle完成后输出 `VALIDATION_GO`、Validation0和新执行ID。正式TRAIN inventory的771文件逐份SHA/size通过；768 raw每份绑定原source和ID，从raw重建整个四组排名与原05完全一致，两方法仍各选(4,0.1)。checkpoint真实字节SHA一致。旧资格配置与基线Git字节SHA一致，历史parents和选参未覆盖。

旧runner从基线Git文本加载，分别调用其实际solve_or_resume与patched版本，以相同TRAIN-only synthetic anchor、seed6301、三方法/三预算，共9对非计时离散SearchOutcome完全相同；patched侧先1024顺序，原侧256/512/1024。每对正确resume不再调用solver。JSON持久化会将tuple转list，等价比较按JSON规范化，排除项只有wall-clock，不忽略科学字段。

8项阶段/身份测试及原GPU身份测试使用合成输入确认：Stage A调度960次且仅1024、无Stage B依赖；Stage B调度1920次且不改Stage A/选择字节；无Stage A或错SHA/来源/设备/batch/checkpoint/parents拒绝；六类S-vs-HRS分别54/54/53/53/53/53，合计320，win/tie/loss107/106/107；每个method合成Return640、scorer320、Grammar960。这些数值是统计oracle，不是正式Validation结果。数学、shared B_WM/cache、所有预算batch16可完成、RNG/order、五分支method-selection重新检查通过。

## Results / Expected vs Actual / Known Issues

`PRE_VALIDATION_BLOCKER_CLOSURE=PASS`，B1/B2已关闭，`TRAIN_REUSE_VALID=true`，`TRAIN_RERUN_REQUIRED=false`；`RUNNER_PATCH_SCIENTIFIC_SEMANTICS_INVARIANT=PASS`。最终Go仅表示执行/统计合同与本次门就绪，不能推断方法优劣或真实模型H4成功率。future Return-birth固定支持限制仍为 `SUPPORTED_WITH_MODEL_OBJECTIVE_SUPPORT_LIMITATION`，不移除困难anchor。

新执行身份只冻结未来3080Ti运行合同，未运行新GPU资格探针或正式search；原资格/模型路径未变，仅runner orchestration改变，不以未运行的GPU结果自证。本 Step 全程Validation results0，locked_test=false，方法未选，无新TRAIN/Validation矩阵、retraining、baseline、ablation、closed loop。

## Context / Git / Next Step

同步AI_CONTEXT00/04/05/06/07/08（06只记录本次明确研究者决定）、相关authority/index/registry与根过程文件。focused tests、compileall、knowledge-index write/check、diff、Context/SHA检查结果见最终过程记录。重要结果收据及审计代码由09 manifest逐项绑定。Git使用独立Conventional Commit并push main，身份以本任务Git历史为准。

唯一下一动作：研究者审阅并单独授权Stage A。即使GO，本次也不开跑；Stage A完成后停止，Stage B仍另行授权。

实际最终低成本验证：`python -m unittest discover -s code/tests -p test_step6_3d*.py` 27/27 PASS（22.152秒）；`python -m compileall -q code/src code/scripts code/tests` exit0；git diff --check exit0。重新执行闭合审计包含11个stage/身份测试，输出GO且正式Validation仍0。

最终Context Consistency PASS（00–08均复核，01研究问题未变）；knowledge-index write/check passed=true、mismatches=[]；git diff和staged diff --check exit0；09 manifest中12项SHA逐项与staged Git blob一致。新Validation execution ID为 `3b3fcfe775f947790a982929a292fb3f3b8f30396996f7f9b33270e7d6c9f215`。正式Validation仍为0，提交推送后停止。
