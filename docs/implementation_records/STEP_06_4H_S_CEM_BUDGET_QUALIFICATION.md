# STEP 6.4H — S-CEM 在线搜索预算资格（CPU gate）

## Step Goal

准备唯一320-case S-CEM B512矩阵，与已验收B1024严格配对。本轮完成CPU门后停止，等待研究者开GPU；不运行搜索或真实环境动作。

## Definition Basis / Initial State

研究者2026-10-08显式授权及末尾CPU-only指令。HEAD=origin/main=aea391f280e5933898cf1804d06179c0d9ec5f2f，git fetch成功，tracked clean；已有未跟踪参考ZIP/TASK/plot脚本保留。6.4G T/A accepted，选S-CEM K4/rho0.2，MH专用B条件停止。

## Files Involved / Changes / Reuse

新增code/scripts/step6_4h_s_cem_budget_v1.py、code/tests/test_step6_4h_budget_v1.py、独立protocol/manifest/config/parentrefs/CPU receipts、合同及本记录。复用已测试的科学solve、6.4G validate_result/atomic、6.4D预算配对与首差统计、6.4G独立数学oracle；不调用或修改旧Phase B条件门。所有code/src/pi_jwm科学文件不变，旧6.4F/6.4G/旧MH结果不变。同步AI_CONTEXT00–08、权威/过程/导航及experiment/question registry。

## Validation

先加测试并确认缺新模块时失败，再实现。新增资格反例、320六类统计/目标首差、完整cohort约束、resume身份/NaN/scorer/预算拒绝、raw summary oracle tests。真实B1024父文件320/320及ZIP内逐SHA、accepted manifest/source/config/checkpoint/模型参数验证PASS。CPU不运行WM搜索，未来字段poison回归使用既有因果bridge/Comm资格测试。

最终focused CPU tests、compileall、knowledge-index write/check、Context Consistency Check、git diff --check输出见05_cpu_validation.json；GPU运行门尚未执行，不把CPU PASS写成GPU运行PASS。

## Results / Expected vs Actual

符合本轮CPU准备预期：320父raw PASS，B512=0/320，READY_FOR_GPU_LAUNCH=true。资格NOT_RUN、final budget待研究者决定。所有性能、scoreability差异和CI待GPU320cases后统计，当前不填预测数值。源身份由03 config绑定；精确Git freeze commit以本次commit为准，启动强制HEAD/origin/main相同。

## Known Issues / Boundaries

future Return-birth fixed-support仍保留。远端GPU型号/容量/无旧runner须续跑前核验，当前未连接/开机。正式闭环NOT_STARTED，hybrid NOT_FROZEN，locked_test=false。预算资格不代表真实闭环性能；0.1s模拟步不是GPU墙钟期限。

## Git / Next Step

本门测试通过后提交并推送main；commit本身为protocol-freeze gate，无自引用SHA。唯一下一动作：研究者开启原RTX3080Ti实例并通知继续，再做远端精确commit/身份/容量/runner检查；不擅自开关服务器。
