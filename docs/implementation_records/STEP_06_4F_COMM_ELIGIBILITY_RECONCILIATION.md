# STEP 6.4F — Comm eligibility reconciliation

## Step Goal

统一Planner当前Comm资格与真实执行，复验通信和候选A真实一步，审计历史搜索证据影响；CPU-only。

## Definition Basis

本轮研究者明确授权的交集定义，见`docs/contracts/PIJWM_STEP_06_4F_COMM_ELIGIBILITY_V1.md`。仅改变Planner eligibility，非FlowLedger/WorldModel科学定义变化。

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

STEP_6_4F=PASS；Comm当前资格修复与真实一步机制通过。共同谓词：Task present、offloading/transmitting、未完成，再与唯一合法活跃无线Flow绑定相交。Flow对象不删除，Flow存在不等于可执行。TRAIN4416/Validation1104完整静态前后审计与输入SHA一致；32/64搜索anchors中28/57的根候选域改变。真实TRAIN anchor0041：Task_24（offloading），UAV_1→vehicle_7，RB0/start0/width1，setter与真实无线传输事件一致，4.5→4.6s一次决策步；候选A独立episode、模拟NO_SCOREABLE_H4、同一bridge、无Comm的canonical合法动作亦4.5→4.6s。Future Target poison不改变当前输入/动作。FORMAL_SEARCH_EVIDENCE_REUSE=TARGETED_REQUALIFICATION_REQUIRED：7个Validation起点有覆盖全部H1–H4分支的规则不变量证明，另外57个旧raw无完整候选/预测轨迹，不足以证明winner/ranking不变。最小建议重新验证57×5×3方法B1024=855及57×5 MH B512=285，共1140cases；仅建议，未授权、未执行。旧TRAIN调参28/32根域受影响，不声称历史调参验证新域；K4/rho0.1按本轮研究者指令保持，不自动retune。B512_EVIDENCE_REQUALIFICATION_REQUIRED=true；PLANNER_V1_CLOSED_LOOP_B_WM=512作为研究者working-budget保留。CLOSED_LOOP_PRE_FORMAL_READINESS=READY_FOR_FALLBACK_DECISION仅表示执行机制就绪，不解除搜索证据重新验证门。FINAL_FALLBACK_POLICY=RESEARCHER_DECISION_PENDING；SEARCH_METHOD=MH-CEM pure-search骨架，非最终hybrid。FORMAL_CLOSED_LOOP_PERFORMANCE=NOT_STARTED；Stage B=NOT_STARTED / DEFERRED；GPU=NOT_USED；locked_test=false。唯一下一动作是研究者审阅旧证据重新验证范围及fallback决定，不自动开跑。

| 数据组 | windows | stale有效Flow绑定 | eligible Tasks | 改变域 | 空域 | modes总和 | exact候选总和 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev_train | 4416 | 18204 | 18385→222 | 3888 | 14→14 | 268763→53429 | 10382182478445→13816272 |
| dev_validation | 1104 | 5165 | 5222→73 | 995 | 6→6 | 67573→13946 | 3575756901757→3172044 |
| train_anchors | 32 | 138 | 139→1 | 28 | 0→0 | 1904→420 | 58816704670→55013 |
| validation_anchors | 64 | 275 | 279→4 | 57 | 0→0 | 4068→883 | 107274078912→158262 |

stale Flow仍留在原state，数量不是after零；after eligible stale Task为0。TRAIN stale lifecycle：failed17921/unknown283；Validation failed5073/unknown92；completed绑定在此数据为0。Flow计数与unique eligible Task计数分母不同，不相减冒充同一量。所有组nonempty→empty/empty→nonempty均0。

真实nonempty Comm：Task24 taskslot7/relation95，U2V UAV_1→vehicle_7，RB0 width1，原setter/allocateRB消费并直接记录transfer delivered_data=0.6604788158018815（模拟器data单位）；outcome-only通道记录不回流决策。4.5→4.6s，Route无动作。候选A独立episode canonical无Comm/Comp、UAV hold，同桥一次4.5→4.6s；NO_SCOREABLE_H4是显式模拟reason，不运行MH搜索。

## Expected vs Actual

符合研究者资格与机制预期。历史搜索证据并非整体PASS：57/64根域改变，动态域也可能改变；不是小修复就默认可复用，也不是全部自动重跑。7证书充分条件覆盖全部预测分支，未把短采样当证明。

## Known Issues / Scientific conclusion boundary

重新验证1140cases只是最小建议、未授权执行；新域最终方法选择和B512比较需研究者审阅。旧TRAIN调参仅历史定义，K/rho保持研究者指令不retune。final fallback pending，behavior候选B未实现独立provider，setter partial mutation仍须终止。future Return-birth限制未变。本轮没有性能结论。

## Archive / Provenance

全量pre/post JSON各约50MB保留本地D:原目录，并生成mtime=0 gzip可逆压缩Git归档；11 inventory记录每文件SHA以及原JSON本地位置，避免提交巨大未压缩raw。本轮2真实step新身份26d7d93a06933c4e554a987e15fe04b3e95debf6a60cb5e6d4c3290b3758357d，执行后源码按SHA核验，旧执行身份不覆盖。

## Git

本Step在main；最终提交为Comm资格修复、CPU机制审计与Context收口，不重写历史运行provenance。具体commit通过`git log -- docs/implementation_records/STEP_06_4F_COMM_ELIGIBILITY_RECONCILIATION.md`和最终报告读取。

## Next Step

仅等待研究者审阅重新验证范围与fallback决定；不启动GPU、Stage B或正式闭环。
