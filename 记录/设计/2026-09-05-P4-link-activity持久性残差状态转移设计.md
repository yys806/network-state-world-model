# P4 link activity 持久性残差状态转移设计

状态：候选方法已完成CPU契约实现和micro-smoke；尚未经过GPU sentinel性能与概率双门，不是通过结果。

## 1. 目标

只修复 formal P4 中链路活动事件的多步低召回风险。上一轮证据显示，主要漏报属于持续活跃链路，且多数漏报在当前 edge GRU 更新前已经低于阈值。本设计只改变 link activity 的读出和递推方式，不重构双图或重新定义数据。

## 2. 固定内容

- 物理图、信息图、节点/边/任务/资源字段、history=8、horizon=20保持不变。
- history encoder、edge GRU、物理消息、信息到物理边耦合、规则反馈、其他输出头保持不变。
- `aggregate_link_activity`仍为`active_task_count > 0`与原有观测/物理边mask的交。
- loss仍为现有weighted BCE，link的train-only `pos_weight`仍使用冻结协议中的50.0。
- raw threshold候选、calibration-only方案B、验证门和seed顺序保持不变。

## 3. 唯一新机制

现有`link_activity_head(edge)`改为输出未加权事件log-odds变化量：

```text
delta_t = link_activity_head(edge_t)
u_1 = persistence_logit(history_last) + delta_1
u_t = u_(t-1) + delta_t, t=2..20
z_t = u_t + log(pos_weight)
```

正式输出字段仍叫`link_activity_logits`，但值为`z_t`，因此loss和既有metric接口形状不变。`u_t`只作为forward内部状态，不作为新tensor输入。

历史最后一帧可观测时，沿用已有persistence基线的有限加权raw logit `+20/-20`，并解析减去`log(pos_weight)`得到`u_0`；因此零变化时正式输出仍精确为已有的`+20/-20`。历史活动缺失时，使用仅由train split有效标签计数得到的先验`pi_train`，取`logit(pi_train)`；没有双类有效训练统计时直接停止，不默认填0.5。

训练和推理都使用同一条递推：h1读取历史最后观测，h2-h20只读取模型自己的`u_(t-1)`。未来target和target mask只能进入loss/metric，不能进入forward。

## 4. 概率和阈值一致性

由于`z=u+log(50)`，方案B的解析反演仍为`z-log(50)`，并得到事件概率`sigmoid(u/T)`。温度T只能在训练后calibration split拟合，绝不能反馈到h2-h20递推。冻结raw threshold仍在z坐标中选择，并沿用已有映射和有限精度下的legacy decision保护。

零残差时，观测到的历史活动在全部20步精确保持已有persistence raw logit（`+20/-20`）与decision；这只是契约测试，不是新模型性能声明。

## 5. checkpoint与版本

新增方法身份`link_activity_persistence_residual_v1`。虽然`Linear(hidden,1)`形状不变，旧checkpoint中的head语义是绝对logit，新checkpoint中的head语义是delta，因此旧checkpoint不能续训或正式评价。加载时必须核对方法/schema身份；仅有`missing/unexpected keys=0/0`不构成兼容证据。

## 6. 验收顺序和停止门

先完成CPU契约测试、未来target不变测试、预测递推测试、概率反演测试、checkpoint版本拒载测试、既有模型回归和micro-smoke。CPU不得运行完整256/128/128训练或多seed性能实验。

CPU全部通过并经Sol Go/No-Go审核后，用户开启GPU才运行sentinel seed `20260831`，配置固定为256/128/128、8 epochs、hidden=32、batch=2、lr=3e-4、weight_decay=1e-5、history/horizon=8/20。sentinel先过validation link-F1和其他保护门，再验收方案B概率门；任一门失败立即停止，不启动其他seed，不调L、不换hazard/bias、不访问`locked_test`。

## 7. 证据边界

当前仍为candidate method，但已完成CPU接口实现。CPU证据不能证明召回会提高；20步log-odds累积漂移是预注册风险，只能由一次冻结GPU sentinel证伪。
