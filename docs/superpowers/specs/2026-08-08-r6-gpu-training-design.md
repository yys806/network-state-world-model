# R6 在线联合策略 GPU 训练设计

## 目标与边界

R6只训练联合候选策略器，冻结R5.1候选B Graph-RSSM。AirFogSim是环境和反馈来源，PI-JWM是双图状态、世界模型与策略接口。训练只使用train，validation只选checkpoint，calibration只做后续阈值/不确定性校准，locked-test在R9前拒绝访问。

## 状态闭环

策略状态必须来自当前动作轨迹，而不是默认教师轨迹。每步固定为：

1. 从当前AirFogSim环境生成6个合法卸载/RB/CPU联合候选；
2. 策略选择并执行候选；
3. AirFogSim推进一步并返回任务、通信、资源和能耗事实；
4. 重新采集节点、通信链路、任务/DAG、传输事件和实际动作；
5. 用冻结R1张量契约把最近8步重映射为严格双图显式状态；
6. 冻结Graph-RSSM由在线显式历史生成隐式belief；
7. 由AirFogSim实际反馈计算服务优先reward，再更新Actor–Critic或PPO。

冻结教师数据只提供对象容量、特征顺序、train-only归一化和协议绑定。禁止在动作分叉后读取教师轨迹状态值。反事实reward代理只有默认动作标签，候选覆盖为1/6，故不得用于正式训练。

## 方法矩阵

矩阵为`{actor_critic, ppo_clipped} × {explicit_only, latent_only, explicit_latent} × {20260803,20260804,20260805}`，共18个run。统一预算上限100,000环境步，rollout 128，minibatch 32，PPO epoch 4，学习率`3e-4`，梯度裁剪`0.5`，每10,000步做validation，patience 5。MPC不进入首轮矩阵。

## 分阶段和恢复

GPU开启后先运行单run 2,000步smoke，检查CUDA显存、吞吐、非默认动作、在线状态变化、真实reward、梯度、参数变化、世界模型哈希、硬约束、checkpoint重载和锁定集边界。smoke通过后用18-run启动器先训练到10,000步；只有训练曲线有限、validation gate正常、失败率和资源成本可接受，才从同一检查点续到100,000步。

每次策略更新后原子保存策略、优化器、步数、run游标、validation gate、候选计数和统计。每个run使用独立目录与日志；完成run跳过，失败run保留，不替换seed。启动器默认6并发，实际并发数由GPU smoke的显存和吞吐决定。

## 本地进入门

进入GPU前必须同时满足：六种方法—状态组合均有真实非零更新；至少执行一个非默认候选；在线显式状态随环境变化；32步闭环覆盖6类候选、CPU动作和通信事件；硬违规为0；同seed复跑一致；2→4步续训连续；validation-only best checkpoint可写入；18-run命令互相隔离；世界模型不变；生成证据不访问locked-test。

本地门禁证据为`代码/artifacts/preflight/pi_jwm_r6_online_gpu_readiness_v2/`。该门只批准GPU smoke，不代表策略性能、收敛或最终方法定型。
