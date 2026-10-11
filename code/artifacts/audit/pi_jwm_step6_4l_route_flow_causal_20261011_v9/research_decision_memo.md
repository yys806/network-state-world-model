# STEP 6.4L Research Decision Memo

## 结论

1. 冻结 World Model 在 6.4K 的受控 Shadow Flow 案例可以完成 H1–H4；但对真实 Task_5 Route 新生 Flow，本轮未通过模型输入支持门禁，不能称为真实 Route 后 H4 兼容。
2. AirFogSim 与 Ledger 的 Route birth 因果关系已对齐：真实 setter 成功，官方 Ledger 建立 `Return::0`。真实传输仍未发生，说明 Route birth 不是 Return completion。
3. 原 Scorer 本轮没有产生合法 SCOREABLE A/B。A 在 Planner domain admission 处异常；B 在 Flow 注入前因缺失 `UAV_0→UAV_1` communication support fail-closed。
4. 不重训尚缺 Route/Comm/Flow 状态合同、模型动态 slot 支持和正式 Planner/Scorer 资格验证；没有充分证据判断必须重训。

## 研究者需决定

- 是否允许正式 Planner 产生 Return Route，以及 Route 与 Comm action 的同一决策顺序。
- 新 Flow 是否只允许已有通信边，还是允许动态建立通信边。
- 新 Flow slot 的 hidden 初始化、训练数据覆盖和 Scorer cohort 语义。

本备忘录是候选方案分析，不是科学方案实施。
