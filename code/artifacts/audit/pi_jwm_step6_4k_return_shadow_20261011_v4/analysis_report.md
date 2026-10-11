# STEP 6.4K Validation Flow Audit

完整 12 条 `dev_validation` Raw 按官方 Causal Flow Ledger 重放。旧 721/724 是上一帧 Route action 代理口径；Ledger 复核结果为 710 条 waiting-to-return 无 Return Flow，713 条 FP32 潜在 fixed-support，710 条 FP64 潜在 fixed-support。8460 条 task records 已知 Return Flow，5675 条仍 active，2785 条 completed，3 条 FP64/FP32 冲突。1237 条缺失 `required_returned_size` 的记录 fail-closed，主要来自 failed/terminal Raw，未补造状态。

独立审计从 12 个 ledger 文件重新计算上述计数，PASS。统计是离线 Raw/Flow 状态，不能等同 Scorer 拒绝，也不能还原 r39 的 3744 个候选原因。
