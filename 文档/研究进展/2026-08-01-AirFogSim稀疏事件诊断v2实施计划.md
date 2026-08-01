# AirFogSim稀疏事件诊断v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为AirFogSim双图张量加入简单基线、训练集类别权重、链路活动预测和任务生命周期预测，并形成四臂公平诊断结果。

**Architecture:** 保持现有tensor-v2数据不变，在窗口数据层统计训练标签，在最小world model中增加两个预测头，在独立诊断模块中统一基线输出和评价逻辑，最后由新runner以相同split和配置运行四个实验臂。所有新增统计只读取`dev_train`，exp07保留不覆盖，新结果写入exp08。

**Tech Stack:** Python 3、NumPy、PyTorch、`unittest`、JSON/Markdown/SHA-256产物。

---

## File Structure

- Modify `代码/src/pi_jwm/airfogsim_window_dataset_v2.py`: 显式链路活动标签、训练split稀疏标签计数、正类权重和生命周期多数类。
- Modify `代码/src/pi_jwm/airfogsim_smoke_model_v2.py`: 链路活动头、任务生命周期头及加权损失。
- Create `代码/src/pi_jwm/airfogsim_sparse_diagnostics_v2.py`: 零活动/持久性基线、AUPRC和统一评价。
- Create `代码/scripts/run_airfogsim_sparse_event_diagnostic_v2.py`: 四臂训练、评价和产物生成。
- Modify `代码/tests/test_airfogsim_window_dataset_v2.py`: train-only统计与权重上限测试。
- Modify `代码/tests/test_airfogsim_smoke_model_v2.py`: 新输出、mask和加权损失测试。
- Create `代码/tests/test_airfogsim_sparse_diagnostics_v2.py`: 基线与AUPRC测试。
- Create `代码/tests/test_run_airfogsim_sparse_event_diagnostic_v2.py`: 端到端四臂产物测试。
- Modify `D:\禹尧珅\人工智能知识库\北大科研\PIJWM\PIJWM推进.md`, `本地计划表.md`, `task_plan.md`, `progress.md`, `findings.md`: 记录结果和边界。

### Task 1: Training-only sparse label statistics

**Files:**
- Modify: `代码/src/pi_jwm/airfogsim_window_dataset_v2.py`
- Test: `代码/tests/test_airfogsim_window_dataset_v2.py`

- [ ] **Step 1: Write the failing tests**

增加两个测试：验证seed1验证数据不会进入统计；验证极稀疏标签的权重被限制为50。

```python
stats = fit_sparse_label_stats(root, split="dev_train", max_pos_weight=50.0)
self.assertEqual("dev_train", stats["source_split"])
self.assertEqual(0, stats["labels"]["link_activity"]["positive_count"])
self.assertNotEqual(100.0, stats["labels"]["task_present"]["positive_rate"])
self.assertLessEqual(stats["labels"]["flow_present"]["pos_weight"], 50.0)
self.assertIn("majority_index", stats["task_lifecycle"])
sample = AirFogSimTensorWindowDataset(root, split="dev_train")[0]
self.assertEqual(torch.bool, sample["target"]["link_activity"].dtype)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest 代码/tests/test_airfogsim_window_dataset_v2.py
```

Expected: FAIL because `fit_sparse_label_stats` does not exist.

- [ ] **Step 3: Implement the minimal statistics API**

Add:

```python
def fit_sparse_label_stats(root: str | Path, *, split: str = "dev_train", max_pos_weight: float = 50.0) -> dict[str, Any]:
    # Iterate only label slices named by window_index rows in split.
    # Count valid positives/negatives for link activity, flow presence and task presence.
    # Count lifecycle labels where task_present and lifecycle_index >= 0.
    # pos_weight = clip(negative / positive, 1, max_pos_weight); use 1 when positive=0.
    return {
        "schema_version": "PI-JWM-AirFogSim-sparse-label-stats-v2",
        "source_split": split,
        "labels": label_reports,
        "task_lifecycle": lifecycle_report,
    }
```

`AirFogSimTensorWindowDataset.__getitem__`在归一化前由原始`physical_edge_state[:, :, EDGE_FEATURES.index("active_task_count")] > 0`生成历史和目标`link_activity`布尔张量。统计只在`physical_edge_present`内使用该标签；flow/task使用各自的`valid`词表mask作为分母，不计padding槽位。

- [ ] **Step 4: Run GREEN and regression tests**

```powershell
python -m unittest 代码/tests/test_airfogsim_window_dataset_v2.py 代码/tests/test_airfogsim_tensor_v2.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add 代码/src/pi_jwm/airfogsim_window_dataset_v2.py 代码/tests/test_airfogsim_window_dataset_v2.py
git commit -m "feat: add train-only sparse label statistics"
```

### Task 2: Link activity and task lifecycle heads

**Files:**
- Modify: `代码/src/pi_jwm/airfogsim_smoke_model_v2.py`
- Modify: `代码/tests/test_airfogsim_smoke_model_v2.py`

- [ ] **Step 1: Write failing output and loss tests**

```python
output = model(history)
self.assertEqual((2, 2, 5), tuple(output["link_activity_logits"].shape))
self.assertEqual((2, 2, 6, 5), tuple(output["task_lifecycle_logits"].shape))

loss, metrics = dual_graph_world_model_loss(
    output,
    target,
    static,
    sparse_pos_weights={"link_activity": 10.0, "flow_present": 5.0, "task_present": 2.0},
)
self.assertIn("link_activity_bce", metrics)
self.assertIn("task_lifecycle_ce", metrics)
```

再将padding任务的生命周期目标改成任意类别，断言损失不变。

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m unittest 代码/tests/test_airfogsim_smoke_model_v2.py
```

Expected: FAIL because both output heads and weighted arguments are missing.

- [ ] **Step 3: Implement both heads**

Make `_EntityRolloutHead.forward` return the latent tensor in addition to state/presence. Add:

```python
self.link_activity_head = nn.Linear(hidden_dim, 1)
self.task_lifecycle_head = nn.Linear(hidden_dim, 5)

output["link_activity_logits"] = self.link_activity_head(edge_latent).squeeze(-1)
output["task_lifecycle_logits"] = self.task_lifecycle_head(task_latent)
```

Extend loss with fixed composition:

```python
total = state_loss + 0.1 * presence_loss + 0.1 * activity_loss + 0.1 * lifecycle_loss
```

Use `binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)` for link activity, flow presence and task presence. Link activity直接使用数据集提供的`target["link_activity"]`，不从归一化连续状态反推。Lifecycle uses cross entropy only where `task_present & (task_lifecycle_index >= 0)`.

- [ ] **Step 4: Run GREEN and backward test**

```powershell
python -m unittest 代码/tests/test_airfogsim_smoke_model_v2.py
```

Expected: all tests pass and gradients are finite.

- [ ] **Step 5: Commit**

```powershell
git add 代码/src/pi_jwm/airfogsim_smoke_model_v2.py 代码/tests/test_airfogsim_smoke_model_v2.py
git commit -m "feat: predict sparse activity and task lifecycle"
```

### Task 3: Shared baselines and diagnostic metrics

**Files:**
- Create: `代码/src/pi_jwm/airfogsim_sparse_diagnostics_v2.py`
- Create: `代码/tests/test_airfogsim_sparse_diagnostics_v2.py`

- [ ] **Step 1: Write failing baseline tests**

```python
zero = build_zero_activity_prediction(batch, stats, lifecycle_majority_index=1)
self.assertTrue(torch.all(zero["link_activity_logits"] < 0))
self.assertTrue(torch.all(zero["flow_presence_logits"] < 0))
self.assertTrue(torch.all(zero["task_presence_logits"] < 0))

persistent = build_last_persistence_prediction(batch, horizon_steps=2)
torch.testing.assert_close(persistent["task_state"][:, 0], batch["history"]["task_state"][:, -1])
torch.testing.assert_close(persistent["task_state"][:, 1], batch["history"]["task_state"][:, -1])
```

Add exact AUPRC cases:

```python
self.assertAlmostEqual(1.0, average_precision([0.9, 0.8, 0.1], [1, 1, 0]))
self.assertIsNone(average_precision([0.1, 0.2], [0, 0]))
self.assertAlmostEqual(2 / 3, average_precision([0.0, 0.0, 0.0], [1, 0, 1]))
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m unittest 代码/tests/test_airfogsim_sparse_diagnostics_v2.py
```

Expected: FAIL with module not found.

- [ ] **Step 3: Implement prediction and evaluation interfaces**

Implement these exact public APIs:

```python
def build_zero_activity_prediction(
    batch: Mapping[str, Any],
    stats: Mapping[str, Any],
    *,
    lifecycle_majority_index: int,
) -> dict[str, torch.Tensor]:
    prediction = build_last_persistence_prediction(batch, horizon_steps=batch["target"]["node_state"].shape[1])
    prediction["link_activity_logits"].fill_(-20.0)
    prediction["flow_presence_logits"].fill_(-20.0)
    prediction["task_presence_logits"].fill_(-20.0)
    prediction["flow_state"] = normalized_raw_zero_like(prediction["flow_state"], stats["features"]["flow_state"])
    prediction["task_state"] = normalized_raw_zero_like(prediction["task_state"], stats["features"]["task_state"])
    force_raw_edge_features_to_zero(prediction["physical_edge_state"], stats, ("rate_sum", "active_task_count", "allocated_rb_count"))
    prediction["task_lifecycle_logits"].fill_(-20.0)
    prediction["task_lifecycle_logits"][:, :, :, lifecycle_majority_index] = 20.0
    return prediction


def build_last_persistence_prediction(
    batch: Mapping[str, Any],
    *,
    horizon_steps: int,
) -> dict[str, torch.Tensor]:
    prediction: dict[str, torch.Tensor] = {}
    for name in ("node", "physical_edge", "flow", "task"):
        prediction[f"{name}_state"] = batch["history"][f"{name}_state"][:, -1:, :, :].repeat(1, horizon_steps, 1, 1)
        present = batch["history"][f"{name}_present"][:, -1:, :].repeat(1, horizon_steps, 1)
        prediction[f"{name}_presence_logits"] = torch.where(present, 20.0, -20.0)
    activity = batch["history"]["link_activity"][:, -1:, :].repeat(1, horizon_steps, 1)
    prediction["link_activity_logits"] = torch.where(activity, 20.0, -20.0)
    prediction["task_lifecycle_logits"] = lifecycle_logits_from_last_history(batch, horizon_steps)
    return prediction


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.sum() == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    labels, scores = labels[order], scores[order]
    cumulative_true = np.cumsum(labels)
    threshold_ends = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
    true_at_threshold = cumulative_true[threshold_ends]
    precision = true_at_threshold / (threshold_ends + 1)
    recall = true_at_threshold / labels.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def evaluate_prediction_batches(
    prediction_batches: Iterable[Mapping[str, torch.Tensor]],
    batches: Iterable[Mapping[str, Any]],
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    accumulator = SparseDiagnosticAccumulator(stats)
    for prediction, batch in zip(prediction_batches, batches):
        accumulator.update(prediction, batch["target"], batch["static"])
    return accumulator.finalize()
```

`normalized_raw_zero_like`、`force_raw_edge_features_to_zero`、`lifecycle_logits_from_last_history`和`SparseDiagnosticAccumulator`均为同模块私有实现；前3个只负责基线构造，accumulator只负责累计和最终化指标，不能读取split或训练数据。

All arms must feed `evaluate_prediction_batches`; it reports link activity precision/recall/F1/AUPRC, active-only rate MAE/RMSE, component presence F1, lifecycle accuracy/macro-F1/support and inverse-normalized state MAE. AUPRC按唯一分数阈值分组计算；所有分数相同时结果等于正类比例，不依赖样本原始顺序。

- [ ] **Step 4: Run GREEN**

```powershell
python -m unittest 代码/tests/test_airfogsim_sparse_diagnostics_v2.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add 代码/src/pi_jwm/airfogsim_sparse_diagnostics_v2.py 代码/tests/test_airfogsim_sparse_diagnostics_v2.py
git commit -m "feat: add sparse event baselines and metrics"
```

### Task 4: Four-arm diagnostic runner

**Files:**
- Create: `代码/scripts/run_airfogsim_sparse_event_diagnostic_v2.py`
- Create: `代码/tests/test_run_airfogsim_sparse_event_diagnostic_v2.py`

- [ ] **Step 1: Write the failing end-to-end test**

Build a temporary tensor fixture and assert:

```python
result = run_diagnostic(
    dataset_dir=tensor,
    output_dir=output,
    epochs=1,
    batch_size=1,
    hidden_dim=8,
    eval_splits=("dev_validation",),
)
self.assertEqual(
    {"zero_activity", "last_persistence", "learned_unweighted", "learned_balanced"},
    set(result["arms"]),
)
self.assertIn("link_activity", json.loads((output / "evaluation.json").read_text())["learned_balanced"]["dev_validation"])
self.assertTrue((output / "learned_balanced_model.pt").is_file())
```

- [ ] **Step 2: Run test and verify RED**

```powershell
python -m unittest 代码/tests/test_run_airfogsim_sparse_event_diagnostic_v2.py
```

Expected: FAIL because runner does not exist.

- [ ] **Step 3: Implement deterministic four-arm execution**

Add `run_diagnostic` with the signature shown in Step 1 that:

1. loads train-only normalization and sparse-label statistics;
2. evaluates both nonlearned baselines;
3. recreates model, optimizer, data generator and initial seed independently for each learned arm;
4. trains both learned arms for the same number of epochs, with weights disabled/enabled only as specified;
5. evaluates every arm on identical split windows through the shared evaluator;
6. writes `config.json`, `class_stats.json`, `training_history.json`, `evaluation.json`, `summary.json`, two model files, `REPORT.md` and `manifest.json`.

- [ ] **Step 4: Run GREEN and runner regressions**

```powershell
python -m unittest 代码/tests/test_run_airfogsim_sparse_event_diagnostic_v2.py 代码/tests/test_run_airfogsim_tensor_smoke_v2.py
```

Expected: all tests pass; exp07 behavior remains unchanged.

- [ ] **Step 5: Commit**

```powershell
git add 代码/scripts/run_airfogsim_sparse_event_diagnostic_v2.py 代码/tests/test_run_airfogsim_sparse_event_diagnostic_v2.py
git commit -m "feat: run four-arm sparse event diagnostic"
```

### Task 5: Real five-epoch run, documentation and verification

**Files:**
- Generate: `代码/artifacts/small_experiments/exp08_airfogsim_sparse_event_diagnostic_v2/`
- Modify: `D:\禹尧珅\人工智能知识库\北大科研\PIJWM\PIJWM推进.md`
- Modify: `本地计划表.md`
- Modify: `task_plan.md`
- Modify: `progress.md`
- Modify: `findings.md`

- [ ] **Step 1: Run the real diagnostic**

```powershell
$env:PYTHONIOENCODING='utf-8'
python 代码/scripts/run_airfogsim_sparse_event_diagnostic_v2.py `
  --dataset-dir 代码/artifacts/datasets/airfogsim_tensor_v2_dev `
  --output-dir 代码/artifacts/small_experiments/exp08_airfogsim_sparse_event_diagnostic_v2 `
  --epochs 5 --batch-size 8 --hidden-dim 32 --learning-rate 0.001 --random-seed 2026
```

Expected: `diagnostic_ready=true`, four arms and two evaluation splits present.

- [ ] **Step 2: Check comparison integrity**

Verify both learned arms use identical initialization hash, data order seed, epochs and samples; confirm class stats source is `dev_train`; recompute every manifest hash.

- [ ] **Step 3: Update documents with measured results**

Add the four-arm table and go/no-go conclusion. Do not call an arm better unless it exceeds both simple baselines on the named metric; preserve `formal_training_ready=false` and JEPA hold.

- [ ] **Step 4: Run full related verification**

```powershell
python -m unittest `
  代码/tests/test_airfogsim_tensor_v2.py `
  代码/tests/test_airfogsim_window_dataset_v2.py `
  代码/tests/test_build_airfogsim_tensor_v2.py `
  代码/tests/test_airfogsim_smoke_model_v2.py `
  代码/tests/test_airfogsim_sparse_diagnostics_v2.py `
  代码/tests/test_run_airfogsim_tensor_smoke_v2.py `
  代码/tests/test_run_airfogsim_sparse_event_diagnostic_v2.py
python -m compileall -q 代码/src/pi_jwm 代码/scripts
git diff --check
```

Expected: zero failures and exit code 0.

- [ ] **Step 5: Record implementation state**

检查`git diff`并只提交能与既有dirty-worktree修改明确分离的直接相关文件。`PIJWM推进.md`位于仓库外，只更新不提交；`本地计划表.md`、`task_plan.md`、`progress.md`和`findings.md`若包含本任务开始前的未提交内容，则保留在工作区并在交付中说明，不把旧修改一并吸收到新提交。

```powershell
git status --short
git diff -- 本地计划表.md task_plan.md progress.md findings.md
```
