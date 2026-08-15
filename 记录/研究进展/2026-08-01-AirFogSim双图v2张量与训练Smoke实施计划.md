# AirFogSim双图v2张量与训练Smoke实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将三条AirFogSim双图v2开发轨迹转换为语义分离、mask完备的定长时序张量，并完成一次不承载性能结论的1-epoch训练smoke。

**Architecture:** 先在v2构图层过滤没有实际通信接口的孤立cloud并重建开发包，再由独立转换器为每个seed生成一次120时隙基础张量。Dataset根据现有8→3窗口索引惰性切片；小型mask-aware编码器只验证前向、masked loss、反向、checkpoint与跨seed推理。

**Tech Stack:** Python 3.10、NumPy、PyTorch 2.8 CPU、`unittest`、JSON/CSV/NPZ、现有PI-JWM v2构图与指标模块。

---

## 文件职责

- 修改`代码/src/pi_jwm/airfogsim_dual_graph_v2.py`：过滤无物理接口的cloud/edge-server类节点。
- 修改`代码/scripts/build_airfogsim_multiseed_v2.py`：只冻结v2图中保留节点的快照，并重建开发数据门。
- 新建`代码/src/pi_jwm/airfogsim_tensor_v2.py`：容量、排序、特征、索引、mask和单seed转换。
- 新建`代码/src/pi_jwm/airfogsim_tensor_dataset_v2.py`：加载每seed NPZ并按窗口惰性切片，计算train-only统计。
- 新建`代码/src/pi_jwm/airfogsim_smoke_model_v2.py`：mask-aware smoke模型与损失。
- 新建`代码/scripts/build_airfogsim_tensor_v2.py`：三seed张量生成、报告和manifest。
- 新建`代码/scripts/run_airfogsim_tensor_smoke_v2.py`：1 epoch训练、验证、checkpoint重载与smoke报告。
- 新建对应的四个测试文件，更新主文档与`本地计划表.md`。

### Task 1: 修正孤立cloud语义并重建开发包

**Files:**
- Modify: `代码/src/pi_jwm/airfogsim_dual_graph_v2.py`
- Modify: `代码/scripts/build_airfogsim_multiseed_v2.py`
- Modify: `代码/tests/test_airfogsim_dual_graph_v2.py`
- Modify: `代码/tests/test_build_airfogsim_multiseed_v2.py`

- [ ] **Step 1: 写失败测试，要求无接口cloud不进入双图**

```python
def test_excludes_cloud_without_an_incident_physical_edge(self):
    bundle = build_dual_graph_v2_bundle(
        trajectory_id="t0",
        physical_nodes=[
            {"id": "vehicle_0", "kind": "vehicle"},
            {"id": "RSU_0", "kind": "rsu"},
            {"id": "cloudServer_0", "kind": "cloud"},
        ],
        physical_edges=[{"id": "pe::vehicle_0::RSU_0", "src": "vehicle_0", "dst": "RSU_0"}],
        task_records=[], dag_edges=[], transfer_events=[],
    )
    self.assertEqual({"vehicle_0", "RSU_0"}, {row["id"] for row in bundle["physical_nodes"]})
    self.assertNotIn("agent::cloudServer_0", {row["id"] for row in bundle["information_nodes"]})
```

- [ ] **Step 2: 运行测试并确认因cloud仍存在而失败**

Run: `python -m unittest 代码/tests/test_airfogsim_dual_graph_v2.py`

Expected: FAIL，实际节点集合额外包含`cloudServer_0`。

- [ ] **Step 3: 实现最小过滤规则**

```python
NETWORK_ATTACHED_ONLY_KINDS = {"cloud", "edge_server"}

def _filter_physical_nodes(nodes, edges):
    incident = {str(row["src"]) for row in edges} | {str(row["dst"]) for row in edges}
    return [
        row for row in nodes
        if str(row.get("kind", "")).lower() not in NETWORK_ATTACHED_ONLY_KINDS
        or str(row["id"]) in incident
    ]
```

在`build_dual_graph_v2_bundle()`中先去重边，再过滤节点，再创建代理与CIP。车辆、UAV和RSU即使临时无业务仍保留；只有要求真实网络接口的cloud/edge-server按incident edge过滤。

- [ ] **Step 4: 写失败测试，要求多seed构建器同步过滤节点快照**

```python
graph = json.loads((output_dir / "seed_000" / "dual_graph_v2_bundle.json").read_text("utf-8"))
self.assertFalse(any(row["kind"] == "cloud" for row in graph["physical_nodes"]))
self.assertFalse(any(row["kind"] == "cloud" for row in graph["source_physical_node_snapshots"]))
```

- [ ] **Step 5: 在构建器中按保留节点ID过滤快照并运行测试**

```python
kept_node_ids = {str(row["id"]) for row in graph["physical_nodes"]}
graph["source_physical_node_snapshots"] = [
    row for row in source.get("physical_node_snapshots", [])
    if str(row.get("id")) in kept_node_ids
]
```

Run: `python -m unittest 代码/tests/test_airfogsim_dual_graph_v2.py 代码/tests/test_build_airfogsim_multiseed_v2.py`

Expected: PASS。

- [ ] **Step 6: 重建seeds 0/1/2开发包并验证容量**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n airfogsim python 代码/scripts/build_airfogsim_multiseed_v2.py --seeds 0 1 2 --max-time 12 --history 8 --horizon 3 --output-dir 代码/artifacts/datasets/airfogsim_multiseed_v2_dev
```

Expected: `development_dataset_ready=true`；最大物理节点18，物理边306，信息流216，任务163，DAG边422；三seed均无孤立cloud。

- [ ] **Step 7: 提交语义修复**

```powershell
git add 代码/src/pi_jwm/airfogsim_dual_graph_v2.py 代码/scripts/build_airfogsim_multiseed_v2.py 代码/tests/test_airfogsim_dual_graph_v2.py 代码/tests/test_build_airfogsim_multiseed_v2.py
git commit -m "fix: exclude disconnected cloud from AirFogSim v2 graph"
```

### Task 2: 实现单seed张量契约和转换器

**Files:**
- Create: `代码/src/pi_jwm/airfogsim_tensor_v2.py`
- Create: `代码/tests/test_airfogsim_tensor_v2.py`

- [ ] **Step 1: 写自然排序、容量和padding失败测试**

```python
def test_natural_task_order_and_padding_contract(self):
    contract = infer_tensor_contract([fake_graph(task_ids=["Task_10", "Task_2"])])
    arrays, report = tensorize_seed_graph(fake_graph(task_ids=["Task_10", "Task_2"]), contract)
    self.assertEqual(["Task_2", "Task_10"], report["task_vocab"])
    self.assertTrue(np.all(arrays["task_state"][:, 2:] == 0.0))
    self.assertTrue(np.all(arrays["task_node_index"][:, 2:] == -1))
```

- [ ] **Step 2: 运行测试并确认模块缺失**

Run: `python -m unittest 代码/tests/test_airfogsim_tensor_v2.py`

Expected: ERROR，`pi_jwm.airfogsim_tensor_v2`不存在。

- [ ] **Step 3: 实现常量、契约和稳定排序**

```python
NODE_FEATURES = ("x", "y", "z", "speed", "acceleration", "cpu", "storage")
EDGE_FEATURES = ("distance", "csi_mean", "rate_sum", "active_task_count", "allocated_rb_count")
FLOW_FEATURES = ("total_data", "remaining_data", "delivered_cumulative", "delivered_this_slot", "age")
TASK_FEATURES = ("task_size", "return_size", "task_cpu", "deadline_remaining", "priority", "transmitted", "computed", "delay")
LIFECYCLE_TYPES = ("to_offload", "computing", "returning", "finished", "failed")

def natural_id_key(value):
    prefix, _, suffix = str(value).rpartition("_")
    return prefix, int(suffix) if suffix.isdigit() else suffix
```

`infer_tensor_contract()`计算三个seed的最大节点、物理边、信息流、任务、DAG容量，写入特征名、类别顺序、history=8、horizon=3和schema版本。

- [ ] **Step 4: 写任务因果mask失败测试**

```python
def test_hides_to_generate_task_before_arrival(self):
    arrays, _ = tensorize_seed_graph(fake_graph_with_prearrival_task(), contract(max_tasks=1))
    self.assertEqual(0, arrays["task_present"][0, 0])
    self.assertTrue(np.all(arrays["task_state"][0, 0] == 0.0))
    self.assertEqual(1, arrays["task_present"][1, 0])
```

- [ ] **Step 5: 实现节点、物理边和任务张量**

为每个seed建立120步时间索引；拒绝重复`(time,id)`。任务只有`arrival_time <= observed_time`且生命周期不是`to_generate`时进入`task_present`，到达前所有状态和节点索引保持padding。

- [ ] **Step 6: 写信息流时序重建失败测试**

```python
def test_reconstructs_flow_from_action_and_direct_events(self):
    arrays, report = tensorize_seed_graph(fake_flow_graph(), contract(max_flows=1))
    self.assertEqual([0, 1, 1, 0], arrays["flow_present"][:, 0].tolist())
    self.assertEqual([0.0, 0.0, 0.4, 0.0], arrays["flow_delivered_this_slot"][:, 0].tolist())
    self.assertEqual(0, report["flow_creation_fallback_count"])
```

- [ ] **Step 7: 实现flow、动作、DAG和耦合索引**

动作按`(time,task_id)`对齐到`task_action[T,Q,5] = [offload, rb, return, rb_count, rb_fraction]`，节点端点另存`task_action_node_index[T,Q,3]`。CIP保存`agent_node_index[N]`；历史CFE保存`flow_bearer_edge_index[T,F]`，无承载为`-1`。DAG保存`dag_edge_index[2,D]`和`dag_edge_valid[D]`。

- [ ] **Step 8: 实现严格验证器并运行全部转换器测试**

`validate_seed_tensors()`检查：形状、有限值、padding零、padding索引`-1`、端点范围、task到达前mask、流剩余量非负、容量未截断和120步均匀时间轴。

Run: `python -m unittest 代码/tests/test_airfogsim_tensor_v2.py`

Expected: PASS。

- [ ] **Step 9: 提交转换器**

```powershell
git add 代码/src/pi_jwm/airfogsim_tensor_v2.py 代码/tests/test_airfogsim_tensor_v2.py
git commit -m "feat: add AirFogSim dual graph v2 tensor contract"
```

### Task 3: 实现惰性窗口Dataset与train-only统计

**Files:**
- Create: `代码/src/pi_jwm/airfogsim_tensor_dataset_v2.py`
- Create: `代码/tests/test_airfogsim_tensor_dataset_v2.py`

- [ ] **Step 1: 写窗口切片和split失败测试**

```python
def test_slices_eight_history_and_three_future_steps(self):
    dataset = AirFogSimTensorWindowDataset(root, split="dev_train", stats=None)
    sample = dataset[0]
    self.assertEqual((8, 18, len(NODE_FEATURES)), sample["history"]["node_state"].shape)
    self.assertEqual((3, 18, len(NODE_FEATURES)), sample["target"]["node_state"].shape)
    self.assertEqual(0, sample["seed"])
```

- [ ] **Step 2: 运行测试并确认模块缺失**

Run: `python -m unittest 代码/tests/test_airfogsim_tensor_dataset_v2.py`

Expected: ERROR，Dataset模块不存在。

- [ ] **Step 3: 实现按seed缓存NPZ和窗口惰性切片**

```python
class AirFogSimTensorWindowDataset(Dataset):
    def __getitem__(self, index):
        row = self.windows[index]
        arrays = self._load_seed(int(row["seed"]))
        return slice_window(arrays, row, self.stats)
```

Dataset只加载当前split对应seed；每seed NPZ在首次访问后缓存。history返回状态、mask、历史动作和历史CIP/CFE；condition返回未来3步实际任务动作；target返回未来状态、存在标签和连续状态mask，不返回未来CFE作为输入。

- [ ] **Step 4: 写归一化泄漏失败测试**

```python
stats = fit_training_stats(root, split="dev_train")
self.assertEqual([0], stats["source_seeds"])
self.assertNotEqual(float(stats["node_state"]["mean"][0]), fake_validation_only_value)
```

- [ ] **Step 5: 实现masked train-only统计和collate**

均值/标准差仅在`dev_train`且对应present/observed mask为1的位置计算；标准差小于`1e-6`置1。`collate_tensor_windows()`堆叠嵌套字典并保留静态索引。

- [ ] **Step 6: 运行测试并提交**

Run: `python -m unittest 代码/tests/test_airfogsim_tensor_dataset_v2.py`

Expected: PASS。

```powershell
git add 代码/src/pi_jwm/airfogsim_tensor_dataset_v2.py 代码/tests/test_airfogsim_tensor_dataset_v2.py
git commit -m "feat: add lazy AirFogSim v2 tensor dataset"
```

### Task 4: 构建三seed张量产物

**Files:**
- Create: `代码/scripts/build_airfogsim_tensor_v2.py`
- Create: `代码/tests/test_build_airfogsim_tensor_v2.py`

- [ ] **Step 1: 写脚本级失败测试**

```python
result = build_tensor_dataset(source_dir, output_dir)
self.assertTrue(result["tensor_dataset_ready"])
self.assertEqual(3, result["seed_count"])
self.assertTrue((output_dir / "tensor_contract.json").is_file())
self.assertTrue((output_dir / "seed_000" / "trajectory_tensors.npz").is_file())
```

- [ ] **Step 2: 运行测试并确认脚本缺失**

Run: `python -m unittest 代码/tests/test_build_airfogsim_tensor_v2.py`

Expected: ERROR，构建脚本不存在。

- [ ] **Step 3: 实现构建入口、报告和manifest**

构建器读取三个`dual_graph_v2_bundle.json`和顶层`window_index.csv`，推断并冻结契约，逐seed写NPZ、词表JSON和转换报告；顶层写train-only统计、验证报告、README和SHA-256 manifest。任何seed验证失败时`tensor_dataset_ready=false`并退出1。

- [ ] **Step 4: 运行单元测试与真实构建**

Run:

```powershell
python -m unittest 代码/tests/test_build_airfogsim_tensor_v2.py
python 代码/scripts/build_airfogsim_tensor_v2.py --source-dir 代码/artifacts/datasets/airfogsim_multiseed_v2_dev --output-dir 代码/artifacts/datasets/airfogsim_tensor_v2_dev
```

Expected: 3个seed、330个窗口、容量18/306/216/163/422，全部tensor验证门通过。

- [ ] **Step 5: 提交构建器**

```powershell
git add 代码/scripts/build_airfogsim_tensor_v2.py 代码/tests/test_build_airfogsim_tensor_v2.py
git commit -m "feat: build AirFogSim v2 tensor development dataset"
```

### Task 5: 实现mask-aware smoke模型与损失

**Files:**
- Create: `代码/src/pi_jwm/airfogsim_smoke_model_v2.py`
- Create: `代码/tests/test_airfogsim_smoke_model_v2.py`

- [ ] **Step 1: 写padding不影响输出和损失的失败测试**

```python
output_a = model(batch)
batch_with_changed_padding = change_only_padding_values(batch, value=999.0)
output_b = model(batch_with_changed_padding)
self.assertTrue(torch.allclose(output_a["task_progress"], output_b["task_progress"]))
self.assertTrue(torch.isfinite(compute_smoke_loss(output_a, batch["target"])["total"]))
```

- [ ] **Step 2: 运行测试并确认模块缺失**

Run: `python -m unittest 代码/tests/test_airfogsim_smoke_model_v2.py`

Expected: ERROR，smoke模型模块不存在。

- [ ] **Step 3: 实现masked pooling与三步动作条件GRU**

```python
def masked_mean(values, mask, dim):
    weight = mask.to(values.dtype).unsqueeze(-1)
    return (values * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)
```

模型分别编码五类历史输入，使用最后历史时隙的有效对象与时间均值形成context；未来任务动作masked pooling后逐步输入GRU。输出节点状态、物理边活动/速率、信息流存在/剩余量和任务存在/进度，形状严格匹配target。

- [ ] **Step 4: 实现分项masked loss**

二元存在/活动使用BCEWithLogits并在词表有效槽位计算；连续状态使用present与observed mask的交集计算MSE。总损失固定为各个有限分项的等权均值，只用于执行链smoke。

- [ ] **Step 5: 写反向梯度有限测试并运行**

```python
losses["total"].backward()
self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
```

Run: `python -m unittest 代码/tests/test_airfogsim_smoke_model_v2.py`

Expected: PASS。

- [ ] **Step 6: 提交模型**

```powershell
git add 代码/src/pi_jwm/airfogsim_smoke_model_v2.py 代码/tests/test_airfogsim_smoke_model_v2.py
git commit -m "feat: add mask aware AirFogSim tensor smoke model"
```

### Task 6: 运行1 epoch smoke并同步文档

**Files:**
- Create: `代码/scripts/run_airfogsim_tensor_smoke_v2.py`
- Create: `代码/tests/test_run_airfogsim_tensor_smoke_v2.py`
- Modify: `本地计划表.md`
- Modify: `D:\禹尧珅\人工智能知识库\北大科研\PIJWM\PIJWM推进.md`

- [ ] **Step 1: 写runner失败测试**

```python
result = run_smoke(tensor_dir, output_dir, epochs=1, batch_size=8, hidden_dim=16, seed=0)
self.assertTrue(result["smoke_passed"])
self.assertFalse(result["formal_training_ready"])
self.assertTrue(result["checkpoint_reload_passed"])
self.assertTrue(math.isfinite(result["train_loss"]))
self.assertTrue(math.isfinite(result["validation_loss"]))
```

- [ ] **Step 2: 运行测试并确认runner缺失**

Run: `python -m unittest 代码/tests/test_run_airfogsim_tensor_smoke_v2.py`

Expected: ERROR，runner脚本不存在。

- [ ] **Step 3: 实现确定性CPU训练、验证和checkpoint审计**

固定Python/NumPy/Torch seed，使用Adam、1 epoch、梯度有限检查。保存`checkpoint.pt`、配置、逐分项loss、checkpoint SHA-256、重载后同batch输出一致性、validation报告和README。报告明确`performance_claim_allowed=false`。

- [ ] **Step 4: 运行测试与真实smoke**

Run:

```powershell
python -m unittest 代码/tests/test_run_airfogsim_tensor_smoke_v2.py
python 代码/scripts/run_airfogsim_tensor_smoke_v2.py --tensor-dir 代码/artifacts/datasets/airfogsim_tensor_v2_dev --output-dir 代码/artifacts/small_experiments/exp07_airfogsim_tensor_smoke_v2 --epochs 1 --batch-size 8 --hidden-dim 32 --seed 0
```

Expected: `smoke_passed=true`、`checkpoint_reload_passed=true`、train/validation loss有限、`formal_training_ready=false`。

- [ ] **Step 5: 更新文档**

主文档记录修正后的三seed节点数量、张量容量、mask规则和smoke事实；`本地计划表.md`把张量与执行链标为完成，把扩充seed与JEPA三臂对照列为下一步。不得使用“性能提升”“泛化有效”等措辞。

- [ ] **Step 6: 运行完整相关验证**

```powershell
python -m unittest 代码/tests/test_airfogsim_dual_graph_v2.py 代码/tests/test_build_airfogsim_multiseed_v2.py 代码/tests/test_airfogsim_tensor_v2.py 代码/tests/test_airfogsim_tensor_dataset_v2.py 代码/tests/test_build_airfogsim_tensor_v2.py 代码/tests/test_airfogsim_smoke_model_v2.py 代码/tests/test_run_airfogsim_tensor_smoke_v2.py
python -m compileall -q 代码/src/pi_jwm 代码/scripts
git diff --check
```

Expected: 全部通过，无编译或空白错误；张量和smoke manifest哈希全部匹配。

- [ ] **Step 7: 提交runner与文档**

```powershell
git add 代码/scripts/run_airfogsim_tensor_smoke_v2.py 代码/tests/test_run_airfogsim_tensor_smoke_v2.py 本地计划表.md
git commit -m "test: complete AirFogSim v2 tensor training smoke"
```

外部知识库主文档不属于当前Git仓库，只更新并核对，不加入提交。
