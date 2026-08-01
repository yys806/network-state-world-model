# PI-JWM 正式 AirFogSim 数据集 v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建包含 60 条 AirFogSim 轨迹、四个互斥 split、原生 DAG 状态和多策略逐任务 CPU 动作的 PI-JWM 正式仿真数据集 v1。

**Architecture:** 在 `代码/src/pi_jwm/` 中新增正式数据协议和 CPU 策略模块，以现有 AirFogSim 事件采集器为只读数据源，通过参数化适配层生成不同场景和 CPU 策略的轨迹。权威轨迹层继续使用双图 v2 稀疏契约，并把 CPU 动作和 DAG 派生状态送入张量层；正式构建器负责 split 隔离、锁定测试保护、manifest 和验收，不修改 AirFogSim 核心代码。

**Tech Stack:** Python 3、NumPy、PyTorch、AirFogSim、NetworkX、`unittest`、JSON/CSV/NPZ、SHA-256。

---

### Task 1: 冻结 60 轨迹协议、场景和 split

**Files:**
- Create: `代码/src/pi_jwm/formal_airfogsim_dataset_v1.py`
- Create: `代码/tests/test_formal_airfogsim_dataset_v1.py`

- [ ] **Step 1: 写协议生成失败测试**

```python
def test_default_protocol_has_balanced_60_trajectories():
    specs = build_formal_trajectory_specs()
    assert len(specs) == 60
    assert Counter(row.split for row in specs) == {
        "train": 36, "validation": 12, "calibration": 6, "locked_test": 6
    }
    assert Counter(row.cpu_policy for row in specs) == {
        "equal_share": 20, "deadline_aware": 20, "feasible_exploration": 20
    }
```

- [ ] **Step 2: 运行测试并确认缺少模块而失败**

Run: `python -m unittest 代码/tests/test_formal_airfogsim_dataset_v1.py -v`
Expected: FAIL，提示 `pi_jwm.formal_airfogsim_dataset_v1` 不存在。

- [ ] **Step 3: 实现协议数据类和确定性生成规则**

```python
@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    load_level: str
    density_level: str
    task_lambda: float
    max_vehicles: int
    vehicle_arrival_lambda: float

@dataclass(frozen=True)
class TrajectorySpec:
    trajectory_id: str
    seed: int
    repetition: int
    split: str
    cpu_policy: str
    scenario: ScenarioSpec

def build_formal_trajectory_specs(scenarios=DEFAULT_SCENARIOS):
    splits = ("train",) * 6 + ("validation",) * 2 + ("calibration", "locked_test")
    policies = ("equal_share", "deadline_aware", "feasible_exploration")
    return [
        TrajectorySpec(
            trajectory_id=f"{scenario.scenario_id}__r{rep:02d}",
            seed=scenario_index * 100 + rep,
            repetition=rep,
            split=splits[rep],
            cpu_policy=policies[(scenario_index + rep) % len(policies)],
            scenario=scenario,
        )
        for scenario_index, scenario in enumerate(scenarios)
        for rep in range(10)
    ]
```

- [ ] **Step 4: 增加协议验证和 locked-test 访问门**

```python
def validate_formal_protocol(specs):
    # 检查 6 场景×10、split 36/12/6/6、策略 20/20/20、ID/seed 唯一和集合互斥。
    return {"protocol_valid": not failed, "failed_checks": failed, "checks": checks}

def require_split_access(split, *, allow_locked_test=False):
    if split == "locked_test" and not allow_locked_test:
        raise PermissionError("locked_test labels are unavailable before model freeze")
```

- [ ] **Step 5: 运行协议测试**

Run: `python -m unittest 代码/tests/test_formal_airfogsim_dataset_v1.py -v`
Expected: PASS。

- [ ] **Step 6: 提交协议模块**

```powershell
git add 代码/src/pi_jwm/formal_airfogsim_dataset_v1.py 代码/tests/test_formal_airfogsim_dataset_v1.py
git commit -m "feat: define formal AirFogSim dataset protocol"
```

### Task 2: 实现三类容量安全 CPU 动作

**Files:**
- Create: `代码/src/pi_jwm/airfogsim_cpu_policy_v1.py`
- Create: `代码/tests/test_airfogsim_cpu_policy_v1.py`

- [ ] **Step 1: 写三策略和容量约束失败测试**

```python
def test_all_policies_are_capacity_safe_and_deterministic():
    for policy in CPU_POLICY_IDS:
        left = CpuPolicyAllocator(policy, seed=7).allocate(env, computing_tasks)
        right = CpuPolicyAllocator(policy, seed=7).allocate(env, computing_tasks)
        assert left.allocations == right.allocations
        assert sum(left.allocations.values()) <= 10.0 + 1e-9
        assert all(value >= 0.0 for value in left.allocations.values())

def test_deadline_aware_prioritizes_urgent_task():
    result = CpuPolicyAllocator("deadline_aware", seed=0).allocate(env, tasks)
    assert result.allocations["urgent"] > result.allocations["relaxed"]
```

- [ ] **Step 2: 运行测试并确认缺少模块而失败**

Run: `python -m unittest 代码/tests/test_airfogsim_cpu_policy_v1.py -v`
Expected: FAIL，提示模块不存在。

- [ ] **Step 3: 实现统一权重归一化分配器**

```python
class CpuPolicyAllocator:
    def __init__(self, policy_id: str, seed: int, max_tasks_per_node: int = 3):
        self.policy_id = validate_cpu_policy_id(policy_id)
        self.rng = np.random.default_rng(seed)
        self.max_tasks_per_node = int(max_tasks_per_node)

    def allocate(self, env, computing_tasks):
        # equal_share: w_i=1
        # deadline_aware: w_i=1/max(arrival+deadline-current_time, 0.1)
        # feasible_exploration: deadline 权重乘有界随机扰动
        # 每节点独立归一化，返回 allocations 和可审计 decision rows。
```

- [ ] **Step 4: 覆盖空队列、零容量、任务上限和非有限输入测试**

Run: `python -m unittest 代码/tests/test_airfogsim_cpu_policy_v1.py -v`
Expected: PASS，三种策略均无跨节点超分配。

- [ ] **Step 5: 提交 CPU 策略**

```powershell
git add 代码/src/pi_jwm/airfogsim_cpu_policy_v1.py 代码/tests/test_airfogsim_cpu_policy_v1.py
git commit -m "feat: add auditable CPU behavior policies"
```

### Task 3: 参数化 AirFogSim 轨迹运行并记录 CPU 决策

**Files:**
- Modify: `代码/scripts/small_experiments/airfogsim_cross_graph_evidence_closure.py`
- Modify: `代码/scripts/small_experiments/task_resource_conservation_audit.py`
- Modify: `代码/tests/small_experiments/test_airfogsim_cross_graph_evidence_closure.py`
- Modify: `代码/tests/small_experiments/test_task_resource_conservation_audit.py`

- [ ] **Step 1: 写默认行为兼容和参数透传失败测试**

```python
def test_formal_overrides_change_only_declared_scenario_fields():
    configured = apply_formal_scenario_overrides(base, scenario)
    assert configured["traffic"]["max_n_vehicles"] == scenario.max_vehicles
    assert configured["traffic"]["arrival_lambda"] == scenario.vehicle_arrival_lambda
    assert configured["task_profile"]["vehicle"]["lambda"] == scenario.task_lambda
    assert configured["task_profile"]["uav"]["lambda"] == scenario.task_lambda

def test_cpu_rows_keep_policy_and_decision_state():
    row = build_cpu_runtime_row(
        record_id="cpu::Task_1::0.1",
        time_value=0.1,
        node_id="RSU_0",
        task_id="Task_1",
        allocated_cpu=4.0,
        node_cpu_capacity=10.0,
        dt=0.1,
        task_cpu=8.0,
        computed_before=1.0,
        computed_after=1.4,
        policy_id="deadline_aware",
        policy_weight=2.0,
        deadline_remaining=0.5,
        queue_size=2,
    )
    assert row["policy_id"] == "deadline_aware"
    assert row["policy_weight"] == 2.0
```

- [ ] **Step 2: 运行相关测试并确认新参数尚不支持**

Run: `python -m unittest 代码/tests/small_experiments/test_airfogsim_cross_graph_evidence_closure.py 代码/tests/small_experiments/test_task_resource_conservation_audit.py -v`
Expected: FAIL 于新场景或 CPU 策略参数。

- [ ] **Step 3: 给运行函数增加可选正式参数且保持旧默认行为**

```python
def run_airfogsim_evidence_seed(seed, max_time, *, scenario=None, cpu_policy="equal_share"):
    config = preflight.build_preflight_config(config, seed, max_time)
    if scenario is not None:
        config = apply_formal_scenario_overrides(config, scenario)
    algorithm = EvidenceLoggingAlgorithm(seed, cpu_policy=cpu_policy)
```

- [ ] **Step 4: 将同一个 CPU allocator 的决策写入资源台账**

```python
def recording_callback(computing_tasks, **kwargs):
    decision = allocator.allocate(env, computing_tasks)
    decision_by_task.update({row["task_id"]: row for row in decision.rows})
    return decision.allocations
```

`build_cpu_runtime_row` 增加 `policy_id`、`policy_weight`、`deadline_remaining`、`queue_size` 和 `allocated_fraction`，并保留计算前后守恒字段。

- [ ] **Step 5: 运行旧小实验测试和 CPU 守恒测试**

Run: `python -m unittest 代码/tests/small_experiments/test_airfogsim_cross_graph_evidence_closure.py 代码/tests/small_experiments/test_task_resource_conservation_audit.py 代码/tests/test_airfogsim_contract_adapter.py -v`
Expected: PASS，旧 equal-share 默认结果继续合法。

- [ ] **Step 6: 提交运行适配**

```powershell
git add 代码/scripts/small_experiments/airfogsim_cross_graph_evidence_closure.py 代码/scripts/small_experiments/task_resource_conservation_audit.py 代码/tests/small_experiments
git commit -m "feat: parameterize formal AirFogSim trajectories"
```

### Task 4: 将 CPU 动作和 DAG 状态送入图包与张量

**Files:**
- Modify: `代码/src/pi_jwm/airfogsim_dual_graph_v2.py`
- Modify: `代码/src/pi_jwm/airfogsim_tensor_v2.py`
- Modify: `代码/scripts/build_airfogsim_multiseed_v2.py`
- Modify: `代码/tests/test_airfogsim_dual_graph_v2.py`
- Modify: `代码/tests/test_airfogsim_tensor_v2.py`
- Modify: `代码/tests/test_build_airfogsim_multiseed_v2.py`
- Modify: `代码/tests/test_airfogsim_smoke_model_v2.py`
- Modify: `代码/tests/test_airfogsim_window_dataset_v2.py`

- [ ] **Step 1: 写 CPU 动作与 DAG 派生状态失败测试**

```python
def test_tensor_contains_cpu_action_and_dag_release_state():
    graph["source_cpu_actions"] = [{
        "task_id": "Task_2", "node_id": "RSU_0", "time": 0.2,
        "allocated_cpu": 3.0, "node_cpu_capacity": 10.0,
    }]
    graph["task_dag_edges"] = [{"id": "d0", "src": "Task_10", "dst": "Task_2", "data_mb": None}]
    arrays, _ = tensorize_seed_graph(graph, infer_tensor_contract([graph]))
    assert arrays["task_action"][1, task_index, ACTION_FEATURES.index("cpu")] == 1.0
    assert arrays["task_action"][1, task_index, ACTION_FEATURES.index("cpu_allocated")] == 3.0
    assert arrays["task_state"][1, task_index, TASK_FEATURES.index("parent_count")] == 1.0
```

- [ ] **Step 2: 运行图包与张量测试并确认失败**

Run: `python -m unittest 代码/tests/test_airfogsim_dual_graph_v2.py 代码/tests/test_airfogsim_tensor_v2.py -v`
Expected: FAIL，CPU action 和 DAG 派生特征尚不存在。

- [ ] **Step 3: 扩展双图包的直接动作字段**

```python
def freeze_cpu_actions(cpu_actions):
    return [copy.deepcopy(dict(row)) for row in cpu_actions]

# 在 build_dual_graph_v2_bundle 的返回对象中增加：
bundle["source_cpu_actions"] = freeze_cpu_actions(cpu_actions)
```

正式构建器从 `resource_bundle["cpu_ledger"]` 传入该字段，不把 CPU 台账变成信息边。

- [ ] **Step 4: 扩展张量动作和 DAG 状态特征**

```python
ACTION_FEATURES = (
    "offload", "rb", "return", "rb_count", "rb_fraction",
    "cpu", "cpu_allocated", "cpu_fraction",
)
TASK_FEATURES = (
    "task_size", "return_size", "task_cpu", "deadline_remaining", "priority",
    "transmitted", "computed", "delay", "parent_count",
    "unfinished_parent_count", "release_ready",
)
```

CPU 动作节点索引扩展为第四个端点；DAG 状态按每个时间点的父任务生命周期确定，失败父任务不伪装为已满足依赖。

- [ ] **Step 5: 更新固定维度测试和 smoke fixture**

Run: `python -m unittest 代码/tests/test_airfogsim_tensor_v2.py 代码/tests/test_airfogsim_window_dataset_v2.py 代码/tests/test_airfogsim_smoke_model_v2.py 代码/tests/test_build_airfogsim_multiseed_v2.py -v`
Expected: PASS，新动作维度为 8、任务状态维度为 11。

- [ ] **Step 6: 提交图包与张量扩展**

```powershell
git add 代码/src/pi_jwm/airfogsim_dual_graph_v2.py 代码/src/pi_jwm/airfogsim_tensor_v2.py 代码/scripts/build_airfogsim_multiseed_v2.py 代码/tests
git commit -m "feat: encode CPU actions and DAG state"
```

### Task 5: 构建正式数据集编排器和锁定测试保护

**Files:**
- Create: `代码/scripts/build_formal_airfogsim_dataset_v1.py`
- Create: `代码/tests/test_build_formal_airfogsim_dataset_v1.py`
- Modify: `代码/scripts/build_airfogsim_tensor_v2.py`
- Modify: `代码/tests/test_build_airfogsim_tensor_v2.py`

- [ ] **Step 1: 写 60 轨迹编排、断点续跑和 locked-test 失败测试**

```python
def test_builder_writes_protocol_manifest_and_skips_completed_trajectory():
    result = build_formal_dataset(
        output_dir=output_dir,
        specs=build_formal_trajectory_specs(),
        runtime_runner=fake_runtime,
        resource_validator=fake_resource_validator,
        max_time=30.0,
    )
    assert result["trajectory_count"] == 60
    assert result["formal_dataset_ready"]
    second = build_formal_dataset(
        output_dir=output_dir,
        specs=build_formal_trajectory_specs(),
        runtime_runner=fail_if_called,
        resource_validator=fake_resource_validator,
        max_time=30.0,
    )
    assert second["reused_trajectory_count"] == 60

def test_locked_test_labels_require_explicit_unlock():
    with self.assertRaises(PermissionError):
        load_formal_split(root, "locked_test", allow_locked_test=False)
```

- [ ] **Step 2: 运行正式构建器测试并确认缺少入口而失败**

Run: `python -m unittest 代码/tests/test_build_formal_airfogsim_dataset_v1.py -v`
Expected: FAIL，构建器不存在。

- [ ] **Step 3: 实现逐轨迹生成、原子目录和断点续跑**

```python
def build_formal_dataset(*, output_dir, specs, runtime_runner, max_time=30.0):
    for spec in specs:
        target = output_dir / "trajectories" / spec.trajectory_id
        if verified_manifest_exists(target):
            reuse(target)
            continue
        build_one_trajectory_in_temporary_dir(spec, target, runtime_runner)
    validate_and_write_top_level_manifests()
```

每条轨迹独立写配置、图包、资源台账、指标、验证报告和 SHA-256；完成验证后再原子移动到目标目录，失败轨迹不留下“已完成”标记。

- [ ] **Step 4: 实现 split 索引和测试锁**

训练窗口只写 train/validation/calibration 的可读索引；locked-test 索引单独保存并带锁定状态。张量构建器的归一化 split 参数从硬编码 `dev_train` 改为由源数据集声明的 `normalization_source_split`。

- [ ] **Step 5: 实现正式验收报告**

检查 60 条轨迹、6 个场景、四个 split、策略平衡、DAG 无环/无 payload、CPU 容量守恒、窗口不跨轨迹、train-only 归一化和 manifest 完整性；所有条件满足才设置 `formal_dataset_ready=true`，同时保持 `formal_training_ready=false`。

- [ ] **Step 6: 运行构建器和张量测试**

Run: `python -m unittest 代码/tests/test_build_formal_airfogsim_dataset_v1.py 代码/tests/test_build_airfogsim_tensor_v2.py -v`
Expected: PASS。

- [ ] **Step 7: 提交正式构建器**

```powershell
git add 代码/scripts/build_formal_airfogsim_dataset_v1.py 代码/scripts/build_airfogsim_tensor_v2.py 代码/tests/test_build_formal_airfogsim_dataset_v1.py 代码/tests/test_build_airfogsim_tensor_v2.py
git commit -m "feat: build locked formal AirFogSim dataset"
```

### Task 6: 校准场景并生成正式轨迹

**Files:**
- Create: `代码/scripts/calibrate_formal_airfogsim_scenarios_v1.py`
- Create: `代码/tests/test_calibrate_formal_airfogsim_scenarios_v1.py`
- Generate: `代码/artifacts/datasets/airfogsim_formal_v1/`

- [ ] **Step 1: 写场景统计和单调性检查测试**

```python
def test_calibration_summary_requires_load_and_density_separation():
    report = summarize_calibration(probe_rows)
    assert report["load_task_count_monotonic"]
    assert report["density_node_count_monotonic"]
```

- [ ] **Step 2: 实现场景短轨迹校准器**

校准器对 6 个候选场景各运行至少 2 个 seed、5 秒，只读取任务到达数、平均并发任务数、物理节点数、物理边数、链路活动率和 CPU 利用率；根据测量结果冻结场景参数，不读取 locked-test 结果。

- [ ] **Step 3: 运行校准测试和真实短轨迹**

Run: `python -m unittest 代码/tests/test_calibrate_formal_airfogsim_scenarios_v1.py -v`
Expected: PASS。

Run: `conda run -n airfogsim python 代码/scripts/calibrate_formal_airfogsim_scenarios_v1.py --seconds 5 --seeds 900 901`
Expected: `calibration_ready=true`，高负载任务数不低于低负载，dense 节点数不低于 sparse。

- [ ] **Step 4: 将校准值冻结到协议模块并重跑协议测试**

Run: `python -m unittest 代码/tests/test_formal_airfogsim_dataset_v1.py 代码/tests/test_airfogsim_cpu_policy_v1.py -v`
Expected: PASS。

- [ ] **Step 5: 生成 60 条正式轨迹**

Run: `conda run -n airfogsim python 代码/scripts/build_formal_airfogsim_dataset_v1.py --max-time 30 --output-dir 代码/artifacts/datasets/airfogsim_formal_v1`
Expected: 60/60 轨迹完成，`formal_dataset_ready=true`、`formal_training_ready=false`。

- [ ] **Step 6: 构建正式张量层**

Run: `python 代码/scripts/build_airfogsim_tensor_v2.py --source-dir 代码/artifacts/datasets/airfogsim_formal_v1 --output-dir 代码/artifacts/datasets/airfogsim_formal_tensor_v1`
Expected: 张量验证通过，归一化来源为 `train`，locked-test 标签仍保持锁定。

- [ ] **Step 7: 提交校准代码，不提交大体积生成数据**

```powershell
git add 代码/scripts/calibrate_formal_airfogsim_scenarios_v1.py 代码/tests/test_calibrate_formal_airfogsim_scenarios_v1.py 代码/src/pi_jwm/formal_airfogsim_dataset_v1.py
git commit -m "feat: calibrate formal AirFogSim scenarios"
```

### Task 7: 全量回归、就绪报告和计划表收口

**Files:**
- Modify: `本地计划表.md`
- Modify: `文档/研究进展/2026-08-01-PI-JWM正式AirFogSim数据集v1实施计划.md`

- [ ] **Step 1: 运行直接相关测试**

Run: `python -m unittest 代码/tests/test_formal_airfogsim_dataset_v1.py 代码/tests/test_airfogsim_cpu_policy_v1.py 代码/tests/test_build_formal_airfogsim_dataset_v1.py 代码/tests/test_airfogsim_dual_graph_v2.py 代码/tests/test_airfogsim_tensor_v2.py 代码/tests/test_build_airfogsim_tensor_v2.py -v`
Expected: 全部 PASS。

- [ ] **Step 2: 运行既有双图和资源回归测试**

Run: `python -m unittest 代码/tests/test_airfogsim_contract_adapter.py 代码/tests/test_airfogsim_dataset_v2.py 代码/tests/test_airfogsim_metrics_v2.py 代码/tests/test_airfogsim_window_dataset_v2.py 代码/tests/test_airfogsim_smoke_model_v2.py 代码/tests/small_experiments/test_airfogsim_cross_graph_evidence_closure.py 代码/tests/small_experiments/test_task_resource_conservation_audit.py -v`
Expected: 全部 PASS。

- [ ] **Step 3: 独立重算 manifest 和数据验收门**

Run: `python 代码/scripts/build_main_experiment_readiness_v1.py`
Expected: 报告明确区分 `formal_dataset_ready=true` 与 `formal_training_ready=false`，且不把仿真数据写成真实测量。

- [ ] **Step 4: 更新任务状态**

在 `本地计划表.md` 记录实际轨迹数、窗口数、各 split 数量、三类 CPU 策略数量、DAG 边数、CPU 容量违例数、manifest 状态和仍未完成的训练/真实验证边界。

- [ ] **Step 5: 提交收口文档**

```powershell
git add 本地计划表.md 文档/研究进展/2026-08-01-PI-JWM正式AirFogSim数据集v1实施计划.md
git commit -m "docs: record formal AirFogSim dataset readiness"
```
