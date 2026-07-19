# PI-JWM v11 Selector Helper 按 Seed 交叉拟合实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成无 helper 样本内泄漏的 schema-v6 OOF train 标签，重新审计候选收益可辨识性，并仅在冻结 validation 硬门通过后训练 selector 和打开 matched test。

**Architecture:** 在 `pi_jwm` 中新增独立的 seed-crossfit 协议与缓存合并模块；现有标签脚本只增加显式的 `seed_crossfit_5fold` 调度，不改候选、world-model rollout 和 outcome 语义。五个 held-out fold 生成 OOF train cache，calibration/validation 使用全部 40 个 train seeds 拟合的 final helper，三者共享同一全局配置 digest。

**Tech Stack:** Python 3、NumPy、PyTorch、scikit-learn、`unittest`、Bash、JSON/NPZ/SHA-256、CUDA 服务器（仅正式标签与 selector 训练）。

**Execution Boundary:** 用户已明确不使用隔离 worktree；在当前 `main` 上小步提交。matched test seeds `18--19` 和 external seeds `60--69` 在 validation 硬门通过前不得生成或读取。

---

## 文件职责映射

- 新建 `代码/src/pi_jwm/v11_crossfit.py`：固定五折、协议 digest、运行索引解析、OOF cache 合并和审计；不加载模型。
- 修改 `代码/src/pi_jwm/v11_labeling.py`：schema-v5/v6 cache 可选保存 `sample_fold_id`，不将其加载为 selector 特征。
- 修改 `代码/scripts/run_v11_selector_candidate_labels.py`：按协议选择 helper-train/label indices，写 fold provenance；继续复用现有候选和 rollout。
- 新建 `代码/scripts/merge_v11_selector_crossfit_labels.py`：将五个 train fold 确定性合并为正式 OOF train cache。
- 新建 `代码/scripts/audit_v11_crossfit_candidate_shift.py`：比较旧 in-sample train 与新 OOF train 的固定候选、收益符号和跨 split 一致性。
- 新建 `代码/scripts/run_v11_selector_helper_crossfit_gpu.sh`：服务器正式生成、合并和 handoff 审计；脚本不包含 locked split。
- 新建 `代码/tests/test_v11_crossfit.py`：协议、索引、provenance、合并与失败分支单测。
- 修改 `代码/tests/test_v11_selector_finalization.py`：标签脚本 CLI、digest、test-lock 和 reproduction command 合同。
- 修改 `代码/tests/test_v11_schema6_interactions.py`：schema-v6 provenance round-trip 和 GPU launcher 合同。
- 修改 `本地计划表.md`：记录根因、执行状态和 validation 决策；不得提前把 selector 写成定型方法。

### Task 1: 固定五折与全局协议

**Files:**
- Create: `代码/src/pi_jwm/v11_crossfit.py`
- Create: `代码/tests/test_v11_crossfit.py`

- [ ] **Step 1: 写固定 fold 和协议红灯测试**

```python
import unittest

from pi_jwm.v11_selector import DEFAULT_SELECTOR_SEEDS


class SeedCrossfitProtocolTest(unittest.TestCase):
    def test_fixed_round_robin_folds_cover_train_without_overlap(self):
        from pi_jwm.v11_crossfit import audit_seed_crossfit_folds, build_seed_crossfit_folds

        folds = build_seed_crossfit_folds(DEFAULT_SELECTOR_SEEDS["train"])
        self.assertEqual(
            [fold.held_out_seeds for fold in folds],
            [
                (0, 5, 10, 15, 24, 29, 34, 39),
                (1, 6, 11, 20, 25, 30, 35, 40),
                (2, 7, 12, 21, 26, 31, 36, 41),
                (3, 8, 13, 22, 27, 32, 37, 42),
                (4, 9, 14, 23, 28, 33, 38, 43),
            ],
        )
        self.assertTrue(audit_seed_crossfit_folds(folds, DEFAULT_SELECTOR_SEEDS)["passed"])

    def test_protocol_digest_is_independent_of_execution_fold(self):
        from pi_jwm.v11_crossfit import build_crossfit_protocol_manifest

        left = build_crossfit_protocol_manifest({"rf_trees": 160, "schema": 6})
        right = build_crossfit_protocol_manifest({"schema": 6, "rf_trees": 160})
        self.assertEqual(left["crossfit_protocol_digest"], right["crossfit_protocol_digest"])
        self.assertNotIn("execution_fold", left["crossfit_protocol_payload"])
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run:

```powershell
cd D:\shen\网络组\代码
python -m unittest tests.test_v11_crossfit.SeedCrossfitProtocolTest -v
```

Expected: `ModuleNotFoundError: No module named 'pi_jwm.v11_crossfit'`。

- [ ] **Step 3: 实现固定类型、fold 构建和协议哈希**

```python
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .v11_selector import DEFAULT_SELECTOR_SEEDS, canonical_sha256


@dataclass(frozen=True)
class SeedCrossfitFold:
    fold_id: int
    held_out_seeds: tuple[int, ...]
    helper_train_seeds: tuple[int, ...]


def build_seed_crossfit_folds(
    train_seeds: Sequence[int] = DEFAULT_SELECTOR_SEEDS["train"],
    num_folds: int = 5,
) -> tuple[SeedCrossfitFold, ...]:
    ordered = tuple(sorted(int(seed) for seed in train_seeds))
    if len(ordered) != len(set(ordered)) or num_folds != 5 or len(ordered) != 40:
        raise ValueError("formal seed crossfit requires 40 unique train seeds and five folds")
    folds = []
    all_seeds = set(ordered)
    for fold_id in range(num_folds):
        held_out = ordered[fold_id::num_folds]
        helper_train = tuple(seed for seed in ordered if seed not in set(held_out))
        folds.append(SeedCrossfitFold(fold_id, held_out, helper_train))
    return tuple(folds)


def audit_seed_crossfit_folds(folds, seed_spec=DEFAULT_SELECTOR_SEEDS) -> dict[str, Any]:
    train = set(int(seed) for seed in seed_spec["train"])
    held = [seed for fold in folds for seed in fold.held_out_seeds]
    locked = set().union(
        *(set(int(seed) for seed in seed_spec[name]) for name in
          ("calibration", "validation", "background", "matched_test", "external_holdout"))
    )
    errors = []
    if len(folds) != 5 or any(len(fold.held_out_seeds) != 8 for fold in folds):
        errors.append("fold_count_or_size")
    if len(held) != len(set(held)) or set(held) != train:
        errors.append("held_out_coverage")
    for fold in folds:
        if set(fold.held_out_seeds) & set(fold.helper_train_seeds):
            errors.append(f"fold_{fold.fold_id}_overlap")
        if set(fold.helper_train_seeds) != train - set(fold.held_out_seeds):
            errors.append(f"fold_{fold.fold_id}_helper_train")
        if set(fold.helper_train_seeds) & locked:
            errors.append(f"fold_{fold.fold_id}_locked_seed")
    return {"passed": not errors, "errors": errors, "held_out_seed_count": len(set(held))}


def build_crossfit_protocol_manifest(base_configuration: Mapping[str, Any]) -> dict[str, Any]:
    folds = build_seed_crossfit_folds()
    audit = audit_seed_crossfit_folds(folds)
    if not audit["passed"]:
        raise ValueError(f"invalid seed crossfit protocol: {audit}")
    payload = {
        "protocol_version": 1,
        "train_helper_mode": "seed_crossfit_5fold",
        "evaluation_helper_mode": "full_selector_train",
        "folds": [asdict(fold) for fold in folds],
        "base_configuration": dict(base_configuration),
    }
    return {
        "crossfit_protocol_payload": payload,
        "crossfit_protocol_digest": canonical_sha256(payload),
        "audit": audit,
    }
```

- [ ] **Step 4: 运行协议测试并确认通过**

Run: `python -m unittest tests.test_v11_crossfit.SeedCrossfitProtocolTest -v`

Expected: 2 tests，`OK`。

- [ ] **Step 5: 提交协议层**

```powershell
git add 代码/src/pi_jwm/v11_crossfit.py 代码/tests/test_v11_crossfit.py
git commit -m "feat: add selector helper seed crossfit protocol"
```

### Task 2: 运行索引解析与泄漏防护

**Files:**
- Modify: `代码/src/pi_jwm/v11_crossfit.py`
- Modify: `代码/tests/test_v11_crossfit.py`

- [ ] **Step 1: 写 helper/label index 隔离红灯测试**

```python
import numpy as np


class CrossfitExecutionTest(unittest.TestCase):
    def test_train_fold_helper_never_reads_held_out_seed(self):
        from pi_jwm.v11_crossfit import resolve_crossfit_execution

        sample_seed = np.repeat(np.asarray(list(range(16)) + list(range(20, 60))), 2)
        result = resolve_crossfit_execution(sample_seed, ("train",), fold_id=0)
        self.assertEqual(set(sample_seed[result.label_indices["train"]]), set(result.held_out_seeds))
        self.assertFalse(set(sample_seed[result.helper_train_indices]) & set(result.held_out_seeds))
        self.assertEqual(set(sample_seed[result.helper_train_indices]), set(result.helper_train_seeds))

    def test_eval_uses_full_train_helper_and_rejects_fold_id(self):
        from pi_jwm.v11_crossfit import resolve_crossfit_execution

        sample_seed = np.repeat(np.asarray(list(range(16)) + list(range(20, 60))), 2)
        result = resolve_crossfit_execution(sample_seed, ("calibration", "validation"), fold_id=None)
        self.assertEqual(set(sample_seed[result.helper_train_indices]), set(DEFAULT_SELECTOR_SEEDS["train"]))
        with self.assertRaisesRegex(ValueError, "fold"):
            resolve_crossfit_execution(sample_seed, ("validation",), fold_id=1)
```

- [ ] **Step 2: 运行测试并确认 `resolve_crossfit_execution` 缺失**

Run: `python -m unittest tests.test_v11_crossfit.CrossfitExecutionTest -v`

Expected: `ImportError`。

- [ ] **Step 3: 实现不可变执行解析结果**

```python
import numpy as np


@dataclass(frozen=True)
class CrossfitExecution:
    mode: str
    fold_id: int | None
    held_out_seeds: tuple[int, ...]
    helper_train_seeds: tuple[int, ...]
    helper_train_indices: np.ndarray
    label_indices: dict[str, np.ndarray]


def resolve_crossfit_execution(
    sample_seed: np.ndarray,
    requested_splits: Sequence[str],
    fold_id: int | None,
) -> CrossfitExecution:
    seeds = np.asarray(sample_seed, dtype=np.int64).reshape(-1)
    requested = tuple(str(name) for name in requested_splits)
    folds = build_seed_crossfit_folds()
    if "train" in requested:
        if requested != ("train",) or fold_id is None or fold_id not in range(5):
            raise ValueError("crossfit train execution requires exactly one valid fold and only train split")
        fold = folds[int(fold_id)]
        helper = np.flatnonzero(np.isin(seeds, fold.helper_train_seeds))
        labels = {"train": np.flatnonzero(np.isin(seeds, fold.held_out_seeds))}
        return CrossfitExecution(
            "crossfit_train_fold", fold.fold_id, fold.held_out_seeds,
            fold.helper_train_seeds, helper, labels,
        )
    if fold_id is not None:
        raise ValueError("evaluation helper execution does not accept a crossfit fold")
    allowed = {"calibration", "validation", "matched_test", "external_holdout"}
    if not requested or not set(requested).issubset(allowed):
        raise ValueError("crossfit evaluation supports only frozen selector evaluation splits")
    helper_seeds = tuple(DEFAULT_SELECTOR_SEEDS["train"])
    labels = {
        name: np.flatnonzero(np.isin(seeds, DEFAULT_SELECTOR_SEEDS[name]))
        for name in requested
    }
    return CrossfitExecution(
        "full_train_eval", None, (), helper_seeds,
        np.flatnonzero(np.isin(seeds, helper_seeds)), labels,
    )
```

- [ ] **Step 4: 运行两组 crossfit 单测**

Run: `python -m unittest tests.test_v11_crossfit -v`

Expected: 全部 `OK`。

- [ ] **Step 5: 提交索引隔离实现**

```powershell
git add 代码/src/pi_jwm/v11_crossfit.py 代码/tests/test_v11_crossfit.py
git commit -m "feat: isolate helper and label indices by seed fold"
```

### Task 3: 缓存 provenance 与确定性 OOF 合并

**Files:**
- Modify: `代码/src/pi_jwm/v11_labeling.py`
- Modify: `代码/src/pi_jwm/v11_crossfit.py`
- Modify: `代码/tests/test_v11_crossfit.py`
- Modify: `代码/tests/test_v11_schema6_interactions.py`

- [ ] **Step 1: 写 provenance round-trip 与重复样本红灯测试**

```python
class CrossfitCacheMergeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write_fold_cache(self, fold_id, sample_ids):
        from pi_jwm.v11_interactions import CandidateInteractionBatch
        from pi_jwm.v11_labeling import save_candidate_interaction_cache
        from pi_jwm.v11_selector import CandidateBatch, CandidateOutcome

        ids = np.asarray(sample_ids, dtype=np.int64)
        count = len(ids)
        batch = CandidateBatch(
            context=np.zeros((count, 2), dtype=np.float32),
            candidate_features=np.zeros((count, 2, 3), dtype=np.float32),
            candidate_mask=np.ones((count, 2), dtype=bool),
            stage=np.asarray(["offload"] * count),
            feature_names=("x", "y", "z"),
            candidate_names=("identity", "ranked_allocation_baseline"),
            context_feature_names=("ctx0", "ctx1"),
        )
        outcome = CandidateOutcome(
            active_sse=np.ones((count, 2), dtype=np.float32),
            active_count=np.ones(count, dtype=np.int64),
            default_index=1,
        )
        interactions = CandidateInteractionBatch(
            tokens=np.zeros((count, 2, 72, 25), dtype=np.float32),
            token_mask=np.zeros((count, 2, 72), dtype=bool),
            edge_index=np.full((count, 2, 72), -1, dtype=np.int32),
            token_feature_names=tuple(f"token_{index}" for index in range(25)),
            pooled_features=np.zeros((count, 2, 234), dtype=np.float32),
            pooled_feature_names=tuple(f"pooled_{index}" for index in range(234)),
        )
        path = self.root / f"fold_{fold_id}_{ids[0]}.npz"
        save_candidate_interaction_cache(
            path,
            split_name="train",
            sample_ids=ids,
            sample_seed=np.full(count, fold_id, dtype=np.int64),
            batch=batch,
            outcome=outcome,
            interactions=interactions,
            action_feature_names=("a0", "a1", "a2", "a3", "a4", "a5"),
            configuration_digest="b" * 64,
            protocol_metadata={
                "crossfit_protocol_digest": "a" * 64,
                "helper_execution": {
                    "mode": "crossfit_train_fold",
                    "fold_id": fold_id,
                    "held_out_seeds": [fold_id],
                    "helper_train_seeds": [seed for seed in range(5) if seed != fold_id],
                },
            },
            sample_fold_id=np.full(count, fold_id, dtype=np.int16),
        )
        self.cache_path = path
        return path

    def test_schema6_cache_keeps_fold_provenance_outside_model_features(self):
        from pi_jwm.v11_labeling import load_candidate_label_metadata

        metadata = load_candidate_label_metadata(self._write_fold_cache(fold_id=2, sample_ids=[9, 10]))
        np.testing.assert_array_equal(metadata["sample_fold_id"], [2, 2])
        batch, _, _, _ = load_candidate_interaction_cache(self.cache_path)
        self.assertFalse(any("fold" in name for name in batch.feature_names))
        self.assertFalse(any("fold" in name for name in batch.context_feature_names))

    def test_merge_rejects_duplicate_sample_id(self):
        from pi_jwm.v11_crossfit import merge_crossfit_label_caches

        paths = [self._write_fold_cache(0, [0, 1]), self._write_fold_cache(1, [1, 2])]
        with self.assertRaisesRegex(ValueError, "duplicate sample"):
            merge_crossfit_label_caches(paths, self.root / "merged.npz", [0, 1, 2], [0, 0, 1], "a" * 64)
```

- [ ] **Step 2: 运行测试并确认 provenance/merge 接口缺失**

Run:

```powershell
python -m unittest tests.test_v11_crossfit.CrossfitCacheMergeTest tests.test_v11_schema6_interactions.CandidateInteractionCacheTest -v
```

Expected: `sample_fold_id` 或 `merge_crossfit_label_caches` 缺失导致失败。

- [ ] **Step 3: 扩展 cache 保存和 metadata 读取**

在 `save_candidate_label_cache()` 和 `save_candidate_interaction_cache()` 增加：

```python
sample_fold_id: np.ndarray | None = None,
```

保存前执行：

```python
fold_ids = np.asarray(
    [] if sample_fold_id is None else sample_fold_id, dtype=np.int16
).reshape(-1)
if fold_ids.size not in (0, sample_count):
    raise ValueError("sample_fold_id must be empty or match candidate batch")
```

在 NPZ 中增加 `sample_fold_id=fold_ids`，并让 `load_candidate_label_metadata()` 返回：

```python
"sample_fold_id": (
    np.asarray(arrays["sample_fold_id"], dtype=np.int16).reshape(-1)
    if "sample_fold_id" in arrays.files and arrays["sample_fold_id"].size
    else None
),
```

manifest 增加：

```python
"sample_provenance_fields": ["sample_fold_id"] if fold_ids.size else [],
```

- [ ] **Step 4: 实现完整 schema-v6 合并逻辑**

在 `v11_crossfit.py` 中实现 `merge_crossfit_label_caches()`。它必须调用 `load_candidate_interaction_cache()`，比较每个 manifest 的 schema、configuration digest、candidate/feature/context/interaction 合同和 crossfit digest。加载结果保存在不可变的 `LoadedFoldCache(path, sample_ids, sample_seed, fold_ids, batch, outcome, interactions, manifest)` 中。随后按 sample id 排序，并先定义：

```python
def cat(name: str) -> np.ndarray:
    return np.concatenate([getattr(item.outcome, name) for item in items], axis=0)


def cat_optional(name: str, order: np.ndarray) -> np.ndarray | None:
    values = [getattr(item.outcome, name) for item in items]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"crossfit outcome field is only partially present: {name}")
    return np.concatenate(values, axis=0)[order]
```

然后用以下全部字段构建输出：

```python
merged_batch = CandidateBatch(
    context=np.concatenate([item.batch.context for item in items], axis=0)[order],
    candidate_features=np.concatenate([item.batch.candidate_features for item in items], axis=0)[order],
    candidate_mask=np.concatenate([item.batch.candidate_mask for item in items], axis=0)[order],
    stage=np.concatenate([item.batch.stage for item in items], axis=0)[order],
    feature_names=items[0].batch.feature_names,
    candidate_names=items[0].batch.candidate_names,
    context_feature_names=items[0].batch.context_feature_names,
)
merged_outcome = CandidateOutcome(
    active_sse=cat("active_sse")[order],
    active_count=cat("active_count")[order],
    link_sse=cat_optional("link_sse", order),
    link_count=cat_optional("link_count", order),
    activity_tp=cat_optional("activity_tp", order),
    activity_fp=cat_optional("activity_fp", order),
    activity_fn=cat_optional("activity_fn", order),
    activity_tn=cat_optional("activity_tn", order),
    action_applied=cat_optional("action_applied", order),
    action_applicable=cat_optional("action_applicable", order),
    default_index=items[0].outcome.default_index,
    task_utility=cat_optional("task_utility", order),
    energy_total=cat_optional("energy_total", order),
    result_kind="diagnostic_only",
)
merged_interactions = CandidateInteractionBatch(
    tokens=np.concatenate([item.interactions.tokens for item in items], axis=0)[order],
    token_mask=np.concatenate([item.interactions.token_mask for item in items], axis=0)[order],
    edge_index=np.concatenate([item.interactions.edge_index for item in items], axis=0)[order],
    token_feature_names=items[0].interactions.token_feature_names,
    pooled_features=np.concatenate(
        [item.interactions.pooled_features for item in items], axis=0
    )[order],
    pooled_feature_names=items[0].interactions.pooled_feature_names,
)
```

其中 `cat_optional()` 要求所有 fold 同时存在或同时缺失该字段；混合状态直接 `ValueError`。合并前执行：

```python
if len(sample_ids) != len(set(sample_ids.tolist())):
    raise ValueError("duplicate sample id across crossfit folds")
if set(sample_ids.tolist()) != set(np.asarray(expected_sample_ids).tolist()):
    raise ValueError("crossfit sample coverage mismatch")
expected_seed_by_id = {
    int(sample_id): int(seed)
    for sample_id, seed in zip(expected_sample_ids, expected_sample_seed)
}
if any(
    expected_seed_by_id[int(sample_id)] != int(seed)
    for sample_id, seed in zip(sample_ids[order], sample_seed[order])
):
    raise ValueError("crossfit sample seed mismatch")
```

最后调用 `save_candidate_interaction_cache(..., split_name="train", sample_fold_id=fold_ids[order])`，并把五个源 cache SHA 写入 merged manifest 的 `protocol_metadata.source_fold_caches`。

- [ ] **Step 5: 运行 cache/merge 测试**

Run:

```powershell
python -m unittest tests.test_v11_crossfit.CrossfitCacheMergeTest tests.test_v11_schema6_interactions.CandidateInteractionCacheTest -v
```

Expected: 全部 `OK`。

- [ ] **Step 6: 提交 cache 合并层**

```powershell
git add 代码/src/pi_jwm/v11_crossfit.py 代码/src/pi_jwm/v11_labeling.py 代码/tests/test_v11_crossfit.py 代码/tests/test_v11_schema6_interactions.py
git commit -m "feat: merge auditable selector crossfit caches"
```

### Task 4: 将标签生成器接入 crossfit 协议

**Files:**
- Modify: `代码/scripts/run_v11_selector_candidate_labels.py`
- Modify: `代码/tests/test_v11_selector_finalization.py`
- Modify: `代码/tests/test_v11_crossfit.py`

- [ ] **Step 1: 写 CLI、digest 和锁测试红灯测试**

```python
class CrossfitLabelRunnerContractTest(unittest.TestCase):
    def test_crossfit_fold_is_execution_metadata_not_global_digest_input(self):
        from run_v11_selector_candidate_labels import _canonical_configuration

        left = self.make_args(helper_protocol="seed_crossfit_5fold", crossfit_fold=0)
        right = self.make_args(helper_protocol="seed_crossfit_5fold", crossfit_fold=4)
        self.assertEqual(_canonical_configuration(left), _canonical_configuration(right))

    def test_crossfit_train_requires_fold_and_eval_rejects_fold(self):
        from run_v11_selector_candidate_labels import validate_helper_execution_args

        with self.assertRaisesRegex(ValueError, "fold"):
            validate_helper_execution_args("seed_crossfit_5fold", None, ("train",))
        with self.assertRaisesRegex(ValueError, "fold"):
            validate_helper_execution_args("seed_crossfit_5fold", 0, ("validation",))

    def test_reproduction_command_contains_crossfit_protocol(self):
        tokens = shlex.split(build_reproduction_command(self.make_args(
            helper_protocol="seed_crossfit_5fold", crossfit_fold=3
        )))
        self.assertEqual(tokens[tokens.index("--helper-protocol") + 1], "seed_crossfit_5fold")
        self.assertEqual(tokens[tokens.index("--crossfit-fold") + 1], "3")
```

- [ ] **Step 2: 运行合同测试并确认新参数缺失**

Run:

```powershell
python -m unittest tests.test_v11_selector_finalization.CandidateLabelRunnerContractTest tests.test_v11_crossfit.CrossfitLabelRunnerContractTest -v
```

Expected: 新 CLI/validation 函数缺失导致失败。

- [ ] **Step 3: 增加显式参数和 reproduction command**

在 `parse_args()` 增加：

```python
parser.add_argument(
    "--helper-protocol",
    choices=("in_sample", "seed_crossfit_5fold"),
    default="in_sample",
)
parser.add_argument("--crossfit-fold", type=int, choices=range(5))
```

在 `build_reproduction_command()` 无条件写 `--helper-protocol`，仅当 fold 非空时写 `--crossfit-fold`。在 `_canonical_configuration()` 中写入完整 `build_crossfit_protocol_manifest(base_configuration)` 的 payload，但不写 `crossfit_fold`；旧 `in_sample` 默认路径保持旧配置字段和历史复现能力。

- [ ] **Step 4: 按协议替换 helper/label indices**

在 `run()` 中保留现有 `build_selector_split()` 和 `SelectorProtocol` test lock，然后对新协议执行：

```python
execution = resolve_crossfit_execution(
    arrays["sample_seed"], tuple(args.splits), args.crossfit_fold
)
helper_indices = limit_indices_seed_balanced(
    execution.helper_train_indices,
    arrays["sample_seed"],
    int(args.helper_train_limit),
)
label_indices = {
    name: limit_indices_seed_balanced(
        indices, arrays["sample_seed"], int(args.split_sample_limit)
    )
    for name, indices in execution.label_indices.items()
}
```

`_fit_helper_models()` 只接收 `helper_indices`。`_make_label_split()` 继续接收同一 `helper_indices` 作为 adaptive dataset 的 train reference，并新增：

```python
sample_fold_id=(
    np.full(len(split_indices), execution.fold_id, dtype=np.int16)
    if execution.fold_id is not None else None
)
```

每个 cache manifest 的 `protocol_metadata` 增加：

```python
"crossfit_protocol_digest": crossfit_manifest["crossfit_protocol_digest"],
"helper_execution": {
    "mode": execution.mode,
    "fold_id": execution.fold_id,
    "held_out_seeds": list(execution.held_out_seeds),
    "helper_train_seeds": list(execution.helper_train_seeds),
},
```

运行开始时把全局协议写到 `crossfit_protocol.json`。summary 明确区分 `helper_train_samples`、`label_samples` 和 `result_kind=diagnostic_only`。

- [ ] **Step 5: 运行标签合同和相关回归测试**

Run:

```powershell
python -m unittest tests.test_v11_crossfit tests.test_v11_selector_finalization.CandidateLabelRunnerContractTest tests.test_v11_schema6_interactions.CandidateInteractionRunnerTest -v
```

Expected: 全部 `OK`。

- [ ] **Step 6: 提交标签生成器接入**

```powershell
git add 代码/scripts/run_v11_selector_candidate_labels.py 代码/tests/test_v11_selector_finalization.py 代码/tests/test_v11_crossfit.py
git commit -m "feat: generate selector labels with seed crossfit helpers"
```

### Task 5: 合并 CLI、服务器 launcher 与合同测试

**Files:**
- Create: `代码/scripts/merge_v11_selector_crossfit_labels.py`
- Create: `代码/scripts/run_v11_selector_helper_crossfit_gpu.sh`
- Modify: `代码/tests/test_v11_crossfit.py`
- Modify: `代码/tests/test_v11_schema6_interactions.py`

- [ ] **Step 1: 写 merge CLI 和 launcher 红灯合同**

```python
class CrossfitLauncherContractTest(unittest.TestCase):
    def test_gpu_launcher_generates_five_folds_then_eval_and_never_locked_splits(self):
        text = (SCRIPTS_ROOT / "run_v11_selector_helper_crossfit_gpu.sh").read_text("utf-8")
        self.assertIn("for FOLD in 0 1 2 3 4", text)
        self.assertIn("--helper-protocol seed_crossfit_5fold", text)
        self.assertIn("merge_v11_selector_crossfit_labels.py", text)
        self.assertIn("--splits calibration validation", text)
        self.assertNotIn("matched_test", text)
        self.assertNotIn("external_holdout", text)
```

- [ ] **Step 2: 运行测试并确认 launcher 不存在**

Run: `python -m unittest tests.test_v11_crossfit.CrossfitLauncherContractTest -v`

Expected: `FileNotFoundError`。

- [ ] **Step 3: 实现 merge CLI**

`merge_v11_selector_crossfit_labels.py` 参数固定为：

```python
parser.add_argument("--fold-cache", type=Path, action="append", required=True)
parser.add_argument("--output-cache", type=Path, required=True)
parser.add_argument("--sample-index-csv", type=Path, required=True)
parser.add_argument("--sample-limit-per-fold", type=int, default=0)
parser.add_argument("--expected-configuration-digest", required=True)
parser.add_argument("--expected-crossfit-protocol-digest", required=True)
```

脚本从 `sample_index.csv` 读取 `sample_id,seed`，只选择 `DEFAULT_SELECTOR_SEEDS["train"]`；smoke 时按每个 held-out fold 内 seed round-robin 复用 `limit_indices_seed_balanced()` 得到预期 sample，正式运行 `sample-limit-per-fold=0` 必须得到 15,600 个样本。调用 `merge_crossfit_label_caches()` 后写 `merge_summary.json`，字段包括 sample 数、40 个 seeds、五个源 SHA、merged SHA 和两个 digest。

- [ ] **Step 4: 实现 GPU launcher**

launcher 使用以下固定结构：

```bash
REPORT_ROOT="${REPORT_ROOT:-artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719}"
FOLD_ROOT="${REPORT_ROOT}/label_cache_schema6/folds"
EVAL_DIR="${REPORT_ROOT}/label_cache_schema6/eval"
MERGED_DIR="${REPORT_ROOT}/label_cache_schema6/merged"

for FOLD in 0 1 2 3 4; do
  python scripts/run_v11_selector_candidate_labels.py \
    --splits train \
    --helper-protocol seed_crossfit_5fold \
    --crossfit-fold "${FOLD}" \
    --cache-schema-version 6 \
    --helper-train-limit 0 --split-sample-limit 0 \
    --device cuda --batch-size "${BATCH_SIZE}" --rf-trees "${RF_TREES}" \
    --output-dir "${FOLD_ROOT}/fold_${FOLD}" \
    2>&1 | tee "${LOG_DIR}/fold_${FOLD}.log"
done

python scripts/run_v11_selector_candidate_labels.py \
  --splits calibration validation \
  --helper-protocol seed_crossfit_5fold \
  --cache-schema-version 6 \
  --helper-train-limit 0 --split-sample-limit 0 \
  --device cuda --batch-size "${BATCH_SIZE}" --rf-trees "${RF_TREES}" \
  --output-dir "${EVAL_DIR}" \
  2>&1 | tee "${LOG_DIR}/evaluation.log"
```

launcher 从 `crossfit_protocol.json` 和 run summaries 读取两个 digest，再调用 merge CLI。最后加载 merged train、calibration、validation，验证 schema=6、split seeds、overflow=0、三个 configuration digest 相同、crossfit protocol digest 相同、train provenance 为 folds `0--4`、源码树干净且 source Git SHA 一致。

- [ ] **Step 5: 运行合同测试和 Bash 静态检查**

Run:

```powershell
python -m unittest tests.test_v11_crossfit.CrossfitLauncherContractTest tests.test_v11_schema6_interactions -v
bash -n 代码/scripts/run_v11_selector_helper_crossfit_gpu.sh
```

Expected: 单测 `OK`，`bash -n` exit code 0。

- [ ] **Step 6: 提交 CLI 和 launcher**

```powershell
git add 代码/scripts/merge_v11_selector_crossfit_labels.py 代码/scripts/run_v11_selector_helper_crossfit_gpu.sh 代码/tests/test_v11_crossfit.py 代码/tests/test_v11_schema6_interactions.py
git commit -m "feat: orchestrate selector crossfit label generation"
```

### Task 6: 本地 CPU smoke 和确定性质量门

**Files:**
- Generated: `代码/artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/smoke/`

- [ ] **Step 1: 运行交叉拟合相关单测**

Run:

```powershell
cd D:\shen\网络组\代码
python -m unittest tests.test_v11_crossfit tests.test_v11_schema6_interactions tests.test_v11_selector_finalization -v
```

Expected: 全部 `OK`，且没有读取 locked split 的测试输出。

- [ ] **Step 2: 对 fold 0 做两次 64-sample 真实 CPU smoke**

Run twice，分别写入 `fold0_a` 和 `fold0_b`：

```powershell
python scripts/run_v11_selector_candidate_labels.py --splits train --helper-protocol seed_crossfit_5fold --crossfit-fold 0 --device cpu --batch-size 16 --helper-train-limit 256 --split-sample-limit 64 --rf-trees 20 --stats-chunk-size 512 --cache-schema-version 6 --output-dir artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/smoke/fold0_a
```

Expected: 每次 64 samples；label seeds 恰为 fold 0 的 8 个 held-out seeds；helper provenance 恰为另外 32 个 train seeds；token overflow 为 0。

- [ ] **Step 3: 比较两次 smoke 的数组级确定性**

Run:

```powershell
@'
import numpy as np
from pathlib import Path
a=Path('artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/smoke/fold0_a/candidate_labels_train.npz')
b=Path('artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/smoke/fold0_b/candidate_labels_train.npz')
with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
    assert x.files == y.files
    for name in x.files:
        assert np.array_equal(x[name], y[name]), name
print('deterministic arrays: PASS')
'@ | python -
```

Expected: `deterministic arrays: PASS`。

- [ ] **Step 4: 对其余四折及 evaluation 各跑 64-sample smoke**

fold 1--4 使用以下 PowerShell 循环，参数与 fold 0 明确保持一致：

```powershell
1..4 | ForEach-Object {
  python scripts/run_v11_selector_candidate_labels.py --splits train --helper-protocol seed_crossfit_5fold --crossfit-fold $_ --device cpu --batch-size 16 --helper-train-limit 256 --split-sample-limit 64 --rf-trees 20 --stats-chunk-size 512 --cache-schema-version 6 --output-dir "artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/smoke/fold$_"
  if ($LASTEXITCODE -ne 0) { throw "fold $_ smoke failed" }
}
```

evaluation 使用：

```powershell
python scripts/run_v11_selector_candidate_labels.py --splits calibration validation --helper-protocol seed_crossfit_5fold --device cpu --batch-size 16 --helper-train-limit 256 --split-sample-limit 64 --rf-trees 20 --stats-chunk-size 512 --cache-schema-version 6 --output-dir artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/smoke/eval
```

Expected: 五折均成功、无 helper/held-out overlap，calibration/validation helper provenance 均为全部 40 个 train seeds。

- [ ] **Step 5: 合并 5×64 smoke caches**

Run：

```powershell
$root='artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/smoke'
$run=Get-Content -Raw "$root/fold0_a/candidate_label_run_summary.json" | ConvertFrom-Json
$protocol=Get-Content -Raw "$root/fold0_a/crossfit_protocol.json" | ConvertFrom-Json
python scripts/merge_v11_selector_crossfit_labels.py `
  --fold-cache "$root/fold0_a/candidate_labels_train.npz" `
  --fold-cache "$root/fold1/candidate_labels_train.npz" `
  --fold-cache "$root/fold2/candidate_labels_train.npz" `
  --fold-cache "$root/fold3/candidate_labels_train.npz" `
  --fold-cache "$root/fold4/candidate_labels_train.npz" `
  --output-cache "$root/merged/candidate_labels_train.npz" `
  --sample-index-csv artifacts/experiments/airfogsim_v0/datasets/dataset_multiseed_active_heavy_v2_60seed_20260619/sample_index.csv `
  --sample-limit-per-fold 64 `
  --expected-configuration-digest $run.configuration_digest `
  --expected-crossfit-protocol-digest $protocol.crossfit_protocol_digest
```

Expected: merged cache 320 samples、40 个 train seeds、fold provenance `{0,1,2,3,4}`、schema=6、overflow=0。

- [ ] **Step 6: 若任一质量门失败则停止；通过则提交 smoke 所需源码修正**

源码没有额外修正时不提交 artifacts。若测试暴露实现缺陷，严格按红灯测试补充后提交：

```powershell
git add 代码/src/pi_jwm 代码/scripts 代码/tests
git commit -m "fix: satisfy selector crossfit smoke invariants"
```

### Task 7: 服务器正式生成 OOF schema-v6 标签

**Files:**
- Generated: `代码/artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/label_cache_schema6/`

- [ ] **Step 1: 同步已提交源码并确认服务器条件**

Run on server:

```bash
cd /root/autodl-tmp/pi-jwm
git status --short
git rev-parse HEAD
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Expected: tracked worktree clean；HEAD 与本地最新提交一致；CUDA GPU 可见。

- [ ] **Step 2: 先在服务器执行相关测试**

Run from the PI-JWM code root:

```bash
PYTHONPATH=src python -m unittest tests.test_v11_crossfit tests.test_v11_schema6_interactions tests.test_v11_selector_finalization -v
```

Expected: 全部 `OK`。

- [ ] **Step 3: 运行正式五折 launcher**

Run:

```bash
bash scripts/run_v11_selector_helper_crossfit_gpu.sh
```

Expected: 5 个 fold 均生成 3,120 samples，merged train 为 15,600；calibration 2,340；validation 3,900；所有 cache schema=6、overflow=0、action-applied gate 通过。

- [ ] **Step 4: 执行正式 handoff 审计**

检查 `gpu_crossfit_handoff_summary.json`：

```python
assert summary["sample_count"] == {"train": 15600, "calibration": 2340, "validation": 3900}
assert summary["train_fold_ids"] == [0, 1, 2, 3, 4]
assert summary["configuration_digest_count"] == 1
assert summary["crossfit_protocol_digest_count"] == 1
assert summary["overflow_count"] == {"train": 0, "calibration": 0, "validation": 0}
assert summary["matched_test_accessed"] is False
assert summary["external_holdout_accessed"] is False
```

- [ ] **Step 5: 若生成失败，仅从失败 fold 续跑**

已通过 SHA 和 manifest 审计的 fold 不覆盖；重跑失败 fold 后必须重新执行 merge 和完整 handoff。禁止手工编辑 manifest 或从旧 in-sample train cache 拼接样本。

### Task 8: 跨 split 候选偏移与可辨识性审计

**Files:**
- Create: `代码/scripts/audit_v11_crossfit_candidate_shift.py`
- Modify: `代码/tests/test_v11_crossfit.py`
- Generated: `代码/artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/audit/`

- [ ] **Step 1: 写固定候选聚合和样本收益符号测试**

```python
def test_candidate_shift_rows_recompute_rmse_and_seed_direction():
    from audit_v11_crossfit_candidate_shift import candidate_shift_rows

    sse = np.asarray([[4.0, 1.0], [4.0, 9.0]], dtype=np.float32)
    count = np.ones(2, dtype=np.int64)
    rows = candidate_shift_rows(
        sse, count, np.asarray([50, 51]), ("default", "repair"), default_index=0
    )
    repair = next(row for row in rows if row["candidate_name"] == "repair")
    self.assertAlmostEqual(repair["rmse"], np.sqrt(5.0))
    self.assertEqual(repair["improved_seed_count"], 1)
    self.assertEqual(repair["positive_pair_rate"], 0.5)
```

- [ ] **Step 2: 实现审计脚本并输出固定文件**

脚本参数：旧 train cache、新 OOF train cache、新 calibration/validation cache 和 output dir。输出：

- `candidate_metrics_by_split.csv`：candidate、family、RMSE、improvement、positive pair rate、improved seed count；
- `helper_candidate_shift.csv`：value-head q50/q75 与 persistent/decayed 候选的新旧 train、calibration、validation 对比；
- `stage_family_opportunity.csv`：逐 stage/family 的 opportunity 与 oracle win；
- `summary.json`：是否消除“value-head persistent 改善 40/40 train seeds”的异常、matched/external access 均为 false。

收益严格使用 `default_sse - candidate_sse`；RMSE 严格使用 `sqrt(sum(sse)/sum(active_count))`，无 active target 的组标记 `unscored`，不写 NaN 胜者。

核心聚合函数固定为：

```python
def candidate_shift_rows(
    active_sse: np.ndarray,
    active_count: np.ndarray,
    sample_seed: np.ndarray,
    candidate_names: tuple[str, ...],
    default_index: int,
) -> list[dict[str, object]]:
    sse = np.asarray(active_sse, dtype=np.float64)
    count = np.asarray(active_count, dtype=np.int64).reshape(-1)
    seeds = np.asarray(sample_seed, dtype=np.int64).reshape(-1)
    if sse.shape[0] != count.size or count.shape != seeds.shape:
        raise ValueError("candidate shift arrays have incompatible sample dimensions")
    total_count = int(count.sum())
    rows = []
    for candidate_index, candidate_name in enumerate(candidate_names):
        benefit = sse[:, default_index] - sse[:, candidate_index]
        seed_directions = []
        for seed in sorted(np.unique(seeds)):
            mask = seeds == seed
            seed_directions.append(float(benefit[mask].sum()) > 0.0)
        rows.append(
            {
                "candidate_index": candidate_index,
                "candidate_name": candidate_name,
                "scored": total_count > 0,
                "rmse": (
                    float(np.sqrt(sse[:, candidate_index].sum() / total_count))
                    if total_count > 0 else None
                ),
                "improvement_sse": float(benefit.sum()),
                "positive_pair_rate": float(np.mean(benefit > 0.0)),
                "improved_seed_count": int(sum(seed_directions)),
                "seed_count": len(seed_directions),
            }
        )
    return rows
```

`main()` 对四个 cache 调用 `load_candidate_interaction_cache()` 和 `load_candidate_label_metadata()`；先核对 candidate order 和 result kind，再写 CSV/JSON。任何旧、新 cache digest 意外相同或 locked split 名称出现时直接失败。

- [ ] **Step 3: 运行审计脚本单测**

Run: `python -m unittest tests.test_v11_crossfit -v`

Expected: 全部 `OK`。

- [ ] **Step 4: 在服务器运行正式候选偏移审计**

Run:

```bash
python scripts/audit_v11_crossfit_candidate_shift.py \
  --old-train-cache artifacts/reports/pi_jwm_v11_schema6_edge_step_interactions_20260718/label_cache_schema6/candidate_labels_train.npz \
  --train-cache artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/label_cache_schema6/merged/candidate_labels_train.npz \
  --calibration-cache artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/label_cache_schema6/eval/candidate_labels_calibration.npz \
  --validation-cache artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/label_cache_schema6/eval/candidate_labels_validation.npz \
  --output-dir artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/audit/candidate_shift
```

- [ ] **Step 5: 运行 schema-v6 可辨识性审计**

```bash
python scripts/audit_v11_candidate_benefit_identifiability.py \
  --train-cache artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/label_cache_schema6/merged/candidate_labels_train.npz \
  --calibration-cache artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/label_cache_schema6/eval/candidate_labels_calibration.npz \
  --validation-cache artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/label_cache_schema6/eval/candidate_labels_validation.npz \
  --output-dir artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/audit/identifiability \
  --sample-limit-per-split 0 --group-cv-folds 3 --required-schema-version 6 \
  --model-kinds linear rf hgb xgb \
  --feature-groups full_schema_v5 interaction_pooled_only full_schema_v6
```

Expected: 输出正式 classification、逐 seed metrics 和 hard-gate；XGBoost 不可用时明确 `skipped`，不得静默改变模型集合。

- [ ] **Step 6: 提交审计代码**

```powershell
git add 代码/scripts/audit_v11_crossfit_candidate_shift.py 代码/tests/test_v11_crossfit.py
git commit -m "feat: audit selector crossfit candidate shift"
```

### Task 9: 条件式 selector 训练、validation 冻结与一次性 test

**Files:**
- Modify: `代码/scripts/train_v11_candidate_set_selector.py`
- Modify: `代码/tests/test_v11_selector_finalization.py`
- Generated: `代码/artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/selector_training/`
- Generated conditionally: `代码/artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/frozen_evaluation/`

- [ ] **Step 1: 写 selector validation 冻结硬门红灯测试**

```python
def test_selector_freeze_requires_performance_and_safety_metrics(self):
    from train_v11_candidate_set_selector import classify_selector_validation_gate

    passing = {
        "rmse": 229.0,
        "improved_seed_count": 7,
        "executed_positive_precision": 0.70,
        "negative_selection_rate": 0.10,
        "activity_f1_drop": 0.001,
        "link_rmse_relative_degradation": 0.01,
        "training_seed_std": 4.0,
    }
    self.assertTrue(classify_selector_validation_gate(passing)["passed"])
    failures = {
        "rmse": 230.8556,
        "improved_seed_count": 6,
        "executed_positive_precision": 0.64,
        "negative_selection_rate": 0.21,
        "activity_f1_drop": 0.0021,
        "link_rmse_relative_degradation": 0.021,
        "training_seed_std": 5.1,
    }
    for field, value in failures.items():
        row = dict(passing)
        row[field] = value
        self.assertFalse(classify_selector_validation_gate(row)["passed"], field)
```

- [ ] **Step 2: 运行测试并确认正式冻结函数缺失**

Run:

```powershell
python -m unittest tests.test_v11_selector_finalization.SelectorTrainingRunnerContractTest.test_selector_freeze_requires_performance_and_safety_metrics -v
```

Expected: `ImportError`。

- [ ] **Step 3: 实现 selection metrics 和冻结硬门**

扩展 `_choice_metrics()`：对 `choice != default_index` 且 `active_count > 0` 的 executed samples 计算：

```python
sample_benefit = (
    outcome.active_sse[np.arange(choice.size), outcome.default_index]
    - outcome.active_sse[np.arange(choice.size), choice]
)
executed = (choice != outcome.default_index) & (outcome.active_count > 0)
executed_positive_precision = (
    float(np.mean(sample_benefit[executed] > 0.0)) if np.any(executed) else 1.0
)
negative_selection_rate = (
    float(np.mean(sample_benefit[executed] < 0.0)) if np.any(executed) else 0.0
)
improved_seed_count = sum(
    row["rmse"] < row["default_rmse"] for row in per_seed
)
activity_f1_drop = float(default_metrics["activity_f1"] - selected_metrics["activity_f1"])
link_rmse_relative_degradation = float(
    (selected_metrics["link_rmse"] - default_metrics["link_rmse"])
    / max(default_metrics["link_rmse"], 1e-12)
)
```

逐 seed row 同时写 `default_rmse`。新增：

```python
def classify_selector_validation_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "rmse": float(metrics["rmse"]) < 230.8556,
        "improved_seed_count": int(metrics["improved_seed_count"]) >= 7,
        "positive_precision": float(metrics["executed_positive_precision"]) >= 0.65,
        "negative_selection_rate": float(metrics["negative_selection_rate"]) <= 0.20,
        "activity_f1_drop": float(metrics["activity_f1_drop"]) <= 0.002,
        "link_rmse_relative_degradation": (
            float(metrics["link_rmse_relative_degradation"]) <= 0.02
        ),
        "training_seed_std": float(metrics["training_seed_std"]) <= 5.0,
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": dict(metrics)}
```

每个 config row 保存这些字段；选出 best 后调用该函数。`configuration_frozen`、`result_kind` 和 matched-test 权限必须使用 `candidate_gate["passed"] and selector_validation_gate["passed"]`，不能再只使用 candidate-library gate。manifest 同时保留两种 gate，避免语义混淆。

- [ ] **Step 4: 运行 trainer 合同和回归测试**

Run:

```powershell
python -m unittest tests.test_v11_selector_finalization.SelectorTrainingRunnerContractTest -v
```

Expected: 全部 `OK`；历史全 defer 结果按新门应为 `configuration_frozen=false`。

- [ ] **Step 5: 根据审计结果决定是否训练**

只有新 OOF 审计至少显示 helper-dependent 候选的 train/calibration/validation 收益方向比旧 in-sample 协议一致，才运行现有 CandidateSetBenefitRanker。若仍为 `not_identifiable` 且 rank Spearman、positive precision 和 validation RMSE 均无改善，则停止扩模并记录根因，不打开 test。

- [ ] **Step 6: 运行冻结的 12×3 selector 网格**

```bash
python scripts/train_v11_candidate_set_selector.py \
  --train-cache artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/label_cache_schema6/merged/candidate_labels_train.npz \
  --calibration-cache artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/label_cache_schema6/eval/candidate_labels_calibration.npz \
  --validation-cache artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/label_cache_schema6/eval/candidate_labels_validation.npz \
  --output-dir artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/selector_training \
  --device cuda --hidden-dim 64 128 --temperature 0.1 0.25 0.5 \
  --dropout 0 0.1 --training-seeds 17 29 41 --epochs 20
```

- [ ] **Step 7: 执行 validation 硬门审计**

从 frozen manifest/summary 验证全部条件：RMSE `<230.8556`、改善 `>=7/10` seeds、positive precision `>=0.65`、negative-selection rate `<=0.20`、F1 drop `<=0.002`、link RMSE degradation `<=0.02`、三个训练 seed 的 RMSE 标准差 `<=5`。任一失败则 `configuration_frozen=false`，matched test 保持未访问。

- [ ] **Step 8: 仅在全部硬门通过时运行 feature ablation**

使用现有 `run_v11_selector_feature_ablation.py`，固定 checkpoint、calibration defer 和 training seeds；去掉 stage/task/resource/energy/uncertainty 五组，输出不得重新选择结构。

- [ ] **Step 9: 仅在 frozen manifest 有效时先生成 external holdout 证据**

```bash
python scripts/run_v11_selector_candidate_labels.py \
  --splits external_holdout --helper-protocol seed_crossfit_5fold \
  --frozen-config-manifest artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/selector_training/frozen_selector_manifest.json \
  --cache-schema-version 6 --helper-train-limit 0 --split-sample-limit 0 \
  --device cuda --batch-size 64 --rf-trees 160 \
  --output-dir artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/locked_labels

python scripts/evaluate_v11_frozen_selector.py \
  --frozen-manifest artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/selector_training/frozen_selector_manifest.json \
  --cache artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/locked_labels/candidate_labels_external_holdout.npz \
  --output-dir artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/frozen_evaluation/external \
  --device cuda
```

external 只用于独立评价，不返回修改结构、阈值或 checkpoint；记录 10 个 seed 中优于 ranked baseline 的数量。

- [ ] **Step 10: 携带 external 证据一次性打开 matched test**

```bash
python scripts/run_v11_selector_candidate_labels.py \
  --splits matched_test --helper-protocol seed_crossfit_5fold \
  --frozen-config-manifest artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/selector_training/frozen_selector_manifest.json \
  --cache-schema-version 6 --helper-train-limit 0 --split-sample-limit 0 \
  --device cuda --batch-size 64 --rf-trees 160 \
  --output-dir artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/locked_labels

python scripts/evaluate_v11_frozen_selector.py \
  --frozen-manifest artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/selector_training/frozen_selector_manifest.json \
  --cache artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/locked_labels/candidate_labels_matched_test.npz \
  --external-summary artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/frozen_evaluation/external/summary_external_holdout.json \
  --output-dir artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/frozen_evaluation/matched \
  --device cuda
```

matched test 只执行上述一次，不调阈值、不重选 checkpoint。若尚无与该 selector freeze digest 对齐的 AirFogSim task--energy safety audit，则 acceptance 中如实标记 safety pending，不伪造 A 级完整验收。

- [ ] **Step 11: 按冻结标准分类结果**

- matched test RMSE `<200`、external 至少 `7/10` seeds 改善且安全证据无 Pareto 违规：A 级完整突破；
- matched test RMSE `<200` 但 external 或安全证据未通过：指标突破，但最终 A 级验收仍 pending；
- `[200,213.160874)`：B 级改善，仍为 `v11 candidate`；
- `>=213.160874`：未定型；
- 任何 oracle/test-best/rank-only：保持 `sample_oracle` 或 `diagnostic_only` 标签。

- [ ] **Step 12: 提交冻结门修复**

```powershell
git add 代码/scripts/train_v11_candidate_set_selector.py 代码/tests/test_v11_selector_finalization.py
git commit -m "fix: gate selector freeze on validation performance"
```

### Task 10: 全量验证、文档状态和结果冻结

**Files:**
- Modify: `本地计划表.md`
- Generated: `代码/artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/summary.json`
- Generated: `代码/artifacts/reports/pi_jwm_v11_selector_helper_seed_crossfit_20260719/sha256_manifest.json`

- [ ] **Step 1: 更新本地计划表**

写明：旧 schema-v6 selector 失败、helper in-sample concept shift 根因、crossfit 实施状态、validation 是否通过、matched test 是否访问。PI-JWM 仍为主线，selector 保持 `v11 candidate`，除非 A 级全部验收完成。

- [ ] **Step 2: 运行相关脚本测试和全量测试**

```powershell
cd D:\shen\网络组\代码
python -m unittest tests.test_v11_crossfit tests.test_v11_schema6_interactions tests.test_v11_benefit_identifiability tests.test_v11_selector_finalization -v
python -m unittest discover -s tests -p "test_*.py"
```

Expected: 全部测试通过；记录准确测试数和 runtime，不能只报告 exit code。

- [ ] **Step 3: 冻结结果清单**

`summary.json` 至少记录：source Git SHA、两个 protocol/config digest、三 split cache SHA、candidate-shift classification、identifiability classification、validation hard-gate 每一项、matched/external accessed 状态和最终 tier。`sha256_manifest.json` 覆盖所有 CSV、JSON、checkpoint、命令和关键日志。

- [ ] **Step 4: 检查产物可追溯性**

每个结果表必须能追溯到 cache SHA、seed、候选名称、配置 digest 和复现命令；任何缺字段、旧新 cache 混用或 result-kind 错标均阻止完成声明。

- [ ] **Step 5: 提交最终源码和文档状态**

```powershell
git add 本地计划表.md 代码/src/pi_jwm 代码/scripts 代码/tests
git commit -m "docs: record selector crossfit validation outcome"
git status --short
```

Expected: commit 成功，tracked worktree clean；生成 artifacts 不因体积被误提交。
