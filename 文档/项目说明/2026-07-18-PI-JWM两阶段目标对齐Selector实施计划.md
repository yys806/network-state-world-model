# PI-JWM Objective-Aligned Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and locally verify a two-stage selector whose opportunity and candidate scores are trained against global-SSE-aligned benefit, stopping immediately before the formal RTX 4090 run.

**Architecture:** Add one focused PI-JWM module for decision-aligned targets, the `OpportunityBenefitRanker`, fitting, calibration, and safe selection. A separate runner consumes the existing schema-v5 train/calibration/validation caches, writes auditable comparison/trace/freeze artifacts, and never exposes matched-test or external data. Existing world-model checkpoints, candidate labels, and historical baselines remain frozen.

**Tech Stack:** Python 3.12, NumPy, PyTorch, scikit-learn, optional existing XGBoost, `unittest`, Bash, JSON/CSV/SHA-256 artifacts.

---

## File map

- Create `代码/src/pi_jwm/v11_objective_aligned_selector.py`: targets, model, losses, fit/predict, calibration, selection, checkpoint I/O.
- Create `代码/tests/test_v11_objective_aligned_selector.py`: pure target/loss/model/calibration/selection and determinism tests.
- Create `代码/scripts/train_v11_objective_aligned_selector.py`: schema-v5 cache loading, fixed grid, validation selection, optional XGBoost baseline, traces and freeze manifest.
- Create `代码/scripts/run_v11_objective_aligned_selector_gpu.sh`: locked-split-free formal GPU entry point.
- Modify `代码/tests/test_v11_selector_finalization.py`: runner and launcher protocol tests.
- Modify `本地计划表.md`: record local quality-gate status and exact next GPU command.
- Generate `代码/artifacts/reports/pi_jwm_v11_objective_aligned_selector_20260718/`: local smoke and later formal artifacts; generated files remain outside Git.

### Task 1: Decision-aligned targets and impact weights

**Files:**
- Create: `代码/src/pi_jwm/v11_objective_aligned_selector.py`
- Create: `代码/tests/test_v11_objective_aligned_selector.py`

- [ ] **Step 1: Write failing target-construction tests**

```python
class DecisionAlignedTargetsTest(unittest.TestCase):
    def test_targets_use_ranked_default_sse_and_masked_oracle(self):
        outcome = CandidateOutcome(
            active_sse=np.asarray([[100.0, 80.0, 20.0], [9.0, 9.0, 12.0]], dtype=np.float32),
            active_count=np.asarray([2, 1]),
            default_index=1,
        )
        mask = np.asarray([[True, True, True], [True, True, False]])
        targets = build_decision_aligned_targets(outcome, mask, weight_cap=5.0)
        np.testing.assert_allclose(targets.candidate_benefit[0], [-20.0, 0.0, 60.0])
        np.testing.assert_allclose(targets.opportunity, [60.0, 0.0])
        self.assertEqual(targets.benefit_scale, 60.0)
        self.assertEqual(targets.positive_opportunity.tolist(), [True, False])

    def test_zero_active_rows_are_audited_but_not_trainable(self):
        outcome = CandidateOutcome(
            active_sse=np.zeros((1, 2), dtype=np.float32),
            active_count=np.zeros((1,), dtype=np.int64),
            default_index=1,
        )
        targets = build_decision_aligned_targets(outcome, np.ones((1, 2), dtype=bool))
        self.assertFalse(targets.valid_sample[0])
        self.assertEqual(targets.sample_weight[0], 0.0)

    def test_weight_cap_limits_high_gain_outlier(self):
        outcome = CandidateOutcome(
            active_sse=np.asarray([[11.0, 10.0], [1001.0, 1000.0]], dtype=np.float32),
            active_count=np.ones(2, dtype=np.int64),
            default_index=1,
        )
        targets = build_decision_aligned_targets(outcome, np.ones((2, 2), dtype=bool), weight_cap=5.0)
        self.assertLessEqual(float(targets.sample_weight.max()), 5.25)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
cd D:\shen\网络组\代码\tests
python -m unittest test_v11_objective_aligned_selector.DecisionAlignedTargetsTest
```

Expected: import failure because `v11_objective_aligned_selector` does not exist.

- [ ] **Step 3: Implement immutable targets and validation**

```python
@dataclass(frozen=True)
class DecisionAlignedTargets:
    candidate_benefit: np.ndarray
    opportunity: np.ndarray
    positive_opportunity: np.ndarray
    valid_sample: np.ndarray
    sample_weight: np.ndarray
    benefit_scale: float
    weight_cap: float


def build_decision_aligned_targets(
    outcome: CandidateOutcome,
    candidate_mask: np.ndarray,
    weight_cap: float = 5.0,
    base_weight: float = 0.25,
    benefit_scale: float | None = None,
) -> DecisionAlignedTargets:
    mask = np.asarray(candidate_mask, dtype=bool)
    if mask.shape != outcome.active_sse.shape or not np.all(mask[:, outcome.default_index]):
        raise ValueError("candidate mask must match outcomes and include the ranked default")
    if weight_cap <= 0.0 or base_weight < 0.0:
        raise ValueError("weight cap must be positive and base weight non-negative")
    default_sse = outcome.active_sse[:, outcome.default_index, None]
    benefit = default_sse - outcome.active_sse
    benefit = np.where(mask, benefit, np.nan).astype(np.float32)
    valid = outcome.active_count > 0
    opportunity = np.zeros(outcome.active_count.shape, dtype=np.float32)
    opportunity[valid] = np.maximum(0.0, np.nanmax(benefit[valid], axis=1))
    positive = valid & (opportunity > 1e-8)
    scale = float(np.median(opportunity[positive])) if benefit_scale is None and np.any(positive) else float(benefit_scale or 1.0)
    scale = max(scale, 1e-6)
    weight = np.zeros_like(opportunity)
    weight[valid] = float(base_weight) + np.minimum(opportunity[valid] / scale, float(weight_cap))
    return DecisionAlignedTargets(benefit, opportunity, positive, valid, weight, scale, float(weight_cap))
```

- [ ] **Step 4: Run target tests and verify GREEN**

Run the Step 2 command. Expected: all target tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- 代码/src/pi_jwm/v11_objective_aligned_selector.py 代码/tests/test_v11_objective_aligned_selector.py
git commit -m "feat: add decision-aligned selector targets"
```

### Task 2: Unified opportunity/benefit model and weighted losses

**Files:**
- Modify: `代码/src/pi_jwm/v11_objective_aligned_selector.py`
- Modify: `代码/tests/test_v11_objective_aligned_selector.py`

- [ ] **Step 1: Write failing loss and permutation tests**

```python
class OpportunityBenefitRankerTest(unittest.TestCase):
    def test_weighted_listwise_prioritizes_high_impact_sample(self):
        predicted = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        benefit = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        mask = torch.ones_like(predicted, dtype=torch.bool)
        high_first = weighted_listwise_benefit_loss(predicted, benefit, mask, torch.tensor([10.0, 1.0]))
        high_second = weighted_listwise_benefit_loss(predicted.flip(1), benefit, mask, torch.tensor([10.0, 1.0]))
        self.assertLess(float(high_first), float(high_second))

    def test_model_is_candidate_permutation_equivariant(self):
        torch.manual_seed(4)
        model = OpportunityBenefitRanker(5, 3, hidden_dim=8)
        candidate = torch.randn(2, 4, 5)
        context = torch.randn(2, 3)
        mask = torch.ones(2, 4, dtype=torch.bool)
        permutation = torch.tensor([2, 0, 3, 1])
        original = model(candidate, context, mask)
        permuted = model(candidate[:, permutation], context, mask[:, permutation])
        inverse = torch.argsort(permutation)
        for field in ("predicted_candidate_benefit", "candidate_uncertainty"):
            torch.testing.assert_close(original[field], permuted[field][:, inverse])
        torch.testing.assert_close(original["predicted_opportunity"], permuted["predicted_opportunity"])
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest test_v11_objective_aligned_selector.OpportunityBenefitRankerTest
```

Expected: missing model/loss symbols.

- [ ] **Step 3: Implement the unified model and per-sample weighted listwise loss**

Implement `OpportunityBenefitRanker` with the existing candidate encoder, masked mean set pooling, context encoder and stage embedding. Use one candidate benefit head, one candidate uncertainty head, one sample opportunity head and one sample opportunity uncertainty head. Masked candidates receive benefit `-1e9` for ranking and uncertainty `0`.

Implement:

```python
def weighted_listwise_benefit_loss(predicted, target_benefit, mask, sample_weight, temperature=0.25):
    valid = mask.any(1) & torch.isfinite(target_benefit.masked_fill(~mask, 0.0)).all(1) & (sample_weight > 0)
    target = torch.softmax((target_benefit / temperature).masked_fill(~mask, -1e9), dim=1)
    per_sample = -(target * torch.log_softmax(predicted.masked_fill(~mask, -1e9), dim=1)).sum(1)
    return (per_sample[valid] * sample_weight[valid]).sum() / sample_weight[valid].sum().clamp_min(1e-8)
```

- [ ] **Step 4: Run model/loss tests and verify GREEN**

Run Step 2. Expected: all tests pass with no warning.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- 代码/src/pi_jwm/v11_objective_aligned_selector.py 代码/tests/test_v11_objective_aligned_selector.py
git commit -m "feat: add opportunity benefit ranker"
```

### Task 3: Deterministic fitting, prediction, and safe checkpoints

**Files:**
- Modify: `代码/src/pi_jwm/v11_objective_aligned_selector.py`
- Modify: `代码/tests/test_v11_objective_aligned_selector.py`

- [ ] **Step 1: Write failing fit/checkpoint tests**

```python
def synthetic_batch_and_outcome():
    rng = np.random.default_rng(4)
    batch = CandidateBatch(
        context=rng.normal(size=(8, 3)).astype(np.float32),
        candidate_features=rng.normal(size=(8, 4, 5)).astype(np.float32),
        candidate_mask=np.ones((8, 4), dtype=bool),
        stage=np.asarray(["offload", "compute", "return", "unknown"] * 2),
        feature_names=("a", "b", "c", "d", "e"),
        candidate_names=("identity", "ranked", "repair_a", "repair_b"),
    )
    outcome = CandidateOutcome(
        active_sse=np.abs(rng.normal(size=(8, 4))).astype(np.float32) + 0.1,
        active_count=np.ones((8,), dtype=np.int64),
        default_index=1,
    )
    return batch, outcome


def test_fit_is_deterministic_and_freezes_train_only_scales(self):
    batch, outcome = synthetic_batch_and_outcome()
    first = fit_objective_aligned_selector(batch, outcome, hidden_dim=8, weight_cap=5, epochs=3, seed=17)
    second = fit_objective_aligned_selector(batch, outcome, hidden_dim=8, weight_cap=5, epochs=3, seed=17)
    self.assertEqual(first.benefit_scale, second.benefit_scale)
    np.testing.assert_array_equal(first.candidate_mean, second.candidate_mean)
    for left, right in zip(first.model.parameters(), second.model.parameters()):
        torch.testing.assert_close(left, right)

def test_checkpoint_loader_rejects_configuration_digest_mismatch(self):
    batch, outcome = synthetic_batch_and_outcome()
    fitted = fit_objective_aligned_selector(batch, outcome, hidden_dim=8, epochs=1)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "selector.pt"
        save_objective_aligned_checkpoint(path, fitted, "a" * 64, training_seed=17)
        with self.assertRaisesRegex(ValueError, "digest"):
            load_objective_aligned_checkpoint(path, expected_configuration_digest="b" * 64)
```

- [ ] **Step 2: Run the fit/checkpoint tests and verify RED**

Expected: missing fit/checkpoint APIs.

- [ ] **Step 3: Implement fitting and prediction**

Add `FittedObjectiveAlignedSelector` with model, normalization arrays, benefit scale, weight cap, history and stage vocabulary. In `fit_objective_aligned_selector`:

- seed Python, NumPy, Torch and CUDA;
- calculate normalization and benefit scale from train only;
- train weighted listwise, weighted candidate Huber, opportunity Huber, worst-seed and heteroscedastic NLL using the fixed coefficients from the design;
- skip listwise when a batch contains no positive opportunity, while retaining benefit/opportunity losses;
- return complete history and frozen statistics.

`predict_objective_aligned_selector` must reuse frozen normalization and return NumPy arrays in original SSE units by multiplying benefit/opportunity outputs and uncertainties by `benefit_scale`.

Checkpoint payloads must use Torch tensors for arrays so `torch.load(..., weights_only=True)` succeeds. Bind candidate/context dimensions, hidden dimension, weight cap, training seed and configuration digest.

- [ ] **Step 4: Run all new module tests and verify GREEN**

```powershell
python -m unittest test_v11_objective_aligned_selector.py
```

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- 代码/src/pi_jwm/v11_objective_aligned_selector.py 代码/tests/test_v11_objective_aligned_selector.py
git commit -m "feat: fit objective-aligned selector"
```

### Task 4: Calibration and objective-aligned defer

**Files:**
- Modify: `代码/src/pi_jwm/v11_objective_aligned_selector.py`
- Modify: `代码/tests/test_v11_objective_aligned_selector.py`

- [ ] **Step 1: Write failing calibration/selection tests**

```python
def test_calibration_uses_only_fixed_opportunity_quantiles(self):
    opportunity_lcb = np.asarray([0.0, 1.0, 2.0, 3.0])
    candidate_choice = np.asarray([1, 1, 1, 1])
    outcome = CandidateOutcome(
        active_sse=np.asarray([[4, 1], [4, 5], [4, 1], [4, 5]], dtype=np.float32),
        active_count=np.ones(4, dtype=np.int64), default_index=0,
    )
    result = calibrate_opportunity_threshold(opportunity_lcb, candidate_choice, outcome)
    self.assertIn(result.quantile, (0.0, 0.25, 0.5, 0.75, 0.9))

def test_selection_requires_both_opportunity_and_candidate_positive_lcb(self):
    decision = select_objective_aligned(
        ensemble_candidate_benefit=np.asarray([[[0, 4]], [[0, 4]], [[0, 4]]], dtype=np.float32),
        ensemble_candidate_uncertainty=np.zeros((3, 1, 2), dtype=np.float32),
        ensemble_opportunity=np.asarray([[5], [5], [5]], dtype=np.float32),
        ensemble_opportunity_uncertainty=np.zeros((3, 1), dtype=np.float32),
        candidate_mask=np.ones((1, 2), dtype=bool), default_index=0, opportunity_threshold=1.0,
    )
    self.assertEqual(decision.candidate_index[0], 1)
    self.assertFalse(decision.deferred[0])
```

- [ ] **Step 2: Run tests and verify RED**

Expected: missing calibration/selection APIs.

- [ ] **Step 3: Implement calibration and selection**

Add immutable `OpportunityCalibration` and `ObjectiveAlignedDecision`. Calibration evaluates only quantiles `(0, .25, .5, .75, .9)` on calibration outcomes, chooses the lowest aggregate RMSE, then higher defer ratio, then higher threshold.

Selection must:

- rank by ensemble mean candidate benefit among legal, non-dominated candidates;
- combine epistemic variance and mean predicted aleatoric variance;
- require opportunity LCB above the frozen threshold and selected candidate LCB above zero;
- otherwise choose default and emit one of `opportunity_below_threshold`, `candidate_nonpositive_lcb`, `pareto_dominated`, or `no_nondefault_candidate`.

- [ ] **Step 4: Run module tests and verify GREEN**

Run all `test_v11_objective_aligned_selector.py` tests.

- [ ] **Step 5: Commit Task 4**

```powershell
git add -- 代码/src/pi_jwm/v11_objective_aligned_selector.py 代码/tests/test_v11_objective_aligned_selector.py
git commit -m "feat: calibrate objective-aligned defer"
```

### Task 5: Train/validate runner and explanation artifacts

**Files:**
- Create: `代码/scripts/train_v11_objective_aligned_selector.py`
- Modify: `代码/tests/test_v11_selector_finalization.py`

- [ ] **Step 1: Write failing runner contract tests**

Test that the runner:

```python
def test_objective_aligned_runner_rejects_non_schema5_cache(self):
    from train_v11_objective_aligned_selector import validate_objective_cache_protocol
    manifests = {
        name: {
            "schema_version": 4,
            "split_name": name,
            "configuration_digest": "a" * 64,
            "candidate_names": ["identity", "ranked"],
            "feature_names": ["x"],
            "context_feature_names": ["ctx"],
        }
        for name in ("train", "calibration", "validation")
    }
    with self.assertRaisesRegex(ValueError, "schema 5"):
        validate_objective_cache_protocol(manifests)

def test_objective_aligned_runner_grid_is_exactly_four_configs(self):
    from train_v11_objective_aligned_selector import build_grid
    self.assertEqual(build_grid([64, 128], [5, 10]), [(64, 5.0), (64, 10.0), (128, 5.0), (128, 10.0)])

def test_validation_success_gate_requires_all_metric_constraints(self):
    from train_v11_objective_aligned_selector import classify_validation_result
    passing = {
        "rmse": 199.0,
        "training_seed_std": 4.0,
        "improved_seed_count": 7,
        "activity_f1_drop": 0.001,
        "link_rmse_relative_degradation": 0.01,
    }
    self.assertEqual(classify_validation_result(passing), "success")
    for key, value in (
        ("rmse", 200.0),
        ("training_seed_std", 5.1),
        ("improved_seed_count", 6),
        ("activity_f1_drop", 0.0021),
        ("link_rmse_relative_degradation", 0.021),
    ):
        failed = dict(passing)
        failed[key] = value
        self.assertNotEqual(classify_validation_result(failed), "success")

def test_runner_source_exposes_no_locked_split_cli(self):
    text = (SCRIPTS_ROOT / "train_v11_objective_aligned_selector.py").read_text(encoding="utf-8")
    self.assertNotIn("--matched-test", text)
    self.assertNotIn("--external-holdout", text)
```

- [ ] **Step 2: Run tests and verify RED**

Expected: runner import failure.

- [ ] **Step 3: Implement the runner**

The runner must accept only `--train-cache`, `--calibration-cache`, `--validation-cache`, `--output-dir`, `--device`, hidden dimensions, weight caps, training seeds and epochs. Reuse `validate_cache_protocol` and `_choice_metrics` from the established runner, requiring schema 5.

For each of four configs and three seeds:

- fit train only;
- predict calibration and validation;
- calibrate opportunity threshold on calibration only;
- save safe checkpoint and hashes;
- calculate rank-only and deployable validation metrics.

Write:

- `selector_grid_results.csv`;
- `selector_comparison.csv` with frozen old baselines plus new configs;
- `opportunity_calibration.csv`;
- `decision_trace_validation.csv` containing sample id/seed/stage/candidate/predicted benefit/LCB/opportunity/defer reason and diagnostic actual benefit;
- `gain_concentration.csv`;
- `feature_and_decision_ablation.csv`;
- `frozen_selector_manifest.json` with `matched_test_accessed=false` and `external_holdout_accessed=false`;
- `summary.json` with success/partial/failure classification.

Use optional XGBoost only when importable. It must use the same flattened legal candidate/context features and sample weights. If unavailable, write a comparison row with status `skipped_dependency_unavailable`.

After freezing the best validation config, run diagnostic-only ablations without changing the selected configuration: `without_opportunity` and `without_uncertainty` reuse the frozen ensemble with the named decision component disabled; `uniform_impact`, `without_stage`, `without_task`, `without_resource`, and `without_energy` retrain only training seed 17 for 100 epochs using the frozen hidden dimension and weight cap. Save all rows as `diagnostic_only`; ablation outcomes cannot alter the selected configuration or calibration rule.

- [ ] **Step 4: Run runner contract tests and a tiny synthetic run**

```powershell
python -m unittest test_v11_selector_finalization.SelectorTrainingRunnerContractTest
python 代码/scripts/train_v11_objective_aligned_selector.py --help
```

Expected: tests pass; help contains no locked-split option.

- [ ] **Step 5: Commit Task 5**

```powershell
git add -- 代码/scripts/train_v11_objective_aligned_selector.py 代码/tests/test_v11_selector_finalization.py
git commit -m "feat: add objective-aligned selector runner"
```

### Task 6: Formal GPU launcher and locked-split audit

**Files:**
- Create: `代码/scripts/run_v11_objective_aligned_selector_gpu.sh`
- Modify: `代码/tests/test_v11_selector_finalization.py`

- [ ] **Step 1: Write a failing launcher-source test**

```python
def test_objective_aligned_gpu_launcher_uses_only_existing_schema5_caches(self):
    text = (SCRIPTS_ROOT / "run_v11_objective_aligned_selector_gpu.sh").read_text(encoding="utf-8")
    self.assertIn("candidate_labels_train.npz", text)
    self.assertIn("candidate_labels_calibration.npz", text)
    self.assertIn("candidate_labels_validation.npz", text)
    self.assertNotIn("matched_test", text)
    self.assertNotIn("external_holdout", text)
    self.assertNotIn("run_v11_selector_candidate_labels.py", text)
```

- [ ] **Step 2: Run test and verify RED**

Expected: launcher file missing.

- [ ] **Step 3: Implement the launcher**

The Bash script resolves the code root, verifies all three cache files and manifests, checks CUDA availability, creates `代码/artifacts/reports/pi_jwm_v11_objective_aligned_selector_20260718/`, and runs exactly:

```bash
python scripts/train_v11_objective_aligned_selector.py \
  --train-cache "$CACHE/candidate_labels_train.npz" \
  --calibration-cache "$CACHE/candidate_labels_calibration.npz" \
  --validation-cache "$CACHE/candidate_labels_validation.npz" \
  --output-dir "$REPORT/selector_training" \
  --device cuda --hidden-dim 64 128 --weight-cap 5 10 \
  --training-seeds 17 29 41 --epochs 200
```

It must not regenerate labels, retrain the world model, repeat classical baselines, or mention locked split names.

- [ ] **Step 4: Validate launcher and tests**

```powershell
bash -n 代码/scripts/run_v11_objective_aligned_selector_gpu.sh
python -m unittest test_v11_selector_finalization.py
```

- [ ] **Step 5: Commit Task 6**

```powershell
git add -- 代码/scripts/run_v11_objective_aligned_selector_gpu.sh 代码/tests/test_v11_selector_finalization.py
git commit -m "feat: add objective-aligned GPU protocol"
```

### Task 7: Local CPU smoke, full verification, and GPU handoff

**Files:**
- Modify: `本地计划表.md`
- Generate: `代码/artifacts/reports/pi_jwm_v11_objective_aligned_selector_20260718/local_smoke64/`
- Generate: `代码/artifacts/reports/pi_jwm_v11_objective_aligned_selector_20260718/local_smoke256/`

- [ ] **Step 1: Run 64-sample CPU smoke**

Use temporary subset caches produced from the existing schema-v5 train/calibration/validation caches without changing sample order or metadata. Run one config, one training seed, 2 epochs. Expected: finite losses, valid calibration threshold, valid decision trace, no locked split access.

- [ ] **Step 2: Run 256-sample CPU smoke**

Run one config, seeds `17,29,41`, 5 epochs. Expected: all checkpoints load with `weights_only=True`, train-only scales match across split predictions, repeated seed 17 decisions are identical.

- [ ] **Step 3: Run complete verification**

```powershell
cd D:\shen\网络组\代码
python -m unittest discover -s tests -p "test_*.py"
python -m unittest discover -s scripts -p "test_*.py"
python -m py_compile src/pi_jwm/v11_objective_aligned_selector.py scripts/train_v11_objective_aligned_selector.py
bash -n scripts/run_v11_objective_aligned_selector_gpu.sh
git diff --check
```

Expected: all commands return zero.

- [ ] **Step 4: Update the local plan**

Record exact test counts, smoke directories, schema-v5 cache hashes, source Git SHA, the GPU launcher command, and the explicit statement that external remains unopened.

- [ ] **Step 5: Commit the local-ready checkpoint**

```powershell
git add -- 本地计划表.md 代码/src/pi_jwm/v11_objective_aligned_selector.py 代码/scripts/train_v11_objective_aligned_selector.py 代码/scripts/run_v11_objective_aligned_selector_gpu.sh 代码/tests/test_v11_objective_aligned_selector.py 代码/tests/test_v11_selector_finalization.py
git commit -m "feat: prepare objective-aligned selector GPU run"
```

- [ ] **Step 6: Stop before GPU and report handoff**

Do not start the server run in this task. Report local verification evidence, exact launcher path, expected 1–2 hour RTX 4090 usage, and whether the server needs to be started or reconnected.

### Task 8: Formal RTX 4090 validation run — later checkpoint

Run only after Task 7 is accepted and a server is available. Sync the exact Git commit, verify the three cache SHA-256 values, run `bash scripts/run_v11_objective_aligned_selector_gpu.sh`, download the complete report, and apply the `<200`/partial/failure gates without changing thresholds or opening external.

### Task 9: External one-shot — conditional later checkpoint

This task is forbidden unless Task 8 produces validation RMSE `<200`, training-seed standard deviation `≤5`, at least 7/10 improved validation seeds, F1 degradation `≤0.002`, and link-RMSE degradation `≤2%`. Freeze the unique configuration first; then and only then generate/evaluate external seeds `60–69` once. Never reopen matched test seeds `18–19`.
