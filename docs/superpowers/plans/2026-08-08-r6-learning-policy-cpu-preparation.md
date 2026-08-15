# R6 Learning Policy CPU Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a CPU-only PI-JWM learning-policy interface with explicit/latent state input, action masks, safety projection, Masked Actor–Critic, PPO loss, frozen-world-model isolation, and an auditable nonlocked preflight bundle.

**Architecture:** Add focused contract, safety, policy, training, and preflight modules under `代码/src/pi_jwm/`. The policy consumes detached explicit/latent state, proposes offload/RB/CPU actions through masked heads, and always passes through a shared safety projector before execution. A thin CPU runner binds the implementation to existing R5/R6 artifacts without changing the world model or AirFogSim.

**Tech Stack:** Python 3.12, PyTorch, NumPy, standard-library `unittest`, JSON/CSV/SHA-256 manifests.

---

### Task 1: Policy state and action contracts

**Files:**
- Create: `代码/src/pi_jwm/r6_learning_policy_contract.py`
- Create: `代码/tests/test_r6_learning_policy_contract.py`

- [ ] **Step 1: Write failing contract tests**

```python
def test_policy_state_detaches_world_model_tensors(self):
    latent = torch.ones(2, 4, requires_grad=True)
    state = PolicyState.create(
        explicit=torch.ones(2, 3), latent=latent,
        offload_mask=torch.tensor([[1, 0], [1, 1]], dtype=torch.bool),
        rb_mask=torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
        cpu_task_mask=torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        cpu_capacity=torch.tensor([10.0, 5.0]),
        scenario_ids=("s0", "s1"), seeds=(7, 8), slots=(1, 1),
        protocol_fingerprint="frozen-r6",
    )
    self.assertFalse(state.latent.requires_grad)
    self.assertIsNone(state.latent.grad_fn)

def test_policy_state_rejects_future_target_and_locked_test(self):
    arguments = make_valid_policy_state_arguments()
    arguments["extra_fields"] = {"future_target": torch.ones(1)}
    with self.assertRaisesRegex(ValueError, "future target"):
        PolicyState.create(**arguments)
    with self.assertRaisesRegex(ValueError, "locked_test"):
        PolicyIdentity("s0", 7, 1, "locked_test", "frozen-r6")
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `cd 代码/tests; python -m unittest test_r6_learning_policy_contract.py`

Expected: import failure because `pi_jwm.r6_learning_policy_contract` does not exist.

- [ ] **Step 3: Implement immutable validated contracts**

```python
@dataclass(frozen=True)
class PolicyIdentity:
    scenario_id: str
    seed: int
    slot: int
    split: str
    protocol_fingerprint: str

@dataclass(frozen=True)
class PolicyState:
    explicit: Tensor
    latent: Tensor
    offload_mask: Tensor
    rb_mask: Tensor
    cpu_task_mask: Tensor
    cpu_capacity: Tensor
    identities: Sequence[PolicyIdentity]

    @classmethod
    def create(cls, *, explicit, latent, offload_mask, rb_mask,
               cpu_task_mask, cpu_capacity, identities, extra_fields=None):
        reject_future_and_locked_fields(extra_fields, identities)
        validate_state_inputs(explicit, latent, offload_mask, rb_mask,
                              cpu_task_mask, cpu_capacity, identities)
        return cls(
            explicit=explicit.detach().clone(),
            latent=latent.detach().clone(),
            offload_mask=offload_mask.detach().clone().bool(),
            rb_mask=rb_mask.detach().clone().bool(),
            cpu_task_mask=cpu_task_mask.detach().clone().bool(),
            cpu_capacity=cpu_capacity.detach().clone(),
            identities=tuple(identities),
        )
```

`ActionSpec`, `ProposedAction`, `ExecutableAction`, `ProjectionRecord`, and `PolicyOutput` must be dataclasses in this module. `ActionSpec` validates action dimensions and no-op indices; no legal discrete action is represented by the configured no-op index, never by an empty distribution.

- [ ] **Step 4: Run contract tests and verify GREEN**

Run: `cd 代码/tests; python -m unittest test_r6_learning_policy_contract.py`

Expected: all contract tests pass.

### Task 2: Shared safety projector

**Files:**
- Create: `代码/src/pi_jwm/r6_learning_policy_safety.py`
- Create: `代码/tests/test_r6_learning_policy_safety.py`

- [ ] **Step 1: Write failing projection tests**

```python
def test_projector_masks_discrete_actions_and_projects_cpu_capacity(self):
    proposed = ProposedAction(
        offload_index=torch.tensor([2]), rb_index=torch.tensor([1]),
        cpu_allocation=torch.tensor([[8.0, 8.0]]),
    )
    result = SafetyProjector().project(state, proposed, spec)
    self.assertEqual(result.action.offload_index.item(), spec.offload_noop_index)
    self.assertLessEqual(result.action.cpu_allocation.sum().item(), 10.0 + 1e-7)
    self.assertTrue(any(row.reason == "masked_offload_to_noop" for row in result.records))

def test_projector_rejects_nonfinite_and_post_projection_violation(self):
    with self.assertRaisesRegex(ValueError, "finite"):
        SafetyProjector().project(state, proposed_with_nan, spec)
```

- [ ] **Step 2: Run and verify RED**

Run: `cd 代码/tests; python -m unittest test_r6_learning_policy_safety.py`

Expected: import failure because the safety module is absent.

- [ ] **Step 3: Implement fixed-order projection**

```python
class SafetyProjector:
    def project(self, state: PolicyState, proposed: ProposedAction,
                spec: ActionSpec) -> ProjectionResult:
        check_finite(proposed)
        offload, offload_rows = project_discrete(
            proposed.offload_index, state.offload_mask, spec.offload_noop_index, "offload"
        )
        rb, rb_rows = project_discrete(
            proposed.rb_index, state.rb_mask, spec.rb_noop_index, "rb"
        )
        cpu, cpu_rows = project_cpu(
            proposed.cpu_allocation, state.cpu_task_mask, state.cpu_capacity
        )
        assert_post_projection_constraints(state, offload, rb, cpu, spec)
        return ProjectionResult(
            ExecutableAction(offload, rb, cpu),
            tuple(offload_rows + rb_rows + cpu_rows),
        )
```

CPU projection first clamps masked tasks to zero and nonnegative values, then scales each batch row by `min(1, capacity / sum)`; post-check tolerance is `1e-7`. Structural shape/mask errors raise instead of falling back.

- [ ] **Step 4: Run safety tests and verify GREEN**

Run: `cd 代码/tests; python -m unittest test_r6_learning_policy_safety.py`

Expected: all tests pass.

### Task 3: Masked Actor–Critic network and deterministic action API

**Files:**
- Create: `代码/src/pi_jwm/r6_learning_policy.py`
- Create: `代码/tests/test_r6_learning_policy.py`

- [ ] **Step 1: Write failing policy tests**

```python
def test_masked_policy_assigns_zero_probability_to_illegal_actions(self):
    policy = MaskedActorCritic(explicit_dim=3, latent_dim=4,
                               hidden_dim=16, spec=spec)
    output = policy(state)
    self.assertTrue(torch.equal(output.offload_prob[state.offload_mask == 0],
                                torch.zeros_like(output.offload_prob[state.offload_mask == 0])))

def test_act_is_reproducible_and_returns_safe_action(self):
    left = policy.act(state, deterministic=False, seed=20260808)
    right = policy.act(state, deterministic=False, seed=20260808)
    self.assertTrue(torch.equal(left.action.offload_index, right.action.offload_index))
    self.assertLessEqual(left.action.cpu_allocation.sum().item(), 10.0 + 1e-7)
```

- [ ] **Step 2: Run and verify RED**

Run: `cd 代码/tests; python -m unittest test_r6_learning_policy.py`

Expected: import failure because `MaskedActorCritic` is absent.

- [ ] **Step 3: Implement masked network**

```python
class MaskedActorCritic(nn.Module):
    def __init__(self, explicit_dim, latent_dim, hidden_dim, spec):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(explicit_dim + latent_dim, hidden_dim), nn.Tanh(),
        )
        self.offload_head = nn.Linear(hidden_dim, spec.offload_count)
        self.rb_head = nn.Linear(hidden_dim, spec.rb_count)
        self.cpu_head = nn.Linear(hidden_dim, spec.cpu_task_count)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        hidden = self.encoder(torch.cat((state.explicit, state.latent), dim=-1))
        offload_logits = masked_logits(self.offload_head(hidden), state.offload_mask,
                                       self.spec.offload_noop_index)
        rb_logits = masked_logits(self.rb_head(hidden), state.rb_mask,
                                  self.spec.rb_noop_index)
        cpu = F.softplus(self.cpu_head(hidden)) * state.cpu_task_mask
        return build_policy_output(offload_logits, rb_logits, cpu,
                                   self.value_head(hidden))
```

`masked_logits` inserts the no-op mask only when a row has no legal action. `act` uses a local `torch.Generator`, calls `SafetyProjector`, and returns both proposed and executable actions. `evaluate` computes log-prob, entropy, and value using the same masks.

- [ ] **Step 4: Run policy tests and verify GREEN**

Run: `cd 代码/tests; python -m unittest test_r6_learning_policy.py`

Expected: all policy tests pass.

### Task 4: Actor–Critic and PPO CPU update objectives

**Files:**
- Create: `代码/src/pi_jwm/r6_learning_policy_training.py`
- Create: `代码/tests/test_r6_learning_policy_training.py`

- [ ] **Step 1: Write failing optimization tests**

```python
def test_actor_critic_update_is_finite_and_does_not_touch_world_model(self):
    frozen_before = tuple(p.detach().clone() for p in world_model.parameters())
    report = actor_critic_cpu_step(policy, batch, optimizer)
    self.assertTrue(all(math.isfinite(v) for v in report.numeric_values()))
    self.assertTrue(report.policy_parameter_changed)
    self.assertTrue(all(p.grad is None for p in world_model.parameters()))
    self.assertTrue(all(torch.equal(a, b) for a, b in zip(frozen_before, world_model.parameters())))

def test_ppo_clipped_step_has_finite_ratio_and_loss(self):
    report = ppo_cpu_step(policy, batch, optimizer, clip_epsilon=0.2)
    self.assertTrue(math.isfinite(report.loss))
    self.assertTrue(math.isfinite(report.ratio_min))
    self.assertTrue(math.isfinite(report.ratio_max))
```

- [ ] **Step 2: Run and verify RED**

Run: `cd 代码/tests; python -m unittest test_r6_learning_policy_training.py`

Expected: import failure because training functions are absent.

- [ ] **Step 3: Implement minimal finite update steps**

```python
def actor_critic_loss(evaluation, advantage, returns,
                      value_coef=0.5, entropy_coef=0.01):
    policy_loss = -(evaluation.log_prob * advantage.detach()).mean()
    value_loss = F.mse_loss(evaluation.value, returns.detach())
    entropy = evaluation.entropy.mean()
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    ensure_finite(total, policy_loss, value_loss, entropy)
    return PolicyLoss(total, policy_loss, value_loss, entropy)

def ppo_loss(evaluation, old_log_prob, advantage, returns,
             clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
    ratio = torch.exp(evaluation.log_prob - old_log_prob.detach())
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = -torch.minimum(ratio * advantage.detach(),
                                 clipped * advantage.detach()).mean()
    value_loss = F.mse_loss(evaluation.value, returns.detach())
    entropy = evaluation.entropy.mean()
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    ensure_finite(total, policy_loss, value_loss, entropy, ratio)
    return PpoLoss(total, policy_loss, value_loss, entropy,
                   ratio.min(), ratio.max())
```

The step functions snapshot policy parameters, zero gradients, backpropagate, reject nonfinite gradients, perform one optimizer step, and report whether at least one policy parameter changed. They accept already-detached state; world-model parameters are never passed to the optimizer.

- [ ] **Step 4: Run training tests and verify GREEN**

Run: `cd 代码/tests; python -m unittest test_r6_learning_policy_training.py`

Expected: all training tests pass.

### Task 5: Auditable CPU preflight runner

**Files:**
- Create: `代码/src/pi_jwm/r6_learning_policy_preflight.py`
- Create: `代码/scripts/run_r6_learning_policy_cpu_preflight.py`
- Create: `代码/tests/test_r6_learning_policy_preflight.py`

- [ ] **Step 1: Write failing preflight tests**

```python
def test_preflight_rejects_locked_test_before_loading(self):
    with self.assertRaisesRegex(ValueError, "locked_test"):
        build_preflight_specs(dataset_root, splits=("locked_test",))

def test_bundle_requires_frozen_bindings_and_all_gates(self):
    summary = run_cpu_preflight(fixture, output_dir)
    self.assertTrue(summary["r6_learning_policy_cpu_ready"])
    self.assertFalse(summary["gpu_started"])
    self.assertFalse(summary["world_model_updated"])
    self.assertFalse(summary["locked_test_accessed"])
    self.assertEqual(summary["hard_constraint_violation_count"], 0)
```

- [ ] **Step 2: Run and verify RED**

Run: `cd 代码/tests; python -m unittest test_r6_learning_policy_preflight.py`

Expected: import failure because the preflight module is absent.

- [ ] **Step 3: Implement preflight and CLI**

The preflight must:

```python
bindings = bind_existing_files(
    r5_candidate_freeze="代码/artifacts/formal_training/pi_jwm_r5_module_confirmation_analysis_v1/candidate_freeze.json",
    r6_paired_summary="代码/artifacts/formal_training/pi_jwm_r6_cpu_paired_closed_loop_v1/summary.json",
    r6_paired_manifest="代码/artifacts/formal_training/pi_jwm_r6_cpu_paired_closed_loop_v1/manifest.json",
)
state = build_detached_real_state(nonlocked_tensor_window, frozen_b_checkpoint)
actor_report = actor_critic_cpu_step(
    policy=actor_policy, batch=training_batch, optimizer=actor_optimizer
)
ppo_report = ppo_cpu_step(
    policy=ppo_policy, batch=training_batch, optimizer=ppo_optimizer,
    clip_epsilon=0.2,
)
write_self_verifying_bundle(output_dir, bindings, state_audit,
                            action_audit, actor_report, ppo_report, failures)
```

The formal bundle contains `summary.json`, `bindings.json`, `state_audit.json`, `action_audit.csv`, `training_smoke.json`, `failures.csv`, `README.md`, and `manifest.json`. The output directory must not already exist. Any failed run stays in `failures.csv`; readiness is false when a gate fails.

- [ ] **Step 4: Run unit tests and one real nonlocked CPU preflight**

Run:

```powershell
cd 代码/tests
python -m unittest test_r6_learning_policy_preflight.py
cd ../scripts
python run_r6_learning_policy_cpu_preflight.py
```

Expected: unit tests pass and `代码/artifacts/preflight/pi_jwm_r6_learning_policy_cpu_preflight_v1/summary.json` reports `r6_learning_policy_cpu_ready=true`, `gpu_started=false`, `world_model_updated=false`, and `locked_test_accessed=false`.

### Task 6: Regression, independent verification, and documentation

**Files:**
- Modify: `task_plan.md`
- Modify: `progress.md`
- Modify: `findings.md`
- Modify: `本地计划表.md`
- Modify: `D:/禹尧珅/人工智能知识库/北大科研/PIJWM/PIJWM推进.md`
- Create: `文档/研究进展/2026-08-08-PI-JWM-R6学习策略CPU预检结果.md`

- [ ] **Step 1: Run focused and historical tests**

```powershell
cd 代码/tests
python -m unittest test_r6_learning_policy_contract.py test_r6_learning_policy_safety.py test_r6_learning_policy.py test_r6_learning_policy_training.py test_r6_learning_policy_preflight.py test_r6_cpu_paired_policy.py test_r6_cpu_paired_analysis.py test_run_r6_cpu_paired_closed_loop.py test_formal_airfogsim_runtime_v1.py
cd ../scripts
python -m unittest test_dual_graph_features.py test_v4_ablation_active_rate.py
```

Expected: all focused tests and all 75 historical tests pass.

- [ ] **Step 2: Independently verify the bundle**

Recompute every manifest SHA-256 and size, confirm no R6 runner remains active, and check the summary gates. Do not trust the writer's own validation result.

- [ ] **Step 3: Synchronize authoritative status**

Record only factual readiness:

```text
r6_learning_policy_cpu_ready=true
r6_gpu_strategy_training_ready=false
final_method_frozen=false
```

Document that Actor–Critic/PPO CPU updates prove interface, numerical, gradient-isolation, and safety correctness only; no strategy performance conclusion is allowed.

- [ ] **Step 4: Run final checks**

Run `python -m py_compile` on all new modules/scripts and `git diff --check`. Expected: zero compile or whitespace errors.
