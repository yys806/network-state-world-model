# PI-JWM

PI-JWM (Physical-Information Joint World Model) studies action-conditioned joint evolution of physical and information networks for connected embodied-intelligence collaboration.

ChatGPT 网页端新对话先读 [`AI_CONTEXT/00_PROJECT_STATE.md`](AI_CONTEXT/00_PROJECT_STATE.md)，再按该目录的研究、架构、数据、模块、实验、决策和问题文件进入真实源码与证据。项目知识导航：[`docs/PROJECT_INDEX.md`](docs/PROJECT_INDEX.md)。当前科研状态：[`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md)。重新学习项目可从 [`docs/LEARNING_PATH.md`](docs/LEARNING_PATH.md) 开始，人与 AI 的职责及解释规则见 [`docs/COLLABORATION_GUIDE.md`](docs/COLLABORATION_GUIDE.md)，重构目标验收见 [`docs/RESTRUCTURE_ACCEPTANCE.md`](docs/RESTRUCTURE_ACCEPTANCE.md)，定位代码可看 [`docs/CODE_INDEX.md`](docs/CODE_INDEX.md)。

常见项目问题可先运行只读检索：`python code/scripts/query_project_knowledge_v1.py --query "你的问题"`。返回结果是导航候选，正式判断仍须打开列出的原始证据。

AirFogSim is a reference simulator and data-generation source. It is not the PI-JWM framework.

## Current Stage

As of 2026-09-10:

- the current gate is P4 formal non-`locked_test` accuracy and generalization acceptance;
- the candidate method is `entity_aligned_dual_graph_rssm_v1`, trained in two stages with a frozen deterministic base and prior-only formal rollout;
- formal seeds `20260831` and `20260830` have each passed independent single-seed acceptance under the same frozen protocol;
- these two results do not establish three-seed stability, close P4, open P6, or authorize a formal performance claim;
- seed `20260832` has not been authorized and must not be started automatically;
- no formal GPU training is currently active, while generated artifacts and synchronization paths remain protected from moves, rewrites, or deletion;
- `locked_test` remains sealed and `formal_performance_claim_ready=false`.

The candidate-action rollout planner is still a CPU mechanism prototype. Do not describe it as an accepted PI-JWM planner or use it to claim that P6 is open.

## Authority

Read these files in order before continuing work:

1. [`AGENTS.md`](AGENTS.md) for permanent repository and evidence rules.
2. [`AI_CONTEXT/00_PROJECT_STATE.md`](AI_CONTEXT/00_PROJECT_STATE.md) for the concise current snapshot used by ChatGPT Web.
3. [`task_plan.md`](task_plan.md) for the current task gate, blocker, and single next action.
4. [`记录/文件树与证据分层_20260826.md`](记录/文件树与证据分层_20260826.md) for file responsibilities and process/final evidence boundaries.
5. [`记录/本地计划表.md`](记录/本地计划表.md) for the single local execution overview.
6. [`记录/PIJWM主文档.md`](记录/PIJWM主文档.md) for fixed theory, method, data, and evaluation definitions.
7. [`记录/8.12之后推进.md`](记录/8.12之后推进.md) for post-2026-08-12 implementation progress and blockers.
8. [`记录/接续记录/新对话接续说明_20260815.md`](记录/接续记录/新对话接续说明_20260815.md) for handoff context only; it cannot override newer evidence.
9. [`docs/README.md`](docs/README.md) and [`记录/README.md`](记录/README.md) for document and record ownership.
10. Machine-readable manifests and reports under `code/artifacts/` for individual acceptance claims.

Historical plans, meeting slides, model runs, and successful smoke tests remain evidence of their stated scope only. They do not override later theory boundaries or prove a complete method.

## Repository Layout

```text
PIJWM/
|-- code/
|   |-- src/pi_jwm/       reusable PI-JWM framework modules
|   |-- scripts/          runnable collection, audit, training, and evaluation entry points
|   |-- tests/            unit, contract, and regression tests
|   |-- reference/        local third-party checkouts and reference implementations
|   `-- artifacts/        local data, reports, checkpoints, figures, and machine evidence
|-- 记录/                  authority theory, progress, plans, handoffs, and migration records
|-- paper/                formal paper manuscripts and submission assets
|-- literature/           authoritative local literature library and PDF categories
|-- meeting/              local meeting materials and historical presentations
|-- docs/                 templates, project notes, and miscellaneous documentation
|-- AI_CONTEXT/           concise ChatGPT Web current-state and source-navigation layer
|-- pyproject.toml
`-- AGENTS.md
```

Generated artifacts, AirFogSim, local literature, meeting binaries, worktrees, caches, and temporary files stay local by default and are excluded by `.gitignore`.

## Environment

PI-JWM core supports Python 3.10-3.13:

```powershell
cd D:\shen\PKU\PIJWM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[experiments]"
```

Core-only dependencies:

```powershell
python -m pip install -r .\code\requirements-core.txt
```

Keep AirFogSim in its independent environment:

```powershell
conda activate airfogsim
cd D:\shen\PKU\PIJWM\code\reference\AirFogSim\examples
```

Do not copy reusable PI-JWM model code into AirFogSim.

## Safe Verification

From the repository root:

```powershell
python -m compileall -q .\code\src .\code\scripts .\code\tests
$env:PYTHONPATH='D:\shen\PKU\PIJWM\code\src'
python -m unittest discover -s .\code\tests -p 'test_*.py'
git diff --check
```

Reference-simulator verification requires `PYTHONUTF8=1` and the `airfogsim` Conda environment. Use the exact commands bound in the relevant plan or artifact manifest; do not substitute a short launch, interface smoke, or stale result for the documented acceptance gate.

## Research Gates

The canonical roadmap is `P0 -> P1 -> P2 -> P4 -> P6 -> P7+`. The current boundary is:

1. the data, tensor, model/loss/metric, CPU reload, frozen protocol, and independent pre-GPU gates are complete for the current P4 candidate;
2. two formal unlocked seeds have passed independent single-seed acceptance;
3. the third frozen seed and the three-seed audit are still missing, so P4 remains open;
4. P6 candidate-action planning cannot begin until P4 closes;
5. method, data, metrics, and reports must be frozen before any one-time `locked_test` access.

Theory, implementation, runtime configuration, data fields, artifacts, tests, and claims must agree item by item throughout this sequence.
