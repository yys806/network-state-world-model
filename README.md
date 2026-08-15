# PI-JWM

PI-JWM (Physical-Information Joint World Model) studies action-conditioned joint evolution of physical and information networks for connected embodied-intelligence collaboration.

AirFogSim is a reference simulator and data-generation source. It is not the PI-JWM framework.

## Current Stage

As of 2026-08-15:

- the first theory-code-data-evidence consistency audit is complete;
- the v4 information-edge minimum viable schema uses five auditable E1 fields instead of filling thirteen unobserved legacy slots;
- the core action is offloading plus sparse RB COO, while CPU follows `PIJWM-CPU-Inner-Rule-v1` after candidate communication effects;
- P2-B v1 full dual-graph non-training preflight is on `main`;
- P2-B v2 Attempt/Reject Ledger is a verified branch candidate and is not yet merged;
- P2-C v2 is blocked only by the scenario matrix, formal scale, and formal split freezes;
- formal v4 data, new-protocol model training, and a true per-candidate world-model rollout planner are not complete.

The current R6 path is a belief-conditioned direct policy with execution feedback. Do not describe it as a world-model candidate-rollout planner.

## Authority

Read these files in order before continuing work:

1. [`AGENTS.md`](AGENTS.md) for permanent repository and evidence rules.
2. [`新对话接续说明_20260815.md`](新对话接续说明_20260815.md) for the current handoff, with verified migration corrections recorded in later governance updates.
3. [`本地计划表.md`](本地计划表.md) for the single local execution overview.
4. [`文档/README.md`](文档/README.md) for current and historical document ownership.
5. Machine-readable manifests and reports under `代码/artifacts/` for individual acceptance claims.

Historical plans, meeting slides, model runs, and successful smoke tests remain evidence of their stated scope only. They do not override later theory boundaries or prove a complete method.

## Repository Layout

```text
PIJWM/
|-- 代码/
|   |-- src/pi_jwm/       reusable PI-JWM framework modules
|   |-- scripts/          runnable collection, audit, training, and evaluation entry points
|   |-- tests/            unit, contract, and regression tests
|   |-- reference/        local third-party checkouts and reference implementations
|   `-- artifacts/        local data, reports, checkpoints, figures, and machine evidence
|-- 文档/
|   |-- 研究进展/         current research designs, results, and internal archives
|   |-- 组会/             local meeting materials and historical presentations
|   `-- 文献/             local literature files; Zotero is the authoritative library
|-- docs/superpowers/     implementation plans and design specifications
|-- 本地计划表.md          single local overview plan
|-- 新对话接续说明_*.md    dated continuation records
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
python -m pip install -r .\代码\requirements-core.txt
```

Keep AirFogSim in its independent environment:

```powershell
conda activate airfogsim
cd D:\shen\PKU\PIJWM\代码\reference\AirFogSim\examples
```

Do not copy reusable PI-JWM model code into AirFogSim.

## Safe Verification

From the repository root:

```powershell
python -m compileall -q .\代码\src .\代码\scripts .\代码\tests
$env:PYTHONPATH='D:\shen\PKU\PIJWM\代码\src'
python -m unittest discover -s .\代码\tests -p 'test_*.py'
git diff --check
```

Reference-simulator verification requires `PYTHONUTF8=1` and the `airfogsim` Conda environment. Use the exact commands bound in the relevant plan or artifact manifest; do not substitute a short launch, interface smoke, or stale result for the documented acceptance gate.

## Research Gates

Before formal data generation, long GPU training, baseline freezing, or locked-test access:

1. close the current P2 ledger evidence chain;
2. freeze the formal scenario matrix, statistical scale, and seed split;
3. regenerate and verify formal v4 data under the approved contract;
4. audit and implement the target world-model rule/rollout semantics;
5. implement and test true per-candidate world-model rollout planning;
6. freeze method, data, metrics, and reports before one-time locked-test access.

Theory, implementation, runtime configuration, data fields, artifacts, tests, and claims must agree item by item throughout this sequence.
