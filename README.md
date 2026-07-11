# PI-JWM

PI-JWM（Physical-Information Joint World Model，物理-信息联合世界模型）面向网联具身智能体协同场景，学习动作条件下的物理网络与信息网络联合状态演化。

AirFogSim 是参考仿真器和数据生成工具，不是项目框架。当前决策方法仍是 **v11 candidate**，不能写成 v12 或最终 v11。

## 当前状态

截至 2026-07-11：

- 双图世界模型能够联合预测节点、链路活动、链路速率和任务状态。
- v10 冻结世界模型证明未来动作是关键条件；真实未来动作结果只作为参考，不是自主策略结果。
- 当前研究瓶颈是自主动作的支持集、RB/CPU 幅值重建和 actual-rollout 候选选择稳定性。
- 60-seed 主数据集必须显式指定 train/val/test seed，代码不再静默套用旧 0-9 seed 协议。
- 完整单元测试入口：`python -m unittest discover -s tests -p "test_*.py"`。

## 安装

推荐 Python 3.10-3.13：

```powershell
cd D:\shen\网络组
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[experiments]"
```

只安装 PI-JWM 核心依赖：

```powershell
python -m pip install -r .\代码\requirements-core.txt
```

AirFogSim 需要独立环境，依赖位于 `代码/reference/AirFogSim/requirements.txt`，不要与 PI-JWM 核心环境混装。

## 目录

```text
网络组/
├─ 代码/
│  ├─ src/pi_jwm/       可复用框架代码
│  ├─ scripts/          训练、评估和诊断入口
│  ├─ tests/            单元与回归测试
│  ├─ reference/        第三方参考代码
│  └─ artifacts/        数据、模型、指标和报告
├─ 文档/
│  ├─ 项目说明/         研究问题与口径
│  ├─ 研究进展/         论文和阶段总结
│  ├─ 组会/             当前与历史组会材料
│  ├─ 文献/             本地论文
│  ├─ 工程治理/         重构设计、计划和结果
│  └─ 归档由各目录内部管理
├─ 数据集构建任务/      仅保留任务说明；WaveFarer 已移入隔离区
├─ 本地计划表.md        当前唯一执行总览
├─ pyproject.toml
└─ AGENTS.md
```

## 验证

```powershell
cd D:\shen\网络组\代码
python -m compileall -q src scripts tests
python -m unittest discover -s tests -p "test_*.py"
python scripts\run_world_model_v6_dual_graph_rollout.py --synthetic-only --device cpu
```

主数据集训练必须显式给出 seed，例如：

```powershell
python scripts\run_world_model_v8_full_training.py `
  --dataset-dir artifacts\experiments\airfogsim_v0\datasets\world_model_dataset_active_heavy_v2_60seed_20260619 `
  --train-seeds "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59" `
  --val-seeds "16 17" `
  --test-seeds "18 19" `
  --device cpu --epochs 1 --max-train-samples 64 --max-val-samples 32 --max-test-samples 32
```

## 结果口径

新评估 JSON 使用 `result_protocol`：

- `deployable`：测试时不使用真实未来标签，且不在测试集拟合或选择配置。
- `true_future_reference`：使用真实未来动作、真实第一步动作、真实活动或真实幅值的参考结果。
- `sample_oracle`：逐样本事后选择的诊断上界。
- `test_best_diagnostic`：根据测试表现事后挑选，不能作为可部署结论。

完整说明见 [实验结果口径](文档/项目说明/实验结果口径.md)。当前任务与下一步见 [本地计划表](本地计划表.md)。

## 隔离备份

2026-07-11 治理使用的可恢复隔离区：

```text
D:\shen\网络组_隔离备份\2026-07-11_PI-JWM治理
```

隔离区不是永久删除区，所有移动内容必须通过 manifest 和恢复脚本管理。
