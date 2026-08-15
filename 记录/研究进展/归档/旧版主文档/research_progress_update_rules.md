# PI-JWM 研究主文档维护规则

适用文件：`research_progress_overview.tex`。本规则用于后续追加实验、补公式、改结论和同步 PPT 之前的主文档整理。

## 1. 文档主线

- 主框架名称统一写为 **PI-JWM: Physical-Information Joint World Model**。
- AirFogSim/SUMO 只能写作参考仿真器、数据来源或数据生成工具，不能写作主框架。
- v5 selector/ranking 只作为决策接口诊断，除非另有明确要求，不作为主方法线。
- 主文档按模型版本演进组织，弱化“第几周”的叙述。

## 2. 每次新增结果必须包含的信息

新增一个版本、候选模型或消融实验时，至少写清：

- 版本名或候选名，例如 `v8 recurrent latent`、`C8a hurdle only`。
- 数据集和 split，例如 `seed0-9 full80`、`active-heavy v1 train 0-15 / val 16-17 / test 18-19`。
- 结果来源文件路径，例如 `代码/artifacts/experiments/.../xxx.csv` 或 `xxx_report.md`。
- 指标表，不能只写文字结论。
- 当前最优项用粗体标出。
- 表头必须带方向：`$\uparrow$` 表示越大越好，`$\downarrow$` 表示越小越好。
- 如果结果是 headroom、oracle、diagnostic、CPU smoke 或小样本 sanity，必须在正文中说明不能当作主模型结果。

## 3. 符号规则

- 每个版本优先使用当时实验报告或代码中实际使用的符号。
- 如果使用后来补充的统一符号，必须说明补充来源，例如“本文统一符号中的 `$G_t^{phy}$` 和 `$G_t^{info}$` 是在 v6 双图整理后补充的论文级写法”。
- 不要把后续版本的概念硬塞回早期版本。比如 v0/v1 可以写 action-conditioned state sequence，不要强行说已经有完整 physical-information dual graph。
- 统一符号可以放在“问题定义与统一符号”中，但版本节要保留历史实现符号，便于追溯。

## 4. 引用规则

- 引用要放在最自然的位置，可以在句中，不必全放在句尾。
- 不要使用 `\cite{a,b,c}`，因为会生成 `[1,2,3]`。
- 多篇文献要分开写：`\cite{a}\cite{b}\cite{c}`，生成 `[1][2][3]`。
- 每个方法模块至少要有对应理论支撑：world model、latent rollout、GNN、attention、STGCN、MoE、multi-task loss、focal loss、hurdle model 等。
- 不要编造参考文献；新增引用前先确认已有 bibitem 或补真实 bibitem。

## 5. 表格规则

- 结果表优先用测试集或正式 same-split 结果。
- 同一表内比较的实验必须来自同一数据集、split 和评价设置；若不是，必须在表注或正文说明。
- 并列最优要全部加粗，不要人为只选一个。
- `--` 表示该指标未报告或不适用，不能参与最优比较。
- 表格过宽时使用：

```tex
\resizebox{\linewidth}{!}{%
\begin{tabular}{...}
...
\end{tabular}}
```

- LaTeX 表格中的下划线必须转义，例如 `world\_model\_v0`；正文路径优先用 `\code{...}`。

## 6. 解释规则

- 解释要服务结果，不要堆概念。
- 每个版本至少回答三件事：做了什么、结果说明什么、不能说明什么。
- 如果后续版本补充了前一版本没有的指标或命名，要明确写“该命名/指标是后续补充，不改变原实验数值”。
- 不要把单一指标改善写成整体胜利。比如 active-rate 改善但 task 或 activity 退化时，要写成 rate-side improvement。
- headroom 或 oracle 结果只能写成诊断上限，不能和端到端模型混为最终性能。

## 7. 完成前检查

每次修改主文档后必须检查：

```powershell
rg -n "\\cite\{[^}]*," 文档\研究进展文档\research_progress_overview.tex
xelatex -interaction=nonstopmode -halt-on-error research_progress_overview.tex
xelatex -interaction=nonstopmode -halt-on-error research_progress_overview.tex
```

如果第一条命令有输出，说明仍有合并引用，需要拆开。XeLaTeX 至少运行两遍，确认 PDF 生成成功。