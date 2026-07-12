"""Build the 2026-05-31 PI-JWM research summary LaTeX/PDF.

The advisor-facing document intentionally contains no cover page and no table of
contents. Concept figures are read from figs/1.png, figs/2.png, and figs/3.png;
experiment figures are generated from measured data.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent

OUT_DIR = WORKSPACE_ROOT / "文档" / "组会" / "5.31"
FIG_DIR = OUT_DIR / "figs"
SUMMARY_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "pi_jwm_v6_eval_full80"
    / "v6_dual_graph_smoke_summary.json"
)

BASE_NAME = "科研实习进组以来工作总结_禹尧珅_20260531"
TEX_PATH = OUT_DIR / f"{BASE_NAME}.tex"
PDF_PATH = OUT_DIR / f"{BASE_NAME}.pdf"
SCRIPT_PATH = OUT_DIR / f"{BASE_NAME}_讲稿.md"
PROMPT_PATH = OUT_DIR / f"{BASE_NAME}_AI绘图提示词.md"


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def get_runs(summary: dict) -> dict:
    return summary["real_data_sanity"]["runs"]


def metric_values(summary: dict) -> dict[str, list[float]]:
    runs = get_runs(summary)
    modes = ["dual", "physical_only", "information_only"]
    return {
        "active-rate RMSE": [
            runs[mode]["test_eval"]["active_rate"]["active_rmse"] for mode in modes
        ],
        "link-rate RMSE": [
            runs[mode]["test_eval"]["link_rate"]["rmse"] for mode in modes
        ],
        "node RMSE": [runs[mode]["test_eval"]["node"]["rmse"] for mode in modes],
        "task RMSE": [runs[mode]["test_eval"]["task"]["rmse"] for mode in modes],
    }


def make_metrics(summary: dict, path: Path) -> None:
    configure_matplotlib()
    labels = ["dual", "physical", "information"]
    colors = ["#2563EB", "#0F766E", "#F97316"]
    metrics = metric_values(summary)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.6), dpi=180)
    fig.patch.set_facecolor("white")
    for ax, (metric, values) in zip(axes.ravel(), metrics.items()):
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(metric, fontsize=14, fontweight="bold", color="#0F172A")
        ax.grid(axis="y", alpha=0.24)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("越低越好", fontsize=10, color="#475569")
        best = min(values)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#0F172A",
            )
            if math.isclose(value, best):
                bar.set_edgecolor("#0F172A")
                bar.set_linewidth(1.8)
    fig.suptitle(
        "PI-JWM v6 full80 测试集指标对比",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.02,
        "三种模式 activity F1 均为 1.0；差异主要来自速率幅值、节点状态和任务状态误差。",
        ha="center",
        fontsize=11,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_training_curves(summary: dict, path: Path) -> None:
    configure_matplotlib()
    runs = get_runs(summary)
    modes = ["dual", "physical_only", "information_only"]
    colors = {
        "dual": "#2563EB",
        "physical_only": "#0F766E",
        "information_only": "#F97316",
    }
    labels = {
        "dual": "dual",
        "physical_only": "physical",
        "information_only": "information",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    for mode in modes:
        history = runs[mode]["history"]
        epochs = [row["epoch"] for row in history]
        val_total = [row["val"]["total"] for row in history]
        val_rate = [row["val"]["rate"] for row in history]
        axes[0].plot(
            epochs,
            val_total,
            color=colors[mode],
            label=labels[mode],
            linewidth=2.2,
        )
        axes[1].plot(
            epochs,
            val_rate,
            color=colors[mode],
            label=labels[mode],
            linewidth=2.2,
        )
    for ax, title in zip(axes, ["验证 total loss", "验证 rate loss"]):
        ax.set_title(title, fontsize=14, fontweight="bold", color="#0F172A")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle("v6 full80 训练过程诊断", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_metric_table(summary: dict, path: Path) -> None:
    configure_matplotlib()
    runs = get_runs(summary)
    modes = ["dual", "physical_only", "information_only"]
    rows = []
    for mode in modes:
        test = runs[mode]["test_eval"]
        rows.append(
            [
                mode,
                str(runs[mode]["best_epoch"]),
                f"{runs[mode]['activity_threshold']:.2f}",
                f"{test['activity']['f1']:.6f}",
                f"{test['active_rate']['active_rmse']:.6f}",
                f"{test['link_rate']['rmse']:.6f}",
                f"{test['node']['rmse']:.6f}",
                f"{test['task']['rmse']:.6f}",
            ]
        )
    headers = [
        "模式",
        "epoch",
        "阈值",
        "F1",
        "active-rate",
        "link-rate",
        "node",
        "task",
    ]

    fig, ax = plt.subplots(figsize=(13.4, 3.4), dpi=180)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(
        "v6 full80 测试集核心数值",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
        pad=18,
    )
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.18, 0.09, 0.08, 0.09, 0.17, 0.15, 0.12, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.9)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row == 0:
            cell.set_facecolor("#DBEAFE")
            cell.set_text_props(weight="bold", color="#0F172A")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F8FAFC")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_figures(summary: dict) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "concept_problem": FIG_DIR / "1.png",
        "concept_pipeline": FIG_DIR / "2.png",
        "concept_architecture": FIG_DIR / "3.png",
        "metrics": FIG_DIR / "metrics.png",
        "curves": FIG_DIR / "curves.png",
        "table": FIG_DIR / "table.png",
    }
    make_metrics(summary, figures["metrics"])
    make_training_curves(summary, figures["curves"])
    make_metric_table(summary, figures["table"])
    return figures


def tex_path(path: Path) -> str:
    return str(path.relative_to(OUT_DIR)).replace("\\", "/")


def figure_placeholder(title: str, note: str) -> str:
    return rf"""
\begin{{figure}}[H]
\centering
\fbox{{%
\begin{{minipage}}[c][4.8cm][c]{{0.92\textwidth}}
\centering
{{\Large\bfseries {title}}}\\[0.55cm]
{{\color{{muted}} {note}}}\\[0.45cm]
{{\small 图位预留，后续替换为概念示意图。}}
\end{{minipage}}}}
\caption{{{title}}}
\end{{figure}}
"""


def figure_image(path: Path, caption: str, width: str = "0.98\\textwidth") -> str:
    if not path.exists():
        return figure_placeholder(caption, f"缺少图片文件：{tex_path(path)}")
    return rf"""
\begin{{figure}}[H]
\centering
\includegraphics[width={width}]{{{tex_path(path)}}}
\caption{{{caption}}}
\end{{figure}}
"""


def write_latex(figures: dict[str, Path]) -> None:
    tex = rf"""
\documentclass[UTF8,12pt,a4paper]{{ctexart}}
\usepackage{{geometry}}
\geometry{{left=2.2cm,right=2.2cm,top=2.2cm,bottom=2.2cm}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{enumitem}}
\usepackage{{caption}}
\usepackage{{float}}
\usepackage{{titlesec}}
\usepackage{{fancyhdr}}
\usepackage{{tcolorbox}}
\tcbuselibrary{{skins,breakable}}

\definecolor{{mainblue}}{{HTML}}{{1D4ED8}}
\definecolor{{deep}}{{HTML}}{{0F172A}}
\definecolor{{muted}}{{HTML}}{{475569}}
\definecolor{{softgray}}{{HTML}}{{F8FAFC}}

\hypersetup{{colorlinks=true,linkcolor=mainblue,urlcolor=mainblue}}
\setlength{{\parindent}}{{2em}}
\setlength{{\parskip}}{{0.28em}}
\setlength{{\headheight}}{{15pt}}
\setlength{{\tabcolsep}}{{4pt}}
\emergencystretch=2em
\linespread{{1.18}}
\pagestyle{{fancy}}
\fancyhf{{}}
\lhead{{PI-JWM 科研实习进展总结}}
\rhead{{2026-05-31}}
\cfoot{{\thepage}}

\titleformat{{\section}}{{\Large\bfseries\color{{deep}}}}{{\thesection}}{{0.8em}}{{}}
\titleformat{{\subsection}}{{\large\bfseries\color{{mainblue}}}}{{\thesubsection}}{{0.8em}}{{}}
\captionsetup{{font=small,labelfont=bf}}
\setlist[itemize]{{leftmargin=2.1em,itemsep=0.18em,topsep=0.2em}}
\setlist[enumerate]{{leftmargin=2.1em,itemsep=0.18em,topsep=0.2em}}
\newcolumntype{{Y}}{{>{{\raggedright\arraybackslash}}X}}

\newtcolorbox{{summarybox}}{{
  enhanced, breakable, colback=softgray, colframe=mainblue,
  arc=2mm, boxrule=0.8pt, left=2mm, right=2mm, top=1.2mm, bottom=1.2mm
}}
\newcommand{{\metric}}[1]{{\textcolor{{mainblue}}{{\textbf{{#1}}}}}}

\begin{{document}}

\begin{{center}}
{{\LARGE\bfseries 科研实习进组以来工作总结：PI-JWM 物理-信息联合世界模型}}\\[0.45em]
{{\large 禹尧珅 \quad 2026 年 5 月 31 日}}
\end{{center}}
\vspace{{0.4em}}

\section{{研究主线与阶段工作}}

本阶段研究围绕 PI-JWM（Physical-Information Joint World Model，物理-信息联合世界模型）展开。核心目标是在联网具身智能体协同场景中学习一个动作条件的状态转移模型，使模型能够根据历史物理状态、信息状态和调度动作预测未来节点、链路、速率、任务和资源变化。PI-JWM 的定位是可训练、可评估、可滚动的近似环境模型，用于后续候选动作评估和在线协同决策。

从进组以来的工作看，研究路径可以分成四个阶段。第一，明确 formulation：把低空/车联网系统中的移动、通信、计算和任务处理统一为动态系统建模问题。第二，打通数据链路：从参考仿真器和交通仿真日志中构建当前状态、动作、下一步状态样本。第三，建立基线和诊断链路：从 state-only、state-action、edge-level、two-stage 模型逐层定位瓶颈。第四，推进 PI-JWM v6：将物理图和信息图分开编码，再进行联合 rollout 和多目标预测。

\begin{{summarybox}}
\textbf{{当前阶段结论：}}PI-JWM v6 已完成真实数据训练、测试和三种图模式消融。三种模式的 link activity F1 均达到 1.0；dual 模式在 active-rate RMSE 与 link-rate RMSE 上最优，说明物理图与信息图联合后，对活跃链路速率幅值和整体链路速率预测更有帮助。information\_only 在任务状态预测上更有优势，physical\_only 在节点状态预测上更有优势，这说明双图结构具有可解释的分工关系。
\end{{summarybox}}

\section{{问题 Formulation}}

\subsection{{研究对象}}

研究对象是基站覆盖下的低空/车联网协同系统。系统中包含无人机、车辆、路侧边缘节点和边缘服务器，节点需要共同完成感知、通信、计算和任务处理。与静态网络优化不同，该场景中的物理状态和信息状态持续耦合：节点位置变化会改变可通信拓扑；任务到达会改变队列压力；信道衰减会改变链路质量；资源分配、任务卸载和回传动作会反过来改变未来任务状态与资源占用。

因此，PI-JWM 的核心问题是学习一个近似状态转移模型：
\[
  \hat{{s}}_{{t+1:t+H}}=f_\theta(s_{{t-k:t}}, a_{{t-k:t}}),
\]
其中 \(s\) 包含物理状态和信息状态，\(a\) 包含卸载、资源分配、回传和边级调度动作，输出覆盖未来节点、链路、速率和任务状态。这个 formulation 优先回答“系统在某个动作条件下会如何演化”，再为后续动作选择提供预测基础。

{figure_image(figures["concept_problem"], "问题 Formulation：当前物理/信息状态、调度动作与未来系统状态预测")}

图 1 对应 PI-JWM 的基本建模对象。左侧表示当前系统状态，其中上半部分是物理网络，包括车辆、无人机、路侧边缘节点、基站和边缘服务器之间的覆盖、距离和连接关系；下半部分是信息网络，包括任务队列、通信链路状态和资源占用。中间表示动作条件，包含任务卸载、RB 分配、CPU 分配和回传动作。右侧表示动作执行后需要预测的未来状态，包括节点位置变化、链路质量变化、队列变化和资源变化。这个图强调 PI-JWM 的输入不是单一网络拓扑，而是物理状态、信息状态和调度动作的组合；输出也不是单个指标，而是一组未来系统状态。

\subsection{{输入、动作与输出}}

\begin{{longtable}}{{p{{2.8cm}}p{{11.4cm}}}}
\toprule
\textbf{{类别}} & \textbf{{内容}}\\
\midrule
物理状态 & 节点位置、速度、高度差、相对距离、覆盖关系、邻近关系、移动历史。\\
信息状态 & 链路活动、链路速率、任务队列、RB/CPU 分配、任务卸载与回传历史。\\
动作变量 & 卸载动作、RB 分配、CPU 分配、任务回传、边级通信动作，后续可接入轨迹控制动作。\\
预测目标 & 节点状态、链路是否活跃、活跃链路速率、整体链路速率、任务状态和资源变化。\\
评价目标 & 状态预测误差、链路活动 F1、活跃链路速率 RMSE、任务状态 RMSE、跨 seed 稳定性与扰动鲁棒性。\\
\bottomrule
\end{{longtable}}

\section{{方法与实现路径}}

\subsection{{数据链路}}

数据来源被固定为“参考仿真器生成日志，PI-JWM 使用日志训练和评估”。围绕这条链路，本阶段完成了三类工作。第一，梳理单步推进顺序，包括交通/移动更新、任务产生、信道衰减、动作注入、无线链路速率计算、资源执行、队列变化和能耗更新。第二，从日志中导出监督学习样本，使每个样本包含当前状态、动作和下一步真实状态。第三，构建 multi-seed、strict action 和 edge action 接口，用于检查模型是否只记住某一次随机轨迹，并让链路级预测能够接触到与候选边相关的动作信息。

具体来说，动态日志中包含四类关键变化：任务到达、信道变化、资源执行事件和系统拓扑/状态轨迹。任务到达决定队列压力，信道变化决定链路速率的随机波动，资源执行决定任务完成进度，拓扑轨迹决定物理连接关系。PI-JWM 的 dataset 构建就是把这些连续演化日志切成监督学习样本，并保留动作与下一步状态之间的对应关系。

\subsection{{模型链路}}

模型演进采用逐层诊断方式，先用简单模型定位瓶颈，再逐步加入结构化建模：

\begin{{itemize}}
  \item \textbf{{state-only baseline：}}只输入历史状态，验证状态自身是否包含可预测结构。
  \item \textbf{{state-action baseline：}}加入调度动作，判断动作是否能解释未来状态差异。
  \item \textbf{{edge-level link model：}}把候选链路单独作为样本，定位链路预测瓶颈。
  \item \textbf{{two-stage link model：}}先判断链路是否活跃，再预测活跃链路 rate，缓解大量零速率样本的影响。
  \item \textbf{{world model v0--v3：}}统一状态/动作/输出接口，加入 latent rollout 与通信图消息传递。
  \item \textbf{{PI-JWM v6：}}将物理图和信息图分开编码，并用动作历史条件化未来状态预测。
\end{{itemize}}

{figure_image(figures["concept_pipeline"], "方法路线：从动态日志到 PI-JWM v6 双图世界模型")}

图 2 展示了完整实现链路。第一步是动态日志与轨迹，包括任务产生、信道变化、资源执行和系统拓扑演化。第二步是样本构建，将日志统一整理为 dataset，并进一步构建 multi-seed、strict action 和 edge action。第三步是基线诊断，使用 state-only、state-action、edge-level 和 two-stage 模型分别检查状态预测、动作条件、边级链路预测和两阶段链路建模的效果。第四步是 PI-JWM v6，将物理图、信息图、融合层和 rollout prediction 接起来，形成当前主线模型。这个流程的作用是避免直接把复杂模型当成黑盒训练，而是先用基线模型明确瓶颈，再用双图结构解决瓶颈。

\section{{PI-JWM 方法设计}}

\subsection{{双图表示}}

PI-JWM 的核心想法是把系统拆成两类图。物理图描述空间和运动关系，例如位置、距离、覆盖和相对速度；信息图描述通信和任务关系，例如链路活动、速率、队列、资源占用和调度动作。两张图表达的是同一个系统的不同侧面，因此需要先分开编码，再在融合层联合预测。

这种设计的意义在于：节点移动、距离变化和覆盖变化主要决定“哪些节点可能通信”；队列、资源、动作历史和链路活动主要决定“通信和任务实际如何执行”。如果只使用单一图结构，模型容易把几何关系、通信关系和任务关系混在同一个表示空间里；双图结构可以把关系类型拆开，使 node、link、rate、task 等不同目标更容易获得对应的信息来源。

{figure_image(figures["concept_architecture"], "PI-JWM v6：物理图、信息图、动作条件与多目标 rollout 结构")}

图 3 是 PI-JWM 的模型结构。左上角的 Physical Graph 用于表达 UAV、Vehicle、Edge Node 等物理实体之间的距离、覆盖和相对运动关系；左下角的 Information Graph 用于表达任务、通信链路、队列、资源和动作历史。两张图分别经过 Physical Encoder 和 Information Encoder 编码后进入 Fusion \& Rollout 模块。该模块维护 latent world state \(z_t\)，并预测 \(z_{{t+1:t+H}}\) 的未来演化。右侧四个输出头分别对应 node state、link activity、link rate 和 task state。底部的 Action Condition 表示模型的预测依赖调度动作历史，因此 PI-JWM 是动作条件世界模型，而不是只根据状态外推的普通预测器。

\subsection{{v0 到 v6 的阶段演进}}

\begin{{small}}
\begin{{tabular}}{{p{{1.1cm}}p{{6.0cm}}p{{6.4cm}}}}
\toprule
\textbf{{版本}} & \textbf{{主要变化}} & \textbf{{阶段意义}}\\
\midrule
v0 & 建立状态转移模型基本输入输出接口。 & 从单点预测转向统一状态转移建模。\\
v1 & 分阶段训练不同输出头。 & 降低多目标训练互相干扰。\\
v2 & 引入隐空间多步滚动预测。 & 开始模拟多步未来状态。\\
v3 & 加入图结构滚动预测和活跃链路速率诊断。 & 将链路结构和速率幅值分开分析。\\
v4 & 引入链路活动校准与双图消融雏形。 & 明确链路活动和速率幅值是不同瓶颈。\\
v5 & 候选动作排序诊断接口。 & 用于候选动作评估，当前主模型仍为状态预测。\\
v6 & 物理图、信息图、动作历史联合建模。 & 当前主线版本，验证双图对状态预测的贡献。\\
\bottomrule
\end{{tabular}}
\end{{small}}

\section{{克服的主要困难}}

\begin{{enumerate}}
  \item \textbf{{问题范围过宽。}}初始方向包含基座模型、world model、资源调度、轨迹控制和网络预测。当前处理方式是先收敛到 PI-JWM：把“预测系统未来状态”作为阶段目标。
  \item \textbf{{仿真机制不透明。}}日志数据来自任务到达、信道衰减、动作执行和资源更新的组合。通过代码和日志级别梳理，将仿真器固定为数据来源，将方法主体放在 PI-JWM。
  \item \textbf{{动作与链路难对齐。}}全局调度动作无法直接解释某条候选边的未来变化，因此构建 strict action 和 edge action，把动作映射到边级样本。
  \item \textbf{{链路标签稀疏。}}多数候选边处于非活跃状态，直接预测速率会受到大量零值影响。因此将任务拆成 activity prediction 和 active-rate regression。
  \item \textbf{{指标容易被总 loss 掩盖。}}rate loss 数值尺度较大，容易主导 total loss。当前评估拆成 activity F1、active-rate RMSE、link-rate RMSE、node RMSE 和 task RMSE。
\end{{enumerate}}

\section{{阶段创新点}}

\begin{{summarybox}}
\textbf{{一句话概括：}}PI-JWM 的创新在于把联网具身智能体系统建模为物理图与信息图的联合动态系统，并显式引入动作条件，使模型能够预测未来状态，而不仅是拟合静态链路或任务指标。
\end{{summarybox}}

\begin{{itemize}}
  \item \textbf{{问题层面：}}将研究对象定义为物理网络和信息网络的联合演化，覆盖节点、链路、任务和资源。
  \item \textbf{{表示层面：}}物理图与信息图分离编码，保留不同关系类型的结构差异。
  \item \textbf{{动作层面：}}从日志中显式抽取调度动作，并构造成边级动作输入。
  \item \textbf{{训练层面：}}将 link activity 与 active-rate 拆分，降低零速率样本对速率回归的干扰。
  \item \textbf{{评估层面：}}用多指标解释不同模块贡献，避免总 loss 掩盖具体瓶颈。
  \item \textbf{{应用层面：}}未来可以作为候选卸载、资源分配和轨迹动作的快速近似评估器。
\end{{itemize}}

当前工作形成的价值主要体现在三个方面。首先，数据链路从仿真日志走到了可复用的 world-model dataset，使后续模型训练不再依赖临时手工处理。其次，指标链路从单一 loss 扩展到 activity、active-rate、link-rate、node 和 task 多目标指标，可以更清楚地定位模型瓶颈。最后，模型链路从普通预测模型推进到双图 action-conditioned rollout，为后续“用模型评估动作”提供了结构基础。

\section{{当前实验结果}}

最新实验为 v6 full80 GPU 版本，数据划分为 train=1520、val=190、test=190，三种模式均训练 80 epoch。三种模式定义如下：

\begin{{itemize}}
  \item \textbf{{dual：}}同时使用物理图和信息图。
  \item \textbf{{physical\_only：}}只使用位置、距离、覆盖和移动关系。
  \item \textbf{{information\_only：}}只使用通信、任务、资源和动作历史关系。
\end{{itemize}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{{tex_path(figures["metrics"])}}}
\caption{{v6 full80 测试集指标对比。除 activity F1 外，RMSE 均为越低越好。}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{{tex_path(figures["table"])}}}
\caption{{v6 full80 测试集原始数值。}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.98\textwidth]{{{tex_path(figures["curves"])}}}
\caption{{训练过程诊断：total loss 与 rate loss 的验证曲线。}}
\end{{figure}}

\subsection{{结果解释}}

三种模式的 activity F1 都达到 1.0，说明链路是否活跃这个二分类问题已经可以稳定学习。真正区分模型能力的是活跃链路速率和整体链路速率。dual 的 active-rate RMSE 为 \metric{{228.318}}，link-rate RMSE 为 \metric{{6.416}}，两项都是三种模式中最低，说明双图联合对速率幅值预测有价值。

information\_only 的 task RMSE 为 \metric{{3.381}}，任务指标最优，说明任务演化更依赖通信、队列、资源和动作历史。physical\_only 的 node RMSE 为 \metric{{37.067}}，节点指标最优，但链路和任务指标相对较弱，说明几何关系对节点状态有帮助，通信任务演化仍需要信息图。

从消融结果看，双图并不是在所有指标上都简单压过单图，而是在链路速率相关指标上体现了最明显收益。这一现象符合建模直觉：速率既受到物理距离、覆盖和相对运动影响，也受到通信链路状态、资源分配和动作历史影响，因此需要 physical graph 与 information graph 联合解释。任务状态更偏向信息图，因为任务队列、资源占用和动作历史直接决定任务进度；节点状态更偏向物理图，因为位置、速度和覆盖关系主要由物理演化决定。这个结果说明双图结构不仅提高了部分指标，也帮助解释不同状态变量依赖的主要信息来源。

训练曲线显示，三种模式的验证损失均能下降，说明当前数据链路和训练流程是可用的；但 rate loss 的数值仍然明显大于其他分量，说明下一阶段主要瓶颈仍在速率幅值预测。后续如果只继续优化 activity F1，提升空间有限；更有效的方向是围绕 active-rate regression、链路速率不确定性、跨 seed 稳定性和动作条件建模继续推进。

\section{{后续计划}}

\begin{{enumerate}}
  \item \textbf{{补普通模型与 world model 对比。}}将 MLP、Ridge、state-action predictor、non-rollout predictor 与 PI-JWM v6 放到同一指标体系下，回答世界模型相较普通模型的收益。
  \item \textbf{{改进 active-rate 建模。}}当前 activity 已稳定，主要误差来自活跃链路速率幅值。后续尝试 active-only regression、加权损失、两阶段 head、分布式输出和不确定性估计。
  \item \textbf{{增强双图融合。}}进一步尝试跨图注意力、门控融合或目标自适应融合，使物理图和信息图在 node、link、rate、task 上发挥不同作用。
  \item \textbf{{补稳健性和泛化实验。}}加入跨 seed、输入扰动、阈值迁移、置信区间和 per-seed 曲线，验证模型在多次随机过程下的稳定性。
  \item \textbf{{接回决策接口。}}在状态 rollout 更稳定后，用 PI-JWM 评估候选卸载、资源分配和轨迹动作，v5 selector/ranking 作为诊断接口接回。
\end{{enumerate}}

\end{{document}}
"""
    TEX_PATH.write_text(tex.strip() + "\n", encoding="utf-8")


def write_script() -> None:
    if SCRIPT_PATH.exists():
        return
    content = """# 科研实习进组以来工作总结讲稿

时间：2026 年 5 月 31 日线上讨论<br>
汇报人：禹尧珅<br>
主线：PI-JWM，Physical-Information Joint World Model

## 0. 开场

各位老师、同学好。我这次汇报围绕进组以来的研究进展展开，主要按四个问题讲：第一，问题如何 formulation；第二，当前用了什么方法；第三，过程中克服了什么困难；第四，目前有什么阶段性创新。后面我再补充当前实验结果和下一步计划。

我目前的研究主线已经收敛为 PI-JWM，也就是 Physical-Information Joint World Model，中文可以叫物理-信息联合世界模型。它的目标是学习一个动作条件下的状态转移模型：给定当前系统状态和调度动作，预测未来节点、链路、速率、任务和资源状态。

这里先说明一下研究边界：参考仿真器和交通仿真主要承担数据来源的角色，用来产生可分析、可训练的动态日志；真正的方法主线是 PI-JWM。也就是说，我现在关注的不是单纯跑仿真，而是从仿真产生的动态数据里学习系统演化规律。

如果用一句话概括本阶段工作，就是：我先把动态网络系统的状态、动作和下一步状态整理成可训练数据，再从普通预测模型逐步推进到物理图和信息图联合建模，最后用 v6 实验验证双图结构在链路速率预测上的作用。

## 1. 问题 Formulation

我理解的研究对象是基站覆盖下的联网具身智能体协同系统。系统里有无人机、车辆、路侧边缘节点、基站和边缘服务器，它们同时涉及移动、通信、计算和任务处理。

这个问题的难点在于状态是动态耦合的。节点位置变化会改变通信距离和覆盖关系；任务到达会改变队列压力；信道变化会影响链路速率；任务卸载、RB 分配、CPU 分配和回传动作又会反过来改变未来任务进度和资源占用。

所以我把问题 formulation 成一个动作条件状态转移问题：

`s_t, a_t -> s_{t+1:t+H}`

这里 `s_t` 是当前状态，包括物理状态和信息状态；`a_t` 是调度动作；输出是未来一段时间的节点状态、链路活动、链路速率和任务状态。

图 1 就是在解释这个 formulation。左侧 Current System State 表示当前状态，上半部分是 physical network，包括车辆、无人机、路侧边缘节点、基站和边缘服务器之间的距离、覆盖和连接关系；下半部分是 information network，包括任务队列、通信链路状态和资源指标。中间 Scheduling Actions 表示动作条件，包括任务卸载、RB 分配、CPU 分配和回传动作。右侧 Future System State Prediction 表示未来状态预测，也就是执行动作以后系统状态会怎样变化。

讲这张图时，重点不是逐个解释所有小图标，而是强调三件事。第一，PI-JWM 的输入是物理状态、信息状态和动作的组合。第二，PI-JWM 的输出是未来系统状态，不只是一个链路速率或一个任务指标。第三，这个模型后续可以作为候选动作评估的近似环境。

## 2. 用了什么方法

这一部分不能只讲图，要讲清楚我实际做了哪几层方法。总体上，我的方法路线分成三层：数据构建、基线诊断、PI-JWM 双图世界模型。

第一层是数据构建。我从动态日志里抽取当前状态、动作和下一步状态，形成 world model dataset。动态日志里主要包含四类变化：任务产生、信道变化、资源执行和系统轨迹。任务产生决定队列压力；信道变化决定链路质量；资源执行决定任务完成进度；系统轨迹决定物理连接关系。把这些日志切成样本后，每个样本都要尽量保留“当前状态—动作—下一步状态”的因果顺序。

在数据构建里，我做了几个关键接口。`dataset` 是统一样本集合；`multi-seed` 是多随机种子数据，用来检查模型是否只记住某一次随机轨迹；`strict action` 是严格动作接口，用来保证动作字段和状态变化对应；`edge action` 是边级动作映射，用来让链路预测看到与具体候选边相关的动作信息。

第二层是基线诊断。我不是直接训练一个复杂模型，而是先用普通模型逐层定位问题。`state-only` 只输入历史状态，用来判断状态自身有没有可预测结构。`state-action` 加入动作，用来判断动作是否能解释未来状态差异。`edge-level` 把候选链路单独作为样本，用来定位链路预测瓶颈。`two-stage` 把链路预测拆成两步：先判断链路是否活跃，再预测活跃链路速率。这样可以减少大量零速率样本对速率回归的干扰。

第三层是 PI-JWM v6。v6 的核心是把系统拆成 physical graph 和 information graph。physical graph 负责编码位置、距离、覆盖和相对运动；information graph 负责编码任务、通信链路、队列、资源和动作历史。两张图分别编码后进入融合与 rollout 模块，再输出 node state、link activity、link rate 和 task state。

图 2 对应的是完整方法路线。左边是动态日志，中间是样本构建和基线诊断，右边是 PI-JWM v6。讲这张图的时候，我会说：这条路线的重点是先把数据链路打通，再用基线模型定位瓶颈，最后把有明确作用的双图结构放进世界模型。

图 3 对应的是 PI-JWM 的模型结构。左上 Physical Graph 处理物理实体和物理关系；左下 Information Graph 处理任务、链路、队列、资源和动作历史；中间 Fusion & Rollout 学习 latent world state，并预测未来多步状态；右侧四个 head 分别预测节点状态、链路活动、链路速率和任务状态。底部 Action Condition 表示模型预测受到动作条件影响，这也是它和普通状态外推模型的区别。

这一部分最后可以总结一句：我的方法不是只做一个单点预测器，而是从日志数据、动作接口、基线诊断一路推进到动作条件双图 rollout 模型。

## 3. 克服了什么困难

第一个困难是问题范围太宽。最开始这个方向容易同时包含基座模型、world model、资源调度、轨迹控制和网络预测。如果全都放在一起，很难形成清晰的阶段目标。所以我把本阶段目标收敛到 PI-JWM：先做好物理-信息联合状态预测，再考虑把预测模型接到动作评估里。

第二个困难是数据机制复杂。日志里的状态变化不是单一因素造成的，而是任务产生、信道衰减、移动变化、动作执行和资源更新共同作用的结果。我的处理方式是先梳理单步推进逻辑，再把仿真器固定为数据来源，把方法主体放在 PI-JWM 的数据构建和模型训练上。

第三个困难是动作和链路难对齐。全局调度动作不一定能直接解释某条候选链路的变化。比如一个动作可能影响任务卸载，也可能影响资源分配，还可能间接影响某条链路是否活跃。因此我构建了 strict action 和 edge action，把动作尽可能映射到边级样本，让链路级预测可以接触到动作条件。

第四个困难是链路标签稀疏。很多候选链路在某些时刻是不活跃的，如果直接回归 link rate，大量零值会影响模型学习。我的处理方式是把问题拆成 link activity 和 active-rate：先判断链路是否活跃，再在活跃链路上预测速率幅值。

第五个困难是指标容易被 total loss 掩盖。rate loss 的数值尺度比较大，如果只看总损失，很难判断模型到底在哪个目标上提升。所以我把评估拆成 activity F1、active-rate RMSE、link-rate RMSE、node RMSE 和 task RMSE。这样可以分别看链路开关、速率幅值、节点状态和任务状态。

第六个困难是主线容易被工具带偏。参考仿真器很重要，但它是数据生成工具。本阶段我重新把主线拉回 PI-JWM：也就是物理图、信息图、动作条件和状态 rollout。

## 4. 有什么创新

第一个创新是问题 formulation。当前工作把研究对象定义为物理网络和信息网络的联合演化，而不是只预测某一个静态链路指标。这个 formulation 能同时覆盖节点、链路、任务和资源。

第二个创新是双图表示。physical graph 和 information graph 分开编码，可以保留几何关系、通信关系、任务关系之间的差异。实验结果也说明，不同预测目标依赖的信息来源不同：node 更依赖物理图，task 更依赖信息图，rate 同时依赖两者。

第三个创新是动作条件建模。我不是只输入状态，而是把调度动作、动作历史和边级动作也作为条件，让模型学习“同一状态下，不同动作会导致不同未来状态”的变化。

第四个创新是两阶段链路建模思路。把 link activity 和 active-rate 拆开，可以缓解零速率样本过多的问题，也能让评估更清楚：activity 负责链路开关判断，active-rate 负责活跃链路速率幅值。

第五个创新是评估体系。当前不只看一个 total loss，而是把指标拆成 activity F1、active-rate RMSE、link-rate RMSE、node RMSE、task RMSE，并进一步计划补跨 seed、扰动、阈值迁移和置信区间。这让模型能力和瓶颈都更容易解释。

第六个创新是后续应用路径。PI-JWM 的直接目标是预测未来状态，但它可以进一步作为候选动作评估器，用来比较不同卸载、资源分配和轨迹动作的未来效果。

## 5. 当前实验结果怎么讲

最新结果是 v6 full80 GPU 实验，数据划分为 train=1520、val=190、test=190，三种模式都训练 80 epoch。

三种模式分别是 dual、physical_only 和 information_only。dual 同时使用物理图和信息图；physical_only 只看几何、位置、距离、覆盖和移动关系；information_only 只看通信、任务、资源和动作历史。

结果上，三种模式 activity F1 都是 1.0，说明链路是否活跃这个二分类问题已经能稳定学习。真正拉开差距的是速率幅值。dual 的 active-rate RMSE 是 228.318，link-rate RMSE 是 6.416，两项都是最低，说明双图联合对链路速率预测有帮助。

task RMSE 上 information_only 最好，是 3.381，这说明任务状态更依赖队列、资源和动作历史。node RMSE 上 physical_only 最好，是 37.067，这说明节点状态更依赖物理图。这个现象支持双图设计：不同预测目标需要的信息来源不同，双图结构可以把这些来源拆开建模。

训练曲线方面，三种模式验证损失都能下降，说明数据链路和训练流程是可用的；但是 rate loss 仍然是主要误差来源，所以后续重点应该放在 active-rate 建模、速率不确定性和跨 seed 稳定性上。

## 6. 后续计划

下一步主要做五件事。

第一，补普通模型和 world model 的统一对比。比如 MLP、Ridge、state-action predictor、non-rollout predictor 和 PI-JWM v6 放到同一指标体系下，回答“世界模型相比普通预测模型到底强在哪里”。

第二，继续改 active-rate 建模。当前 activity F1 已经稳定，主要误差来自活跃链路速率幅值。可以尝试 active-only regression、加权损失、两阶段 head、分布式输出和不确定性估计。

第三，增强双图融合。现在已经验证双图有价值，后续可以尝试跨图注意力、门控融合或目标自适应融合，让 node、link、rate、task 不同目标自动选择物理图和信息图的贡献。

第四，补稳健性和泛化实验。包括跨 seed、输入扰动、阈值迁移、置信区间和 per-seed 曲线，验证模型在随机过程变化下是否稳定。

第五，接回候选动作评估接口。在状态 rollout 更稳定后，用 PI-JWM 评估候选卸载、资源分配和轨迹动作，把 v5 selector/ranking 作为诊断接口接回。

## 7. 可能提问与回答

### Q1：PI-JWM 和参考仿真器是什么关系？

回答：参考仿真器主要用于生成动态日志和真实标签，相当于数据来源。PI-JWM 是我们训练出来的近似世界模型，目标是从这些日志中学习状态转移规律。后续如果 PI-JWM 足够稳定，就可以用它快速评估候选动作，减少每次都调用完整仿真的成本。

### Q2：为什么要做双图，单图不行吗？

回答：单图会把几何关系、通信关系、队列关系和资源关系混在一起。我们的消融结果也说明，不同目标依赖的信息来源不同：node 更依赖 physical graph，task 更依赖 information graph，rate 同时依赖两者。所以双图的意义是把关系类型拆开，再在融合层联合预测。

### Q3：现在已经是完整 world model 了吗？

回答：目前已经具备 world model 的核心形式，也就是输入状态和动作，输出未来状态，并且有 rollout 结构。但从最终目标看，还需要继续补多步稳定性、跨 seed 泛化、扰动鲁棒性和候选动作评估。所以当前可以说是 PI-JWM 的阶段性版本，已经从普通预测模型推进到了双图动作条件 rollout。

### Q4：为什么 activity F1 都是 1.0，还要继续做 link modeling？

回答：activity F1 只说明链路是否活跃这个二分类问题已经比较稳定，但速率幅值预测仍然有误差。实际调度里，知道链路开不开还不够，还要知道可用速率大概是多少，所以 active-rate RMSE 和 link-rate RMSE 更能反映下一阶段瓶颈。

### Q5：active-rate RMSE 是什么意思？

回答：rate 是链路速率，RMSE 是均方根误差。active-rate RMSE 只在真实活跃链路上计算速率误差，避免大量非活跃链路的零速率把结果冲淡。它衡量的是“链路真的开着时，模型预测速率幅值准不准”。

### Q6：dual 模式为什么 link rate 最好，但 task 不是最好？

回答：这说明不同目标的主要信息来源不一样。速率同时受物理距离、覆盖、移动关系和信息侧资源/动作影响，所以 dual 更有优势。任务状态更直接依赖队列、资源和动作历史，所以 information_only 在 task RMSE 上更低。这不是矛盾，反而说明双图消融能解释不同模块的作用。

### Q7：下一步为什么要补普通模型对比？

回答：因为如果只报告 PI-JWM 内部消融，还不能充分回答“世界模型相比普通模型有什么收益”。补 MLP、Ridge、state-action predictor 和 non-rollout predictor 之后，可以更清楚地证明 PI-JWM 的结构和 rollout 是否带来提升。

### Q8：后续会不会直接做决策？

回答：会接回决策接口，但顺序上先把状态 rollout 做稳。因为如果世界模型预测本身不稳定，用它做候选动作排序会放大误差。当前计划是先补稳定性、泛化和不确定性，再把 v5 selector/ranking 作为动作评估接口接回。

### Q9：这个方向最后可以形成什么完整成果？

回答：完整成果可以包括三部分：第一，一个物理-信息联合的状态转移建模框架；第二，一套面向节点、链路、任务和资源的多目标评估体系；第三，一个可用于候选动作评估的近似环境模型。这样可以从“理解系统如何演化”推进到“辅助系统如何决策”。
"""
    SCRIPT_PATH.write_text(content, encoding="utf-8")


def write_ai_prompts() -> None:
    content = """# AI 绘图提示词

说明：正式 LaTeX 文档中已经把非数据概念图位置空出来。以下提示词用于自行生成图片后替换对应占位图。建议生成 16:9 横图，清晰、学术风、无水印。为了避免错字，建议图片里不要放中文正文，必要标签可以后期手动加。

## 图 1：问题 Formulation 示意图

中文提示词：
面向低空/车联网协同系统的科研示意图。画面左侧是当前系统状态：无人机、车辆、路侧边缘节点、基站和边缘服务器，通过无线链路连接，并带有任务队列和资源图标；中间是调度动作：任务卸载、RB 分配、CPU 分配、回传动作，以箭头或控制模块表示；右侧是未来系统状态：节点位置变化、链路质量变化、任务状态变化。整体表达“当前物理/信息状态 + 调度动作 -> 未来节点/链路/任务状态”。风格为干净的学术信息图，白色背景，蓝色表示物理网络，青绿色表示信息网络，橙色表示动作和未来预测。不要出现水印，不要出现复杂小字，不要出现真实品牌。

English prompt:
Clean academic scientific infographic for a low-altitude and vehicular edge-network system. Left side shows current system state: UAVs, vehicles, roadside edge nodes, base stations, and edge servers connected by wireless links, with task queues and resource icons. Middle shows scheduling actions: task offloading, RB allocation, CPU allocation, and backhaul actions represented as arrows or a control module. Right side shows future system state: changed node positions, changed link quality, and changed task states. The core visual meaning is “current physical/information state + scheduling action -> future node/link/task state”. White background, blue physical network layer, teal information network layer, orange action and prediction highlights. No watermark, no dense tiny text, no brand logos.

## 图 2：方法路线示意图

中文提示词：
科研流程图，展示从动态系统日志到 PI-JWM 双图世界模型的研究链路。画面从左到右分为四段：第一段是动态网络日志和轨迹，包含任务产生、信道变化、资源执行的抽象图标；第二段是样本构建，包含 dataset、多 seed、strict action、edge action 的数据块；第三段是基线诊断，包含 state-only、state-action、edge-level、two-stage 的模型模块；第四段是 PI-JWM v6，包含物理图、信息图、融合层和 rollout 输出。风格简洁、学术、适合论文/汇报文档，白色背景，少量蓝/绿/橙配色。不要大段文字，不要水印。

English prompt:
Academic research pipeline diagram showing the workflow from dynamic system logs to the PI-JWM dual-graph world model. Left-to-right four stages: dynamic network logs and trajectories with abstract icons for task generation, channel variation, and resource execution; dataset construction with data blocks for dataset, multi-seed, strict action, and edge action; baseline diagnostics with model modules for state-only, state-action, edge-level, and two-stage models; PI-JWM v6 with physical graph, information graph, fusion layer, and rollout outputs. Clean paper/report style, white background, restrained blue/green/orange palette, no dense text, no watermark.

## 图 3：PI-JWM 双图结构示意图

中文提示词：
PI-JWM 物理-信息联合世界模型结构图。左上为物理图，节点代表无人机、车辆、边缘节点，边代表距离、覆盖、相对运动关系，用蓝色；左下为信息图，节点和边代表任务、通信链路、队列、资源和动作历史，用青绿色；中间是融合与 rollout 模块，表现为神经网络/模型核心；右侧是四个输出 head：node state、link activity、link rate、task state，用不同颜色的输出箭头表示。画面要清晰、专业、适合学术文档。不要水印，不要品牌，不要复杂背景。

English prompt:
Architecture diagram for PI-JWM, a Physical-Information Joint World Model. Upper-left physical graph: nodes represent UAVs, vehicles, and edge nodes; edges represent distance, coverage, and relative mobility, colored blue. Lower-left information graph: nodes and edges represent tasks, communication links, queues, resources, and action history, colored teal. Center module is fusion and rollout, shown as a neural model core. Right side has four output heads: node state, link activity, link rate, and task state, shown with colored arrows. Clear professional academic style, suitable for a research report. No watermark, no brand logos, no complex background.
"""
    PROMPT_PATH.write_text(content, encoding="utf-8")


def update_plan_file() -> None:
    plan_path = WORKSPACE_ROOT / "本地计划表.md"
    if not plan_path.exists():
        return
    try:
        text = plan_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = plan_path.read_text(encoding="gbk", errors="ignore")

    marker = "## 2026-05-31 进组以来工作总结材料"
    addition = f"""

{marker}

- 已重写 5.31 线上讨论材料，正式文档只保留标题、正文、真实数据图和概念图位。
- 非数据概念图不再直接生成，统一在 LaTeX 中留空位；AI 绘图提示词单独放在 `文档/组会/5.31/{PROMPT_PATH.name}`。
- 正式文档文件：`文档/组会/5.31/{TEX_PATH.name}`。
- PDF 文件：`文档/组会/5.31/{PDF_PATH.name}`。
- 主线保持 PI-JWM，参考仿真器只作为数据来源。
"""
    if marker not in text:
        plan_path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")


def compile_latex() -> None:
    xelatex = shutil.which("xelatex")
    if xelatex is None:
        raise RuntimeError("No xelatex executable found. Please install TeX Live or MiKTeX.")

    cmd = [xelatex, "-interaction=nonstopmode", "-halt-on-error", TEX_PATH.name]
    subprocess.run(cmd, cwd=OUT_DIR, check=True)
    subprocess.run(cmd, cwd=OUT_DIR, check=True)

    produced = OUT_DIR / f"{TEX_PATH.stem}.pdf"
    if produced.exists() and produced != PDF_PATH:
        produced.replace(PDF_PATH)
    if not PDF_PATH.exists():
        raise RuntimeError(f"PDF was not produced: {PDF_PATH}")


def cleanup_unused_outputs() -> None:
    for docx in OUT_DIR.glob("*.docx"):
        docx.unlink()
    for old_dir_name in ["figures", "figures_latex"]:
        old_dir = OUT_DIR / old_dir_name
        if old_dir.exists() and old_dir.is_dir():
            shutil.rmtree(old_dir)
    for stale_figure in ["cover.png", "problem.png", "pipeline.png", "dualgraph.png"]:
        stale_path = FIG_DIR / stale_figure
        if stale_path.exists():
            stale_path.unlink()


def cleanup_latex_aux() -> None:
    for suffix in [".aux", ".log", ".out", ".toc", ".synctex.gz"]:
        aux = OUT_DIR / f"{TEX_PATH.stem}{suffix}"
        if aux.exists():
            aux.unlink()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_unused_outputs()

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    figures = build_figures(summary)
    write_latex(figures)
    write_script()
    write_ai_prompts()
    compile_latex()
    cleanup_latex_aux()
    update_plan_file()

    print(TEX_PATH)
    print(PDF_PATH)
    print(SCRIPT_PATH)
    print(PROMPT_PATH)


if __name__ == "__main__":
    main()
