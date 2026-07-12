"""Local CPU smoke run for the PI-JWM v8 full-world-model skeleton."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pi_jwm.paths import ARTIFACTS_DIR
from pi_jwm.v6_dual_graph import V6DualGraphBatch
from pi_jwm.v8_full_world_model import V8FullWorldModelConfig, V8FullWorldModelRollout


DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "pi_jwm_v8_full_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PI-JWM v8 synthetic CPU smoke.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument(
        "--graph-mode",
        choices=("dual", "physical_only", "information_only"),
        default="dual",
    )
    return parser.parse_args()


def build_synthetic_batch(config: V8FullWorldModelConfig, seed: int) -> V6DualGraphBatch:
    generator = torch.Generator().manual_seed(seed)
    batch_size = 4
    history = 5
    num_nodes = 6
    num_edges = int(config.edge_src_idx.numel())
    return V6DualGraphBatch(
        node_history=torch.randn(batch_size, history, num_nodes, config.node_dim, generator=generator),
        physical_edge_history=torch.randn(
            batch_size,
            history,
            num_edges,
            config.physical_edge_dim,
            generator=generator,
        ),
        info_edge_history=torch.randn(batch_size, history, num_edges, config.info_edge_dim, generator=generator),
        action_history=torch.randn(batch_size, history, num_edges, config.action_dim, generator=generator),
        future_actions=torch.randn(batch_size, config.horizon, num_edges, config.action_dim, generator=generator),
        task_history=torch.randn(batch_size, history, config.task_dim, generator=generator),
    )


def run_smoke(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    edge_src_idx = torch.tensor([0, 0, 1, 1, 2, 3, 4, 5, 5, 3])
    edge_dst_idx = torch.tensor([1, 2, 2, 3, 4, 4, 5, 0, 3, 1])
    config = V8FullWorldModelConfig(
        node_dim=6,
        physical_edge_dim=8,
        info_edge_dim=5,
        action_dim=4,
        task_dim=3,
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        graph_mode=args.graph_mode,
        edge_src_idx=edge_src_idx,
        edge_dst_idx=edge_dst_idx,
        return_message_diagnostics=True,
    )
    model = V8FullWorldModelRollout(config)
    batch = build_synthetic_batch(config, args.seed)

    with torch.no_grad():
        outputs = model(batch)

    return {
        "status": "smoke",
        "framework": "PI-JWM",
        "version": "v8",
        "note": "Synthetic CPU interface smoke only; not a trained experiment result.",
        "config": {
            "node_dim": config.node_dim,
            "physical_edge_dim": config.physical_edge_dim,
            "info_edge_dim": config.info_edge_dim,
            "action_dim": config.action_dim,
            "task_dim": config.task_dim,
            "hidden_dim": config.hidden_dim,
            "horizon": config.horizon,
            "graph_mode": config.graph_mode,
            "num_edges": int(edge_src_idx.numel()),
        },
        "output_shapes": {name: list(value.shape) for name, value in outputs.items()},
    }


def render_report(summary: dict) -> str:
    return "\n".join(
        [
            "# PI-JWM v8 Full World Model Smoke",
            "",
            f"- framework: {summary['framework']}",
            f"- version: {summary['version']}",
            f"- status: {summary['status']}",
            f"- note: {summary['note']}",
            f"- config: `{summary['config']}`",
            f"- output_shapes: `{summary['output_shapes']}`",
            "",
            "This smoke run verifies the local v8 M1 message-passing interface only.",
        ]
    )


def main() -> None:
    args = parse_args()
    summary = run_smoke(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "v8_full_smoke_summary.json"
    report_path = args.output_dir / "v8_full_smoke_report.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
