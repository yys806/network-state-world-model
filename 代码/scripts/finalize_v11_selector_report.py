"""Freeze the auditable PI-JWM v11 selector result bundle after evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sha256_manifest(root: Path) -> list[dict[str, Any]]:
    """Return a deterministic manifest without recursively hashing itself."""
    excluded = {"sha256_manifest.json"}
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if path.name in excluded or "/logs/" in f"/{relative}":
            continue
        rows.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_task_energy_pareto(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted({row["action_family"] for row in rows})
    figure, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    for family in families:
        selected = [row for row in rows if row["action_family"] == family]
        x = [float(row["predicted_energy_delta_proxy"]) for row in selected]
        y = [float(row["predicted_task_delta_proxy"]) for row in selected]
        axis.scatter(x, y, s=24, alpha=0.72, label=family)
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.axvline(0.0, color="#777777", linewidth=0.8)
    axis.set_xlabel("Predicted UAV-energy proxy delta (lower is better)")
    axis.set_ylabel("Predicted task-progress proxy delta (higher is better)")
    axis.set_title("PI-JWM v11 selected-candidate task–energy trade-off")
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    report_root = args.report_root.resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
    matched = json.loads(args.matched_summary.read_text(encoding="utf-8"))
    if not bool(frozen.get("configuration_frozen")):
        raise PermissionError("final report requires a validation-frozen selector")
    if str(matched.get("split_name")) != "matched_test":
        raise ValueError("final report requires a matched_test summary")
    for key in ("configuration_digest", "selector_freeze_digest"):
        if str(matched.get(key)) != str(frozen.get(key)):
            raise ValueError(f"matched summary {key} differs from frozen selector")

    decision_rows = _read_csv(args.matched_decision_trace)
    if not decision_rows:
        raise ValueError("matched decision trace is empty")
    pareto_path = report_root / "task_energy_pareto_selected.png"
    _plot_task_energy_pareto(decision_rows, pareto_path)

    acceptance = matched.get("acceptance", {})
    facts = [
        f"matched-test active-rate RMSE = {matched['metrics']['active_rate_rmse']}",
        f"validation-selected tier = {acceptance.get('final_tier', 'not_available')}",
        f"defer ratio = {matched.get('defer_ratio')}",
    ]
    pending = [name for name, passed in acceptance.get("gates", {}).items() if not bool(passed)]
    conclusion = (
        "# PI-JWM v11 Selector 结论底稿\n\n"
        "## 观测事实\n\n"
        + "\n".join(f"- {value}" for value in facts)
        + "\n\n## 合理解释\n\n"
        "- selector 只按冻结后的 listwise score 排名，并由 calibration 增益下界与可观测 Pareto 规则决定执行或回退。\n"
        "- 逐 seed、stage、action-family 结果应结合 decision trace 与 CSV 解释，不能用 sample oracle 替代 deployable 结果。\n\n"
        "## 待验证假设\n\n"
        + ("\n".join(f"- 尚未通过：{value}" for value in pending) if pending else "- 当前冻结验收门均已通过。")
        + "\n"
    )
    (report_root / "selector_conclusion_draft.md").write_text(conclusion, encoding="utf-8")
    reproduction = (
        "bash 代码/scripts/run_v11_selector_finalization_gpu.sh\n"
        "python 代码/scripts/finalize_v11_selector_report.py "
        "--report-root <REPORT_ROOT> --frozen-manifest <FROZEN_MANIFEST> "
        "--matched-summary <SUMMARY_MATCHED_TEST> --matched-decision-trace <DECISION_TRACE>\n"
    )
    (report_root / "reproduction_commands.txt").write_text(reproduction, encoding="utf-8")
    summary = {
        "framework": "PI-JWM",
        "candidate": "v11",
        "configuration_digest": frozen["configuration_digest"],
        "selector_freeze_digest": frozen["selector_freeze_digest"],
        "result_kind": "deployable" if acceptance.get("final_tier") in {"A", "B"} else "diagnostic_only",
        "matched_test": matched,
        "deliverables": {
            "conclusion_draft": "selector_conclusion_draft.md",
            "task_energy_pareto": pareto_path.name,
            "reproduction_commands": "reproduction_commands.txt",
        },
    }
    (report_root / "selector_final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = build_sha256_manifest(report_root)
    (report_root / "sha256_manifest.json").write_text(
        json.dumps({"files": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--matched-summary", type=Path, required=True)
    parser.add_argument("--matched-decision-trace", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
