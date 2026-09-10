"""Query PI-JWM's curated registries before opening raw project evidence."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


REGISTRY_ROOT = Path("docs/registries")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _flatten(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten(child)}" for key, child in value.items())
    if isinstance(value, list):
        return " ".join(_flatten(child) for child in value)
    if value is None:
        return ""
    return str(value)


def _normalized(value: str) -> str:
    return re.sub(r"\s+", "", value.casefold())


def _query_tokens(question: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9_-]{2,}", question.casefold()))
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", question):
        for width in (2, 3, 4):
            if len(chunk) < width:
                continue
            tokens.update(chunk[index : index + width] for index in range(len(chunk) - width + 1))
    return tokens


def _record(
    *,
    identifier: str,
    kind: str,
    title: str,
    summary: str,
    keywords: list[str],
    sources: list[str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": identifier,
        "kind": kind,
        "title": title,
        "summary": summary,
        "keywords": keywords,
        "sources": list(dict.fromkeys(source for source in sources if source)),
        "search_text": _flatten(payload),
    }


def _load_records(repo_root: Path) -> list[dict[str, Any]]:
    root = repo_root / REGISTRY_ROOT
    routes = _read_json(root / "question_routes.json")["routes"]
    experiments = _read_json(root / "experiment_registry.json")["experiments"]
    results = _read_json(root / "results_registry.json")["results"]
    history = _read_json(root / "historical_method_registry.json")["methods"]
    deferred = _read_json(root / "deferred_work.json")["items"]
    artifacts = _read_csv(root / "generated" / "artifact_catalog.csv")
    files = _read_csv(root / "generated" / "tracked_file_inventory.csv")
    records: list[dict[str, Any]] = []

    for row in routes:
        records.append(
            _record(
                identifier=row["id"],
                kind="question_route",
                title=row["title"],
                summary=row["summary"],
                keywords=list(row["keywords"]),
                sources=list(row["primary_sources"]) + list(row.get("verification_sources", [])),
                payload=row,
            )
        )
    for row in experiments:
        keywords = [row["id"], row["name"], str(row.get("seed") or ""), row["method"]]
        if row["method"] == "entity_aligned_dual_graph_rssm_v1":
            keywords.append("实体级RSSM")
        if row["id"].startswith("P4-EARSSM-SEED-") and row["status"] == "passed_single_seed":
            keywords.extend(["当前方法", "当前模型", "训练入口", "现在使用"])
        records.append(
            _record(
                identifier=row["id"],
                kind="experiment",
                title=row["name"],
                summary=row.get("summary_zh", row["conclusion"]),
                keywords=keywords,
                sources=[
                    row.get("code"),
                    row.get("configuration"),
                    row.get("protocol"),
                    row.get("data"),
                    row.get("result"),
                    row.get("audit"),
                ],
                payload=row,
            )
        )
    for row in results:
        records.append(
            _record(
                identifier=row["id"],
                kind="result",
                title=f"formal result for seed {row['seed']}",
                summary=row.get("summary_zh", f"{row['status']}; {row['claim_boundary']}"),
                keywords=[row["id"], row["experiment_id"], str(row["seed"]), "结果", "指标", "数字", "checkpoint"],
                sources=[row["audit"], "docs/registries/results_registry.json", "docs/registries/experiment_registry.json"],
                payload=row,
            )
        )
    for row in history:
        records.append(
            _record(
                identifier=row["id"],
                kind="historical_method",
                title=row["name"],
                summary=row.get("summary_zh", row["why_not_current"]),
                keywords=list(row["keywords"]) + [row["name"], row["id"]],
                sources=list(row["evidence_paths"]) + list(row["experiment_paths"]),
                payload=row,
            )
        )
    for row in deferred:
        keywords = [row["id"], row["type"], row["status"], "延期", "什么时候运行"]
        if row.get("seed") == 20260832:
            keywords.extend(["第三个seed", "第三个 seed", "20260832"])
        if row.get("type") == "remote_sync":
            keywords.extend(["远端同步", "同步"])
        records.append(
            _record(
                identifier=row["id"],
                kind="deferred_work",
                title=row["id"],
                summary=row.get("summary_zh", row["reason"]),
                keywords=keywords,
                sources=["docs/registries/deferred_work.json", row.get("entrypoint")],
                payload=row,
            )
        )
    for row in artifacts:
        artifact_id = row["artifact_id"]
        artifact_path = f"code/artifacts/{artifact_id}"
        sources = [artifact_path]
        for field in ("config_path", "manifest_path", "primary_result_path"):
            relative = row.get(field)
            if relative:
                sources.append(f"code/artifacts/{relative}")
        records.append(
            _record(
                identifier=f"ARTIFACT::{artifact_id}",
                kind="artifact",
                title=Path(artifact_id).name,
                summary=(
                    f"自动目录记录：family={row.get('family') or 'unknown'}，"
                    f"status={row.get('status') or 'unclassified'}，"
                    f"seed={row.get('seed') or 'unknown'}；需打开控制文件确认含义。"
                ),
                keywords=[
                    artifact_id,
                    Path(artifact_id).name,
                    row.get("method", ""),
                    row.get("seed", ""),
                ],
                sources=sources,
                payload=row,
            )
        )
    for row in files:
        path = row["path"]
        if not path.startswith(("code/src/pi_jwm/", "code/scripts/", "code/tests/")):
            continue
        records.append(
            _record(
                identifier=f"FILE::{path}",
                kind="file",
                title=Path(path).name,
                summary=(
                    f"{row.get('lifecycle_status', 'unknown')} / "
                    f"{row.get('category', 'project_file')}；状态标签用于导航，不替代代码审查。"
                ),
                keywords=[path, Path(path).name, Path(path).stem],
                sources=[path],
                payload=row,
            )
        )
    return records


def _score(question: str, record: dict[str, Any]) -> int:
    query = _normalized(question)
    score = 0
    for keyword in record["keywords"]:
        normalized_keyword = _normalized(str(keyword))
        if normalized_keyword and normalized_keyword in query:
            score += 20 + min(len(normalized_keyword), 20)
    searchable = _normalized(record["search_text"] + " " + record["title"])
    for token in _query_tokens(question):
        normalized_token = _normalized(token)
        if normalized_token and normalized_token in searchable:
            score += 2 + min(len(normalized_token), 8)
    if any(word in question for word in ("以前", "之前", "历史", "弃用", "为什么不用")):
        if record["kind"] in {"historical_method", "question_route"}:
            score += 15
    if any(word in question for word in ("当前", "现在", "入口", "在哪里")):
        if record["kind"] in {"experiment", "question_route"}:
            score += 10
    if any(word in question for word in ("结果", "指标", "数字")) and record["kind"] == "result":
        score += 15
    if any(word in question for word in ("什么时候", "延期", "第三个", "同步")) and record["kind"] == "deferred_work":
        score += 15
    return score


def query_knowledge(repo_root: Path, question: str, *, limit: int = 8) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("query must not be empty")
    ranked: list[dict[str, Any]] = []
    for record in _load_records(repo_root):
        score = _score(question, record)
        if score <= 0:
            continue
        ranked.append(
            {
                "id": record["id"],
                "kind": record["kind"],
                "title": record["title"],
                "summary": record["summary"],
                "score": score,
                "sources": record["sources"],
            }
        )
    ranked.sort(key=lambda row: (-row["score"], row["kind"], row["id"]))
    return {
        "schema_version": "PI-JWM-project-knowledge-query-v1",
        "query": question,
        "matches": ranked[: max(1, limit)],
        "verification_required": True,
        "verification_rule": "Use these matches for navigation, then verify scientific claims against original code/config/metrics/manifest/audit.",
        "locked_test_accessed": False,
    }


def _render_text(result: dict[str, Any]) -> str:
    lines = [f"问题：{result['query']}", "候选入口："]
    for row in result["matches"]:
        lines.append(f"- [{row['kind']}] {row['id']}: {row['summary']}")
        for source in row["sources"][:4]:
            lines.append(f"  - {source}")
    lines.append("说明：以上仅用于定位；科研结论仍须核对原始证据。")
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = query_knowledge(args.repo_root.resolve(), args.query, limit=args.limit)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(_render_text(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
