"""项目导出/导入：身份、顺序、分支与决定历史完整往返。

导出物为自描述 JSON；digest 覆盖内容表。导入时：
  * 字节/JSON 损坏           -> format_corrupt
  * 引用断裂、顺序不一致等   -> semantic_unsolvable
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .errors import Diagnostic, Outcome
from .project import digest

FORMAT = "itc-project/1"
CONTENT_TABLES = ["branches", "materials", "results", "events"]


def _rows(conn, table: str) -> list[dict[str, Any]]:
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY " + (
        "seq" if table == "events" else
        "material_id,version" if table == "materials" else
        "result_id" if table == "results" else "branch_id")).fetchall()
    return [dict(r) for r in rows]


def export_project(store) -> dict[str, Any]:
    doc = {
        "format": FORMAT,
        "project": store.get_meta("project_name", "ITC 滴定热谱审议案例"),
        "tables": {t: _rows(store.conn, t) for t in CONTENT_TABLES},
    }
    doc["digest"] = hashlib.sha256(
        json.dumps({"format": doc["format"], "project": doc["project"],
                    "tables": doc["tables"]}, ensure_ascii=False,
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return doc


def import_project(store, doc: Any) -> Outcome:
    diagnostics: list[Diagnostic] = []
    if not isinstance(doc, dict):
        return Outcome.fail("format_corrupt", [Diagnostic(
            "format_corrupt", "导出物不是 JSON 对象", "root")])
    if doc.get("format") != FORMAT:
        return Outcome.fail("format_corrupt", [Diagnostic(
            "format_mismatch", f"格式标识 {doc.get('format')!r} != {FORMAT}",
            "format")])
    tables = doc.get("tables")
    if not isinstance(tables, dict):
        return Outcome.fail("format_corrupt", [Diagnostic(
            "format_corrupt", "缺少 tables 对象", "tables")])
    want = hashlib.sha256(
        json.dumps({"format": doc["format"], "project": doc.get("project"),
                    "tables": tables}, ensure_ascii=False,
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if doc.get("digest") != want:
        return Outcome.fail("format_corrupt", [Diagnostic(
            "digest_mismatch", "内容摘要不一致，导出物已损坏", "digest")])

    # 语义校验：身份、顺序、引用关系
    for t in CONTENT_TABLES:
        if t not in tables or not isinstance(tables[t], list):
            return Outcome.fail("format_corrupt", [Diagnostic(
                "format_corrupt", f"缺少表 {t}", f"tables.{t}")])
    seqs = [r.get("seq") for r in tables["events"]]
    if seqs != list(range(1, len(seqs) + 1)):
        return Outcome.fail("semantic_unsolvable", [Diagnostic(
            "event_order_broken", "事件 seq 不是从 1 开始的连续序列",
            "tables.events")])
    mat_keys = {(r["material_id"], r["version"])
                for r in tables["materials"]}
    result_ids = {r["result_id"] for r in tables["results"]}
    branch_ids = {r["branch_id"] for r in tables["branches"]}
    for ev in tables["events"]:
        if ev.get("branch") not in branch_ids:
            return Outcome.fail("semantic_unsolvable", [Diagnostic(
                "dangling_branch",
                f"事件 seq={ev.get('seq')} 引用未知分支 {ev.get('branch')}",
                f"events[{ev.get('seq')}].branch")])
    for r in tables["results"]:
        try:
            deps = json.loads(r["deps"])
        except (TypeError, json.JSONDecodeError):
            return Outcome.fail("format_corrupt", [Diagnostic(
                "format_corrupt", f"结果 {r.get('result_id')} deps 非法 JSON",
                f"results.{r.get('result_id')}.deps")])
        for name, spec in deps.items():
            if not spec:
                continue
            if name == "integration":
                if spec[0] not in result_ids:
                    return Outcome.fail("semantic_unsolvable", [Diagnostic(
                        "dangling_dependency",
                        f"结果 {r['result_id']} 依赖缺失的积分 {spec[0]}",
                        f"results.{r['result_id']}.deps.{name}")])
            elif tuple(spec) not in mat_keys:
                return Outcome.fail("semantic_unsolvable", [Diagnostic(
                    "dangling_dependency",
                    f"结果 {r['result_id']} 依赖缺失材料 {spec}",
                    f"results.{r['result_id']}.deps.{name}")])

    # 全量恢复（先清空，再按原始身份写回；不走 journal 批量路径，
    # 但整个恢复放在单事务内，保证原子性）
    with store.conn:
        for t in ("journal", "events", "results", "materials",
                  "branches", "meta"):
            store.conn.execute(f"DELETE FROM {t}")
        for b in tables["branches"]:
            store.conn.execute(
                "INSERT INTO branches(branch_id,name,head_seq,"
                "parent_branch,parent_seq) VALUES(?,?,?,?,?)",
                (b["branch_id"], b["name"], b["head_seq"],
                 b.get("parent_branch"), b.get("parent_seq", 0)))
        for m in tables["materials"]:
            store.conn.execute(
                "INSERT INTO materials(material_id,version,kind,payload,raw,"
                "created_seq) VALUES(?,?,?,?,?,?)",
                (m["material_id"], m["version"], m["kind"], m["payload"],
                 m.get("raw", ""), m["created_seq"]))
        for r in tables["results"]:
            store.conn.execute(
                "INSERT INTO results(result_id,kind,payload,deps,"
                "input_digest,param_version,created_seq,branch) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (r["result_id"], r["kind"], r["payload"], r["deps"],
                 r["input_digest"], r["param_version"], r["created_seq"],
                 r["branch"]))
        for e in tables["events"]:
            store.conn.execute(
                "INSERT INTO events(seq,event_id,branch,kind,payload,parents,"
                "created_at) VALUES(?,?,?,?,?,?,?)",
                (e["seq"], e["event_id"], e["branch"], e["kind"],
                 e["payload"], e["parents"], e["created_at"]))
        store.set_meta("project_name",
                       doc.get("project", "ITC 滴定热谱审议案例"))
    return Outcome.good({"imported_events": len(seqs),
                         "imported_materials": len(mat_keys),
                         "imported_results": len(result_ids)})
