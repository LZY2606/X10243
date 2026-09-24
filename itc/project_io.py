"""项目导出 / 导入。

- 导出包含全部事件、材料版本、结果、结果状态历史与分支，附校验和。
- 导入在单个事务中完成：任何失败整体回滚（半写恢复）。
- 格式损坏（JSON / 校验和 / 结构）与语义无解（悬空引用）是不同结果。
- 身份、顺序与来源关系逐项保留。
"""

import json

from .outcomes import CORRUPT_FORMAT, SEMANTIC_UNSOLVABLE, diag, failure, ok
from .store import canonical, digest

FORMAT = "itc-gsb-project/1"


def export_project(store):
    events = store.db.execute(
        "SELECT seq,id,branch,type,payload,parents FROM events ORDER BY seq").fetchall()
    materials = store.db.execute(
        "SELECT stream,version,kind,payload,seq FROM material_versions"
        " ORDER BY stream,version").fetchall()
    results = store.db.execute(
        "SELECT id,kind,payload,parents,input_digest,params,created_seq"
        " FROM results ORDER BY created_seq,id").fetchall()
    statuses = store.db.execute(
        "SELECT result_id,seq,status,reason FROM result_status"
        " ORDER BY result_id,seq").fetchall()
    branches = store.db.execute(
        "SELECT name,base_seq FROM branches ORDER BY name").fetchall()
    body = {
        "format": FORMAT,
        "events": [dict(zip(("seq", "id", "branch", "type", "payload", "parents"),
                            (r["seq"], r["id"], r["branch"], r["type"],
                             json.loads(r["payload"]), json.loads(r["parents"]))))
                   for r in events],
        "materials": [dict(zip(("stream", "version", "kind", "payload", "seq"),
                               (r["stream"], r["version"], r["kind"],
                                json.loads(r["payload"]), r["seq"])))
                      for r in materials],
        "results": [dict(zip(("id", "kind", "payload", "parents", "input_digest",
                              "params", "created_seq"),
                             (r["id"], r["kind"], json.loads(r["payload"]),
                              json.loads(r["parents"]), r["input_digest"],
                              json.loads(r["params"]), r["created_seq"])))
                    for r in results],
        "result_status": [dict(zip(("result_id", "seq", "status", "reason"),
                                   (r["result_id"], r["seq"], r["status"],
                                    r["reason"]))) for r in statuses],
        "branches": [dict(zip(("name", "base_seq"), (r["name"], r["base_seq"])))
                     for r in branches],
    }
    body["checksum"] = digest({k: v for k, v in body.items() if k != "checksum"})
    return body


def dumps_project(store):
    return json.dumps(export_project(store), ensure_ascii=False, indent=1)


def _validate(body):
    diags = []
    if not isinstance(body, dict) or body.get("format") != FORMAT:
        return [diag("bad_format", "format", "缺少或未知的 format 标记")]
    expect = body.get("checksum")
    actual = digest({k: v for k, v in body.items() if k != "checksum"})
    if expect != actual:
        diags.append(diag("checksum_mismatch", "checksum",
                          "校验和不匹配，文件可能损坏"))
    for key in ("events", "materials", "results", "result_status", "branches"):
        if not isinstance(body.get(key), list):
            diags.append(diag("bad_section", key, f"段 {key} 缺失或不是列表"))
    return diags


def _semantic_check(body):
    diags = []
    event_seqs = {e["seq"] for e in body["events"]}
    for m in body["materials"]:
        if m["seq"] not in event_seqs:
            diags.append(diag("dangling_material_seq",
                              f"materials.{m['stream']}#v{m['version']}",
                              "材料版本指向不存在的事件序号"))
    result_ids = {r["id"] for r in body["results"]}
    for s in body["result_status"]:
        if s["result_id"] not in result_ids:
            diags.append(diag("dangling_status", s["result_id"],
                              "状态历史指向不存在的结果"))
    for r in body["results"]:
        for p in r["parents"]:
            if p.get("type") == "result" and p.get("id") not in result_ids:
                diags.append(diag("dangling_parent", r["id"],
                                  f"父级结果 {p.get('id')} 不存在"))
    return diags


def import_project(store, body):
    """导入项目。成功返回 ok；失败分类返回，且库保持原状。"""
    if isinstance(body, (str, bytes)):
        try:
            body = json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            return failure(CORRUPT_FORMAT, [diag(
                "json_parse_error", "$", f"JSON 解析失败: {exc}")])
    diags = _validate(body)
    if diags:
        return failure(CORRUPT_FORMAT, diags)
    diags = _semantic_check(body)
    if diags:
        return failure(SEMANTIC_UNSOLVABLE, diags)
    try:
        with store._lock, store.db:
            store.db.execute("DELETE FROM result_status")
            store.db.execute("DELETE FROM results")
            store.db.execute("DELETE FROM material_versions")
            store.db.execute("DELETE FROM branches")
            store.db.execute("DELETE FROM events")
            store.db.execute(
                "DELETE FROM sqlite_sequence WHERE name='events'")
            for e in body["events"]:
                store.db.execute(
                    "INSERT INTO events(seq,id,branch,type,payload,parents)"
                    " VALUES(?,?,?,?,?,?)",
                    (e["seq"], e["id"], e["branch"], e["type"],
                     canonical(e["payload"]), canonical(e["parents"])))
            for m in body["materials"]:
                store.db.execute(
                    "INSERT INTO material_versions(stream,version,kind,payload,seq)"
                    " VALUES(?,?,?,?,?)",
                    (m["stream"], m["version"], m["kind"],
                     canonical(m["payload"]), m["seq"]))
            for r in body["results"]:
                store.db.execute(
                    "INSERT INTO results(id,kind,payload,parents,input_digest,"
                    "params,created_seq) VALUES(?,?,?,?,?,?,?)",
                    (r["id"], r["kind"], canonical(r["payload"]),
                     canonical(r["parents"]), r["input_digest"],
                     canonical(r["params"]), r["created_seq"]))
            for s in body["result_status"]:
                store.db.execute(
                    "INSERT INTO result_status(result_id,seq,status,reason)"
                    " VALUES(?,?,?,?)",
                    (s["result_id"], s["seq"], s["status"], s["reason"]))
            for b in body["branches"]:
                store.db.execute(
                    "INSERT INTO branches(name,base_seq) VALUES(?,?)",
                    (b["name"], b["base_seq"]))
    except Exception as exc:  # 事务整体回滚，库保持原状
        return failure(CORRUPT_FORMAT, [diag(
            "import_aborted", "$", f"导入中断，已整体回滚: {exc}")])
    return ok(imported={"events": len(body["events"]),
                        "materials": len(body["materials"]),
                        "results": len(body["results"])})
