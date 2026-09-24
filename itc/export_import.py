"""项目导出 / 导入：身份、分支、顺序与来源关系在往返后保持。

导出包含全部事件（含 seq）、材料版本、计算与决定；导入幂等，
格式损坏返回 format_error，与语义无解、预算耗尽是不同结果。
"""
from __future__ import annotations

import json

from .store import Store

FORMAT = "itc-review/1"


def export_project(store: Store) -> dict:
    c = store.conn
    return {
        "format": FORMAT,
        "events": [dict(r) for r in c.execute(
            "SELECT id, seq, branch, type, payload, created FROM events"
            " ORDER BY seq")],
        "materials": [dict(r) for r in c.execute(
            "SELECT id, version, kind, raw, event_id FROM materials"
            " ORDER BY id, version")],
        "computations": [dict(r) for r in c.execute(
            "SELECT id, kind, param_version, parents, status, result,"
            " event_id FROM computations ORDER BY id")],
        "decisions": [dict(r) for r in c.execute(
            "SELECT id, subject, action, rationale, branch, event_id"
            " FROM decisions ORDER BY id")],
    }


def import_project(store: Store, data: dict) -> dict:
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        return {"status": "format_error",
                "diagnostics": [{"path": "format", "code": "format_error",
                                 "message": f"期望 format={FORMAT}"}]}
    for key in ("events", "materials", "computations", "decisions"):
        if not isinstance(data.get(key), list):
            return {"status": "format_error",
                    "diagnostics": [{"path": key, "code": "format_error",
                                     "message": f"缺少数组字段 {key}"}]}
    seqs = [e.get("seq") for e in data["events"]]
    if len(seqs) != len(set(seqs)):
        return {"status": "format_error",
                "diagnostics": [{"path": "events", "code": "format_error",
                                 "message": "事件 seq 重复"}]}
    effects = ([("events", e["id"]) for e in data["events"]]
               + [("computations", c["id"]) for c in data["computations"]]
               + [("decisions", d["id"]) for d in data["decisions"]])
    intent = store.begin_intent(effects)
    try:
        for e in data["events"]:
            store.conn.execute(
                "INSERT OR IGNORE INTO events(id, seq, branch, type, payload,"
                " created) VALUES(?,?,?,?,?,?)",
                (e["id"], e["seq"], e["branch"], e["type"],
                 e["payload"] if isinstance(e["payload"], str)
                 else json.dumps(e["payload"], ensure_ascii=False),
                 e.get("created", "")))
        for m in data["materials"]:
            store.conn.execute(
                "INSERT OR IGNORE INTO materials(id, version, kind, raw,"
                " event_id) VALUES(?,?,?,?,?)",
                (m["id"], m["version"], m["kind"],
                 m["raw"] if isinstance(m["raw"], str)
                 else json.dumps(m["raw"], ensure_ascii=False),
                 m["event_id"]))
        for comp in data["computations"]:
            store.conn.execute(
                "INSERT OR IGNORE INTO computations(id, kind, param_version,"
                " parents, status, result, event_id) VALUES(?,?,?,?,?,?,?)",
                (comp["id"], comp["kind"], comp["param_version"],
                 comp["parents"] if isinstance(comp["parents"], str)
                 else json.dumps(comp["parents"], sort_keys=True),
                 comp["status"],
                 comp["result"] if isinstance(comp["result"], str)
                 else json.dumps(comp["result"], ensure_ascii=False),
                 comp["event_id"]))
        for d in data["decisions"]:
            store.conn.execute(
                "INSERT OR IGNORE INTO decisions(id, subject, action,"
                " rationale, branch, event_id) VALUES(?,?,?,?,?,?)",
                (d["id"], d["subject"], d["action"], d["rationale"],
                 d["branch"], d["event_id"]))
        store.conn.commit()
    except Exception as exc:  # 半写：事务回滚，意图留给恢复流程
        store.conn.rollback()
        return {"status": "format_error",
                "diagnostics": [{"path": "import", "code": "format_error",
                                 "message": str(exc)}], "intent": intent}
    store.complete_intent(intent)
    return {"status": "ok", "imported": {"events": len(data["events"]),
                                         "materials": len(data["materials"]),
                                         "computations": len(data["computations"]),
                                         "decisions": len(data["decisions"])}}
