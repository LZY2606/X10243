"""项目导出/导入：完整事件溯源包，保留稳定身份与来源关系。"""
from __future__ import annotations

import io
import zipfile
from typing import Any

from ..core.constants import SCHEMA_VERSION
from ..core.hashing import digest
from .db import dumps, loads


def export_bundle(store, project_id: str) -> bytes:
    project = store.conn.execute(
        "SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    branches = store.list_branches(project_id)
    events = []
    for b in branches:
        rows = store.conn.execute(
            "SELECT * FROM events WHERE project_id=? AND branch_id=? "
            "ORDER BY seq", (project_id, b["id"])).fetchall()
        for r in rows:
            d = dict(r)
            d["payload"] = loads(d.pop("payload_json"))
            events.append(d)
    materials = [dict(m, payload=loads(m["payload_json"]))
                 for m in store.conn.execute(
                     "SELECT * FROM materials WHERE project_id=?",
                     (project_id,))]
    bundle = {
        "bundle_kind": "titration_heats_project",
        "schema_version": SCHEMA_VERSION,
        "project": dict(project),
        "branches": branches,
        "events": events,
        "materials": materials,
    }
    body = dumps(bundle).encode("utf-8")
    checksum = digest(bundle)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", body)
        zf.writestr("SHA256", checksum)
    return buf.getvalue()


def read_bundle(data: bytes) -> tuple[dict[str, Any], str]:
    """解析并校验；结构损坏抛 ValueError(format_corrupt 语义由上层映射)。"""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        body = zf.read("project.json")
        checksum = zf.read("SHA256").decode().strip()
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"format_corrupt: 导出包结构损坏 ({exc})")
    bundle = loads(body.decode("utf-8"))
    if digest(bundle) != checksum:
        raise ValueError("format_corrupt: 校验和不匹配")
    if bundle.get("bundle_kind") != "titration_heats_project":
        raise ValueError("format_corrupt: 非 ITC 滴定热谱项目包")
    return bundle, checksum


def import_bundle(store, data: bytes,
                  new_project_id: str | None = None) -> dict[str, Any]:
    bundle, checksum = read_bundle(data)
    pid = new_project_id or bundle["project"]["id"]
    cur = store.conn.execute(
        "SELECT 1 FROM projects WHERE id=?", (pid,)).fetchone()
    if cur:
        raise ValueError(f"项目 {pid} 已存在")
    with store.conn:
        p = bundle["project"]
        store.conn.execute(
            "INSERT INTO projects(id,name,created_at,parent_event_id,"
            "head_event_id) VALUES(?,?,?,?,?)",
            (pid, p["name"], p["created_at"], p.get("parent_event_id"),
             p.get("head_event_id")))
        for b in bundle["branches"]:
            store.conn.execute(
                "INSERT INTO branches(id,project_id,name,parent_branch_id,"
                "fork_event_id,created_at,head_event_id) "
                "VALUES(?,?,?,?,?,?,?)",
                (b["id"], pid, b["name"], b.get("parent_branch_id"),
                 b.get("fork_event_id", ""), b["created_at"],
                 b.get("head_event_id")))
        for m in bundle["materials"]:
            store.conn.execute(
                "INSERT INTO materials(id,version,project_id,slot,label,"
                "payload_json,raw_digest,introduced_event_id,"
                "superseded_event_id) VALUES(?,?,?,?,?,?,?,?,?)",
                (m["id"], m["version"], pid, m["slot"], m.get("label"),
                 dumps(m["payload"]), m["raw_digest"],
                 m["introduced_event_id"], m.get("superseded_event_id")))
        for e in bundle["events"]:
            store.conn.execute(
                "INSERT INTO events(id,project_id,branch_id,seq,"
                "parent_event_id,type,payload_json,actor,note,created_at,"
                "txn_id) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (e["id"], pid, e["branch_id"], e["seq"],
                 e.get("parent_event_id"), e["type"], dumps(e["payload"]),
                 e.get("actor", "reviewer"), e.get("note"), e["created_at"],
                 e.get("txn_id", "imported")))
    return {"project_id": pid, "events": len(bundle["events"]),
            "materials": len(bundle["materials"]),
            "branches": len(bundle["branches"]), "checksum": checksum}
