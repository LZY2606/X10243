"""项目导出/导入：身份、顺序、分支与决定历史保持；半写导入可恢复。"""
from __future__ import annotations

import io
import json
import zipfile

from .canonicalid import canonical, key_hash
from .storage import Store


def export_project(store: Store, pid: str) -> bytes:
    project = store.get_project(pid)
    if not project:
        raise KeyError(pid)
    branches = {}
    for b in store.list_branches(pid):
        branches[b] = {"head": store.branch_head(pid, b),
                       "events": store.branch_events(pid, b)}
    materials = store.all_materials(pid)
    manifest = {
        "format": "itcpkg", "format_version": 1,
        "project": project, "branches": list(branches),
        "material_count": len(materials),
        "payload": {"project": project, "branches": branches, "materials": materials},
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.json", canonical(project))
        z.writestr("branches.json", canonical(branches))
        z.writestr("materials.json", canonical(materials))
        z.writestr("manifest.json", canonical(manifest))
    return buf.getvalue()


class PackageCorrupt(Exception):
    pass


def import_project(path_or_bytes, db_path: str, _fault_hook=None) -> str:
    """原子导入：全部进同一事务；中途崩溃 => 无残留项目（半写恢复）。"""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        raw = bytes(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            raw = f.read()
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            project = json.loads(z.read("project.json"))
            branches = json.loads(z.read("branches.json"))
            materials = json.loads(z.read("materials.json"))
            manifest = json.loads(z.read("manifest.json"))
    except (KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise PackageCorrupt(f"format_corrupt: 案例包结构损坏 ({exc})") from exc

    if manifest.get("format") != "itcpkg":
        raise PackageCorrupt("format_corrupt: 不是 itcpkg 格式")
    if canonical(manifest["payload"]["materials"]) != canonical(materials):
        raise PackageCorrupt("semantic_unsolvable: 材料载荷与清单摘要不一致")

    old_hook = None
    store = Store(db_path)
    if store.get_project(project["pid"]):
        store.close()
        raise PackageCorrupt("semantic_unsolvable: 项目 id 已存在，拒绝覆盖")
    try:
        from . import storage as _s
        old_hook = _s.FAULT_HOOK
        if _fault_hook is not None:
            _s.FAULT_HOOK = _fault_hook
        with store.txn() as c:
            c.execute("INSERT INTO projects(pid,name,created_at) VALUES(?,?,?)",
                      (project["pid"], project["name"], project["created_at"]))
            for b in branches:
                c.execute("INSERT INTO branches(pid,branch,head) VALUES(?,?,?)",
                          (project["pid"], b, branches[b]["head"]))
            for m in materials:
                c.execute("INSERT INTO materials(mid,pid,stream,version,kind,source,"
                          "payload,diagnostics) VALUES(?,?,?,?,?,?,?,?)",
                          (m["mid"], m["pid"], m["stream"], m["version"], m["kind"],
                           m["source"], canonical(m["payload"]),
                           canonical(m["diagnostics"])))
            for b in branches:
                prev = None
                prev_eid = None
                for e in branches[b]["events"]:
                    deps = e.get("deps", "[]") if isinstance(e.get("deps"), str) else canonical(e.get("deps", []))
                    body = {"type": e["type"], "payload": e["payload"],
                            "parent": e["parent"], "seq": e["seq"],
                            "pid": project["pid"], "branch": b}
                    expect_hash = key_hash({"body": body, "prev": prev,
                                            "deps": json.loads(deps)})
                    if expect_hash != e["hash"]:
                        raise PackageCorrupt(
                            f"semantic_unsolvable: hash 链在 {b}#seq{e['seq']} 被破坏")
                    expected_parent = prev_eid
                    if e["parent"] != expected_parent:
                        raise PackageCorrupt(
                            f"semantic_unsolvable: 顺序/父子关系在 {b}#seq{e['seq']} 不一致")
                    c.execute("INSERT INTO events(eid,pid,branch,seq,parent,type,payload,"
                              "hash,deps) VALUES(?,?,?,?,?,?,?,?,?)",
                              (e["eid"], project["pid"], b, e["seq"], e["parent"],
                               e["type"], canonical(e["payload"]), e["hash"], deps))
                    prev = expect_hash
                    prev_eid = e["eid"]
    except PackageCorrupt:
        store.close()
        raise
    except Exception:
        store.close()
        raise
    finally:
        from . import storage as _s
        _s.FAULT_HOOK = old_hook
    store.close()
    return project["pid"]
