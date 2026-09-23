"""导出 / 导入。往返后身份、顺序、来源关系保持不变。

格式损坏 (format_corrupt) 是与 budget_exhausted / semantically_unsolvable
不同的独立结果。
"""
from __future__ import annotations

import json

from .store import canonical

BUNDLE_FORMAT = "itc-review-bundle/1"


class FormatCorrupt(Exception):
    """导出包格式损坏。"""


def export_bundle(store):
    dump = store.dump()
    return {
        "format": BUNDLE_FORMAT,
        "meta": dump["meta"],
        "materials": dump["materials"],
        "events": dump["events"],
        "results": dump["results"],
    }


def export_json(store):
    return json.dumps(export_bundle(store), ensure_ascii=False, indent=1)


def _validate(bundle):
    if not isinstance(bundle, dict):
        raise FormatCorrupt("导出包不是 JSON 对象")
    if bundle.get("format") != BUNDLE_FORMAT:
        raise FormatCorrupt(f"未知格式 {bundle.get('format')!r}, 期望 {BUNDLE_FORMAT}")
    for key in ("materials", "events", "results"):
        if not isinstance(bundle.get(key), list):
            raise FormatCorrupt(f"导出包缺少 {key} 列表")
    for m in bundle["materials"]:
        if not all(k in m for k in ("id", "kind", "version", "payload", "digest", "seq")):
            raise FormatCorrupt(f"材料记录不完整: {m.get('id')!r}")


def import_bundle(store, bundle):
    """导入到(可为空的)库。保留原 id 与 seq, 幂等: 已存在的身份跳过。"""
    _validate(bundle)
    with store.operation("import", detail={"format": bundle["format"]}):
        for m in sorted(bundle["materials"], key=lambda m: m["seq"]):
            if store.material(m["id"]):
                continue
            store.conn.execute(
                "INSERT INTO materials(id,kind,version,payload,digest,seq,supersedes) "
                "VALUES(?,?,?,?,?,?,?)",
                (m["id"], m["kind"], m["version"], canonical(m["payload"]),
                 m["digest"], m["seq"], m.get("supersedes")))
        for e in sorted(bundle["events"], key=lambda e: e["seq"]):
            if store.event(e["id"]):
                continue
            store.conn.execute(
                "INSERT INTO events(id,seq,kind,payload,parents,input_digest,"
                "param_version,status) VALUES(?,?,?,?,?,?,?,?)",
                (e["id"], e["seq"], e["kind"], canonical(e["payload"]),
                 canonical(e["parents"]), e["input_digest"], e["param_version"],
                 e["status"]))
        for r in sorted(bundle["results"], key=lambda r: r["seq"]):
            if store.result(r["id"]):
                continue
            store.conn.execute(
                "INSERT INTO results(id,kind,event_id,seq,payload,digest,status) "
                "VALUES(?,?,?,?,?,?,?)",
                (r["id"], r["kind"], r["event_id"], r["seq"],
                 canonical(r["payload"]), r["digest"], r["status"]))
        # 序号计数器推进到导入内容之上, 保证后续写入不与导入身份冲突
        max_seq = max(
            [m["seq"] for m in bundle["materials"]]
            + [e["seq"] for e in bundle["events"]]
            + [r["seq"] for r in bundle["results"]] + [store.head_seq()])
        store.conn.execute(
            "INSERT INTO meta(key,value) VALUES('seq',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(max_seq),))
    return {"imported_materials": len(bundle["materials"]),
            "imported_events": len(bundle["events"]),
            "imported_results": len(bundle["results"])}


def import_json(store, text):
    try:
        bundle = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise FormatCorrupt(f"JSON 解析失败: {exc}") from exc
    return import_bundle(store, bundle)
