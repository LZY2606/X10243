"""SQLite 事件溯源存储。

- materials：不可变材料（内容寻址，独立版本化；去重但不覆盖）。
- events：机器可读取事件，构成按 parent 链接的链/分支 DAG；
  计算事件记录 inputs（内容摘要+版本）、params（规则版本）、parents（父级依赖）。
- artifacts：计算产物（积分、拟合候选、决定），内容身份稳定。
- op_ledger：半写恢复——任何多语句写操作先登记 PENDING，提交后标 DONE；
  重开时 PENDING 记录对应事务随回滚消失，恢复结果必须一致。
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any

from .models import (
    Diagnostic, ITCError, BudgetExhausted, FormatCorrupted,
    EV_POWER_IMPORTED, EV_INJECTIONS_IMPORTED, EV_CONCENTRATION_IMPORTED,
    EV_CONDITIONS_IMPORTED, EV_BASELINE_IMPORTED, EV_BLANK_IMPORTED,
)

SCHEMA_VERSION = 1
IMPORT_EVENT = {
    "power": EV_POWER_IMPORTED,
    "injections": EV_INJECTIONS_IMPORTED,
    "concentration": EV_CONCENTRATION_IMPORTED,
    "conditions": EV_CONDITIONS_IMPORTED,
    "baseline": EV_BASELINE_IMPORTED,
    "blank": EV_BLANK_IMPORTED,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS materials (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  raw_text TEXT NOT NULL,
  parsed_json TEXT NOT NULL,
  created_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  seq INTEGER NOT NULL,
  chain_id TEXT NOT NULL,
  type TEXT NOT NULL,
  parent_id TEXT,
  payload_json TEXT NOT NULL,
  inputs_json TEXT NOT NULL DEFAULT '[]',
  params_json TEXT NOT NULL DEFAULT '{}',
  parents_json TEXT NOT NULL DEFAULT '[]',
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  created_event_id TEXT NOT NULL,
  json_text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS op_ledger (
  op_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  started_at REAL NOT NULL,
  detail TEXT NOT NULL
);
"""


def digest(data: Any) -> str:
    return hashlib.sha256(
        json.dumps(data, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # check_same_thread=False + 显式锁，便于 FastAPI 与测试共用
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._crash_after_ledger = False
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),))
            self._conn.commit()
            self.recover()

    # ---- 半写恢复 ----
    def recover(self) -> list[str]:
        """重开/故障后清理：回滚未提交事务，报告未完成操作。"""
        cleaned = []
        with self._lock:
            self._conn.rollback()
            rows = self._conn.execute(
                "SELECT op_id, detail FROM op_ledger WHERE status='PENDING'"
            ).fetchall()
            for r in rows:
                cleaned.append(r["op_id"])
                self._conn.execute(
                    "UPDATE op_ledger SET status='ABANDONED' WHERE op_id=?",
                    (r["op_id"],))
            self._conn.commit()
        return cleaned

    def set_fault_injection(self, crash_after_ledger: bool) -> None:
        """测试钩子：登记 PENDING 后模拟进程崩溃（不提交事务）。"""
        self._crash_after_ledger = crash_after_ledger

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- 读 ----
    def get_material(self, mid: str) -> dict | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM materials WHERE id=?", (mid,)).fetchone()
        if not r:
            return None
        return {
            "id": r["id"], "kind": r["kind"], "raw_text": r["raw_text"],
            "parsed": json.loads(r["parsed_json"]),
            "created_seq": r["created_seq"],
        }

    def list_materials(self, kind: str | None = None) -> list[dict]:
        with self._lock:
            if kind:
                rows = self._conn.execute(
                    "SELECT id,kind,created_seq FROM materials WHERE kind=? "
                    "ORDER BY created_seq,id", (kind,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id,kind,created_seq FROM materials "
                    "ORDER BY created_seq,id").fetchall()
        return [dict(r) for r in rows]

    def get_event(self, eid: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM events WHERE id=?",
                                   (eid,)).fetchone()
        return _event_row(r) if r else None

    def all_events(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events ORDER BY seq").fetchall()
        return [_event_row(r) for r in rows]

    def head_events(self) -> dict[str, dict]:
        """每个 chain 当前叶子（head）事件。"""
        events = self.all_events()
        has_child = set()
        by_chain: dict[str, list[dict]] = {}
        for ev in events:
            if ev["parent_id"]:
                has_child.add(ev["parent_id"])
            by_chain.setdefault(ev["chain_id"], []).append(ev)
        heads = {}
        for chain, evs in by_chain.items():
            leaves = [e for e in evs if e["id"] not in has_child]
            # 链上不允许分叉（分叉用新 chain + fork）；取唯一叶子
            leaves.sort(key=lambda e: e["seq"])
            heads[chain] = leaves[-1]
        return heads

    def chain_events(self, chain_id: str) -> list[dict]:
        return [e for e in self.all_events() if e["chain_id"] == chain_id]

    def ancestors(self, event_id: str) -> list[dict]:
        """沿 parent 回到根的有序事件（旧→新）。fork 事件 parent 指向另一链。"""
        index = {e["id"]: e for e in self.all_events()}
        out: list[dict] = []
        cur = index.get(event_id)
        seen = set()
        while cur is not None and cur["id"] not in seen:
            seen.add(cur["id"])
            out.append(cur)
            cur = index.get(cur["parent_id"]) if cur["parent_id"] else None
        out.reverse()
        return out

    def get_artifact(self, aid: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM artifacts WHERE id=?",
                                   (aid,)).fetchone()
        if not r:
            return None
        d = json.loads(r["json_text"])
        d.setdefault("id", r["id"])
        d["kind"] = r["kind"]
        d["created_event_id"] = r["created_event_id"]
        return d

    def list_artifacts(self, kind: str | None = None) -> list[dict]:
        with self._lock:
            if kind:
                rows = self._conn.execute(
                    "SELECT id,kind,created_event_id FROM artifacts "
                    "WHERE kind=? ORDER BY id", (kind,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id,kind,created_event_id FROM artifacts "
                    "ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def meta(self, key: str, default=None):
        with self._lock:
            r = self._conn.execute("SELECT value FROM meta WHERE key=?",
                                   (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    def set_meta(self, key: str, value) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value, ensure_ascii=False)))
            self._conn.commit()

    # ---- 写（全部经 op ledger，保证原子/可恢复）----
    def material_exists(self, mid: str) -> bool:
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM materials WHERE id=?", (mid,)).fetchone() is not None

    def put_material(self, mid: str, kind: str, raw_text: str,
                     parsed: dict, op_id: str | None = None) -> bool:
        own = op_id is None
        if own:
            op_id = self._begin_op(f"put_material:{mid}")
        try:
            with self._lock:
                exists = self._conn.execute(
                    "SELECT 1 FROM materials WHERE id=?", (mid,)).fetchone()
                if not exists:
                    seq = self._next_mat_seq()
                    self._conn.execute(
                        "INSERT INTO materials(id,kind,raw_text,parsed_json,created_seq)"
                        " VALUES(?,?,?,?,?)",
                        (mid, kind, raw_text,
                         json.dumps(parsed, ensure_ascii=False), seq))
                if own:
                    self._commit_op(op_id)
            return not exists
        except Exception:
            if own:
                self._rollback_op(op_id)
            raise

    def add_event(self, ev_type: str, chain_id: str, payload: dict,
                  parent_id: str | None = None, inputs: list | None = None,
                  params: dict | None = None, parents: list | None = None,
                  artifacts: list[tuple[str, str, dict]] | None = None,
                  op_id: str | None = None) -> dict:
        """追加事件并可选同事务写入产物。事件 ID 内容寻址 → 重放稳定。"""
        own = op_id is None
        if own:
            op_id = self._begin_op(f"add_event:{chain_id}:{ev_type}")
        try:
            with self._lock:
                if parent_id is not None:
                    pr = self._conn.execute(
                        "SELECT id FROM events WHERE id=?", (parent_id,)).fetchone()
                    if pr is None:
                        raise KeyError(f"父事件不存在: {parent_id}")
                seq = self._next_event_seq()
                basis = {
                    "type": ev_type, "chain_id": chain_id, "parent_id": parent_id,
                    "payload": payload, "seq": seq,
                }
                eid = "ev_" + digest(basis)[:20]
                if self._conn.execute("SELECT 1 FROM events WHERE id=?",
                                      (eid,)).fetchone():
                    raise ValueError(f"事件已存在（幂等冲突）: {eid}")
                created_at = time.time()
                self._conn.execute(
                    "INSERT INTO events(id,seq,chain_id,type,parent_id,payload_json,"
                    "inputs_json,params_json,parents_json,created_at)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (eid, seq, chain_id, ev_type, parent_id,
                     json.dumps(payload, ensure_ascii=False),
                     json.dumps(inputs or [], ensure_ascii=False),
                     json.dumps(params or {}, ensure_ascii=False),
                     json.dumps(parents or [], ensure_ascii=False), created_at))
                for aid, akind, adoc in (artifacts or []):
                    self._conn.execute(
                        "INSERT OR REPLACE INTO artifacts(id,kind,created_event_id,"
                        "json_text) VALUES(?,?,?,?)",
                        (aid, akind, eid, json.dumps(adoc, ensure_ascii=False)))
                if own:
                    self._commit_op(op_id)
                return {
                    "id": eid, "seq": seq, "chain_id": chain_id, "type": ev_type,
                    "parent_id": parent_id, "payload": payload,
                    "inputs": inputs or [], "params": params or {},
                    "parents": parents or [], "created_at": created_at,
                    "artifact_ids": [a[0] for a in (artifacts or [])],
                }
        except Exception:
            if own:
                self._rollback_op(op_id)
            raise

    def _next_mat_seq(self) -> int:
        r = self._conn.execute(
            "SELECT COALESCE(MAX(created_seq),0)+1 FROM materials").fetchone()
        return int(r[0])

    def _next_event_seq(self) -> int:
        r = self._conn.execute("SELECT COALESCE(MAX(seq),0)+1 FROM events").fetchone()
        return int(r[0])

    def _begin_op(self, detail: str) -> str:
        op_id = "op_" + digest([detail, time.time_ns(), os.urandom(8).hex()])[:16]
        with self._lock:
            self._conn.execute(
                "INSERT INTO op_ledger(op_id,status,started_at,detail)"
                " VALUES('PLACEHOLDER','X',0,'')")
            self._conn.execute("DELETE FROM op_ledger WHERE op_id='PLACEHOLDER'")
            self._conn.execute(
                "INSERT INTO op_ledger(op_id,status,started_at,detail)"
                " VALUES(?,?,?,?)",
                (op_id, "PENDING", time.time(), detail))
            if self._crash_after_ledger:
                self._crash_after_ledger = False
                # 模拟进程崩溃：连接关闭，未提交事务随 WAL/rollback 丢弃；
                # PENDING 行在同一事务内，因此不持久化。
                self._conn.rollback()
                raise ConnectionAbortedError("injected crash after op ledger write")
        return op_id

    def _commit_op(self, op_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE op_ledger SET status='DONE' WHERE op_id=?", (op_id,))
            self._conn.commit()

    def _rollback_op(self, op_id: str) -> None:
        with self._lock:
            self._conn.rollback()
            self._conn.execute(
                "UPDATE op_ledger SET status='ABANDONED' WHERE op_id=?", (op_id,))
            self._conn.commit()


def _event_row(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"], "seq": r["seq"], "chain_id": r["chain_id"],
        "type": r["type"], "parent_id": r["parent_id"],
        "payload": json.loads(r["payload_json"]),
        "inputs": json.loads(r["inputs_json"]),
        "params": json.loads(r["params_json"]),
        "parents": json.loads(r["parents_json"]),
        "created_at": r["created_at"],
    }


def export_archive(store: Store) -> dict:
    """完整项目导出：材料 + 事件 + 产物 + 元数据（身份、顺序、血缘全保留）。"""
    with store._lock:
        mats = [
            {"id": r["id"], "kind": r["kind"], "raw_text": r["raw_text"],
             "parsed": json.loads(r["parsed_json"]), "created_seq": r["created_seq"]}
            for r in store._conn.execute(
                "SELECT * FROM materials ORDER BY created_seq,id").fetchall()
        ]
        events = [_event_row(r) for r in store._conn.execute(
            "SELECT * FROM events ORDER BY seq").fetchall()]
        arts = [
            {"id": r["id"], "kind": r["kind"],
             "created_event_id": r["created_event_id"],
             "doc": json.loads(r["json_text"])}
            for r in store._conn.execute(
                "SELECT * FROM artifacts ORDER BY id").fetchall()
        ]
        meta = {r["key"]: json.loads(r["value"])
                for r in store._conn.execute("SELECT key,value FROM meta").fetchall()}
    return {
        "format": "itc-archive",
        "format_version": 1,
        "exported_at": time.time(),
        "meta": meta,
        "materials": mats,
        "events": events,
        "artifacts": arts,
    }


def import_archive(store: Store, archive: dict, *, verify_ids: bool = True) -> dict:
    """导入归档：内容身份、seq 顺序、parent 关系逐一校验后整体写入。

    三类失败在这里同样区分：格式损坏 / 语义无解（身份不一致）/ 预算由上层保证。
    """
    if not isinstance(archive, dict) or archive.get("format") != "itc-archive":
        raise FormatCorrupted(
            "不是 itc-archive 归档",
            [Diagnostic("bad_archive", "format 字段不是 itc-archive", "$")])
    op_id = store._begin_op("import_archive")
    try:
        with store._lock:
            n_mat = len(archive.get("materials", []))
            if n_mat > 10_000:
                raise BudgetExhausted("归档材料数超预算")
            # 1) 材料：内容哈希必须与其 id 一致（身份不可被导入篡改）
            for m in archive.get("materials", []):
                expect = None
                from .materials import material_id
                expect = material_id(m["kind"], m["raw_text"])
                if verify_ids and m["id"] != expect:
                    raise ITCError(
                        f"材料身份校验失败: {m['id']} != {expect}",
                        [Diagnostic("id_mismatch",
                                    "材料内容与身份不一致", m["id"])])
                store._conn.execute(
                    "INSERT OR IGNORE INTO materials(id,kind,raw_text,parsed_json,"
                    "created_seq) VALUES(?,?,?,?,?)",
                    (m["id"], m["kind"], m["raw_text"],
                     json.dumps(m["parsed"], ensure_ascii=False), m["created_seq"]))
            # 2) 事件：按 seq 顺序、内容寻址身份、parent 先于 child
            evs = sorted(archive.get("events", []),
                         key=lambda e: (e["seq"], e["id"]))
            known = {r["id"] for r in
                     store._conn.execute("SELECT id FROM events").fetchall()}
            last_seq = 0
            for e in evs:
                basis = {
                    "type": e["type"], "chain_id": e["chain_id"],
                    "parent_id": e["parent_id"], "payload": e["payload"],
                    "seq": e["seq"],
                }
                expect_id = "ev_" + digest(basis)[:20]
                if verify_ids and e["id"] != expect_id:
                    raise ITCError(
                        f"事件身份校验失败: {e['id']}",
                        [Diagnostic("id_mismatch",
                                    "事件内容与身份不一致", e["id"])])
                if e["parent_id"] and e["parent_id"] not in known:
                    raise ITCError(
                        f"事件父级缺失: {e['parent_id']}",
                        [Diagnostic("missing_parent",
                                    "父事件不在归档中", e["parent_id"])])
                store._conn.execute(
                    "INSERT OR IGNORE INTO events(id,seq,chain_id,type,parent_id,"
                    "payload_json,inputs_json,params_json,parents_json,created_at)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (e["id"], e["seq"], e["chain_id"], e["type"], e["parent_id"],
                     json.dumps(e["payload"], ensure_ascii=False),
                     json.dumps(e.get("inputs", []), ensure_ascii=False),
                     json.dumps(e.get("params", {}), ensure_ascii=False),
                     json.dumps(e.get("parents", []), ensure_ascii=False),
                     e.get("created_at", 0.0)))
                known.add(e["id"])
                last_seq = max(last_seq, e["seq"])
            # 3) 产物：创建事件必须存在
            for a in archive.get("artifacts", []):
                if a["created_event_id"] not in known:
                    raise ITCError(
                        f"产物的创建事件缺失: {a['id']}",
                        [Diagnostic("orphan_artifact",
                                    "产物创建事件缺失", a["id"])])
                store._conn.execute(
                    "INSERT OR IGNORE INTO artifacts(id,kind,created_event_id,"
                    "json_text) VALUES(?,?,?,?)",
                    (a["id"], a["kind"], a["created_event_id"],
                     json.dumps(a["doc"], ensure_ascii=False)))
            # 4) 元数据
            for k, v in (archive.get("meta") or {}).items():
                store._conn.execute(
                    "INSERT INTO meta(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (k, json.dumps(v, ensure_ascii=False)))
            store._commit_op(op_id)
        return {"materials": n_mat, "events": len(evs),
                "artifacts": len(archive.get("artifacts", []))}
    except Exception:
        store._rollback_op(op_id)
        raise


def reset_store(store: Store) -> None:
    op_id = store._begin_op("reset")
    try:
        with store._lock:
            for t in ("materials", "events", "artifacts", "op_ledger", "meta"):
                store._conn.execute(f"DELETE FROM {t}")
            store._conn.execute(
                "INSERT INTO meta(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),))
            store._commit_op(op_id)
    except Exception:
        store._rollback_op(op_id)
        raise
