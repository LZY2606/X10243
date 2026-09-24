"""SQLite 版本化存储。

分层：
- materials   不可变材料，按 series 分开版本化（功率采样/注射/浓度/温度/池体积/基线/空白曲线）
- decisions   可修改的人工决定（追加式，旧决定不删）
- computations 计算结果：记录输入摘要、参数版本、父级依赖；创建后不改写
- staleness   失效标记（追加式，可定位到单次注射）
- events      机器可读事件日志，seq 全局单调，支持回到任意旧事件重放状态
- intents     写意图；打开库时清理 pending 意图及其残留行（半写恢复）
"""
from __future__ import annotations

import json
import os
import sqlite3

from .domain import canonical

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS intents(id TEXT PRIMARY KEY, status TEXT, seq INTEGER);
CREATE TABLE IF NOT EXISTS materials(
  id TEXT PRIMARY KEY, series TEXT, kind TEXT, version INTEGER,
  supersedes TEXT, payload TEXT, seq INTEGER, intent TEXT);
CREATE TABLE IF NOT EXISTS decisions(
  id TEXT PRIMARY KEY, kind TEXT, payload TEXT, seq INTEGER, intent TEXT);
CREATE TABLE IF NOT EXISTS computations(
  id TEXT PRIMARY KEY, kind TEXT, input_digest TEXT, param_version TEXT,
  parents TEXT, status TEXT, result TEXT, seq INTEGER, intent TEXT);
CREATE TABLE IF NOT EXISTS staleness(
  id INTEGER PRIMARY KEY AUTOINCREMENT, comp_id TEXT, scope TEXT,
  detail TEXT, seq INTEGER, intent TEXT);
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY, kind TEXT, payload TEXT, intent TEXT);
"""

TABLES_WITH_INTENT = ("materials", "decisions", "computations", "staleness", "events")


def _parse(row: sqlite3.Row, json_fields=()) -> dict:
    d = dict(row)
    for f in json_fields:
        if f in d and isinstance(d[f], str):
            d[f] = json.loads(d[f])
    return d


class Store:
    def __init__(self, path: str):
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()
        self.recover()

    def close(self):
        self.db.close()

    def is_empty(self) -> bool:
        return self.db.execute("SELECT COUNT(*) FROM materials").fetchone()[0] == 0

    # --- 全局顺序 ---
    def _next_seq(self, cur) -> int:
        cur.execute("INSERT INTO meta(key,value) VALUES('seq','0') ON CONFLICT(key) DO NOTHING")
        cur.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key='seq'")
        return int(cur.execute("SELECT value FROM meta WHERE key='seq'").fetchone()[0])

    def current_seq(self) -> int:
        row = self.db.execute("SELECT value FROM meta WHERE key='seq'").fetchone()
        return int(row[0]) if row else 0

    def _event(self, cur, seq: int, kind: str, payload: dict, intent: str):
        cur.execute("INSERT INTO events(seq,kind,payload,intent) VALUES(?,?,?,?)",
                    (seq, kind, canonical(payload), intent))

    def _write(self, fn):
        """单次原子写：分配一个 seq、一个已提交意图，执行 fn(cur, seq, intent)。"""
        with self.db:
            cur = self.db.cursor()
            seq = self._next_seq(cur)
            intent = f"intent-{seq}"
            cur.execute("INSERT INTO intents(id,status,seq) VALUES(?,?,?)", (intent, "committed", seq))
            return fn(cur, seq, intent)

    def recover(self):
        """打开时清理 pending 写意图及其残留行，并记录恢复事件。"""
        pending = [r[0] for r in self.db.execute("SELECT id FROM intents WHERE status='pending'").fetchall()]
        if not pending:
            return
        marks = ",".join("?" * len(pending))
        with self.db:
            cur = self.db.cursor()
            for table in TABLES_WITH_INTENT:
                cur.execute(f"DELETE FROM {table} WHERE intent IN ({marks})", pending)
            cur.execute(f"DELETE FROM intents WHERE id IN ({marks})", pending)
            seq = self._next_seq(cur)
            self._event(cur, seq, "recovery", {"dropped_intents": pending, "reason": "half_write"}, "")

    # --- 材料 ---
    def add_material(self, series: str, kind: str, payload: dict) -> dict:
        def fn(cur, seq, intent):
            row = cur.execute("SELECT MAX(version) AS v FROM materials WHERE series=?", (series,)).fetchone()
            version = (row["v"] or 0) + 1
            supersedes = f"{series}@v{version - 1}" if version > 1 else None
            mid = f"{series}@v{version}"
            cur.execute("INSERT INTO materials VALUES(?,?,?,?,?,?,?,?)",
                        (mid, series, kind, version, supersedes, canonical(payload), seq, intent))
            self._event(cur, seq, "material_revised" if version > 1 else "material_imported",
                        {"id": mid, "series": series, "kind": kind, "version": version,
                         "supersedes": supersedes}, intent)
            return {"id": mid, "series": series, "kind": kind, "version": version,
                    "supersedes": supersedes, "payload": payload, "seq": seq}
        return self._write(fn)

    def get_material(self, mid: str) -> dict | None:
        row = self.db.execute("SELECT * FROM materials WHERE id=?", (mid,)).fetchone()
        return _parse(row, ("payload",)) if row else None

    # --- 决定 ---
    def add_decision(self, kind: str, payload: dict) -> dict:
        def fn(cur, seq, intent):
            did = f"dec-{seq}"
            cur.execute("INSERT INTO decisions VALUES(?,?,?,?,?)", (did, kind, canonical(payload), seq, intent))
            self._event(cur, seq, f"decision:{kind}", {"id": did, **payload}, intent)
            return {"id": did, "kind": kind, "payload": payload, "seq": seq}
        return self._write(fn)

    # --- 计算 ---
    def add_computation(self, kind, input_digest, param_version, parents, status, result):
        """等价输入得到同一 id：重复计算幂等，不产生新行。"""
        cid = f"{kind}-{input_digest[:10]}"
        row = self.db.execute("SELECT * FROM computations WHERE id=?", (cid,)).fetchone()
        if row:
            return _parse(row, ("parents", "result")), False

        def fn(cur, seq, intent):
            cur.execute("INSERT INTO computations VALUES(?,?,?,?,?,?,?,?,?)",
                        (cid, kind, input_digest, param_version, canonical(parents),
                         status, canonical(result), seq, intent))
            self._event(cur, seq, f"computed:{kind}",
                        {"id": cid, "status": status, "param_version": param_version,
                         "parents": parents}, intent)
            return {"id": cid, "kind": kind, "input_digest": input_digest,
                    "param_version": param_version, "parents": parents,
                    "status": status, "result": result, "seq": seq}
        return self._write(fn), True

    def mark_stale(self, comp_id: str, scope: str, detail: dict):
        def fn(cur, seq, intent):
            cur.execute("INSERT INTO staleness(comp_id,scope,detail,seq,intent) VALUES(?,?,?,?,?)",
                        (comp_id, scope, canonical(detail), seq, intent))
            self._event(cur, seq, "computation_stale",
                        {"comp_id": comp_id, "scope": scope, "detail": detail}, intent)
        return self._write(fn)

    def record_event(self, kind: str, payload: dict):
        def fn(cur, seq, intent):
            self._event(cur, seq, kind, payload, intent)
        return self._write(fn)

    # --- 查询（均可按事件序号回放） ---
    def materials_at(self, seq: int | None = None) -> list[dict]:
        q, args = "SELECT * FROM materials", ()
        if seq is not None:
            q, args = q + " WHERE seq<=?", (seq,)
        return [_parse(r, ("payload",)) for r in self.db.execute(q + " ORDER BY seq,id", args)]

    def decisions_at(self, seq: int | None = None) -> list[dict]:
        q, args = "SELECT * FROM decisions", ()
        if seq is not None:
            q, args = q + " WHERE seq<=?", (seq,)
        return [_parse(r, ("payload",)) for r in self.db.execute(q + " ORDER BY seq,id", args)]

    def computations_at(self, seq: int | None = None) -> list[dict]:
        q, args = "SELECT * FROM computations", ()
        if seq is not None:
            q, args = q + " WHERE seq<=?", (seq,)
        return [_parse(r, ("parents", "result")) for r in self.db.execute(q + " ORDER BY seq,id", args)]

    def staleness_at(self, seq: int | None = None) -> list[dict]:
        q, args = "SELECT * FROM staleness", ()
        if seq is not None:
            q, args = q + " WHERE seq<=?", (seq,)
        return [_parse(r, ("detail",)) for r in self.db.execute(q + " ORDER BY seq,id", args)]

    def events_at(self, seq: int | None = None) -> list[dict]:
        q, args = "SELECT * FROM events", ()
        if seq is not None:
            q, args = q + " WHERE seq<=?", (seq,)
        return [_parse(r, ("payload",)) for r in self.db.execute(q + " ORDER BY seq", args)]

    def latest_comp(self, kind: str, seq: int | None = None) -> dict | None:
        q, args = "SELECT * FROM computations WHERE kind=?", (kind,)
        if seq is not None:
            q, args = q + " AND seq<=?", (kind, seq)
        row = self.db.execute(q + " ORDER BY seq DESC LIMIT 1", args).fetchone()
        return _parse(row, ("parents", "result")) if row else None
