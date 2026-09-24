"""SQLite 存储层：不可变材料 + 仅追加事件；每个提交自带 hash 链与依赖摘要。

半写恢复：所有写操作先登记 journal，提交后清除；崩溃后重开自动回滚未提交事务。
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager

from .canonicalid import canonical, cid, key_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects(
  pid TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS branches(
  pid TEXT NOT NULL, branch TEXT NOT NULL, head TEXT,
  PRIMARY KEY(pid, branch));
CREATE TABLE IF NOT EXISTS events(
  eid TEXT NOT NULL, pid TEXT NOT NULL, branch TEXT NOT NULL, seq INTEGER NOT NULL,
  parent TEXT, type TEXT NOT NULL, payload TEXT NOT NULL, hash TEXT NOT NULL,
  deps TEXT NOT NULL DEFAULT '[]', PRIMARY KEY(eid, branch));
CREATE TABLE IF NOT EXISTS materials(
  mid TEXT PRIMARY KEY, pid TEXT NOT NULL, stream TEXT NOT NULL, version INTEGER NOT NULL,
  kind TEXT NOT NULL, source TEXT NOT NULL, payload TEXT NOT NULL, diagnostics TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS journal(op TEXT PRIMARY KEY, detail TEXT);
CREATE INDEX IF NOT EXISTS idx_ev_branch ON events(pid, branch, seq);
CREATE INDEX IF NOT EXISTS idx_mat_pid ON materials(pid, stream, version);
"""

# 故障注入钩子（测试半写恢复）；设为抛异常的 callable
FAULT_HOOK = None


class Store:
    def __init__(self, path: str = "itc.db"):
        self.path = path
        fresh = not os.path.exists(path)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        with self.conn:
            self.conn.executescript(SCHEMA)
        self._recover(fresh)

    def _recover(self, fresh: bool) -> None:
        rows = self.conn.execute("SELECT op, detail FROM journal").fetchall()
        for r in rows:
            if r["op"] == "txn":
                self.conn.rollback()
        self.conn.execute("DELETE FROM journal")
        self.conn.commit()
        self.had_half_write = bool(rows) and not fresh

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def txn(self):
        self.conn.execute("INSERT OR REPLACE INTO journal(op,detail) VALUES('txn','open')")
        self.conn.commit()
        try:
            yield self.conn
        except Exception:
            self.conn.rollback()
            self.conn.execute("DELETE FROM journal")
            self.conn.commit()
            raise
        else:
            if FAULT_HOOK is not None:
                FAULT_HOOK()
            self.conn.commit()
            self.conn.execute("DELETE FROM journal")
            self.conn.commit()

    # ---- 项目 / 分支 ----
    def create_project(self, name: str, at: str) -> str:
        pid = cid("proj", {"name": name, "created_at": at})
        with self.txn() as c:
            c.execute("INSERT OR IGNORE INTO projects(pid,name,created_at) VALUES(?,?,?)",
                      (pid, name, at))
            c.execute("INSERT OR IGNORE INTO branches(pid,branch,head) VALUES(?,?,NULL)",
                      (pid, "main"))
        return pid

    def list_projects(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM projects ORDER BY name")]

    def get_project(self, pid: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM projects WHERE pid=?", (pid,)).fetchone()
        return dict(r) if r else None

    def list_branches(self, pid: str) -> list[str]:
        return [r["branch"] for r in self.conn.execute(
            "SELECT branch FROM branches WHERE pid=? ORDER BY branch", (pid,))]

    def branch_head(self, pid: str, branch: str) -> str | None:
        r = self.conn.execute("SELECT head FROM branches WHERE pid=? AND branch=?",
                              (pid, branch)).fetchone()
        return r["head"] if r else None

    def create_branch(self, pid: str, branch: str, from_head: str) -> None:
        with self.txn() as c:
            c.execute("INSERT INTO branches(pid,branch,head) VALUES(?,?,?)",
                      (pid, branch, from_head))

    # ---- 材料（内容寻址、全局去重；同一项目内按 stream 递增版本号） ----
    def put_material(self, pid: str, stream: str, kind: str, source: str,
                     payload: dict, diagnostics: list[dict]) -> dict:
        mid = cid("mat", {"stream": stream, "kind": kind, "source": source,
                          "payload": payload, "diagnostics": diagnostics})
        with self.txn() as c:
            existing = c.execute(
                "SELECT version FROM materials WHERE mid=?", (mid,)).fetchone()
            if existing:
                return {"id": mid, "version": existing["version"], "deduplicated": True}
            row = c.execute(
                "SELECT COALESCE(MAX(version),0)+1 AS v FROM materials "
                "WHERE pid=? AND stream=?", (pid, stream)).fetchone()
            version = row["v"]
            c.execute("INSERT INTO materials(mid,pid,stream,version,kind,source,payload,diagnostics)"
                      " VALUES(?,?,?,?,?,?,?,?)",
                      (mid, pid, stream, version, kind, source,
                       canonical(payload), canonical(diagnostics)))
        return {"id": mid, "version": version, "deduplicated": False}

    def get_material(self, mid: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM materials WHERE mid=?", (mid,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        d["diagnostics"] = json.loads(d["diagnostics"])
        return d

    def material_versions(self, pid: str, stream: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT mid,version,kind,source FROM materials WHERE pid=? AND stream=? "
            "ORDER BY version", (pid, stream)).fetchall()
        return [dict(r) for r in rows]

    def all_materials(self, pid: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT mid,pid,stream,version,kind,source,payload,diagnostics FROM materials "
            "WHERE pid=? ORDER BY stream,version", (pid,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            d["diagnostics"] = json.loads(d["diagnostics"])
            out.append(d)
        return out

    # ---- 仅追加事件 ----
    def append_event(self, pid: str, branch: str, etype: str, payload: dict,
                     deps: list[str] | None = None) -> dict:
        deps = deps or []
        with self.txn() as c:
            head = c.execute("SELECT head FROM branches WHERE pid=? AND branch=?",
                             (pid, branch)).fetchone()["head"]
            n = c.execute("SELECT COALESCE(MAX(seq),0)+1 AS n FROM events "
                          "WHERE pid=? AND branch=?", (pid, branch)).fetchone()["n"]
            body = {"type": etype, "payload": payload, "parent": head,
                    "seq": n, "pid": pid, "branch": branch}
            eid = cid("evt", body)
            prev_hash = c.execute("SELECT hash FROM events WHERE eid=? AND branch=?",
                                  (head, branch)).fetchone() if head else None
            h = key_hash({"body": body,
                          "prev": prev_hash["hash"] if prev_hash else None,
                          "deps": deps})
            c.execute("INSERT INTO events(eid,pid,branch,seq,parent,type,payload,hash,deps)"
                      " VALUES(?,?,?,?,?,?,?,?,?)",
                      (eid, pid, branch, n, head, etype, canonical(payload), h,
                       canonical(deps)))
            c.execute("UPDATE branches SET head=? WHERE pid=? AND branch=?",
                      (eid, pid, branch))
        return {"id": eid, "seq": n, "parent": head}

    def branch_events(self, pid: str, branch: str, upto_seq: int | None = None) -> list[dict]:
        q = "SELECT eid,pid,branch,seq,parent,type,payload,hash,deps FROM events WHERE pid=? AND branch=?"
        args: list = [pid, branch]
        if upto_seq is not None:
            q += " AND seq<=?"
            args.append(upto_seq)
        q += " ORDER BY seq"
        rows = self.conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    def get_event(self, eid: str, branch: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM events WHERE eid=? AND branch=?",
                              (eid, branch)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        return d

    def verify_chain(self, pid: str, branch: str) -> list[str]:
        """重算 hash 链，返回被破坏的事件 eid 列表（空=完整）。"""
        broken: list[str] = []
        prev_hash = None
        for e in self.branch_events(pid, branch):
            body = {"type": e["type"], "payload": e["payload"], "parent": e["parent"],
                    "seq": e["seq"], "pid": pid, "branch": branch}
            deps = json.loads(e["deps"])
            h = key_hash({"body": body, "prev": prev_hash, "deps": deps})
            if h != e["hash"]:
                broken.append(e["eid"])
            prev_hash = e["hash"]
        return broken

    def raw_insert_event(self, pid, branch, seq, eid, parent, etype, payload_canon, h,
                         deps_canon="[]"):
        """仅供导入/破坏测试使用。"""
        with self.txn() as c:
            c.execute("INSERT INTO events(eid,pid,branch,seq,parent,type,payload,hash,deps)"
                      " VALUES(?,?,?,?,?,?,?,?,?)",
                      (eid, pid, branch, seq, parent, etype, payload_canon, h, deps_canon))
            c.execute("UPDATE branches SET head=? WHERE pid=? AND branch=?",
                      (eid, pid, branch))
