"""SQLite 事件溯源存储。

分层：
* materials        不可变材料版本（功率、注射时序、浓度、温度、池体积、基线、空白分开版本化）
* results          派生结果（积分/稀释/拟合），带输入摘要、参数版本与父级依赖
* events           事件流（分支 + 父事件构成有向无环链，fork 后继承历史）
* branches         分支头
所有 id 由内容决定；同一分支重复导入相同内容是幂等的。
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager

from .identity import stable_id

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL,
    title TEXT
);
CREATE TABLE IF NOT EXISTS branches (
    branch_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    name TEXT NOT NULL,
    parent_branch_id TEXT,
    fork_event_id TEXT,
    head_event_id TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    parent_event_id TEXT,
    type TEXT NOT NULL,
    timestamp REAL NOT NULL,
    actor TEXT NOT NULL,
    payload TEXT NOT NULL,
    material_id TEXT,
    result_id TEXT,
    PRIMARY KEY (branch_id, seq)
);
CREATE TABLE IF NOT EXISTS materials (
    material_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    lineage TEXT NOT NULL,
    version INTEGER NOT NULL,
    branch_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    parsed TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE (case_id, kind, lineage, content_hash)
);
CREATE TABLE IF NOT EXISTS results (
    result_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    content TEXT NOT NULL,
    inputs_digest TEXT NOT NULL,
    parents TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS import_journals (
    journal_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    finished_at REAL,
    tmp_path TEXT,
    final_path TEXT,
    status TEXT NOT NULL,
    detail TEXT
);
"""


class Store:
    def __init__(self, path: str = "itc_lab.db"):
        self.path = path
        self._init()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init(self):
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")

    # ---------- cases / branches ----------
    def create_case(self, case_id: str, name: str, title: str = "",
                    created_at: float | None = None) -> str:
        ts = created_at if created_at is not None else time.time()
        branch_id = f"br_{case_id}_main"
        with self.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO cases VALUES (?,?,?,?)",
                         (case_id, name, ts, title))
            conn.execute(
                "INSERT OR IGNORE INTO branches VALUES (?,?,?,?,?,?,?)",
                (branch_id, case_id, "main", None, None, None, ts))
        return branch_id

    def list_cases(self) -> list[dict]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM events e WHERE e.branch_id=b.branch_id) AS event_count "
                "FROM cases c JOIN branches b ON b.case_id=c.case_id "
                "WHERE b.parent_branch_id IS NULL ORDER BY c.created_at")]

    def create_branch(self, case_id: str, name: str, parent_branch_id: str,
                      fork_event_id: str | None, created_at: float | None = None) -> str:
        ts = created_at if created_at is not None else time.time()
        branch_id = stable_id("br", {"case": case_id, "name": name,
                                     "parent": parent_branch_id, "fork": fork_event_id})
        with self.connect() as conn:
            exists = conn.execute("SELECT 1 FROM branches WHERE branch_id=?", (branch_id,)).fetchone()
            if not exists:
                conn.execute("INSERT INTO branches VALUES (?,?,?,?,?,?,?)",
                             (branch_id, case_id, name, parent_branch_id, fork_event_id,
                              fork_event_id, ts))
        return branch_id

    def list_branches(self, case_id: str) -> list[dict]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM branches WHERE case_id=? ORDER BY created_at", (case_id,))]

    def branch_chain(self, branch_id: str) -> list[str]:
        chain: list[str] = []
        with self.connect() as conn:
            cur = branch_id
            while cur:
                chain.append(cur)
                row = conn.execute("SELECT parent_branch_id, fork_event_id FROM branches WHERE branch_id=?",
                                   (cur,)).fetchone()
                if row is None:
                    break
                cur = row["parent_branch_id"]
        return chain

    # ---------- materials ----------
    def put_material(self, case_id: str, kind: str, lineage: str, branch_id: str,
                     raw_text: str, parsed: dict, content_hash: str,
                     created_at: float | None = None) -> tuple[str, int, bool]:
        ts = created_at if created_at is not None else time.time()
        material_id = f"mat_{kind}_{content_hash[:12]}"
        with self.connect() as conn:
            row = conn.execute(
                "SELECT material_id, version FROM materials WHERE case_id=? AND kind=? AND lineage=? "
                "ORDER BY version DESC LIMIT 1", (case_id, kind, lineage)).fetchone()
            same = conn.execute(
                "SELECT material_id, version FROM materials WHERE case_id=? AND kind=? AND lineage=? "
                "AND content_hash=?", (case_id, kind, lineage, content_hash)).fetchone()
            if same:
                return same["material_id"], same["version"], False
            version = (row["version"] + 1) if row else 1
            conn.execute("INSERT INTO materials VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (material_id, case_id, kind, lineage, version, branch_id,
                          content_hash, raw_text, json.dumps(parsed, ensure_ascii=False), ts))
        return material_id, version, True

    def get_material(self, material_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM materials WHERE material_id=?", (material_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["parsed"] = json.loads(d["parsed"])
            return d

    # ---------- results ----------
    def put_result(self, result_id: str, case_id: str, kind: str, branch_id: str,
                   content: dict, inputs_digest: str, parents: list[str],
                   created_at: float | None = None) -> bool:
        ts = created_at if created_at is not None else time.time()
        with self.connect() as conn:
            exists = conn.execute("SELECT 1 FROM results WHERE result_id=?", (result_id,)).fetchone()
            if exists:
                return False
            conn.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?,?)",
                         (result_id, case_id, kind, branch_id,
                          json.dumps(content, ensure_ascii=False), inputs_digest,
                          json.dumps(parents), ts))
            return True

    def get_result(self, result_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM results WHERE result_id=?", (result_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["content"] = json.loads(d["content"])
            d["parents"] = json.loads(d["parents"])
            return d

    # ---------- events ----------
    def append_event(self, case_id: str, branch_id: str, etype: str, payload: dict,
                     actor: str = "reviewer", material_id: str | None = None,
                     result_id: str | None = None,
                     event_id: str | None = None, ts: float | None = None) -> dict:
        timestamp = ts if ts is not None else time.time()
        with self.connect() as conn:
            head = conn.execute("SELECT head_event_id, COALESCE(MAX(seq),0) AS m FROM branches b "
                                "LEFT JOIN events e ON e.branch_id=b.branch_id "
                                "WHERE b.branch_id=? GROUP BY b.branch_id", (branch_id,)).fetchone()
            if head is None:
                raise KeyError(f"未知分支 {branch_id}")
            seq = head["m"] + 1
            parent = head["head_event_id"]
            eid = event_id or stable_id("evt", {"branch": branch_id, "seq": seq,
                                                "type": etype, "payload": payload})
            conn.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (eid, branch_id, seq, parent, etype, timestamp, actor,
                          json.dumps(payload, ensure_ascii=False), material_id, result_id))
            conn.execute("UPDATE branches SET head_event_id=? WHERE branch_id=?", (eid, branch_id))
        return {"event_id": eid, "branch_id": branch_id, "seq": seq,
                "parent_event_id": parent, "type": etype, "timestamp": timestamp,
                "payload": payload}

    def timeline(self, branch_id: str) -> list[dict]:
        """分支可见事件 = 本分支全部事件 + 祖先分支在 fork 点（含）之前的事件，按时间/拓扑顺序。"""
        chain = self.branch_chain(branch_id)
        events: list[dict] = []
        with self.connect() as conn:
            for i, bid in enumerate(chain):
                if i == 0:
                    rows = conn.execute(
                        "SELECT * FROM events WHERE branch_id=? ORDER BY seq", (bid,)).fetchall()
                else:
                    child = chain[i - 1]
                    fk = conn.execute("SELECT fork_event_id FROM branches WHERE branch_id=?",
                                      (child,)).fetchone()["fork_event_id"]
                    rows = conn.execute(
                        "SELECT * FROM events WHERE branch_id=? AND seq<=COALESCE("
                        "(SELECT seq FROM events WHERE event_id=?), 0) ORDER BY seq",
                        (bid, fk)).fetchall() if fk else []
                for r in rows:
                    d = dict(r)
                    d["payload"] = json.loads(d["payload"])
                    events.append(d)
        events.sort(key=lambda e: (e["timestamp"], e["branch_id"], e["seq"]))
        return events

    def event_by_id(self, event_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["payload"] = json.loads(d["payload"])
            return d

    def head(self, branch_id: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT head_event_id FROM branches WHERE branch_id=?",
                               (branch_id,)).fetchone()
            return row["head_event_id"] if row else None

    # ---------- export / import ----------
    def dump_case(self, case_id: str) -> dict:
        with self.connect() as conn:
            case = dict(conn.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone())
            branches = [dict(r) for r in conn.execute(
                "SELECT * FROM branches WHERE case_id=? ORDER BY created_at", (case_id,))]
            bset = [b["branch_id"] for b in branches]
            qmarks = ",".join("?" * len(bset))
            materials = [dict(r) for r in conn.execute(
                f"SELECT * FROM materials WHERE case_id=? ORDER BY kind, lineage, version", (case_id,))]
            results = [dict(r) for r in conn.execute(
                f"SELECT * FROM results WHERE case_id=? ORDER BY created_at", (case_id,))]
            events = [dict(r) for r in conn.execute(
                f"SELECT * FROM events WHERE branch_id IN ({qmarks}) ORDER BY timestamp, branch_id, seq",
                bset)]
        for m in materials:
            m["parsed"] = json.loads(m["parsed"])
        for r in results:
            r["content"] = json.loads(r["content"])
            r["parents"] = json.loads(r["parents"])
        for e in events:
            e["payload"] = json.loads(e["payload"])
        return {"format": "itc-case/1", "case": case, "branches": branches,
                "materials": materials, "results": results, "events": events}

    def load_case(self, bundle: dict, journal_id: str, tmp_path: str, final_path: str,
                  update_journal) -> dict:
        """单事务导入；任何不一致整体回滚（半写恢复：journal 保留 failed 状态）。"""
        if bundle.get("format") != "itc-case/1":
            raise ValueError("bundle format 不是 itc-case/1")
        with self.connect() as conn:
            try:
                case = bundle["case"]
                conn.execute("INSERT OR REPLACE INTO cases VALUES (?,?,?,?)",
                             (case["case_id"], case["name"], case["created_at"], case.get("title", "")))
                for b in bundle["branches"]:
                    conn.execute("INSERT OR REPLACE INTO branches VALUES (?,?,?,?,?,?,?)",
                                 (b["branch_id"], b["case_id"], b["name"],
                                  b.get("parent_branch_id"), b.get("fork_event_id"),
                                  b.get("head_event_id"), b["created_at"]))
                for m in bundle["materials"]:
                    conn.execute("INSERT OR REPLACE INTO materials VALUES (?,?,?,?,?,?,?,?,?,?)",
                                 (m["material_id"], m["case_id"], m["kind"], m["lineage"],
                                  m["version"], m["branch_id"], m["content_hash"], m["raw_text"],
                                  json.dumps(m["parsed"], ensure_ascii=False), m["created_at"]))
                for r in bundle["results"]:
                    conn.execute("INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?)",
                                 (r["result_id"], r["case_id"], r["kind"], r["branch_id"],
                                  json.dumps(r["content"], ensure_ascii=False),
                                  r["inputs_digest"], json.dumps(r["parents"]),
                                  r["created_at"]))
                for e in bundle["events"]:
                    conn.execute("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,?,?)",
                                 (e["event_id"], e["branch_id"], e["seq"],
                                  e.get("parent_event_id"), e["type"], e["timestamp"],
                                  e.get("actor", "reviewer"),
                                  json.dumps(e["payload"], ensure_ascii=False),
                                  e.get("material_id"), e.get("result_id")))
            except Exception:
                update_journal(conn, journal_id, "failed", "事务回滚")
                raise
            update_journal(conn, journal_id, "committed", "导入完成")
        return {"case_id": bundle["case"]["case_id"],
                "events": len(bundle["events"]),
                "materials": len(bundle["materials"]),
                "results": len(bundle["results"])}
