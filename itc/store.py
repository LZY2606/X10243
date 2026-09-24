"""SQLite 事件溯源存储。

分层：
- 不可变材料（material_versions）：功率采样、注射体积和时间、浓度、
  温度、池体积、基线、空白曲线，每条流分开版本化，版本只增不改。
- 事件（events）：所有导入 / 计算 / 决定都是追加式机器可读事件。
- 结果（results + result_status）：计算结果记录输入摘要、参数版本
  与父级依赖；状态变化（accepted/stale/...）本身也是事件。
- 分支（branches）：从某个事件序号分叉，互不改写历史。

任意旧事件可回放：state(asof=seq) 恢复当时的依赖版本与可见结论。
"""

import hashlib
import json
import sqlite3
import threading
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS events(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL,
    branch TEXT NOT NULL DEFAULT 'main',
    type TEXT NOT NULL,
    payload TEXT NOT NULL,
    parents TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS material_versions(
    stream TEXT NOT NULL,
    version INTEGER NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    seq INTEGER NOT NULL,
    PRIMARY KEY(stream, version)
);
CREATE TABLE IF NOT EXISTS results(
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    parents TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    params TEXT NOT NULL,
    created_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS result_status(
    result_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    PRIMARY KEY(result_id, seq)
);
CREATE TABLE IF NOT EXISTS branches(
    name TEXT PRIMARY KEY,
    base_seq INTEGER NOT NULL
);
"""

STREAM_KINDS = {
    "power": "功率采样",
    "injections": "注射体积和时间",
    "concentration": "浓度",
    "temperature": "温度",
    "cell_volume": "池体积",
    "baseline": "基线",
    "blank": "空白曲线",
}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def new_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Store:
    def __init__(self, path=":memory:"):
        self.path = path
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self):
        self.db.close()

    # ---------------- 事件 ----------------
    def append_event(self, type_, payload, parents=(), branch="main", event_id=None):
        eid = event_id or new_id("ev")
        with self._lock, self.db:
            cur = self.db.execute(
                "INSERT INTO events(id,branch,type,payload,parents) VALUES(?,?,?,?,?)",
                (eid, branch, type_, canonical(payload), canonical(list(parents))),
            )
            seq = cur.lastrowid
        return {"seq": seq, "id": eid, "branch": branch, "type": type_,
                "payload": payload, "parents": list(parents)}

    def events(self, branch=None, asof=None):
        q = "SELECT * FROM events WHERE 1=1"
        args = []
        if asof is not None:
            q += " AND seq<=?"
            args.append(asof)
        if branch is not None:
            base = self.branch_base(branch)
            q += " AND (branch=? OR seq<=?)"
            args += [branch, base]
        q += " ORDER BY seq"
        rows = self.db.execute(q, args).fetchall()
        return [self._row_to_event(r) for r in rows]

    @staticmethod
    def _row_to_event(r):
        return {"seq": r["seq"], "id": r["id"], "branch": r["branch"],
                "type": r["type"], "payload": json.loads(r["payload"]),
                "parents": json.loads(r["parents"])}

    def max_seq(self):
        row = self.db.execute("SELECT COALESCE(MAX(seq),0) AS m FROM events").fetchone()
        return row["m"]

    def is_empty(self):
        return self.max_seq() == 0

    # ---------------- 材料 ----------------
    def put_material(self, stream, payload, seq, kind=None):
        kind = kind or STREAM_KINDS.get(stream, stream)
        with self._lock, self.db:
            row = self.db.execute(
                "SELECT COALESCE(MAX(version),0) AS v FROM material_versions WHERE stream=?",
                (stream,)).fetchone()
            version = row["v"] + 1
            self.db.execute(
                "INSERT INTO material_versions(stream,version,kind,payload,seq) VALUES(?,?,?,?,?)",
                (stream, version, kind, canonical(payload), seq))
        return version

    def get_material(self, stream, version=None, asof=None):
        q = "SELECT * FROM material_versions WHERE stream=?"
        args = [stream]
        if version is not None:
            q += " AND version=?"
            args.append(version)
        if asof is not None:
            q += " AND seq<=?"
            args.append(asof)
        q += " ORDER BY version DESC LIMIT 1"
        row = self.db.execute(q, args).fetchone()
        if row is None:
            return None
        return {"stream": row["stream"], "version": row["version"],
                "kind": row["kind"], "payload": json.loads(row["payload"]),
                "seq": row["seq"], "digest": digest(json.loads(row["payload"]))}

    def material_versions(self, stream):
        rows = self.db.execute(
            "SELECT version,seq FROM material_versions WHERE stream=? ORDER BY version",
            (stream,)).fetchall()
        return [{"version": r["version"], "seq": r["seq"]} for r in rows]

    # ---------------- 结果 ----------------
    def put_result(self, kind, payload, parents, params, seq, result_id=None,
                   status="candidate"):
        rid = result_id or new_id(kind)
        input_digest = digest({"parents": parents, "params": params})
        with self._lock, self.db:
            self.db.execute(
                "INSERT INTO results(id,kind,payload,parents,input_digest,params,created_seq)"
                " VALUES(?,?,?,?,?,?,?)",
                (rid, kind, canonical(payload), canonical(parents), input_digest,
                 canonical(params), seq))
            self.db.execute(
                "INSERT INTO result_status(result_id,seq,status,reason) VALUES(?,?,?,?)",
                (rid, seq, status, "created"))
        return rid

    def set_result_status(self, result_id, status, reason, seq):
        with self._lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO result_status(result_id,seq,status,reason)"
                " VALUES(?,?,?,?)", (result_id, seq, status, reason))

    def get_result(self, result_id):
        row = self.db.execute("SELECT * FROM results WHERE id=?", (result_id,)).fetchone()
        return self._row_to_result(row) if row else None

    def results(self, kind=None, asof=None, branch=None):
        q = "SELECT * FROM results WHERE 1=1"
        args = []
        if kind:
            q += " AND kind=?"
            args.append(kind)
        if asof is not None:
            q += " AND created_seq<=?"
            args.append(asof)
        rows = self.db.execute(q, args).fetchall()
        out = []
        for r in rows:
            item = self._row_to_result(r)
            item["status"], item["status_reason"] = self._status_asof(r["id"], asof)
            out.append(item)
        # 等价输入得到稳定顺序：按 (创建序号, id) 排序。
        out.sort(key=lambda x: (x["created_seq"], x["id"]))
        return out

    @staticmethod
    def _row_to_result(r):
        return {"id": r["id"], "kind": r["kind"], "payload": json.loads(r["payload"]),
                "parents": json.loads(r["parents"]), "input_digest": r["input_digest"],
                "params": json.loads(r["params"]), "created_seq": r["created_seq"]}

    def _status_asof(self, result_id, asof):
        q = "SELECT status,reason FROM result_status WHERE result_id=?"
        args = [result_id]
        if asof is not None:
            q += " AND seq<=?"
            args.append(asof)
        q += " ORDER BY seq DESC LIMIT 1"
        row = self.db.execute(q, args).fetchone()
        return (row["status"], row["reason"]) if row else ("candidate", None)

    # ---------------- 分支 ----------------
    def create_branch(self, name, base_seq):
        with self._lock, self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO branches(name,base_seq) VALUES(?,?)",
                (name, base_seq))

    def branch_base(self, name):
        if name == "main":
            return self.max_seq()
        row = self.db.execute("SELECT base_seq FROM branches WHERE name=?",
                              (name,)).fetchone()
        return row["base_seq"] if row else 0

    def branches(self):
        rows = self.db.execute("SELECT * FROM branches ORDER BY name").fetchall()
        out = [{"name": "main", "base_seq": 0}]
        out += [{"name": r["name"], "base_seq": r["base_seq"]} for r in rows]
        return out

    # ---------------- 状态视图 ----------------
    def state(self, asof=None, branch="main"):
        """恢复某个事件序号 / 分支下当时可见的依赖版本与结论。"""
        asof = self.max_seq() if asof is None else asof
        base = self.branch_base(branch)
        streams = {}
        for stream in STREAM_KINDS:
            m = self.get_material(stream, asof=asof)
            if m:
                streams[stream] = {k: m[k] for k in ("version", "kind", "digest")}
                streams[stream]["payload"] = m["payload"]
        events = self.events(branch=branch, asof=asof)
        visible_seqs = {e["seq"] for e in events}
        results = [r for r in self.results(asof=asof)
                   if r["created_seq"] in visible_seqs or r["created_seq"] <= base]
        return {"asof": asof, "branch": branch, "streams": streams,
                "results": results, "events": events,
                "branches": self.branches()}
