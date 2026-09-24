"""SQLite 存储层。

分层原则：
- materials：不可变材料，按 kind 分开版本化（功率采样 / 注射体积和时间 /
  浓度 / 温度 / 池体积 / 基线 / 空白曲线）。每个版本保留 raw 原始表示。
- events：机器可读取事件，携带输入摘要、参数版本、父级依赖。
- results：计算结果，携带输入摘要、参数版本、父级依赖与分支。
- decisions：可修改的人工决定（接受 / 否定 / 参数调整），冲突决定并存。

所有表共享一个单调 seq，使任意 as_of 时刻的可见状态可重建。
"""
import json
import sqlite3
import threading

from .canon import canonical_json, digest
from .errors import FormatCorrupt, SemanticUnsolvable

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS materials(
  id TEXT PRIMARY KEY, kind TEXT, version INTEGER, seq INTEGER,
  raw TEXT, units TEXT, payload TEXT, digest TEXT);
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY, id TEXT, kind TEXT, status TEXT, branch TEXT,
  payload TEXT, parents TEXT, input_digest TEXT, param_version TEXT,
  committed INTEGER);
CREATE TABLE IF NOT EXISTS results(
  id TEXT PRIMARY KEY, seq INTEGER, kind TEXT, status TEXT, branch TEXT,
  payload TEXT, parents TEXT, input_digest TEXT, param_version TEXT);
CREATE TABLE IF NOT EXISTS decisions(
  seq INTEGER PRIMARY KEY, id TEXT, target TEXT, action TEXT,
  rationale TEXT, reviewer TEXT, branch TEXT);
"""

MATERIAL_KINDS = (
    "power_trace", "injection_schedule", "concentrations",
    "temperature", "cell_volume", "baseline", "blank_curve",
)


class Store:
    def __init__(self, path=":memory:"):
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self._lock = threading.RLock()
        if self.get_meta("seq") is None:
            self.set_meta("seq", "0")

    # ---- 基础 ----
    def close(self):
        self.db.close()

    def get_meta(self, key):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key, value):
        self.db.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        self.db.commit()

    def next_seq(self):
        with self._lock:
            seq = int(self.get_meta("seq")) + 1
            self.set_meta("seq", str(seq))
            return seq

    def is_empty(self):
        row = self.db.execute("SELECT COUNT(*) c FROM events").fetchone()
        return row["c"] == 0

    # ---- 材料（不可变、分版本）----
    def put_material(self, kind, raw, units, payload):
        if kind not in MATERIAL_KINDS:
            raise SemanticUnsolvable(f"未知材料类别 {kind!r}")
        with self._lock:
            row = self.db.execute(
                "SELECT MAX(version) v FROM materials WHERE kind=?", (kind,)).fetchone()
            version = (row["v"] or 0) + 1
            seq = self.next_seq()
            mid = f"mat-{kind}-v{version}"
            self.db.execute(
                "INSERT INTO materials VALUES(?,?,?,?,?,?,?,?)",
                (mid, kind, version, seq,
                 canonical_json(raw), canonical_json(units),
                 canonical_json(payload), digest(payload)))
            self.db.commit()
            return {"id": mid, "kind": kind, "version": version, "seq": seq,
                    "digest": digest(payload)}

    def get_material(self, kind, version=None, as_of=None):
        sql = "SELECT * FROM materials WHERE kind=?"
        args = [kind]
        if version is not None:
            sql += " AND version=?"
            args.append(version)
        if as_of is not None:
            sql += " AND seq<=?"
            args.append(as_of)
        sql += " ORDER BY version DESC LIMIT 1"
        row = self.db.execute(sql, args).fetchone()
        return self._mat_from_row(row) if row else None

    def list_materials(self, as_of=None):
        sql = "SELECT * FROM materials"
        args = []
        if as_of is not None:
            sql += " WHERE seq<=?"
            args.append(as_of)
        sql += " ORDER BY kind, version"
        return [self._mat_from_row(r) for r in self.db.execute(sql, args)]

    @staticmethod
    def _mat_from_row(row):
        return {"id": row["id"], "kind": row["kind"], "version": row["version"],
                "seq": row["seq"], "raw": json.loads(row["raw"]),
                "units": json.loads(row["units"]),
                "payload": json.loads(row["payload"]), "digest": row["digest"]}

    # ---- 事件 ----
    def add_event(self, kind, payload, parents=None, input_digest=None,
                  param_version=None, branch="main", status="ok"):
        with self._lock:
            seq = self.next_seq()
            eid = f"ev-{seq:05d}"
            self.db.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,0)",
                (seq, eid, kind, status, branch, canonical_json(payload),
                 canonical_json(parents or []), input_digest, param_version))
            self.db.execute("UPDATE events SET committed=1 WHERE seq=?", (seq,))
            self.db.commit()
            return {"seq": seq, "id": eid, "kind": kind, "status": status}

    def insert_uncommitted_event(self, kind, payload):
        """模拟半写崩溃：写入 committed=0 后不再提交标记。"""
        with self._lock:
            seq = self.next_seq()
            eid = f"ev-{seq:05d}"
            self.db.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,0)",
                (seq, eid, kind, "ok", "main", canonical_json(payload),
                 canonical_json([]), None, None))
            self.db.commit()
            return seq

    def recover(self):
        """半写恢复：把未提交事件标记为 aborted（committed=-1），返回清单。"""
        with self._lock:
            rows = self.db.execute(
                "SELECT seq, id, kind FROM events WHERE committed=0").fetchall()
            aborted = [{"seq": r["seq"], "id": r["id"], "kind": r["kind"]}
                       for r in rows]
            self.db.execute("UPDATE events SET committed=-1 WHERE committed=0")
            if aborted:
                seq = self.next_seq()
                self.db.execute(
                    "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,1)",
                    (seq, f"ev-{seq:05d}", "recovery", "ok", "main",
                     canonical_json({"aborted": aborted}), "[]", None, None))
            self.db.commit()
            return aborted

    def list_events(self, as_of=None, include_aborted=False):
        sql = "SELECT * FROM events"
        conds, args = [], []
        if as_of is not None:
            conds.append("seq<=?")
            args.append(as_of)
        if not include_aborted:
            conds.append("committed=1")
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY seq"
        return [self._event_from_row(r) for r in self.db.execute(sql, args)]

    def get_event(self, seq):
        row = self.db.execute("SELECT * FROM events WHERE seq=?", (seq,)).fetchone()
        return self._event_from_row(row) if row else None

    @staticmethod
    def _event_from_row(row):
        return {"seq": row["seq"], "id": row["id"], "kind": row["kind"],
                "status": row["status"], "branch": row["branch"],
                "payload": json.loads(row["payload"]),
                "parents": json.loads(row["parents"]),
                "input_digest": row["input_digest"],
                "param_version": row["param_version"],
                "committed": row["committed"]}

    # ---- 结果 ----
    def put_result(self, rid, kind, status, payload, parents=None,
                   input_digest=None, param_version=None, branch="main"):
        with self._lock:
            seq = self.next_seq()
            self.db.execute(
                "INSERT OR REPLACE INTO results VALUES(?,?,?,?,?,?,?,?,?)",
                (rid, seq, kind, status, branch, canonical_json(payload),
                 canonical_json(parents or []), input_digest, param_version))
            self.db.commit()
            return {"id": rid, "seq": seq}

    def list_results(self, kind=None, branch=None, as_of=None):
        sql = "SELECT * FROM results"
        conds, args = [], []
        if kind is not None:
            conds.append("kind=?")
            args.append(kind)
        if branch is not None:
            conds.append("branch=?")
            args.append(branch)
        if as_of is not None:
            conds.append("seq<=?")
            args.append(as_of)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY seq, id"
        return [self._result_from_row(r) for r in self.db.execute(sql, args)]

    def get_result(self, rid):
        row = self.db.execute("SELECT * FROM results WHERE id=?", (rid,)).fetchone()
        return self._result_from_row(row) if row else None

    @staticmethod
    def _result_from_row(row):
        return {"id": row["id"], "seq": row["seq"], "kind": row["kind"],
                "status": row["status"], "branch": row["branch"],
                "payload": json.loads(row["payload"]),
                "parents": json.loads(row["parents"]),
                "input_digest": row["input_digest"],
                "param_version": row["param_version"]}

    # ---- 决定（可修改层，冲突并存）----
    def add_decision(self, target, action, rationale, reviewer, branch="main"):
        with self._lock:
            seq = self.next_seq()
            did = f"dec-{seq:05d}"
            self.db.execute(
                "INSERT INTO decisions VALUES(?,?,?,?,?,?,?)",
                (seq, did, target, action, rationale, reviewer, branch))
            self.db.commit()
            return {"seq": seq, "id": did}

    def list_decisions(self, as_of=None, branch=None):
        sql = "SELECT * FROM decisions"
        conds, args = [], []
        if as_of is not None:
            conds.append("seq<=?")
            args.append(as_of)
        if branch is not None:
            conds.append("branch=?")
            args.append(branch)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY seq"
        return [{"seq": r["seq"], "id": r["id"], "target": r["target"],
                 "action": r["action"], "rationale": r["rationale"],
                 "reviewer": r["reviewer"], "branch": r["branch"]}
                for r in self.db.execute(sql, args)]

    # ---- 导出 / 导入 ----
    def export_bundle(self):
        return {
            "format": "itc-gsb/1",
            "meta": {"seq": self.get_meta("seq")},
            "materials": [
                {"id": r["id"], "kind": r["kind"], "version": r["version"],
                 "seq": r["seq"], "raw": json.loads(r["raw"]),
                 "units": json.loads(r["units"]),
                 "payload": json.loads(r["payload"]), "digest": r["digest"]}
                for r in self.db.execute("SELECT * FROM materials ORDER BY seq")],
            "events": [self._event_from_row(r) for r in self.db.execute(
                "SELECT * FROM events ORDER BY seq")],
            "results": [self._result_from_row(r) for r in self.db.execute(
                "SELECT * FROM results ORDER BY seq")],
            "decisions": self.list_decisions(),
        }

    def import_bundle(self, text):
        """事务性导入：任何失败整体回滚，库保持原状（半写安全）。"""
        try:
            bundle = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise FormatCorrupt(f"导入内容不是合法 JSON：{exc}")
        if not isinstance(bundle, dict) or bundle.get("format") != "itc-gsb/1":
            raise FormatCorrupt("缺少 format=itc-gsb/1 头，格式不被识别")
        for key in ("materials", "events", "results", "decisions"):
            if key not in bundle or not isinstance(bundle[key], list):
                raise FormatCorrupt(f"导出包缺少数组字段 {key!r}")
        if not self.is_empty():
            raise SemanticUnsolvable("目标库非空，拒绝导入以免混淆身份")
        with self._lock:
            try:
                self.db.execute("BEGIN")
                for m in bundle["materials"]:
                    self.db.execute(
                        "INSERT INTO materials VALUES(?,?,?,?,?,?,?,?)",
                        (m["id"], m["kind"], m["version"], m["seq"],
                         canonical_json(m["raw"]), canonical_json(m["units"]),
                         canonical_json(m["payload"]), m["digest"]))
                for e in bundle["events"]:
                    self.db.execute(
                        "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (e["seq"], e["id"], e["kind"], e["status"], e["branch"],
                         canonical_json(e["payload"]), canonical_json(e["parents"]),
                         e["input_digest"], e["param_version"], e["committed"]))
                for r in bundle["results"]:
                    self.db.execute(
                        "INSERT INTO results VALUES(?,?,?,?,?,?,?,?,?)",
                        (r["id"], r["seq"], r["kind"], r["status"], r["branch"],
                         canonical_json(r["payload"]), canonical_json(r["parents"]),
                         r["input_digest"], r["param_version"]))
                for d in bundle["decisions"]:
                    self.db.execute(
                        "INSERT INTO decisions VALUES(?,?,?,?,?,?,?)",
                        (d["seq"], d["id"], d["target"], d["action"],
                         d["rationale"], d["reviewer"], d["branch"]))
                self.db.execute(
                    "INSERT INTO meta(key,value) VALUES('seq',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(bundle["meta"]["seq"]),))
                self.db.commit()
            except (sqlite3.Error, KeyError, TypeError) as exc:
                self.db.rollback()
                raise FormatCorrupt(f"导入中途失败，已整体回滚：{exc}")
        return {"imported_events": len(bundle["events"]),
                "imported_results": len(bundle["results"])}
