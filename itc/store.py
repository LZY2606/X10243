"""SQLite 存储层 + 事件溯源 + 两阶段半写恢复 + 导出/导入。

所有业务对象内容寻址，身份不依赖自增主键；因此导出 -> 删库 -> 导入后，
身份、顺序与来源关系保持。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any

from .errors import FormatCorrupt
from .identity import canonical, digest, event_id

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS materials (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  source_ref TEXT NOT NULL,
  revision INTEGER NOT NULL,
  name TEXT NOT NULL,
  payload TEXT NOT NULL,
  op_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rules (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  rule_set TEXT NOT NULL,
  version TEXT NOT NULL,
  payload TEXT NOT NULL,
  op_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  op_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  id TEXT NOT NULL,
  project_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  branch TEXT NOT NULL,
  action TEXT NOT NULL,
  payload TEXT NOT NULL,
  parent_event_id TEXT,
  op_id TEXT NOT NULL,
  PRIMARY KEY (project_id, branch, seq)
);
CREATE TABLE IF NOT EXISTS branches (
  project_id TEXT NOT NULL,
  name TEXT NOT NULL,
  head_event_id TEXT,
  op_id TEXT NOT NULL,
  PRIMARY KEY (project_id, name)
);
CREATE TABLE IF NOT EXISTS decisions (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  payload TEXT NOT NULL,
  op_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS op_log (
  op_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  state TEXT NOT NULL,
  created_at REAL NOT NULL
);
"""


class Store:
    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        self.recovery: list[dict[str, Any]] = []
        self._recover()

    def _init_schema(self) -> None:
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # ---------------------------------------------------------- 恢复
    def _recover(self) -> None:
        """半写恢复：所有写入绑定 op_id，只有 COMMITTED 的 op 生效；
        PENDING 的 op 在重新打开时回滚（删除其全部行）。"""
        with self.lock:
            rows = self.conn.execute(
                "SELECT op_id, project_id, state FROM op_log WHERE state != 'COMMITTED'"
            ).fetchall()
            for r in rows:
                op_id, pid, state = r["op_id"], r["project_id"], r["state"]
                for tbl in ("materials", "rules", "results", "events", "branches", "decisions"):
                    self.conn.execute(f"DELETE FROM {tbl} WHERE op_id=?", (op_id,))
                self.conn.execute("DELETE FROM op_log WHERE op_id=?", (op_id,))
                self.recovery.append({"op_id": op_id, "project_id": pid, "aborted_state": state})
            self.conn.commit()

    def begin_op(self, project_id: str, op_id: str | None = None) -> str:
        import time
        op_id = op_id or digest("op", [project_id, time.time_ns(), os.urandom(8).hex()])
        with self.lock:
            self.conn.execute(
                "INSERT INTO op_log(op_id, project_id, state, created_at) VALUES (?,?,?,?)",
                (op_id, project_id, "PENDING", time.time()),
            )
            self.conn.commit()
        return op_id

    def commit_op(self, op_id: str) -> None:
        with self.lock:
            self.conn.execute("UPDATE op_log SET state='COMMITTED' WHERE op_id=?", (op_id,))
            self.conn.commit()

    def crash_op(self, op_id: str) -> None:
        """测试钩子：模拟进程在 COMMITTED 前崩溃（op 停在 PENDING）。"""
        self.conn.close()
        # 重新打开即触发恢复
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._recover()

    # ---------------------------------------------------------- 写入
    def put_material(self, project_id: str, op_id: str, mat: dict[str, Any]) -> None:
        with self.lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO materials
                   (id, project_id, kind, source_ref, revision, name, payload, op_id)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (mat["id"], project_id, mat["kind"], mat["source_ref"], mat["revision"],
                 mat["name"], canonical(mat), op_id),
            )

    def put_rule(self, project_id: str, op_id: str, rule_set: str, version: str,
                 rid: str, payload: dict[str, Any]) -> None:
        with self.lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO rules
                   (id, project_id, rule_set, version, payload, op_id)
                   VALUES (?,?,?,?,?,?)""",
                (rid, project_id, rule_set, version, canonical(payload), op_id),
            )

    def put_result(self, project_id: str, op_id: str, result: dict[str, Any]) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO results (id, project_id, kind, payload, op_id) VALUES (?,?,?,?,?)",
                (result["id"], project_id, result["kind"], canonical(result), op_id),
            )

    def next_seq(self, project_id: str, branch: str) -> int:
        with self.lock:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(seq),0)+1 AS s FROM events WHERE project_id=? AND branch=?",
                (project_id, branch),
            ).fetchone()
            return int(row["s"])

    def append_event(self, project_id: str, op_id: str, branch: str, action: str,
                     payload: dict[str, Any], parent_event_id: str | None) -> dict[str, Any]:
        with self.lock:
            seq = self.next_seq(project_id, branch)
            eid = event_id(seq, branch, action, payload, parent_event_id)
            self.conn.execute(
                """INSERT INTO events
                   (id, project_id, seq, branch, action, payload, parent_event_id, op_id)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (eid, project_id, seq, branch, action, canonical(payload), parent_event_id, op_id),
            )
            self.conn.execute(
                """INSERT OR REPLACE INTO branches (project_id, name, head_event_id, op_id)
                   VALUES (?,?,?,?)""",
                (project_id, branch, eid, op_id),
            )
            return {"id": eid, "seq": seq, "branch": branch, "action": action,
                    "payload": payload, "parent_event_id": parent_event_id}

    def put_decision(self, project_id: str, op_id: str, event: dict[str, Any],
                     payload: dict[str, Any]) -> None:
        did = digest("dec", [event["id"], payload])
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO decisions (id, project_id, event_id, payload, op_id) VALUES (?,?,?,?,?)",
                (did, project_id, event["id"], canonical(payload), op_id),
            )

    # ---------------------------------------------------------- 读取
    def list_projects(self) -> list[str]:
        with self.lock:
            rows = self.conn.execute("SELECT DISTINCT project_id FROM materials").fetchall()
            return [r["project_id"] for r in rows]

    def materials(self, project_id: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT payload FROM materials WHERE project_id=?", (project_id,)).fetchall()
            return [json.loads(r["payload"]) for r in rows]

    def material_by_id(self, mid: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute("SELECT payload FROM materials WHERE id=?", (mid,)).fetchone()
            return json.loads(row["payload"]) if row else None

    def results(self, project_id: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT payload FROM results WHERE project_id=?", (project_id,)).fetchall()
            return [json.loads(r["payload"]) for r in rows]

    def result_by_id(self, rid: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute("SELECT payload FROM results WHERE id=?", (rid,)).fetchone()
            return json.loads(row["payload"]) if row else None

    def rules(self, project_id: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT payload FROM rules WHERE project_id=?", (project_id,)).fetchall()
            return [json.loads(r["payload"]) for r in rows]

    def events(self, project_id: str, branch: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            if branch is None:
                rows = self.conn.execute(
                    "SELECT * FROM events WHERE project_id=? ORDER BY branch, seq",
                    (project_id,)).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM events WHERE project_id=? AND branch=? ORDER BY seq",
                    (project_id, branch)).fetchall()
            out = []
            for r in rows:
                out.append({"id": r["id"], "seq": r["seq"], "branch": r["branch"],
                            "action": r["action"], "payload": json.loads(r["payload"]),
                            "parent_event_id": r["parent_event_id"]})
            return out

    def branches(self, project_id: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM branches WHERE project_id=?", (project_id,)).fetchall()
            return [{"name": r["name"], "head_event_id": r["head_event_id"]} for r in rows]

    def decisions(self, project_id: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM decisions WHERE project_id=?", (project_id,)).fetchall()
            return [{"id": r["id"], "event_id": r["event_id"],
                     "payload": json.loads(r["payload"])} for r in rows]

    # ---------------------------------------------------------- 导出 / 导入
    def export_project(self, project_id: str) -> dict[str, Any]:
        return {
            "export_format": "itc-project/1.0",
            "project_id": project_id,
            "materials": sorted(self.materials(project_id), key=lambda m: (m["kind"], m["revision"])),
            "rules": sorted(self.rules(project_id), key=lambda r: r.get("id", "")),
            "results": sorted(self.results(project_id), key=lambda r: r["id"]),
            "events": self.events(project_id),
            "branches": sorted(self.branches(project_id), key=lambda b: b["name"]),
            "decisions": sorted(self.decisions(project_id), key=lambda d: d["id"]),
        }

    def import_project(self, data: dict[str, Any], *, expect_id: str | None = None) -> str:
        if not isinstance(data, dict) or data.get("export_format") != "itc-project/1.0":
            raise FormatCorrupt("导出包格式损坏：缺少 itc-project/1.0 标记", {"keys": list(data) if isinstance(data, dict) else None})
        pid = data["project_id"]
        if expect_id is not None and pid != expect_id:
            raise FormatCorrupt("项目身份不一致", {"expected": expect_id, "got": pid})
        for key in ("materials", "results", "events", "branches", "decisions"):
            if not isinstance(data.get(key), list):
                raise FormatCorrupt(f"导出包字段 {key} 损坏", {"field": key})
        # 事件链完整性（父指针 + 分支顺序）
        seen: set[tuple[str, int]] = set()
        for ev in data["events"]:
            if ev["parent_event_id"] is not None:
                parent = next((e for e in data["events"] if e["id"] == ev["parent_event_id"]
                               and e["branch"] == ev["branch"]), None)
                if parent is None or parent["seq"] != ev["seq"] - 1:
                    raise FormatCorrupt("事件父链断裂", {"event": ev["id"], "seq": ev["seq"]})
            seen.add((ev["branch"], ev["seq"]))

        op_id = self.begin_op(pid)
        try:
            for m in data["materials"]:
                self.put_material(pid, op_id, m)
            for r in data["rules"]:
                self.conn.execute(
                    "INSERT OR REPLACE INTO rules (id, project_id, rule_set, version, payload, op_id) "
                    "VALUES (?,?,?,?,?,?)",
                    (r["id"], pid, r.get("rule_set", ""), r.get("version", ""), canonical(r), op_id))
            for res in data["results"]:
                self.put_result(pid, op_id, res)
            for ev in data["events"]:
                self.conn.execute(
                    """INSERT OR REPLACE INTO events
                       (id, project_id, seq, branch, action, payload, parent_event_id, op_id)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (ev["id"], pid, ev["seq"], ev["branch"], ev["action"],
                     canonical(ev["payload"]), ev["parent_event_id"], op_id))
            for b in data["branches"]:
                self.conn.execute(
                    "INSERT OR REPLACE INTO branches (project_id, name, head_event_id, op_id) VALUES (?,?,?,?)",
                    (pid, b["name"], b["head_event_id"], op_id))
            for d in data["decisions"]:
                self.conn.execute(
                    "INSERT OR REPLACE INTO decisions (id, project_id, event_id, payload, op_id) VALUES (?,?,?,?,?)",
                    (d["id"], pid, d["event_id"], canonical(d["payload"]), op_id))
            self.commit_op(op_id)
        except Exception:
            self.conn.rollback()
            raise
        return pid

    def wipe_project(self, project_id: str) -> None:
        with self.lock:
            for tbl in ("materials", "rules", "results", "events", "branches", "decisions"):
                self.conn.execute(f"DELETE FROM {tbl} WHERE project_id=?", (project_id,))
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            self.conn.close()
