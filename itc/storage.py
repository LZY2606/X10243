from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from .core import canonical_json_bytes, stable_hash

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS projects (
  project_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  branch TEXT NOT NULL,
  seq INTEGER NOT NULL,
  kind TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  actor TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  parents_json TEXT NOT NULL DEFAULT '[]',
  input_summary_json TEXT NOT NULL DEFAULT '{}',
  rules_version TEXT NOT NULL,
  result_ids_json TEXT NOT NULL DEFAULT '[]',
  UNIQUE(project_id, branch, seq)
);
CREATE TABLE IF NOT EXISTS materials (
  project_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  material_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  event_id TEXT NOT NULL,
  value_json TEXT NOT NULL,
  original_json TEXT NOT NULL,
  diagnostics_json TEXT NOT NULL DEFAULT '[]',
  PRIMARY KEY(project_id, kind, material_id, version)
);
CREATE TABLE IF NOT EXISTS results (
  project_id TEXT NOT NULL,
  result_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  branch TEXT NOT NULL,
  event_id TEXT NOT NULL,
  status TEXT NOT NULL,
  value_json TEXT NOT NULL,
  parents_json TEXT NOT NULL,
  input_summary_json TEXT NOT NULL,
  rules_version TEXT NOT NULL,
  PRIMARY KEY(project_id, result_id)
);
CREATE TABLE IF NOT EXISTS branch_heads (
  project_id TEXT NOT NULL,
  branch TEXT NOT NULL,
  event_id TEXT NOT NULL,
  PRIMARY KEY(project_id, branch)
);
CREATE TABLE IF NOT EXISTS decisions (
  project_id TEXT NOT NULL,
  decision_id TEXT NOT NULL,
  branch TEXT NOT NULL,
  event_id TEXT NOT NULL,
  decision TEXT NOT NULL,
  target_type TEXT NOT NULL,
  target_id TEXT NOT NULL,
  rationale TEXT NOT NULL,
  conflicts_with TEXT,
  payload_json TEXT NOT NULL,
  PRIMARY KEY(project_id, decision_id)
);
"""


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def loads(text, fallback=None):
    if text is None:
        return fallback
    return json.loads(text)


class Repository:
    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self):
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def project_exists(self, project_id):
        with self.connect() as conn:
            return conn.execute("SELECT 1 FROM projects WHERE project_id=?", (project_id,)).fetchone() is not None

    def create_project(self, project_id, name, created_at=None):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO projects(project_id,name,created_at) VALUES(?,?,?)",
                (project_id, name, created_at or now_iso()),
            )

    def head(self, project_id, branch="main"):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT event_id FROM branch_heads WHERE project_id=? AND branch=?", (project_id, branch)
            ).fetchone()
            return row["event_id"] if row else None

    def branches(self, project_id):
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT branch,event_id FROM branch_heads WHERE project_id=? ORDER BY branch", (project_id,)
            )]

    def next_seq(self, conn, project_id, branch):
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS seq FROM events WHERE project_id=? AND branch=?",
            (project_id, branch),
        ).fetchone()
        return int(row["seq"])

    def append_event(self, project_id, kind, payload, branch="main", parents=None, actor="system",
                     input_summary=None, rules_version="", results=None, timestamp=None, material_records=None,
                     decisions=None):
        parents = parents or []
        results = results or []
        material_records = material_records or []
        decisions = decisions or []
        with self.connect() as conn:
            seq = self.next_seq(conn, project_id, branch)
            timestamp = timestamp or now_iso()
            preimage = {
                "project_id": project_id, "branch": branch, "seq": seq, "kind": kind,
                "timestamp": timestamp, "actor": actor, "payload": payload, "parents": parents,
                "input_summary": input_summary or {}, "rules_version": rules_version,
            }
            event_id = "evt_" + stable_hash(preimage)
            conn.execute(
                """INSERT INTO events(event_id,project_id,branch,seq,kind,timestamp,actor,payload_json,
                   parents_json,input_summary_json,rules_version,result_ids_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (event_id, project_id, branch, seq, kind, timestamp, actor, dumps(payload), dumps(parents),
                 dumps(input_summary or {}), rules_version, dumps([r["result_id"] for r in results])),
            )
            for record in material_records:
                conn.execute(
                    """INSERT INTO materials(project_id,kind,material_id,version,event_id,value_json,
                       original_json,diagnostics_json) VALUES(?,?,?,?,?,?,?,?)""",
                    (project_id, record["kind"], record["material_id"], record["version"], event_id,
                     dumps(record.get("value")), dumps(record.get("original", record.get("value"))),
                     dumps(record.get("diagnostics", []))),
                )
            for result in results:
                conn.execute(
                    """INSERT INTO results(project_id,result_id,kind,branch,event_id,status,value_json,
                       parents_json,input_summary_json,rules_version) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (project_id, result["result_id"], result["kind"], branch, event_id, result["status"],
                     dumps(result["value"]), dumps(result.get("parents", [event_id])),
                     dumps(result.get("input_summary", {})), result.get("rules_version", rules_version)),
                )
            for decision in decisions:
                conn.execute(
                    """INSERT INTO decisions(project_id,decision_id,branch,event_id,decision,target_type,
                       target_id,rationale,conflicts_with,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (project_id, decision["decision_id"], branch, event_id, decision["decision"],
                     decision["target_type"], decision["target_id"], decision.get("rationale", ""),
                     decision.get("conflicts_with"), dumps(decision.get("payload", {}))),
                )
            conn.execute(
                """INSERT INTO branch_heads(project_id,branch,event_id) VALUES(?,?,?)
                   ON CONFLICT(project_id,branch) DO UPDATE SET event_id=excluded.event_id""",
                (project_id, branch, event_id),
            )
            return event_id

    def events(self, project_id, branch=None):
        sql = "SELECT * FROM events WHERE project_id=?"
        args = [project_id]
        if branch:
            sql += " AND branch=?"
            args.append(branch)
        sql += " ORDER BY seq"
        with self.connect() as conn:
            return [self._event(row) for row in conn.execute(sql, args)]

    def event(self, project_id, event_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM events WHERE project_id=? AND event_id=?", (project_id, event_id)).fetchone()
            return self._event(row) if row else None

    def _event(self, row):
        return {
            "event_id": row["event_id"], "project_id": row["project_id"], "branch": row["branch"],
            "seq": row["seq"], "kind": row["kind"], "timestamp": row["timestamp"], "actor": row["actor"],
            "payload": loads(row["payload_json"]), "parents": loads(row["parents_json"], []),
            "input_summary": loads(row["input_summary_json"], {}), "rules_version": row["rules_version"],
            "result_ids": loads(row["result_ids_json"], []),
        }

    def _lineage_ids(self, project_id, branch, until_event=None):
        head = until_event or self.head(project_id, branch)
        if not head:
            return set()
        all_events = self.events(project_id, None)
        by_id = {event["event_id"]: event for event in all_events}
        stack = [head]
        ids = set()
        while stack:
            current = stack.pop()
            if current in ids or current not in by_id:
                continue
            ids.add(current)
            stack.extend(parent for parent in by_id[current]["parents"] if parent in by_id)
        return ids

    def materials_at(self, project_id, branch="main", until_event=None):
        ids = self._lineage_ids(project_id, branch, until_event)
        materials = {}
        with self.connect() as conn:
            if ids:
                marks = ",".join("?" for _ in ids)
                rows = conn.execute(
                    f"SELECT * FROM materials WHERE project_id=? AND event_id IN ({marks})",
                    [project_id, *ids],
                )
                for row in rows:
                    materials.setdefault(row["kind"], {})[row["material_id"]] = {
                        "material_id": row["material_id"], "kind": row["kind"], "version": row["version"],
                        "event_id": row["event_id"], "value": loads(row["value_json"]),
                        "original": loads(row["original_json"]), "diagnostics": loads(row["diagnostics_json"], []),
                    }
        return materials

    def results_at(self, project_id, branch="main", until_event=None):
        event_ids = list(self._lineage_ids(project_id, branch, until_event))
        if not event_ids:
            return {}
        out = {}
        with self.connect() as conn:
            chunks = [event_ids[i:i + 400] for i in range(0, len(event_ids), 400)]
            for chunk in chunks:
                marks = ",".join("?" for _ in chunk)
                for row in conn.execute(f"SELECT * FROM results WHERE project_id=? AND event_id IN ({marks}) ORDER BY event_id", [project_id, *chunk]):
                    result = {
                        "result_id": row["result_id"], "kind": row["kind"], "branch": row["branch"],
                        "event_id": row["event_id"], "status": row["status"], "value": loads(row["value_json"]),
                        "parents": loads(row["parents_json"]), "input_summary": loads(row["input_summary_json"]),
                        "rules_version": row["rules_version"],
                    }
                    out[result["result_id"]] = result
        return out

    def decisions_at(self, project_id, branch="main", until_event=None):
        ids = self._lineage_ids(project_id, branch, until_event)
        results = []
        if not ids:
            return results
        with self.connect() as conn:
            marks = ",".join("?" for _ in ids)
            rows = conn.execute(
                f"SELECT * FROM decisions WHERE event_id IN ({marks}) ORDER BY event_id", list(ids)
            )
            for row in rows:
                results.append(dict(row))
        return results

    def _ancestors(self, project_id, event_id, branch):
        ordered = self.events(project_id, branch)
        by_id = {event["event_id"]: event for event in ordered}
        if event_id not in by_id:
            return []
        out = []
        stack = [event_id]
        seen = set()
        while stack:
            current = stack.pop()
            if current in seen or current not in by_id:
                continue
            seen.add(current)
            out.append(current)
            stack.extend(parent for parent in by_id[current]["parents"] if parent in by_id)
        return [event["event_id"] for event in ordered if event["event_id"] in seen]

    def export_project(self, project_id):
        with self.connect() as conn:
            project = conn.execute("SELECT * FROM projects WHERE project_id=?", (project_id,)).fetchone()
            tables = ["events", "materials", "results", "branch_heads", "decisions"]
            payload = {"format": "itc-review-archive/v1", "project": dict(project), "tables": {}}
            for table in tables:
                payload["tables"][table] = [dict(row) for row in conn.execute(f"SELECT * FROM {table} WHERE project_id=?", (project_id,))]
        payload["archive_hash"] = stable_hash({k: v for k, v in payload.items() if k != "archive_hash"})
        return payload

    def import_project(self, payload, replace=False):
        expected_hash = payload.get("archive_hash")
        actual = stable_hash({k: v for k, v in payload.items() if k != "archive_hash"})
        if payload.get("format") != "itc-review-archive/v1" or expected_hash != actual:
            raise ValueError("corrupt ITC archive")
        project = payload["project"]
        with self.connect() as conn:
            if replace:
                for table in ["events", "materials", "results", "branch_heads", "decisions", "projects"]:
                    conn.execute(f"DELETE FROM {table} WHERE project_id=?", (project["project_id"],))
            conn.execute("INSERT INTO projects VALUES(:project_id,:name,:created_at)", project)
            for row in payload["tables"]["events"]:
                conn.execute("INSERT INTO events VALUES(:event_id,:project_id,:branch,:seq,:kind,:timestamp,:actor,:payload_json,:parents_json,:input_summary_json,:rules_version,:result_ids_json)", row)
            for row in payload["tables"]["materials"]:
                conn.execute("INSERT INTO materials VALUES(:project_id,:kind,:material_id,:version,:event_id,:value_json,:original_json,:diagnostics_json)", row)
            for row in payload["tables"]["results"]:
                conn.execute("INSERT INTO results VALUES(:project_id,:result_id,:kind,:branch,:event_id,:status,:value_json,:parents_json,:input_summary_json,:rules_version)", row)
            for row in payload["tables"]["branch_heads"]:
                conn.execute("INSERT INTO branch_heads VALUES(:project_id,:branch,:event_id)", row)
            for row in payload["tables"]["decisions"]:
                conn.execute("INSERT INTO decisions VALUES(:project_id,:decision_id,:branch,:event_id,:decision,:target_type,:target_id,:rationale,:conflicts_with,:payload_json)", row)

    def recover_if_needed(self):
        with self.connect() as conn:
            conn.execute("PRAGMA integrity_check")
