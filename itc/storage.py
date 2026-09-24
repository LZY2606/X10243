from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS projects (
  project_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  created_at TEXT NOT NULL,
  case_pack_id TEXT,
  resource_limits_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS branches (
  project_id TEXT NOT NULL,
  branch_id TEXT NOT NULL,
  name TEXT NOT NULL,
  parent_branch_id TEXT,
  fork_event_id TEXT,
  created_event_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(project_id, branch_id)
);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  branch_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  kind TEXT NOT NULL,
  actor TEXT NOT NULL,
  summary TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(project_id, branch_id, seq)
);
CREATE TABLE IF NOT EXISTS event_parents (
  event_id TEXT NOT NULL,
  parent_event_id TEXT NOT NULL,
  position INTEGER NOT NULL,
  PRIMARY KEY(event_id, parent_event_id)
);
CREATE TABLE IF NOT EXISTS branch_heads (
  project_id TEXT NOT NULL,
  branch_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  PRIMARY KEY(project_id, branch_id)
);
CREATE TABLE IF NOT EXISTS material_versions (
  material_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  branch_id TEXT NOT NULL,
  material_type TEXT NOT NULL,
  version INTEGER NOT NULL,
  event_id TEXT NOT NULL,
  source_id TEXT NOT NULL,
  source_revision TEXT NOT NULL,
  status TEXT NOT NULL,
  raw_document_json TEXT NOT NULL,
  canonical_json TEXT NOT NULL,
  diagnostics_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(project_id, branch_id, material_type, version)
);
CREATE TABLE IF NOT EXISTS decisions (
  event_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  branch_id TEXT NOT NULL,
  decision_key TEXT NOT NULL,
  action TEXT NOT NULL,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS derived_results (
  result_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  result_type TEXT NOT NULL,
  dependency_digest TEXT NOT NULL,
  object_key TEXT NOT NULL,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  input_summary_json TEXT NOT NULL,
  parameter_versions_json TEXT NOT NULL,
  parent_dependencies_json TEXT NOT NULL,
  created_event_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(project_id, result_type, dependency_digest)
);
CREATE TABLE IF NOT EXISTS state_results (
  project_id TEXT NOT NULL,
  branch_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  result_type TEXT NOT NULL,
  object_key TEXT NOT NULL,
  result_id TEXT NOT NULL,
  PRIMARY KEY(project_id, branch_id, event_id, result_type, object_key)
);
CREATE INDEX IF NOT EXISTS idx_materials_lookup
  ON material_versions(project_id, branch_id, material_type, version);
CREATE INDEX IF NOT EXISTS idx_results_lookup
  ON derived_results(project_id, result_type, object_key);
CREATE INDEX IF NOT EXISTS idx_state_lookup
  ON state_results(project_id, branch_id, event_id, result_type);
"""


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def loads(value: str | bytes | None, default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value)


def read_project(conn: sqlite3.Connection, project_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM projects WHERE project_id=?", (project_id,)).fetchone()
    if row is None:
        return None
    project = dict(row)
    project["resource_limits"] = loads(project.pop("resource_limits_json"))
    return project


def list_projects(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM projects ORDER BY created_at, project_id").fetchall()
    projects = []
    for row in rows:
        item = dict(row)
        item["resource_limits"] = loads(item.pop("resource_limits_json"))
        projects.append(item)
    return projects


def all_events(conn: sqlite3.Connection, project_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT e.*, GROUP_CONCAT(p.parent_event_id, CHAR(31)) AS parents
        FROM events e
        LEFT JOIN event_parents p ON p.event_id=e.event_id
        WHERE e.project_id=?
        GROUP BY e.event_id
        ORDER BY e.created_at, e.event_id
        """,
        (project_id,),
    ).fetchall()
    events = []
    for row in rows:
        item = dict(row)
        item["payload"] = loads(item.pop("payload_json"))
        item["parents"] = [p for p in (item.pop("parents") or "").split("\x1f") if p]
        events.append(item)
    return events


def insert_event(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    project_id: str,
    branch_id: str,
    kind: str,
    actor: str,
    summary: str,
    payload: dict[str, Any],
    parents: Iterable[str],
    created_at: str,
) -> None:
    seq_row = conn.execute(
        "SELECT COALESCE(MAX(seq),0)+1 AS next_seq FROM events WHERE project_id=? AND branch_id=?",
        (project_id, branch_id),
    ).fetchone()
    conn.execute(
        """
        INSERT INTO events(event_id,project_id,branch_id,seq,kind,actor,summary,payload_json,created_at)
        VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (event_id, project_id, branch_id, seq_row["next_seq"], kind, actor, summary, dumps(payload), created_at),
    )
    for position, parent in enumerate(parents):
        conn.execute(
            "INSERT INTO event_parents(event_id,parent_event_id,position) VALUES(?,?,?)",
            (event_id, parent, position),
        )
    conn.execute(
        """
        INSERT INTO branch_heads(project_id,branch_id,event_id)
        VALUES(?,?,?)
        ON CONFLICT(project_id,branch_id) DO UPDATE SET event_id=excluded.event_id
        """,
        (project_id, branch_id, event_id),
    )


def insert_material_version(
    conn: sqlite3.Connection,
    *,
    material_id: str,
    project_id: str,
    branch_id: str,
    material_type: str,
    version: int,
    event_id: str,
    source_id: str,
    source_revision: str,
    status: str,
    raw_document: dict[str, Any],
    canonical: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO material_versions(
          material_id,project_id,branch_id,material_type,version,event_id,source_id,
          source_revision,status,raw_document_json,canonical_json,diagnostics_json,created_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            material_id, project_id, branch_id, material_type, version, event_id, source_id,
            source_revision, status, dumps(raw_document), dumps(canonical), dumps(diagnostics), created_at,
        ),
    )


def insert_derived_result(
    conn: sqlite3.Connection,
    *,
    result_id: str,
    project_id: str,
    result_type: str,
    dependency_digest: str,
    object_key: str,
    status: str,
    payload: dict[str, Any],
    input_summary: dict[str, Any],
    parameter_versions: dict[str, Any],
    parent_dependencies: dict[str, Any],
    created_event_id: str,
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO derived_results(
          result_id,project_id,result_type,dependency_digest,object_key,status,payload_json,
          input_summary_json,parameter_versions_json,parent_dependencies_json,created_event_id,created_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(project_id,result_type,dependency_digest) DO UPDATE SET
          status=excluded.status,
          payload_json=excluded.payload_json,
          input_summary_json=excluded.input_summary_json,
          parameter_versions_json=excluded.parameter_versions_json,
          parent_dependencies_json=excluded.parent_dependencies_json
        """,
        (
            result_id, project_id, result_type, dependency_digest, object_key, status, dumps(payload),
            dumps(input_summary), dumps(parameter_versions), dumps(parent_dependencies), created_event_id, created_at,
        ),
    )


def set_state_result(
    conn: sqlite3.Connection,
    project_id: str,
    branch_id: str,
    event_id: str,
    result_type: str,
    object_key: str,
    result_id: str,
) -> None:
    conn.execute(
        """
        INSERT INTO state_results(project_id,branch_id,event_id,result_type,object_key,result_id)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(project_id,branch_id,event_id,result_type,object_key)
        DO UPDATE SET result_id=excluded.result_id
        """,
        (project_id, branch_id, event_id, result_type, object_key, result_id),
    )


def replace_project_database(target: Path, source: sqlite3.Connection) -> None:
    source.commit()
    target.parent.mkdir(parents=True, exist_ok=True)
    backup_path = target.with_suffix(target.suffix + ".restored")
    if backup_path.exists():
        backup_path.unlink()
    if target.exists():
        target.replace(backup_path)
    source.backup(sqlite3.connect(str(target)))
    if backup_path.exists():
        backup_path.unlink()
