"""SQLite persistence: immutable materials, append-only decisions, derived results.

Identity strategy
-----------------
* stable ids are human-meaningful, deterministic and preserved on export/import
  (``mat_<project8>_<kind>_v<n>`` / ``evt_<project8>_b<branch>_<n>`` ...);
* every derived row carries ``input_summary``, ``rule_version`` and
  ``parent_versions``;
* history is never overwritten: revisions insert new versions, stale derived
  rows get ``status='stale'`` but remain readable.

Half-written imports use a staging table + single transactional promotion.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    slug TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    head_event_id TEXT
);
CREATE TABLE IF NOT EXISTS branches (
    branch_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    parent_branch_id TEXT,
    fork_event_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(project_id, name)
);
CREATE TABLE IF NOT EXISTS materials (
    material_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    version INTEGER NOT NULL,
    role TEXT NOT NULL,
    source_filename TEXT,
    content_sha256 TEXT NOT NULL,
    raw_text TEXT,
    parsed_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    supersedes TEXT,
    UNIQUE(project_id, kind, role, version)
);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    parent_event_id TEXT,
    adopted INTEGER NOT NULL DEFAULT 1,
    rule_version TEXT,
    UNIQUE(branch_id, seq)
);
CREATE TABLE IF NOT EXISTS results (
    result_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    branch_id TEXT,
    event_id TEXT,
    input_summary_json TEXT NOT NULL,
    rule_version TEXT NOT NULL,
    parent_versions_json TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS import_staging (
    staging_id TEXT PRIMARY KEY,
    project_id TEXT,
    bundle_sha256 TEXT,
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

RULE_VERSION = "itc.rules.v1"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def short(project_id: str) -> str:
    return project_id.replace("prj_", "")[:8]


def material_id(project_id, kind, version, role="main"):
    return f"mat_{short(project_id)}_{kind}_{role}_v{version}"


def event_id(project_id, branch_seq, seq):
    return f"evt_{short(project_id)}_b{branch_seq}_{seq}"


def result_id(project_id, kind, branch_id, event_id_, extra=""):
    h = hashlib.sha256(
        f"{kind}|{branch_id}|{event_id_}|{extra}".encode()).hexdigest()[:10]
    return f"res_{short(project_id)}_{kind}_{h}"


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection):
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def transaction(conn):
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def create_project(conn, project_id: str, slug: str, title: str,
                   created_at: str | None = None):
    created_at = created_at or now_iso()
    with transaction(conn):
        conn.execute(
            "INSERT INTO projects(project_id,slug,title,created_at) VALUES(?,?,?,?)",
            (project_id, slug, title, created_at))
        conn.execute(
            "INSERT INTO branches(branch_id,project_id,name,parent_branch_id,"
            "fork_event_id,created_at) VALUES(?,?,?,?,?,?)",
            (f"br_{short(project_id)}_1", project_id, "main", None, None,
             created_at))


def get_project(conn, project_id: str):
    return conn.execute("SELECT * FROM projects WHERE project_id=?",
                        (project_id,)).fetchone()


def list_projects(conn):
    return conn.execute("SELECT * FROM projects ORDER BY slug").fetchall()


def get_branch(conn, branch_id: str):
    return conn.execute("SELECT * FROM branches WHERE branch_id=?",
                        (branch_id,)).fetchone()


def main_branch(conn, project_id: str):
    return conn.execute(
        "SELECT * FROM branches WHERE project_id=? ORDER BY created_at LIMIT 1",
        (project_id,)).fetchone()


def next_material_version(conn, project_id, kind, role="main"):
    row = conn.execute(
        "SELECT MAX(version) AS v FROM materials WHERE project_id=? AND kind=? "
        "AND role=?", (project_id, kind, role)).fetchone()
    return (row["v"] or 0) + 1


def insert_material(conn, project_id, kind, parsed: dict, raw_text: str,
                    role="main", source_filename=None, supersedes=None,
                    created_at=None):
    created_at = created_at or now_iso()
    version = next_material_version(conn, project_id, kind, role)
    mid = material_id(project_id, kind, version, role)
    with transaction(conn):
        conn.execute(
            "INSERT INTO materials(material_id,project_id,kind,version,role,"
            "source_filename,content_sha256,raw_text,parsed_json,created_at,"
            "supersedes) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (mid, project_id, kind, version, role, source_filename,
             sha256_text(raw_text), raw_text, json.dumps(parsed, sort_keys=True,
                                                         ensure_ascii=False),
             created_at, supersedes))
    return mid


def latest_material(conn, project_id, kind, role="main"):
    return conn.execute(
        "SELECT * FROM materials WHERE project_id=? AND kind=? AND role=? "
        "ORDER BY version DESC LIMIT 1", (project_id, kind, role)).fetchone()


def material_by_id(conn, material_id):
    return conn.execute("SELECT * FROM materials WHERE material_id=?",
                        (material_id,)).fetchone()


def all_materials(conn, project_id):
    return conn.execute(
        "SELECT * FROM materials WHERE project_id=? ORDER BY kind,role,version",
        (project_id,)).fetchall()


def next_event_seq(conn, branch_id):
    row = conn.execute("SELECT MAX(seq) AS s FROM events WHERE branch_id=?",
                       (branch_id,)).fetchone()
    return (row["s"] or 0) + 1


def insert_event(conn, project_id, branch_id, etype, payload,
                 parent_event_id=None, adopted=True, rule_version=RULE_VERSION,
                 created_at=None):
    created_at = created_at or now_iso()
    branch = get_branch(conn, branch_id)
    branch_seq = int(branch["branch_id"].split("_")[-1])
    seq = next_event_seq(conn, branch_id)
    eid = event_id(project_id, branch_seq, seq)
    with transaction(conn):
        conn.execute(
            "INSERT INTO events(event_id,project_id,branch_id,seq,type,"
            "payload_json,created_at,parent_event_id,adopted,rule_version) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (eid, project_id, branch_id, seq, etype,
             json.dumps(payload, sort_keys=True, ensure_ascii=False), created_at,
             parent_event_id, 1 if adopted else 0, rule_version))
        conn.execute("UPDATE projects SET head_event_id=? WHERE project_id=?",
                     (eid, project_id))
    return eid


def branch_events(conn, branch_id):
    return conn.execute(
        "SELECT * FROM events WHERE branch_id=? ORDER BY seq",
        (branch_id,)).fetchall()


def event_by_id(conn, event_id):
    return conn.execute("SELECT * FROM events WHERE event_id=?",
                        (event_id,)).fetchone()


def insert_result(conn, project_id, kind, branch_id, event_id_, input_summary,
                  parent_versions, payload, status="ok",
                  rule_version=RULE_VERSION, extra="", created_at=None):
    created_at = created_at or now_iso()
    rid = result_id(project_id, kind, branch_id, event_id_, extra)
    with transaction(conn):
        conn.execute(
            "INSERT OR REPLACE INTO results(result_id,project_id,kind,branch_id,"
            "event_id,input_summary_json,rule_version,parent_versions_json,"
            "status,payload_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (rid, project_id, kind, branch_id, event_id_,
             json.dumps(input_summary, sort_keys=True, ensure_ascii=False),
             rule_version,
             json.dumps(parent_versions, sort_keys=True, ensure_ascii=False),
             status,
             json.dumps(payload, sort_keys=True, ensure_ascii=False), created_at))
    return rid


def mark_stale_results(conn, project_id, kinds: list[str], branch_id=None):
    q = "UPDATE results SET status='stale' WHERE project_id=? AND status='ok' "
    args = [project_id]
    placeholders = ",".join("?" for _ in kinds)
    q += f"AND kind IN ({placeholders})"
    args.extend(kinds)
    if branch_id:
        q += " AND branch_id=?"
        args.append(branch_id)
    with transaction(conn):
        conn.execute(q, args)


def results_for_event(conn, event_id_):
    return conn.execute("SELECT * FROM results WHERE event_id=? ORDER BY kind",
                        (event_id_,)).fetchall()


def create_branch(conn, project_id, name, fork_event_id, created_at=None):
    created_at = created_at or now_iso()
    parent = main_branch(conn, project_id)
    n = conn.execute("SELECT COUNT(*) AS c FROM branches WHERE project_id=?",
                     (project_id,)).fetchone()["c"] + 1
    bid = f"br_{short(project_id)}_{n}"
    with transaction(conn):
        conn.execute(
            "INSERT INTO branches(branch_id,project_id,name,parent_branch_id,"
            "fork_event_id,created_at) VALUES(?,?,?,?,?,?)",
            (bid, project_id, name, parent["branch_id"], fork_event_id,
             created_at))
    return bid


def list_branches(conn, project_id):
    return conn.execute(
        "SELECT * FROM branches WHERE project_id=? ORDER BY branch_id",
        (project_id,)).fetchall()


def stage_import(conn, staging_id, project_id, bundle_sha, payload,
                 created_at=None):
    created_at = created_at or now_iso()
    with transaction(conn):
        conn.execute(
            "INSERT INTO import_staging(staging_id,project_id,bundle_sha256,"
            "payload_json,state,created_at) VALUES(?,?,?,?,?,?)",
            (staging_id, project_id, bundle_sha,
             json.dumps(payload, sort_keys=True, ensure_ascii=False), "staged",
             created_at))


def get_staging(conn, staging_id):
    return conn.execute("SELECT * FROM import_staging WHERE staging_id=?",
                        (staging_id,)).fetchone()


def list_staging(conn, state=None):
    if state:
        return conn.execute("SELECT * FROM import_staging WHERE state=?",
                            (state,)).fetchall()
    return conn.execute("SELECT * FROM import_staging").fetchall()


def delete_staging(conn, staging_id):
    with transaction(conn):
        conn.execute("DELETE FROM import_staging WHERE staging_id=?",
                     (staging_id,))


def set_staging_state(conn, staging_id, state):
    with transaction(conn):
        conn.execute("UPDATE import_staging SET state=? WHERE staging_id=?",
                     (state, staging_id))


def cleanup_stale_staging(conn):
    """Recover half-written imports left by a crashed process."""
    with transaction(conn):
        cur = conn.execute(
            "DELETE FROM import_staging WHERE state!='promoted'")
    return cur.rowcount


def dump_project(conn, project_id) -> dict[str, Any]:
    return {
        "project": dict(get_project(conn, project_id)),
        "branches": [dict(b) for b in list_branches(conn, project_id)],
        "materials": [dict(m) for m in all_materials(conn, project_id)],
        "events": [dict(e) for e in conn.execute(
            "SELECT * FROM events WHERE project_id=? ORDER BY branch_id,seq",
            (project_id,)).fetchall()],
        "results": [dict(r) for r in conn.execute(
            "SELECT * FROM results WHERE project_id=? ORDER BY created_at",
            (project_id,)).fetchall()],
    }


def load_project(conn, data: dict[str, Any]):
    """Idempotent restore inside one transaction; ids are preserved."""
    with transaction(conn) as c:
        c.execute("INSERT OR REPLACE INTO projects VALUES(?,?,?,?,?)",
                  tuple(data["project"][k] for k in
                        ("project_id", "slug", "title", "created_at",
                         "head_event_id")))
        for b in data["branches"]:
            c.execute("INSERT OR REPLACE INTO branches VALUES(?,?,?,?,?,?)",
                      tuple(b[k] for k in
                            ("branch_id", "project_id", "name",
                             "parent_branch_id", "fork_event_id", "created_at")))
        for m in data["materials"]:
            c.execute("INSERT OR REPLACE INTO materials VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      tuple(m[k] for k in
                            ("material_id", "project_id", "kind", "version",
                             "role", "source_filename", "content_sha256",
                             "raw_text", "parsed_json", "created_at",
                             "supersedes")))
        for e in data["events"]:
            c.execute("INSERT OR REPLACE INTO events VALUES(?,?,?,?,?,?,?,?,?,?)",
                      tuple(e[k] for k in
                            ("event_id", "project_id", "branch_id", "seq",
                             "type", "payload_json", "created_at",
                             "parent_event_id", "adopted", "rule_version")))
        for r in data["results"]:
            c.execute("INSERT OR REPLACE INTO results VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      tuple(r[k] for k in
                            ("result_id", "project_id", "kind", "branch_id",
                             "event_id", "input_summary_json", "rule_version",
                             "parent_versions_json", "status", "payload_json",
                             "created_at")))
