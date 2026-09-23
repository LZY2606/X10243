"""SQLite persistence.

Two physical tables (plus project metadata): the append-only event log and the
content-addressed material blob store.  Current state is derived by replay
(:mod:`itc_app.service`); we never mutate history.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    project_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    created_seq  INTEGER NOT NULL DEFAULT 0,
    metadata     TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS materials (
    material_id  TEXT NOT NULL,
    project_id   TEXT NOT NULL,
    kind         TEXT NOT NULL,
    file_name    TEXT NOT NULL,
    content      TEXT NOT NULL,
    media_type   TEXT NOT NULL,
    PRIMARY KEY (project_id, material_id)
);
CREATE TABLE IF NOT EXISTS events (
    event_id     TEXT NOT NULL,
    project_id   TEXT NOT NULL,
    branch       TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    parent_event_id TEXT,
    type         TEXT NOT NULL,
    payload      TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    PRIMARY KEY (project_id, event_id),
    UNIQUE (project_id, branch, seq)
);
CREATE INDEX IF NOT EXISTS events_branch ON events(project_id, branch, seq);
"""


def connect(db_path: str) -> sqlite3.Connection:
    first = not os.path.exists(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    if first:
        conn.commit()
    return conn


@contextmanager
def transaction(conn):
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def get_project(conn, project_id: str):
    return conn.execute(
        "SELECT * FROM projects WHERE project_id=?", (project_id,)).fetchone()


def list_projects(conn):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM projects ORDER BY name")]


def upsert_project(conn, project_id, name, metadata=None):
    conn.execute(
        "INSERT INTO projects(project_id, name, metadata) VALUES(?,?,?)\n"
        "ON CONFLICT(project_id) DO UPDATE SET name=excluded.name, "
        "metadata=excluded.metadata",
        (project_id, name, json.dumps(metadata or {}, ensure_ascii=False)))


def put_material(conn, project_id, material_id, kind, file_name, content,
                 media_type="text/plain"):
    conn.execute(
        "INSERT OR IGNORE INTO materials(project_id, material_id, kind, "
        "file_name, content, media_type) VALUES(?,?,?,?,?,?)",
        (project_id, material_id, kind, file_name, content, media_type))


def get_material(conn, project_id, material_id):
    row = conn.execute(
        "SELECT * FROM materials WHERE project_id=? AND material_id=?",
        (project_id, material_id)).fetchone()
    return dict(row) if row else None


def all_materials(conn, project_id):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM materials WHERE project_id=?", (project_id,))]


def append_event(conn, project_id, event_id, branch, seq, parent_event_id,
                 event_type, payload, payload_hash):
    conn.execute(
        "INSERT INTO events(event_id, project_id, branch, seq, "
        "parent_event_id, type, payload, payload_hash) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (event_id, project_id, branch, seq, parent_event_id, event_type,
         json.dumps(payload, ensure_ascii=False, default=str), payload_hash))


def branch_events(conn, project_id, branch, upto_seq=None):
    q = ("SELECT * FROM events WHERE project_id=? AND branch=? "
         "ORDER BY seq ASC")
    args: list[Any] = [project_id, branch]
    if upto_seq is not None:
        q += " LIMIT ?"
        args.append(int(upto_seq))
    return [dict(r) for r in conn.execute(q, args)]


def branch_head(conn, project_id, branch):
    row = conn.execute(
        "SELECT * FROM events WHERE project_id=? AND branch=? ORDER BY seq DESC "
        "LIMIT 1", (project_id, branch)).fetchone()
    return dict(row) if row else None


def list_branches(conn, project_id):
    rows = conn.execute(
        "SELECT branch, COUNT(*) n, MAX(seq) max_seq FROM events WHERE "
        "project_id=? GROUP BY branch", (project_id,)).fetchall()
    return {r["branch"]: {"events": r["n"], "max_seq": r["max_seq"]}
            for r in rows}


def wipe_project(conn, project_id):
    conn.execute("DELETE FROM events WHERE project_id=?", (project_id,))
    conn.execute("DELETE FROM materials WHERE project_id=?", (project_id,))
    conn.execute("DELETE FROM projects WHERE project_id=?", (project_id,))
