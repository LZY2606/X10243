"""SQLite 存储层。

- 材料版本不可变；事件只追加；结果缓存可由事件重放重建。
- 导入在单事务内完成；写前先登记 import_journal，崩溃后重启按
  journal 的状态（prepared/committed）恢复或回滚，即“半写恢复”。
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from typing import Any

from .models import MaterialVersion, canon_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL,
    current_branch_id TEXT
);
CREATE TABLE IF NOT EXISTS branches (
    branch_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    name TEXT NOT NULL,
    parent_event_id TEXT,
    created_at REAL NOT NULL,
    tip_event_id TEXT
);
CREATE TABLE IF NOT EXISTS material_versions (
    version_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    material_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    role TEXT NOT NULL,
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    origin TEXT NOT NULL,
    created_event TEXT NOT NULL,
    supersedes TEXT,
    content_digest TEXT NOT NULL,
    UNIQUE(project_id, material_id, version)
);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    parent_event_id TEXT,
    at REAL NOT NULL,
    actor TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    supersedes_event_id TEXT
);
CREATE TABLE IF NOT EXISTS results (
    result_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    parent_event_id TEXT NOT NULL,
    rule_version TEXT NOT NULL,
    created_at REAL NOT NULL,
    body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS import_journal (
    journal_id TEXT PRIMARY KEY,
    project_id TEXT,
    status TEXT NOT NULL,
    prepared_at REAL NOT NULL,
    committed_at REAL,
    summary TEXT,
    error TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_events_seq
    ON events(branch_id, seq);
CREATE INDEX IF NOT EXISTS ix_results_branch ON results(branch_id);
CREATE INDEX IF NOT EXISTS ix_materials_ident
    ON material_versions(project_id, material_id, version);
"""


def connect(db_path: str) -> sqlite3.Connection:
    fresh = not os.path.exists(db_path)
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def begin_import(conn, journal_id: str, project_id: str | None, summary: dict):
    conn.execute(
        "INSERT INTO import_journal(journal_id,project_id,status,"
        "prepared_at,summary) VALUES(?,?,?,?,?)",
        (journal_id, project_id, "prepared", time.time(),
         canon_json(summary)))


def mark_import_committed(conn, journal_id: str, project_id: str):
    conn.execute(
        "UPDATE import_journal SET status='committed',committed_at=?,"
        "project_id=? WHERE journal_id=?",
        (time.time(), project_id, journal_id))


def recover_half_writes(conn) -> list[dict]:
    """启动时调用：committed 但无项目 / prepared 未提交的 journal 回滚清理。"""
    rows = conn.execute("SELECT * FROM import_journal").fetchall()
    recovered = []
    for r in rows:
        if r["status"] == "committed" and r["project_id"]:
            exists = conn.execute(
                "SELECT 1 FROM projects WHERE project_id=?",
                (r["project_id"],)).fetchone()
            if not exists:
                recovered.append({"journal_id": r["journal_id"],
                                  "action": "orphan_committed",
                                  "project_id": r["project_id"]})
                conn.execute("DELETE FROM import_journal WHERE journal_id=?",
                             (r["journal_id"],))
        elif r["status"] == "prepared":
            recovered.append({"journal_id": r["journal_id"],
                              "action": "rolled_back_prepared",
                              "project_id": r["project_id"]})
            conn.execute("DELETE FROM import_journal WHERE journal_id=?",
                         (r["journal_id"],))
    return recovered
