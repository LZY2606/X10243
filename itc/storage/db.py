"""连接、建表与半写恢复。

每个写事务先在 txn_journal 记 pending，COMMIT 成功后记 committed。
进程在中途被杀（simulate_crash=1 或外部破坏）后重启时，recover_pending()
会回滚所有没有 committed 记录的暂存行——这是“半写恢复”的可重放证据。
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from ..core.constants import SCHEMA_VERSION

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS projects(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL,
  parent_event_id TEXT,
  head_event_id TEXT
);
CREATE TABLE IF NOT EXISTS branches(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  name TEXT NOT NULL,
  parent_branch_id TEXT,
  fork_event_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  head_event_id TEXT
);
CREATE TABLE IF NOT EXISTS events(
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  branch_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  parent_event_id TEXT,
  type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  actor TEXT NOT NULL DEFAULT 'reviewer',
  note TEXT,
  created_at TEXT NOT NULL,
  txn_id TEXT NOT NULL,
  UNIQUE(branch_id, seq)
);
CREATE TABLE IF NOT EXISTS materials(
  id TEXT NOT NULL,
  version INTEGER NOT NULL,
  project_id TEXT NOT NULL,
  slot TEXT NOT NULL,
  label TEXT,
  payload_json TEXT NOT NULL,
  raw_digest TEXT NOT NULL,
  introduced_event_id TEXT NOT NULL,
  superseded_event_id TEXT,
  PRIMARY KEY(id, version)
);
CREATE TABLE IF NOT EXISTS nodes_cache(
  project_id TEXT NOT NULL,
  branch_id TEXT NOT NULL,
  head_event_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  node_json TEXT NOT NULL,
  ord INTEGER NOT NULL,
  PRIMARY KEY(project_id, branch_id, head_event_id, node_id)
);
CREATE TABLE IF NOT EXISTS txn_journal(
  txn_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  branch_id TEXT NOT NULL,
  state TEXT NOT NULL,
  created_at TEXT NOT NULL,
  finished_at TEXT
);
"""


def connect(path: str) -> sqlite3.Connection:
    fresh = not os.path.exists(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    if fresh:
        conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
            (SCHEMA_VERSION,))
        conn.commit()
    return conn


def recover_pending(conn: sqlite3.Connection) -> list[str]:
    """删除未提交事务写入的半行，返回事务 id 列表（恢复证据）。"""
    rows = conn.execute(
        "SELECT txn_id FROM txn_journal WHERE state='pending'").fetchall()
    recovered = []
    for r in rows:
        txn = r["txn_id"]
        conn.execute("DELETE FROM events WHERE txn_id=?", (txn,))
        conn.execute(
            """DELETE FROM materials WHERE introduced_event_id IN
               (SELECT DISTINCT introduced_event_id FROM materials
                WHERE introduced_event_id NOT IN (SELECT id FROM events))""")
        conn.execute(
            """DELETE FROM nodes_cache WHERE head_event_id
               NOT IN (SELECT id FROM events)""")
        conn.execute("DELETE FROM txn_journal WHERE txn_id=?", (txn,))
        recovered.append(txn)
    conn.commit()
    return recovered


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def loads(text: str) -> Any:
    return json.loads(text)
