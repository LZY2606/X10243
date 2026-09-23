"""SQLite 存储层：不可变材料版本、派生结果、事件流、分支、写入组日志。

半写恢复：每次跨表写组先在 journal 登记 pending 与已落库行，
崩溃重开时未 committed 的写组被回滚（记录为 rolled_back）。
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS branches (
    branch_id     TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    head_seq      INTEGER NOT NULL DEFAULT 0,
    parent_branch TEXT,
    parent_seq    INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS materials (
    material_id TEXT NOT NULL,
    version     INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    raw         TEXT NOT NULL DEFAULT '',
    created_seq INTEGER NOT NULL,
    PRIMARY KEY (material_id, version)
);
CREATE TABLE IF NOT EXISTS results (
    result_id   TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    deps        TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    param_version TEXT NOT NULL,
    created_seq INTEGER NOT NULL,
    branch      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   TEXT NOT NULL UNIQUE,
    branch     TEXT NOT NULL,
    kind       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    parents    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS journal (
    group_id   TEXT PRIMARY KEY,
    label      TEXT NOT NULL,
    status     TEXT NOT NULL,          -- pending | committed | rolled_back
    inserts    TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
"""

_PK = {
    "materials": ("material_id", "version"),
    "results": ("result_id",),
    "events": ("event_id",),
    "branches": ("branch_id",),
    "meta": ("key",),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()
        self._ensure_main_branch()
        self._recover()

    # ---- 基础 ----------------------------------------------------------

    def _migrate(self) -> None:
        with self.conn:
            self.conn.executescript(SCHEMA)

    def _ensure_main_branch(self) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO branches(branch_id,name,head_seq,"
                "parent_branch,parent_seq) VALUES(?,?,0,NULL,0)",
                ("main", "main"))

    def create_branch(self, branch_id: str, name: str,
                      parent_branch: str, parent_seq: int) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO branches(branch_id,name,head_seq,"
                "parent_branch,parent_seq) VALUES(?,?,0,?,?)",
                (branch_id, name, parent_branch, parent_seq))

    def close(self) -> None:
        self.conn.close()

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value))

    # ---- 写入组与半写恢复 ----------------------------------------------

    def _journal_start(self, group_id: str, label: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO journal(group_id,label,status,inserts,created_at) "
                "VALUES(?,?,?,?,?)", (group_id, label, "pending", "[]", _now()))

    def _journal_record(self, group_id: str, table: str, pk: tuple) -> None:
        row = self.conn.execute(
            "SELECT inserts FROM journal WHERE group_id=?", (group_id,)).fetchone()
        items = json.loads(row["inserts"])
        items.append([table, list(pk)])
        with self.conn:
            self.conn.execute(
                "UPDATE journal SET inserts=? WHERE group_id=?",
                (json.dumps(items, ensure_ascii=False, sort_keys=True), group_id))

    def _journal_commit(self, group_id: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE journal SET status='committed' WHERE group_id=?", (group_id,))

    def _journal_rollback(self, group_id: str) -> None:
        self._undo_group(group_id)
        with self.conn:
            self.conn.execute(
                "UPDATE journal SET status='rolled_back' WHERE group_id=?", (group_id,))

    def _undo_group(self, group_id: str) -> None:
        row = self.conn.execute(
            "SELECT inserts FROM journal WHERE group_id=?", (group_id,)).fetchone()
        if row is None:
            return
        for table, pk in json.loads(row["inserts"]):
            cols = _PK[table]
            where = " AND ".join(f"{c}=?" for c in cols)
            with self.conn:
                self.conn.execute(f"DELETE FROM {table} WHERE {where}", tuple(pk))

    def _recover(self) -> list[str]:
        """重开时把所有 pending 写组回滚；返回回滚的 group_id。"""
        rows = self.conn.execute(
            "SELECT group_id FROM journal WHERE status='pending'").fetchall()
        recovered: list[str] = []
        for r in rows:
            self._journal_rollback(r["group_id"])
            recovered.append(r["group_id"])
        return recovered

    @contextmanager
    def write_group(self, label: str = "write") -> Iterator["WriteGroup"]:
        gid = f"{label}-{uuid.uuid4().hex[:10]}"
        self._journal_start(gid, label)
        group = WriteGroup(self, gid)
        try:
            yield group
        except BaseException:
            self._journal_rollback(gid)
            raise
        self._journal_commit(gid)

    # ---- 行操作（写入组内） --------------------------------------------

    def insert_row(self, group_id: str, table: str, row: dict[str, Any]) -> None:
        # payload/parents 等字段由调用方预先 json.dumps
        keys = list(row)
        ph = ",".join("?" for _ in keys)
        with self.conn:
            self.conn.execute(
                f"INSERT INTO {table}({','.join(keys)}) VALUES({ph})",
                [row[k] for k in keys])
        self._journal_record(group_id, table, tuple(row[c] for c in _PK[table]))

    def update_head(self, branch: str, seq: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE branches SET head_seq=? WHERE branch_id=?", (seq, branch))

    # ---- 读取 ----------------------------------------------------------

    def head(self, branch: str = "main") -> int:
        row = self.conn.execute(
            "SELECT head_seq FROM branches WHERE branch_id=?", (branch,)).fetchone()
        return int(row["head_seq"]) if row else 0

    def branches(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM branches ORDER BY branch_id").fetchall())

    def append_event(
        self, group: "WriteGroup", branch: str, kind: str,
        payload: dict, parents: list[int]
    ) -> int:
        event_id = f"evt-{uuid.uuid4().hex[:12]}"
        group.insert("events", {
            "event_id": event_id,
            "branch": branch,
            "kind": kind,
            "payload": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            "parents": json.dumps(parents, sort_keys=True),
            "created_at": _now(),
        })
        seq = self.conn.execute("SELECT seq FROM events WHERE event_id=?",
                                (event_id,)).fetchone()["seq"]
        self.update_head(branch, seq)
        return seq

    def events(self, branch: str | None = None) -> list[sqlite3.Row]:
        if branch is None:
            return list(self.conn.execute("SELECT * FROM events ORDER BY seq"))
        return list(self.conn.execute(
            "SELECT * FROM events WHERE branch=? ORDER BY seq", (branch,)))

    def latest_material(self, material_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM materials WHERE material_id=? ORDER BY version DESC LIMIT 1",
            (material_id,)).fetchone()

    def material_at(self, material_id: str, version: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM materials WHERE material_id=? AND version=?",
            (material_id, version)).fetchone()

    def all_materials(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM materials ORDER BY material_id, version"))

    def get_result(self, result_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM results WHERE result_id=?", (result_id,)).fetchone()

    def all_results(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM results ORDER BY created_seq, result_id"))

    def has_any_event(self) -> bool:
        return self.conn.execute("SELECT 1 FROM events LIMIT 1").fetchone() is not None

    def reset_all(self) -> None:
        with self.conn:
            for t in ("journal", "events", "results", "materials", "branches", "meta"):
                self.conn.execute(f"DELETE FROM {t}")
        self._ensure_main_branch()



class WriteGroup:
    def __init__(self, store: "Store", group_id: str):
        self.store = store
        self.group_id = group_id

    def insert(self, table: str, row: dict[str, Any]) -> None:
        self.store.insert_row(self.group_id, table, row)
