"""SQLite-backed event store.

Layers:
  * materials  - immutable, separately versioned per kind
                 (power_trace / injections / concentrations / conditions /
                  baseline / blank_trace)
  * results    - derived computations; every result records an input digest,
                 a parameter version and its parent dependencies
  * events     - the decision/revision journal (branches supported)

Every write happens inside a batch.  A batch is only visible after its commit
marker lands; on open, recover() removes rows from batches that were half
written (e.g. process killed mid-transaction).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    committed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS materials (
    material_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    hash TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    PRIMARY KEY (material_id, version)
);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    branch TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    parents TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    result_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    param_version TEXT NOT NULL,
    parents TEXT NOT NULL,
    payload TEXT NOT NULL,
    batch_id TEXT NOT NULL
);
"""

# result statuses - the three failure modes are deliberately distinct
STATUS_OK = "ok"
STATUS_INCOMPLETE = "incomplete_input"     # missing/invalid inputs, diagnostics attached
STATUS_BUDGET = "budget_exhausted"         # resource budget ran out
STATUS_CORRUPT = "corrupt_format"          # bytes/JSON could not be parsed
STATUS_UNSOLVABLE = "semantic_unsolvable"  # parsed fine, but no solution exists


def digest(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Store:
    def __init__(self, path: str):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.recover()

    def close(self):
        self.db.close()

    # ---- half-write recovery -------------------------------------------------
    def recover(self):
        """Drop every row whose batch never received its commit marker."""
        pending = [r["batch_id"] for r in self.db.execute(
            "SELECT batch_id FROM batches WHERE committed = 0")]
        for bid in pending:
            for table in ("materials", "events", "results"):
                self.db.execute(f"DELETE FROM {table} WHERE batch_id = ?", (bid,))
            self.db.execute("DELETE FROM batches WHERE batch_id = ?", (bid,))
        self.db.commit()
        return len(pending)

    # ---- batches -------------------------------------------------------------
    def begin_batch(self) -> str:
        bid = new_id("batch")
        self.db.execute(
            "INSERT INTO batches VALUES (?, 0, ?)", (bid, _now()))
        self.db.commit()
        return bid

    def commit_batch(self, bid: str):
        self.db.execute("UPDATE batches SET committed = 1 WHERE batch_id = ?", (bid,))
        self.db.commit()

    # ---- materials -----------------------------------------------------------
    def put_material(self, kind: str, content: dict, batch_id: str,
                     material_id: str | None = None) -> tuple[str, int]:
        mid = material_id or new_id(kind)
        row = self.db.execute(
            "SELECT MAX(version) v FROM materials WHERE material_id = ?", (mid,)
        ).fetchone()
        version = (row["v"] or 0) + 1
        self.db.execute(
            "INSERT INTO materials VALUES (?,?,?,?,?,?)",
            (mid, version, kind, json.dumps(content, sort_keys=True),
             digest(content), batch_id))
        self.db.commit()
        return mid, version

    def get_material(self, material_id: str, version: int | None = None):
        if version is None:
            row = self.db.execute(
                "SELECT * FROM materials WHERE material_id = ? "
                "ORDER BY version DESC LIMIT 1", (material_id,)).fetchone()
        else:
            row = self.db.execute(
                "SELECT * FROM materials WHERE material_id = ? AND version = ?",
                (material_id, version)).fetchone()
        if row is None:
            return None
        return {"material_id": row["material_id"], "version": row["version"],
                "kind": row["kind"], "content": json.loads(row["content"]),
                "hash": row["hash"]}

    def material_versions(self, kind: str | None = None):
        q = "SELECT * FROM materials"
        args = ()
        if kind:
            q += " WHERE kind = ?"
            args = (kind,)
        q += " ORDER BY material_id, version"
        return [{"material_id": r["material_id"], "version": r["version"],
                 "kind": r["kind"], "hash": r["hash"],
                 "content": json.loads(r["content"])}
                for r in self.db.execute(q, args)]

    # ---- events --------------------------------------------------------------
    def _next_seq(self) -> int:
        row = self.db.execute("SELECT MAX(seq) s FROM events").fetchone()
        return (row["s"] or 0) + 1

    def add_event(self, kind: str, payload: dict, parents: list[str],
                  branch: str = "main", batch_id: str | None = None,
                  event_id: str | None = None, seq: int | None = None,
                  created_at: str | None = None) -> dict:
        own_batch = batch_id is None
        bid = batch_id or self.begin_batch()
        eid = event_id or new_id("evt")
        s = seq if seq is not None else self._next_seq()
        evt = {"event_id": eid, "seq": s, "branch": branch, "kind": kind,
               "payload": payload, "parents": parents,
               "created_at": created_at or _now()}
        self.db.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?,?)",
            (eid, s, branch, kind, json.dumps(payload, sort_keys=True),
             json.dumps(parents), bid, evt["created_at"]))
        if own_batch:
            self.commit_batch(bid)
        else:
            self.db.commit()
        return evt

    def events(self, upto_seq: int | None = None):
        q = "SELECT * FROM events"
        args = ()
        if upto_seq is not None:
            q += " WHERE seq <= ?"
            args = (upto_seq,)
        q += " ORDER BY seq, event_id"   # stable order for equivalent inputs
        return [{"event_id": r["event_id"], "seq": r["seq"], "branch": r["branch"],
                 "kind": r["kind"], "payload": json.loads(r["payload"]),
                 "parents": json.loads(r["parents"]),
                 "created_at": r["created_at"]}
                for r in self.db.execute(q, args)]

    # ---- results -------------------------------------------------------------
    def add_result(self, kind: str, status: str, payload: dict,
                   parents: list[str], param_version: str,
                   batch_id: str | None = None,
                   result_id: str | None = None, seq: int | None = None) -> dict:
        own_batch = batch_id is None
        bid = batch_id or self.begin_batch()
        rid = result_id or new_id("res")
        s = seq if seq is not None else self._next_seq()
        res = {"result_id": rid, "seq": s, "kind": kind, "status": status,
               "input_digest": digest({"parents": parents, "param": param_version,
                                       "payload_in": payload.get("inputs")}),
               "param_version": param_version, "parents": parents,
               "payload": payload}
        self.db.execute(
            "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?)",
            (rid, s, kind, status, res["input_digest"], param_version,
             json.dumps(parents), json.dumps(payload, sort_keys=True), bid))
        if own_batch:
            self.commit_batch(bid)
        else:
            self.db.commit()
        return res

    def results(self, kind: str | None = None, upto_seq: int | None = None):
        q, args = "SELECT * FROM results WHERE 1=1", []
        if kind:
            q += " AND kind = ?"
            args.append(kind)
        if upto_seq is not None:
            q += " AND seq <= ?"
            args.append(upto_seq)
        q += " ORDER BY seq, result_id"
        return [{"result_id": r["result_id"], "seq": r["seq"], "kind": r["kind"],
                 "status": r["status"], "input_digest": r["input_digest"],
                 "param_version": r["param_version"],
                 "parents": json.loads(r["parents"]),
                 "payload": json.loads(r["payload"])}
                for r in self.db.execute(q, args)]

    # ---- export / import -----------------------------------------------------
    def export(self) -> dict:
        return {
            "format": "itc-workbench-export",
            "format_version": 1,
            "exported_at": _now(),
            "materials": [
                {"material_id": r["material_id"], "version": r["version"],
                 "kind": r["kind"], "content": json.loads(r["content"]),
                 "hash": r["hash"]}
                for r in self.db.execute(
                    "SELECT * FROM materials ORDER BY material_id, version")],
            "events": self.events(),
            "results": self.results(),
        }

    def import_(self, data: dict) -> dict:
        if data.get("format") != "itc-workbench-export":
            raise ValueError("corrupt_format: not an itc-workbench-export document")
        bid = self.begin_batch()
        try:
            for m in data["materials"]:
                self.db.execute(
                    "INSERT OR REPLACE INTO materials VALUES (?,?,?,?,?,?)",
                    (m["material_id"], m["version"], m["kind"],
                     json.dumps(m["content"], sort_keys=True), m["hash"], bid))
            for e in data["events"]:
                self.db.execute(
                    "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?)",
                    (e["event_id"], e["seq"], e["branch"], e["kind"],
                     json.dumps(e["payload"], sort_keys=True),
                     json.dumps(e["parents"]), bid, e["created_at"]))
            for r in data["results"]:
                self.db.execute(
                    "INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?,?)",
                    (r["result_id"], r["seq"], r["kind"], r["status"],
                     r["input_digest"], r["param_version"],
                     json.dumps(r["parents"]),
                     json.dumps(r["payload"], sort_keys=True), bid))
            self.commit_batch(bid)
        except Exception:
            self.db.execute("DELETE FROM batches WHERE batch_id = ?", (bid,))
            self.db.commit()
            self.recover()
            raise
        return {"imported_materials": len(data["materials"]),
                "imported_events": len(data["events"]),
                "imported_results": len(data["results"])}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
