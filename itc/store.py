"""SQLite storage layer.

Layers:
- materials:    immutable, versioned source data (power sampling, injection
                volumes/times, concentrations, temperature, cell volume,
                baseline, blank curves -- each versioned separately).
- decisions:    modifiable human judgements, branch-aware.
- computations: derived results; every one records an input digest, a
                parameter version and its parent dependencies.
- events:       machine-readable judgements, the replayable evidence trail.

Nothing is ever updated in place except lifecycle markers
(retired_event / staled_event), so any past event seq can be re-visited and
shows the dependency versions and conclusions visible at that time.
"""
import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS materials(
  id TEXT PRIMARY KEY,
  run_id TEXT,
  kind TEXT NOT NULL,
  version INTEGER NOT NULL,
  meta TEXT NOT NULL DEFAULT '{}',
  raw TEXT NOT NULL,
  digest TEXT NOT NULL,
  created_event INTEGER NOT NULL,
  retired_event INTEGER);
CREATE TABLE IF NOT EXISTS computations(
  id TEXT PRIMARY KEY,
  run_id TEXT,
  kind TEXT NOT NULL,
  subject TEXT NOT NULL DEFAULT '',
  branch TEXT NOT NULL DEFAULT 'main',
  status TEXT NOT NULL,
  input_digest TEXT NOT NULL,
  param_version TEXT NOT NULL,
  parents TEXT NOT NULL,
  result TEXT,
  diagnostics TEXT NOT NULL DEFAULT '[]',
  created_event INTEGER NOT NULL,
  staled_event INTEGER);
CREATE TABLE IF NOT EXISTS decisions(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  branch TEXT NOT NULL DEFAULT 'main',
  payload TEXT NOT NULL,
  created_event INTEGER NOT NULL,
  retired_event INTEGER);
"""

STATUS_OK = "ok"
STATUS_DIAGNOSTIC = "diagnostic"
STATUS_BUDGET = "budget_exhausted"
STATUS_FORMAT = "format_corrupt"
STATUS_SEMANTIC = "semantic_unsolvable"


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def new_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@contextmanager
def _tx(conn):
    try:
        yield
        conn.commit()
    except Exception:
        conn.rollback()
        raise


class Store:
    def __init__(self, path=":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.close()

    def transaction(self):
        return _tx(self.conn)

    # -- events ------------------------------------------------------------
    def event(self, kind, payload, seq=None):
        if seq is None:
            cur = self.conn.execute(
                "INSERT INTO events(kind,payload) VALUES(?,?)",
                (kind, json.dumps(payload, sort_keys=True)))
        else:
            cur = self.conn.execute(
                "INSERT INTO events(seq,kind,payload) VALUES(?,?,?)",
                (seq, kind, json.dumps(payload, sort_keys=True)))
        return cur.lastrowid

    def events(self, at_seq=None):
        sql = "SELECT * FROM events"
        args = []
        if at_seq is not None:
            sql += " WHERE seq<=?"
            args.append(at_seq)
        sql += " ORDER BY seq"
        return [dict(r) for r in self.conn.execute(sql, args)]

    def max_seq(self):
        return self.conn.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0]

    # -- materials ---------------------------------------------------------
    def add_material(self, run_id, kind, raw, meta=None, event_seq=None,
                     mid=None, version=None):
        mid = mid or new_id("mat")
        if version is None:
            version = 1 + self.conn.execute(
                "SELECT COUNT(*) FROM materials WHERE run_id IS ? AND kind=?",
                (run_id, kind)).fetchone()[0]
        meta = meta or {}
        self.conn.execute(
            "INSERT INTO materials(id,run_id,kind,version,meta,raw,digest,created_event)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (mid, run_id, kind, version, json.dumps(meta, sort_keys=True), raw,
             digest({"kind": kind, "raw": raw, "meta": meta}), event_seq))
        return mid

    def retire_material(self, mid, seq):
        self.conn.execute("UPDATE materials SET retired_event=? WHERE id=?", (seq, mid))

    def materials(self, run_id=None, kind=None, at_seq=None):
        sql = "SELECT * FROM materials WHERE 1=1"
        args = []
        if run_id is not None:
            sql += " AND run_id=?"
            args.append(run_id)
        if kind is not None:
            sql += " AND kind=?"
            args.append(kind)
        if at_seq is not None:
            sql += " AND created_event<=? AND (retired_event IS NULL OR retired_event>?)"
            args += [at_seq, at_seq]
        sql += " ORDER BY created_event, id"
        return [dict(r) for r in self.conn.execute(sql, args)]

    def latest_material(self, run_id, kind, at_seq=None):
        rows = self.materials(run_id=run_id, kind=kind, at_seq=at_seq)
        if not rows:
            return None
        return max(rows, key=lambda r: (r["version"], r["created_event"]))

    # -- computations ------------------------------------------------------
    def add_computation(self, run_id, kind, subject, branch, status, input_digest,
                        param_version, parents, result=None, diagnostics=None,
                        event_seq=None, cid=None):
        cid = cid or new_id("cmp")
        self.conn.execute(
            "INSERT INTO computations(id,run_id,kind,subject,branch,status,input_digest,"
            "param_version,parents,result,diagnostics,created_event)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (cid, run_id, kind, subject, branch, status, input_digest, param_version,
             json.dumps(list(parents), sort_keys=True),
             json.dumps(result, sort_keys=True) if result is not None else None,
             json.dumps(diagnostics or [], sort_keys=True), event_seq))
        return cid

    def update_result(self, cid, result):
        self.conn.execute("UPDATE computations SET result=? WHERE id=?",
                          (json.dumps(result, sort_keys=True), cid))

    def stale_computation(self, cid, seq):
        self.conn.execute("UPDATE computations SET staled_event=? WHERE id=?", (seq, cid))

    def computations(self, run_id=None, kind=None, branch=None, at_seq=None,
                     include_staled=False):
        sql = "SELECT * FROM computations WHERE 1=1"
        args = []
        if run_id is not None:
            sql += " AND run_id=?"
            args.append(run_id)
        if kind is not None:
            sql += " AND kind=?"
            args.append(kind)
        if branch is not None:
            sql += " AND branch=?"
            args.append(branch)
        if at_seq is not None:
            sql += " AND created_event<=? AND (staled_event IS NULL OR staled_event>?)"
            args += [at_seq, at_seq]
        elif not include_staled:
            sql += " AND staled_event IS NULL"
        sql += " ORDER BY created_event, id"
        return [dict(r) for r in self.conn.execute(sql, args)]

    def latest_computation(self, run_id, kind, branch=None, at_seq=None, subject=None):
        rows = self.computations(run_id=run_id, kind=kind, branch=branch, at_seq=at_seq)
        if subject is not None:
            rows = [r for r in rows if r["subject"] == subject]
        return rows[-1] if rows else None

    # -- decisions ---------------------------------------------------------
    def add_decision(self, kind, payload, branch="main", event_seq=None, did=None):
        did = did or new_id("dec")
        self.conn.execute(
            "INSERT INTO decisions(id,kind,branch,payload,created_event) VALUES(?,?,?,?,?)",
            (did, kind, branch, json.dumps(payload, sort_keys=True), event_seq))
        return did

    def retire_decision(self, did, seq):
        self.conn.execute("UPDATE decisions SET retired_event=? WHERE id=?", (seq, did))

    def decisions(self, kind=None, branch=None, at_seq=None):
        sql = "SELECT * FROM decisions WHERE 1=1"
        args = []
        if kind is not None:
            sql += " AND kind=?"
            args.append(kind)
        if branch is not None:
            sql += " AND branch=?"
            args.append(branch)
        if at_seq is not None:
            sql += " AND created_event<=? AND (retired_event IS NULL OR retired_event>?)"
            args += [at_seq, at_seq]
        sql += " ORDER BY created_event, id"
        return [dict(r) for r in self.conn.execute(sql, args)]

    # -- lookup / provenance ----------------------------------------------
    def entity(self, eid):
        for table in ("materials", "computations", "decisions"):
            row = self.conn.execute(f"SELECT * FROM {table} WHERE id=?", (eid,)).fetchone()
            if row:
                return table, dict(row)
        return None, None

    def children(self, eid):
        out = []
        for r in self.conn.execute("SELECT * FROM computations ORDER BY created_event, id"):
            if eid in json.loads(r["parents"]):
                out.append(dict(r))
        return out

    def reset(self):
        with self.transaction():
            for table in ("events", "materials", "computations", "decisions"):
                self.conn.execute(f"DELETE FROM {table}")

    def counts(self):
        return {t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("events", "materials", "computations", "decisions")}
