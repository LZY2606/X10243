"""SQLite-backed event-sourced store.

Immutable materials (separately versioned per kind), an append-only
event log, and derived computations that record input digests, parameter
versions and parent dependencies. Branches are movable head pointers;
checking out an old event restores the dependency versions and visible
conclusions of that moment instead of overwriting history.
"""

import hashlib
import json
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS materials (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    version INTEGER NOT NULL,
    units_json TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    hash TEXT NOT NULL,
    created_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    branch TEXT NOT NULL,
    kind TEXT NOT NULL,
    summary TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    parents_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS computations (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    inputs_json TEXT NOT NULL,
    params_json TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL,
    created_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS branches (
    name TEXT PRIMARY KEY,
    forked_from TEXT,
    fork_seq INTEGER,
    head_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

MATERIAL_KINDS = (
    "power_samples", "injection_schedule", "concentrations",
    "temperature", "cell_volume", "baseline_curve", "blank_curve",
)


def content_hash(payload):
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


class Store:
    def __init__(self, path=":memory:"):
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self):
        self.db.close()

    # -- atomic multi-write support (half-write recovery) ----------------
    def transaction(self):
        return _Transaction(self)

    # -- meta -------------------------------------------------------------
    def set_meta(self, key, value):
        self.db.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)))
        self.db.commit()

    def get_meta(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    # -- materials ---------------------------------------------------------
    def add_material(self, kind, raw, units, seq, db=None):
        assert kind in MATERIAL_KINDS, kind
        conn = db or self.db
        version = self.latest_version(kind, db=conn) + 1
        mid = f"mat-{kind}-{version}"
        conn.execute(
            "INSERT INTO materials(id,kind,version,units_json,raw_json,hash,created_seq)"
            " VALUES(?,?,?,?,?,?,?)",
            (mid, kind, version, json.dumps(units), json.dumps(raw),
             content_hash({"kind": kind, "raw": raw, "units": units}), seq))
        if db is None:
            self.db.commit()
        return mid

    def latest_version(self, kind, db=None):
        conn = db or self.db
        row = conn.execute(
            "SELECT COALESCE(MAX(version),0) v FROM materials WHERE kind=?",
            (kind,)).fetchone()
        return row["v"]

    def get_material(self, mid):
        row = self.db.execute("SELECT * FROM materials WHERE id=?", (mid,)).fetchone()
        return _material(row) if row else None

    def materials(self):
        return [_material(r) for r in self.db.execute(
            "SELECT * FROM materials ORDER BY kind, version")]

    # -- events ------------------------------------------------------------
    def add_event(self, branch, kind, summary, payload, parents, seq=None, db=None):
        conn = db or self.db
        if seq is None:
            seq = self.next_seq(db=conn)
        eid = f"evt-{seq:04d}"
        conn.execute(
            "INSERT INTO events(id,seq,branch,kind,summary,payload_json,parents_json)"
            " VALUES(?,?,?,?,?,?,?)",
            (eid, seq, branch, kind, summary, json.dumps(payload), json.dumps(parents)))
        if db is None:
            self.db.commit()
        return eid, seq

    def next_seq(self, db=None):
        conn = db or self.db
        row = conn.execute("SELECT COALESCE(MAX(seq),0) s FROM events").fetchone()
        return row["s"] + 1

    def events(self):
        return [_event(r) for r in self.db.execute(
            "SELECT * FROM events ORDER BY seq, id")]

    def get_event(self, eid):
        row = self.db.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
        return _event(row) if row else None

    # -- computations ------------------------------------------------------
    def add_computation(self, kind, input_digest, inputs, params, status,
                        result, diagnostics, seq, db=None):
        conn = db or self.db
        cid = "cmp-" + content_hash({
            "kind": kind, "digest": input_digest, "params": params})[7:19]
        conn.execute(
            "INSERT OR IGNORE INTO computations"
            "(id,kind,input_digest,inputs_json,params_json,status,result_json,"
            " diagnostics_json,created_seq) VALUES(?,?,?,?,?,?,?,?,?)",
            (cid, kind, input_digest, json.dumps(inputs), json.dumps(params),
             status, json.dumps(result), json.dumps(diagnostics), seq))
        if db is None:
            self.db.commit()
        return cid

    def computations(self):
        return [_computation(r) for r in self.db.execute(
            "SELECT * FROM computations ORDER BY created_seq, id")]

    def get_computation(self, cid):
        row = self.db.execute(
            "SELECT * FROM computations WHERE id=?", (cid,)).fetchone()
        return _computation(row) if row else None

    # -- branches ----------------------------------------------------------
    def ensure_branch(self, name, forked_from=None, fork_seq=0, head_seq=0, db=None):
        conn = db or self.db
        conn.execute(
            "INSERT OR IGNORE INTO branches(name,forked_from,fork_seq,head_seq)"
            " VALUES(?,?,?,?)", (name, forked_from, fork_seq, head_seq))
        if db is None:
            self.db.commit()

    def set_head(self, name, head_seq, db=None):
        conn = db or self.db
        conn.execute("UPDATE branches SET head_seq=? WHERE name=?", (head_seq, name))
        if db is None:
            self.db.commit()

    def branches(self):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM branches ORDER BY name")]

    def get_branch(self, name):
        row = self.db.execute(
            "SELECT * FROM branches WHERE name=?", (name,)).fetchone()
        return dict(row) if row else None

    # -- export / import ---------------------------------------------------
    def dump(self):
        return {
            "format": "itc-review/1",
            "meta": {r["key"]: json.loads(r["value"])
                     for r in self.db.execute("SELECT * FROM meta ORDER BY key")},
            "materials": [dict(r) for r in self.db.execute(
                "SELECT * FROM materials ORDER BY kind, version")],
            "events": [dict(r) for r in self.db.execute(
                "SELECT * FROM events ORDER BY seq, id")],
            "computations": [dict(r) for r in self.db.execute(
                "SELECT * FROM computations ORDER BY created_seq, id")],
            "branches": self.branches(),
        }

    def counts(self):
        return {t: self.db.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
                for t in ("materials", "events", "computations", "branches")}


class _Transaction:
    """All-or-nothing writes: a failed import leaves the DB untouched."""

    def __init__(self, store):
        self.store = store

    def __enter__(self):
        self.store._lock.acquire()
        self.store.db.execute("BEGIN IMMEDIATE")
        return self.store.db

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.store.db.commit()
            else:
                self.store.db.rollback()
        finally:
            self.store._lock.release()
        return False


def _material(row):
    return {"id": row["id"], "kind": row["kind"], "version": row["version"],
            "units": json.loads(row["units_json"]), "raw": json.loads(row["raw_json"]),
            "hash": row["hash"], "created_seq": row["created_seq"]}


def _event(row):
    return {"id": row["id"], "seq": row["seq"], "branch": row["branch"],
            "kind": row["kind"], "summary": row["summary"],
            "payload": json.loads(row["payload_json"]),
            "parents": json.loads(row["parents_json"])}


def _computation(row):
    return {"id": row["id"], "kind": row["kind"],
            "input_digest": row["input_digest"],
            "inputs": json.loads(row["inputs_json"]),
            "params": json.loads(row["params_json"]),
            "status": row["status"], "result": json.loads(row["result_json"]),
            "diagnostics": json.loads(row["diagnostics_json"]),
            "created_seq": row["created_seq"]}
