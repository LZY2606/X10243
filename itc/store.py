"""SQLite persistence: immutable versioned materials, computations,
decisions, machine-readable events, diagnostics, write-intent recovery.

Layering:
  * materials   - immutable, versioned separately per kind
  * decisions   - mutable human layer (adopted / rejected / superseded)
  * computations- derived results with input digest, parameter version
                  and parent dependencies
  * events      - append-only, machine-readable judgement log
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS materials(
  uid TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  version INTEGER NOT NULL,
  seq INTEGER NOT NULL,
  payload TEXT NOT NULL,
  derived TEXT,
  UNIQUE(kind, version)
);
CREATE TABLE IF NOT EXISTS computations(
  uid TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  branch TEXT NOT NULL DEFAULT 'main',
  seq INTEGER NOT NULL,
  input_digest TEXT NOT NULL,
  param_version TEXT NOT NULL,
  parents TEXT NOT NULL,
  status TEXT NOT NULL,
  result TEXT
);
CREATE TABLE IF NOT EXISTS decisions(
  uid TEXT PRIMARY KEY,
  seq INTEGER NOT NULL,
  branch TEXT NOT NULL DEFAULT 'main',
  target TEXT NOT NULL,
  action TEXT NOT NULL,
  value TEXT,
  rationale TEXT,
  status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  uid TEXT NOT NULL,
  branch TEXT NOT NULL DEFAULT 'main',
  type TEXT NOT NULL,
  payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS diagnostics(
  uid TEXT PRIMARY KEY,
  computation_uid TEXT,
  seq INTEGER NOT NULL,
  severity TEXT NOT NULL,
  position TEXT NOT NULL,
  code TEXT NOT NULL,
  message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS write_intents(
  uid TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  description TEXT
);
"""

MATERIAL_KINDS = (
    "power_sampling", "injection_schedule", "concentrations",
    "temperature", "cell_volume", "baseline", "blank_curve",
)

# computation kind -> material kinds it depends on
DEPENDS_ON = {
    "integration": ("power_sampling", "injection_schedule", "baseline"),
    "blank_correction": ("blank_curve", "injection_schedule"),
    "fit": ("concentrations", "temperature", "cell_volume",
            "injection_schedule"),
}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def new_uid():
    return uuid.uuid4().hex


class Store:
    def __init__(self, path=":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self.recover()

    # ---- half-write recovery -------------------------------------------
    def recover(self):
        """Any intent left 'pending' crashed before its transaction
        committed, so its effects are absent (transactions are atomic);
        mark it aborted.  Returns the aborted intent uids."""
        rows = self.conn.execute(
            "SELECT uid FROM write_intents WHERE status='pending'").fetchall()
        for r in rows:
            self.conn.execute(
                "UPDATE write_intents SET status='aborted' WHERE uid=?",
                (r["uid"],))
        self.conn.commit()
        return [r["uid"] for r in rows]

    def transaction(self, description=""):
        intent = new_uid()
        self.conn.execute(
            "INSERT INTO write_intents(uid,status,description) "
            "VALUES(?,?,?)", (intent, "pending", description))
        self.conn.commit()
        return _Transaction(self, intent)

    # ---- events ---------------------------------------------------------
    def record_event(self, type_, payload, branch="main", conn=None):
        c = conn or self.conn
        uid = new_uid()
        cur = c.execute(
            "INSERT INTO events(uid,branch,type,payload) VALUES(?,?,?,?)",
            (uid, branch, type_, canonical(payload)))
        return cur.lastrowid, uid

    # ---- materials ------------------------------------------------------
    def add_material(self, kind, payload, derived, branch="main", conn=None,
                     uid=None, version=None, seq=None):
        c = conn or self.conn
        if version is None:
            row = c.execute(
                "SELECT MAX(version) v FROM materials WHERE kind=?",
                (kind,)).fetchone()
            version = (row["v"] or 0) + 1
        uid = uid or new_uid()
        if seq is None:
            _, seq = self._event(c, "material_versioned", {
                "kind": kind, "version": version, "uid": uid,
                "supersedes": version - 1 if version > 1 else None,
            }, branch)
        c.execute(
            "INSERT INTO materials(uid,kind,version,seq,payload,derived) "
            "VALUES(?,?,?,?,?,?)",
            (uid, kind, version, seq, canonical(payload),
             canonical(derived) if derived is not None else None))
        return uid, version, seq

    def _event(self, c, type_, payload, branch):
        uid = new_uid()
        cur = c.execute(
            "INSERT INTO events(uid,branch,type,payload) VALUES(?,?,?,?)",
            (uid, branch, type_, canonical(payload)))
        return uid, cur.lastrowid

    def latest_material(self, kind, at_seq=None):
        sql = "SELECT * FROM materials WHERE kind=?"
        args = [kind]
        if at_seq is not None:
            sql += " AND seq<=?"
            args.append(at_seq)
        sql += " ORDER BY version DESC LIMIT 1"
        row = self.conn.execute(sql, args).fetchone()
        return _mat(row) if row else None

    def materials(self, at_seq=None):
        out = []
        for kind in MATERIAL_KINDS:
            m = self.latest_material(kind, at_seq)
            if m:
                out.append(m)
        return out

    # ---- computations ---------------------------------------------------
    def add_computation(self, kind, input_digest, param_version, parents,
                        status, result, branch="main", conn=None, uid=None,
                        seq=None):
        c = conn or self.conn
        uid = uid or new_uid()
        if seq is None:
            _, seq = self._event(c, "computation_recorded", {
                "kind": kind, "uid": uid, "status": status,
                "input_digest": input_digest,
                "param_version": param_version,
                "parents": parents,
            }, branch)
        c.execute(
            "INSERT INTO computations(uid,kind,branch,seq,input_digest,"
            "param_version,parents,status,result) VALUES(?,?,?,?,?,?,?,?,?)",
            (uid, kind, branch, seq, input_digest, param_version,
             canonical(parents), status,
             canonical(result) if result is not None else None))
        return uid, seq

    def add_diagnostics(self, diags, computation_uid=None, conn=None,
                        seq=None):
        c = conn or self.conn
        for d in diags:
            c.execute(
                "INSERT INTO diagnostics(uid,computation_uid,seq,severity,"
                "position,code,message) VALUES(?,?,?,?,?,?,?)",
                (new_uid(), computation_uid,
                 seq or self._now_seq(c), d["severity"], d["position"],
                 d["code"], d["message"]))

    def _now_seq(self, c):
        row = c.execute("SELECT MAX(seq) s FROM events").fetchone()
        return row["s"] or 0

    def computations(self, kind=None, at_seq=None, branch=None):
        sql, args = "SELECT * FROM computations WHERE 1=1", []
        if kind:
            sql += " AND kind=?"; args.append(kind)
        if at_seq is not None:
            sql += " AND seq<=?"; args.append(at_seq)
        if branch:
            sql += " AND branch=?"; args.append(branch)
        sql += " ORDER BY seq, uid"
        return [_comp(r) for r in self.conn.execute(sql, args)]

    # ---- decisions ------------------------------------------------------
    def add_decision(self, target, action, value, rationale, status,
                     branch="main", conn=None, uid=None, seq=None):
        c = conn or self.conn
        uid = uid or new_uid()
        if seq is None:
            _, seq = self._event(c, "decision_recorded", {
                "uid": uid, "target": target, "action": action,
                "value": value, "rationale": rationale, "status": status,
            }, branch)
        c.execute(
            "INSERT INTO decisions(uid,seq,branch,target,action,value,"
            "rationale,status) VALUES(?,?,?,?,?,?,?,?)",
            (uid, seq, branch, target, action,
             canonical(value) if value is not None else None, rationale,
             status))
        return uid, seq

    def decisions(self, at_seq=None, branch=None):
        sql, args = "SELECT * FROM decisions WHERE 1=1", []
        if at_seq is not None:
            sql += " AND seq<=?"; args.append(at_seq)
        if branch:
            sql += " AND branch=?"; args.append(branch)
        sql += " ORDER BY seq, uid"
        return [dict(r) for r in self.conn.execute(sql, args)]

    # ---- queries --------------------------------------------------------
    def events(self, at_seq=None, branch=None):
        sql, args = "SELECT * FROM events WHERE 1=1", []
        if at_seq is not None:
            sql += " AND seq<=?"; args.append(at_seq)
        if branch:
            sql += " AND branch=?"; args.append(branch)
        sql += " ORDER BY seq"
        return [dict(r) for r in self.conn.execute(sql, args)]

    def diagnostics(self, computation_uid=None):
        sql, args = "SELECT * FROM diagnostics WHERE 1=1", []
        if computation_uid:
            sql += " AND computation_uid=?"; args.append(computation_uid)
        sql += " ORDER BY seq, position"
        return [dict(r) for r in self.conn.execute(sql, args)]

    def max_seq(self):
        row = self.conn.execute("SELECT MAX(seq) s FROM events").fetchone()
        return row["s"] or 0

    def stale_uids(self, at_seq=None):
        """Computations whose parent material versions are superseded by a
        newer version visible at ``at_seq`` (None = now)."""
        latest = {}
        for m in self.materials(at_seq):
            latest[m["kind"]] = m["version"]
        dec_seq = self.max_seq() if at_seq is None else at_seq
        stale = []
        for c in self.computations(at_seq=at_seq):
            for kind, ver, _uid in c["parents"]:
                if kind == "decisions":
                    if self._decision_seq_after(ver, dec_seq):
                        stale.append(c["uid"]); break
                elif kind in latest and latest[kind] > ver:
                    stale.append(c["uid"]); break
        return stale

    def _decision_seq_after(self, seq, upto):
        row = self.conn.execute(
            "SELECT 1 FROM decisions WHERE seq>? AND seq<=? AND "
            "status='adopted' LIMIT 1", (seq, upto)).fetchone()
        return row is not None

    def close(self):
        self.conn.close()


class _Transaction:
    def __init__(self, store, intent):
        self.store, self.intent = store, intent

    def __enter__(self):
        self.conn = self.store.conn
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.conn.commit()
            self.conn.execute(
                "UPDATE write_intents SET status='committed' WHERE uid=?",
                (self.intent,))
            self.conn.commit()
        else:
            self.conn.rollback()
            self.conn.execute(
                "UPDATE write_intents SET status='aborted' WHERE uid=?",
                (self.intent,))
            self.conn.commit()
        return False


def _mat(row):
    return {"uid": row["uid"], "kind": row["kind"], "version": row["version"],
            "seq": row["seq"], "payload": json.loads(row["payload"]),
            "derived": json.loads(row["derived"]) if row["derived"] else None}


def _comp(row):
    return {"uid": row["uid"], "kind": row["kind"], "branch": row["branch"],
            "seq": row["seq"], "input_digest": row["input_digest"],
            "param_version": row["param_version"],
            "parents": json.loads(row["parents"]),
            "status": row["status"],
            "result": json.loads(row["result"]) if row["result"] else None}
