"""Append-only SQLite storage.

Layers immutable materials (versioned per kind) from mutable decisions.
Every computation records input digest, parameter version and parent
dependencies. All multi-row writes happen in a single transaction so a crash
can only leave a detectable half-write (status='writing'), never a silent
corruption.
"""
import json
import sqlite3

from .ids import digest, short

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY, value TEXT
);
CREATE TABLE IF NOT EXISTS materials(
  kind TEXT, version INTEGER, content TEXT, digest TEXT,
  seq INTEGER, branch TEXT, PRIMARY KEY(kind, version)
);
CREATE TABLE IF NOT EXISTS events(
  event_id TEXT PRIMARY KEY, seq INTEGER, branch TEXT,
  kind TEXT, payload TEXT, parents TEXT
);
CREATE TABLE IF NOT EXISTS computations(
  result_id TEXT PRIMARY KEY, seq INTEGER, branch TEXT, kind TEXT,
  status TEXT, input_digest TEXT, param_version TEXT,
  parents TEXT, payload TEXT, diagnostics TEXT
);
CREATE TABLE IF NOT EXISTS decisions(
  decision_id TEXT PRIMARY KEY, seq INTEGER, branch TEXT,
  target TEXT, action TEXT, rationale TEXT, payload TEXT
);
"""

STATUS_OK = "ok"
STATUS_BUDGET = "budget_exhausted"
STATUS_FORMAT = "format_corrupt"
STATUS_UNSAT = "unsatisfiable"
STATUS_WRITING = "writing"
STATUS_ABORTED = "aborted"


class Store:
    def __init__(self, path=":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('seq', '0')")

    def close(self):
        self.conn.close()

    def _next_seq(self, cur):
        cur.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key='seq'")
        cur.execute("SELECT value FROM meta WHERE key='seq'")
        return int(cur.fetchone()["value"])

    # ---------------- materials (immutable, versioned per kind) ----------
    def add_material(self, kind, content, branch="main"):
        with self.conn:
            cur = self.conn.cursor()
            seq = self._next_seq(cur)
            cur.execute("SELECT COALESCE(MAX(version),0) AS v FROM materials WHERE kind=:k",
                        {"k": kind})
            version = int(cur.fetchone()["v"]) + 1
            cur.execute(
                "INSERT INTO materials VALUES(:k,:v,:c,:d,:s,:b)",
                {"k": kind, "v": version, "c": json.dumps(content),
                 "d": digest(content), "s": seq, "b": branch})
        return {"material_id": f"{kind}@v{version}", "kind": kind,
                "version": version, "digest": digest(content), "seq": seq}

    def materials(self, seq=None, kind=None):
        q = "SELECT * FROM materials"
        cond, args = [], {}
        if seq is not None:
            cond.append("seq <= :s"); args["s"] = seq
        if kind is not None:
            cond.append("kind = :k"); args["k"] = kind
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY kind, version"
        rows = self.conn.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def current_materials(self, seq=None):
        """Latest version per kind visible at seq (None = latest overall)."""
        out = {}
        for r in self.materials(seq=seq):
            out[r["kind"]] = r  # rows ordered by version asc -> last wins
        return out

    def material(self, kind, version):
        r = self.conn.execute(
            "SELECT * FROM materials WHERE kind=? AND version=?",
            (kind, version)).fetchone()
        return dict(r) if r else None

    # ---------------- events --------------------------------------------
    def add_event(self, kind, payload, parents=None, branch="main"):
        with self.conn:
            cur = self.conn.cursor()
            seq = self._next_seq(cur)
            event_id = f"evt_{seq:04d}"
            cur.execute(
                "INSERT INTO events VALUES(:id,:s,:b,:k,:p,:par)",
                {"id": event_id, "s": seq, "b": branch, "k": kind,
                 "p": json.dumps(payload), "par": json.dumps(parents or [])})
        return {"event_id": event_id, "seq": seq}

    def events(self, seq=None, branch=None):
        q, cond, args = "SELECT * FROM events", [], {}
        if seq is not None:
            cond.append("seq <= :s"); args["s"] = seq
        if branch is not None:
            cond.append("branch = :b"); args["b"] = branch
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY seq"
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    # ---------------- computations --------------------------------------
    def computation_id(self, kind, input_digest, param_version, parents):
        return "cmp_" + short(digest({
            "kind": kind, "input": input_digest,
            "params": param_version, "parents": parents}))

    def add_computation(self, kind, status, input_digest, param_version,
                        parents, payload, diagnostics, branch="main"):
        """Idempotent: identical inputs+params+parents reuse the same id."""
        result_id = self.computation_id(kind, input_digest, param_version, parents)
        existing = self.conn.execute(
            "SELECT result_id FROM computations WHERE result_id=?", (result_id,)
        ).fetchone()
        if existing:
            return {"result_id": result_id, "reused": True}
        with self.conn:
            cur = self.conn.cursor()
            seq = self._next_seq(cur)
            cur.execute(
                "INSERT INTO computations VALUES(:id,:s,:b,:k,:st,:inp,:pv,:par,:pl,:dg)",
                {"id": result_id, "s": seq, "b": branch, "k": kind, "st": status,
                 "inp": input_digest, "pv": param_version,
                 "par": json.dumps(parents), "pl": json.dumps(payload),
                 "dg": json.dumps(diagnostics)})
        return {"result_id": result_id, "reused": False, "seq": seq}

    def computations(self, seq=None, kind=None, branch=None):
        q, cond, args = "SELECT * FROM computations", [], {}
        if seq is not None:
            cond.append("seq <= :s"); args["s"] = seq
        if kind is not None:
            cond.append("kind = :k"); args["k"] = kind
        if branch is not None:
            cond.append("branch = :b"); args["b"] = branch
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY seq"
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    def computation(self, result_id):
        r = self.conn.execute(
            "SELECT * FROM computations WHERE result_id=?", (result_id,)).fetchone()
        return dict(r) if r else None

    # ---------------- decisions -----------------------------------------
    def add_decision(self, target, action, rationale, payload=None, branch="main"):
        with self.conn:
            cur = self.conn.cursor()
            seq = self._next_seq(cur)
            decision_id = f"dec_{seq:04d}"
            cur.execute(
                "INSERT INTO decisions VALUES(:id,:s,:b,:t,:a,:r,:p)",
                {"id": decision_id, "s": seq, "b": branch, "t": target,
                 "a": action, "r": rationale, "p": json.dumps(payload or {})})
        return {"decision_id": decision_id, "seq": seq}

    def decisions(self, seq=None, branch=None):
        q, cond, args = "SELECT * FROM decisions", [], {}
        if seq is not None:
            cond.append("seq <= :s"); args["s"] = seq
        if branch is not None:
            cond.append("branch = :b"); args["b"] = branch
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY seq"
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    # ---------------- misc ----------------------------------------------
    def max_seq(self):
        r = self.conn.execute("SELECT value FROM meta WHERE key='seq'").fetchone()
        return int(r["value"])

    def is_empty(self):
        r = self.conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()
        return r["c"] == 0

    def branches(self):
        found = set()
        for table in ("events", "decisions", "computations", "materials"):
            rows = self.conn.execute(
                f"SELECT DISTINCT branch FROM {table}").fetchall()
            found.update(r["branch"] for r in rows)
        return sorted(found)
