"""SQLite-backed store for the ITC review application.

Layers:
  * materials    - immutable, separately versioned raw inputs
                   (power sampling, injection volumes/times, concentrations,
                   temperature, cell volume, baseline curve, blank curve,
                   blank schedule).  Raw representation is preserved verbatim.
  * decisions    - modifiable human/rule decisions (append-only history).
  * computations - derived results; each records an input digest, a parameter
                   version and its parent dependencies.
  * events       - machine-readable journal; every judgement is an event.

All ids are content-derived (stable): equivalent inputs produce identical
ids and identical ordering, which is what makes export -> delete -> import
preserve identity, order and provenance.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS materials(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  stream_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  raw TEXT NOT NULL,
  units TEXT NOT NULL,
  basis TEXT NOT NULL,
  source TEXT NOT NULL,
  created_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  branch TEXT NOT NULL DEFAULT 'main',
  created_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS computations(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  input_digest TEXT NOT NULL,
  param_version TEXT NOT NULL,
  parents TEXT NOT NULL,
  result TEXT NOT NULL,
  diagnostics TEXT NOT NULL,
  branch TEXT NOT NULL DEFAULT 'main',
  created_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY,
  type TEXT NOT NULL,
  payload TEXT NOT NULL,
  branch TEXT NOT NULL DEFAULT 'main',
  ts TEXT NOT NULL
);
"""

STATUS_OK = "ok"
STATUS_INCOMPLETE = "incomplete_input"
STATUS_INVALID = "invalid_format"
STATUS_UNSAT = "unsatisfiable"
STATUS_BUDGET = "budget_exhausted"
STATUS_RUNNING = "running"
STATUS_INTERRUPTED = "interrupted"


def stable_id(prefix: str, payload) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{prefix}-{hashlib.sha1(blob.encode()).hexdigest()[:12]}"


def digest(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class ImportFormatError(Exception):
    """Raised when an export payload is corrupt or of an unknown format."""


class ImportConflictError(Exception):
    """Raised when an imported id already exists with different content."""


class Store:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._recover_half_writes()

    def close(self):
        self.conn.close()

    # -- half-write recovery ------------------------------------------------
    def _recover_half_writes(self):
        stuck = [r["id"] for r in self.conn.execute(
            "SELECT id FROM computations WHERE status=?", (STATUS_RUNNING,))]
        if stuck:
            self.conn.execute(
                "UPDATE computations SET status=? WHERE status=?",
                (STATUS_INTERRUPTED, STATUS_RUNNING))
            self.conn.commit()
            self.event("recovery.performed", {
                "interrupted": stuck,
                "reason": "computation left in 'running' state by an "
                          "interrupted write; marked 'interrupted' at open",
            })

    # -- events --------------------------------------------------------------
    def next_seq(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 AS s FROM events").fetchone()
        return int(row["s"])

    def event(self, type_: str, payload, branch: str = "main") -> int:
        seq = self.next_seq()
        self.conn.execute(
            "INSERT INTO events(seq,type,payload,branch,ts) VALUES(?,?,?,?,?)",
            (seq, type_, json.dumps(payload, sort_keys=True), branch, utcnow()))
        self.conn.commit()
        return seq

    def events(self, as_of: int | None = None, branch: str | None = None):
        q = "SELECT * FROM events WHERE 1=1"
        args: list = []
        if as_of is not None:
            q += " AND seq<=?"
            args.append(as_of)
        if branch is not None:
            q += " AND branch=?"
            args.append(branch)
        q += " ORDER BY seq"
        return [self._row(r, ("payload",)) for r in
                self.conn.execute(q, args).fetchall()]

    # -- materials -----------------------------------------------------------
    def add_material(self, kind, raw, units, basis, source,
                     revises=None, branch="main"):
        mid = stable_id("mat", {"kind": kind, "raw": raw})
        existing = self.get_material(mid)
        if existing:
            return existing, False
        if revises:
            parent = self.get_material(revises)
            if parent is None:
                raise KeyError(f"revises unknown material {revises}")
            stream_id, version = parent["stream_id"], parent["version"] + 1
        else:
            stream_id, version = mid, 1
        seq = self.event(
            "material.revised" if revises else "material.imported",
            {"id": mid, "kind": kind, "version": version,
             "stream_id": stream_id, "revises": revises, "source": source},
            branch=branch)
        self.conn.execute(
            "INSERT INTO materials(id,kind,stream_id,version,raw,units,basis,"
            "source,created_seq) VALUES(?,?,?,?,?,?,?,?,?)",
            (mid, kind, stream_id, version,
             json.dumps(raw, sort_keys=True), json.dumps(units, sort_keys=True),
             basis, source, seq))
        self.conn.commit()
        return self.get_material(mid), True

    def get_material(self, mid):
        r = self.conn.execute(
            "SELECT * FROM materials WHERE id=?", (mid,)).fetchone()
        return self._row(r, ("raw", "units")) if r else None

    def materials(self, as_of: int | None = None):
        q = "SELECT * FROM materials"
        args: list = []
        if as_of is not None:
            q += " WHERE created_seq<=?"
            args.append(as_of)
        q += " ORDER BY kind, stream_id, version"
        return [self._row(r, ("raw", "units")) for r in
                self.conn.execute(q, args).fetchall()]

    def latest_material(self, kind, as_of: int | None = None):
        q = ("SELECT * FROM materials WHERE kind=?")
        args: list = [kind]
        if as_of is not None:
            q += " AND created_seq<=?"
            args.append(as_of)
        q += " ORDER BY version DESC, created_seq DESC LIMIT 1"
        r = self.conn.execute(q, args).fetchone()
        return self._row(r, ("raw", "units")) if r else None

    # -- decisions -----------------------------------------------------------
    def add_decision(self, kind, payload, branch="main"):
        did = stable_id("dec", {"kind": kind, "payload": payload,
                                "branch": branch})
        existing = self.get_decision(did)
        if existing:
            return existing, False
        seq = self.event("decision.made",
                         {"id": did, "kind": kind, "payload": payload},
                         branch=branch)
        self.conn.execute(
            "INSERT INTO decisions(id,kind,payload,branch,created_seq)"
            " VALUES(?,?,?,?,?)",
            (did, kind, json.dumps(payload, sort_keys=True), branch, seq))
        self.conn.commit()
        return self.get_decision(did), True

    def get_decision(self, did):
        r = self.conn.execute(
            "SELECT * FROM decisions WHERE id=?", (did,)).fetchone()
        return self._row(r, ("payload",)) if r else None

    def decisions(self, kind=None, as_of=None, branch=None):
        q = "SELECT * FROM decisions WHERE 1=1"
        args: list = []
        if kind is not None:
            q += " AND kind=?"
            args.append(kind)
        if as_of is not None:
            q += " AND created_seq<=?"
            args.append(as_of)
        if branch is not None:
            q += " AND branch=?"
            args.append(branch)
        q += " ORDER BY created_seq, id"
        return [self._row(r, ("payload",)) for r in
                self.conn.execute(q, args).fetchall()]

    def latest_decision(self, kind, as_of=None, branch=None):
        ds = self.decisions(kind=kind, as_of=as_of, branch=branch)
        return ds[-1] if ds else None

    # -- computations --------------------------------------------------------
    def add_computation(self, kind, status, input_digest, param_version,
                        parents, result, diagnostics, branch="main"):
        cid = stable_id("cmp", {"kind": kind, "digest": input_digest,
                                "param_version": param_version})
        existing = self.get_computation(cid)
        if existing:
            return existing, False
        seq = self.event(
            "computation.completed" if status == STATUS_OK
            else "computation.failed",
            {"id": cid, "kind": kind, "status": status,
             "diagnostics": diagnostics}, branch=branch)
        self.conn.execute(
            "INSERT INTO computations(id,kind,status,input_digest,"
            "param_version,parents,result,diagnostics,branch,created_seq)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (cid, kind, status, input_digest, param_version,
             json.dumps(parents, sort_keys=True),
             json.dumps(result, sort_keys=True),
             json.dumps(diagnostics, sort_keys=True), branch, seq))
        self.conn.commit()
        return self.get_computation(cid), True

    def get_computation(self, cid):
        r = self.conn.execute(
            "SELECT * FROM computations WHERE id=?", (cid,)).fetchone()
        return self._row(r, ("parents", "result", "diagnostics")) if r else None

    def computations(self, kind=None, as_of=None, branch=None):
        q = "SELECT * FROM computations WHERE 1=1"
        args: list = []
        if kind is not None:
            q += " AND kind=?"
            args.append(kind)
        if as_of is not None:
            q += " AND created_seq<=?"
            args.append(as_of)
        if branch is not None:
            q += " AND branch=?"
            args.append(branch)
        q += " ORDER BY created_seq, id"
        return [self._row(r, ("parents", "result", "diagnostics")) for r in
                self.conn.execute(q, args).fetchall()]

    def latest_computation(self, kind, as_of=None, branch=None):
        cs = self.computations(kind=kind, as_of=as_of, branch=branch)
        return cs[-1] if cs else None

    # -- export / import ------------------------------------------------------
    def export(self) -> dict:
        return {
            "format": "itc-project/1",
            "materials": [dict(r) for r in self.conn.execute(
                "SELECT * FROM materials ORDER BY created_seq, id")],
            "decisions": [dict(r) for r in self.conn.execute(
                "SELECT * FROM decisions ORDER BY created_seq, id")],
            "computations": [dict(r) for r in self.conn.execute(
                "SELECT * FROM computations ORDER BY created_seq, id")],
            "events": [dict(r) for r in self.conn.execute(
                "SELECT * FROM events ORDER BY seq")],
        }

    def import_(self, data):
        """Transactional import.  Any failure rolls back the whole import so
        a half-written transfer never leaves partial state behind."""
        if not isinstance(data, dict) or data.get("format") != "itc-project/1":
            raise ImportFormatError(
                "invalid_format: expected an object with "
                "format == 'itc-project/1'")
        for key in ("materials", "decisions", "computations", "events"):
            if not isinstance(data.get(key), list):
                raise ImportFormatError(
                    f"invalid_format: missing list '{key}'")
        try:
            self.conn.execute("BEGIN")
            for row in data["events"]:
                self._insert_row("events", row,
                                 ("seq", "type", "payload", "branch", "ts"),
                                 key="seq")
            for row in data["materials"]:
                self._insert_row("materials", row,
                                 ("id", "kind", "stream_id", "version", "raw",
                                  "units", "basis", "source", "created_seq"))
            for row in data["decisions"]:
                self._insert_row("decisions", row,
                                 ("id", "kind", "payload", "branch",
                                  "created_seq"))
            for row in data["computations"]:
                self._insert_row("computations", row,
                                 ("id", "kind", "status", "input_digest",
                                  "param_version", "parents", "result",
                                  "diagnostics", "branch", "created_seq"))
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        self.event("import.performed", {
            "materials": len(data["materials"]),
            "decisions": len(data["decisions"]),
            "computations": len(data["computations"]),
            "events": len(data["events"]),
        })

    def _insert_row(self, table, row, cols, key="id"):
        missing = [c for c in cols if c not in row]
        if missing:
            raise ImportFormatError(
                f"invalid_format: {table} row missing {missing}")
        kval = row[key]
        existing = self.conn.execute(
            f"SELECT * FROM {table} WHERE {key}=?", (kval,)).fetchone()
        if existing is not None:
            for c in cols:
                if str(existing[c]) != str(row[c]):
                    raise ImportConflictError(
                        f"import conflict: {table}.{key}={kval} already "
                        f"exists with different content")
            return  # identical row already present: identity preserved
        self.conn.execute(
            f"INSERT INTO {table}({','.join(cols)})"
            f" VALUES({','.join('?' * len(cols))})",
            tuple(row[c] if not isinstance(row[c], (dict, list))
                  else json.dumps(row[c], sort_keys=True) for c in cols))

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _row(r, json_cols):
        d = dict(r)
        for c in json_cols:
            if c in d and isinstance(d[c], str):
                d[c] = json.loads(d[c])
        return d

    def counts(self):
        return {t: self.conn.execute(
            f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
            for t in ("materials", "decisions", "computations", "events")}
