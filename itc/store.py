"""SQLite storage: immutable versioned materials, derived computations,
machine-readable events and mutable decisions, in separate layers.

- Materials are immutable and versioned per (kind, branch); a revision
  inserts a new row superseding the previous one. Nothing is overwritten.
- Computations record input digest, parameter version and parent
  dependencies, and are written begin/finish so a crash mid-write is
  detectable and recovered on open (never presented as valid).
- Export/import preserves identities, ordering and provenance edges.
"""

import hashlib
import json
import os
import sqlite3

from .errors import FORMAT_CORRUPT, OK, RECOVERED_INCOMPLETE, diag, make_result

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS materials(
  material_id TEXT PRIMARY KEY, kind TEXT, branch TEXT, version INTEGER,
  supersedes TEXT, hash TEXT, raw_json TEXT, seq INTEGER);
CREATE TABLE IF NOT EXISTS computations(
  compute_id TEXT PRIMARY KEY, kind TEXT, branch TEXT, status TEXT,
  input_digest TEXT, param_version TEXT, parents_json TEXT,
  result_json TEXT, diagnostics_json TEXT, seq INTEGER, complete INTEGER);
CREATE TABLE IF NOT EXISTS events(
  event_id TEXT PRIMARY KEY, seq INTEGER, branch TEXT, type TEXT,
  payload_json TEXT, links_json TEXT);
CREATE TABLE IF NOT EXISTS decisions(
  decision_id TEXT PRIMARY KEY, seq INTEGER, branch TEXT,
  target TEXT, action TEXT, payload_json TEXT);
"""

PREFIX = {"materials": "mat", "computations": "cmp", "events": "evt",
          "decisions": "dec"}


def canonical_hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Store:
    def __init__(self, path):
        self.path = path
        conn = self._connect()
        try:
            conn.executescript(SCHEMA)
            if conn.execute(
                    "SELECT value FROM meta WHERE key='seq'").fetchone() is None:
                conn.execute("INSERT INTO meta(key, value) VALUES('seq','0')")
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),))
            conn.commit()
            self._recover(conn)
        finally:
            conn.close()

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # -- recovery ---------------------------------------------------------
    def _recover(self, conn):
        """Half-written computations (complete=0) are quarantined, never
        presented as valid results, and a recovery event is recorded."""
        rows = conn.execute(
            "SELECT compute_id, kind, branch FROM computations WHERE complete=0"
        ).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE computations SET complete=-1, status=? "
                "WHERE compute_id=?",
                (RECOVERED_INCOMPLETE, row["compute_id"]))
        if rows:
            seq = self._next_seq(conn)
            conn.execute(
                "INSERT INTO events(event_id, seq, branch, type, payload_json,"
                " links_json) VALUES(?,?,?,?,?,?)",
                (f"evt-{seq:06d}", seq, "system", "recovery.incomplete_computation",
                 json.dumps({"quarantined": [r["compute_id"] for r in rows],
                             "kinds": [r["kind"] for r in rows]}),
                 json.dumps([r["compute_id"] for r in rows])))
        conn.commit()

    # -- identity / ordering ----------------------------------------------
    def _next_seq(self, conn):
        row = conn.execute("SELECT value FROM meta WHERE key='seq'").fetchone()
        seq = int(row[0]) + 1
        conn.execute("UPDATE meta SET value=? WHERE key='seq'", (str(seq),))
        return seq

    def is_empty(self):
        conn = self._connect()
        try:
            n = conn.execute("SELECT COUNT(*) FROM materials").fetchone()[0]
            return n == 0
        finally:
            conn.close()

    # -- materials (immutable, versioned) ----------------------------------
    def put_material(self, kind, branch, raw):
        conn = self._connect()
        try:
            prev = conn.execute(
                "SELECT material_id, version FROM materials WHERE kind=? AND"
                " branch=? ORDER BY version DESC LIMIT 1",
                (kind, branch)).fetchone()
            version = (prev["version"] + 1) if prev else 1
            supersedes = prev["material_id"] if prev else None
            seq = self._next_seq(conn)
            mid = f"mat-{seq:06d}"
            conn.execute(
                "INSERT INTO materials(material_id, kind, branch, version,"
                " supersedes, hash, raw_json, seq) VALUES(?,?,?,?,?,?,?,?)",
                (mid, kind, branch, version, supersedes,
                 canonical_hash(raw), json.dumps(raw), seq))
            conn.commit()
            return mid
        finally:
            conn.close()

    @staticmethod
    def _mat_row(row):
        return {"material_id": row["material_id"], "kind": row["kind"],
                "branch": row["branch"], "version": row["version"],
                "supersedes": row["supersedes"], "hash": row["hash"],
                "raw": json.loads(row["raw_json"]), "seq": row["seq"]}

    def latest_material(self, kind, branch, at_seq=None):
        conn = self._connect()
        try:
            q = ("SELECT * FROM materials WHERE kind=? AND branch=?")
            args = [kind, branch]
            if at_seq is not None:
                q += " AND seq<=?"
                args.append(at_seq)
            q += " ORDER BY version DESC LIMIT 1"
            row = conn.execute(q, args).fetchone()
            return self._mat_row(row) if row else None
        finally:
            conn.close()

    def get_material(self, material_id):
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM materials WHERE material_id=?",
                (material_id,)).fetchone()
            return self._mat_row(row) if row else None
        finally:
            conn.close()

    def materials(self, branch=None, at_seq=None):
        conn = self._connect()
        try:
            q, args = "SELECT * FROM materials WHERE 1=1", []
            if branch is not None:
                q += " AND branch=?"
                args.append(branch)
            if at_seq is not None:
                q += " AND seq<=?"
                args.append(at_seq)
            q += " ORDER BY seq"
            return [self._mat_row(r) for r in conn.execute(q, args)]
        finally:
            conn.close()

    # -- computations -------------------------------------------------------
    def begin_computation(self, kind, branch, param_version, parents,
                          input_digest):
        """Phase 1 of a two-phase write: the row exists with complete=0."""
        conn = self._connect()
        try:
            seq = self._next_seq(conn)
            cid = f"cmp-{seq:06d}"
            conn.execute(
                "INSERT INTO computations(compute_id, kind, branch, status,"
                " input_digest, param_version, parents_json, result_json,"
                " diagnostics_json, seq, complete) VALUES(?,?,?,?,?,?,?,?,?,?,0)",
                (cid, kind, branch, "computing", input_digest, param_version,
                 json.dumps(parents), None, None, seq))
            conn.commit()
            return cid
        finally:
            conn.close()

    def finish_computation(self, compute_id, status, result, diagnostics):
        """Phase 2: results attached and the row marked complete."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE computations SET status=?, result_json=?,"
                " diagnostics_json=?, complete=1 WHERE compute_id=?",
                (status,
                 json.dumps(result) if result is not None else None,
                 json.dumps(diagnostics or []), compute_id))
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _cmp_row(row):
        return {"compute_id": row["compute_id"], "kind": row["kind"],
                "branch": row["branch"], "status": row["status"],
                "input_digest": row["input_digest"],
                "param_version": row["param_version"],
                "parents": json.loads(row["parents_json"]),
                "result": (json.loads(row["result_json"])
                           if row["result_json"] else None),
                "diagnostics": (json.loads(row["diagnostics_json"])
                                if row["diagnostics_json"] else []),
                "seq": row["seq"], "complete": row["complete"]}

    def computations(self, kind=None, branch=None, at_seq=None,
                     include_incomplete=False):
        conn = self._connect()
        try:
            q, args = "SELECT * FROM computations WHERE 1=1", []
            if not include_incomplete:
                q += " AND complete=1"
            if kind is not None:
                q += " AND kind=?"
                args.append(kind)
            if branch is not None:
                q += " AND branch=?"
                args.append(branch)
            if at_seq is not None:
                q += " AND seq<=?"
                args.append(at_seq)
            q += " ORDER BY seq"
            return [self._cmp_row(r) for r in conn.execute(q, args)]
        finally:
            conn.close()

    def latest_computation(self, kind, branch, at_seq=None):
        rows = self.computations(kind=kind, branch=branch, at_seq=at_seq)
        return rows[-1] if rows else None

    # -- events ---------------------------------------------------------------
    def add_event(self, branch, type_, payload, links=None):
        conn = self._connect()
        try:
            seq = self._next_seq(conn)
            eid = f"evt-{seq:06d}"
            conn.execute(
                "INSERT INTO events(event_id, seq, branch, type, payload_json,"
                " links_json) VALUES(?,?,?,?,?,?)",
                (eid, seq, branch, type_, json.dumps(payload),
                 json.dumps(links or [])))
            conn.commit()
            return eid
        finally:
            conn.close()

    def events(self, branch=None, at_seq=None):
        conn = self._connect()
        try:
            q, args = "SELECT * FROM events WHERE 1=1", []
            if branch is not None:
                q += " AND branch=?"
                args.append(branch)
            if at_seq is not None:
                q += " AND seq<=?"
                args.append(at_seq)
            q += " ORDER BY seq"
            return [{"event_id": r["event_id"], "seq": r["seq"],
                     "branch": r["branch"], "type": r["type"],
                     "payload": json.loads(r["payload_json"]),
                     "links": json.loads(r["links_json"])}
                    for r in conn.execute(q, args)]
        finally:
            conn.close()

    # -- decisions --------------------------------------------------------------
    def add_decision(self, branch, target, action, payload=None):
        conn = self._connect()
        try:
            seq = self._next_seq(conn)
            did = f"dec-{seq:06d}"
            conn.execute(
                "INSERT INTO decisions(decision_id, seq, branch, target, action,"
                " payload_json) VALUES(?,?,?,?,?,?)",
                (did, seq, branch, target, action, json.dumps(payload or {})))
            conn.commit()
            return did
        finally:
            conn.close()

    def decisions(self, branch=None, at_seq=None):
        conn = self._connect()
        try:
            q, args = "SELECT * FROM decisions WHERE 1=1", []
            if branch is not None:
                q += " AND branch=?"
                args.append(branch)
            if at_seq is not None:
                q += " AND seq<=?"
                args.append(at_seq)
            q += " ORDER BY seq"
            return [{"decision_id": r["decision_id"], "seq": r["seq"],
                     "branch": r["branch"], "target": r["target"],
                     "action": r["action"], "payload": json.loads(r["payload_json"])}
                    for r in conn.execute(q, args)]
        finally:
            conn.close()

    # -- export / import ---------------------------------------------------------
    def export_archive(self):
        return {
            "archive": "itc-review",
            "schema_version": SCHEMA_VERSION,
            "meta": {"seq": self._current_seq()},
            "materials": [self._export_row("materials", r) for r in self._dump("materials")],
            "computations": [self._export_row("computations", r) for r in self._dump("computations")],
            "events": [self._export_row("events", r) for r in self._dump("events")],
            "decisions": [self._export_row("decisions", r) for r in self._dump("decisions")],
        }

    def _current_seq(self):
        conn = self._connect()
        try:
            return int(conn.execute(
                "SELECT value FROM meta WHERE key='seq'").fetchone()[0])
        finally:
            conn.close()

    def _dump(self, table):
        conn = self._connect()
        try:
            return [dict(r) for r in conn.execute(
                f"SELECT * FROM {table} ORDER BY seq")]
        finally:
            conn.close()

    @staticmethod
    def _export_row(table, row):
        # JSON columns stay as text so identity/hash survive the round trip
        return row

    REQUIRED_TABLES = ("materials", "computations", "events", "decisions")

    def import_archive(self, data):
        """Replace local content with an exported archive. Structural
        problems yield FORMAT_CORRUPT; nothing is partially imported."""
        if not isinstance(data, dict) or data.get("archive") != "itc-review":
            return make_result(FORMAT_CORRUPT, None,
                               [diag("archive", "not_an_archive",
                                     "payload is not an itc-review archive")])
        if data.get("schema_version") != SCHEMA_VERSION:
            return make_result(FORMAT_CORRUPT, None,
                               [diag("archive/schema_version", "version_mismatch",
                                     "unsupported archive schema version")])
        for table in self.REQUIRED_TABLES:
            if not isinstance(data.get(table), list):
                return make_result(FORMAT_CORRUPT, None,
                                   [diag(f"archive/{table}", "section_missing",
                                         f"archive section {table} missing")])
        try:
            conn = self._connect()
            try:
                conn.execute("BEGIN")
                for table in self.REQUIRED_TABLES + ("meta",):
                    conn.execute(f"DELETE FROM {table}")
                for table in self.REQUIRED_TABLES:
                    for row in data[table]:
                        cols = list(row.keys())
                        conn.execute(
                            f"INSERT INTO {table}({','.join(cols)}) "
                            f"VALUES({','.join('?' * len(cols))})",
                            [row[c] for c in cols])
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES('seq', ?)",
                    (str(data.get("meta", {}).get("seq", 0)),))
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        except (sqlite3.Error, TypeError, KeyError) as exc:
            return make_result(FORMAT_CORRUPT, None,
                               [diag("archive", "import_failed", str(exc))])
        conn = self._connect()
        try:
            self._recover(conn)
        finally:
            conn.close()
        return make_result(OK, {"imported": {t: len(data[t])
                                             for t in self.REQUIRED_TABLES}})
