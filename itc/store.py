"""SQLite event-sourced store.

Layers:
- ``materials``: immutable, separately versioned per kind;
- ``events``: append-only log (imports, revisions, runs, decisions,
  rule judgments), carries branch and parents;
- ``results``: computed results with input digest, parameter version and
  parent dependencies; identical inputs reproduce the identical id.

Export/import preserves identity, order, branches and relations.  Import is
atomic: a half-written or corrupt payload leaves the database untouched and
reports ``format_corrupt`` (distinct from ``budget_exhausted`` and
``unsatisfiable``).
"""

import contextlib
import hashlib
import json
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS materials(
  mat_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  version INTEGER NOT NULL,
  raw_json TEXT NOT NULL,
  norm_json TEXT,
  diag_json TEXT NOT NULL DEFAULT '[]',
  created_seq INTEGER NOT NULL,
  branch TEXT NOT NULL DEFAULT 'main'
);
CREATE TABLE IF NOT EXISTS events(
  event_id TEXT PRIMARY KEY,
  seq INTEGER NOT NULL,
  branch TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  parents_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS results(
  result_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  input_digest TEXT NOT NULL,
  param_version TEXT NOT NULL,
  parents_json TEXT NOT NULL,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_seq INTEGER NOT NULL,
  branch TEXT NOT NULL DEFAULT 'main'
);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
"""

EXPORT_FORMAT = "itc-project/1"


def digest_of(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class Store:
    def __init__(self, path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.close()

    @contextlib.contextmanager
    def atomic(self):
        self.conn.execute("BEGIN")
        try:
            yield
        except Exception:
            self.conn.rollback()
            raise
        self.conn.commit()

    # -- sequence ---------------------------------------------------------
    def current_seq(self):
        row = self.conn.execute("SELECT v FROM meta WHERE k='seq'").fetchone()
        return int(row["v"]) if row else 0

    def _bump_seq(self):
        s = self.current_seq() + 1
        self.conn.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('seq',?)", (str(s),))
        return s

    def is_empty(self):
        return self.conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"] == 0

    # -- writes -----------------------------------------------------------
    def add_material(self, kind, raw, norm, diagnostics, branch="main"):
        row = self.conn.execute(
            "SELECT COALESCE(MAX(version),0) v FROM materials WHERE kind=?",
            (kind,)).fetchone()
        version = row["v"] + 1
        seq = self._bump_seq()
        mat_id = f"mat:{kind}:v{version}"
        self.conn.execute(
            "INSERT INTO materials(mat_id,kind,version,raw_json,norm_json,diag_json,created_seq,branch)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (mat_id, kind, version, json.dumps(raw),
             json.dumps(norm) if norm is not None else None,
             json.dumps(diagnostics), seq, branch))
        return {"mat_id": mat_id, "kind": kind, "version": version,
                "raw": raw, "norm": norm, "diagnostics": diagnostics,
                "created_seq": seq, "branch": branch}

    def add_event(self, branch, kind, payload, parents=None):
        seq = self._bump_seq()
        event_id = f"evt:{seq:05d}"
        parents = parents or []
        self.conn.execute(
            "INSERT INTO events(event_id,seq,branch,kind,payload_json,parents_json)"
            " VALUES(?,?,?,?,?,?)",
            (event_id, seq, branch, kind, json.dumps(payload), json.dumps(parents)))
        return {"event_id": event_id, "seq": seq, "branch": branch, "kind": kind,
                "payload": payload, "parents": parents}

    def add_result(self, kind, input_digest, param_version, parents, status, payload,
                   branch="main"):
        result_id = f"res:{kind}:{input_digest[:12]}"
        self.conn.execute(
            "INSERT OR IGNORE INTO results"
            "(result_id,kind,input_digest,param_version,parents_json,status,payload_json,created_seq,branch)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (result_id, kind, input_digest, param_version, json.dumps(parents),
             status, json.dumps(payload), self.current_seq(), branch))
        return result_id

    # -- reads ------------------------------------------------------------
    def _rows(self, sql, args=()):
        return [dict(r) for r in self.conn.execute(sql, args)]

    def materials(self):
        out = []
        for r in self._rows("SELECT * FROM materials ORDER BY created_seq, kind, version"):
            out.append({"mat_id": r["mat_id"], "kind": r["kind"], "version": r["version"],
                        "raw": json.loads(r["raw_json"]),
                        "norm": json.loads(r["norm_json"]) if r["norm_json"] else None,
                        "diagnostics": json.loads(r["diag_json"]),
                        "created_seq": r["created_seq"], "branch": r["branch"]})
        return out

    def events(self):
        out = []
        for r in self._rows("SELECT * FROM events ORDER BY seq"):
            out.append({"event_id": r["event_id"], "seq": r["seq"], "branch": r["branch"],
                        "kind": r["kind"], "payload": json.loads(r["payload_json"]),
                        "parents": json.loads(r["parents_json"])})
        return out

    def results(self):
        out = []
        for r in self._rows("SELECT * FROM results ORDER BY created_seq, result_id"):
            out.append({"result_id": r["result_id"], "kind": r["kind"],
                        "input_digest": r["input_digest"],
                        "param_version": r["param_version"],
                        "parents": json.loads(r["parents_json"]),
                        "status": r["status"], "payload": json.loads(r["payload_json"]),
                        "created_seq": r["created_seq"], "branch": r["branch"]})
        return out

    def latest_result(self, kind, branch="main"):
        rows = self._rows(
            "SELECT * FROM results WHERE kind=? AND branch=? ORDER BY created_seq DESC LIMIT 1",
            (kind, branch))
        if not rows:
            return None
        return [r for r in self.results() if r["result_id"] == rows[0]["result_id"]][0]

    # -- export / import --------------------------------------------------
    def export(self):
        return {"format": EXPORT_FORMAT,
                "meta": {"seq": self.current_seq()},
                "materials": self._rows("SELECT * FROM materials ORDER BY created_seq, kind, version"),
                "events": self._rows("SELECT * FROM events ORDER BY seq"),
                "results": self._rows("SELECT * FROM results ORDER BY created_seq, result_id")}

    def import_data(self, data):
        if not isinstance(data, dict) or data.get("format") != EXPORT_FORMAT:
            return {"status": "format_corrupt",
                    "diagnostics": [{"path": "$", "code": "format_corrupt",
                                     "message": "not an itc-project/1 export"}]}
        for key in ("materials", "events", "results", "meta"):
            if key not in data or not isinstance(data[key], (list, dict)):
                return {"status": "format_corrupt",
                        "diagnostics": [{"path": f"$.{key}", "code": "format_corrupt",
                                         "message": f"missing or invalid section {key!r}"}]}
        imported = skipped = 0
        conflicts = []
        with self.atomic():
            for row in data["materials"]:
                if self._merge("materials", "mat_id", row,
                               ("mat_id", "kind", "version", "raw_json", "norm_json",
                                "diag_json", "created_seq", "branch"), conflicts):
                    imported += 1
                else:
                    skipped += 1
            for row in data["events"]:
                if self._merge("events", "event_id", row,
                               ("event_id", "seq", "branch", "kind", "payload_json",
                                "parents_json"), conflicts):
                    imported += 1
                else:
                    skipped += 1
            for row in data["results"]:
                if self._merge("results", "result_id", row,
                               ("result_id", "kind", "input_digest", "param_version",
                                "parents_json", "status", "payload_json", "created_seq",
                                "branch"), conflicts):
                    imported += 1
                else:
                    skipped += 1
            seq = max(self.current_seq(), int(data["meta"].get("seq", 0)))
            self.conn.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('seq',?)", (str(seq),))
        return {"status": "ok", "imported": imported, "skipped": skipped,
                "conflicts": conflicts}

    def _merge(self, table, key_col, row, cols, conflicts):
        existing = self.conn.execute(
            f"SELECT * FROM {table} WHERE {key_col}=?", (row[key_col],)).fetchone()
        if existing is None:
            self.conn.execute(
                f"INSERT INTO {table}({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                tuple(row.get(c) for c in cols))
            return True
        if any((existing[c] != row.get(c)) for c in cols):
            conflicts.append({"id": row[key_col], "table": table,
                              "code": "identity_conflict",
                              "message": f"{row[key_col]} exists with different content"})
        return False
