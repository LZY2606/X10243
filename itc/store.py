"""SQLite persistence layer.

Immutable materials (power sampling, injection volumes/times, concentrations,
temperature, cell volume, baseline, blank curves) are versioned separately.
Events form an append-only log; every computation records its input digest,
parameter version and parent dependencies. Results are content-addressed so
equal inputs yield stable identities and stable ordering.
"""

import hashlib
import json
import sqlite3
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS materials (
  id TEXT PRIMARY KEY,
  category TEXT NOT NULL,
  version INTEGER NOT NULL,
  branch TEXT NOT NULL,
  payload TEXT NOT NULL,
  digest TEXT NOT NULL,
  created_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT NOT NULL,
  branch TEXT NOT NULL,
  type TEXT NOT NULL,
  payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
  id TEXT PRIMARY KEY,
  branch TEXT NOT NULL,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  inputs_digest TEXT NOT NULL,
  params TEXT NOT NULL,
  payload TEXT NOT NULL,
  diagnostics TEXT NOT NULL,
  event_seq INTEGER NOT NULL
);
"""

EXPORT_FORMAT = "itc-review/1"


def digest(obj):
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _dump(obj):
    return json.dumps(obj, ensure_ascii=False)


class Store:
    def __init__(self, path=":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.close()

    def is_empty(self):
        row = self.conn.execute(
            "SELECT (SELECT COUNT(*) FROM materials) + (SELECT COUNT(*) FROM events) AS n"
        ).fetchone()
        return row["n"] == 0

    def _max_event_seq(self):
        return self.conn.execute("SELECT COALESCE(MAX(seq),0) AS s FROM events").fetchone()["s"]

    # ---------------- materials ----------------
    def add_material(self, category, payload, branch="main", id=None,
                     version=None, created_seq=None):
        if version is None:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(version),0) AS v FROM materials WHERE category=? AND branch=?",
                (category, branch)).fetchone()
            version = row["v"] + 1
        mid = id or f"{category}-{uuid.uuid4().hex[:12]}"
        if created_seq is None:
            created_seq = self._max_event_seq()
        dgst = digest({"category": category, "payload": payload})
        self.conn.execute(
            "INSERT OR IGNORE INTO materials(id,category,version,branch,payload,digest,created_seq)"
            " VALUES(?,?,?,?,?,?,?)",
            (mid, category, version, branch, _dump(payload), dgst, created_seq))
        self.conn.commit()
        return {"id": mid, "category": category, "version": version,
                "branch": branch, "digest": dgst}

    def materials(self, branches=None, as_of=None):
        sql = "SELECT * FROM materials WHERE 1=1"
        args = []
        if branches:
            sql += " AND branch IN (%s)" % ",".join("?" * len(branches))
            args += list(branches)
        if as_of is not None:
            sql += " AND created_seq <= ?"
            args.append(as_of)
        sql += " ORDER BY category, version, id"
        rows = self.conn.execute(sql, args).fetchall()
        return [dict(r, payload=json.loads(r["payload"])) for r in rows]

    def latest_materials(self, branches=("main",), as_of=None):
        latest = {}
        for m in self.materials(branches, as_of):
            key = m["category"]
            if key not in latest or m["version"] > latest[key]["version"]:
                latest[key] = m
        return latest

    # ---------------- events ----------------
    def add_event(self, type, payload, branch="main", id=None, seq=None):
        eid = id or uuid.uuid4().hex
        if seq is None:
            cur = self.conn.execute(
                "INSERT INTO events(id,branch,type,payload) VALUES(?,?,?,?)",
                (eid, branch, type, _dump(payload)))
            seq = cur.lastrowid
        else:
            self.conn.execute(
                "INSERT OR IGNORE INTO events(seq,id,branch,type,payload) VALUES(?,?,?,?,?)",
                (seq, eid, branch, type, _dump(payload)))
        self.conn.commit()
        return {"seq": seq, "id": eid, "type": type, "branch": branch}

    def events(self, branches=None, as_of=None):
        sql = "SELECT * FROM events WHERE 1=1"
        args = []
        if branches:
            sql += " AND branch IN (%s)" % ",".join("?" * len(branches))
            args += list(branches)
        if as_of is not None:
            sql += " AND seq <= ?"
            args.append(as_of)
        sql += " ORDER BY seq"
        rows = self.conn.execute(sql, args).fetchall()
        return [dict(r, payload=json.loads(r["payload"])) for r in rows]

    # ---------------- results ----------------
    def add_result(self, id, kind, status, inputs_digest, params, payload,
                   diagnostics, event_seq, branch="main"):
        self.conn.execute(
            "INSERT OR IGNORE INTO results(id,branch,kind,status,inputs_digest,params,"
            "payload,diagnostics,event_seq) VALUES(?,?,?,?,?,?,?,?,?)",
            (id, branch, kind, status, inputs_digest, _dump(params), _dump(payload),
             _dump(diagnostics), event_seq))
        self.conn.commit()
        return id

    def results(self, branches=None, as_of=None):
        sql = "SELECT * FROM results WHERE 1=1"
        args = []
        if branches:
            sql += " AND branch IN (%s)" % ",".join("?" * len(branches))
            args += list(branches)
        if as_of is not None:
            sql += " AND event_seq <= ?"
            args.append(as_of)
        sql += " ORDER BY event_seq, id"
        rows = self.conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("params", "payload", "diagnostics"):
                d[k] = json.loads(d[k])
            out.append(d)
        return out

    # ---------------- export / import ----------------
    def export_data(self):
        return {
            "format": EXPORT_FORMAT,
            "materials": self.materials(),
            "events": self.events(),
            "results": self.results(),
        }

    def import_data(self, data):
        """Atomic import. Any failure rolls back and returns format_corrupt."""
        try:
            if not isinstance(data, dict) or data.get("format") != EXPORT_FORMAT:
                raise ValueError("unknown export format")
            for key in ("materials", "events", "results"):
                if not isinstance(data.get(key), list):
                    raise ValueError(f"missing section: {key}")
            cur = self.conn.cursor()
            try:
                cur.execute("BEGIN")
                for m in data["materials"]:
                    for k in ("id", "category", "version", "branch", "payload", "created_seq"):
                        if k not in m:
                            raise ValueError(f"material missing field: {k}")
                    dgst = digest({"category": m["category"], "payload": m["payload"]})
                    cur.execute(
                        "INSERT OR IGNORE INTO materials(id,category,version,branch,payload,"
                        "digest,created_seq) VALUES(?,?,?,?,?,?,?)",
                        (m["id"], m["category"], int(m["version"]), m["branch"],
                         _dump(m["payload"]), m.get("digest", dgst), int(m["created_seq"])))
                for e in data["events"]:
                    for k in ("seq", "id", "branch", "type", "payload"):
                        if k not in e:
                            raise ValueError(f"event missing field: {k}")
                    cur.execute(
                        "INSERT OR IGNORE INTO events(seq,id,branch,type,payload)"
                        " VALUES(?,?,?,?,?)",
                        (int(e["seq"]), e["id"], e["branch"], e["type"], _dump(e["payload"])))
                for r in data["results"]:
                    for k in ("id", "branch", "kind", "status", "inputs_digest", "params",
                              "payload", "diagnostics", "event_seq"):
                        if k not in r:
                            raise ValueError(f"result missing field: {k}")
                    cur.execute(
                        "INSERT OR IGNORE INTO results(id,branch,kind,status,inputs_digest,"
                        "params,payload,diagnostics,event_seq) VALUES(?,?,?,?,?,?,?,?,?)",
                        (r["id"], r["branch"], r["kind"], r["status"], r["inputs_digest"],
                         _dump(r["params"]), _dump(r["payload"]), _dump(r["diagnostics"]),
                         int(r["event_seq"])))
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
            return {"status": "ok",
                    "imported": {"materials": len(data["materials"]),
                                 "events": len(data["events"]),
                                 "results": len(data["results"])}}
        except Exception as exc:
            return {"status": "format_corrupt",
                    "diagnostics": [{"code": "import_failed", "detail": str(exc)}]}
