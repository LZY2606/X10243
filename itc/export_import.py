"""Project export / import.

Export is a single JSON document containing every table plus per-table
digests. Import validates structure and digests *before* touching the
database and applies everything in one transaction: a corrupt file is a
'format_corrupt' result and leaves the existing database untouched; a crash
mid-import rolls back cleanly. Identity (ids, seq, branches) is preserved
verbatim so export -> delete -> import is a perfect round-trip.
"""
import json

from .ids import digest
from .store import STATUS_FORMAT, STATUS_OK

FORMAT = "itc-review-export/1"
TABLES = ("materials", "events", "computations", "decisions")


def export_project(store):
    tables = {}
    digests = {}
    for name in TABLES:
        rows = [dict(r) for r in store.conn.execute(
            f"SELECT * FROM {name} ORDER BY rowid").fetchall()]
        tables[name] = rows
        digests[name] = digest(rows)
    meta = {r["key"]: r["value"] for r in
            store.conn.execute("SELECT * FROM meta").fetchall()}
    return {"format": FORMAT, "tables": tables, "meta": meta, "digests": digests}


def _validate(doc):
    diags = []
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        return [{"code": "format_unknown", "path": "format",
                 "message": "not an itc-review-export/1 document",
                 "severity": "error"}]
    tables = doc.get("tables")
    if not isinstance(tables, dict):
        return [{"code": "tables_missing", "path": "tables",
                 "message": "export has no tables object", "severity": "error"}]
    for name in TABLES:
        rows = tables.get(name)
        if not isinstance(rows, list):
            diags.append({"code": "table_missing", "path": f"tables.{name}",
                          "message": f"table {name} missing or not a list",
                          "severity": "error"})
            continue
        expected = doc.get("digests", {}).get(name)
        if expected is not None and digest(rows) != expected:
            diags.append({"code": "digest_mismatch", "path": f"tables.{name}",
                          "message": f"table {name} failed digest verification",
                          "severity": "error"})
    return diags


def import_project(store, doc):
    """Replace database contents with the exported document (all-or-nothing)."""
    diags = _validate(doc)
    if diags:
        return {"status": STATUS_FORMAT, "diagnostics": diags}
    try:
        with store.conn:
            cur = store.conn.cursor()
            for name in TABLES:
                cur.execute(f"DELETE FROM {name}")
            cur.execute("DELETE FROM meta")
            for key, value in doc.get("meta", {}).items():
                cur.execute("INSERT INTO meta(key, value) VALUES(?,?)", (key, value))
            for name in TABLES:
                rows = doc["tables"][name]
                if not rows:
                    continue
                cols = list(rows[0].keys())
                marks = ",".join("?" for _ in cols)
                for row in rows:
                    if list(row.keys()) != cols:
                        raise ValueError(f"inconsistent columns in {name}")
                    cur.execute(
                        f"INSERT INTO {name}({','.join(cols)}) VALUES({marks})",
                        [row[c] for c in cols])
    except (ValueError, Exception) as exc:  # sqlite errors included
        return {"status": STATUS_FORMAT, "diagnostics": [{
            "code": "import_failed", "path": "tables",
            "message": f"import rolled back: {exc}", "severity": "error"}]}
    return {"status": STATUS_OK, "diagnostics": [],
            "counts": {n: len(doc["tables"][n]) for n in TABLES}}
