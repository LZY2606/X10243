"""Project export / import.  Round-trips preserve identity (uids), order
(event seq) and provenance (parent links, supersedes chains).  A malformed
bundle is a *format_corrupt* result; a well-formed bundle that conflicts
with existing identity is *semantic_unsolvable*; success is *ok*.
"""
from __future__ import annotations

import json

from .store import Store, new_uid

FORMAT = "itc-review-export"
FORMAT_VERSION = 1

TABLES = ("materials", "computations", "decisions", "diagnostics", "events")


def export_project(store: Store):
    def rows(table):
        return [dict(r) for r in store.conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid")]
    bundle = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "exported_at_seq": store.max_seq(),
        "tables": {t: rows(t) for t in TABLES},
    }
    store.record_event("export_created", {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "counts": {t: len(bundle["tables"][t]) for t in TABLES}})
    store.conn.commit()
    return bundle


def _validate(bundle):
    """Return positioned diagnostics for structural problems."""
    diags = []
    if not isinstance(bundle, dict):
        return [{"severity": "error", "position": "$",
                 "code": "bundle_not_object",
                 "message": "export bundle must be a JSON object"}]
    if bundle.get("format") != FORMAT:
        diags.append({"severity": "error", "position": "$.format",
                      "code": "format_marker_missing",
                      "message": f"expected format {FORMAT!r}"})
    if bundle.get("format_version") != FORMAT_VERSION:
        diags.append({"severity": "error", "position": "$.format_version",
                      "code": "format_version_unsupported",
                      "message": "unsupported export format version"})
    tables = bundle.get("tables")
    if not isinstance(tables, dict):
        diags.append({"severity": "error", "position": "$.tables",
                      "code": "tables_missing",
                      "message": "bundle has no tables object"})
        return diags
    for t in TABLES:
        if not isinstance(tables.get(t), list):
            diags.append({"severity": "error", "position": f"$.tables.{t}",
                          "code": "table_missing",
                          "message": f"table {t} missing or not a list"})
    return diags


def import_project(store: Store, raw):
    """raw: str/bytes of the export bundle (or already-parsed dict)."""
    if isinstance(raw, (str, bytes)):
        try:
            bundle = json.loads(raw)
        except json.JSONDecodeError as exc:
            return {"status": "format_corrupt", "diagnostics": [{
                "severity": "error", "position": f"line {exc.lineno} "
                f"column {exc.colno}", "code": "json_malformed",
                "message": str(exc)}]}
    else:
        bundle = raw
    diags = _validate(bundle)
    if diags:
        return {"status": "format_corrupt", "diagnostics": diags}
    tables = bundle["tables"]
    # identity conflict check: same uid, different content
    for t in TABLES:
        for row in tables[t]:
            uid = row.get("uid")
            if uid is None:
                continue
            existing = store.conn.execute(
                f"SELECT * FROM {t} WHERE uid=?", (uid,)).fetchone()
            if existing and dict(existing) != {k: row[k] for k in
                                               existing.keys()}:
                return {"status": "semantic_unsolvable", "diagnostics": [{
                    "severity": "error", "position": f"$.tables.{t}",
                    "code": "identity_conflict",
                    "message": f"uid {uid} exists with different content"}]}
    with store.transaction("import project") as conn:
        for t in TABLES:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
            for row in tables[t]:
                present = {k: v for k, v in row.items() if k in cols}
                if "uid" in cols and present.get("uid") is not None:
                    dup = conn.execute(
                        f"SELECT 1 FROM {t} WHERE uid=?",
                        (present["uid"],)).fetchone()
                    if dup:
                        continue
                if t == "events":
                    dup = conn.execute(
                        "SELECT 1 FROM events WHERE uid=?",
                        (present["uid"],)).fetchone()
                    if dup:
                        continue
                    conn.execute(
                        "INSERT INTO events(seq,uid,branch,type,payload) "
                        "VALUES(?,?,?,?,?)",
                        (present["seq"], present["uid"],
                         present.get("branch", "main"), present["type"],
                         present["payload"]))
                    continue
                ph = ",".join("?" for _ in present)
                conn.execute(
                    f"INSERT INTO {t}({','.join(present)}) VALUES({ph})",
                    list(present.values()))
        store._event(conn, "import_completed", {
            "format": FORMAT,
            "counts": {t: len(tables[t]) for t in TABLES},
            "rule": "identity_order_provenance_preserved",
        }, "main")
    return {"status": "ok", "diagnostics": []}
