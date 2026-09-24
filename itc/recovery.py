"""Half-write recovery.

Computations are written with status 'writing' and finalized in the same
transaction; a crash between the two leaves a detectable 'writing' row. On
startup, recovery marks them 'aborted' and records a machine-readable
recovery event. Aborted rows are never reused as valid results.
"""
import json

from .store import STATUS_ABORTED, STATUS_WRITING


def recover_half_writes(store, branch="main"):
    cur = store.conn.execute(
        "SELECT result_id, kind, seq FROM computations WHERE status=?",
        (STATUS_WRITING,))
    stale = [dict(r) for r in cur.fetchall()]
    if not stale:
        return []
    with store.conn:
        store.conn.execute(
            "UPDATE computations SET status=? WHERE status=?",
            (STATUS_ABORTED, STATUS_WRITING))
    ev = store.add_event("recovery", {
        "rule": "half_write_recovery",
        "aborted": stale,
        "message": "incomplete computations from an interrupted write were "
                   "marked aborted; no partial result is treated as valid",
    }, parents=[s["result_id"] for s in stale], branch=branch)
    return [{"aborted": stale, "event_id": ev["event_id"]}]
