"""Project export / import preserving ids, branches and decision history.

Bundle (zip):
  manifest.json   - schema, project id, event/blob counts, hash chain summary
  events.jsonl    - events in created_order (same ids/seq/hashes)
  blobs/<id>      - immutable raw material blobs

Import verifies the hash chain before applying; a corrupted bundle is rejected
as FORMAT_CORRUPTED.  Re-import keeps ids stable.
"""

from __future__ import annotations

import io
import json
import zipfile

from .errors import FormatCorrupted
from .util import digest

MANIFEST = "manifest.json"
EVENTS = "events.jsonl"


def export_bundle(store, project_id: str, *, verify=True) -> bytes:
    snap = store.snapshot()
    events = [e for e in snap["events"] if e["project_id"] == project_id]
    blob_ids = {bid for e in events for bid in e["blobs"].values()}
    blobs = [b for b in snap["blobs"] if b["id"] in blob_ids]
    if verify:
        _verify_chain(events)
    payload = {
        "schema": "itc-bundle/1",
        "project_id": project_id,
        "event_count": len(events),
        "blob_count": len(blobs),
        "hash_chain_head": events[-1]["hash"] if events else None,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(MANIFEST, json.dumps(payload, ensure_ascii=False, indent=2))
        with z.open(EVENTS, "w") as f:
            for e in events:
                f.write((json.dumps(_event_json(e), ensure_ascii=False) + "\n")
                        .encode("utf-8"))
        for b in blobs:
            z.writestr(f"blobs/{b['id']}", b["content"])
            z.writestr(f"blobs/{b['id']}.meta",
                       json.dumps({"kind": b["kind"], "media_type": b["media_type"]}))
    return buf.getvalue()


def _event_json(e):
    return {k: e[k] for k in ("project_id", "branch", "seq", "id", "type",
                              "payload", "blobs", "prev_hash", "hash",
                              "created_order")}


def _verify_chain(events):
    ordered = sorted(events, key=lambda e: (e["branch"], e["seq"]))
    heads: dict[str, str] = {}
    for e in ordered:
        if e.get("prev_hash") != heads.get(e["branch"]):
            raise FormatCorrupted(
                "hash chain broken at event",
                path=EVENTS, line=e["seq"],
                diagnostics=[{"code": "HASH_CHAIN_BROKEN", "branch": e["branch"],
                              "seq": e["seq"], "event_id": e["id"]}])
        body = {"id": e["id"], "project": e["project_id"], "branch": e["branch"],
                "seq": e["seq"], "type": e["type"], "payload": e["payload"],
                "blobs": e["blobs"], "prev": e["prev_hash"]}
        if digest(body, length=40) != e["hash"]:
            raise FormatCorrupted(
                "event hash mismatch", path=EVENTS, line=e["seq"],
                diagnostics=[{"code": "EVENT_HASH_MISMATCH", "branch": e["branch"],
                              "seq": e["seq"]}])
        heads[e["branch"]] = e["hash"]


def import_bundle(store, data: bytes, *, project_id=None, verify=True):
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        bad = z.testzip()
        if bad is not None:
            raise FormatCorrupted(f"corrupted zip member {bad}", path=bad)
        manifest = json.loads(z.read(MANIFEST).decode("utf-8"))
        lines = z.read(EVENTS).decode("utf-8").splitlines()
        events = [json.loads(line) for line in lines if line.strip()]
    except FormatCorrupted:
        raise
    except (KeyError, zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FormatCorrupted(f"bundle unreadable: {exc}") from exc

    if verify:
        _verify_chain(events)
    pid = project_id or manifest["project_id"]
    existing = {e["id"] for br in store.branches(pid) for e in store.list_events(pid, br)}
    applied = 0
    for e in sorted(events, key=lambda x: x["created_order"]):
        meta_name = None
        for bid in e["blobs"].values():
            if store.get_blob(bid) is None:
                content = z.read(f"blobs/{bid}").decode("utf-8")
                meta = json.loads(z.read(f"blobs/{bid}.meta").decode("utf-8"))
                store.put_blob(content, kind=meta["kind"], media_type=meta["media_type"])
        if e["id"] in existing:
            continue
        if store.list_events(pid, e["branch"]) and \
                max(x["seq"] for x in store.list_events(pid, e["branch"])) >= e["seq"]:
            raise FormatCorrupted(
                "event seq conflicts with existing project",
                diagnostics=[{"code": "SEQ_CONFLICT", "branch": e["branch"],
                              "seq": e["seq"]}])
        store.append(pid, e["branch"], e["type"], e["payload"], blobs=e["blobs"])
        applied += 1
    return {"project_id": pid, "events_imported": applied,
            "branches": store.branches(pid)}
