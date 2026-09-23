"""FastAPI entrypoint: local ITC injection-integration / binding-model
review app.  No accounts, no cloud, no extra configuration.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, Body
from fastapi.responses import FileResponse, JSONResponse

from itc.store import Store
from itc.service import Service
from itc import casepack, export_import

DB_PATH = os.environ.get("ITC_DB", os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "itc_review.db"))

store = Store(DB_PATH)
service = Service(store)
if not store.materials():
    casepack.seed(service)
    casepack.replay_narrative(service)

app = FastAPI(title="ITC titration review")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/state")
def state(at_seq: int | None = None, branch: str | None = None):
    return service.state(at_seq=at_seq, branch=branch)


@app.get("/api/events")
def events():
    return {"events": store.events()}


@app.get("/api/events/{seq}/view")
def view_at_event(seq: int):
    """Restore dependency versions and visible conclusions as of an
    event; historical views are never overwritten by newer results."""
    return service.view_at_event(seq)


@app.post("/api/materials/{kind}")
def import_material(kind: str, payload: dict = Body(...)):
    try:
        return service.import_material(kind, payload)
    except ValueError as exc:
        return JSONResponse({"status": "format_corrupt", "diagnostics": [{
            "severity": "error", "position": kind, "code": "unknown_kind",
            "message": str(exc)}]}, status_code=422)


@app.post("/api/compute/integration")
def compute_integration(params: dict = Body(default={})):
    return service.run_integration(params=params or None)


@app.post("/api/compute/blank")
def compute_blank(params: dict = Body(default={})):
    return service.run_blank_correction(params=params or None)


@app.post("/api/compute/fit")
def compute_fit(body: dict = Body(default={})):
    return service.run_fit(models=body.get("models"),
                           budget=body.get("budget", 20000),
                           discarded=body.get("discarded"))


@app.post("/api/decisions")
def decide(body: dict = Body(...)):
    return service.decide(body["target"], body["action"],
                          body.get("value"), body.get("rationale", ""),
                          branch=body.get("branch", "main"))


@app.get("/api/export")
def export():
    return export_import.export_project(store)


@app.post("/api/import")
def import_(body: dict = Body(...)):
    return export_import.import_project(store, body)


@app.post("/api/reset")
def reset():
    global store, service
    store.close()
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    for suffix in ("-wal", "-shm"):
        if os.path.exists(DB_PATH + suffix):
            os.remove(DB_PATH + suffix)
    store = Store(DB_PATH)
    service = Service(store)
    casepack.seed(service)
    log = casepack.replay_narrative(service)
    return {"status": "ok", "steps": {k: v.get("status") for k, v in
                                      log.items() if isinstance(v, dict)}}
