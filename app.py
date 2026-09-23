"""ITC review app: raw power curves, injection schedules and blank controls in;
per-injection integration and binding-model candidates reviewed. All business
rules live server-side; the browser only renders state."""

import os
import threading

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from itc import casepack, pipeline
from itc.state import build_curve, build_state
from itc.store import CorruptFormat, Store

DB_PATH = os.environ.get("ITC_DB", "itc_review.db")
CASEPACK_DIR = os.environ.get("ITC_CASEPACK", "casepack")

app = FastAPI(title="ITC injection integration & binding model review")
_lock = threading.Lock()
_holder = {"store": None}


def get_store():
    if _holder["store"] is None:
        _holder["store"] = Store(DB_PATH)
        if _holder["store"].is_empty():
            casepack.load_casepack(_holder["store"], CASEPACK_DIR)
    return _holder["store"]


@app.on_event("startup")
def startup():
    get_store()


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/state")
def api_state(as_of: int | None = Query(None), branch: str | None = Query(None)):
    with _lock:
        return build_state(get_store(), as_of, branch)


@app.get("/api/curve")
def api_curve(as_of: int | None = Query(None)):
    with _lock:
        curve = build_curve(get_store(), as_of)
    if curve is None:
        raise HTTPException(404, "no power curve material")
    return curve


@app.post("/api/integrate")
def api_integrate(params: dict = Body(default={})):
    with _lock:
        result_id, event = pipeline.run_integration(get_store(), params)
    return {"result_id": result_id, "event": event}


@app.post("/api/blank_correct")
def api_blank_correct(params: dict = Body(default={})):
    with _lock:
        result_id, event = pipeline.run_blank_correction(get_store(), params)
    return {"result_id": result_id, "event": event}


@app.post("/api/fit")
def api_fit(params: dict = Body(default={})):
    with _lock:
        result_id, event = pipeline.run_fit(get_store(), params)
    return {"result_id": result_id, "event": event}


@app.post("/api/materials/{kind}/revise")
def api_revise(kind: str, body: dict = Body(...)):
    payload = body.get("payload")
    if payload is None:
        raise HTTPException(422, "payload is required")
    with _lock:
        store = get_store()
        with store.txn() as (cur, txn_id):
            event = store.add_event(cur, txn_id, f"revise:{kind}",
                                    {"kind": kind, "reason": body.get("reason", "")},
                                    [e["id"] for e in store.events()][-1:])
            mat = store.put_material(cur, txn_id, kind, payload, event["id"])
    return {"material": mat, "event": event}


@app.post("/api/decision")
def api_decision(body: dict = Body(...)):
    for field in ("branch", "target", "action"):
        if not body.get(field):
            raise HTTPException(422, f"{field} is required")
    if body["action"] not in ("accept", "reject", "adjust"):
        raise HTTPException(422, "action must be accept|reject|adjust")
    with _lock:
        store = get_store()
        with store.txn() as (cur, txn_id):
            event = store.add_event(
                cur, txn_id, f"decision:{body['action']}",
                {"branch": body["branch"], "target": body["target"],
                 "rationale": body.get("rationale", "")},
                [e["id"] for e in store.events()][-1:])
            decision_id = store.put_decision(cur, txn_id, event["id"], body["branch"],
                                             body["target"], body["action"],
                                             body.get("rationale", ""))
    return {"decision_id": decision_id, "event": event}


@app.get("/api/export")
def api_export():
    with _lock:
        return get_store().export()


@app.post("/api/import")
def api_import(data: dict = Body(...)):
    with _lock:
        try:
            stats = get_store().import_(data)
        except CorruptFormat as exc:
            return JSONResponse(status_code=422,
                                content={"status": "format_corrupt",
                                         "detail": str(exc)})
    return {"status": "ok", **stats}


@app.post("/api/reset")
def api_reset():
    with _lock:
        if _holder["store"] is not None:
            _holder["store"].close()
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        _holder["store"] = Store(DB_PATH)
        casepack.load_casepack(_holder["store"], CASEPACK_DIR)
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory="static"), name="static")
