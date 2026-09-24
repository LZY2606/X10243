"""Local ITC review app: raw power curves + injection schedules + blank
controls in, per-injection integrals and binding-model candidates reviewed."""
from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from itc import casepack, counterexamples
from itc.domain import MATERIAL_KINDS
from itc.service import Session
from itc.store import Store

DB_PATH = os.environ.get("ITC_DB", "itc_review.db")
STATIC = os.path.join(os.path.dirname(__file__), "static")

store = Store(DB_PATH)
session = Session(store)
if not store.events():
    casepack.replay(session)

app = FastAPI(title="ITC Review")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/state")
def api_state(seq: int | None = None):
    return session.view(seq=seq)


@app.get("/api/events")
def api_events():
    return session.store.events()


@app.post("/api/decisions")
async def api_decide(request: Request):
    body = await request.json()
    ev = session.decide(body.get("action", "accept"), body["target"],
                        body.get("rationale", ""), body.get("branch", "main"),
                        body.get("reviewer", "reviewer"))
    return ev


@app.post("/api/materials/{kind}")
async def api_material(kind: str, request: Request):
    if kind not in MATERIAL_KINDS:
        return JSONResponse({"kind": "semantic_unsolvable",
                             "message": f"unknown material kind '{kind}'"}, status_code=422)
    payload = await request.json()
    mid = session.import_material(kind, payload.get("payload", payload),
                                  note=payload.get("note") if isinstance(payload, dict) else None)
    return {"material": mid}


@app.post("/api/compute/{kind}")
async def api_compute(kind: str, request: Request):
    body = await request.json() if request.headers.get("content-length") else {}
    budget = int(body.get("budget", 100000))
    if kind == "integration":
        rid = session.run_integration(config=body.get("config"), budget=budget)
    elif kind == "dilution":
        rid = session.run_dilution(rule=body.get("rule", "by_index"), budget=budget)
    elif kind == "fit":
        rid = session.run_fit(budget=budget)
    else:
        return JSONResponse({"kind": "semantic_unsolvable",
                             "message": f"unknown computation '{kind}'"}, status_code=422)
    return session.store.result(rid)


@app.get("/api/export")
def api_export():
    return session.store.export()


@app.post("/api/import")
async def api_import(request: Request):
    global store, session
    archive = await request.json()
    new_store, status = Store.import_archive(DB_PATH + ".imported", archive)
    if new_store is None:
        return JSONResponse(status, status_code=422)
    store.close()
    os.replace(DB_PATH + ".imported", DB_PATH)
    store = Store(DB_PATH)
    session = Session(store)
    return {"kind": "ok", "events": len(store.events()),
            "recovered_writes": store.recovered}


@app.get("/api/counterexamples")
def api_counterexamples():
    with tempfile.TemporaryDirectory() as tmp:
        return counterexamples.run_archive(tmp)
