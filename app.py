"""Local ITC review application.

Run:  .venv/bin/uvicorn app:app --host 127.0.0.1 --port 5354
The root opens directly on the bundled, fully-seeded review page. Business
rules live in the itc.* services/storage layers and are tested directly, not
in the browser.
"""

from __future__ import annotations

import json
import os
import pathlib

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from itc import case as case_pkg
from itc.errors import FormatCorrupt, ITCError, ResourceExhausted, SemanticImpossible
from itc.service import Workflow
from itc.storage import Store
from itc import svg as svg_pkg

BASE_DIR = pathlib.Path(__file__).parent
DB_PATH = os.environ.get("ITC_DB", str(BASE_DIR / "data" / "itc.sqlite3"))
CASE_DIR = BASE_DIR / "cases" / "itc-titration-heat-review"
DEMO_ID = "demo-itc-case"

app = FastAPI(title="Local ITC Review", version="1.0")
store: Store | None = None
wf: Workflow | None = None


def _ensure_store() -> tuple[Store, Workflow]:
    global store, wf
    if store is None:
        path = DB_PATH if os.environ.get("ITC_PERSIST", "1") != "0" else ":memory:"
        store = Store(path)
        wf = Workflow(store)
        if store.get_project(DEMO_ID) is None:
            case_pkg.seed_case(wf, CASE_DIR, project_id=DEMO_ID)
    return store, wf


@app.on_event("startup")
def _startup() -> None:
    _ensure_store()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (BASE_DIR / "static" / "index.html").read_text()


@app.exception_handler(ITCError)
async def _itc_error_handler(request: Request, exc: ITCError):
    status = 422
    if isinstance(exc, FormatCorrupt):
        status = 400
    elif isinstance(exc, ResourceExhausted):
        status = 409
    return JSONResponse(status_code=status, content=exc.to_dict())


@app.get("/api/health")
def health():
    s, w = _ensure_store()
    return {"ok": True, "projects": len(s.projects())}


@app.get("/api/projects")
def projects():
    s, w = _ensure_store()
    return {"projects": s.projects(),
            "branches": {p["project_id"]: s.list_branches(p["project_id"])
                         for p in s.projects()}}


@app.get("/api/view")
def view(project: str = DEMO_ID, branch: str = "main", upto: str | None = None):
    s, w = _ensure_store()
    try:
        return w.view(project, branch, upto=upto)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/event")
def event(project: str, event_id: str):
    s, w = _ensure_store()
    ev = s.get_event(project, event_id)
    if ev is None:
        raise HTTPException(status_code=404, detail="event not found")
    return ev


def _series_at(view: dict, kind: str):
    from itc.materials import MaterialVersion

    versions = view["materials"].get(kind)
    if not versions:
        return None, None
    s, w = _ensure_store()
    ev = s.get_event(view["project_id"], versions[-1]["id"])
    mv = MaterialVersion.from_dict(ev["payload"]["material"])
    t, v = mv.series.arrays()
    return t, v


@app.get("/api/panels/power", response_class=Response)
def panel_power(project: str = DEMO_ID, branch: str = "main",
                upto: str | None = None):
    s, w = _ensure_store()
    v = w.view(project, branch, upto=upto)
    integ = v["current_integration"]
    if integ is None:
        raise HTTPException(404, "run integration first")
    tp, vp = _series_at(v, "power")
    tb = vb = None
    if v["materials"].get("baseline"):
        tb, vb = _series_at(v, "baseline")
    shots = [dict(sh, integration_event_id=integ["id"]) for sh in integ["shots"]]
    svg = svg_pkg.power_svg(power_t=tp, power_v=vp, baseline_t=tb, baseline_v=vb,
                            shots=shots)
    return Response(svg, media_type="image/svg+xml")


@app.get("/api/panels/integrals", response_class=Response)
def panel_integrals(project: str = DEMO_ID, branch: str = "main",
                    upto: str | None = None):
    s, w = _ensure_store()
    v = w.view(project, branch, upto=upto)
    integ = v["current_integration"]
    if integ is None:
        raise HTTPException(404, "run integration first")
    svg = svg_pkg.integral_svg(integ["shots"], integration_event_id=integ["id"])
    return Response(svg, media_type="image/svg+xml")


@app.get("/api/panels/residuals", response_class=Response)
def panel_residuals(project: str = DEMO_ID, branch: str = "main",
                    candidate: str = "one_site", upto: str | None = None):
    s, w = _ensure_store()
    v = w.view(project, branch, upto=upto)
    fit = v["current_fit"]
    if fit is None:
        raise HTTPException(404, "run fit first")
    cand = next((c for c in fit["candidates"] if c["model"] == candidate),
                fit["candidates"][0])
    indices = fit["input_summary"]["used_shots"]
    svg = svg_pkg.residual_svg(cand, indices, candidate_id=cand["id"])
    return Response(svg, media_type="image/svg+xml")
