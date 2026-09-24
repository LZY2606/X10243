"""FastAPI 入口：本地 ITC 注射积分 / 稀释热 / 绑定模型审议应用。

启动即进入带样例的审阅页；无账号、无云端、无额外配置。
业务规则全部在服务端（itc 包），浏览器只做呈现与联动。
"""
from __future__ import annotations

import os
import pathlib

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from itc import casepack, export_import, pipeline, units
from itc.store import Store

DB_PATH = os.environ.get(
    "ITC_DB", str(pathlib.Path(__file__).parent / "data" / "itc_review.db"))
STATIC = pathlib.Path(__file__).parent / "static" / "review.html"

store = Store(DB_PATH)
if store.is_empty():
    casepack.seed(store)

app = FastAPI(title="ITC 注射积分 / 绑定模型审议")


def _material_view(kind, mat):
    """派生层规范化视图：原始表示不动，此处产出显示用数组与诊断。"""
    raw = mat["raw"]
    view = {"id": mat["id"], "version": mat["version"], "kind": kind,
            "raw": raw}
    if kind in ("power_trace", "baseline", "blank_trace"):
        norm = units.normalize_trace(raw, f"materials.{kind}")
        view["normalized"] = norm
    elif kind == "injection_schedule":
        view["normalized"] = units.normalize_schedule(raw, f"materials.{kind}")
    elif kind == "concentration":
        out, diags = {}, []
        for key in ("syringe", "cell"):
            v, d = units.normalize_scalar(raw.get(key), "concentration",
                                          f"materials.concentration.{key}")
            out[key] = v
            if d:
                diags.append(d.to_json())
        view["normalized"] = {"value": out, "unit": "µM", "diagnostics": diags}
    elif kind in ("temperature", "cell_volume"):
        k = "temperature" if kind == "temperature" else "volume"
        v, d = units.normalize_scalar(raw.get("value"), k,
                                      f"materials.{kind}")
        view["normalized"] = {"value": v, "unit": units.CANONICAL_UNITS[k],
                              "diagnostics": [d.to_json()] if d else []}
    return view


def _full_state(seq=None):
    state = store.state_at(seq)
    state["material_views"] = {k: _material_view(k, m)
                               for k, m in state["materials"].items()}
    return state


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(STATIC.read_text(encoding="utf-8"))


@app.get("/api/state")
def api_state(seq: int | None = None):
    return _full_state(seq)


@app.post("/api/decision")
async def api_decision(req: Request):
    body = await req.json()
    action = body.get("action")
    if action not in ("accept", "reject", "adjust"):
        return JSONResponse({"status": "format_error",
                             "message": "action 需为 accept/reject/adjust"},
                            status_code=400)
    dec = store.add_decision(body["subject"], action,
                             body.get("rationale", ""),
                             branch=body.get("branch", "main"))
    return {"status": "ok", "decision": dec, "state": _full_state()}


@app.post("/api/recompute")
def api_recompute():
    pipeline.compute_integrations(store, "power_trace",
                                  {"discard_first": True})
    pipeline.compute_blank_subtraction(store, {"matching": "index"})
    fit_id, _, _ = pipeline.compute_model_fit(store)
    return {"status": "ok", "model_fit": fit_id, "state": _full_state()}


@app.get("/api/export")
def api_export():
    return export_import.export_project(store)


@app.post("/api/import")
async def api_import(req: Request):
    body = await req.json()
    result = export_import.import_project(store, body)
    code = 200 if result["status"] == "ok" else 400
    return JSONResponse(result, status_code=code)
