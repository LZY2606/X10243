"""ITC 注射积分与绑定模型审阅 — FastAPI 入口。

工作流: 导入 -> 计算 -> 保存 -> 导出 -> 再次打开。
业务规则全部在服务端领域层 (itc/), 浏览器只做展示与联动。
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from itc import casepack, export, svg, views
from itc.store import Store

DB_PATH = os.environ.get("ITC_DB", "itc_review.db")
CASEPACK_PATH = os.environ.get(
    "ITC_CASEPACK", os.path.join(os.path.dirname(__file__),
                                 "casepack", "itc_titration_case.json"))


def open_store(path=DB_PATH):
    store = Store(path)
    if store.head_seq() == 0:
        if os.path.exists(CASEPACK_PATH):
            pack = casepack.load_casepack(CASEPACK_PATH)
        else:
            pack = casepack.build_casepack()
        casepack.replay(store, pack)
    return store


store = open_store()
app = FastAPI(title="ITC 注射积分与绑定模型审阅")
app.mount("/static", StaticFiles(directory=os.path.join(
    os.path.dirname(__file__), "static")), name="static")


def _case_meta():
    if os.path.exists(CASEPACK_PATH):
        pack = casepack.load_casepack(CASEPACK_PATH)
    else:
        pack = casepack.build_casepack()
    return {"title": pack["title"], "narrative": pack["narrative"],
            "default_milestone": pack.get("default_milestone")}


def _milestone_seq(name):
    for e in store.events(kind="milestone"):
        if e["payload"].get("name") == name:
            return e["seq"]
    return None


def _select_integrals(as_of):
    cands = [r for r in store.results(kind="injection_integrals", as_of=as_of)
             if r["status"] in ("ok", "partial")]
    return cands[-1] if cands else None


def _fits_by_model(as_of):
    out = {}
    for r in store.results(kind="model_fit", as_of=as_of):
        out[r["payload"].get("model")] = r  # 按 seq 升序, 后者覆盖
    return out


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__),
                           "static", "index.html"), encoding="utf-8") as fh:
        return fh.read()


@app.get("/api/state")
def api_state(as_of: int | None = Query(default=None),
              integrals: str | None = Query(default=None),
              fit: str | None = Query(default=None),
              candidate: int = Query(default=0)):
    head = store.head_seq()
    if as_of is None:
        as_of = _milestone_seq("main_complete") or head
    as_of = min(as_of, head)
    state = views.review_state(store, as_of=as_of)
    state["case"] = _case_meta()
    state["milestones"] = [
        {"name": e["payload"]["name"], "label": e["payload"].get("label", ""),
         "seq": e["seq"]} for e in store.events(kind="milestone")]

    integ = store.result(integrals) if integrals else _select_integrals(as_of)
    if integ and integ["seq"] > as_of:
        integ = _select_integrals(as_of)
    fits = _fits_by_model(as_of)
    fit_res = store.result(fit) if fit else fits.get("one_site")
    if fit_res and fit_res["seq"] > as_of:
        fit_res = fits.get("one_site")

    selection = {"integrals": None, "fits": {}, "fit": None}
    if integ:
        inputs = integ["payload"].get("inputs", {})
        power_id = inputs.get("power_curve", {}).get("id")
        power = store.material(power_id) if power_id else None
        selection["integrals"] = {
            "id": integ["id"], "status": integ["status"],
            "stale": views.staleness(store, integ, as_of=as_of)[0],
            "stale_reasons": views.staleness(store, integ, as_of=as_of)[1],
            "units": integ["payload"].get("units", {}),
            "time_basis": integ["payload"].get("time_basis"),
            "injections": integ["payload"]["injections"],
            "diagnostics": integ["payload"].get("diagnostics", []),
            "power_svg": svg.power_curve_svg(power["payload"], integ["payload"])
            if power else None,
        }
    for model, r in fits.items():
        stale, reasons = views.staleness(store, r, as_of=as_of)
        selection["fits"][model] = {"id": r["id"], "stale": stale,
                                    "stale_reasons": reasons,
                                    "n_candidates": len(r["payload"].get("candidates", [])),
                                    "status": r["status"]}
    if fit_res:
        fv = views.fit_view(store, fit_res["id"], as_of=as_of)
        fv["residuals_svg"] = svg.residuals_svg(
            fit_res["payload"], candidate_idx=candidate)
        fv["candidate_index"] = candidate
        selection["fit"] = fv
    state["selection"] = selection
    return state


@app.post("/api/decide")
async def api_decide(request: Request):
    body = await request.json()
    reviewer = (body.get("reviewer") or "").strip()
    action = (body.get("action") or "").strip()
    rationale = (body.get("rationale") or "").strip()
    if not reviewer or not action:
        raise HTTPException(400, "reviewer 与 action 必填")
    decision = {"reviewer": reviewer, "action": action,
                "rationale": rationale}
    parents = []
    if body.get("target_result"):
        target = store.result(body["target_result"])
        if target is None:
            raise HTTPException(404, f"结果不存在: {body['target_result']}")
        decision["target"] = target["id"]
        if body.get("target_candidate"):
            decision["target_candidate"] = body["target_candidate"]
        parents.append(target["id"])
    if body.get("target_injection"):
        decision["target"] = body["target_injection"]
    with store.operation("decide", {"action": action}):
        evt = store.put_event("decision", {"decision": decision},
                              parents=parents, param_version="review/1.0")
    return {"event": evt, "head_seq": store.head_seq()}


@app.get("/api/export")
def api_export():
    return Response(content=export.export_json(store),
                    media_type="application/json",
                    headers={"Content-Disposition":
                             "attachment; filename=itc_review_bundle.json"})


@app.post("/api/import")
async def api_import(request: Request):
    text = (await request.body()).decode("utf-8")
    try:
        summary = export.import_json(store, text)
    except export.FormatCorrupt as exc:
        return JSONResponse(status_code=400,
                            content={"status": "format_corrupt",
                                     "detail": str(exc)})
    return {"status": "ok", **summary, "head_seq": store.head_seq()}


@app.get("/api/op_log")
def api_op_log():
    return {"op_log": store.dump()["op_log"]}
