"""FastAPI 入口：ITC 注射积分与绑定模型审阅。

业务规则全部在服务端领域层（itc.*），浏览器只做展示与交互。
"""

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from itc import casepack, project_io, service
from itc.store import Store

STATIC = os.path.join(os.path.dirname(__file__), "static")


class MaterialIn(BaseModel):
    stream: str
    payload: dict
    note: str | None = None
    branch: str = "main"


class IntegrateIn(BaseModel):
    params: dict
    branch: str = "main"


class FitIn(BaseModel):
    budget: dict | None = None
    integration_id: str | None = None
    branch: str = "main"


class DecisionIn(BaseModel):
    action: str
    target_id: str
    rationale: str = ""
    branch: str = "main"


def create_app(db_path="itc.db"):
    app = FastAPI(title="ITC 注射积分与绑定模型审阅")
    store = Store(db_path)
    casepack.load_case_into(store)
    app.state.store = store
    app.state.db_path = db_path

    def db():
        return app.state.store

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/api/state")
    def state(asof: int | None = None, branch: str = "main"):
        return db().state(asof=asof, branch=branch)

    @app.post("/api/material")
    def import_material(body: MaterialIn):
        return service.import_material(db(), body.stream, body.payload,
                                       branch=body.branch, note=body.note)

    @app.post("/api/integrate")
    def integrate(body: IntegrateIn):
        return service.run_integration(db(), body.params, branch=body.branch)

    @app.post("/api/fit")
    def fit(body: FitIn):
        return service.run_fit(db(), budget=body.budget,
                               integration_id=body.integration_id,
                               branch=body.branch)

    @app.post("/api/decision")
    def decide(body: DecisionIn):
        return service.decide(db(), body.action, body.target_id,
                              body.rationale, branch=body.branch)

    @app.get("/api/export")
    def export():
        return JSONResponse(project_io.export_project(db()))

    @app.post("/api/import")
    def import_(body: dict):
        return project_io.import_project(db(), body)

    @app.post("/api/reset")
    def reset():
        db().close()
        fresh = Store(app.state.db_path)
        app.state.store = fresh
        ids = casepack.build_case(fresh)
        return {"status": "ok", "ids": ids}

    return app


app = create_app(os.environ.get("ITC_DB", "itc.db"))
