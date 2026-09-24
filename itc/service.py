"""Workflow orchestration: import -> integrate -> dilution subtraction -> fit
-> decide, with machine-readable rule judgments and replayable views.

Distinct result statuses used across the service:
``ok`` / ``incomplete_input`` / ``budget_exhausted`` / ``format_corrupt`` /
``unsatisfiable``.
"""

import json

from . import binding, materials
from .integrate import IntegParams, integrate
from .store import digest_of

REQUIRED_FOR_INTEGRATION = ["power_trace", "injection_schedule", "baseline", "cell_volume"]


# ---------------------------------------------------------------- materials
def import_material(store, kind, raw, branch="main"):
    norm, diags = materials.normalize(kind, raw)
    mat = store.add_material(kind, raw, norm, diags, branch)
    store.add_event(
        branch,
        "material_imported" if mat["version"] == 1 else "material_revised",
        {"material": mat["mat_id"], "kind": kind, "version": mat["version"],
         "diagnostics": diags},
        parents=[mat["mat_id"]])
    return mat


def revise_material(store, kind, raw, branch="main"):
    """A revision creates a new immutable version.  Concentration revision
    makes model fits stale but never touches stored integrations; this is
    emitted as a machine-readable rule judgment."""
    mat = import_material(store, kind, raw, branch)
    if kind == "concentrations":
        fits = [r["result_id"] for r in store.results() if r["kind"] == "fit"]
        integs = [r["result_id"] for r in store.results() if r["kind"] == "integration"]
        store.add_event(branch, "rule_judgment", {
            "rule": "concentration_revision",
            "statement": "浓度修订使模型过期但不改原积分",
            "basis": {"superseded_by": mat["mat_id"]},
            "stale_results": fits,
            "unchanged_results": integs,
        }, parents=[mat["mat_id"]])
    return mat


# -------------------------------------------------------------- integration
def _current_materials(store):
    current = {}
    for m in store.materials():
        k = m["kind"]
        if k not in current or m["version"] > current[k]["version"]:
            current[k] = m
    return current


def run_integration(store, branch="main", params=None, budget=None):
    params = params or IntegParams()
    mats = _current_materials(store)
    missing = [k for k in REQUIRED_FOR_INTEGRATION
               if k not in mats or mats[k]["norm"] is None]
    parents = [mats[k]["mat_id"] for k in REQUIRED_FOR_INTEGRATION if k in mats]
    if missing:
        diags = [{"path": k, "code": "missing_material",
                  "message": f"required material {k!r} is missing or not convertible"}
                 for k in missing]
        payload = {"status": "incomplete_input", "diagnostics": diags, "injections": []}
        rid = store.add_result("integration", digest_of({"parents": parents,
                                                         "missing": missing,
                                                         "params": params.as_dict()}),
                               params.version, parents, "incomplete_input", payload, branch)
        store.add_event(branch, "integration_run",
                        {"result_id": rid, "status": "incomplete_input",
                         "diagnostics": diags}, parents=parents)
        return {"result_id": rid, "status": "incomplete_input", "payload": payload}

    payload = integrate(mats["power_trace"]["norm"], mats["injection_schedule"]["norm"],
                        mats["baseline"]["norm"], params, budget=budget)
    prev = store.latest_result("integration", branch)
    affected = None
    if prev and prev["status"] == "ok" and payload["status"] == "ok":
        old = {i["index"]: i for i in prev["payload"]["injections"]}
        affected = [inj["index"] for inj in payload["injections"]
                    if inj["index"] not in old
                    or old[inj["index"]]["window"] != inj["window"]
                    or (old[inj["index"]]["heat_uJ"] or 0) != (inj["heat_uJ"] or 0)]
    payload["affected_injections"] = affected
    digest = digest_of({"parents": parents, "params": params.as_dict()})
    rid = store.add_result("integration", digest, params.version, parents,
                           payload["status"], payload, branch)
    store.add_event(branch, "integration_run", {
        "result_id": rid, "status": payload["status"],
        "params": params.as_dict(),
        "n_injections": len(payload.get("injections", [])),
        "gap_truncated": [i["index"] for i in payload.get("injections", [])
                          if i["status"] == "gap_truncated"],
        "affected_injections": affected,
    }, parents=parents)
    if payload["status"] == "ok":
        store.add_event(branch, "rule_judgment", {
            "rule": "integration_window",
            "statement": "注射后峰可重叠，积分窗不跨越数据缺口",
            "basis": {"param_version": params.version, "endpoint": params.endpoint},
            "checks": [{"injection": i["index"], "window": i["window"],
                        "gap_limited": i["gap_limited"]}
                       for i in payload["injections"]],
            "outcome": "windows_do_not_cross_gaps",
        }, parents=[rid])
    return {"result_id": rid, "status": payload["status"], "payload": payload}


# ------------------------------------------------------------------ dilution
def run_dilution(store, rule="by_index", branch="main"):
    mats = _current_materials(store)
    integ = store.latest_result("integration", branch)
    missing = []
    if integ is None or integ["status"] != "ok":
        missing.append("integration_result")
    if "blank_curve" not in mats or mats["blank_curve"]["norm"] is None:
        missing.append("blank_curve")
    parents = ([integ["result_id"]] if integ else []) + \
              ([mats["blank_curve"]["mat_id"]] if "blank_curve" in mats else [])
    if missing:
        diags = [{"path": m, "code": "missing_material",
                  "message": f"{m} required for dilution subtraction"} for m in missing]
        payload = {"status": "incomplete_input", "diagnostics": diags, "injections": []}
        rid = store.add_result("dilution", digest_of({"parents": parents, "rule": rule,
                                                      "missing": missing}),
                               "dil-1.0", parents, "incomplete_input", payload, branch)
        store.add_event(branch, "dilution_run",
                        {"result_id": rid, "status": "incomplete_input"}, parents=parents)
        return {"result_id": rid, "status": "incomplete_input", "payload": payload}

    heats = mats["blank_curve"]["norm"]["heats_uJ"]
    out = []
    for inj in integ["payload"]["injections"]:
        i = inj["index"]
        entry = dict(inj)
        entry["blank_uJ"] = None
        entry["corrected_uJ"] = None
        if inj["heat_uJ"] is None:
            entry["status"] = inj["status"]
        elif rule == "by_index":
            j = inj["order"]
            if j < len(heats) and heats[j] is not None:
                entry["blank_uJ"] = heats[j]
                entry["corrected_uJ"] = inj["heat_uJ"] - heats[j]
                entry["status"] = "ok"
            else:
                entry["status"] = "no_blank"
                entry["diagnostics"] = entry["diagnostics"] + [{
                    "path": f"blank_curve.heats[{j}]", "code": "blank_missing",
                    "message": f"no blank heat corresponds to injection {i} "
                               f"under rule 'by_index'"}]
        else:
            entry["status"] = "rule_unknown"
            entry["diagnostics"] = entry["diagnostics"] + [{
                "path": "dilution.rule", "code": "rule_unknown",
                "message": f"unknown correspondence rule {rule!r}"}]
        out.append(entry)
    payload = {"status": "ok", "rule": rule,
               "correspondence": "blank_curve.heats[i] <-> injection[i] (by_index)",
               "injections": out}
    digest = digest_of({"parents": parents, "rule": rule})
    rid = store.add_result("dilution", digest, "dil-1.0", parents, "ok", payload, branch)
    store.add_event(branch, "dilution_run", {
        "result_id": rid, "status": "ok", "rule": rule,
        "no_blank": [e["index"] for e in out if e["status"] == "no_blank"],
    }, parents=parents)
    store.add_event(branch, "rule_judgment", {
        "rule": "dilution_blank_correspondence",
        "statement": "稀释热扣除必须说明对应规则",
        "basis": {"correspondence": rule,
                  "mapping": "blank_curve.heats[i] <-> injection[i]"},
        "missing_blank": [e["index"] for e in out if e["status"] == "no_blank"],
        "outcome": "ok",
    }, parents=[rid])
    return {"result_id": rid, "status": "ok", "payload": payload}


# ---------------------------------------------------------------------- fit
def discarded_injections(store, branch="main", seq=None):
    out = set()
    for e in store.events():
        if e["branch"] != branch or e["kind"] != "decision":
            continue
        if seq is not None and e["seq"] > seq:
            continue
        p = e["payload"]
        if p.get("action") == "discard_injection" and "injection" in p:
            out.add(p["injection"])
    return out


def run_fit(store, models=binding.MODELS, equal_rel=0.05, equal_abs=1e-4,
            budget=6000, branch="main"):
    mats = _current_materials(store)
    dil = store.latest_result("dilution", branch)
    missing = []
    if dil is None or dil["status"] != "ok":
        missing.append("dilution_result")
    for k in ("concentrations", "temperature", "cell_volume", "injection_schedule"):
        if k not in mats or mats[k]["norm"] is None:
            missing.append(k)
    parents = ([dil["result_id"]] if dil else []) + \
              [mats[k]["mat_id"] for k in
               ("concentrations", "temperature", "cell_volume", "injection_schedule")
               if k in mats]
    if missing:
        diags = [{"path": m, "code": "missing_material",
                  "message": f"{m} required for model fitting"} for m in missing]
        payload = {"status": "incomplete_input", "diagnostics": diags, "candidates": []}
        rid = store.add_result("fit", digest_of({"parents": parents, "missing": missing}),
                               "fit-1.0", parents, "incomplete_input", payload, branch)
        store.add_event(branch, "fit_run",
                        {"result_id": rid, "status": "incomplete_input"}, parents=parents)
        return {"result_id": rid, "status": "incomplete_input", "payload": payload}

    xs = mats["concentrations"]["norm"]["syringe_mM"]
    mt = mats["concentrations"]["norm"]["cell_mM"]
    v0 = mats["cell_volume"]["norm"]["mL"]
    discarded = discarded_injections(store, branch)
    obs, vols, used, excluded = [], [], [], []
    for inj in dil["payload"]["injections"]:
        i = inj["index"]
        if i in discarded:
            excluded.append({"index": i, "reason": "discarded"})
            continue
        if inj.get("corrected_uJ") is None:
            excluded.append({"index": i, "reason": inj["status"]})
            continue
        obs.append(inj["corrected_uJ"] / (inj["volume_uL"] * xs))
        vols.append(inj["volume_uL"])
        used.append(i)
    fit = binding.fit_candidates(obs, vols, mt, xs, v0, models=models,
                                 budget=budget, equal_rel=equal_rel, equal_abs=equal_abs)
    payload = {"status": fit["status"], "candidates": fit["candidates"],
               "used_injections": used, "excluded": excluded,
               "obs_kJ_mol": obs, "volumes_uL": vols,
               "equal_rel": equal_rel, "equal_abs": equal_abs,
               "unit": "kJ/mol injectant"}
    digest = digest_of({"parents": parents, "models": list(models),
                        "equal_rel": equal_rel, "equal_abs": equal_abs})
    rid = store.add_result("fit", digest, "fit-1.0", parents, fit["status"], payload, branch)
    store.add_event(branch, "fit_run", {
        "result_id": rid, "status": fit["status"],
        "candidates": [{"model": c["model"], "sse": c["sse"],
                        "equally_good": c["equally_good"]} for c in fit["candidates"]],
        "used_injections": used, "excluded": excluded,
    }, parents=parents)
    return {"result_id": rid, "status": fit["status"], "payload": payload}


# ----------------------------------------------------------------- decisions
def decide(store, action, target, rationale, reviewer, branch="main", payload=None):
    p = {"action": action, "target": target, "rationale": rationale,
         "reviewer": reviewer}
    if payload:
        p.update(payload)
    return store.add_event(branch, "decision", p,
                           parents=[target] if target else [])


# -------------------------------------------------------------------- views
def view_at(store, seq=None, branch="main"):
    """Reconstruct the visible state at an event sequence number: dependency
    versions and conclusions of that moment, never overwritten by later
    computation."""
    seq = store.current_seq() if seq is None else seq
    current = {}
    for m in store.materials():
        if m["created_seq"] <= seq:
            k = m["kind"]
            if k not in current or m["version"] > current[k]["version"]:
                current[k] = m
    all_results = store.results()
    by_id = {r["result_id"]: r for r in all_results}
    results = [r for r in all_results
               if r["created_seq"] <= seq and r["branch"] == branch]
    stale_memo = {}

    def is_stale(r):
        if r["result_id"] in stale_memo:
            return stale_memo[r["result_id"]]
        stale_memo[r["result_id"]] = False  # guard against cycles
        stale = False
        for p in r["parents"]:
            if p.startswith("mat:"):
                kind = p.split(":")[1]
                cur = current.get(kind)
                if cur is None or cur["mat_id"] != p:
                    stale = True
            elif p.startswith("res:"):
                pr = by_id.get(p)
                if pr is not None and is_stale(pr):
                    stale = True
        stale_memo[r["result_id"]] = stale
        return stale

    decisions = [e for e in store.events()
                 if e["kind"] == "decision" and e["branch"] == branch and e["seq"] <= seq]
    accepted = {}
    for d in sorted(decisions, key=lambda e: e["seq"]):
        p = d["payload"]
        key = (p.get("target"), p.get("model"))
        if p.get("action") == "accept_model":
            accepted[key] = d
        elif p.get("action") == "reject_model":
            accepted.pop(key, None)
    accepted_models = sorted({k[1] for k in accepted if k[1]})
    events = [e for e in store.events()
              if e["seq"] <= seq and (e["branch"] == branch
                                      or e["kind"].startswith("material_"))]
    return {
        "seq": seq, "branch": branch,
        "materials": current,
        "results": [dict(r, stale=is_stale(r)) for r in results],
        "decisions": decisions,
        "accepted_models": accepted_models,
        "conflict": len(accepted_models) > 1,
        "discarded_injections": sorted(discarded_injections(store, branch, seq)),
        "events": events,
    }


# ------------------------------------------------------------ export/import
def export_project(store, path):
    with open(path, "w") as fh:
        json.dump(store.export(), fh, indent=1)
    return {"status": "ok", "path": str(path)}


def import_project(store, path):
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {"status": "format_corrupt",
                "diagnostics": [{"path": str(path), "code": "format_corrupt",
                                 "message": str(exc)}]}
    return store.import_data(data)
