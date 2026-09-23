import numpy as np

from . import binding, integrate

NEED_INTEGRATION = ["mat:power", "mat:schedule", "mat:baseline", "mat:blank",
                    "mat:params.integration"]
NEED_NORM = ["mat:conc", "mat:schedule"]
NEED_FITS = ["mat:conc", "mat:temp", "mat:cellvol", "mat:params.fit"]

R_GAS = 8.314e-3


def run_integration(store, phase=None):
    mats = store.current_materials()
    missing = [m for m in NEED_INTEGRATION if m not in mats]
    parents = {m: mats[m]["version"] for m in NEED_INTEGRATION if m in mats}
    pv = "mat:params.integration@v%s" % parents.get("mat:params.integration", "?")
    with store.write_set("integration") as ws:
        if missing:
            diags = [{"code": "missing_input", "location": m} for m in missing]
            rid, _ = store.add_result(ws, "integration", "incomplete", parents, pv, None, diags)
            store.add_event(ws, "integration_run", {
                "result_id": rid, "status": "incomplete", "missing": missing,
                "phase": phase})
            return rid
        corrupt = [m for m in ("mat:power", "mat:schedule", "mat:baseline", "mat:blank")
                   if mats[m]["status"] == "corrupt"]
        if corrupt:
            diags = [{"code": "format_corrupt", "location": m} for m in corrupt]
            rid, _ = store.add_result(ws, "integration", "format_corrupt", parents, pv, None, diags)
            store.add_event(ws, "integration_run", {
                "result_id": rid, "status": "format_corrupt", "corrupt": corrupt,
                "phase": phase})
            return rid
        params = mats["mat:params.integration"]["norm"]
        power = mats["mat:power"]["norm"]
        sched = mats["mat:schedule"]["norm"]["injections"]
        base = mats["mat:baseline"]["norm"]
        blank = mats["mat:blank"]["norm"]
        cell = integrate.integrate_power(power, sched, base, params)
        bl = integrate.integrate_power(blank["power"], blank["schedule"]["injections"],
                                       base, params)
        payload = {
            "params": params,
            "blank_rule": params.get("blank_rule", "per_index"),
            "injections": cell,
            "blank_injections": bl,
            "units": {"heat": "uJ", "time": "s", "time_basis": "seconds since run start"},
        }
        diags = [d for inj in cell for d in inj["diagnostics"]]
        rid, cached = store.add_result(ws, "integration", "ok", parents, pv, payload, diags)
        if not cached:
            store.add_event(ws, "integration_run", {
                "result_id": rid,
                "status": "ok",
                "phase": phase,
                "blank_rule": payload["blank_rule"],
                "discarded": [i["index"] for i in cell if i["status"] == "discarded"],
                "overlaps": [i["index"] for i in cell
                             if any(d["code"] == "window_overlap" for d in i["diagnostics"])],
                "truncated": [i["index"] for i in cell
                              if any(d["code"] == "window_truncated_at_gap"
                                     for d in i["diagnostics"])],
                "unsolvable": [i["index"] for i in cell
                               if i["status"] == "semantic_unsolvable"],
                "gap_guard": "integration windows never cross data gaps",
            })
    return rid


def run_normheats(store, phase=None):
    mats = store.current_materials()
    integ = store.latest_result("integration")
    with store.write_set("normheats") as ws:
        if integ is None or integ["status"] != "ok":
            rid, _ = store.add_result(ws, "normheats", "incomplete", {}, "n/a", None,
                                      [{"code": "missing_input",
                                        "location": "result:integration"}])
            store.add_event(ws, "normalization", {
                "result_id": rid, "status": "incomplete", "phase": phase})
            return rid
        missing = [m for m in NEED_NORM if m not in mats]
        parents = {"result:integration": integ["result_id"]}
        parents.update({m: mats[m]["version"] for m in NEED_NORM if m in mats})
        pv = integ["param_version"]
        if missing:
            rid, _ = store.add_result(ws, "normheats", "incomplete", parents, pv, None,
                                      [{"code": "missing_input", "location": m}
                                       for m in missing])
            store.add_event(ws, "normalization", {
                "result_id": rid, "status": "incomplete", "phase": phase})
            return rid
        conc = mats["mat:conc"]["norm"]
        sched = mats["mat:schedule"]["norm"]
        ip = integ["payload"]
        rows, fit_points = integrate.normalize_heats(
            ip["injections"], ip["blank_injections"], sched, conc, ip["blank_rule"])
        if conc.get("syringe_mM") is None:
            status = "semantic_unsolvable"
        else:
            status = "ok"
        payload = {
            "injections": rows,
            "fit_points": fit_points,
            "blank_rule": ip["blank_rule"],
            "units": {"norm": "kJ/mol injectant", "heat": "uJ"},
        }
        diags = [d for r in rows for d in r["diagnostics"]]
        rid, cached = store.add_result(ws, "normheats", status, parents, pv, payload, diags)
        if not cached:
            store.add_event(ws, "normalization", {
                "result_id": rid,
                "status": status,
                "phase": phase,
                "blank_rule": ip["blank_rule"],
                "missing_blank": [r["index"] for r in rows
                                  if any(d["code"] == "missing_blank_correspondence"
                                         for d in r["diagnostics"])],
                "fit_points": fit_points,
            })
    return rid


def run_fits(store, phase=None, budget_override=None):
    mats = store.current_materials()
    norm = store.latest_result("normheats")
    with store.write_set("fits") as ws:
        if norm is None or norm["status"] != "ok":
            rid, _ = store.add_result(ws, "fits", "incomplete", {}, "n/a", None,
                                      [{"code": "missing_input",
                                        "location": "result:normheats"}])
            store.add_event(ws, "model_fit", {
                "result_id": rid, "status": "incomplete", "phase": phase})
            return rid
        missing = [m for m in NEED_FITS if m not in mats]
        parents = {"result:normheats": norm["result_id"]}
        parents.update({m: mats[m]["version"] for m in NEED_FITS if m in mats})
        pv = "mat:params.fit@v%s" % parents.get("mat:params.fit", "?")
        if missing:
            rid, _ = store.add_result(ws, "fits", "incomplete", parents, pv, None,
                                      [{"code": "missing_input", "location": m}
                                       for m in missing])
            store.add_event(ws, "model_fit", {
                "result_id": rid, "status": "incomplete", "phase": phase})
            return rid
        params = mats["mat:params.fit"]["norm"]
        budget = int(budget_override or params.get("budget", 4000))
        pv = pv + ";budget=%d" % budget
        fit_points = norm["payload"]["fit_points"]
        if not fit_points:
            rid, _ = store.add_result(ws, "fits", "semantic_unsolvable", parents, pv, None,
                                      [{"code": "no_fit_points",
                                        "location": "normheats.fit_points"}])
            store.add_event(ws, "model_fit", {
                "result_id": rid, "status": "semantic_unsolvable", "phase": phase})
            return rid
        conc = mats["mat:conc"]["norm"]
        temp = mats["mat:temp"]["norm"]
        cellvol = mats["mat:cellvol"]["norm"]
        sched = mats["mat:schedule"]["norm"]
        vols = [inj["vol"] for inj in sched["injections"]]
        y = [norm["payload"]["injections"][i]["norm_kJmol"] for i in fit_points]
        kinds = params.get("models", ["one-site", "two-site", "dilution"])
        tol = float(params.get("equivalence_tol", 0.05))
        try:
            cands = binding.fit_all(kinds, fit_points, y, conc["cell_mM"],
                                    conc["syringe_mM"], cellvol["mL"], vols,
                                    budget, tol_tss=tol)
        except binding.BudgetExhausted:
            rid, _ = store.add_result(ws, "fits", "budget_exhausted", parents, pv, None,
                                      [{"code": "budget_exhausted",
                                        "location": "fits",
                                        "budget": budget}])
            store.add_event(ws, "model_fit", {
                "result_id": rid, "status": "budget_exhausted",
                "budget": budget, "phase": phase})
            return rid
        rid, _ = store.result_id_for("fits", parents, pv)
        t_k = temp["K"]
        candidates = []
        for c in cands:
            cid = "%s#%s" % (rid, c["kind"])
            entry = {
                "id": cid,
                "kind": c["kind"],
                "params": c["params"],
                "sse": c["sse"],
                "evals": c["evals"],
                "equally_good": c["equally_good"],
                "notes": c["notes"],
            }
            if c["kind"] == "one-site":
                entry["dG_kJmol"] = -R_GAS * t_k * float(np.log(c["params"]["K"] * 1000.0))
            candidates.append(entry)
        payload = {
            "candidates": candidates,
            "fit_points": fit_points,
            "equivalence_rule": "sse <= best + tol*TSS, tol=%g" % tol,
            "retained": "all",
            "units": {"norm": "kJ/mol injectant", "T": "K"},
        }
        rid2, cached = store.add_result(ws, "fits", "ok", parents, pv, payload, [])
        assert rid2 == rid
        if not cached:
            store.add_event(ws, "model_fit", {
                "result_id": rid,
                "status": "ok",
                "phase": phase,
                "candidates": [{"id": c["id"], "kind": c["kind"], "sse": c["sse"],
                                "equally_good": c["equally_good"]} for c in candidates],
                "equally_good": [c["id"] for c in candidates if c["equally_good"]],
                "retained": "all",
            })
    return rid


def run_all(store, phase=None, budget_override=None):
    rid_i = run_integration(store, phase=phase)
    rid_n = run_normheats(store, phase=phase)
    rid_f = run_fits(store, phase=phase, budget_override=budget_override)
    return {"integration": rid_i, "normheats": rid_n, "fits": rid_f}
