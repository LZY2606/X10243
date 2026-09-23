"""Orchestration: pull current material versions, run a computation, and
record the result with input digest, parameter version and parent deps.
Incomplete input produces a located diagnostic result, never a fake success."""

import hashlib
import json

from . import integrate as integ
from . import models
from .units import convert, temperature_to_kelvin, UnitUnconvertible

PARAMS_VERSION = "pipeline-v1"


def digest(kind, deps, params):
    blob = json.dumps({"kind": kind, "deps": deps, "params": params,
                       "params_version": PARAMS_VERSION}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _missing(name, kind):
    return {"code": "missing_input", "location": {"material": f"mat-{name}"},
            "message": f"required material of kind {kind!r} is not present"}


def _record(store, kind, deps, params, status, diagnostics, payload, parents):
    dig = digest(kind, deps, params)
    with store.txn() as (cur, txn_id):
        event = store.add_event(cur, txn_id, f"compute:{kind}",
                                {"kind": kind, "status": status,
                                 "params": params, "digest": dig}, parents)
        result_id = store.put_result(cur, txn_id, kind, event["id"], dig,
                                     params, deps, status, diagnostics, payload)
    return result_id, event


def run_integration(store, params, parents=None):
    mats = store.current_materials()
    need = [("power", "power_curve"), ("injections", "injection_schedule"),
            ("baseline", "baseline")]
    missing = [_missing(n, k) for n, k in need if f"mat-{n}" not in mats]
    deps = {"materials": {mats[f"mat-{n}"]["id"]: mats[f"mat-{n}"]["version"]
                          for n, _ in need if f"mat-{n}" in mats},
            "results": []}
    if missing:
        return _record(store, "integration", deps, params, "diagnostic",
                       missing, {}, parents or [])
    try:
        payload = integ.integrate(mats["mat-power"]["payload"],
                                  mats["mat-injections"]["payload"],
                                  mats["mat-baseline"]["payload"], params)
        return _record(store, "integration", deps, params, payload["status"],
                       payload["diagnostics"], payload, parents or [])
    except integ.SemanticUnsolvable as exc:
        diag = [{"code": "semantic_unsolvable", "location": {"material": "mat-power"},
                 "message": str(exc)}]
        return _record(store, "integration", deps, params,
                       "semantic_unsolvable", diag, {}, parents or [])


def _latest_result(store, kind):
    candidates = [r for r in store.results() if r["kind"] == kind]
    if not candidates:
        return None
    seq_by_event = {e["id"]: e["seq"] for e in store.events()}
    return max(candidates, key=lambda r: seq_by_event[r["event_id"]])


def run_blank_correction(store, params, parents=None):
    mats = store.current_materials()
    integration = _latest_result(store, "integration")
    missing = []
    if "mat-blank" not in mats:
        missing.append(_missing("blank", "blank_curve"))
    if integration is None or integration["status"] not in ("ok", "diagnostic") \
            or not integration["payload"].get("injections"):
        missing.append({"code": "missing_input",
                        "location": {"result": "integration"},
                        "message": "no usable integration result to correct"})
    deps = {"materials": ({"mat-blank": mats["mat-blank"]["version"]}
                          if "mat-blank" in mats else {}),
            "results": [integration["id"]] if integration else []}
    if missing:
        return _record(store, "blank_correction", deps, params, "diagnostic",
                       missing, {}, parents or [])
    payload = integ.correct_blank(integration["payload"],
                                  mats["mat-blank"]["payload"], params)
    return _record(store, "blank_correction", deps, params, payload["status"],
                   payload["diagnostics"] + [d for inj in payload["injections"]
                                             for d in inj["diagnostics"]],
                   payload, parents or [])


def _conditions(mats, diagnostics):
    conc = mats["mat-concentrations"]["payload"]
    def conc_of(field):
        entry = conc[field]
        return convert(entry["value"], entry["unit"], "mM", "concentration") * 1e-3
    for field, entry in conc.items():
        try:
            convert(entry["value"], entry["unit"], "mM", "concentration")
        except (UnitUnconvertible, KeyError, TypeError) as exc:
            diagnostics.append({"code": "unit_unconvertible",
                                "location": {"material": "mat-concentrations",
                                             "field": field},
                                "message": f"value for {field!r} kept raw: {exc}"})
    vol = mats["mat-cell_volume"]["payload"]
    v0 = convert(vol["value"], vol["unit"], "uL", "volume") * 1e-6
    temp = mats["mat-temperature"]["payload"]
    temperature_to_kelvin(temp["value"], temp["unit"])
    inj = mats["mat-injections"]["payload"]
    dv = [convert(e["volume"], inj.get("volume_unit", "uL"), "uL", "volume") * 1e-6
          for e in sorted(inj["entries"], key=lambda e: e["index"])]
    return {"V0": v0, "M0": conc_of("cell"), "Xs": conc_of("syringe"), "dv": dv}


def run_fit(store, params, parents=None):
    mats = store.current_materials()
    correction = _latest_result(store, "blank_correction")
    missing = []
    for name, kind in [("concentrations", "concentrations"),
                       ("cell_volume", "cell_volume"), ("temperature", "temperature"),
                       ("injections", "injection_schedule")]:
        if f"mat-{name}" not in mats:
            missing.append(_missing(name, kind))
    if correction is None or not correction["payload"].get("injections"):
        missing.append({"code": "missing_input", "location": {"result": "blank_correction"},
                        "message": "no usable blank correction to fit"})
    deps = {"materials": {mats[f"mat-{n}"]["id"]: mats[f"mat-{n}"]["version"]
                          for n in ("concentrations", "cell_volume", "temperature",
                                    "injections") if f"mat-{n}" in mats},
            "results": [correction["id"]] if correction else []}
    if missing:
        return _record(store, "model_fit", deps, params, "diagnostic",
                       missing, {}, parents or [])
    unit_diags = []
    try:
        cond = _conditions(mats, unit_diags)
    except (UnitUnconvertible, KeyError, TypeError) as exc:
        diag = [{"code": "semantic_unsolvable",
                 "location": {"material": "mat-concentrations"}, "message": str(exc)}]
        return _record(store, "model_fit", deps, params,
                       "semantic_unsolvable", diag, {}, parents or [])
    injections = correction["payload"]["injections"]
    by_index = {inj["index"]: inj for inj in injections}
    heats, included = [], []
    for i in range(len(cond["dv"])):
        rec = by_index.get(i)
        usable = rec is not None and rec["status"] == "ok" and not rec["excluded"] \
            and rec["corrected_ucal"] is not None
        heats.append(rec["corrected_ucal"] if usable else 0.0)
        included.append(usable)
    only = params.get("only_injections")
    if only is not None:
        included = [ok and i in only for i, ok in enumerate(included)]
    try:
        fit = models.fit_models(
            heats, included, cond,
            budget=int(params.get("budget", 20000)),
            equipotent_rel_tol=float(params.get("equipotent_rel_tol", 0.05)),
            equipotent_abs_tol=float(params.get("equipotent_abs_tol", 1e-6)))
    except models.BudgetExhausted as exc:
        diag = [{"code": "budget_exhausted", "location": {"result": "model_fit"},
                 "message": str(exc)}]
        return _record(store, "model_fit", deps, params,
                       "budget_exhausted", diag, {}, parents or [])
    except models.SemanticUnsolvable as exc:
        diag = [{"code": "semantic_unsolvable", "location": {"result": "model_fit"},
                 "message": str(exc)}]
        return _record(store, "model_fit", deps, params,
                       "semantic_unsolvable", diag, {}, parents or [])
    fit["conditions"] = {"V0_L": cond["V0"], "M0_M": cond["M0"], "Xs_M": cond["Xs"]}
    fit["heats_ucal"] = heats
    fit["included"] = included
    return _record(store, "model_fit", deps, params, "ok",
                   unit_diags, fit, parents or [])
