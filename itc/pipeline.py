"""Derived layer: parse raw materials (kept verbatim) into normalized
quantities, run integration / dilution subtraction / model fitting, and
assemble review states. Unconvertible values stay in place and produce
positioned diagnostics; nothing is silently defaulted.
"""
import json

from . import binding, integrate
from .domain import normalize_quantity
from .ids import digest
from .store import STATUS_OK

MATERIAL_KINDS = ("power_curve", "injection_schedule", "concentrations",
                  "temperature", "cell_volume", "baseline", "blank_curve")


# ---------------- raw -> derived ----------------------------------------
def parse_power(content, path):
    diags, t, p = [], [], []
    units = content.get("units", {})
    for i, row in enumerate(content.get("rows", [])):
        try:
            t_raw = {"value": row[0], "unit": units.get("time", "s")}
            p_raw = {"value": row[1], "unit": units.get("power", "uJ/s")}
            tv, _, d1 = normalize_quantity(t_raw, f"{path}.rows[{i}][0]")
            pv, _, d2 = normalize_quantity(p_raw, f"{path}.rows[{i}][1]")
            for d in (d1, d2):
                if d:
                    diags.append(d)
            if tv is None or pv is None:
                continue  # value stays in raw material; excluded here
            t.append(tv)
            p.append(pv)
        except (IndexError, TypeError) as exc:
            diags.append({"code": "row_malformed", "path": f"{path}.rows[{i}]",
                          "message": str(exc), "severity": "error"})
    return t, p, diags


def parse_schedule(content, path="injection_schedule"):
    diags, injections = [], []
    units = content.get("units", {})
    for i, row in enumerate(content.get("rows", [])):
        try:
            idx = int(row[0])
            tv, _, d = normalize_quantity(
                {"value": row[1], "unit": units.get("time", "s")},
                f"{path}.rows[{i}][1]")
            if d:
                diags.append(d)
            vv, _, d = normalize_quantity(
                {"value": row[2], "unit": units.get("volume", "uL")},
                f"{path}.rows[{i}][2]")
            if d:
                diags.append(d)
            if tv is None or vv is None:
                continue
            injections.append({
                "index": idx, "time_s": tv, "volume_uL": vv * 1e6,
                "volume_L": vv, "duration_s": float(row[3]),
                "first": str(row[4]) == "1"})
        except (IndexError, TypeError, ValueError) as exc:
            diags.append({"code": "row_malformed", "path": f"{path}.rows[{i}]",
                          "message": str(exc), "severity": "error"})
    injections.sort(key=lambda j: (j["time_s"], j["index"]))
    return injections, diags


def parse_baseline(content):
    if content.get("mode") == "constant":
        v, _, d = normalize_quantity(content["value"], "baseline.value")
        return ({"mode": "constant", "value_uJ_s": v} if v is not None else None), \
               ([d] if d else [])
    return None, [{"code": "baseline_mode_unknown", "path": "baseline.mode",
                   "message": f"unsupported baseline mode {content.get('mode')!r}",
                   "severity": "error"}]


def parse_concentrations(content):
    diags = []
    out = {}
    for key in ("syringe", "cell"):
        v, _, d = normalize_quantity(content.get(key), f"concentrations.{key}")
        if d:
            diags.append(d)
        out[key] = v
    # anything else in the material is validated but kept raw
    for key in content:
        if key in ("syringe", "cell", "label"):
            continue
        raw = content[key]
        if isinstance(raw, dict) and "unit" in raw:
            _, _, d = normalize_quantity(raw, f"concentrations.{key}")
            if d:
                diags.append(d)
    return out, diags


def parse_cell_volume(content):
    v, _, d = normalize_quantity(content, "cell_volume")
    return v, ([d] if d else [])


def parse_temperature(content):
    v, _, d = normalize_quantity(content, "temperature")
    return v, ([d] if d else [])


def _content(store, kind, seq=None, branch=None):
    mats = store.current_materials(seq=seq)
    row = mats.get(kind)
    return (json.loads(row["content"]), row) if row else (None, None)


# ---------------- computations ------------------------------------------
def run_integration(store, branch="main", budget=None):
    """Integrate sample and blank against current materials; subtract blank."""
    pc, pc_row = _content(store, "power_curve")
    sc, sc_row = _content(store, "injection_schedule")
    bl, bl_row = _content(store, "baseline")
    bc, bc_row = _content(store, "blank_curve")
    missing = [k for k, r in (("power_curve", pc_row), ("injection_schedule", sc_row),
                              ("baseline", bl_row), ("blank_curve", bc_row)) if r is None]
    if missing:
        return None, [{"code": "material_missing", "path": ",".join(missing),
                       "message": f"materials missing: {missing}",
                       "severity": "error"}], "unsatisfiable"

    t, p, d1 = parse_power(pc, "power_curve")
    injections, d2 = parse_schedule(sc)
    baseline, d3 = parse_baseline(bl)
    tb, pb, d4 = parse_power(bc, "blank_curve")
    parse_diags = d1 + d2 + d3 + d4
    if baseline is None:
        return None, parse_diags, "unsatisfiable"

    parents = {"power_curve": pc_row["version"],
               "injection_schedule": sc_row["version"],
               "baseline": bl_row["version"]}
    inp = digest({"t": t, "p": p, "inj": injections, "baseline": baseline})
    payload, diags, status = integrate.integrate(t, p, injections, baseline,
                                                 budget=budget)
    sample_cmp = store.add_computation(
        "integration", status, inp, integrate.PARAM_VERSION, parents,
        {**payload, "series": "sample"}, parse_diags + diags, branch=branch)

    parents_b = {"blank_curve": bc_row["version"],
                 "injection_schedule": sc_row["version"],
                 "baseline": bl_row["version"]}
    inp_b = digest({"t": tb, "p": pb, "inj": injections, "baseline": baseline})
    payload_b, diags_b, status_b = integrate.integrate(tb, pb, injections,
                                                       baseline, budget=budget)
    blank_cmp = store.add_computation(
        "integration", status_b, inp_b, integrate.PARAM_VERSION, parents_b,
        {**payload_b, "series": "blank"}, parse_diags + diags_b, branch=branch)

    # dilution subtraction requires an explicit correspondence rule
    rule = {"type": "by_index"}
    blank_inj = [dict(index=b["index"], heat_uJ=b["heat_uJ"],
                      volume_uL=injections[b["index"]]["volume_uL"]
                      if b["index"] < len(injections) else None)
                 for b in payload_b["injections"]]
    sample_inj = [dict(s, volume_uL=injections[s["index"]]["volume_uL"])
                  for s in payload["injections"] if s["index"] < len(injections)]
    blank_inj = [b for b in blank_inj if b["index"] < len(injections)]
    sub, sub_diags, sub_status = integrate.apply_blank(sample_inj, blank_inj, rule)
    parents_s = {"sample_integration": sample_cmp["result_id"],
                 "blank_integration": blank_cmp["result_id"]}
    sub_cmp = store.add_computation(
        "dilution_subtraction", sub_status,
        digest({"rule": rule, "sample": sample_cmp["result_id"],
                "blank": blank_cmp["result_id"]}),
        integrate.PARAM_VERSION, parents_s, sub or {}, sub_diags, branch=branch)

    return {"sample": sample_cmp, "blank": blank_cmp, "subtraction": sub_cmp,
            "rule": rule}, parse_diags + diags + diags_b + sub_diags, STATUS_OK


def run_fit(store, excluded=(), decision_parents=(), branch="main", budget=None):
    subs = store.computations(kind="dilution_subtraction")
    subs = [s for s in subs if s["status"] == STATUS_OK]
    if not subs:
        return None, [{"code": "no_dilution_subtraction", "path": "computations",
                       "message": "no successful dilution subtraction to fit",
                       "severity": "error"}], "unsatisfiable"
    sub = json.loads(subs[-1]["payload"])
    conc, conc_row = _content(store, "concentrations")
    cv, cv_row = _content(store, "cell_volume")
    tp, tp_row = _content(store, "temperature")
    conc_v, d1 = parse_concentrations(conc)
    V0, d2 = parse_cell_volume(cv)
    T, d3 = parse_temperature(tp)
    parse_diags = d1 + d2 + d3

    entries = [e for e in sub.get("injections", [])
               if e.get("corrected_uJ") is not None
               and e["index"] not in excluded]
    if not entries:
        return None, parse_diags + [{
            "code": "no_fittable_injections", "path": "injections",
            "message": "no injections with corrected heat remain",
            "severity": "error"}], "unsatisfiable"

    Xs = conc_v.get("syringe")
    M0 = conc_v.get("cell")
    if Xs is None or M0 is None or V0 is None:
        return None, parse_diags + [{
            "code": "derived_input_missing", "path": "concentrations/cell_volume",
            "message": "required derived quantities unavailable",
            "severity": "error"}], "unsatisfiable"

    dV = [e["volume_uL"] * 1e-6 for e in entries]
    y = [e["corrected_uJ"] * 1e-9 / (e["volume_uL"] * 1e-6 * Xs)
         for e in entries]  # uJ -> kJ/mol of injectant
    candidates, diags, status = binding.fit_all(M0, Xs, V0, dV, y, budget=budget)
    import numpy as np
    f = np.cumprod(1.0 - np.asarray(dV) / V0)
    molar_ratios = [float(Xs * (1.0 - fi) / (M0 * fi)) for fi in f]
    for c in candidates:
        c["injection_indices"] = [e["index"] for e in entries]
        c["molar_ratios"] = molar_ratios
    parents = {"dilution_subtraction": subs[-1]["result_id"],
               "concentrations": conc_row["version"],
               "cell_volume": cv_row["version"],
               "temperature": tp_row["version"],
               "decisions": list(decision_parents)}
    inp = digest({"y": y, "dV": dV, "M0": M0, "Xs": Xs, "V0": V0,
                  "excluded": sorted(excluded)})
    cmp_ = store.add_computation(
        "model_fit", status, inp, binding.PARAM_VERSION, parents,
        {"candidates": candidates, "excluded": sorted(excluded),
         "y_kJ_mol": y, "molar_ratios": molar_ratios, "temperature_K": T},
        parse_diags + diags, branch=branch)
    return cmp_, parse_diags + diags, status


# ---------------- review state ------------------------------------------
def build_state(store, seq=None, branch=None):
    mats = store.current_materials(seq=seq)
    materials = {}
    norm_diags = []
    for kind, row in mats.items():
        content = json.loads(row["content"])
        materials[kind] = {
            "material_id": f"{kind}@v{row['version']}", "kind": kind,
            "version": row["version"], "digest": row["digest"],
            "seq": row["seq"], "content": content}
    derived = {}
    if "power_curve" in mats:
        t, p, d = parse_power(json.loads(mats["power_curve"]["content"]), "power_curve")
        norm_diags += d
        gaps = integrate.find_gaps(t) if t else []
        derived["power"] = {"time_s": t, "power_uJ_s": p, "gaps": gaps,
                            "unit": "uJ/s", "time_basis": "s since run start"}
    if "blank_curve" in mats:
        t, p, d = parse_power(json.loads(mats["blank_curve"]["content"]), "blank_curve")
        norm_diags += d
        derived["blank"] = {"time_s": t, "power_uJ_s": p, "unit": "uJ/s"}
    if "injection_schedule" in mats:
        inj, d = parse_schedule(json.loads(mats["injection_schedule"]["content"]))
        norm_diags += d
        derived["injections"] = inj

    comps = store.computations(seq=seq)
    integrations = [c for c in comps if c["kind"] == "integration"]
    subtraction = [c for c in comps if c["kind"] == "dilution_subtraction"]
    fits = [c for c in comps if c["kind"] == "model_fit"]

    def unpack(c):
        return {"result_id": c["result_id"], "status": c["status"],
                "seq": c["seq"], "branch": c["branch"],
                "param_version": c["param_version"],
                "parents": json.loads(c["parents"]),
                "payload": json.loads(c["payload"]),
                "diagnostics": json.loads(c["diagnostics"])}

    integ = {c["payload"] and json.loads(c["payload"]).get("series"): unpack(c)
             for c in integrations}
    sub = unpack(subtraction[-1]) if subtraction else None

    # model staleness is relative to the viewed seq
    conc_version = (materials.get("concentrations") or {}).get("version")
    models = []
    for c in fits:
        u = unpack(c)
        u["stale"] = (conc_version is not None
                      and u["parents"].get("concentrations") != conc_version)
        models.append(u)

    decisions = [dict(d, payload=json.loads(d["payload"]))
                 for d in store.decisions(seq=seq)]
    events = [dict(e, payload=json.loads(e["payload"]),
                   parents=json.loads(e["parents"]))
              for e in store.events(seq=seq)]
    all_diags = norm_diags
    for c in comps:
        all_diags += json.loads(c["diagnostics"])

    return {"seq": seq if seq is not None else store.max_seq(),
            "branches": store.branches(), "materials": materials,
            "derived": derived, "integrations": integ, "subtraction": sub,
            "models": models, "decisions": decisions, "events": events,
            "diagnostics": all_diags}
