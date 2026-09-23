import numpy as np

try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz


def baseline_at(points, t):
    pts = np.asarray(points, dtype=float)
    return np.interp(t, pts[:, 0], pts[:, 1])


def integrate_power(power_norm, injections, baseline_norm, params):
    """Integrate each injection peak against the baseline.

    Rules encoded here (mirrored as machine-readable diagnostics):
    - post-injection peaks may overlap; overlapping windows are allowed and
      recorded, never silently merged;
    - an integration window never crosses a data gap: it is truncated at the
      gap edge (partial) or reported unsolvable when it starts inside a gap;
    - the first injection can be flagged discarded by parameter, its heat is
      still computed and archived.
    """
    window = float(params.get("window_s", 150.0))
    discard_first = bool(params.get("discard_first", False))
    t = np.asarray(power_norm["t"], dtype=float)
    p = np.asarray(power_norm["p"], dtype=float)
    gaps = power_norm.get("gaps", [])
    base = baseline_at(baseline_norm["points"], t)
    out = []
    for i, inj in enumerate(injections):
        t0 = float(inj["t"])
        t1 = t0 + window
        diags = []
        status = "ok"
        if any(g0 <= t0 < g1 for g0, g1 in gaps):
            gap = next(g for g in gaps if g[0] <= t0 < g[1])
            diags.append({
                "code": "window_start_in_gap",
                "location": "injections[%d]" % i,
                "gap": gap,
            })
            out.append({
                "index": i, "t_inj": t0, "window": [t0, t1],
                "status": "semantic_unsolvable", "heat_uJ": None,
                "diagnostics": diags,
            })
            continue
        for g0, g1 in gaps:
            if t0 < g0 < t1:
                t1 = g0
                status = "partial"
                diags.append({
                    "code": "window_truncated_at_gap",
                    "location": "injections[%d]" % i,
                    "gap": [g0, g1],
                    "rule": "integration window never crosses a data gap",
                })
        if i + 1 < len(injections) and t1 > float(injections[i + 1]["t"]):
            diags.append({
                "code": "window_overlap",
                "location": "injections[%d]" % i,
                "with": i + 1,
                "rule": "post-injection peaks may overlap; shared region kept",
            })
        mask = (t >= t0) & (t <= t1)
        heat = None
        if int(mask.sum()) < 3:
            status = "semantic_unsolvable"
            diags.append({
                "code": "insufficient_samples",
                "location": "injections[%d]" % i,
                "n_samples": int(mask.sum()),
            })
        else:
            heat = float(_trapezoid((p - base)[mask], t[mask]))
        if i == 0 and discard_first and status == "ok":
            status = "discarded"
            diags.append({
                "code": "first_injection_discarded",
                "location": "injections[0]",
                "rule": "discard_first parameter",
            })
        out.append({
            "index": i, "t_inj": t0, "window": [t0, t1],
            "status": status, "heat_uJ": heat, "diagnostics": diags,
        })
    return out


def normalize_heats(cell_injections, blank_injections, schedule_norm, conc_norm, blank_rule):
    """Blank-subtract and molar-normalize integrated heats.

    The blank (dilution-heat) subtraction always names its correspondence
    rule; a missing blank counterpart yields a located diagnostic and a None
    value, never a zero or a default.
    """
    syringe = conc_norm.get("syringe_mM")
    out = []
    for i, ci in enumerate(cell_injections):
        diags = []
        blank_heat = None
        if i < len(blank_injections):
            blank_heat = blank_injections[i]["heat_uJ"]
        else:
            diags.append({
                "code": "missing_blank_correspondence",
                "location": "injections[%d]" % i,
                "rule": blank_rule,
                "message": "no blank injection corresponds under rule %s" % blank_rule,
            })
        vol = schedule_norm["injections"][i]["vol"]
        if vol is None:
            diags.append({
                "code": "missing_volume",
                "location": "injections[%d].vol" % i,
                "cause": "unconvertible_value",
            })
        if syringe is None:
            diags.append({
                "code": "missing_concentration",
                "location": "conc.syringe",
            })
        norm = None
        if (ci["heat_uJ"] is not None and blank_heat is not None
                and vol is not None and syringe is not None):
            n_umol = vol * syringe * 1e-3
            norm = (ci["heat_uJ"] - blank_heat) / (n_umol * 1000.0)
        out.append({
            "index": i,
            "status": ci["status"],
            "heat_uJ": ci["heat_uJ"],
            "blank_uJ": blank_heat,
            "vol_uL": vol,
            "norm_kJmol": norm,
            "diagnostics": diags,
        })
    fit_points = [r["index"] for r in out
                  if r["status"] == "ok" and r["norm_kJmol"] is not None]
    return out, fit_points
