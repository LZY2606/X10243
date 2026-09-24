"""Injection integration and blank (dilution heat) subtraction.

Rules encoded here:
- Post-injection peaks may overlap; the integration window ends at the next
  injection and the overlap is recorded, never silently extended.
- An integration window never crosses a data gap; it is truncated at the gap
  edge and a positioned diagnostic is emitted.
- The first injection can be discarded (excluded, not deleted).
- Blank subtraction must state its correspondence rule; missing blank
  correspondence keeps the raw heat in place with a diagnostic.
"""

import copy

from . import units


def _factor(from_unit, to_unit, path, diagnostics):
    f, diag = units.factor(from_unit, to_unit)
    if diag is not None:
        diag = dict(diag, path=path)
        diagnostics.append(diag)
        return 1.0, False
    return f, True


def integrate_injections(power, schedule, baseline, params=None):
    """Integrate each injection peak of a power trace.

    power:     {"time": {"values": [...], "unit": "s", "basis": ...},
                "power": {"values": [...], "unit": "uJ/s"},
                "gaps": [[t0, t1], ...]}
    schedule:  {"injections": [{"time": {"value","unit"},
                                "volume": {"value","unit"}}, ...]}
    baseline:  {"kind": "linear", "a": {"value","unit"}, "b": {"value","unit"},
                "t_ref": {"value","unit"}}
    params:    {"discard_first": bool, "max_window_s": float}
    """
    params = dict(params or {})
    diagnostics = []

    tf, t_ok = _factor(power["time"]["unit"], "s", "power_curve.time.unit", diagnostics)
    pf, p_ok = _factor(power["power"]["unit"], "J/s", "power_curve.power.unit", diagnostics)
    t = [v * tf for v in power["time"]["values"]]
    p = [v * pf for v in power["power"]["values"]]
    heat_unit = "J" if (t_ok and p_ok) else (
        f"{power['power']['unit']}*{power['time']['unit']}")
    gaps = [[g[0] * tf, g[1] * tf] for g in power.get("gaps", [])]

    a = baseline["a"]["value"] * pf
    b = baseline["b"]["value"] * pf
    t_ref = baseline.get("t_ref", {}).get("value", 0.0) * tf

    def base_at(x):
        return a + b * (x - t_ref)

    inj_t, inj_vol, inj_vol_l = [], [], []
    for i, inj in enumerate(schedule["injections"]):
        tv, diag = units.normalize(inj["time"]["value"], inj["time"]["unit"], "s")
        if diag:
            diagnostics.append(dict(diag, path=f"injection_schedule.injections[{i}].time"))
        inj_t.append(tv)
        vv, vdiag = units.normalize(inj["volume"]["value"], inj["volume"]["unit"], "L")
        if vdiag:
            diagnostics.append(dict(vdiag, path=f"injection_schedule.injections[{i}].volume"))
            inj_vol_l.append(None)
        else:
            inj_vol_l.append(vv)
        inj_vol.append({"value": inj["volume"]["value"], "unit": inj["volume"]["unit"]})

    max_window = params.get("max_window_s", 240.0)
    discard_first = bool(params.get("discard_first", False))
    n = len(inj_t)
    heats = []
    for i in range(n):
        t0 = inj_t[i]
        ended_by_next = i + 1 < n and inj_t[i + 1] <= min(t0 + max_window, t[-1])
        t1 = min(inj_t[i + 1] if i + 1 < n else t[-1], t0 + max_window, t[-1])
        truncated = False
        for g in gaps:
            if t0 < g[0] and t1 > g[0]:
                t1 = g[0]
                truncated = True
                diagnostics.append({
                    "code": "window_truncated_at_gap",
                    "injection": i,
                    "window_end": t1,
                    "gap": g,
                })
        in_gap = any(g[0] <= t0 < g[1] for g in gaps)
        seg = [(tk, pk) for tk, pk in zip(t, p) if t0 <= tk <= t1]
        entry = {
            "index": i,
            "time_s": t0,
            "window": [t0, t1],
            "volume": inj_vol[i],
            "volume_L": inj_vol_l[i],
            "excluded": False,
            "reason": None,
            "overlap": False,
            "truncated": truncated,
            "status": "ok",
            "heat": None,
        }
        if in_gap or len(seg) < 2:
            entry["status"] = "no_data"
            diagnostics.append({
                "code": "injection_window_no_data",
                "injection": i,
                "position": t0,
            })
        else:
            area = 0.0
            for (ta, pa), (tb, pb) in zip(seg, seg[1:]):
                area += 0.5 * ((pa - base_at(ta)) + (pb - base_at(tb))) * (tb - ta)
            entry["heat"] = {"value": area, "unit": heat_unit}
            amp = max(pk - base_at(tk) for tk, pk in seg)
            end_res = seg[-1][1] - base_at(seg[-1][0])
            if ended_by_next and amp != 0 and abs(end_res) > 0.05 * abs(amp):
                entry["overlap"] = True
        if discard_first and i == 0 and entry["status"] == "ok":
            entry["excluded"] = True
            entry["reason"] = "first_injection_discarded"
        heats.append(entry)

    ok_count = sum(1 for h in heats if h["status"] == "ok")
    if not (t_ok and p_ok):
        status = "incomplete"
    elif not diagnostics:
        status = "ok"
    elif ok_count:
        status = "partial"
    else:
        status = "incomplete"
    return {
        "status": status,
        "heats": heats,
        "diagnostics": diagnostics,
        "basis": {
            "time": "s",
            "time_basis": power["time"].get("basis", "unknown"),
            "heat": heat_unit,
        },
    }


def subtract_blank(sample_result, blank_result, rule):
    """Subtract blank (dilution) heats. The correspondence rule is mandatory
    and recorded; missing correspondences keep raw heats with diagnostics."""
    diagnostics = []
    heats = copy.deepcopy(sample_result["heats"])
    kind = (rule or {}).get("kind")

    def blank_ok(h):
        return h is not None and h.get("status") == "ok" and h.get("heat") is not None

    if kind == "by_index":
        bmap = {h["index"]: h for h in blank_result["heats"]}
        total = missing = 0
        for h in heats:
            if h["excluded"] or h["status"] != "ok" or h["heat"] is None:
                continue
            total += 1
            b = bmap.get(h["index"])
            if not blank_ok(b):
                diagnostics.append({
                    "code": "blank_correspondence_missing",
                    "injection": h["index"],
                })
                h["blank_corrected"] = False
                missing += 1
            else:
                h["heat_blank"] = b["heat"]
                h["heat_net"] = {
                    "value": h["heat"]["value"] - b["heat"]["value"],
                    "unit": h["heat"]["unit"],
                }
                h["blank_corrected"] = True
        if total and missing == total:
            return {"status": "semantic_unsolvable", "rule": rule, "heats": heats,
                    "diagnostics": diagnostics + [{
                        "code": "no_blank_correspondence",
                        "detail": "no injection has a usable blank counterpart"}],
                    "basis": sample_result["basis"]}
    elif kind == "mean_tail":
        count = int(rule.get("count", 3))
        tail = [h for h in blank_result["heats"] if blank_ok(h)][-count:]
        if not tail:
            return {"status": "semantic_unsolvable", "rule": rule, "heats": heats,
                    "diagnostics": [{"code": "no_blank_correspondence",
                                     "detail": "blank curve has no usable heats"}],
                    "basis": sample_result["basis"]}
        mean = sum(h["heat"]["value"] for h in tail) / len(tail)
        for h in heats:
            if h["excluded"] or h["status"] != "ok" or h["heat"] is None:
                continue
            h["heat_blank"] = {"value": mean, "unit": h["heat"]["unit"]}
            h["heat_net"] = {"value": h["heat"]["value"] - mean,
                             "unit": h["heat"]["unit"]}
            h["blank_corrected"] = True
    else:
        return {"status": "semantic_unsolvable", "rule": rule, "heats": heats,
                "diagnostics": [{"code": "unknown_blank_rule", "rule": rule}],
                "basis": sample_result["basis"]}

    corrected = sum(1 for h in heats if h.get("blank_corrected"))
    if not diagnostics:
        status = "ok"
    elif corrected:
        status = "partial"
    else:
        status = "incomplete"
    return {"status": status, "rule": rule, "heats": heats,
            "diagnostics": diagnostics, "basis": sample_result["basis"]}
