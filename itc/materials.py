from .units import convert_value


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _qty(q, kind, location, diags, default_unit):
    if isinstance(q, dict):
        value = q.get("value")
        unit = q.get("unit", default_unit)
    else:
        value = q
        unit = default_unit
    val, diag = convert_value(value, unit, kind, location)
    if diag:
        diags.append(diag)
        return None
    return val


def _norm_power(raw, loc, diags):
    ut = raw.get("unit_t", "s")
    up = raw.get("unit_p", "uW")
    samples = raw.get("samples")
    if not isinstance(samples, list):
        diags.append({
            "code": "format_corrupt",
            "location": loc + "samples",
            "message": "samples is not a list",
        })
        return None
    t = []
    p = []
    for i, row in enumerate(samples):
        if not (isinstance(row, (list, tuple)) and len(row) == 2
                and _is_num(row[0]) and _is_num(row[1])):
            diags.append({
                "code": "format_corrupt",
                "location": "%ssamples[%d]" % (loc, i),
                "raw": row,
                "message": "sample row must be [t, power] numbers",
            })
            continue
        tv, td = convert_value(row[0], ut, "time", "%ssamples[%d][0]" % (loc, i))
        pv, pd = convert_value(row[1], up, "power", "%ssamples[%d][1]" % (loc, i))
        if td:
            diags.append(td)
        if pd:
            diags.append(pd)
        if tv is None or pv is None:
            continue
        t.append(tv)
        p.append(pv)
    gaps = []
    for j, g in enumerate(raw.get("gaps", [])):
        if isinstance(g, (list, tuple)) and len(g) == 2 and _is_num(g[0]) and _is_num(g[1]):
            g0, d0 = convert_value(g[0], ut, "time", "%sgaps[%d][0]" % (loc, j))
            g1, d1 = convert_value(g[1], ut, "time", "%sgaps[%d][1]" % (loc, j))
            for d in (d0, d1):
                if d:
                    diags.append(d)
            if g0 is not None and g1 is not None:
                gaps.append([g0, g1])
        else:
            diags.append({
                "code": "format_corrupt",
                "location": "%sgaps[%d]" % (loc, j),
                "raw": g,
            })
    return {"t": t, "p": p, "gaps": gaps, "units": {"t": "s", "p": "uW"}}


def _norm_schedule(raw, loc, diags):
    ut = raw.get("t_unit", "s")
    uv = raw.get("vol_unit", "uL")
    injections = []
    rows = raw.get("injections")
    if not isinstance(rows, list):
        diags.append({
            "code": "format_corrupt",
            "location": loc + "injections",
            "message": "injections is not a list",
        })
        return None
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or "t" not in row:
            diags.append({
                "code": "format_corrupt",
                "location": "%sinjections[%d]" % (loc, i),
                "raw": row,
            })
            continue
        tv, td = convert_value(row.get("t"), ut, "time", "%sinjections[%d].t" % (loc, i))
        if td:
            diags.append(td)
        vol = _qty(row.get("vol"), "volume", "%sinjections[%d].vol" % (loc, i), diags, uv)
        injections.append({"t": tv, "vol": vol, "vol_raw": row.get("vol")})
    return {"injections": injections, "units": {"t": "s", "vol": "uL"}}


def _norm_conc(raw, loc, diags):
    unit = raw.get("unit", "mM")
    return {
        "syringe_mM": _qty(raw.get("syringe"), "conc", loc + "syringe", diags, unit),
        "cell_mM": _qty(raw.get("cell"), "conc", loc + "cell", diags, unit),
        "units": "mM",
    }


def _norm_temp(raw, loc, diags):
    unit = raw.get("unit") if isinstance(raw, dict) else None
    return {"K": _qty(raw, "temp", loc + "temperature", diags, unit), "units": "K"}


def _norm_cellvol(raw, loc, diags):
    unit = raw.get("unit", "mL") if isinstance(raw, dict) else "mL"
    return {"mL": _qty(raw, "cellvol", loc + "cell_volume", diags, unit), "units": "mL"}


def _norm_baseline(raw, loc, diags):
    ut = raw.get("unit_t", "s")
    up = raw.get("unit_p", "uW")
    points = []
    for i, row in enumerate(raw.get("points", [])):
        if not (isinstance(row, (list, tuple)) and len(row) == 2 and _is_num(row[0]) and _is_num(row[1])):
            diags.append({
                "code": "format_corrupt",
                "location": "%spoints[%d]" % (loc, i),
                "raw": row,
            })
            continue
        tv, td = convert_value(row[0], ut, "time", "%spoints[%d][0]" % (loc, i))
        pv, pd = convert_value(row[1], up, "power", "%spoints[%d][1]" % (loc, i))
        for d in (td, pd):
            if d:
                diags.append(d)
        if tv is not None and pv is not None:
            points.append([tv, pv])
    return {"kind": raw.get("kind", "linear"), "points": points,
            "units": {"t": "s", "p": "uW"}}


def _norm_blank(raw, loc, diags):
    power = _norm_power(raw.get("power", {}), loc + "power.", diags)
    schedule = _norm_schedule(raw.get("schedule", {}), loc + "schedule.", diags)
    return {"power": power, "schedule": schedule}


_NORMALIZERS = {
    "power": _norm_power,
    "schedule": _norm_schedule,
    "conc": _norm_conc,
    "temp": _norm_temp,
    "cellvol": _norm_cellvol,
    "baseline": _norm_baseline,
    "blank": _norm_blank,
}


def normalize(material_id, kind, raw):
    """Normalize a raw material into the derived layer.

    The raw representation is always preserved by the store; normalization
    only adds a derived view plus located diagnostics. Returns
    (norm, diagnostics, status) with status in ok/degraded/corrupt.
    """
    diags = []
    fn = _NORMALIZERS.get(kind)
    if fn is None:
        norm = raw
    else:
        norm = fn(raw, "", diags)
    if any(d["code"] == "format_corrupt" for d in diags):
        status = "corrupt"
    elif diags:
        status = "degraded"
    else:
        status = "ok"
    return norm, diags, status
