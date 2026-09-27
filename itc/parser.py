"""Import raw ITC bundles.

Input format is JSON (a domain bundle, NOT a generic spreadsheet). The parser
keeps every original string (raw) and only adds a normalized layer. It returns
structured, positioned diagnostics and distinguishes the three required failure
outcomes: format_corrupt / semantic_unsolvable / budget_exhausted.
"""
from __future__ import annotations

import json
import math
from typing import Any

from . import materials as M
from .diagnostics import (FormatCorruptError, SemanticUnsolvable,
                          BudgetExhausted, diag)
from .units import quantity

BUNDLE_SCHEMA = "itc.bundle/v1"
MAX_POINTS_DEFAULT = 200_000


def parse_bytes(blob: bytes | str, *, max_points: int = MAX_POINTS_DEFAULT):
    """Stage 1: structural decode. Failure => FormatCorruptError."""
    try:
        text = blob.decode("utf-8") if isinstance(blob, bytes) else blob
    except UnicodeDecodeError as exc:
        raise FormatCorruptError(f"payload is not valid UTF-8: {exc}",
                                 position="$")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FormatCorruptError(
            f"invalid JSON at line {exc.lineno} col {exc.colno}: {exc.msg}",
            position=f"$[{exc.pos}]")
    if not isinstance(obj, dict):
        raise FormatCorruptError("bundle root must be an object",
                                 position="$")
    return parse_bundle(obj, max_points=max_points)


def _require(obj: dict, key: str, typ, pos: str):
    if key not in obj:
        raise SemanticUnsolvable(f"missing required field '{key}'",
                                 position=pos)
    val = obj[key]
    if not isinstance(val, typ) or (typ is not bool and isinstance(val, bool)
                                    and typ in (int, float)):
        name = getattr(typ, "__name__", str(typ))
        raise SemanticUnsolvable(
            f"field '{key}' must be {name}", position=pos)
    return val


def parse_bundle(obj: dict, *, max_points: int = MAX_POINTS_DEFAULT,
                 budget: dict | None = None):
    if obj.get("schema") != BUNDLE_SCHEMA:
        raise FormatCorruptError(
            f"unsupported/missing schema; expected {BUNDLE_SCHEMA!r}, "
            f"got {obj.get('schema')!r}", position="$.schema")
    project = _require(obj, "project", dict, "$.project")
    name = _require(project, "name", str, "$.project.name")

    diags = []
    out: list[M.Material] = []

    def add(kind, raw, data, role="sample", version=1, replaces=None,
            extra_diags=()):
        out.append(M.make_material(kind, raw, data,
                                   tuple(diags) + tuple(extra_diags),
                                   role=role, version=version,
                                   replaces=replaces))

    # ---- power series (sample) ----
    ps = _require(obj, "power_series", dict, "$.power_series")
    pdata, pdiags = _parse_power(ps, "$.power_series", max_points, budget)
    diags.extend(pdiags)
    add("power_series", ps, pdata)

    # ---- blanks: separate, versioned, never merged into the sample ----
    blank_corr = None
    for i, b in enumerate(obj.get("blank_series", [])):
        if not isinstance(b, dict):
            raise SemanticUnsolvable("blank entry must be an object",
                                     position=f"$.blank_series[{i}]")
        role = b.get("label", f"blank{i + 1}")
        bdata, bdiags = _parse_power(b, f"$.blank_series[{i}]",
                                     max_points, budget)
        add("blank_series", b, bdata, role=str(role),
            extra_diags=bdiags)
        if b.get("corresponds_to") is not None:
            blank_corr = b.get("corresponds_to")

    # ---- injection schedule ----
    sch = _require(obj, "injections", dict, "$.injections")
    sdata, sdiags = _parse_injections(sch, "$.injections")
    diags.extend(sdiags)
    add("injection_schedule", sch, sdata)

    # ---- concentrations ----
    con = _require(obj, "concentrations", dict, "$.concentrations")
    cdata, cdiags = _parse_concentrations(con, "$.concentrations")
    diags.extend(cdiags)
    add("concentrations", con, cdata)

    # ---- conditions: temperature + cell volume ----
    cond = _require(obj, "conditions", dict, "$.conditions")
    kdata, kdiags = _parse_conditions(cond, "$.conditions")
    diags.extend(kdiags)
    add("conditions", cond, kdata)

    # ---- baseline (separate versioned material) ----
    base = _require(obj, "baseline", dict, "$.baseline")
    bdata, bdiags = _parse_baseline(base, "$.baseline", pdata)
    diags.extend(bdiags)
    add("baseline", base, bdata)

    rules = obj.get("review", {})
    if not isinstance(rules, dict):
        raise SemanticUnsolvable("review must be an object",
                                 position="$.review")

    return {
        "project": {"name": name,
                    "external_id": project.get("id", name)},
        "materials": out,
        "review_rules": {
            "first_injection": rules.get("first_injection", "discard"),
            "blank_correspondence": rules.get("blank_correspondence",
                                              blank_corr or "auto"),
            "integration": rules.get("integration", {}),
        },
        "diagnostics": diags,
    }


def _check_budget(n: int, limit: int, where: str):
    if n > limit:
        raise BudgetExhausted(
            f"sample count {n} exceeds budget {limit}",
            position=where, detail={"count": n, "budget": limit})


def _parse_power(ps: dict, pos: str, max_points: int,
                 budget: dict | None):
    samples = _require(ps, "samples", list, f"{pos}.samples")
    time_unit = ps.get("time_unit", "s")
    power_unit = _require(ps, "power_unit", str, f"{pos}.power_unit")
    limit = (budget or {}).get("max_points", max_points)
    _check_budget(len(samples), limit, f"{pos}.samples")

    rows, gaps_declared, diags = [], [], []
    t_prev = None
    for i, s in enumerate(samples):
        if not isinstance(s, dict):
            raise FormatCorruptError("each sample must be an object",
                                     position=f"{pos}.samples[{i}]")
        traw, praw = s.get("t"), s.get("p")
        tq = quantity(traw, time_unit, "time")
        pq = quantity(praw, power_unit, "power")
        if not tq.ok:
            diags.append(diag("bad_time", f"sample time {traw!r} cannot be "
                              f"converted ({tq.error})",
                              position=f"{pos}.samples[{i}].t",
                              subject=f"sample[{i}]"))
        if not pq.ok:
            diags.append(diag("bad_power", f"sample power {praw!r} cannot be "
                              f"converted ({pq.error})",
                              position=f"{pos}.samples[{i}].p",
                              subject=f"sample[{i}]"))
        rows.append({"i": i, "t_raw": "" if traw is None else str(traw),
                     "p_raw": "" if praw is None else str(praw),
                     "t": tq.value, "p": pq.value,
                     "t_unit": "s", "p_unit": "W"})
        if tq.ok and t_prev is not None and tq.value <= t_prev:
            raise SemanticUnsolvable(
                f"time must be strictly increasing: t[{i-1}]={t_prev} "
                f">= t[{i}]={tq.value}",
                position=f"{pos}.samples[{i}].t")
        if tq.ok:
            t_prev = tq.value

    for j, g in enumerate(ps.get("gaps", [])):
        if not isinstance(g, dict) or "from_t" not in g or "to_t" not in g:
            raise SemanticUnsolvable(
                "gap needs from_t and to_t", position=f"{pos}.gaps[{j}]")
        fq = quantity(g["from_t"], time_unit, "time")
        tq = quantity(g["to_t"], time_unit, "time")
        if not fq.ok or not tq.ok or fq.value >= tq.value:
            raise SemanticUnsolvable(
                "gap from_t/to_t invalid", position=f"{pos}.gaps[{j}]")
        gaps_declared.append({"from_t": fq.value, "to_t": tq.value,
                              "reason": g.get("reason", "unspecified")})

    data = {"rows": rows, "time_unit": "s", "power_unit": "W",
            "time_unit_raw": time_unit, "power_unit_raw": power_unit,
            "gaps": gaps_declared, "label": ps.get("label", "sample")}
    return data, diags


def _parse_injections(sch: dict, pos: str):
    items = _require(sch, "list", list, f"{pos}.list")
    time_unit = sch.get("time_unit", "s")
    vol_unit = sch.get("volume_unit", "uL")
    out, diags = [], []
    t_prev = -math.inf
    for i, inj in enumerate(items):
        if not isinstance(inj, dict):
            raise FormatCorruptError("injection must be an object",
                                     position=f"{pos}.list[{i}]")
        tq = quantity(inj.get("t"), time_unit, "time")
        vq = quantity(inj.get("volume"), vol_unit, "volume")
        if not tq.ok:
            diags.append(diag("bad_inj_time", tq.error,
                              position=f"{pos}.list[{i}].t",
                              subject=f"injection[{i}]"))
        elif tq.value < t_prev:
            raise SemanticUnsolvable(
                "injections must be ordered by time",
                position=f"{pos}.list[{i}].t")
        else:
            t_prev = tq.value
        if not vq.ok or vq.value <= 0:
            diags.append(diag("bad_inj_volume",
                              f"injection volume invalid: {vq.error}",
                              position=f"{pos}.list[{i}].volume",
                              subject=f"injection[{i}]"))
        out.append({"index": i,
                    "t_raw": str(inj.get("t", "")),
                    "volume_raw": str(inj.get("volume", "")),
                    "t": tq.value, "volume": vq.value,
                    "t_unit": "s", "volume_unit": "L",
                    "label": inj.get("label", f"I{i+1}")})
    if not out:
        raise SemanticUnsolvable("at least one injection is required",
                                 position=f"{pos}.list")
    return {"items": out, "time_unit_raw": time_unit,
            "volume_unit_raw": vol_unit}, diags


def _parse_concentrations(con: dict, pos: str):
    syr_unit = con.get("syringe_unit", "mM")
    cell_present = "cell" in con
    syr = quantity(con.get("syringe"), syr_unit, "concentration")
    cell = quantity(con.get("cell", 0), con.get("cell_unit", syr_unit),
                    "concentration")
    diags = []
    if not syr.ok or syr.value is None or syr.value <= 0:
        diags.append(diag("bad_syringe_conc",
                          f"syringe concentration invalid: {syr.error}",
                          position=f"{pos}.syringe"))
    if cell_present and not cell.ok:
        diags.append(diag("bad_cell_conc", f"cell concentration invalid: "
                          f"{cell.error}", position=f"{pos}.cell"))
    return {"syringe": syr.value, "cell": cell.value,
            "syringe_raw": syr.raw, "cell_raw": cell.raw,
            "syringe_unit_raw": syr_unit,
            "cell_unit_raw": con.get("cell_unit", syr_unit),
            "unit": "M", "cell_present": cell_present}, diags


def _parse_conditions(cond: dict, pos: str):
    tq = quantity(cond.get("temperature"), cond.get("temperature_unit", "C"),
                  "temperature")
    vq = quantity(cond.get("cell_volume"),
                  cond.get("cell_volume_unit", "mL"), "volume")
    diags = []
    if not tq.ok:
        diags.append(diag("bad_temperature", tq.error,
                          position=f"{pos}.temperature"))
    if not vq.ok or vq.value <= 0:
        diags.append(diag("bad_cell_volume", vq.error,
                          position=f"{pos}.cell_volume"))
    return {"temperature": tq.value, "cell_volume": vq.value,
            "temperature_raw": tq.raw, "temperature_unit_raw": tq.raw_unit,
            "cell_volume_raw": vq.raw,
            "cell_volume_unit_raw": vq.raw_unit,
            "temperature_unit": "K", "cell_volume_unit": "L"}, diags


def _parse_baseline(base: dict, pos: str, power_data: dict):
    kind = _require(base, "kind", str, f"{pos}.kind")
    diags = []
    data = {"kind": kind, "unit": "W"}
    if kind == "constant":
        vq = quantity(_require(base, "value", (str, int, float),
                               f"{pos}.value"),
                      base.get("power_unit",
                               power_data["power_unit_raw"]), "power")
        if not vq.ok:
            diags.append(diag("bad_baseline", vq.error,
                              position=f"{pos}.value"))
        data["value"] = vq.value
        data["value_raw"] = vq.raw
        data["power_unit_raw"] = vq.raw_unit
    elif kind == "anchored_linear":
        anchors = _require(base, "anchors", list, f"{pos}.anchors")
        rows = []
        pu = base.get("power_unit", power_data["power_unit_raw"])
        for k, a in enumerate(anchors):
            tq = quantity(a.get("t"), base.get("time_unit", "s"), "time")
            pq = quantity(a.get("p"), pu, "power")
            if not tq.ok or not pq.ok:
                diags.append(diag("bad_baseline_anchor",
                                  f"anchor t/p invalid",
                                  position=f"{pos}.anchors[{k}]"))
            rows.append({"t": tq.value, "p": pq.value})
        if len(rows) < 2:
            raise SemanticUnsolvable(
                "anchored_linear baseline needs >= 2 anchors",
                position=f"{pos}.anchors")
        data["anchors"] = rows
    elif kind == "series":
        series = _require(base, "samples", list, f"{pos}.samples")
        rows = []
        pu = base.get("power_unit", power_data["power_unit_raw"])
        for k, s in enumerate(series):
            tq = quantity(s.get("t"), base.get("time_unit", "s"), "time")
            pq = quantity(s.get("p"), pu, "power")
            if not tq.ok or not pq.ok:
                diags.append(diag("bad_baseline_point",
                                  "baseline point t/p invalid",
                                  position=f"{pos}.samples[{k}]"))
            rows.append({"t": tq.value, "p": pq.value})
        data["samples"] = rows
    else:
        raise SemanticUnsolvable(f"unknown baseline kind {kind!r}",
                                 position=f"{pos}.kind")
    return data, diags
