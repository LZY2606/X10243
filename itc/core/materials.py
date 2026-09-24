"""把分槽版本化的原始材料解析为规范化视图。

原则：原始表示永远留在材料载荷里；本层只在派生计算时规范化。
任何样本都不会因为无法转换而被删除，它以 mask=False 留在原坐标，
并形成显式缺口区间（gap interval），供积分窗判定。
"""
from __future__ import annotations

from typing import Any

from .errors import Diagnostic, Location
from .units import parse_quantity


def _curve_view(payload: dict[str, Any], slot: str) -> dict[str, Any]:
    samples = payload.get("samples", [])
    tu = payload.get("time_unit", "s")
    pu = payload.get("power_unit", "ucal/s")
    n = len(samples)
    t: list[float | None] = [None] * n
    p: list[float | None] = [None] * n
    valid_t = [False] * n
    valid_p = [False] * n
    diags: list[dict[str, Any]] = []
    declared_gap = [False] * n
    for i, s in enumerate(samples):
        qt = parse_quantity(s.get("time_raw"), "time", tu,
                            Location(slot=slot, row=i, field="time"))
        qp = parse_quantity(s.get("power_raw"), "power", pu,
                            Location(slot=slot, row=i, field="power"))
        if qt.value is not None:
            t[i], valid_t[i] = qt.value, True
        if qp.value is not None:
            p[i], valid_p[i] = qp.value, True
        declared_gap[i] = bool(s.get("gap"))
        for d in (*qt.diagnostics, *qp.diagnostics):
            diags.append(d.to_dict())
    missing = [i for i in range(n) if not valid_p[i] or declared_gap[i]]
    intervals = _runs(missing)
    return {
        "n": n,
        "t": t,
        "p": p,
        "valid_t": valid_t,
        "valid_p": valid_p,
        "missing_rows": missing,
        "gap_intervals": intervals,
        "gap_time_intervals": [(t[a], t[b]) for a, b in intervals
                               if t[a] is not None and t[b] is not None],
        "declared_gap": declared_gap,
        "time_unit": "s",
        "power_unit": "ucal/s",
        "time_base": payload.get("time_base",
                                 "seconds_from_acquisition_start"),
        "diagnostics": diags,
        "raw_ref": payload.get("raw"),
    }


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    out: list[tuple[int, int]] = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append((start, prev))
        start = prev = i
    out.append((start, prev))
    return out


def power_view(material: dict[str, Any]) -> dict[str, Any]:
    return _curve_view(material["payload"], "power")


def baseline_view(material: dict[str, Any]) -> dict[str, Any]:
    payload = material["payload"]
    if payload.get("format") == "polynomial":
        return {"model": "polynomial", "coeff": payload.get("coeff_raw", []),
                "unit": payload.get("power_unit", "ucal/s")}
    if payload.get("format") == "constant":
        return {"model": "constant",
                "value_raw": payload.get("value_raw", 0),
                "unit": payload.get("power_unit", "ucal/s")}
    return _curve_view(payload, "baseline")


def injections_view(material: dict[str, Any]) -> list[dict[str, Any]]:
    payload = material["payload"]
    vu = payload.get("volume_unit", "uL")
    tu = payload.get("time_unit", "s")
    out = []
    for r in payload.get("injections", []):
        qv = parse_quantity(r["volume_raw"], "volume", vu,
                            Location(slot="injections", row=r["ordinal"] - 1,
                                     field="volume"))
        qt = parse_quantity(r["time_raw"], "time", tu,
                            Location(slot="injections", row=r["ordinal"] - 1,
                                     field="time"))
        out.append({
            "ordinal": r["ordinal"],
            "label": r.get("label", f"I{r['ordinal']}"),
            "t": qt.value,
            "volume": qv.value,
            "t_raw": r["time_raw"],
            "volume_raw": r["volume_raw"],
            "diagnostics": [d.to_dict() for d in
                            (*qv.diagnostics, *qt.diagnostics)],
        })
    return out


def scalar_view(material: dict[str, Any], dimension: str, slot: str,
                field: str = "value_raw") -> dict[str, Any]:
    payload = material["payload"]
    unit = payload.get("unit")
    q = parse_quantity(payload.get(field), dimension, unit,
                       Location(slot=slot, field=dimension))
    return {"value": q.value, "raw": q.raw, "unit": q.unit,
            "canonical_unit": q.canonical_unit,
            "diagnostics": [d.to_dict() for d in q.diagnostics]}


def concentration_view(material: dict[str, Any], which: str) -> dict[str, Any]:
    payload = material["payload"]
    entry = payload[which]
    q = parse_quantity(entry.get("raw"), "concentration",
                       entry.get("unit"),
                       Location(slot="concentrations", field=which))
    return {"value": q.value, "raw": q.raw, "unit": q.unit,
            "canonical_unit": q.canonical_unit,
            "diagnostics": [d.to_dict() for d in q.diagnostics]}


def blank_view(material: dict[str, Any]) -> dict[str, Any]:
    payload = material["payload"]
    curve = _curve_view(payload["curve"], "blank")
    inj_payload = {
        "payload": {
            "volume_unit": payload.get("volume_unit", "uL"),
            "time_unit": payload.get("time_unit", "s"),
            "injections": payload.get("injections", []),
        }
    }
    curve["injections"] = injections_view(inj_payload)
    curve["source_kind"] = payload.get("source_kind", "blank_run")
    return curve


def first_diagnostic(views: list[dict[str, Any]]
                     ) -> Diagnostic | None:
    for v in views:
        for d in v.get("diagnostics", []):
            return d
    return None
