"""Immutable domain materials.

Roles are versioned independently:
    power / injections / syringe_concentration / cell_concentration /
    temperature / cell_volume / baseline / blank_power / blank_injections

Parsing preserves the *original representation*. Normalisation happens only in
the derived layer. Values that cannot be converted remain in ``raw`` with a
positioned diagnostic instead of being replaced by a default.
"""
from __future__ import annotations

from typing import Any

from .canonical import content_id
from .errors import Diagnostic
from .units import convert_scalar, convert_temperature

ROLES = (
    "power",
    "injections",
    "syringe_concentration",
    "cell_concentration",
    "temperature",
    "cell_volume",
    "baseline",
    "blank_power",
    "blank_injections",
)
OPTIONAL_ROLES = ("blank_power", "blank_injections", "baseline")

SCALAR_ROLES = {
    "syringe_concentration": ("concentration", "M"),
    "cell_concentration": ("concentration", "M"),
    "temperature": ("temperature", "K"),
    "cell_volume": ("volume", "L"),
}

ROLE_PREFIX = {
    "power": "v_pwr",
    "injections": "v_inj",
    "syringe_concentration": "v_conc_syr",
    "cell_concentration": "v_conc_cell",
    "temperature": "v_temp",
    "cell_volume": "vcell",
    "baseline": "v_base",
    "blank_power": "v_bpwr",
    "blank_injections": "v_binj",
}


def _samples(raw: Any):
    """Accept [{t,p}...] or {time:[...], power:[...]} -> list[(t_raw,p_raw,i)]."""
    out: list[tuple[Any, Any, int]] = []
    if isinstance(raw, dict):
        ts = raw.get("time") or raw.get("t") or []
        ps = raw.get("power") or raw.get("p") or []
        if not isinstance(ts, list) or not isinstance(ps, list):
            return None
        if len(ts) != len(ps):
            return None
        for i, (t, p) in enumerate(zip(ts, ps)):
            out.append((t, p, i))
        return out
    if isinstance(raw, list):
        for i, pt in enumerate(raw):
            if not isinstance(pt, dict):
                return None
            t = pt.get("t", pt.get("time"))
            p = pt.get("p", pt.get("power"))
            out.append((t, p, i))
        return out
    return None


def parse_series(role: str, node: Any, time_unit: str | None, power_unit: str | None,
                 time_kind: str = "time", power_kind: str = "power"):
    """Parse a power/time series preserving raw points.

    Returns dict with raw + normalized arrays. Unconvertible points stay in
    raw and are omitted from normalized arrays with positioned diagnostics.
    """
    diags: list[dict[str, Any]] = []
    samples = _samples(node.get("samples", node))
    if samples is None:
        raise _format(node, f"{role}: samples must be a list of {{t,p}} points "
                            "or an object with equal-length time/power arrays")
    raw_samples = [{"i": i, "t": t, "p": p} for (t, p, i) in samples]
    tn, pn = [], []
    for (t, p, i) in samples:
        tv, _, d1 = convert_scalar(t, node.get("time_unit", time_unit), time_kind,
                                   f"{role}.samples[{i}].t")
        pv, cu, d2 = convert_scalar(p, node.get("power_unit", power_unit), power_kind,
                                    f"{role}.samples[{i}].p")
        diags.extend(d1)
        diags.extend(d2)
        if tv is not None and pv is not None:
            tn.append(tv)
            pn.append(pv)
    if not tn:
        diags.append(Diagnostic("error", "NO_USABLE_SAMPLES",
                                f"{role}: no convertible samples remain",
                                position=f"{role}.samples").to_dict())
    raw = {
        "samples": raw_samples,
        "time_unit": node.get("time_unit", time_unit),
        "power_unit": node.get("power_unit", power_unit),
        "time_basis": node.get("time_basis", "elapsed seconds from first sample"),
    }
    normalized = {
        "t": tn,
        "p": pn,
        "t_unit": "s",
        "p_unit": "ucal/s",
        "time_basis": raw["time_basis"],
    }
    version_id = content_id(ROLE_PREFIX[role], raw)
    return {"role": role, "version_id": version_id, "raw": raw,
            "normalized": normalized, "diagnostics": diags}


def parse_baseline(node: Any):
    """Baseline is its own versioned source: constant or sampled curve."""
    diags: list[dict[str, Any]] = []
    mode = node.get("mode", "constant" if "value" in node else "points")
    raw = {"mode": mode}
    normalized: dict[str, Any] = {"mode": mode}
    if mode == "constant":
        v, cu, d = convert_scalar(node.get("value"), node.get("power_unit"), "power",
                                  "baseline.value")
        diags.extend(d)
        raw["value"] = node.get("value")
        raw["power_unit"] = node.get("power_unit")
        normalized["value"] = v
        normalized["p_unit"] = cu or "ucal/s"
    elif mode == "points":
        series = parse_series("baseline", node, None, None)
        raw["samples"] = series["raw"]["samples"]
        raw["time_unit"] = series["raw"]["time_unit"]
        raw["power_unit"] = series["raw"]["power_unit"]
        raw["time_basis"] = series["raw"]["time_basis"]
        normalized.update({"t": series["normalized"]["t"],
                           "p": series["normalized"]["p"],
                           "p_unit": "ucal/s", "t_unit": "s",
                           "time_basis": series["normalized"]["time_basis"]})
        diags.extend(series["diagnostics"])
    else:
        raise _format(node, f"baseline.mode {mode!r} must be 'constant' or 'points'")
    version_id = content_id(ROLE_PREFIX["baseline"], raw)
    return {"role": "baseline", "version_id": version_id, "raw": raw,
            "normalized": normalized, "diagnostics": diags}


def parse_injections(role: str, node: Any):
    diags: list[dict[str, Any]] = []
    items = node.get("injections")
    if not isinstance(items, list) or not items:
        raise _format(node, f"{role}: non-empty injections list required")
    t_unit = node.get("time_unit")
    v_unit = node.get("volume_unit")
    raw_rows = []
    norm_rows = []
    seen_ordinal = set()
    for i, row in enumerate(items):
        if not isinstance(row, dict):
            raise _format(row, f"{role}.injections[{i}] must be an object")
        ordinal = row.get("ordinal", i)
        if ordinal in seen_ordinal:
            diags.append(Diagnostic("error", "DUPLICATE_ORDINAL",
                                    "duplicate injection ordinal",
                                    position=f"{role}.injections[{i}].ordinal",
                                    value=ordinal).to_dict())
        seen_ordinal.add(ordinal)
        tv, _, d1 = convert_scalar(row.get("time", row.get("t")),
                                   row.get("time_unit", t_unit), "time",
                                   f"{role}.injections[{i}].time")
        vv, vu, d2 = convert_scalar(row.get("volume", row.get("v")),
                                    row.get("volume_unit", v_unit), "volume",
                                    f"{role}.injections[{i}].volume")
        diags.extend(d1)
        diags.extend(d2)
        raw_rows.append({"ordinal": ordinal,
                         "time": row.get("time", row.get("t")),
                         "volume": row.get("volume", row.get("v")),
                         "time_unit": row.get("time_unit", t_unit),
                         "volume_unit": row.get("volume_unit", v_unit)})
        if tv is None or vv is None:
            continue
        norm_rows.append({"ordinal": ordinal, "time_s": tv, "volume_l": vv})
    norm_rows.sort(key=lambda r: r["ordinal"])
    raw = {"injections": raw_rows, "time_unit": t_unit, "volume_unit": v_unit,
           "time_basis": node.get("time_basis", "elapsed seconds from first sample")}
    normalized = {"injections": norm_rows, "t_unit": "s", "v_unit": "L",
                  "time_basis": raw["time_basis"]}
    version_id = content_id(ROLE_PREFIX[role], raw)
    return {"role": role, "version_id": version_id, "raw": raw,
            "normalized": normalized, "diagnostics": diags}


def parse_scalar(role: str, node: Any):
    kind, _canon = SCALAR_ROLES[role]
    if isinstance(node, dict) and "value" in node:
        value, unit = node["value"], node.get("unit")
        raw = {"value": value, "unit": unit,
               "basis": node.get("basis", "absolute")}
    else:
        value, unit = node, None
        raw = {"value": value, "unit": unit, "basis": "absolute"}
    if kind == "temperature":
        nv, cu, diags = convert_temperature(value, unit, f"{role}.value")
    else:
        nv, cu, diags = convert_scalar(value, unit, kind, f"{role}.value")
    normalized = {"value": nv, "unit": cu, "basis": raw["basis"]}
    version_id = content_id(ROLE_PREFIX[role], raw)
    return {"role": role, "version_id": version_id, "raw": raw,
            "normalized": normalized, "diagnostics": diags}


def _format(node: Any, msg: str):
    # local import to avoid cycle
    from .errors import FormatCorrupted
    return FormatCorrupted(msg, position="<bundle>")


def parse_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Parse an import-bundle dict into versioned role materials."""
    if not isinstance(bundle, dict) or "materials" not in bundle:
        from .errors import FormatCorrupted
        raise FormatCorrupted("root must be an object with a 'materials' object",
                              position="<root>")
    mats_in = bundle["materials"]
    if not isinstance(mats_in, dict):
        from .errors import FormatCorrupted
        raise FormatCorrupted("'materials' must be an object", position="materials")
    materials: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    missing: list[str] = []

    for role in ROLES:
        node = mats_in.get(role)
        if node is None:
            if role not in OPTIONAL_ROLES:
                missing.append(role)
                diagnostics.append(Diagnostic("error", "MISSING_SOURCE",
                                              f"required source {role!r} is absent",
                                              position=f"materials.{role}").to_dict())
            else:
                diagnostics.append(Diagnostic("info", "OPTIONAL_SOURCE_ABSENT",
                                              f"optional source {role!r} is absent",
                                              position=f"materials.{role}").to_dict())
            continue
        try:
            if role in ("power", "blank_power"):
                mat = parse_series(role, node,
                                   mats_in.get("power", {}).get("time_unit")
                                   if role == "blank_power" else None,
                                   mats_in.get("power", {}).get("power_unit")
                                   if role == "blank_power" else None)
            elif role in ("injections", "blank_injections"):
                mat = parse_injections(role, node)
            elif role == "baseline":
                mat = parse_baseline(node)
            else:
                mat = parse_scalar(role, node)
        except _FormatErrorShim:  # pragma: no cover - safety
            raise
        materials[role] = mat
        diagnostics.extend(mat.get("diagnostics", []))

    parsed = {
        "project_id": bundle.get("project_id", "local-project"),
        "project_name": bundle.get("project_name", bundle.get("project_id", "local project")),
        "bundle_format": bundle.get("bundle_format", "itc-bundle/1"),
        "materials": materials,
        "missing_roles": missing,
        "diagnostics": diagnostics,
    }
    return parsed


class _FormatErrorShim(Exception):  # pragma: no cover
    pass
