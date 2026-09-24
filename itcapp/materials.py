"""Typed, separately-versioned material types.

Seven kinds of immutable material are versioned independently:
power sampling, injection schedule, concentration, temperature, pool
volume, baseline and blank curve. Each parsed value keeps its raw
representation; conversion to canonical units happens here in the
derived layer and unconvertible values stay in place with diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from .errors import Diagnostic, SemanticUnsolvableError
from .itcparse import ItcDocument, parse_itc, require_sections
from .units import Quantity, convert

KINDS = ("power", "injections", "concentration", "temperature",
         "pool", "baseline", "blank")


def _q(doc: ItcDocument, sec: str, row: int, col: str, kind: str
       ) -> Tuple[Optional[Quantity], Optional[Diagnostic]]:
    table = doc.sections[sec]
    if col not in table.columns:
        return None, Diagnostic("MISSING_COLUMN", "error",
                                f"{sec}#header:{col}",
                                f"section @{sec} lacks column {col!r}")
    raw = table.rows[row][table.columns.index(col)]
    unit_col = col + "_unit"
    if unit_col in table.columns:
        unit = table.rows[row][table.columns.index(unit_col)]
    else:
        unit = _default_unit(col)
    location = f"@{sec}#L{row + 1}:{col}"
    q = convert(raw, unit, kind, location)
    return q, q.diagnostic


def _default_unit(col: str) -> str:
    return {
        "t": "s", "time": "s", "time_s": "s",
        "p": "uJ/s", "power": "uJ/s", "power_uJ_s": "uJ/s",
        "volume": "uL",
        "syringe_concentration": "mM", "cell_concentration": "mM",
        "temperature": "K",
        "cell_volume": "uL",
        "b": "uJ/s", "baseline_power": "uJ/s",
    }.get(col, "")


@dataclass
class Material:
    id: str
    kind: str
    name: str
    version: int
    replaces: Optional[str]  # id of prior material version
    source: str              # original filename
    raw_text: str            # immutable raw representation
    parameters: Dict[str, str] = field(default_factory=dict)
    derived: dict = field(default_factory=dict)
    diagnostics: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_material(raw_text: str, source: str, material_id: str,
                  version: int = 1, replaces: Optional[str] = None) -> Material:
    doc = parse_itc(raw_text, source)
    kind = doc.header.get("material")
    name = doc.header.get("name", source)
    if kind not in KINDS:
        raise SemanticUnsolvableError(
            f"unknown material kind {kind!r}",
            [Diagnostic("UNKNOWN_MATERIAL", "error", f"{source}:1",
                        f"material must be one of {KINDS}")])
    builder = {
        "power": _build_power,
        "injections": _build_injections,
        "concentration": _build_concentration,
        "temperature": _build_temperature,
        "pool": _build_pool,
        "baseline": _build_baseline,
        "blank": _build_blank,
    }[kind]
    derived, diagnostics = builder(doc, source)
    return Material(
        id=material_id, kind=kind, name=name, version=version,
        replaces=replaces, source=source, raw_text=raw_text,
        parameters=dict(doc.header), derived=derived,
        diagnostics=[d.to_dict() for d in diagnostics])


def _samples(doc: ItcDocument, sec: str, src: str, power_col: str
             ) -> Tuple[list, List[Diagnostic]]:
    require_sections(doc, [sec], src)
    table = doc.sections[sec]
    diagnostics: List[Diagnostic] = []
    samples = []
    tcol = "t" if "t" in table.columns else (
        "time" if "time" in columns_of(table) else None)
    if tcol is None:
        raise SemanticUnsolvableError(
            f"@{sec} needs a time column t or time",
            [Diagnostic("MISSING_COLUMN", "error", f"{src}@{sec}:header",
                        "time column 't' missing")])
    for i in range(len(table.rows)):
        tq, td = _q(doc, sec, i, tcol, "time")
        pq, pd = _q(doc, sec, i, power_col, "power")
        for d in (td, pd):
            if d:
                diagnostics.append(d)
        samples.append({
            "t_raw": tq.raw, "t_unit": tq.unit, "t": tq.value,
            "p_raw": pq.raw, "p_unit": pq.unit, "p": pq.value,
            "row": i + 1,
        })
    if not samples:
        raise SemanticUnsolvableError(
            f"@{sec} has no samples",
            [Diagnostic("EMPTY_SERIES", "error", f"{src}@{sec}",
                        "no sample rows")])
    # time must strictly increase and be fully numeric
    prev = None
    for s in samples:
        if s["t"] is None or s["p"] is None:
            continue  # diagnostics already attached
        if prev is not None and s["t"] <= prev:
            raise SemanticUnsolvableError(
                "sample times must be strictly increasing",
                [Diagnostic("TIME_NOT_INCREASING", "error",
                            f"{src}@{sec}#L{s['row']}:t",
                            f"t={s['t_raw']} is not after previous time")])
        prev = s["t"]
    return samples, diagnostics


def columns_of(table):
    return table.columns


def _build_power(doc, src):
    samples, diags = _samples(doc, "power", src, "p")
    gaps = _find_gaps(samples, doc.sections["power"], src)
    return {"samples": samples, "gaps": gaps}, diags


def _find_gaps(samples, table, src):
    """A gap is either an explicit gap marker row or a time jump.

    Marker rows use p_raw == 'GAP' and carry only a numeric t; jump rows
    are flagged in the header parameter gap_threshold (canonical seconds).
    """
    gaps = []
    numeric_ts = [s["t"] for s in samples if s["t"] is not None]
    diffs = [numeric_ts[i + 1] - numeric_ts[i]
             for i in range(len(numeric_ts) - 1)]
    median = sorted(diffs)[len(diffs) // 2] if diffs else 1.0
    threshold_raw = doc.header_param(table)
    for s in samples:
        if s["p_raw"].upper() == "GAP":
            gaps.append({"at_row": s["row"], "at_t": s["t"],
                         "kind": "marker", "explicit": True})
    threshold = median * 5.0
    for i in range(len(samples) - 1):
        a, b = samples[i], samples[i + 1]
        if a["t"] is None or b["t"] is None:
            continue
        if b["t"] - a["t"] > threshold and not (
                b["p_raw"].upper() == "GAP" or a["p_raw"].upper() == "GAP"):
            gaps.append({"after_row": a["row"], "before_row": b["row"],
                         "t0": a["t"], "t1": b["t"],
                         "kind": "jump", "explicit": False})
    return gaps


def doc_header_param(table):
    return None


def _build_injections(doc, src):
    require_sections(doc, ["injections"], src)
    table = doc.sections["injections"]
    diags: List[Diagnostic] = []
    inj = []
    for i, _row in enumerate(table.rows):
        tq, td = _q(doc, "injections", i, "time", "time")
        vq, vd = _q(doc, "injections", i, "volume", "volume")
        for d in (td, vd):
            if d:
                diags.append(d)
        inj.append({
            "index": i + 1,
            "time_raw": tq.raw, "time_unit": tq.unit, "time": tq.value,
            "volume_raw": vq.raw, "volume_unit": vq.unit, "volume": vq.value,
            "row": i + 1,
        })
    if not inj:
        raise SemanticUnsolvableError(
            "no injections declared",
            [Diagnostic("EMPTY_SCHEDULE", "error", f"{src}@injections",
                        "at least one injection required")])
    prev = None
    for e in inj:
        if e["time"] is None or e["volume"] is None:
            continue
        if prev is not None and e["time"] <= prev:
            raise SemanticUnsolvableError(
                "injection times must be strictly increasing",
                [Diagnostic("INJECTION_ORDER", "error",
                            f"{src}@injections#L{e['row']}:time",
                            f"injection {e['index']} not after previous")])
        if e["volume"] <= 0:
            diags.append(Diagnostic("NONPOSITIVE_VOLUME", "error",
                                    f"{src}@injections#L{e['row']}:volume",
                                    f"injection {e['index']} volume "
                                    f"{e['volume_raw']} is non-positive"))
        prev = e["time"]
    return {"injections": inj}, diags


def _build_concentration(doc, src):
    require_sections(doc, ["concentration"], src)
    table = doc.sections["concentration"]
    diags: List[Diagnostic] = []
    rows = []
    for i, _ in enumerate(table.rows):
        out = {"row": i + 1}
        for col, key in (("syringe_concentration", "syringe"),
                         ("cell_concentration", "cell")):
            if col in table.columns:
                q, d = _q(doc, "concentration", i, col, "concentration")
                out[key + "_raw"] = q.raw
                out[key + "_unit"] = q.unit
                out[key] = q.value
                if d:
                    diags.append(d)
        rows.append(out)
    if len(rows) != 1:
        raise SemanticUnsolvableError(
            "concentration material needs exactly one row",
            [Diagnostic("BAD_CONCENTRATION", "error",
                        f"{src}@concentration",
                        f"expected 1 row, got {len(rows)}")])
    row = rows[0]
    for key in ("syringe", "cell"):
        if row.get(key) is not None and row[key] < 0:
            diags.append(Diagnostic("NEGATIVE_CONCENTRATION", "error",
                                    f"{src}@concentration#L1:{key}",
                                    f"{key} concentration is negative; "
                                    "semantically invalid"))
    return {"concentration": row}, diags


def _build_temperature(doc, src):
    require_sections(doc, ["conditions"], src)
    table = doc.sections["conditions"]
    diags: List[Diagnostic] = []
    if "temperature" not in table.columns:
        raise SemanticUnsolvableError(
            "conditions needs temperature column",
            [Diagnostic("MISSING_COLUMN", "error",
                        f"{src}@conditions:header", "temperature missing")])
    if len(table.rows) != 1:
        raise SemanticUnsolvableError(
            "temperature material needs exactly one condition row",
            [Diagnostic("BAD_CONDITIONS", "error", f"{src}@conditions",
                        f"expected 1 row, got {len(table.rows)}")])
    q, d = _q(doc, "conditions", 0, "temperature", "temperature")
    if d:
        diags.append(d)
    val = q.value
    if val is not None and not (200.0 <= val <= 400.0):
        diags.append(Diagnostic("TEMPERATURE_RANGE", "error",
                                f"{src}@conditions#L1:temperature",
                                f"temperature {q.raw} {q.unit} -> "
                                f"{val} K is outside 200..400 K"))
    return {"temperature": {"raw": q.raw, "unit": q.unit, "kelvin": val}}, diags


def _build_pool(doc, src):
    require_sections(doc, ["cell"], src)
    table = doc.sections["cell"]
    diags: List[Diagnostic] = []
    if "cell_volume" not in table.columns:
        raise SemanticUnsolvableError(
            "cell needs cell_volume column",
            [Diagnostic("MISSING_COLUMN", "error",
                        f"{src}@cell:header", "cell_volume missing")])
    if len(table.rows) != 1:
        raise SemanticUnsolvableError(
            "pool material needs exactly one row",
            [Diagnostic("BAD_CELL", "error", f"{src}@cell",
                        f"expected 1 row, got {len(table.rows)}")])
    q, d = _q(doc, "cell", 0, "cell_volume", "volume")
    if d:
        diags.append(d)
    if q.value is not None and q.value <= 0:
        diags.append(Diagnostic("NONPOSITIVE_VOLUME", "error",
                                f"{src}@cell#L1:cell_volume",
                                f"cell volume {q.raw} {q.unit} non-positive"))
    return {"cell_volume": {"raw": q.raw, "unit": q.unit, "uL": q.value}}, diags


def _anchor_series(doc, src, sec, col):
    require_sections(doc, [sec], src)
    table = doc.sections[sec]
    diags: List[Diagnostic] = []
    pts = []
    for i, _ in enumerate(table.rows):
        tq, td = _q(doc, sec, i, "t", "time")
        bq, bd = _q(doc, sec, i, col, "power")
        for d in (td, bd):
            if d:
                diags.append(d)
        pts.append({"t": tq.value, "p": bq.value, "t_raw": tq.raw,
                    "p_raw": bq.raw, "row": i + 1})
    if len(pts) < 2:
        raise SemanticUnsolvableError(
            f"@{sec} needs at least 2 anchor points",
            [Diagnostic("SHORT_SERIES", "error", f"{src}@{sec}",
                        "at least 2 rows required")])
    return pts, diags


def _build_baseline(doc, src):
    kind = doc.header.get("baseline_kind", "linear")
    pts, diags = _anchor_series(doc, src, "baseline", "b")
    if kind not in ("linear", "anchor"):
        diags.append(Diagnostic("UNKNOWN_BASELINE_KIND", "warning",
                                f"{src}:1",
                                f"baseline_kind {kind!r} treated as 'anchor'"))
        kind = "anchor"
    return {"kind": kind, "anchors": pts}, diags


def _build_blank(doc, src):
    pts, diags = _anchor_series(doc, src, "blank", "p")
    label = doc.header.get("blank_id", "B1")
    return {"blank_id": label, "samples": pts}, diags
