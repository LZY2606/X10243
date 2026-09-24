"""Turn structurally parsed documents into versioned *material* records.

Materials are immutable laboratory inputs: power samples, injection timing
and volumes, concentrations, conditions, baselines and blank runs.  Every
parsed item keeps its source line, raw tokens and (where conversion
succeeds) canonical SI values.  Unconvertible values stay in place with a
positioned diagnostic instead of being replaced by nulls/defaults.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from .errors import Diagnostic, FormatCorrupted, ResourceExhausted, SemanticImpasse
from .parser import Document, parse_document
from .units import parse_quantity

MAX_SERIES_POINTS = 200_000
MAX_INJECTIONS = 1000
IMPLIED_GAP_FACTOR = 10.0


@dataclass
class Budgets:
    max_series_points: int = MAX_SERIES_POINTS
    max_injections: int = MAX_INJECTIONS


def _err(doc: Document, code: str, message: str, line: int | None = None,
         section: str | None = None, raw: str | None = None) -> Diagnostic:
    return Diagnostic(code=code, message=message, file=doc.file, line=line,
                      section=section, raw=raw)


def _require_meta(doc: Document, key: str, diags: list[Diagnostic]) -> str:
    if key not in doc.meta:
        diags.append(_err(doc, "S010", f"meta '{key}' is required",
                          line=doc.meta_lines.get(key), section="meta"))
        return ""
    return doc.meta[key]


def _meta_q(doc: Document, key: str, dimension: str,
            diags: list[Diagnostic]):
    if key not in doc.meta:
        diags.append(_err(doc, "S010", f"meta '{key}' is required",
                          line=doc.meta_lines.get(key), section="meta"))
        return None
    q = parse_quantity(doc.meta[key], dimension)
    if q.canonical_value is None:
        diags.append(_err(
            doc, "S011",
            f"meta '{key}' value {doc.meta[key]!r} cannot be converted to a "
            f"canonical {dimension}; kept as raw text",
            line=doc.meta_lines[key], section="meta", raw=doc.meta[key]))
        return q
    return q


def _series_points(doc: Document, section: str, time_unit: str,
                   power_unit: str, diags: list[Diagnostic],
                   budgets: Budgets) -> list[dict[str, Any]]:
    rows = doc.sections.get(section, [])
    if len(rows) > budgets.max_series_points:
        raise ResourceExhausted(
            f"@{section} has {len(rows)} points, budget "
            f"{budgets.max_series_points}",
            [_err(doc, "R001",
                  f"series point budget exhausted: {len(rows)} > "
                  f"{budgets.max_series_points}", section=section)])
    points: list[dict[str, Any]] = []
    for row in rows:
        t = parse_quantity(f"{row.tokens[0]} {time_unit}", "time")
        p = parse_quantity(f"{row.tokens[1]} {power_unit}", "power")
        if t.canonical_value is None:
            diags.append(_err(doc, "S012",
                              f"time token {row.tokens[0]!r} cannot be converted; "
                              "point kept at source position", row.line, section,
                              row.raw))
        if p.canonical_value is None:
            diags.append(_err(doc, "S013",
                              f"power token {row.tokens[1]!r} cannot be converted; "
                              "point kept at source position", row.line, section,
                              row.raw))
        points.append({
            "line": row.line,
            "raw": row.raw,
            "time": t.to_dict(),
            "power": p.to_dict(),
            "comment": row.comment,
        })
    good = [p for p in points if p["time"]["canonical_value"] is not None]
    good.sort(key=lambda p: (p["time"]["canonical_value"], p["line"]))
    return good


def _injection_items(doc: Document, diags: list[Diagnostic],
                     budgets: Budgets) -> list[dict[str, Any]]:
    rows = doc.sections.get("injections", [])
    if len(rows) > budgets.max_injections:
        raise ResourceExhausted(
            f"{len(rows)} injections exceed budget {budgets.max_injections}",
            [_err(doc, "R002", "injection budget exhausted", section="injections")])
    seen: set[int] = set()
    items: list[dict[str, Any]] = []
    for row in rows:
        try:
            idx = int(row.tokens[0])
        except ValueError:
            diags.append(_err(doc, "S020",
                              f"injection index {row.tokens[0]!r} is not an integer",
                              row.line, "injections", row.raw))
            continue
        if idx in seen:
            diags.append(_err(doc, "S021", f"duplicate injection index {idx}",
                              row.line, "injections", row.raw))
        seen.add(idx)
        t = parse_quantity(" ".join(row.tokens[1:]), "time")
        volume = None
        # volume is the last whitespace token; anything between is a unit tail
        # supported formats: "1  12.0 s  2.0 uL" -> tokens after idx
        rest = row.tokens[1:]
        t = parse_quantity(" ".join(rest[:2]) if len(rest) >= 2 else " ".join(rest),
                           "time")
        if len(rest) >= 3:
            volume = parse_quantity(" ".join(rest[2:4]), "volume")
        if t.canonical_value is None:
            diags.append(_err(doc, "S022",
                              f"injection time cannot be converted from "
                              f"{' '.join(rest)!r}", row.line, "injections", row.raw))
        if volume is None or volume.canonical_value is None:
            diags.append(_err(doc, "S023",
                              "injection volume missing or unconvertible",
                              row.line, "injections", row.raw))
        items.append({
            "index": idx,
            "line": row.line,
            "raw": row.raw,
            "time": t.to_dict(),
            "volume": volume.to_dict() if volume else None,
            "comment": row.comment,
        })
    items.sort(key=lambda it: (it["index"], it["line"]))
    return items


def _gaps(doc: Document, diags: list[Diagnostic], time_unit: str,
          points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for row in doc.sections.get("gaps", []):
        a = parse_quantity(f"{row.tokens[0]} {time_unit}", "time")
        b = parse_quantity(f"{row.tokens[1]} {time_unit}", "time")
        if a.canonical_value is None or b.canonical_value is None:
            diags.append(_err(doc, "S030", "gap bounds unconvertible", row.line,
                              "gaps", row.raw))
            continue
        if b.canonical_value <= a.canonical_value:
            diags.append(_err(doc, "S031", "gap end must be greater than gap start",
                              row.line, "gaps", row.raw))
            continue
        gaps.append({"start": a.to_dict(), "end": b.to_dict(), "kind": "declared",
                     "line": row.line, "note": row.comment})
    # implied gaps: irregular sampling much larger than the median interval
    if len(points) >= 5:
        dts = [points[i + 1]["time"]["canonical_value"] -
               points[i]["time"]["canonical_value"] for i in range(len(points) - 1)]
        positive = [d for d in dts if d > 0]
        if positive:
            median_dt = statistics.median(positive)
            for i, dt in enumerate(dts):
                if dt > IMPLIED_GAP_FACTOR * median_dt and dt > 1.0:
                    start = points[i]["time"]["canonical_value"]
                    end = points[i + 1]["time"]["canonical_value"]
                    if not any(abs(g["start"]["canonical_value"] - start) < 1e-9
                               and abs(g["end"]["canonical_value"] - end) < 1e-9
                               for g in gaps):
                        gaps.append({
                            "start": {**points[i]["time"]},
                            "end": {**points[i + 1]["time"]},
                            "kind": "implied",
                            "line": points[i + 1]["line"],
                            "note": "sampling interval > 10x median; no points "
                                    "exist across this span",
                        })
    gaps.sort(key=lambda g: (g["start"]["canonical_value"], g["line"]))
    return gaps


def _base_material(doc: Document) -> dict[str, Any]:
    return {
        "kind": doc.kind(),
        "file": doc.file,
        "run": doc.meta.get("run"),
        "meta": {k: {"raw": v, "line": doc.meta_lines[k]}
                 for k, v in doc.meta.items()},
        "time_basis": doc.meta.get("time_basis"),
        "items": {},
        "diagnostics": [],
    }


def _raise_if_errors(diags: list[Diagnostic]) -> None:
    errors = [d for d in diags if d.severity == "error"]
    if errors:
        raise SemanticImpasse(
            f"{len(errors)} semantic problem(s) in laboratory material", errors)


def build_material(text: str, file: str = "input.itc",
                   budgets: Budgets | None = None) -> dict[str, Any]:
    """Parse + semantically validate one laboratory file into a material."""
    budgets = budgets or Budgets()
    doc = parse_document(text, file)
    kind = doc.kind()
    assert kind is not None
    diags: list[Diagnostic] = list(doc.diagnostics)
    mat = _base_material(doc)

    if kind in ("power", "blank"):
        time_unit = _require_meta(doc, "time_unit", diags) or "s"
        power_unit = _require_meta(doc, "power_unit", diags) or "uW"
        _meta_q(doc, "temperature", "temperature", diags)
        injections = _injection_items(doc, diags, budgets)
        if not injections:
            diags.append(_err(doc, "S040", "@injections must list at least one shot",
                              section="injections"))
        section = "blank_series" if kind == "blank" else "series"
        points = _series_points(doc, section, time_unit, power_unit, diags, budgets)
        if not points:
            diags.append(_err(doc, "S041", f"@{section} must contain power samples",
                              section=section))
        baseline_points = _series_points(doc, "baseline", time_unit, power_unit,
                                         diags, Budgets(max_series_points=10**9,
                                                        max_injections=MAX_INJECTIONS)) \
            if doc.sections.get("baseline") else []
        gaps = _gaps(doc, diags, time_unit, points)
        mat["items"] = {"injections": injections, "series": points,
                        "baseline": baseline_points, "gaps": gaps}
        mat["units"] = {"time": time_unit, "power": power_unit}
    elif kind == "injections":
        injections = _injection_items(doc, diags, budgets)
        if not injections:
            diags.append(_err(doc, "S040", "@injections must list at least one shot",
                              section="injections"))
        mat["items"] = {"injections": injections}
    elif kind == "concentration":
        unit = doc.meta.get("concentration_unit", "M")
        rows = doc.sections.get("concentration", [])
        if not rows:
            diags.append(_err(doc, "S050", "@concentration must list revised values",
                              section="concentration"))
        entries = []
        for row in rows:
            key = row.tokens[0]
            q = parse_quantity(" ".join(row.tokens[1:]), "concentration")
            if q.canonical_value is None:
                diags.append(_err(doc, "S051",
                                  f"concentration for {key!r} unconvertible",
                                  row.line, "concentration", row.raw))
            entries.append({"key": key, "line": row.line, "raw": row.raw,
                            "quantity": q.to_dict(), "comment": row.comment})
        mat["items"] = {"entries": entries}
        mat["units"] = {"concentration": unit}
    elif kind == "conditions":
        pool = _meta_q(doc, "pool_volume", "volume", diags)
        temp = _meta_q(doc, "temperature", "temperature", diags)
        mat["items"] = {"pool_volume": pool.to_dict() if pool else None,
                        "temperature": temp.to_dict() if temp else None}
    elif kind == "baseline":
        time_unit = _require_meta(doc, "time_unit", diags) or "s"
        power_unit = _require_meta(doc, "power_unit", diags) or "uW"
        points = _series_points(doc, "baseline", time_unit, power_unit, diags,
                                budgets)
        if not points:
            diags.append(_err(doc, "S041", "@baseline must contain samples",
                              section="baseline"))
        mat["items"] = {"series": points}
        mat["units"] = {"time": time_unit, "power": power_unit}

    mat["diagnostics"] = [d.to_dict() for d in diags]
    _raise_if_errors(diags)
    return mat
