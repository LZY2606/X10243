"""Injection integration and blank (dilution heat) correction.

Rules enforced here:
- peaks after an injection may overlap; each injection keeps its own window;
- an integration window must never cross a data gap: windows that would are
  rejected with a located diagnostic instead of a fabricated number;
- the first injection can be discarded (flagged excluded, never deleted);
- blank subtraction requires an explicit correspondence rule; a missing
  counterpart keeps the raw value in place with a diagnostic."""

import numpy as np

from .units import convert, UnitUnconvertible


class SemanticUnsolvable(Exception):
    status = "semantic_unsolvable"


def diag(code, location, message):
    return {"code": code, "location": location, "message": message}


def _segments(power):
    segs = []
    for i, seg in enumerate(power.get("segments", [])):
        t = np.asarray(seg["t"], dtype=float)
        p = np.asarray(seg["p"], dtype=float)
        if t.size:
            segs.append((i, t, p))
    return segs


def gaps(power):
    segs = _segments(power)
    out = []
    for a, b in zip(segs, segs[1:]):
        out.append([float(a[1][-1]), float(b[1][0])])
    return out


def integrate(power, injections, baseline, params):
    diagnostics = []
    window = float(params.get("window_seconds", 0))
    if window <= 0:
        raise SemanticUnsolvable("integration window must be positive seconds")
    try:
        power_factor = convert(1.0, power.get("power_unit", "ucal/s"), "ucal/s", "power")
    except UnitUnconvertible as exc:
        diagnostics.append(diag("unit_unconvertible",
                                {"material": "mat-power", "field": "power_unit"},
                                str(exc)))
        return {"status": "diagnostic", "diagnostics": diagnostics, "injections": []}
    segs = _segments(power)
    if not segs:
        raise SemanticUnsolvable("power curve has no samples")
    coeffs = baseline.get("coeffs", [0.0])
    entries = sorted(injections["entries"], key=lambda e: e["index"])
    times = [e["time"] for e in entries]
    out = []
    for pos, entry in enumerate(entries):
        idx = entry["index"]
        t0 = float(entry["time"])
        t1 = t0 + window
        loc = {"material": "mat-power", "injection_index": idx}
        record = {"index": idx, "time": t0, "volume": entry["volume"],
                  "window": [t0, t1], "status": "ok", "heat_ucal": None,
                  "excluded": False, "exclude_reason": None,
                  "peak_overlap": False, "diagnostics": []}
        if params.get("discard_first") and pos == 0:
            record["excluded"] = True
            record["exclude_reason"] = "first_injection_discarded"
        seg = next((s for s in segs if s[1][0] <= t0 and t1 <= s[1][-1]), None)
        if seg is None:
            record["status"] = "invalid"
            record["diagnostics"].append(
                diag("integration_window_crosses_gap", loc,
                     f"window [{t0}, {t1}] is not contained in any continuous "
                     f"data segment; not integrated"))
        else:
            _, t, p = seg
            mask = (t >= t0) & (t <= t1)
            if mask.sum() < 3:
                record["status"] = "invalid"
                record["diagnostics"].append(
                    diag("insufficient_samples", loc,
                         f"only {int(mask.sum())} samples inside window"))
            else:
                base = np.polyval(coeffs, t[mask])
                record["heat_ucal"] = float(np.trapezoid(
                    (p[mask] - base) * power_factor, t[mask]))
                pre = float(p[t <= t0][-1] - np.polyval(coeffs, t0)) if (t <= t0).any() else 0.0
                if abs(pre) > float(params.get("overlap_threshold", 0.05)):
                    record["peak_overlap"] = True
        if any(t0 < other <= t1 for other in times):
            record["window_neighbor"] = True
        out.append(record)
    status = "ok" if all(r["status"] == "ok" for r in out) else "diagnostic"
    return {"status": status, "diagnostics": diagnostics, "injections": out,
            "params_used": {"window_seconds": window,
                            "discard_first": bool(params.get("discard_first"))}}


def correct_blank(integration_payload, blank, params):
    rule = params.get("rule")
    diagnostics = []
    if not rule:
        diagnostics.append(diag("blank_rule_unspecified", {"material": "mat-blank"},
                                "dilution heat subtraction requires an explicit "
                                "correspondence rule (e.g. by_index)"))
        return {"status": "diagnostic", "diagnostics": diagnostics, "injections": []}
    blank_by_index = {e["injection_index"]: e for e in blank.get("entries", [])}
    corrected = []
    for rec in integration_payload["injections"]:
        item = dict(rec)
        item["blank_ucal"] = None
        item["corrected_ucal"] = None
        item["blank_applied"] = False
        item["diagnostics"] = list(rec.get("diagnostics", []))
        if rec["status"] != "ok":
            corrected.append(item)
            continue
        match = blank_by_index.get(rec["index"]) if rule == "by_index" else None
        if match is None:
            item["diagnostics"].append(
                diag("blank_correspondence_missing",
                     {"material": "mat-blank", "injection_index": rec["index"]},
                     f"no blank injection corresponds under rule {rule!r}; "
                     f"raw heat kept in place"))
            item["corrected_ucal"] = rec["heat_ucal"]
        else:
            heat = convert(match["heat"], match.get("heat_unit", "ucal"), "ucal", "heat")
            item["blank_ucal"] = heat
            item["corrected_ucal"] = rec["heat_ucal"] - heat
            item["blank_applied"] = True
        corrected.append(item)
    status = "ok" if not any(c["diagnostics"] for c in corrected) else "diagnostic"
    return {"status": status, "diagnostics": diagnostics, "rule": rule,
            "injections": corrected}
