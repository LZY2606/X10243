"""Per-injection power-curve integration.

Hard domain rules implemented here as structured state (not log lines):

* Post-injection peaks may overlap.  When the next injection arrives before
  the previous peak settles, both windows are still integrated and the
  injection is flagged ``OVERLAP_PEAKS``.
* An integration window never crosses a data gap.  Gaps split the window:
  the valid segments are integrated, coverage is reported, and the result
  is flagged ``WINDOW_CLIPPED_AT_GAP``.
* Integration endpoints are real sample coordinates snapped inward,
  never extrapolated.
* The first injection can be discarded (a rule decision); discarding does
  not delete its raw integral, it marks ``FIRST_INJECTION_DISCARDED``.
"""

from __future__ import annotations

import numpy as np

from .diagnostics import SEMANTIC_INVALID, diagnostic
from .hashing import digest
from .units import is_finite

OVERLAP_PEAKS = "OVERLAP_PEAKS"
WINDOW_CLIPPED_AT_GAP = "WINDOW_CLIPPED_AT_GAP"
WINDOW_FULL = "WINDOW_FULL"
FIRST_INJECTION_DISCARDED = "FIRST_INJECTION_DISCARDED"
BASELINE_GAP = "BASELINE_ANCHOR_GAP"


def _valid_sample_arrays(power_payload):
    samples = power_payload["samples"]
    ts, ps, lines = [], [], []
    for s in samples:
        t, p = s["time"], s["power"]
        if is_finite_q(t) and is_finite_q(p):
            ts.append(t["value"])
            ps.append(p["value"])
            lines.append(s["line"])
    order = np.argsort(ts)
    return (np.asarray(ts, dtype=float)[order],
            np.asarray(ps, dtype=float)[order],
            [lines[i] for i in order])


def is_finite_q(cell):
    return bool(cell and cell.get("value") is not None
                and np.isfinite(cell["value"]))


def baseline_at(t, anchors):
    """Piecewise-linear baseline; constant outside the anchor range."""
    pts = [(a["time"]["value"], a["value"]["value"]) for a in anchors
           if is_finite_q(a["time"]) and is_finite_q(a["value"])]
    pts.sort()
    if not pts:
        return None
    bt = np.array([p[0] for p in pts])
    bv = np.array([p[1] for p in pts])
    return float(np.interp(t, bt, bv, bv[0], bv[-1]))


def detect_gaps(times, gap_threshold_s):
    """Return list of (start_index, end_index, gap_seconds) gaps."""
    gaps = []
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt > gap_threshold_s:
            gaps.append((i - 1, i, float(dt)))
    return gaps


def integrate(power_material, schedule_material, baseline_material, rules,
              sampling_period_s=2.0, gap_factor=5.0):
    """Integrate every injection. Pure function over material versions.

    Returns a dict: identity, per-injection records, diagnostics.
    """
    diags = []
    t, p, lines = _valid_sample_arrays(power_material.payload)
    if len(t) < 2:
        return None, [diagnostic(SEMANTIC_INVALID,
                                 "power curve has fewer than 2 valid samples")]
    sampling_period_s = float(np.median(np.diff(t))) or sampling_period_s
    gap_threshold = gap_factor * sampling_period_s
    gaps = detect_gaps(t, gap_threshold)

    anchors = baseline_material.payload.get("anchors", [])
    bad_anchors = [a for a in anchors if not (is_finite_q(a["time"]) and is_finite_q(a["value"]))]
    if bad_anchors:
        diags.append(diagnostic(
            SEMANTIC_INVALID,
            f"{len(bad_anchors)} baseline anchor(s) unconvertible; using remaining anchors",
            lines=[a["line"] for a in bad_anchors]))

    injections = [i for i in schedule_material.payload["injections"] if i["index"] is not None]
    injections.sort(key=lambda i: (i["time"]["value"] if is_finite_q(i["time"]) else np.inf,
                                   i["index"]))
    start_off = rules["start_offset_s"]
    end_off = rules["end_offset_s"]
    settle = rules["settle_s"]
    discard_first = rules.get("discard_first_injection", True)
    first_index = injections[0]["index"] if injections else None

    records = []
    for pos, inj in enumerate(injections):
        if not (is_finite_q(inj["time"]) and is_finite_q(inj["volume"])):
            records.append(_bad_injection(inj))
            continue
        t0 = inj["time"]["value"]
        w_lo = t0 + start_off
        w_hi = t0 + end_off
        if pos + 1 < len(injections) and is_finite_q(injections[pos + 1]["time"]):
            w_hi = min(w_hi, injections[pos + 1]["time"]["value"] + start_off)
        seg_mask = (t >= w_lo) & (t <= w_hi)
        seg_t, seg_p = t[seg_mask], p[seg_mask]
        seg_lines = [lines[i] for i, m in enumerate(seg_mask) if m]

        flags = []
        seg_diags = []
        coverage = 1.0
        if len(seg_t) >= 2:
            # gap clipping: break at gaps, integrate valid segments only
            integral_j = 0.0
            valid_span = 0.0
            clipped = False
            run_start = 0
            for gi, (a, b, dt_gap) in enumerate(
                    detect_gaps(seg_t, gap_threshold)):
                tt = seg_t[run_start:b]
                pp = seg_p[run_start:b]
                integral_j += _integrate_segment(tt, pp, anchors)
                valid_span += float(np.sum(np.diff(tt))) if len(tt) > 1 else 0.0
                run_start = b
                clipped = True
            tt = seg_t[run_start:]
            pp = seg_p[run_start:]
            integral_j += _integrate_segment(tt, pp, anchors)
            valid_span += float(np.sum(np.diff(tt))) if len(tt) > 1 else 0.0
            nominal_span = float(np.sum(np.diff(seg_t)))
            coverage = (valid_span / nominal_span) if nominal_span > 0 else 1.0
            if clipped:
                flags.append(WINDOW_CLIPPED_AT_GAP)
                seg_diags.append(diagnostic(
                    SEMANTIC_INVALID,
                    f"injection {inj['index']}: integration window clipped at data gap(s); "
                    f"coverage {coverage:.3f}",
                    injection=inj["index"]))
            # overlap: previous peak still above settle threshold at window start
            if pos > 0:
                prev_t0 = injections[pos - 1]["time"]["value"]
                age = t0 - prev_t0
                if age < settle:
                    flags.append(OVERLAP_PEAKS)
                    seg_diags.append(diagnostic(
                        SEMANTIC_INVALID,
                        f"injection {inj['index']}: peak overlap (interval {age:.1f}s < settle {settle:.1f}s)",
                        injection=inj["index"]))
            status = WINDOW_FULL if not clipped else WINDOW_CLIPPED_AT_GAP
        else:
            integral_j = None
            coverage = 0.0
            status = WINDOW_CLIPPED_AT_GAP
            flags.append(WINDOW_CLIPPED_AT_GAP)
            seg_diags.append(diagnostic(
                SEMANTIC_INVALID,
                f"injection {inj['index']}: no valid samples inside window [{w_lo}, {w_hi}]",
                injection=inj["index"]))

        if discard_first and inj["index"] == first_index:
            flags.append(FIRST_INJECTION_DISCARDED)

        records.append({
            "index": inj["index"],
            "injection_time_s": t0,
            "window_start_s": _snap_in(t, w_lo),
            "window_end_s": _snap_in(t, w_hi),
            "window_start_nominal_s": w_lo,
            "window_end_nominal_s": w_hi,
            "volume_l": inj["volume"]["value"],
            "volume_raw": inj["volume"]["raw"],
            "integral_j": integral_j,
            "coverage": coverage,
            "status": status,
            "flags": flags,
            "sample_lines": seg_lines[:500],
            "diagnostics": seg_diags,
            "usable_in_fit": (FIRST_INJECTION_DISCARDED not in flags
                              and integral_j is not None
                              and coverage > 0.5),
        })
        diags.extend(seg_diags)

    result = {
        "records": records,
        "gaps_s": [{"between_lines": [lines[a], lines[b]], "gap_s": g}
                   for a, b, g in gaps],
        "sampling_period_s": sampling_period_s,
        "diagnostics": diags,
    }
    result["result_id"] = result_identity(power_material.version_id,
                                          schedule_material.version_id,
                                          baseline_material.version_id, rules, result)
    result["parents"] = {
        "power": power_material.version_id,
        "schedule": schedule_material.version_id,
        "baseline": baseline_material.version_id,
    }
    result["rule_id"] = digest("rule_integration_rules", rules)
    return result, []


def _integrate_segment(tt, pp, anchors):
    if len(tt) < 2:
        return 0.0
    base = np.array([baseline_at(x, anchors) if anchors else 0.0 for x in tt])
    base = np.where(np.isnan(base), 0.0, base)
    return float(np.trapz(pp - base, tt))


def _snap_in(t, tq):
    idx = int(np.searchsorted(t, tq))
    candidates = []
    if idx < len(t):
        candidates.append(float(t[idx]))
    if idx > 0:
        candidates.append(float(t[idx - 1]))
    if not candidates:
        return None
    return min(candidates, key=lambda z: abs(z - tq))


def _bad_injection(inj):
    return {
        "index": inj["index"],
        "injection_time_s": inj["time"].get("value") if inj.get("time") else None,
        "window_start_s": None,
        "window_end_s": None,
        "window_start_nominal_s": None,
        "window_end_nominal_s": None,
        "volume_l": inj["volume"].get("value") if inj.get("volume") else None,
        "volume_raw": inj["volume"]["raw"] if inj.get("volume") else None,
        "integral_j": None,
        "coverage": 0.0,
        "status": "INPUT_INVALID",
        "flags": ["INPUT_INVALID"],
        "sample_lines": [],
        "diagnostics": [diagnostic(SEMANTIC_INVALID,
                                   f"injection {inj['index']} has unconvertible time/volume",
                                   injection=inj["index"])],
        "usable_in_fit": False,
    }


def result_identity(power_id, schedule_id, baseline_id, rules, result):
    payload = {
        "parents": {"power": power_id, "schedule": schedule_id, "baseline": baseline_id},
        "rule_id": digest("rule_integration_rules", rules),
        # integrals derive only from the consumed numeric content, so a
        # concentration revision cannot change them.
        "records": [
            {"index": r["index"], "integral_j": round(r["integral_j"], 12)
                if r["integral_j"] is not None else None,
             "window_start_s": r["window_start_s"], "window_end_s": r["window_end_s"],
             "flags": r["flags"], "coverage": round(r["coverage"], 9)}
            for r in result["records"]
        ],
    }
    return digest("integral", payload)
