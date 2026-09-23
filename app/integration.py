"""Per-injection integration engine.

Pure NumPy functions, callable directly from tests (no server, no browser).

Hard rules implemented here:
  * post-injection peaks may overlap, but an integration window never spans a
    data gap (missing sample or time discontinuity);
  * dilution-heat subtraction requires an explicit correspondence rule
    (``by_index`` / ``by_volume_tolerance``);
  * the first injection is never silently dropped here: it is integrated like
    any other and marked ``discarded`` by a decision in the event layer;
  * unobserved endpoints (series ends before recovery) are flagged, not hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass
class Gap:
    t_before: float
    t_after: float
    row_before: int
    row_after: int
    reason: str

    def to_dict(self):
        return asdict(self)


def _valid_arrays(samples: list[dict]):
    rows, t, p = [], [], []
    for s in samples:
        if s.get("missing") or s.get("time_s") is None or s.get("power_ucal_s") is None:
            continue
        rows.append(s["row"])
        t.append(s["time_s"])
        p.append(s["power_ucal_s"])
    return np.array(rows), np.array(t, dtype=float), np.array(p, dtype=float)


def detect_gaps(samples: list[dict], jump_factor: float = 5.0) -> list[Gap]:
    gaps: list[Gap] = []
    prev = None
    for s in samples:
        cur = (s["row"], s.get("time_s"), s.get("power_ucal_s"), s.get("missing"))
        if prev is not None:
            prow, pt, _, _ = prev
            row, t, p, missing = cur
            if missing or t is None or p is None:
                gaps.append(Gap(float("nan"), float("nan"), prow, row,
                                "missing_sample"))
            else:
                if pt is not None:
                    dt = t - pt
                    if dt <= 0:
                        gaps.append(Gap(pt, t, prow, row, "non_monotonic_time"))
                    # population threshold computed lazily below
        prev = cur

    rows, t, _p = _valid_arrays(samples)
    if len(t) >= 3:
        dts = np.diff(t)
        med = float(np.median(dts))
        for i, dt in enumerate(dts):
            if med > 0 and dt > jump_factor * med:
                gaps.append(Gap(float(t[i]), float(t[i + 1]), int(rows[i]),
                                int(rows[i + 1]), "time_jump"))
    # de-duplicate by row pair, keep deterministic order
    seen, out = set(), []
    for g in sorted(gaps, key=lambda x: (x.row_before, x.row_after, x.reason)):
        key = (g.row_before, g.row_after, g.reason)
        if key not in seen:
            seen.add(key)
            out.append(g)
    return out


def baseline_at(baseline_samples: list[dict] | None, times: np.ndarray) -> np.ndarray:
    """Linearly interpolate a separately versioned baseline curve."""
    if not baseline_samples:
        return np.zeros_like(times)
    _r, bt, bp = _valid_arrays(baseline_samples)
    if len(bt) == 0:
        return np.zeros_like(times)
    if len(bt) == 1:
        return np.full_like(times, bp[0])
    return np.interp(times, bt, bp, left=bp[0], right=bp[-1])


def _segment_contains_gap(t0: float, t1: float, power_samples: list[dict],
                          jump_factor: float = 5.0) -> dict | None:
    """Check the original sample sequence strictly inside [t0, t1].

    Walk every row whose timestamp is inside the window; a missing/unparseable
    sample there, or a cadence jump measured between two in-window samples,
    invalidates the window. Gaps before t0 or after t1 never invalidate it.
    """
    in_window = [s_ for s_ in power_samples
                 if s_.get("time_s") is not None
                 and t0 - 1e-9 <= s_["time_s"] <= t1 + 1e-9]
    for s_ in in_window:
        if s_.get("missing") or s_.get("power_ucal_s") is None:
            return {"code": "missing_sample",
                    "row_before": s_["row"] - 1, "row_after": s_["row"],
                    "t_before": None, "t_after": None}
    good = [s_ for s_ in in_window
            if not s_.get("missing") and s_.get("time_s") is not None
            and s_.get("power_ucal_s") is not None]
    if len(good) < 2:
        return None
    dts = [good[i]["time_s"] - good[i - 1]["time_s"]
           for i in range(1, len(good))]
    med = float(np.median(dts))
    if med <= 0:
        return {"code": "non_monotonic_time"}
    for i, dt in enumerate(dts, start=1):
        if dt > jump_factor * med:
            return {"code": "time_jump",
                    "row_before": good[i - 1]["row"],
                    "row_after": good[i]["row"],
                    "t_before": good[i - 1]["time_s"],
                    "t_after": good[i]["time_s"]}
    return None


@dataclass
class InjectionResult:
    index: int
    t_start_s: float
    t_end_s: float | None
    t_end_observed_s: float | None
    q_ucal: float | None
    baseline_ucal_s: float
    overlap_with_next: bool
    endpoint_open: bool
    status: str
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    window: dict[str, Any] | None = None
    n_points: int = 0

    def to_dict(self):
        return asdict(self)


def integrate_injections(power_samples: list[dict],
                         injections: list[dict],
                         baseline_samples: list[dict] | None,
                         noise_sigma_floor: float = 0.02) -> list[InjectionResult]:
    rows, times, power = _valid_arrays(power_samples)
    base = baseline_at(baseline_samples, times)
    corrected = power - base
    gaps = detect_gaps(power_samples)

    results: list[InjectionResult] = []
    n = len(injections)
    for k, inj in enumerate(injections):
        t0 = inj["time_s"]
        t_next = injections[k + 1]["time_s"] if k + 1 < n else None
        diag: list[dict[str, Any]] = []

        if t0 is None:
            results.append(InjectionResult(
                inj["index"], float("nan"), None, None, None, 0.0,
                False, False, "rejected_unparseable_time",
                [{"code": "unparseable_injection_time", "index": inj["index"]}]))
            continue

        i0 = int(np.searchsorted(times, t0, side="left"))
        pre_mask = (times >= t0 - 90.0) & (times <= max(t0 - 5.0, t0))
        pre = corrected[pre_mask] if pre_mask.any() else corrected[:0]
        level = max(3.0 * float(np.std(pre)) if len(pre) > 4 else 0.0,
                    noise_sigma_floor)

        # search forward for the last excited sample
        if t_next is not None:
            search = np.where((times > t0) & (times < t_next))[0]
        else:
            search = np.where(times > t0)[0]
        excited = search[np.abs(corrected[search]) > level] if len(search) else search
        t_end_obs = float(times[excited[-1]]) if len(excited) else None

        overlap = False
        endpoint_open = False
        status = "ok"
        if t_end_obs is None:
            # no detectable peak: zero-area degenerate window
            t_end = t0
        elif t_next is not None and t_end_obs >= t_next - 1e-9:
            overlap = True
            t_end = t_next
            diag.append({"code": "overlapping_peak",
                         "index": inj["index"],
                         "next_index": injections[k + 1]["index"],
                         "detail": "peak had not recovered before next injection; "
                                   "window truncated at next injection boundary"})
        elif k == n - 1 and t_end_obs >= float(times[-1]) - 1e-9:
            endpoint_open = True
            t_end = float(times[-1])
            status = "endpoint_unobserved"
            diag.append({"code": "endpoint_unobserved", "index": inj["index"],
                         "detail": "series ends before peak returns to baseline"})
        else:
            t_end = t_end_obs

        gap = _segment_contains_gap(t0, t_end, power_samples)
        if gap is not None:
            diag.append({"code": "window_crosses_gap", "index": inj["index"],
                         "gap": gap,
                         "detail": "integration window cannot span a data gap"})
            results.append(InjectionResult(
                inj["index"], t0, None, t_end_obs, None, 0.0, overlap,
                endpoint_open, "rejected_gap", diag))
            continue

        mask = (times >= t0) & (times <= t_end)
        seg_t = times[mask]
        seg_y = corrected[mask]
        q = float(np.trapezoid(seg_y, seg_t)) if len(seg_t) >= 2 else 0.0
        b0 = float(np.interp(t0, times, base))
        results.append(InjectionResult(
            inj["index"], t0, t_end, t_end_obs, q, b0, overlap, endpoint_open,
            status, diag,
            window={"start_s": t0, "end_s": t_end,
                    "start_raw": inj.get("time_raw"),
                    "time_basis": "experiment_clock(s)",
                    "power_unit": "µcal/s", "energy_unit": "µcal"},
            n_points=int(len(seg_t))))
    return results


def correspondence(main_indices: list[int], blank_injections: list[dict],
                   rule: str, volume_tolerance_ul: float = 0.5
                   ) -> dict[int, dict[str, Any]]:
    """Map main injection index -> matched blank injection, with explicit rule."""
    mapping: dict[int, dict[str, Any]] = {}
    for idx in main_indices:
        match, reason = None, None
        if rule == "by_index":
            for b in blank_injections:
                if b["index"] == idx:
                    match, reason = b, "index_equal"
                    break
            if match is None:
                reason = "no_blank_with_equal_index"
        elif rule == "by_volume_tolerance":
            # Need main volume to compare; caller attaches via index order.
            raise ValueError("by_volume_tolerance needs main injections; "
                             "use correspondence_with_volumes")
        else:
            raise ValueError(f"unknown correspondence rule: {rule}")
        mapping[idx] = {"blank_index": match["index"] if match else None,
                        "rule": rule, "reason": reason,
                        "matched": match is not None}
    return mapping


def correspondence_with_volumes(main_injections: list[dict],
                                blank_injections: list[dict],
                                rule: str,
                                volume_tolerance_ul: float = 0.5
                                ) -> dict[int, dict[str, Any]]:
    if rule == "by_index":
        return correspondence([i["index"] for i in main_injections],
                              blank_injections, rule)
    if rule != "by_volume_tolerance":
        raise ValueError(f"unknown correspondence rule: {rule}")
    mapping: dict[int, dict[str, Any]] = {}
    for inj in main_injections:
        best, best_diff = None, None
        if inj.get("volume_ul") is not None:
            for b in blank_injections:
                if b.get("volume_ul") is None:
                    continue
                diff = abs(b["volume_ul"] - inj["volume_ul"])
                if diff <= volume_tolerance_ul and (best_diff is None or diff < best_diff):
                    best, best_diff = b, diff
        mapping[inj["index"]] = {
            "blank_index": best["index"] if best else None,
            "rule": rule,
            "tolerance_ul": volume_tolerance_ul,
            "reason": "volume_within_tolerance" if best else "no_blank_volume_within_tolerance",
            "matched": best is not None}
    return mapping


def apply_dilution(main_results: list[InjectionResult],
                   blank_results: list[InjectionResult],
                   mapping: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    blank_q = {r.index: r.q_ucal for r in blank_results}
    out = []
    for r in main_results:
        entry = r.to_dict()
        info = mapping.get(r.index, {"matched": False, "rule": "none",
                                     "reason": "no_rule_declared"})
        entry["blank_correspondence"] = info
        if r.q_ucal is None:
            entry["net_q_ucal"] = None
            entry["dilution_status"] = "not_applicable"
        elif info.get("matched") and blank_q.get(info["blank_index"]) is not None:
            entry["net_q_ucal"] = r.q_ucal - blank_q[info["blank_index"]]
            entry["dilution_status"] = "subtracted"
        else:
            entry["net_q_ucal"] = None
            entry["dilution_status"] = "unsubtracted_missing_correspondence"
            entry["diagnostics"].append({
                "code": "blank_correspondence_missing",
                "index": r.index,
                "rule": info.get("rule"),
                "reason": info.get("reason"),
                "detail": "dilution heat cannot be subtracted without a matching "
                          "blank injection under the declared rule"})
        out.append(entry)
    return out
