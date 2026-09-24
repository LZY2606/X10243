"""Per-injection integration over possibly overlapping peaks and data gaps.

Rule set version: ``integrate-v1``.

* Every injection is processed; the first may be ``discarded`` (its volume /
  diffusion is commonly perturbed).
* Integration windows are half-open ``[t0, t1)`` on the shared *seconds since
  run start* basis.
* Endpoint policies:
    - ``next_injection`` (default): window ends at the next injection start.
    - ``return_to_baseline``: window ends when the baseline-corrected signal
      stays within ``rtb_band`` ucal/s for ``rtb_hold_s`` seconds.
    - ``fixed``: explicit ``end_s`` per injection.
  A custom end may never exceed the next injection start, which is how windows
  for *overlapping peaks* are cut without mixing the next peak in.
* A window whose numeric samples do not continuously cover ``[t0, t1]`` is
  ``blocked_gap``: it is clipped at the first interruption and the injection
  is not integrated. Windows never cross a data gap.
* The baseline is linearly interpolated at each sample time (identity when
  absent). Unconvertible samples inside the window block it as well.
* Peak overlap is derived from effective windows: peak ``i`` overlaps ``j``
  when peak ``i``'s actual end exceeds peak ``j``'s start. Overlap is allowed
  and reported; only gap-crossing windows are forbidden.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any

from .errors import Diagnostic, SemanticImpasse
from .materials import Material, content_hash

RULE_VERSION = "integrate-v1"
ENDPOINT_POLICIES = ("next_injection", "return_to_baseline", "fixed")


@dataclass(frozen=True)
class NumericSeries:
    t: list[float]
    y: list[float]
    rows: list[int]
    gap_intervals: tuple[tuple[float, float, int], ...]
    bad_intervals: tuple[tuple[float, float, str, int], ...]
    span: tuple[float, float]
    diagnostics: tuple[dict[str, Any], ...]


def _intervals(points: list[tuple[float, str, int, str]],
               edge_t: tuple[float, float]):
    gap: list[tuple[float, float, int]] = []
    bad: list[tuple[float, float, str, int]] = []
    for idx, (pt, kind, row, raw) in enumerate(points):
        prev = points[idx - 1][0] if idx > 0 else edge_t[0]
        nxt = points[idx + 1][0] if idx + 1 < len(points) else edge_t[1]
        if kind == "gap":
            gap.append((prev, nxt, row))
        else:
            bad.append((prev, nxt, raw, row))
    return tuple(gap), tuple(bad)


def material_series(mat: Material) -> NumericSeries:
    payload = mat.payload
    samples = payload["samples"]
    t: list[float] = []
    y: list[float] = []
    rows: list[int] = []
    broken: list[tuple[float, str, int, str]] = []
    for s in samples:
        ts = s.get("t_s")
        if ts is None:
            continue
        if s["kind"] == "sample":
            t.append(ts)
            y.append(s["quantity"]["canonical"])
            rows.append(s["row"])
        elif s["kind"] == "gap":
            broken.append((ts, "gap", s["row"], s.get("raw", "")))
        elif s["kind"] == "unconvertible":
            broken.append((ts, "bad", s["row"], s.get("raw", "")))
    if not t:
        raise SemanticImpasse(
            f"{mat.file}: no numeric samples available",
            [Diagnostic("no_numeric_samples",
                        "every sample is gap/unconvertible", mat.file)])
    order = sorted(range(len(t)), key=lambda i: t[i])
    t = [t[i] for i in order]
    y = [y[i] for i in order]
    rows = [rows[i] for i in order]
    broken.sort(key=lambda p: p[0])
    gap_iv, bad_iv = _intervals(broken, (t[0], t[-1]))
    return NumericSeries(t=t, y=y, rows=rows, gap_intervals=gap_iv,
                         bad_intervals=bad_iv, span=(t[0], t[-1]),
                         diagnostics=tuple(payload.get("diagnostics", [])))


def interp_baseline(bt: list[float], by: list[float], times: list[float],
                    diag_file: str) -> tuple[list[float], list[Diagnostic]]:
    import numpy as np
    if not bt:
        return [0.0] * len(times), []
    diags: list[Diagnostic] = []
    arr_b = np.asarray(bt, dtype=float)
    arr_y = np.asarray(by, dtype=float)
    out: list[float] = []
    lo, hi = arr_b[0], arr_b[-1]
    for ts in times:
        if ts < lo or ts > hi:
            diags.append(Diagnostic(
                "baseline_out_of_span",
                f"baseline does not cover t={ts:g}s", diag_file,
                t_s=ts, detail={"baseline_span": [float(lo), float(hi)]}))
        out.append(float(np.interp(ts, arr_b, arr_y)))
    return out, diags


def _first_break(power: NumericSeries, t0: float, t1: float
                 ) -> tuple[float, str, int, str] | None:
    for a, b, row in power.gap_intervals:
        if b > t0 and a < t1:
            return (max(a, t0), "gap", row, "")
    for a, b, raw, row in power.bad_intervals:
        if b > t0 and a < t1:
            return (max(a, t0), "unconvertible", row, raw)
    return None


def _window_end(power: NumericSeries, corr: list[float], times: list[float],
                start_idx: int, start_s: float, next_start: float | None,
                policy: str, policy_opts: dict[str, Any]
                ) -> tuple[float, str]:
    hard_end = next_start if next_start is not None else power.span[1]
    if policy == "next_injection":
        return hard_end, "next_injection"
    if policy == "fixed":
        end = float(policy_opts["end_s"])
        if end <= start_s:
            raise SemanticImpasse(
                "fixed endpoint must be after injection start",
                [Diagnostic("bad_endpoint", "end <= start", t_s=end)])
        if next_start is not None and end > next_start + 1e-9:
            raise SemanticImpasse(
                "fixed endpoint crosses next injection start",
                [Diagnostic("endpoint_crosses_injection",
                            "custom end exceeds next injection start",
                            t_s=end, detail={"next_start": next_start})])
        return min(end, hard_end), "fixed"
    if policy == "return_to_baseline":
        band = float(policy_opts.get("rtb_band", 0.2))
        hold = float(policy_opts.get("rtb_hold_s", 2.0))
        run_start: float | None = None
        for idx in range(start_idx, len(times)):
            if next_start is not None and times[idx] > next_start:
                break
            if abs(corr[idx]) <= band:
                if run_start is None:
                    run_start = times[idx]
                if times[idx] - run_start >= hold:
                    return times[idx], "return_to_baseline"
            else:
                run_start = None
        return hard_end, "next_injection_fallback"
    raise SemanticImpasse(f"unknown endpoint policy {policy!r}",
                          [Diagnostic("unknown_policy", "bad endpoint policy",
                                      detail={"policy": policy})])


def _injection_values(mat: Material) -> list[dict[str, Any]]:
    out = []
    for inj in mat.payload["injections"]:
        if not inj["time"]["convertible"] or not inj["volume"]["convertible"]:
            raise SemanticImpasse(
                f"injection {inj['id']} has unconvertible time/volume",
                [Diagnostic("unconvertible_injection",
                            "injection schedule must be fully convertible",
                            inj.get("row"), detail={"id": inj["id"]})])
        out.append({"id": inj["id"], "row": inj["row"],
                    "t": inj["time"]["canonical"],
                    "v": inj["volume"]["canonical"],
                    "t_raw": inj["time"]["raw"], "v_raw": inj["volume"]["raw"]})
    return out


def integrate_all(power_mat: Material, injection_mat: Material,
                  baseline_mat: Material | None, *,
                  policy: str = "next_injection",
                  policy_options: dict[str, Any] | None = None,
                  fixed_ends: dict[str, float] | None = None,
                  discard_first: bool = False) -> dict[str, Any]:
    """Integrate every injection; never crosses gaps or unconvertible samples."""
    opts = dict(policy_options or {})
    fixed_ends = fixed_ends or {}
    if policy not in ENDPOINT_POLICIES:
        raise SemanticImpasse(f"unknown endpoint policy {policy!r}",
                              [Diagnostic("unknown_policy", "bad policy",
                                          detail={"policy": policy})])
    power = material_series(power_mat)
    base_t: list[float] = []
    base_y: list[float] = []
    if baseline_mat is not None:
        bs = material_series(baseline_mat)
        base_t, base_y = bs.t, bs.y
    injections = _injection_values(injection_mat)

    # preliminary windows for overlap detection (generous: up to next shot)
    raw_windows: list[dict[str, Any]] = []
    for i, inj in enumerate(injections):
        nxt = injections[i + 1]["t"] if i + 1 < len(injections) else power.span[1]
        end_s = fixed_ends[inj["id"]] if policy == "fixed" else nxt
        raw_windows.append({"id": inj["id"], "t0": inj["t"], "t1": end_s,
                            "next_start": nxt, "reason": policy})

    results: list[dict[str, Any]] = []
    base_at_all, base_diags = interp_baseline(base_t, base_y, power.t,
                                              baseline_mat.file if baseline_mat else "")
    corr_all = [power.y[k] - base_at_all[k] for k in range(len(power.t))]

    for i, inj in enumerate(injections):
        t0 = inj["t"]
        nxt = injections[i + 1]["t"] if i + 1 < len(injections) else None
        win = raw_windows[i]
        diagnostics: list[dict[str, Any]] = []
        status = "integrated"
        if discard_first and i == 0:
            status = "discarded_first"
        start_idx = bisect_left(power.t, t0)
        if start_idx >= len(power.t) or power.t[start_idx] > (nxt or power.span[1]):
            status = "no_samples"
        pol = policy
        popts = dict(opts)
        if policy == "fixed":
            if inj["id"] not in fixed_ends:
                raise SemanticImpasse(
                    f"fixed endpoint missing for injection {inj['id']}",
                    [Diagnostic("missing_fixed_end",
                                "fixed policy requires end_s for every injection",
                                detail={"id": inj["id"]})])
            popts["end_s"] = fixed_ends[inj["id"]]
        t1, end_reason = _window_end(power, corr_all, power.t, start_idx, t0,
                                     nxt, pol, popts)
        t0_actual, t1_actual = t0, t1
        area = None
        n_points = 0
        clipped_at: float | None = None
        if status not in ("discarded_first", "no_samples"):
            brk = _first_break(power, t0, t1)
            if brk is not None:
                at, kind, row, raw = brk
                status = "blocked_gap" if kind == "gap" else "blocked_unconvertible"
                diagnostics.append(Diagnostic(
                    "window_crosses_gap" if kind == "gap"
                    else "window_crosses_unconvertible",
                    "integration window crosses interrupted data; not integrated",
                    power_mat.file, row, t_s=at,
                    detail={"window": [t0, t1], "kind": kind, "raw": raw}).to_dict())
                t1_actual = at
                clipped_at = at
            else:
                lo = bisect_left(power.t, t0)
                hi = bisect_right(power.t, t1)
                seg_t = power.t[lo:hi]
                seg_c = corr_all[lo:hi]
                rows_seg = power.rows[lo:hi]
                if not seg_t:
                    status = "no_samples"
                else:
                    if seg_t[0] > t0 + 1e-9:
                        diagnostics.append(Diagnostic(
                            "window_starts_after_sample",
                            "first usable sample is after injection start",
                            power_mat.file, rows_seg[0], t_s=seg_t[0],
                            detail={"injection_s": t0}).to_dict())
                    if nxt is None and seg_t[-1] < power.span[1] - 1e-9:
                        pass
                    n_points = len(seg_t)
                    area = float(_trapz_actual(seg_t, seg_c))
        overlaps = [w["id"] for j, w in enumerate(raw_windows)
                    if j != i and w["t0"] < win["t1"] - 1e-9
                    and win["t0"] < w["t1"] - 1e-9
                    and (w["t0"] > win["t0"] - 1e-9)]
        results.append({
            "injection_id": inj["id"], "order": i, "source_row": inj["row"],
            "status": status,
            "injection_time_s": t0, "injection_time_raw": inj["t_raw"],
            "injection_volume_uL": inj["v"], "injection_volume_raw": inj["v_raw"],
            "window": {"t0_s": t0_actual, "t1_s": t1_actual,
                       "endpoint_policy": policy, "endpoint_reason": end_reason,
                       "requested_t1_s": t1, "clipped_at_s": clipped_at},
            "area_ucal": area, "n_points": n_points,
            "overlapping_injections": sorted(set(overlaps)),
            "diagnostics": diagnostics,
        })

    summary = {
        "n_injections": len(injections),
        "n_integrated": sum(r["status"] == "integrated" for r in results),
        "n_discarded": sum(r["status"] == "discarded_first" for r in results),
        "n_blocked": sum(r["status"].startswith("blocked") for r in results),
        "n_overlapping_windows": sum(bool(r["overlapping_injections"])
                                     for r in results),
        "first_injection_discarded": discard_first,
    }
    inputs = {"power_sha1": power_mat.sha1, "injections_sha1": injection_mat.sha1,
              "baseline_sha1": baseline_mat.sha1 if baseline_mat else None,
              "policy": policy, "policy_options": opts,
              "fixed_ends": fixed_ends, "discard_first": discard_first}
    result_id = "res:int:" + content_hash(
        {"rule": RULE_VERSION, "inputs": inputs})[5:21]
    return {"id": result_id, "kind": "integrations", "rule_version": RULE_VERSION,
            "inputs": inputs, "summary": summary, "injections": results,
            "diagnostics": [d.to_dict() for d in base_diags]}


def _trapz_actual(times: list[float], values: list[float]) -> float:
    import numpy as np
    return float(np.trapz(values, times))


DILUTION_RULE_VERSION = "dilution-v1"


def subtract_dilution(sample_result: dict[str, Any], blank_result: dict[str, Any],
                      cell_volume_uL: float, *,
                      correspondence: str = "order",
                      volume_tolerance: float = 0.02) -> dict[str, Any]:
    """Dilution-heat correction with an explicit correspondence rule.

    ``order``: sample injection *k* corresponds to blank injection *k* (same
    ordinal position in each schedule), provided the injection volumes agree
    within ``volume_tolerance`` (2 % by default). A missing blank counterpart,
    an unmatched status, or a volume mismatch leaves the net heat un-set and
    attaches a positional diagnostic. Nothing is silently zero-filled.
    """
    if correspondence != "order":
        raise SemanticImpasse(
            f"unknown correspondence rule {correspondence!r}",
            [Diagnostic("unknown_correspondence",
                        "dilution correspondence rule not recognised",
                        detail={"rule": correspondence})])
    blank_by_order = {r["order"]: r for r in blank_result["injections"]}
    adjusted: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    n_net = 0
    for r in sample_result["injections"]:
        entry = {k: v for k, v in r.items()}
        entry["dilution_area_ucal"] = None
        entry["net_area_ucal"] = None
        entry["dilution_status"] = "unmatched"
        entry["correspondence_rule"] = correspondence
        entry["corresponds_to"] = None
        if r["status"] != "integrated":
            entry["dilution_status"] = f"sample_{r['status']}"
            diagnostics.append(Diagnostic(
                "sample_not_integrated",
                "sample injection was not integrated; dilution not applied",
                detail={"injection_id": r["injection_id"],
                        "status": r["status"]}).to_dict())
            adjusted.append(entry)
            continue
        b = blank_by_order.get(r["order"])
        if b is None:
            entry["dilution_status"] = "missing_blank_counterpart"
            diagnostics.append(Diagnostic(
                "missing_blank_counterpart",
                f"no blank injection at ordinal {r['order']} ({r['injection_id']})",
                detail={"order": r["order"],
                        "injection_id": r["injection_id"]}).to_dict())
            adjusted.append(entry)
            continue
        entry["corresponds_to"] = {"order": b["order"],
                                   "injection_id": b["injection_id"]}
        if b["status"] != "integrated":
            entry["dilution_status"] = f"blank_{b['status']}"
            diagnostics.append(Diagnostic(
                "blank_not_integrated",
                "matched blank injection was not integrated",
                detail={"injection_id": r["injection_id"],
                        "blank_id": b["injection_id"],
                        "blank_status": b["status"]}).to_dict())
            adjusted.append(entry)
            continue
        rel = abs(b["injection_volume_uL"] - r["injection_volume_uL"]) / r["injection_volume_uL"]
        if rel > volume_tolerance:
            entry["dilution_status"] = "volume_mismatch"
            diagnostics.append(Diagnostic(
                "blank_volume_mismatch",
                "matched blank injection volume outside tolerance",
                detail={"injection_id": r["injection_id"],
                        "sample_volume_uL": r["injection_volume_uL"],
                        "blank_volume_uL": b["injection_volume_uL"],
                        "relative": rel, "tolerance": volume_tolerance}).to_dict())
            adjusted.append(entry)
            continue
        # Scale the per-shot blank heat by the volume ratio, then subtract.
        ratio = r["injection_volume_uL"] / b["injection_volume_uL"]
        dilution = b["area_ucal"] * ratio
        entry["dilution_area_ucal"] = dilution
        entry["net_area_ucal"] = r["area_ucal"] - dilution
        entry["dilution_status"] = "matched"
        n_net += 1
        adjusted.append(entry)

    inputs = {"sample_integrations": sample_result["id"],
              "blank_integrations": blank_result["id"],
              "cell_volume_uL": cell_volume_uL,
              "correspondence": correspondence,
              "volume_tolerance": volume_tolerance}
    result_id = "res:dil:" + content_hash(
        {"rule": DILUTION_RULE_VERSION, "inputs": inputs})[5:21]
    summary = {"n_injections": len(adjusted), "n_net": n_net,
               "n_missing_blank": sum(
                   e["dilution_status"] == "missing_blank_counterpart"
                   for e in adjusted),
               "n_volume_mismatch": sum(
                   e["dilution_status"] == "volume_mismatch" for e in adjusted)}
    return {"id": result_id, "kind": "dilution_adjusted",
            "rule_version": DILUTION_RULE_VERSION, "inputs": inputs,
            "summary": summary, "injections": adjusted,
            "diagnostics": diagnostics}
