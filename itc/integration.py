import math

import numpy as np

from .errors import Diagnostic


def _diag(severity, code, message, path, details=None):
    return Diagnostic(severity, code, message, path, details or {}).to_dict()


def adjacent_gaps(points, threshold):
    gaps = []
    for left, right in zip(points, points[1:]):
        if right["t"] - left["t"] > threshold:
            gaps.append({"start": left["t"], "end": right["t"], "path": f"{left['path']}->{right['path']}"})
    return gaps


def clipped_series(points, start, end):
    inside = [(p["t"], p["p"]) for p in points if start <= p["t"] <= end]
    times = np.array([p["t"] for p in points], dtype=float)
    values = np.array([p["p"] for p in points], dtype=float)
    for boundary, label in ((start, "start"), (end, "end")):
        if not any(t == boundary for t, _ in inside):
            idx = np.searchsorted(times, boundary)
            if idx <= 0 or idx >= len(times):
                return None
            inside.append((boundary, float(np.interp(boundary, times, values))))
    return sorted(inside)


def _window_gap(gaps, start, end):
    for gap in gaps:
        if start < gap["end"] and end > gap["start"]:
            return gap
    return None


def _overlap_signal(points, start, baseline, threshold):
    nearest = min(points, key=lambda p: abs(p["t"] - start))
    if abs(nearest["t"] - start) <= 1e-9:
        return abs(nearest["p"] - baseline) > threshold
    return False


def integrate_injection(injection, power, baseline_material, gap_threshold=6.0,
                        overlap_threshold=0.02, endpoint=None):
    start = endpoint["start"] if endpoint else injection["integrate_start"]
    end = endpoint["end"] if endpoint else injection["integrate_end"]
    ident = injection["id"]
    path = f"injections[{ident}]"
    gaps = adjacent_gaps(power["points"], gap_threshold)
    gap = _window_gap(gaps, start, end)
    baseline = baseline_material["power"]
    base_result = {
        "id": ident,
        "window": {"start": start, "end": end, "unit": "s",
                   "basis": "seconds since acquisition start"},
        "baseline": {"value": baseline, "unit": "µW"},
        "parents": {},
    }
    if gap:
        base_result.update({
            "status": "invalid", "valid": False, "raw_heat": None,
            "diagnostics": [_diag("error", "window_crosses_data_gap",
                                  "Integration window must not cross a data gap",
                                  f"{path}.integrate_window", gap)]})
        return base_result
    series = clipped_series(power["points"], start, end)
    if series is None:
        base_result.update({
            "status": "invalid", "valid": False, "raw_heat": None,
            "diagnostics": [_diag("error", "endpoint_unresolved",
                                  "Integration endpoint cannot be resolved against power coordinates",
                                  f"{path}.integrate_end")]})
        return base_result
    sample_times = np.array([row[0] for row in series], dtype=float)
    sample_values = np.array([row[1] for row in series], dtype=float) - baseline
    heat = float(np.trapezoid(sample_values, sample_times)) if hasattr(np, "trapezoid") else float(np.trapz(sample_values, sample_times))
    diagnostics = []
    if _overlap_signal(power["points"], start, baseline, overlap_threshold):
        diagnostics.append(_diag(
            "warning", "overlapping_post_injection_peaks",
            "Post-injection peaks overlap; this window uses the agreed partition boundary",
            f"{path}.integrate_start", {"boundary_time_s": start}))
    if endpoint and (start != injection["integrate_start"] or end != injection["integrate_end"]):
        diagnostics.append(_diag(
            "info", "integration_endpoint_revised",
            "Integration endpoint revised; original schedule remains immutable",
            f"{path}.endpoint",
            {"original": {"start": injection["integrate_start"], "end": injection["integrate_end"]}}))
    base_result.update({"status": "valid", "valid": True, "raw_heat": heat,
                        "raw_heat_unit": "µJ", "diagnostics": diagnostics})
    return base_result


def apply_blank_correction(integration, injection, blank):
    path = f"blank.injections[{injection['id']}]"
    record = blank["heats"].get(injection["id"])
    if record is None:
        corrected = dict(integration)
        corrected.update({
            "status": "unresolved", "valid": False,
            "dilution_heat": None, "net_heat": None,
            "blank_rule": blank["rule"],
            "diagnostics": integration.get("diagnostics", []) + [_diag(
                "error", "blank_correspondence_missing",
                "Dilution heat cannot be subtracted because no blank row corresponds to this injection",
                path, {"injection_id": injection["id"], "expected_rule": blank["rule"]})]})
        return corrected
    expected = record["heat"] * injection["volume"] / record["volume"]
    heat_ucal = integration["raw_heat"] / 4.184
    corrected = dict(integration)
    corrected.update({
        "dilution_heat": expected, "dilution_heat_unit": "µcal",
        "net_heat": heat_ucal - expected, "net_heat_unit": "µcal",
        "blank_rule": blank["rule"], "blank_record_path": record["path"],
        "diagnostics": integration.get("diagnostics", []) + [_diag(
            "info", "blank_correspondence_applied",
            "Dilution heat applied by explicit same-id, volume-scaled rule",
            path, {"observed_blank_volume_µL": record["volume"],
                   "injection_volume_µL": injection["volume"]})]})
    return corrected


def integrate_schedule(state, overrides=None):
    overrides = overrides or {}
    results = []
    power = state["materials"]["power"][-1]
    baseline = state["materials"]["baseline"][-1]
    blank = state["materials"]["blank"][-1]
    schedule = state["materials"]["schedule"][-1]
    first_id = schedule["injections"][0]["id"]
    for injection in schedule["injections"]:
        result = integrate_injection(
            injection, power, baseline, endpoint=overrides.get(injection["id"]))
        if state.get("discard_first") and injection["id"] == first_id:
            result = dict(result)
            result["status"] = "discarded"
            result["valid"] = False
            result["diagnostics"] = result.get("diagnostics", []) + [_diag(
                "info", "first_injection_discarded",
                "First injection is retained but excluded from model fitting",
                f"injections[{injection['id']}].discard_policy")]
        elif result["valid"]:
            result = apply_blank_correction(result, injection, blank)
        results.append(result)
    return results
