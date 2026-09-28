from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any

import numpy as np

from .errors import Diagnostic, FormatCorruptError, SemanticImpossibleError
from .hashing import canonical_hash
from . import units


MATERIAL_KINDS = {
    "power_curve",
    "injection_schedule",
    "concentration",
    "temperature",
    "cell_volume",
    "baseline_curve",
    "blank_curve",
}

ANALYSIS_VERSION = "rules:itc-review/v1"
MODEL_ORDER = ("one_site", "two_site", "dilution")


@dataclass(frozen=True)
class MaterialVersion:
    kind: str
    version: int
    raw: dict[str, Any]
    canonical: dict[str, Any]
    diagnostics: list[dict[str, str]]
    direct_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "raw": self.raw,
            "canonical": self.canonical,
            "diagnostics": self.diagnostics,
            "direct_source": self.direct_source,
            "source_hash": canonical_hash(self.raw),
        }


def _diag(code: str, message: str, location: str, severity: str = "error") -> Diagnostic:
    return Diagnostic(code, message, location, severity)


def require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FormatCorruptError(
            "object expected",
            [_diag("FORMAT_NOT_OBJECT", f"expected object, got {type(value).__name__}", location)],
        )
    return value


def require_array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise FormatCorruptError(
            "array expected",
            [_diag("FORMAT_NOT_ARRAY", f"expected array, got {type(value).__name__}", location)],
        )
    return value


def safe_number(value: Any, location: str, diagnostics: list[Diagnostic]) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        diagnostics.append(_diag("UNCONVERTIBLE_VALUE", f"cannot convert {value!r}", location))
        return None
    if not np.isfinite(result):
        diagnostics.append(_diag("UNCONVERTIBLE_VALUE", f"non-finite value {value!r}", location))
        return None
    return result


def _points(raw: dict[str, Any], location: str, diagnostics: list[Diagnostic]) -> dict[str, Any]:
    points = require_array(raw.get("points"), f"{location}.points")
    if not points:
        raise SemanticImpossibleError(
            "empty series",
            [_diag("EMPTY_SERIES", "series requires at least one point", f"{location}.points")],
        )
    raw_times: list[Any] = []
    raw_values: list[Any] = []
    time_unit = raw.get("time", {}).get("unit", "s") if isinstance(raw.get("time"), dict) else raw.get("time_unit", "s")
    value_kind = raw.get("value_kind", "power")
    value_unit = (
        raw.get("power", {}).get("unit", "µW")
        if value_kind == "power" and isinstance(raw.get("power"), dict)
        else raw.get("value_unit", "µW")
    )
    canonical_time: list[float | None] = []
    canonical_value: list[float | None] = []
    previous: float | None = None
    for index, point in enumerate(points):
        point_location = f"{location}.points[{index}]"
        point_obj = require_object(point, point_location)
        rt = point_obj.get("time")
        rv = point_obj.get("value")
        raw_times.append(rt)
        raw_values.append(rv)
        ct = safe_number(rt, f"{point_location}.time", diagnostics)
        cv = safe_number(rv, f"{point_location}.value", diagnostics)
        if ct is not None:
            try:
                ct = units.time_seconds(rt, time_unit, f"{point_location}.time").canonical_value
            except SemanticImpossibleError as exc:
                diagnostics.extend(exc.diagnostics)
                ct = None
        if cv is not None:
            try:
                if value_kind == "power":
                    cv = units.power_microwatt(rv, value_unit, f"{point_location}.value").canonical_value
                else:
                    cv = units.energy_microjoule(rv, value_unit, f"{point_location}.value").canonical_value
            except SemanticImpossibleError as exc:
                diagnostics.extend(exc.diagnostics)
                cv = None
        if ct is not None and previous is not None and ct <= previous:
            diagnostics.append(_diag("NON_MONOTONIC_TIME", "time coordinates must strictly increase", f"{point_location}.time"))
        if ct is not None:
            previous = ct
        canonical_time.append(ct)
        canonical_value.append(cv)
    return {
        "raw_times": raw_times,
        "raw_values": raw_values,
        "time_unit": time_unit,
        "value_unit": value_unit,
        "time": canonical_time,
        "value": canonical_value,
    }


def normalize_material(kind: str, raw_input: dict[str, Any], direct_source: str, version: int) -> MaterialVersion:
    if kind not in MATERIAL_KINDS:
        raise SemanticImpossibleError(
            "unknown material",
            [_diag("UNKNOWN_MATERIAL", f"unsupported material kind {kind!r}", "material.kind")],
        )
    raw = deepcopy(require_object(raw_input, f"material:{kind}"))
    diagnostics: list[Diagnostic] = []
    canonical: dict[str, Any] = {"kind": kind}
    if kind in {"power_curve", "baseline_curve", "blank_curve"}:
        series = _points(raw, f"material:{kind}", diagnostics)
        canonical["series"] = series
        gaps = require_array(raw.get("gaps", []), f"material:{kind}.gaps")
        gap_out = []
        time_unit = series["time_unit"]
        for idx, gap in enumerate(gaps):
            obj = require_object(gap, f"material:{kind}.gaps[{idx}]")
            rs, re = obj.get("start"), obj.get("end")
            cs = safe_number(rs, f"material:{kind}.gaps[{idx}].start", diagnostics)
            ce = safe_number(re, f"material:{kind}.gaps[{idx}].end", diagnostics)
            if cs is not None:
                cs = units.time_seconds(rs, time_unit, f"material:{kind}.gaps[{idx}].start").canonical_value
            if ce is not None:
                ce = units.time_seconds(re, time_unit, f"material:{kind}.gaps[{idx}].end").canonical_value
            if cs is not None and ce is not None and ce <= cs:
                diagnostics.append(_diag("INVALID_GAP", "gap end must be after start", f"material:{kind}.gaps[{idx}]"))
            gap_out.append({"start": cs, "end": ce, "raw_start": rs, "raw_end": re, "unit": time_unit, "reason": obj.get("reason", "unspecified")})
        canonical["gaps"] = gap_out
        if kind == "blank_curve":
            canonical["blank_identity"] = raw.get("blank_identity", "blank")
    elif kind == "injection_schedule":
        injections = require_array(raw.get("injections"), f"material:{kind}.injections")
        out = []
        prev_time = None
        for idx, item in enumerate(injections):
            obj = require_object(item, f"material:{kind}.injections[{idx}]")
            vol_raw = obj.get("volume", {})
            time_raw = obj.get("time", {})
            vol_obj = require_object(vol_raw, f"material:{kind}.injections[{idx}].volume") if isinstance(vol_raw, dict) else {"value": vol_raw, "unit": "µL"}
            time_obj = require_object(time_raw, f"material:{kind}.injections[{idx}].time") if isinstance(time_raw, dict) else {"value": time_raw, "unit": "s"}
            v = safe_number(vol_obj.get("value"), f"material:{kind}.injections[{idx}].volume.value", diagnostics)
            t = safe_number(time_obj.get("value"), f"material:{kind}.injections[{idx}].time.value", diagnostics)
            cv = units.volume_microlitre(vol_obj.get("value"), vol_obj.get("unit", "µL"), f"material:{kind}.injections[{idx}].volume").canonical_value if v is not None else None
            ct = units.time_seconds(time_obj.get("value"), time_obj.get("unit", "s"), f"material:{kind}.injections[{idx}].time").canonical_value if t is not None else None
            if cv is not None and cv <= 0:
                diagnostics.append(_diag("NON_POSITIVE_VOLUME", "injection volume must be positive", f"material:{kind}.injections[{idx}].volume"))
            if ct is not None and prev_time is not None and ct <= prev_time:
                diagnostics.append(_diag("NON_MONOTONIC_TIME", "injection times must strictly increase", f"material:{kind}.injections[{idx}].time"))
            if ct is not None:
                prev_time = ct
            out.append({
                "index": idx + 1,
                "raw": obj,
                "volume": {"raw": vol_obj, "value": cv, "unit": "µL"},
                "time": {"raw": time_obj, "value": ct, "unit": "s"},
            })
        canonical["injections"] = out
    elif kind in {"concentration", "temperature", "cell_volume"}:
        measurement = require_object(raw.get("measurement"), f"material:{kind}.measurement")
        raw_value = measurement.get("value")
        unit = measurement.get("unit")
        location = f"material:{kind}.measurement"
        numeric = safe_number(raw_value, f"{location}.value", diagnostics)
        try:
            if kind == "concentration":
                converted = units.concentration_mm(raw_value, unit, location)
            elif kind == "temperature":
                converted = units.temperature_celsius(raw_value, unit, location)
            else:
                converted = units.volume_microlitre(raw_value, unit, location)
            canonical.update({"measurement": converted.display()})
        except SemanticImpossibleError as exc:
            diagnostics.extend(exc.diagnostics)
            canonical.update({"measurement": {"raw": measurement, "value": numeric, "unit": unit, "canonical_value": None, "canonical_unit": None}})
    blocking = [d.to_dict() for d in diagnostics if d.severity == "error"]
    canonical["valid"] = not blocking
    canonical["diagnostics"] = blocking
    return MaterialVersion(kind, version, raw, canonical, blocking, direct_source)


def _interpolate_baseline(times: np.ndarray, baseline: dict[str, Any]) -> np.ndarray:
    bt = np.array([v for v in baseline["series"]["time"] if v is not None], dtype=float)
    bv = np.array([v for v, t in zip(baseline["series"]["value"], baseline["series"]["time"]) if t is not None], dtype=float)
    order = np.argsort(bt)
    return np.interp(times, bt[order], bv[order], left=bv[order][0], right=bv[order][-1])


def _interval_gap(window_start: float, window_end: float, gaps: list[dict[str, Any]]) -> dict[str, Any] | None:
    for gap in gaps:
        if gap["start"] is None or gap["end"] is None:
            continue
        if window_start < gap["end"] and gap["start"] < window_end:
            return gap
    return None


def _series_array(material: MaterialVersion) -> tuple[np.ndarray, np.ndarray, list[int]]:
    times: list[float] = []
    values: list[float] = []
    indices: list[int] = []
    for idx, (time_value, power_value) in enumerate(zip(material.canonical["series"]["time"], material.canonical["series"]["value"])):
        if time_value is not None and power_value is not None:
            times.append(time_value)
            values.append(power_value)
            indices.append(idx)
    return np.asarray(times, dtype=float), np.asarray(values, dtype=float), indices


def _integrate_series(
    times: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
) -> dict[str, Any]:
    mask = (times >= start) & (times <= end)
    selected_t = times[mask]
    selected_v = values[mask]
    endpoint_issues: list[str] = []
    if selected_t.size < 2:
        raise SemanticImpossibleError(
            "integration window has fewer than two valid samples",
            [_diag("WINDOW_TOO_SPARSE", f"only {selected_t.size} samples in window", "integration_window")],
        )
    first_gap = float(selected_t[0] - start)
    last_gap = float(end - selected_t[-1])
    if first_gap > 0:
        endpoint_issues.append("start endpoint interpolated because no exact sample")
    if last_gap > 0:
        endpoint_issues.append("end endpoint interpolated because no exact sample")
    if first_gap > 0 or last_gap > 0:
        eval_t = np.concatenate(([start], selected_t, [end]))
        eval_v = np.interp(eval_t, times, values, left=float(values[0]), right=float(values[-1]))
    else:
        eval_t, eval_v = selected_t, selected_v
    heat = float(np.trapz(eval_v, eval_t))
    return {
        "heat_uJ": heat,
        "start_s": start,
        "end_s": end,
        "sample_count": int(eval_t.size),
        "first_sample_s": float(selected_t[0]),
        "last_sample_s": float(selected_t[-1]),
        "endpoint_issues": endpoint_issues,
        "points": [
            {"id": f"sample#{i + 1}", "time_s": float(t), "time_base": "experiment_elapsed_time", "power_uW": float(v), "unit": "µW"}
            for i, (t, v) in enumerate(zip(eval_t, eval_v))
        ],
    }


def integrate_injections(materials: dict[str, MaterialVersion], rules: dict[str, Any] | None = None) -> dict[str, Any]:
    rules = rules or {"version": ANALYSIS_VERSION, "first_injection": "discard", "window_rule": "next_injection"}
    required = ["power_curve", "injection_schedule", "concentration", "temperature", "cell_volume", "baseline_curve"]
    missing = [kind for kind in required if kind not in materials]
    if missing:
        raise SemanticImpossibleError(
            "incomplete inputs",
            [_diag("MISSING_MATERIAL", f"required material {kind} is missing", f"material:{kind}") for kind in missing],
        )
    invalid = [d for material in materials.values() for d in material.diagnostics]
    if invalid:
        raise SemanticImpossibleError("invalid source material", invalid)

    power = materials["power_curve"]
    baseline = materials["baseline_curve"]
    blank = materials.get("blank_curve")
    injections = materials["injection_schedule"].canonical["injections"]
    power_times, power_values, power_indices = _series_array(power)
    corrected_values = power_values - _interpolate_baseline(power_times, baseline)
    blank_times = blank_values = None
    if blank is not None:
        blank_times, raw_blank_values, _ = _series_array(blank)
        blank_values = raw_blank_values - _interpolate_baseline(blank_times, baseline)

    entries: list[dict[str, Any]] = []
    post_window_s = float(rules.get("post_window_s", 170.0))
    blank_tolerance = float(rules.get("blank_volume_tolerance_ratio", 0.02))
    blank_correspondence = rules.get("blank_correspondence", "same_injection_index_and_matching_volume")
    for pos, injection in enumerate(injections):
        index = injection["index"]
        start = float(injection["time"]["value"])
        next_time = float(injections[pos + 1]["time"]["value"]) if pos + 1 < len(injections) else None
        nominal_end = start + post_window_s
        end = min(nominal_end, next_time) if next_time is not None else nominal_end
        next_volume = injections[pos + 1]["volume"]["value"] if pos + 1 < len(injections) else None
        overlap = next_time is not None and nominal_end > next_time
        gap = _interval_gap(start, end, power.canonical.get("gaps", []))
        status = "valid"
        reasons: list[str] = []
        discarded = rules.get("first_injection", "discard") == "discard" and index == 1
        if discarded:
            status = "discarded"
            reasons.append("first injection discarded by configured rule")
        if overlap:
            reasons.append("post-injection peak overlaps next injection; integration window closes at next injection")
        if gap is not None:
            status = "invalid"
            reasons.append("integration window intersects a data gap; no heat was synthesized across it")

        integral: dict[str, Any] | None = None
        corrected: dict[str, Any] | None = None
        blank_match: dict[str, Any] | None = None
        if status != "invalid":
            try:
                integral = _integrate_series(power_times, corrected_values, start, end)
            except SemanticImpossibleError as exc:
                status = "invalid"
                reasons.extend(d.message for d in exc.diagnostics)
                integral = None

        if integral is not None and not discarded:
            if blank is None:
                blank_match = {
                    "status": "missing",
                    "rule": blank_correspondence,
                    "reason": "no versioned blank curve supplied; dilution heat is left uncorrected",
                }
                corrected = {"status": "uncorrected", "heat_uJ": integral["heat_uJ"]}
            else:
                same_index = next((item for item in blank.canonical.get("injections", []) if item["index"] == index), None)
                blank_candidates = blank.canonical.get("injections") or injections
                candidate = same_index or (blank_candidates[pos] if pos < len(blank_candidates) else None)
                candidate_volume = candidate["volume"]["value"] if candidate else None
                volume = injection["volume"]["value"]
                volume_matches = candidate_volume is not None and abs(candidate_volume - volume) <= volume * blank_tolerance
                if same_index is None and blank.canonical.get("injections") or not volume_matches:
                    blank_match = {
                        "status": "missing",
                        "rule": blank_correspondence,
                        "reason": "blank correspondence requires same injection index and matching volume",
                    }
                    corrected = {"status": "uncorrected", "heat_uJ": integral["heat_uJ"]}
                else:
                    b_start = candidate["time"]["value"]
                    b_next = blank_candidates[pos + 1]["time"]["value"] if pos + 1 < len(blank_candidates) else None
                    b_end = min(b_start + post_window_s, b_next) if b_next is not None else b_start + post_window_s
                    b_gap = _interval_gap(b_start, b_end, blank.canonical.get("gaps", []))
                    if b_gap is not None:
                        status = "invalid"
                        reasons.append("matched blank window intersects a data gap")
                    else:
                        blank_integral = _integrate_series(blank_times, blank_values, b_start, b_end)
                        corrected_heat = integral["heat_uJ"] - blank_integral["heat_uJ"]
                        blank_match = {
                            "status": "matched",
                            "rule": blank_correspondence,
                            "blank_index": candidate["index"],
                            "volume_ratio": candidate_volume / volume,
                            "blank_heat_uJ": blank_integral["heat_uJ"],
                        }
                        corrected = {"status": "corrected", "heat_uJ": corrected_heat}

        entries.append({
            "id": f"injection:{index}",
            "stable_identity": f"injection:{index}",
            "index": index,
            "injection_time_s": start,
            "injection_volume_uL": injection["volume"]["value"],
            "next_injection_time_s": next_time,
            "next_injection_volume_uL": next_volume,
            "overlaps_next_peak": bool(overlap),
            "window": {"start_s": start, "end_s": end, "time_base": "experiment_elapsed_time"},
            "data_gap": gap,
            "status": status,
            "reasons": reasons,
            "raw_baseline_subtracted_integral": integral,
            "blank_correspondence": blank_match,
            "dilution_corrected_integral": corrected,
        })
    material_versions = {kind: f"{kind}:v{material.version}" for kind, material in materials.items()}
    input_summary = {
        "material_versions": material_versions,
        "source_hashes": {kind: material.to_dict()["source_hash"] for kind, material in materials.items()},
        "injection_count": len(injections),
        "rules_version": rules.get("version", ANALYSIS_VERSION),
    }
    result_id = "result:integration:" + canonical_hash({"summary": input_summary, "entries": entries})[7:19]
    return {"id": result_id, "kind": "integration", "input_summary": input_summary, "rules_version": rules.get("version", ANALYSIS_VERSION), "entries": entries}


def _nonnegative_lstsq(design: np.ndarray, y: np.ndarray, iterations: int = 300) -> np.ndarray:
    coef = np.zeros(design.shape[1], dtype=float)
    cross = design.T @ design
    rhs = design.T @ y
    for _ in range(iterations):
        previous = coef.copy()
        for j in range(design.shape[1]):
            coordinate = coef[j]
            other = cross[j, j]
            if other == 0:
                continue
            value = (rhs[j] - cross[j] @ coef + other * coordinate) / other
            coef[j] = max(0.0, value)
        if np.allclose(coef, previous, atol=1e-12, rtol=1e-12):
            break
    return coef


def _fit_linear_family(name: str, design: np.ndarray, y: np.ndarray, parameter_count: int, grids: list[dict[str, Any]], observations: int) -> dict[str, Any]:
    coef = _nonnegative_lstsq(design, y)
    predicted = design @ coef
    residual = y - predicted
    rss = float(np.sum(residual * residual))
    params = {"coefficients": [float(x) for x in coef]}
    for grid, value in zip(grids, coef):
        params[grid["role"]] = {"value": float(grid["value"]), "unit": grid["unit"], "grid_index": int(grid["index"])}
    if observations - parameter_count > 0 and rss > 1e-12:
        aic = float(observations * np.log(rss / observations) + 2 * parameter_count)
    else:
        aic = 0.0
    return {
        "model": name,
        "parameter_count": parameter_count,
        "parameters": params,
        "predicted_heat_uJ": [float(x) for x in predicted],
        "residual_uJ": [float(x) for x in residual],
        "rss_uJ2": rss,
        "aic": aic,
        "identifiability": "boundary_or_non_identifiable" if rss <= 1e-12 else "identified_on_fixed_grid",
    }


def fit_binding_models(
    materials: dict[str, MaterialVersion],
    integration: dict[str, Any],
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rules = rules or {"version": ANALYSIS_VERSION}
    entries = [e for e in integration["entries"] if e["status"] == "valid" and e.get("dilution_corrected_integral")]
    if len(entries) < 3:
        raise SemanticImpossibleError(
            "not enough valid injections for model comparison",
            [_diag("INSUFFICIENT_OBSERVATIONS", f"need at least 3 valid corrected injections, found {len(entries)}", "integration.entries")],
        )
    stock = materials["concentration"].canonical["measurement"]["canonical_value"]
    cell = materials["cell_volume"].canonical["measurement"]["canonical_value"]
    temperature = materials["temperature"].canonical["measurement"]["canonical_value"] + 273.15
    macromolecule = float(rules.get("cell_macromolecule_concentration_mM", 0.05))
    cumulative_volume = 0.0
    ligand = []
    volumes = []
    heats = []
    injection_indices = []
    for entry in entries:
        cumulative_volume += float(entry["injection_volume_uL"])
        ligand.append(stock * cumulative_volume / cell)
        volumes.append(float(entry["injection_volume_uL"]))
        heats.append(float(entry["dilution_corrected_integral"]["heat_uJ"]))
        injection_indices.append(int(entry["index"]))
    ligand_arr = np.asarray(ligand, dtype=float)
    volumes_arr = np.asarray(volumes, dtype=float)
    y = np.asarray(heats, dtype=float)
    m_amount = macromolecule * cell

    kd_values = np.geomspace(0.001, 100.0, 25)
    one_candidates: list[dict[str, Any]] = []
    for idx, kd in enumerate(kd_values):
        total_q = 4.184 * m_amount * ligand_arr / (kd + ligand_arr)
        increments = np.diff(np.concatenate(([0.0], total_q)))
        design = increments.reshape(-1, 1)
        one_candidates.append(_fit_linear_family("one_site", design, y, 2, [{"role": "Kd", "value": kd, "unit": "mM", "index": idx}], len(y)))
    one_site = min(one_candidates, key=lambda item: (round(item["rss_uJ2"], 12), item["parameters"]["Kd"]["grid_index"]))
    one_site["parameters"]["n_sites"] = {"value": 1.0, "unit": "sites/cell_component"}
    one_site["parameters"]["delta_H"] = {"value": one_site["parameters"]["coefficients"][0], "unit": "kcal/mol"}

    two_candidates: list[dict[str, Any]] = []
    for i, kd1 in enumerate(kd_values):
        for j, kd2 in enumerate(kd_values):
            if j < i:
                continue
            q1 = 4.184 * m_amount * ligand_arr / (kd1 + ligand_arr)
            q2 = 4.184 * m_amount * ligand_arr / (kd2 + ligand_arr)
            inc1 = np.diff(np.concatenate(([0.0], q1)))
            inc2 = np.diff(np.concatenate(([0.0], q2)))
            design = np.column_stack((inc1, inc2))
            fit = _fit_linear_family(
                "two_site",
                design,
                y,
                4,
                [
                    {"role": "Kd1", "value": kd1, "unit": "mM", "index": i},
                    {"role": "Kd2", "value": kd2, "unit": "mM", "index": j},
                ],
                len(y),
            )
            two_candidates.append(fit)
    two_site = min(two_candidates, key=lambda item: (round(item["rss_uJ2"], 12), item["parameters"]["Kd1"]["grid_index"], item["parameters"]["Kd2"]["grid_index"]))
    two_site["parameters"]["n_sites_1"] = {"value": 1.0, "unit": "sites/cell_component"}
    two_site["parameters"]["n_sites_2"] = {"value": 1.0, "unit": "sites/cell_component"}
    two_site["parameters"]["delta_H_1"] = {"value": two_site["parameters"]["coefficients"][0], "unit": "kcal/mol"}
    two_site["parameters"]["delta_H_2"] = {"value": two_site["parameters"]["coefficients"][1], "unit": "kcal/mol"}

    design_d = np.column_stack((volumes_arr, np.ones_like(volumes_arr)))
    dilution = _fit_linear_family("dilution", design_d, y, 2, [], len(y))
    dilution["parameters"]["dilution_per_volume"] = {"value": dilution["parameters"]["coefficients"][0], "unit": "µJ/µL"}
    dilution["parameters"]["intercept"] = {"value": dilution["parameters"]["coefficients"][1], "unit": "µJ"}

    fits = [one_site, two_site, dilution]
    scale = max(float(np.sum(y * y)), 1e-12)
    best_rss = min(fit["rss_uJ2"] for fit in fits)
    for fit in fits:
        fit["equivalent_best"] = bool(abs(fit["rss_uJ2"] - best_rss) <= max(1e-10, 1e-6 * scale))
        fit["delta_aic"] = float(fit["aic"] - min(other["aic"] for other in fits))
    accepted = [fit for fit in fits if fit["equivalent_best"]]
    accepted.sort(key=lambda item: MODEL_ORDER.index(item["model"]))
    input_summary = {
        "integration_result": integration["id"],
        "material_versions": {
            "concentration": f"concentration:v{materials['concentration'].version}",
            "temperature": f"temperature:v{materials['temperature'].version}",
            "cell_volume": f"cell_volume:v{materials['cell_volume'].version}",
        },
        "rules_version": rules.get("version", ANALYSIS_VERSION),
        "temperature_K": temperature,
        "observations": [{"injection": index, "ligand_mM": ligand_value, "volume_uL": volume, "heat_uJ": heat}
                         for index, ligand_value, volume, heat in zip(injection_indices, ligand, volumes, heats)],
    }
    result_id = "result:binding:" + canonical_hash({"summary": input_summary, "fits": fits})[7:19]
    return {
        "id": result_id,
        "kind": "binding_models",
        "input_summary": input_summary,
        "rules_version": rules.get("version", ANALYSIS_VERSION),
        "fits": {fit["model"]: fit for fit in fits},
        "equivalent_best_models": [fit["model"] for fit in accepted],
        "selection_status": "multiple_equivalent" if len(accepted) > 1 else "unique",
        "residual_reference": {"y": "injection_increment", "x": "injection_index", "heat_unit": "µJ"},
    }
