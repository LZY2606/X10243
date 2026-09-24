"""Immutable materials, versioned separately per kind.

Kinds: power_trace / injection_schedule / concentrations / temperature /
cell_volume / baseline / blank_curve.  ``normalize`` derives the canonical
layer from the raw import; the raw layer is stored verbatim by the store.
"""

from . import units

KINDS = [
    "power_trace", "injection_schedule", "concentrations",
    "temperature", "cell_volume", "baseline", "blank_curve",
]


def normalize(kind, raw):
    """Return (norm_dict_or_None, diagnostics)."""
    fn = {
        "power_trace": _power,
        "injection_schedule": _schedule,
        "concentrations": _concentrations,
        "temperature": _temperature,
        "cell_volume": _cell_volume,
        "baseline": _baseline,
        "blank_curve": _blank,
    }.get(kind)
    if fn is None:
        return None, [units.diagnostic(kind, "kind_unknown", f"unknown material kind {kind!r}")]
    try:
        return fn(raw)
    except (KeyError, TypeError, IndexError) as exc:
        return None, [units.diagnostic(kind, "field_missing", f"missing or malformed field: {exc}")]


def _power(raw):
    t, d1 = units.convert_series(raw["time"], raw.get("time_unit", "s"), "time", "power_trace.time")
    p, d2 = units.convert_series(p_raw := raw["power"], raw.get("power_unit", "uJ/s"), "power", "power_trace.power")
    diags = d1 + d2
    if len(t) != len(p_raw):
        diags.append(units.diagnostic("power_trace", "length_mismatch",
                                      f"time has {len(t)} samples, power has {len(p_raw)}"))
        return None, diags
    gaps = [[float(g[0]), float(g[1])] for g in raw.get("gaps", [])]
    return {"t": t, "p": p, "gaps": gaps, "time_unit": "s", "power_unit": "uJ/s"}, diags


def _schedule(raw):
    t, d1 = units.convert_series(raw["time"], raw.get("time_unit", "s"), "time", "injection_schedule.time")
    v, d2 = units.convert_series(raw["volume"], raw.get("volume_unit", "uL"), "volume", "injection_schedule.volume")
    diags = d1 + d2
    if len(t) != len(v):
        diags.append(units.diagnostic("injection_schedule", "length_mismatch",
                                      f"time has {len(t)} entries, volume has {len(v)}"))
        return None, diags
    return {"t": t, "v_uL": v, "time_unit": "s", "volume_unit": "uL"}, diags


def _concentrations(raw):
    unit = raw.get("unit", "mM")
    sy, d1 = units.convert_scalar(raw["syringe"], unit, "concentration", "concentrations.syringe")
    ce, d2 = units.convert_scalar(raw["cell"], unit, "concentration", "concentrations.cell")
    diags = [d for d in (d1, d2) if d]
    norm = {"syringe_mM": sy, "cell_mM": ce, "unit": "mM"}
    return (None if diags else norm), diags if diags else []


def _temperature(raw):
    val, d = units.convert_scalar(raw["value"], raw.get("unit", "K"), "temperature", "temperature.value")
    diags = [d] if d else []
    return (None if d else {"K": val, "unit": "K"}), diags


def _cell_volume(raw):
    val, d = units.convert_scalar(raw["value"], raw.get("unit", "mL"), "cell_volume", "cell_volume.value")
    diags = [d] if d else []
    return (None if d else {"mL": val, "unit": "mL"}), diags


def _baseline(raw):
    mode = raw.get("mode", "constant")
    if mode == "constant":
        val, d = units.convert_scalar(raw["value"], raw.get("power_unit", "uJ/s"), "power", "baseline.value")
        diags = [d] if d else []
        return (None if d else {"mode": "constant", "c": val, "power_unit": "uJ/s"}), diags
    t, d1 = units.convert_series(raw["time"], raw.get("time_unit", "s"), "time", "baseline.time")
    p, d2 = units.convert_series(raw["power"], raw.get("power_unit", "uJ/s"), "power", "baseline.power")
    return {"mode": "curve", "t": t, "p": p, "power_unit": "uJ/s"}, d1 + d2


def _blank(raw):
    heats, diags = units.convert_series(raw["heats"], raw.get("unit", "uJ"), "energy", "blank_curve.heats")
    return {"heats_uJ": heats, "unit": "uJ"}, diags
