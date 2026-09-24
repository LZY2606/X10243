import csv
import json
from io import StringIO

from .errors import Diagnostic, FormatCorrupt
from .units import canonical


def _series_from_csv(text):
    reader = csv.DictReader(StringIO(text))
    required = {"time", "power"}
    if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
        raise FormatCorrupt("Power series requires 'time' and 'power' columns", "materials.power.csv")
    points = []
    for index, row in enumerate(reader.rows if hasattr(reader, "rows") else list(reader)):
        path = f"row[{index + 2}]"
        try:
            t = float(row["time"])
            p = float(row["power"])
        except (TypeError, ValueError) as exc:
            raise FormatCorrupt("Non-numeric power sample", path) from exc
        points.append({"t": t, "p": p, "raw_time": row["time"], "raw_power": row["power"], "path": path})
    return points


def _series_from_json(payload):
    if not isinstance(payload.get("points"), list) or not payload["points"]:
        raise FormatCorrupt("Power material requires non-empty points", "points")
    points = []
    for index, row in enumerate(payload["points"]):
        path = f"points[{index}]"
        if not isinstance(row, dict) or "time" not in row or "power" not in row:
            raise FormatCorrupt("Power sample requires time and power", path)
        t = canonical(row["time"], "time", f"{path}.time")
        p = canonical(row["power"], "power", f"{path}.power")
        points.append({"t": t, "p": p, "raw_time": str(row["time"]), "raw_power": str(row["power"]), "path": path})
    return points


def parse_power(payload):
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise FormatCorrupt("Power material must be an object", "$")
    time_unit = payload.get("time_unit", "s")
    power_unit = payload.get("power_unit", "µW")
    if "csv" in payload:
        wrapped = {"points": []}
        for row in csv.DictReader(StringIO(payload["csv"])):
            wrapped["points"].append({"time": f"{row['time']} {time_unit}", "power": f"{row['power']} {power_unit}"})
        points = _series_from_json(wrapped)
    else:
        points = _series_from_json(payload)
    times = [p["t"] for p in points]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise FormatCorrupt("Power sample times must be strictly increasing", "points")
    return {
        "kind": "power",
        "raw": payload,
        "time_unit": str(time_unit),
        "power_unit": str(power_unit),
        "coordinate_basis": payload.get("coordinate_basis", "seconds since acquisition start"),
        "points": points,
    }


def parse_baseline(payload):
    if isinstance(payload, str):
        payload = json.loads(payload)
    value = canonical(payload["power"], "power", "power")
    return {
        "kind": "baseline",
        "raw": payload,
        "power": value,
        "coordinate_basis": payload.get("coordinate_basis", "global constant baseline"),
    }


def parse_blank(payload):
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload.get("injections"), list):
        raise FormatCorrupt("Blank control requires injection heats", "injections")
    heats = {}
    for index, row in enumerate(payload["injections"]):
        path = f"injections[{index}]"
        ident = str(row.get("injection_id", ""))
        if not ident:
            raise FormatCorrupt("Blank row requires injection_id", f"{path}.injection_id")
        volume = canonical(row["volume"], "volume", f"{path}.volume")
        heat = canonical(row["heat"], "energy", f"{path}.heat")
        heats[ident] = {"volume": volume, "heat": heat, "path": path, "raw": row}
    return {
        "kind": "blank",
        "raw": payload,
        "rule": payload.get("rule", "linear volume-scaled dilution heat"),
        "coordinate_basis": payload.get("coordinate_basis", "per blank injection; µcal vs µL"),
        "heats": heats,
    }


def parse_concentration(payload):
    if isinstance(payload, str):
        payload = json.loads(payload)
    return {
        "kind": "concentration",
        "raw": payload,
        "macromolecule": canonical(payload["macromolecule"], "concentration", "macromolecule"),
        "ligand_syringe": canonical(payload["ligand_syringe"], "concentration", "ligand_syringe"),
    }


def parse_cell(payload):
    if isinstance(payload, str):
        payload = json.loads(payload)
    return {
        "kind": "cell",
        "raw": payload,
        "temperature": canonical(payload["temperature"], "temperature", "temperature"),
        "cell_volume": canonical(payload["cell_volume"], "volume", "cell_volume"),
    }


def parse_schedule(payload):
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload.get("injections"), list):
        raise FormatCorrupt("Injection schedule requires injections", "injections")
    injections = []
    for index, row in enumerate(payload["injections"]):
        path = f"injections[{index}]"
        injections.append({
            "id": str(row.get("id") or row.get("injection_id") or f"I{index + 1}"),
            "time": canonical(row["time"], "time", f"{path}.time"),
            "volume": canonical(row["volume"], "volume", f"{path}.volume"),
            "integrate_start": canonical(row["integrate_start"], "time", f"{path}.integrate_start"),
            "integrate_end": canonical(row["integrate_end"], "time", f"{path}.integrate_end"),
            "path": path,
            "raw": row,
        })
    return {"kind": "schedule", "raw": payload, "injections": injections}


def materialize_material(kind, payload):
    return {
        "power": parse_power,
        "baseline": parse_baseline,
        "blank": parse_blank,
        "concentration": parse_concentration,
        "cell": parse_cell,
        "schedule": parse_schedule,
    }[kind](payload)
