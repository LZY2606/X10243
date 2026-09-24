"""Pure pipeline functions used by both the service and event replay."""
from __future__ import annotations

import math

from . import binding as B
from . import integration as I
from . import parser as P
from .blank import (MATCH_BY_ORDINAL, BlankParams, subtract)
from .events import IdFactory

CONCENTRATION_KINDS = {
    "macromolecule": "concentration",
    "ligand": "concentration",
    "blank_ligand": "concentration",
}
CONDITION_KINDS = {"temperature": "temperature", "cell_volume": "volume"}


def parse_material(kind: str, role: str, version: str, raw_text: str) -> dict:
    record = {
        "kind": kind, "role": role, "version": version,
        "raw_text": raw_text, "status": P.CORRUPT,
        "diagnostics": [], "summary": {},
    }
    if kind in ("power", "baseline", "blank"):
        ps = P.parse_power(raw_text, kind=kind)
        record["status"] = ps.status
        record["diagnostics"] = [d.to_dict() for d in ps.diagnostics]
        record["summary"] = {
            "n": ps.n, "power_unit": ps.declared_power_unit,
            "median_dt_s": ps.median_dt_s, "gaps": ps.gaps,
        }
    elif kind == "schedule":
        inj, status, diags = P.parse_schedule(raw_text)
        record["status"] = status
        record["diagnostics"] = [d.to_dict() for d in diags]
        record["summary"] = {
            "injections": [injection_dict(x) for x in inj],
        }
    elif kind in ("concentration", "conditions"):
        entries, status, diags = P._parse_keyvalue(raw_text, kind)
        kind_map = (CONCENTRATION_KINDS if kind == "concentration"
                    else CONDITION_KINDS)
        ndiags = P.normalise_keyvalues(entries, kind_map)
        record["status"] = (P.CORRUPT if status == P.CORRUPT
                            else (P.WITH_DIAGNOSTICS if
                                  (diags or ndiags) else P.PARSED))
        record["diagnostics"] = [d.to_dict() for d in diags + ndiags]
        record["summary"] = {"entries": entries}
    else:
        raise ValueError(f"unknown material kind {kind}")
    return record


def injection_dict(x: P.Injection) -> dict:
    return {"index": x.index, "nominal_time_s": _f(x.nominal_time_s),
            "raw_time": x.raw_time, "raw_time_unit": x.raw_time_unit,
            "volume_uL": _f(x.volume_uL), "raw_volume": x.raw_volume,
            "raw_volume_unit": x.raw_volume_unit, "flags": x.flags}


def _f(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def _power_object(record: dict) -> P.PowerSeries:
    return P.parse_power(record["raw_text"], kind=record["kind"])


def _injections(schedule_record: dict) -> list[P.Injection]:
    if schedule_record is None:
        return []
    inj, _, _ = P.parse_schedule(schedule_record["raw_text"])
    return inj


def _entry(records_by_kind, kind, name):
    rec = records_by_kind.get(kind)
    if rec is None:
        return None
    return rec["summary"]["entries"].get(name)


def canonical_value(records_by_kind, kind, name):
    e = _entry(records_by_kind, kind, name)
    if e is None or not e.get("convertible"):
        return None
    return e["canonical_value"]


def run_integration(power_rec, schedule_rec, baseline_rec, params: I.IntegrationParams):
    power = _power_object(power_rec)
    injections = _injections(schedule_rec)
    baseline = _power_object(baseline_rec) if baseline_rec is not None else None
    results = I.integrate(power, injections, params, baseline)
    set_id = IdFactory.integral_id(power_rec["material_id"], params.version)
    for r in results:
        r.stable_id = f"{set_id}:inj{r.injection_index}"
    return results, injections, set_id


def run_blank_integration(blank_rec, params: I.IntegrationParams):
    # blank schedules are committed as kind 'schedule' with role 'blank'
    blank = _power_object(blank_rec["power"])
    injections = _injections(blank_rec["schedule"])
    baseline = (_power_object(blank_rec["baseline"])
                if blank_rec.get("baseline") else None)
    results = I.integrate(blank, injections, params, baseline)
    set_id = IdFactory.blank_id(blank_rec["power"]["material_id"],
                                params.version)
    for r in results:
        r.stable_id = f"{set_id}:inj{r.injection_index}"
    return results, injections, set_id


def run_net(sample_results, blank_results, sample_injections,
            blank_injections, bp: BlankParams):
    nets = subtract(sample_results, blank_results, sample_injections,
                    blank_injections, bp)
    return nets


def run_fits(net_heats, volumes, conditions_rec, concentration_rec,
             discard_first: bool, equiopt_tol: float, max_evals: int,
             net_set_id: str, params_version: str):
    v0 = _kv(conditions_rec, "cell_volume")
    m0 = _kv_conc(concentration_rec, "macromolecule")
    l0 = _kv_conc(concentration_rec, "ligand")
    diags = []
    if v0 is None:
        diags.append({"code": "CELL_VOLUME_UNAVAILABLE",
                      "message": "cell volume missing/unconvertible"})
    if m0 is None:
        diags.append({"code": "MACROMOLECULE_CONC_UNAVAILABLE",
                      "message": "macromolecule concentration unavailable"})
    if l0 is None:
        diags.append({"code": "LIGAND_CONC_UNAVAILABLE",
                      "message": "ligand concentration unavailable"})
    if diags:
        return {"status": "SEMANTIC_UNRESOLVED", "fits": [],
                "diagnostics": diags}
    req = B.FitRequest(net_heats, volumes, v0, m0, l0, discard_first,
                       equiopt_tol, max_evals)
    fits = [B.fit_model(mid, req, params_version, net_set_id)
            for mid in B.MODEL_IDS]
    fits = B.compare(fits, equiopt_tol)
    q_used, v_used = B.prepare_vectors(req)
    return {
        "status": "FITTED",
        "fits": [f.to_dict() for f in fits],
        "diagnostics": [],
        "used_indices_heat": _used_indices(net_heats, discard_first),
        "concentration_version": concentration_rec["version"],
        "concentration_material_id": concentration_rec["material_id"],
        "cell_volume_uL": v0,
        "macromolecule_uM": m0,
        "ligand_uM": l0,
    }


def _used_indices(net_heats, discard_first):
    out = []
    for i, q in enumerate(net_heats):
        if q is None or (isinstance(q, float) and math.isnan(q)):
            continue
        if discard_first and i == 0:
            continue
        out.append(i)
    return out


def _kv(rec, name):
    if rec is None:
        return None
    e = rec["summary"]["entries"].get(name)
    return e["canonical_value"] if e and e.get("convertible") else None


def _kv_conc(rec, name):
    return _kv(rec, name)
