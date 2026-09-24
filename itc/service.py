"""Orchestration: immutable materials in, versioned derived results out.

Layering rules:
  * materials are immutable and separately versioned per kind;
  * every derived result records input digest, parameter version and parents;
  * incomplete input yields located diagnostics, never fabricated defaults;
  * concentration revision stales fits (they depend on it) but never the
    original integrals (they do not).
"""
from __future__ import annotations

from .binding import PARAM_VERSION as FIT_PV
from .binding import fit_models
from .domain import RESULT_OK, RESULT_SEMANTIC, convert, diag, digest
from .integrate import DEFAULT_CONFIG, PARAM_VERSION as INTEG_PV, integrate

DIL_PV = "dilution/1"


def _norm_power(payload, path):
    diags = []
    tu = payload.get("time_unit", "s")
    pu = payload.get("power_unit", "uW")
    samples = []
    for i, s in enumerate(payload.get("samples", [])):
        t = convert(s.get("t"), tu, "s")
        p = convert(s.get("p"), pu, "uW")
        if t is None or p is None:
            diags.append(diag("unconvertible-unit",
                              f"sample unit not convertible ({tu}/{pu}); raw sample kept in material",
                              f"{path}/samples/{i}"))
            continue
        samples.append({"t_s": t, "power_uW": p})
    return {"samples": samples, "time_base": payload.get("time_base", "elapsed_s")}, diags


def _norm_injections(payload, path):
    diags, out = [], []
    for i, inj in enumerate(payload.get("injections", [])):
        entry = {"id": inj.get("id", f"I{i+1:02d}"), "index": i,
                 "time_s": None, "volume_uL": None, "diagnostics": []}
        tv = inj.get("time", {})
        t = convert(tv.get("value"), tv.get("unit"), "s")
        if t is None:
            d = diag("unconvertible-unit", "injection time not convertible to seconds",
                     f"{path}/injections/{i}/time")
            entry["diagnostics"].append(d)
            diags.append(d)
        else:
            entry["time_s"] = t
        vv = inj.get("volume", {})
        v = convert(vv.get("value"), vv.get("unit"), "uL")
        if v is None:
            d = diag("unconvertible-unit",
                     f"injection volume unit '{vv.get('unit')}' not convertible; raw value kept in place",
                     f"{path}/injections/{i}/volume")
            entry["diagnostics"].append(d)
            entry["volume_raw"] = vv
            diags.append(d)
        else:
            entry["volume_uL"] = v
        out.append(entry)
    return out, diags


class Session:
    def __init__(self, store):
        self.store = store

    # ---------------- materials ----------------
    def import_material(self, kind, payload, note=None):
        mid = f"mat:{kind}:v{self.store.next_version(kind)}"
        ev = self.store.append_event(
            "material.import", {"material": mid, "kind": kind, "note": note})
        self.store.add_material(kind, payload, ev["seq"])
        return mid

    # ---------------- decisions (mutable layer, branchable) ----------------
    def decide(self, action, target, rationale, branch="main", reviewer="reviewer"):
        base = target.split("#")[0]
        parents = [base] if base.startswith(("res:", "mat:")) else []
        return self.store.append_event(
            f"decision.{action}",
            {"target": target, "rationale": rationale, "branch": branch, "reviewer": reviewer},
            parents)

    # ---------------- derived computations ----------------
    def _record(self, kind, status, ind, pv, parents, payload):
        indigest = ind if isinstance(ind, str) else digest(ind)
        rid = f"res:{kind}:{indigest[:12]}"
        ev = self.store.append_event(
            f"compute.{kind}", {"result": rid, "status": status, "param_version": pv}, parents)
        self.store.put_result(kind, status, indigest, pv, parents, payload, ev["seq"], rid=rid)
        return rid

    def _latest(self, kind):
        rs = self.store.results(kind=kind)
        return rs[-1] if rs else None

    def run_integration(self, config=None, budget=10000):
        mats = self.store.materials()
        cfg = dict(DEFAULT_CONFIG, **(config or {}))
        missing = [k for k in ("power_trace", "injection_schedule", "baseline") if k not in mats]
        if missing:
            return self._record("integration", RESULT_SEMANTIC, {"missing": missing, "cfg": cfg},
                                INTEG_PV, [],
                                {"diagnostics": [diag("missing-input",
                                                      f"required material '{k}' not imported",
                                                      f"materials/{k}") for k in missing],
                                 "config": cfg})
        trace, d1 = _norm_power(mats["power_trace"]["payload"], "power_trace")
        base, d2 = _norm_power(mats["baseline"]["payload"], "baseline")
        injs, d3 = _norm_injections(mats["injection_schedule"]["payload"], "injection_schedule")
        status, payload = integrate(trace, injs, base, cfg, budget)
        parents = [mats[k]["id"] for k in ("power_trace", "injection_schedule", "baseline")]
        ind = digest({"trace": trace, "base": base, "injs": injs, "cfg": cfg})
        if status != RESULT_OK:
            payload = {"config": cfg, "diagnostics": d1 + d2 + d3}
        else:
            for entry, inj in zip(payload["injections"], injs):
                entry["volume_uL"] = inj["volume_uL"]
                entry["diagnostics"].extend(inj["diagnostics"])
            payload["diagnostics"] = d1 + d2 + d3 + payload["diagnostics"]
        return self._record("integration", status, ind, INTEG_PV, parents, payload)

    def run_dilution(self, rule="by_index", budget=10000):
        mats = self.store.materials()
        integ = self._latest("integration")
        if integ is None or integ["status"] != RESULT_OK:
            return self._record("dilution", RESULT_SEMANTIC, {"why": "no-ok-integration"},
                                DIL_PV, [],
                                {"diagnostics": [diag("missing-input",
                                                      "no successful integration result",
                                                      "results/integration")]})
        if "blank_trace" not in mats:
            return self._record("dilution", RESULT_SEMANTIC, {"why": "no-blank"},
                                DIL_PV, [integ["id"]],
                                {"diagnostics": [diag("missing-input",
                                                      "blank control trace not imported",
                                                      "materials/blank_trace")]})
        blank = mats["blank_trace"]["payload"]
        btrace, d1 = _norm_power(blank.get("trace", {}), "blank_trace/trace")
        binjs, d2 = _norm_injections(blank.get("schedule", {}), "blank_trace/schedule")
        base, d3 = _norm_power(mats["baseline"]["payload"], "baseline")
        cfg = integ["payload"]["config"]
        bstatus, bpayload = integrate(btrace, binjs, base, cfg, budget)
        parents = [integ["id"], mats["blank_trace"]["id"], mats["baseline"]["id"]]
        if bstatus != RESULT_OK:
            return self._record("dilution", bstatus, {"blank": bstatus}, DIL_PV, parents,
                                {"diagnostics": [diag("blank-integration-failed", bstatus,
                                                      "blank_trace")]})
        blank_heats = {e["index"]: e["heat_uJ"] for e in bpayload["injections"]
                       if e["status"] in (RESULT_OK, "truncated_gap")}
        entries, diags = [], list(d1 + d2 + d3)
        for e in integ["payload"]["injections"]:
            out = {"injection_id": e["injection_id"], "index": e["index"],
                   "volume_uL": e.get("volume_uL"), "sample_heat_uJ": e["heat_uJ"],
                   "blank_heat_uJ": None, "corrected_uJ": None,
                   "status": e["status"], "diagnostics": list(e["diagnostics"])}
            if e["status"] in (RESULT_OK, "truncated_gap"):
                bh = blank_heats.get(e["index"])
                if bh is None:
                    out["status"] = "missing_blank"
                    d = diag("blank-correspondence-missing",
                             f"no blank injection corresponds under rule '{rule}'",
                             f"dilution/injections/{e['index']}")
                    out["diagnostics"].append(d)
                    diags.append(d)
                else:
                    out["blank_heat_uJ"] = bh
                    out["corrected_uJ"] = e["heat_uJ"] - bh
            entries.append(out)
        status = "partial" if any(e["status"] == "missing_blank" for e in entries) else RESULT_OK
        payload = {"rule": rule, "injections": entries, "diagnostics": diags}
        ind = digest({"integ": integ["input_digest"],
                      "blank": mats["blank_trace"]["id"], "rule": rule})
        return self._record("dilution", status, ind, DIL_PV, parents, payload)

    def run_fit(self, budget=100000):
        mats = self.store.materials()
        dil = self._latest("dilution")
        missing = [k for k in ("concentrations", "cell_volume", "temperature") if k not in mats]
        if dil is None or dil["status"] not in (RESULT_OK, "partial") or missing:
            diags = [diag("missing-input", f"required material '{k}' not imported",
                          f"materials/{k}") for k in missing]
            if dil is None or dil["status"] not in (RESULT_OK, "partial"):
                diags.append(diag("missing-input", "no usable dilution result",
                                  "results/dilution"))
            return self._record("fit", RESULT_SEMANTIC, {"missing": missing}, FIT_PV, [],
                                {"diagnostics": diags})
        conc = mats["concentrations"]["payload"]
        Mt = convert(conc["cell"]["value"], conc["cell"]["unit"], "mM")
        Cs = convert(conc["syringe"]["value"], conc["syringe"]["unit"], "mM")
        V0 = convert(mats["cell_volume"]["payload"]["value"],
                     mats["cell_volume"]["payload"]["unit"], "uL")
        T = convert(mats["temperature"]["payload"]["value"],
                    mats["temperature"]["payload"]["unit"], "K")
        parents = [dil["id"], mats["concentrations"]["id"],
                   mats["cell_volume"]["id"], mats["temperature"]["id"]]
        if None in (Mt, Cs, V0, T):
            return self._record("fit", RESULT_SEMANTIC, {"why": "unconvertible-constants"},
                                FIT_PV, parents,
                                {"diagnostics": [diag("unconvertible-unit",
                                                      "concentration/volume/temperature not convertible",
                                                      "materials")]})
        diags, points, cumv = [], [], 0.0
        for e in dil["payload"]["injections"]:
            v = e["volume_uL"]
            if v is None:
                diags.append(diag("ligand-concentration-uncertain",
                                  "injection volume unconvertible; point excluded, later ligand concentrations approximate",
                                  f"fit/points/{e['injection_id']}"))
                continue
            cumv += v
            if e["status"] != RESULT_OK or e["corrected_uJ"] is None:
                continue
            points.append({"injection_id": e["injection_id"],
                           "heat_uJ": e["corrected_uJ"],
                           "ligand_mM": Cs * cumv / V0})
        if not points:
            return self._record("fit", RESULT_SEMANTIC, {"why": "no-valid-points"},
                                FIT_PV, parents,
                                {"diagnostics": [diag("no-valid-points",
                                                      "no usable injection heats for fitting",
                                                      "fit/points")]})
        status, fpayload = fit_models([p["heat_uJ"] for p in points], Mt,
                                      [p["ligand_mM"] for p in points], V0, budget)
        if status != RESULT_OK:
            fpayload = {}
        fpayload["points"] = points
        fpayload["temperature_K"] = T
        fpayload["cell_mM"] = Mt
        fpayload["diagnostics"] = diags
        ind = digest({"dil": dil["input_digest"], "conc": mats["concentrations"]["id"],
                      "V0": V0, "T": T})
        return self._record("fit", status, ind, FIT_PV, parents, fpayload)

    # ---------------- views (time travel + staleness) ----------------
    def view(self, seq=None):
        mats = self.store.materials(upto_seq=seq)
        results = self.store.results(upto_seq=seq)
        events = self.store.events(upto_seq=seq)
        latest = {k: m["version"] for k, m in mats.items()}
        stale = set()
        changed = True
        while changed:
            changed = False
            for r in results:
                if r["id"] in stale:
                    continue
                for p in r["parents"]:
                    is_stale = False
                    if p.startswith("mat:"):
                        _, kind, ver = p.split(":")
                        is_stale = int(ver[1:]) < latest.get(kind, 0)
                    elif p.startswith("res:"):
                        is_stale = p in stale
                    if is_stale:
                        stale.add(r["id"])
                        changed = True
                        break
        for r in results:
            r["freshness"] = "stale" if r["id"] in stale else "current"
        decisions = [e for e in events if e["kind"].startswith("decision.")]
        return {"materials": mats, "results": results, "events": events,
                "decisions": decisions, "stale": sorted(stale),
                "recovered_writes": list(self.store.recovered),
                "latest_seq": events[-1]["seq"] if events else 0}
