"""Service-layer orchestration, callable directly by tests.

Every derived result records:
  inputs   - input summary (material ids + versions + blobs),
  params   - decision/parameter version digest,
  parents  - parent dependency result ids (layered provenance).
"""

from __future__ import annotations

from typing import Optional

from . import integration as integ
from .budget import Budget
from .errors import SemanticNoSolution
from .blank import match_blanks, normalize
from .models import fit_all, stable_model_id
from .util import digest


def _decision_digest(decisions):
    basis = [{k: d.get(k) for k in ("kind", "params", "event_id")} for d in decisions]
    return digest(basis, length=16, prefix="dec_")


def _active(decisions, kind):
    items = [d for d in decisions if d.get("kind") == kind]
    if not items:
        return None
    return items[-1]


def _stable_integral_id(sid, mat, decisions):
    basis = ["integrals", sid,
             {k: mat.get(k, {}).get("id") for k in ("power", "schedule", "baseline")},
             _decision_digest([d for d in decisions if d.get("kind") in
                               ("integration_window", "first_injection_discard")])]
    return digest(basis, length=16, prefix="int_")


def _stable_norm_id(sid, integral_id, conc_id, link):
    return digest(["norm", sid, integral_id, conc_id,
                   link.get("blank_series_id") if link else None,
                   link.get("rule") if link else None],
                  length=16, prefix="norm_")


def _get_decisions(state, sid):
    return state["decisions"].get(sid, [])


def compute_series(state, sid, *, budget: Optional[Budget] = None):
    s = state["series"][sid]
    mats = s["materials"]
    needed = ("power", "schedule")
    missing = [k for k in needed if k not in mats]
    result = {"series_id": sid, "label": s.get("label"), "status": "INCOMPLETE",
              "missing": missing, "integrals": None, "normalization": None,
              "models": None, "decisions": _get_decisions(state, sid),
              "material_digests": {k: r["id"] for k, r in mats.items()},
              "material_versions": {k: r["version"] for k, r in mats.items()}}
    if missing:
        result["diagnostics"] = [{
            "code": "INCOMPLETE_INPUTS", "series_id": sid, "missing": missing,
            "message": "inputs incomplete; no default substitution made"}]
        return result

    decisions = _get_decisions(state, sid)
    window = _active(decisions, "integration_window")
    pre = float((window or {}).get("params", {}).get("pre_s", integ.DEFAULT_PRE_S))
    post = float((window or {}).get("params", {}).get("post_s", integ.DEFAULT_POST_S))

    base = mats.get("baseline")
    integ_out = integ.integrate_series(
        mats["power"]["parsed"], mats["schedule"]["parsed"],
        base["parsed"] if base else None, pre_s=pre, post_s=post)
    integrals = integ_out["integrals"]

    discard = _active(decisions, "first_injection_discard")
    if discard is not None:
        integrals = integ.apply_first_discard(
            integrals, {"id": discard.get("event_id"),
                        "count": int(discard.get("params", {}).get("count", 1))})
    else:
        integrals = [{**r, "excluded": False} for r in integrals]

    integral_id = _stable_integral_id(sid, mats, decisions)
    result.update({
        "status": "INTEGRATED",
        "integrals": integrals,
        "gaps": integ_out["gaps"],
        "integral_id": integral_id,
        "windows": {"pre_s": pre, "post_s": post},
    })

    link = state["blank_links"].get(sid)
    blank_view = None
    matched, blank_diags = {}, []
    if link:
        blank_view = compute_blank(state, link["blank_series_id"], pre=pre, post=post)
        matched, blank_diags = match_blanks(
            integrals, blank_view["integrals"], link["rule"])
        for d in blank_diags:
            pass
    if "concentrations" in mats and "conditions" in mats:
        cv = mats["conditions"]["parsed"]["entries"].get("cell_volume", {}).get("value_si")
        norm = normalize(integrals, matched,
                         mats["concentrations"]["parsed"]["entries"], cv,
                         blank_series_id=link["blank_series_id"] if link else None)
        norm["diagnostics"] = list(norm["diagnostics"])
        if link:
            norm["blank_rule"] = link["rule"]
            for d in blank_diags:
                norm["diagnostics"].append(d)
        else:
            norm["blank_rule"] = None
            norm["diagnostics"].append({
                "code": "BLANK_LINK_ABSENT", "series_id": sid,
                "message": "no blank series linked; dilution heats uncorrected"})
        norm_id = _stable_norm_id(sid, integral_id, mats["concentrations"]["id"],
                                  link or {})
        result.update({"status": "NORMALIZED", "normalization": norm,
                       "norm_id": norm_id, "blank": blank_view,
                       "blank_matched_ordinals": sorted(matched)})

        # binding models
        if any(r["usable"] for r in norm["rows"]):
            fit_budget = budget or Budget("model_evaluations", 200_000)
            try:
                fitted = fit_all(norm, fit_budget)
            except SemanticNoSolution as exc:
                result.update({"status": "NO_MODEL_SOLUTION",
                               "models": {"error": exc.to_dict()}})
            else:
                conc_digest = mats["concentrations"]["id"]
                for fit in fitted["fits"]:
                    fit["id"] = stable_model_id(
                        sid, conc_digest, _decision_digest(decisions),
                        fit["model"], fit["params"])
                    fit["parents"] = [norm_id]
                result.update({"status": "MODELED", "models": fitted})
                _annotate_decisions(result, decisions, fitted, norm_id, conc_digest)
    else:
        result["diagnostics"] = result.get("diagnostics", []) + [{
            "code": "INCOMPLETE_INPUTS",
            "missing": [k for k in ("concentrations", "conditions") if k not in mats],
            "message": "concentrations/conditions missing; normalization withheld"}]

    _annotate_staleness(result, decisions, mats)
    return result


def compute_blank(state, blank_sid, *, pre, post):
    s = state["series"][blank_sid]
    out = integ.integrate_series(
        s["materials"]["power"]["parsed"],
        s["materials"]["schedule"]["parsed"],
        s["materials"].get("baseline", {}).get("parsed")
        if "baseline" in s["materials"] else None,
        pre_s=pre, post_s=post)
    return {"series_id": blank_sid, "integrals": out["integrals"], "gaps": out["gaps"]}


def _annotate_decisions(result, decisions, fitted, norm_id, conc_digest):
    accepts = [d for d in decisions if d.get("kind") == "model_accepted"]
    conflict = len({d.get("params", {}).get("model") for d in accepts}) > 1
    result["model_conflict"] = conflict
    if accepts:
        latest = accepts[-1]
        chosen_model = latest.get("params", {}).get("model")
        result["accepted_model"] = chosen_model
        result["accepted_decision_event"] = latest.get("event_id")
        fit_ids = {f["model"]: f["id"] for f in fitted["fits"]}
        result["accepted_stale"] = fit_ids.get(chosen_model) is None
        # recompute the accepted model's id as it would have been at decision time
        params_then = latest.get("params", {}).get("params_snapshot")
        if params_then is not None:
            result["accepted_result_id_at_time"] = stable_model_id(
                result["series_id"],
                latest.get("params", {}).get("conc_digest", conc_digest),
                _decision_digest(decisions[: decisions.index(latest)]),
                chosen_model, params_then)


def _annotate_staleness(result, decisions, mats):
    # A concentration revision after an accepted model marks the model stale:
    # original integrals are NOT affected; only normalized/model layers are.
    conc = mats.get("concentrations")
    if conc and conc["version"] > 1:
        accepted = [d for d in decisions if d.get("kind") == "model_accepted"]
        if accepted and conc.get("seq", 10**9) > max(d.get("seq", 0) for d in accepted):
            result["stale_layers"] = ["normalization", "models"]
            result["stale_reason"] = ("concentration revised after model acceptance: "
                                      "model stale; original integrals unchanged")
        elif accepted:
            result["stale_layers"] = ["models"]
            result["stale_reason"] = ("concentration revised: accepted model computed "
                                      "from an older concentration version")
