"""Read-side assembly: reconstruct the review view as of any event.
Staleness is derived, never stored: a result is stale when any material it
depends on has a newer version, or any parent result is stale/superseded.
Going back to an old event restores the dependency versions and the visible
conclusions of that moment."""

from .integrate import gaps as curve_gaps


def compute_flags(store, as_of=None):
    mats = store.current_materials(as_of)
    results = store.results(as_of)
    seq_by_event = {e["id"]: e["seq"] for e in store.events(as_of)}
    superseded = set()
    by_kind = {}
    for r in results:
        by_kind.setdefault(r["kind"], []).append(r)
    for rs in by_kind.values():
        newest = max(rs, key=lambda r: seq_by_event[r["event_id"]])
        for r in rs:
            if r["id"] != newest["id"]:
                superseded.add(r["id"])
    stale = set()
    changed = True
    while changed:
        changed = False
        for r in results:
            if r["id"] in stale:
                continue
            for mid, ver in r["deps"].get("materials", {}).items():
                if mid in mats and mats[mid]["version"] > ver:
                    stale.add(r["id"])
                    changed = True
            for rid in r["deps"].get("results", []):
                if rid in stale or rid in superseded:
                    stale.add(r["id"])
                    changed = True
    return stale, superseded


def _latest(store, kind, as_of):
    seq_by_event = {e["id"]: e["seq"] for e in store.events(as_of)}
    rs = [r for r in store.results(as_of) if r["kind"] == kind]
    return max(rs, key=lambda r: seq_by_event[r["event_id"]], default=None)


def build_state(store, as_of=None, branch=None):
    as_of = as_of if as_of is not None else store.latest_seq()
    stale, superseded = compute_flags(store, as_of)
    events = store.events(as_of)
    materials = sorted(store.current_materials(as_of).values(),
                       key=lambda m: m["id"])
    integration = _latest(store, "integration", as_of)
    correction = _latest(store, "blank_correction", as_of)
    fit = _latest(store, "model_fit", as_of)

    injections = []
    if integration and integration["payload"].get("injections"):
        corr_by_index = {}
        if correction and correction["payload"].get("injections"):
            corr_by_index = {c["index"]: c
                             for c in correction["payload"]["injections"]}
        for rec in integration["payload"]["injections"]:
            item = dict(rec)
            item["id"] = f"inj-{rec['index']:02d}"
            corr = corr_by_index.get(rec["index"])
            if corr:
                item["blank_ucal"] = corr.get("blank_ucal")
                item["corrected_ucal"] = corr.get("corrected_ucal")
                item["blank_applied"] = corr.get("blank_applied", False)
                item["diagnostics"] = rec["diagnostics"] + [
                    d for d in corr.get("diagnostics", [])
                    if d not in rec["diagnostics"]]
            injections.append(item)

    def result_view(r):
        if r is None:
            return None
        return {"id": r["id"], "kind": r["kind"], "status": r["status"],
                "event_id": r["event_id"], "params": r["params"],
                "deps": r["deps"], "stale": r["id"] in stale,
                "superseded": r["id"] in superseded,
                "diagnostics": r["diagnostics"], "payload": r["payload"]}

    fit_view = result_view(fit)
    if fit_view and fit_view["payload"].get("candidates"):
        decided = {d["target"]: d for d in store.decisions(as_of, branch)}
        for cand in fit_view["payload"]["candidates"]:
            cand["decision"] = decided.get(cand["id"])

    all_results = [result_view(r) for r in store.results(as_of)]
    diagnostics = []
    for r in (integration, correction, fit):
        if r:
            diagnostics.extend(r["diagnostics"])
    for inj in injections:
        diagnostics.extend(inj.get("diagnostics", []))

    return {"as_of": as_of, "latest_seq": store.latest_seq(),
            "events": events, "materials": materials,
            "injections": injections,
            "integration": result_view(integration),
            "blank_correction": result_view(correction),
            "fit": fit_view, "results": all_results,
            "decisions": store.decisions(as_of, branch),
            "diagnostics": diagnostics}


def build_curve(store, as_of=None):
    mats = store.current_materials(as_of)
    if "mat-power" not in mats:
        return None
    power = mats["mat-power"]["payload"]
    integration = _latest(store, "integration", as_of)
    windows = []
    if integration and integration["payload"].get("injections"):
        windows = [{"index": r["index"], "window": r["window"],
                    "status": r["status"], "excluded": r["excluded"]}
                   for r in integration["payload"]["injections"]]
    return {"time_unit": power.get("time_unit"), "power_unit": power.get("power_unit"),
            "time_base": power.get("time_base"),
            "power_version": mats["mat-power"]["version"],
            "segments": power["segments"], "gaps": curve_gaps(power),
            "windows": windows}
