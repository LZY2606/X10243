from .integration import integrate_schedule
from .materials import materialize_material
from .models import fit_models


def initial_state():
    return {
        "materials": {"power": [], "baseline": [], "blank": [],
                      "concentration": [], "cell": [], "schedule": []},
        "discard_first": False,
        "integrations": [],
        "models": [],
        "decisions": [],
        "overrides": {},
    }


def _summary_for(kind, material):
    if kind == "power":
        return {"samples": len(material["points"]),
                "time_range": [material["points"][0]["t"], material["points"][-1]["t"]],
                "time_unit": "s", "power_unit": "µW"}
    if kind == "schedule":
        return {"injections": len(material["injections"])}
    if kind == "concentration":
        return {"macromolecule_µM": material["macromolecule"],
                "ligand_syringe_µM": material["ligand_syringe"]}
    if kind == "cell":
        return {"temperature_K": material["temperature"], "cell_volume_µL": material["cell_volume"]}
    if kind == "baseline":
        return {"power_µW": material["power"]}
    if kind == "blank":
        return {"injections": sorted(material["heats"]), "rule": material["rule"]}
    return {}


def _material_parents(state, kind):
    parents = {}
    for other, versions in state["materials"].items():
        if versions:
            parents[other] = versions[-1]["id"]
    return parents


def _validated_fit(state, event_id, payload):
    if not state["integrations"]:
        from .errors import SemanticUnresolvable
        raise SemanticUnresolvable("Model fit requires a prior integration version",
                                   f"{event_id}.parents.integration_version")
    integration_version = state["integrations"][-1]
    fit = fit_models(state, integration_version, payload.get("manual_net_heat_overrides"))
    required = ["power", "schedule", "concentration", "cell", "baseline", "blank"]
    fit["id"] = f"models-{event_id}"
    fit["event_id"] = event_id
    fit["parents"] = {kind: state["materials"][kind][-1]["id"] for kind in required}
    fit["parents"]["integrations"] = integration_version["id"]
    fit["input_summary"] = {
        "valid_points": len(fit["candidates"][0]["points"]) if fit["candidates"] else 0,
        "integration_version": integration_version["id"],
        "concentration_version": fit["parents"]["concentration"],
    }
    fit["stale"] = False
    return fit


def apply_event(state, event):
    event_id = event["id"]
    event_type = event["type"]
    payload = event.get("payload", {})
    next_state = {
        "materials": {key: list(value) for key, value in state["materials"].items()},
        "discard_first": state["discard_first"],
        "integrations": list(state["integrations"]),
        "models": list(state["models"]),
        "decisions": list(state["decisions"]),
        "overrides": dict(state["overrides"]),
    }
    if event_type == "project_created":
        next_state["project"] = {"id": payload.get("project_id"), "title": payload.get("title")}
    elif event_type == "branch_created":
        next_state.setdefault("branches", []).append({
            "id": event.get("branch"),
            "event_id": event_id,
            "parent_event": payload.get("parent_event"),
            "conflict_subject": payload.get("conflict_subject"),
            "reason": payload.get("reason"),
        })
    elif event_type == "material_added":
        kind = payload["kind"]
        version = len(next_state["materials"][kind]) + 1
        material = materialize_material(kind, payload["material"])
        material["id"] = f"mat-{kind}-v{version}"
        material["event_id"] = event_id
        material["version"] = version
        material["revision_reason"] = payload.get("revision_reason", "initial_import")
        material["input_summary"] = _summary_for(kind, material)
        material["parents"] = _material_parents(next_state, kind)
        next_state["materials"][kind].append(material)
        if kind == "concentration" and version > 1:
            for fit in next_state["models"]:
                fit["stale"] = True
        if kind == "schedule" and version > 1:
            for result_set in next_state["integrations"]:
                result_set["superseded_by_local_source_change"] = True
    elif event_type == "first_injection_policy":
        next_state["discard_first"] = bool(payload.get("discard", True))
    elif event_type == "integrations_computed":
        if not all(next_state["materials"][kind] for kind in ("power", "baseline", "blank", "schedule")):
            from .errors import SemanticUnresolvable
            raise SemanticUnresolvable("Integration requires power, schedule, baseline and blank materials",
                                       f"{event_id}.parents")
        results = integrate_schedule(next_state, payload.get("endpoints"))
        version = len(next_state["integrations"]) + 1
        record = {
            "id": f"integrations-v{version}",
            "event_id": event_id,
            "version": version,
            "results": results,
            "parents": {kind: next_state["materials"][kind][-1]["id"]
                        for kind in ("power", "schedule", "baseline", "blank")},
            "input_summary": {
                "injections": len(results),
                "valid": sum(1 for item in results if item.get("valid")),
                "discarded": sum(1 for item in results if item.get("status") == "discarded"),
                "unresolved": sum(1 for item in results if item.get("status") == "unresolved"),
                "invalid": sum(1 for item in results if item.get("status") == "invalid"),
            },
        }
        next_state["integrations"].append(record)
    elif event_type == "models_fitted":
        next_state["models"].append(_validated_fit(next_state, event_id, payload))
    elif event_type == "manual_decision":
        next_state["decisions"].append({
            "id": f"decision-{event_id}",
            "event_id": event_id,
            "action": payload["action"],
            "subject": payload["subject"],
            "model_id": payload.get("model_id"),
            "rationale": payload.get("rationale", ""),
            "parents": payload.get("parents", {}),
        })
    return next_state


def replay(events):
    state = initial_state()
    for event in events:
        state = apply_event(state, event)
    return state
