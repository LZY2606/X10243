import json
from pathlib import Path


BASELINE = 10.0
TARGET_RAW = -3.3472
BOUNDARY_RESIDUAL = -0.10


def q(value, unit):
    return {"value": round(value, 6), "unit": unit, "raw": f"{round(value, 6)} {unit}"}


def point(time, value):
    return {"time": q(time, "s"), "power": q(value, "µW")}


def set_value(samples, time_s, excess):
    samples[time_s] = BASELINE + excess


def ramp(samples, start, end, start_excess, end_excess, step=2.0):
    count = int(round((end - start) / step))
    for index in range(count + 1):
        fraction = index / count if count else 1.0
        set_value(samples, start + index * step,
                  start_excess + (end_excess - start_excess) * fraction)


def first_pulse(samples):
    ramp(samples, 0.0, 2.0, 0.0, 0.0)
    area = -3.2
    peak = 2.0 * area / 16.0
    ramp(samples, 2.0, 10.0, 0.0, peak)
    ramp(samples, 10.0, 18.0, peak, 0.0)
    ramp(samples, 18.0, 20.0, 0.0, 0.0)


def full_pulse(samples, start, incoming, outgoing, target):
    ramp(samples, start, start + 2.0, incoming, 0.0)
    ramp(samples, start + 2.0, start + 4.0, 0.0, 0.0)
    peak = 2.0 * (target - incoming - outgoing) / 32.0
    ramp(samples, start + 4.0, start + 20.0, 0.0, peak)
    ramp(samples, start + 20.0, start + 36.0, peak, 0.0)
    ramp(samples, start + 36.0, start + 38.0, 0.0, 0.0)
    ramp(samples, start + 38.0, start + 40.0, 0.0, outgoing)


def truncated_pulse(samples, start, incoming, target):
    ramp(samples, start, start + 2.0, incoming, 0.0)
    ramp(samples, start + 2.0, start + 4.0, 0.0, 0.0)
    peak = 2.0 * (target - incoming) / 26.0
    ramp(samples, start + 4.0, start + 16.0, 0.0, peak)
    ramp(samples, start + 16.0, start + 30.0, peak, 0.0)


def build_power():
    samples = {}
    first_pulse(samples)
    full_pulse(samples, 20.0, 0.0, BOUNDARY_RESIDUAL, TARGET_RAW)
    full_pulse(samples, 60.0, BOUNDARY_RESIDUAL, BOUNDARY_RESIDUAL, TARGET_RAW)
    full_pulse(samples, 100.0, BOUNDARY_RESIDUAL, BOUNDARY_RESIDUAL, TARGET_RAW)
    # I4 tail carries into I5; original I5 window crosses the 170-182 s gap.
    truncated_pulse(samples, 140.0, BOUNDARY_RESIDUAL, TARGET_RAW)
    # Acquisition resumes at 182 s; no synthetic interpolation is stored.
    full_pulse(samples, 182.0, 0.0, 0.0, TARGET_RAW)
    set_value(samples, 224.0, 0.0)
    points = [point(t, samples[t]) for t in sorted(samples)]
    return {
        "time_unit": "s",
        "power_unit": "µW",
        "coordinate_basis": "seconds since acquisition start",
        "points": points,
    }


def build_schedule():
    starts = [0, 20, 60, 100, 140, 182]
    ends = [20, 60, 100, 140, 180, 222]
    return {"injections": [
        {"id": f"I{i + 1}", "time": q(starts[i], "s"), "volume": q(2.0, "µL"),
         "integrate_start": q(starts[i], "s"), "integrate_end": q(ends[i], "s")}
        for i in range(6)
    ]}


def build_blank():
    return {
        "rule": "same injection_id; blank heat scaled linearly by injection volume",
        "coordinate_basis": "per matched blank injection; µcal heat and µL volume",
        "injections": [
            {"injection_id": "I1", "volume": q(2.0, "µL"), "heat": q(0.20, "µcal")},
            {"injection_id": "I2", "volume": q(2.0, "µL"), "heat": q(0.20, "µcal")},
            {"injection_id": "I3", "volume": q(2.0, "µL"), "heat": q(0.20, "µcal")},
            {"injection_id": "I5", "volume": q(2.0, "µL"), "heat": q(0.20, "µcal")},
            {"injection_id": "I6", "volume": q(2.0, "µL"), "heat": q(0.20, "µcal")},
        ],
    }


def concentration(ligand):
    return {"macromolecule": q(10.0, "µM"), "ligand_syringe": q(ligand, "µM")}


def material_event(event_id, kind, material, reason="initial_import"):
    return {"id": event_id, "type": "material_added", "branch": "main",
            "payload": {"kind": kind, "material": material, "revision_reason": reason}}


def build_events(power, schedule, blank):
    events = [
        {"id": "e01", "type": "project_created", "branch": "main",
         "payload": {"project_id": "itc-titration-heat-review",
                     "title": "滴定热谱：ITC 注射积分、稀释热与绑定模型审议"}},
        material_event("e02", "power", power),
        material_event("e03", "schedule", schedule),
        material_event("e04", "concentration", concentration(100.0)),
        material_event("e05", "cell", {"temperature": q(25.0, "°C"),
                                       "cell_volume": q(200.0, "µL")}),
        material_event("e06", "baseline", {"power": q(BASELINE, "µW"),
                                           "coordinate_basis": "acquisition-wide constant baseline"}),
        material_event("e07", "blank", blank),
        {"id": "e08", "type": "first_injection_policy", "branch": "main",
         "payload": {"discard": True, "rule": "discard first injection before fitting; retain raw integration"}},
        {"id": "e09", "type": "integrations_computed", "branch": "main",
         "payload": {"note": "initial review exposes overlap, first discard, missing blank, and I5 data gap"}},
        {"id": "e10", "type": "models_fitted", "branch": "main",
         "payload": {"note": "three valid corrected injections yield multiple legal candidates"}},
        material_event("e11", "concentration", concentration(1000.0),
                       "syringe stock concentration corrected from laboratory notebook revision"),
        {"id": "e12", "type": "integrations_computed", "branch": "main",
         "payload": {"note": "integrations recalculated after concentration version; raw heats remain unchanged",
                     "endpoints": {"I5": {"start": 140.0, "end": 170.0}}}},
        {"id": "e13", "type": "models_fitted", "branch": "main",
         "payload": {"note": "I5 endpoint now stops before the acquisition gap; prior model is stale but retained"}},
        {"id": "e14", "type": "branch_created", "branch": "alt-dilution",
         "payload": {"parent_event": "e13", "conflict_subject": "model_candidate",
                     "reason": "preserve a conflicting human decision without replacing the main decision"}},
        {"id": "e15", "type": "manual_decision", "branch": "main",
         "payload": {"action": "accept", "subject": "model_candidate", "model_id": "one_site",
                     "rationale": "accepted on main review despite equivalent fit set",
                     "parents": {"models": "models-e13"}}},
        {"id": "e16", "type": "manual_decision", "branch": "main",
         "payload": {"action": "reject", "subject": "model_candidate", "model_id": "multi_site",
                     "rationale": "extra sites are not uniquely supported; rejected candidate remains readable",
                     "parents": {"models": "models-e13"}}},
        {"id": "e17", "type": "manual_decision", "branch": "alt-dilution",
         "payload": {"action": "accept", "subject": "model_candidate", "model_id": "dilution",
                     "rationale": "conflicting branch accepts dilution explanation",
                     "parents": {"models": "models-e13"}}},
    ]
    return events


def build_case(output_dir="casepkg"):
    root = Path(output_dir)
    materials = root / "materials"
    materials.mkdir(parents=True, exist_ok=True)
    power = build_power()
    schedule = build_schedule()
    blank = build_blank()
    files = {
        "power.v1.json": power,
        "injection_schedule.v1.json": schedule,
        "concentration.v1.json": concentration(100.0),
        "concentration.v2.json": concentration(1000.0),
        "cell.v1.json": {"temperature": q(25.0, "°C"), "cell_volume": q(200.0, "µL")},
        "baseline.v1.json": {"power": q(BASELINE, "µW"),
                             "coordinate_basis": "acquisition-wide constant baseline"},
        "blank.v1.json": blank,
    }
    for name, payload in files.items():
        (materials / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    events = build_events(power, schedule, blank)
    manifest = {
        "case_id": "itc-titration-heat-review",
        "domain": "滴定热谱：ITC 注射积分、稀释热与绑定模型审议",
        "narrative": [
            "Import raw power, injection timing, concentration, temperature, cell volume, separately versioned baseline and blank.",
            "Discard the first injection, retain its raw integration, and inspect overlapping post-injection peaks.",
            "Leave I4 without a corresponding blank row; dilution subtraction reports a positioned unresolved state.",
            "Revise syringe concentration; derived binding models become stale while original integrated heats are unchanged.",
            "Move only the I5 integration endpoint so the partition ends at, and never crosses, the acquisition gap.",
            "Compare one-site, multi-site and dilution candidates; retain all legal equivalent candidates.",
            "Create a conflicting human decision branch and preserve accepted and rejected outcomes with parent versions.",
        ],
        "boundary_cases": [
            "overlapping_post_injection_peaks", "first_injection_discarded",
            "blank_correspondence_missing", "concentration_revision_stales_models",
            "multiple_equivalent_models", "data_gap", "integration_endpoint"
        ],
        "events": events,
    }
    (root / "case_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    build_case()
