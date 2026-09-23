import math

from . import binding, pipeline

GROUND_TRUTH = {
    "sites": [[1.0, 100.0, -40.0]],
    "M0_mM": 0.1,
    "Xs_mM": 1.0,
    "V0_mL": 1.4,
    "tau_s": 25.0,
    "dilution_uJ": 60.0,
}

TIMES = [300.0, 540.0, 780.0, 1020.0, 1260.0, 1500.0, 1560.0, 1740.0]
VOLS = [2.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
GAP = [1130.0, 1150.0]
T_END = 1900

PARAMS_INTEGRATION_V1 = {
    "window_s": 150.0,
    "discard_first": True,
    "blank_rule": "per_index",
    "pre_s": 20.0,
}
PARAMS_FIT_V1 = {
    "models": ["one-site", "two-site", "dilution"],
    "equivalence_tol": 0.05,
    "budget": 4000,
}


def _pulse_train(times, heats_uJ, tau, gap, bump=None):
    ts = []
    ps = []
    for t in range(0, T_END + 1):
        if gap[0] <= t < gap[1]:
            continue
        p = -0.5 + 0.3 * t / T_END
        for ti, q in zip(times, heats_uJ):
            if t >= ti:
                p += q / tau * math.exp(-(t - ti) / tau)
        if bump is not None and bump[0] <= t < bump[1]:
            p += 8.0 * math.exp(-(t - bump[0]) / 15.0)
        p += 0.3 * math.sin(2.0 * math.pi * t / 7.3) + 0.2 * math.sin(2.0 * math.pi * t / 3.1)
        ts.append(float(t))
        ps.append(p)
    return ts, ps


def synth_cell_power(bump=None):
    g = GROUND_TRUTH
    heats = binding.injection_heats_uJ(g["sites"], g["M0_mM"], g["Xs_mM"],
                                       g["V0_mL"], VOLS)
    heats = [h + g["dilution_uJ"] for h in heats]
    return _pulse_train(TIMES, heats, g["tau_s"], GAP, bump=bump)


def synth_blank_power():
    g = GROUND_TRUTH
    heats = [g["dilution_uJ"]] * 7
    return _pulse_train(TIMES[:7], heats, g["tau_s"], gap=(-1.0, -1.0))


def raw_materials():
    t, p = synth_cell_power()
    tb, pb = synth_blank_power()
    power = {
        "unit_t": "s", "unit_p": "uW",
        "samples": [[t[i], p[i]] for i in range(len(t))],
        "gaps": [list(GAP)],
        "meta": {"instrument": "VP-ITC", "role": "cell",
                 "time_basis": "seconds since run start"},
    }
    schedule = {
        "t_unit": "s",
        "injections": [{"t": tt, "vol": {"value": v, "unit": "uL"}}
                       for tt, v in zip(TIMES, VOLS)],
    }
    conc = {"syringe": {"value": 1.0, "unit": "mM"},
            "cell": {"value": 100.0, "unit": "uM"}}
    temp = {"value": 25.0, "unit": "degC"}
    cellvol = {"value": 1.4, "unit": "mL"}
    baseline = {"kind": "linear", "unit_t": "s", "unit_p": "uW",
                "points": [[0.0, -0.5], [float(T_END), -0.2]]}
    blank_injections = [{"t": tt, "vol": {"value": v, "unit": "uL"}}
                        for tt, v in zip(TIMES[:7], VOLS[:7])]
    blank_injections[2]["vol"] = {"value": "n/a", "unit": "uL"}
    blank = {
        "power": {"unit_t": "s", "unit_p": "uW",
                  "samples": [[tb[i], pb[i]] for i in range(len(tb))],
                  "gaps": [],
                  "meta": {"role": "blank", "time_basis": "seconds since run start"}},
        "schedule": {"t_unit": "s", "injections": blank_injections},
    }
    return [
        ("mat:power", "power", power),
        ("mat:schedule", "schedule", schedule),
        ("mat:conc", "conc", conc),
        ("mat:temp", "temp", temp),
        ("mat:cellvol", "cellvol", cellvol),
        ("mat:baseline", "baseline", baseline),
        ("mat:blank", "blank", blank),
        ("mat:params.integration", "params", dict(PARAMS_INTEGRATION_V1)),
        ("mat:params.fit", "params", dict(PARAMS_FIT_V1)),
    ]


def _candidate_id(store, kind):
    fits = store.latest_result("fits")
    for c in fits["payload"]["candidates"]:
        if c["kind"] == kind:
            return c["id"]
    raise KeyError(kind)


def build(store):
    with store.write_set("case:baseline-import") as ws:
        for mid, kind, raw in raw_materials():
            store.add_material(ws, mid, kind, raw, phase="baseline-import")

    pipeline.run_all(store, phase="initial-review")
    store.record_decision(_candidate_id(store, "one-site"), "accept",
                          "initial review: one-site isotherm explains the curve",
                          phase="initial-review")
    store.record_decision(_candidate_id(store, "dilution"), "reject",
                          "dilution-only hypothesis cannot explain the curve shape",
                          phase="initial-review")

    with store.write_set("case:parameter-revision") as ws:
        revised = dict(PARAMS_INTEGRATION_V1)
        revised["window_s"] = 100.0
        store.add_material(ws, "mat:params.integration", "params", revised,
                           cause="parameter-revision", phase="parameter-revision")
    pipeline.run_all(store, phase="parameter-revision")

    t2, p2 = synth_cell_power(bump=(1260.0, 1410.0))
    power_v2 = {
        "unit_t": "s", "unit_p": "uW",
        "samples": [[t2[i], p2[i]] for i in range(len(t2))],
        "gaps": [list(GAP)],
        "meta": {"instrument": "VP-ITC", "role": "cell",
                 "time_basis": "seconds since run start",
                 "edit": "localized bump inside injection 5 window only"},
    }
    with store.write_set("case:localized-source-change") as ws:
        store.add_material(ws, "mat:power", "power", power_v2,
                           cause="localized-source-change",
                           phase="localized-source-change")
    pipeline.run_all(store, phase="localized-source-change")

    with store.write_set("case:concentration-revision") as ws:
        store.add_material(ws, "mat:conc", "conc",
                           {"syringe": {"value": 0.9, "unit": "mM"},
                            "cell": {"value": 100.0, "unit": "uM"}},
                           cause="concentration-revision",
                           phase="concentration-revision")
    pipeline.run_all(store, phase="concentration-revision")

    store.record_decision(_candidate_id(store, "one-site"), "accept",
                          "after concentration revision one-site still preferred",
                          phase="conflicting-decisions")
    store.record_decision(_candidate_id(store, "two-site"), "accept",
                          "second reviewer insists the equally-good two-site candidate",
                          phase="conflicting-decisions")
    return store
