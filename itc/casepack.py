"""Offline domain case pack: 滴定热谱 — ITC 注射积分、稀释热与绑定模型审议.

Built by executing real service calls (never canned JSON), so replay actually
passes through: overlapping injection peaks, first-injection discard, missing
blank correspondence, concentration revision, multiple equally-good models,
data gaps and integration endpoints.

Narrative:
1. initial import of all seven materials, each versioned separately
   (power sampling, injection volumes/times, concentrations, temperature,
   cell volume, baseline, blank curve);
2. integration run -> decision discarding the first (small) injection ->
   dilution subtraction with an explicit correspondence rule (two injections
   lack blank heats) -> fit of one-site / two-site / dilution candidates;
3. two reviewers accept different models: conflicting decisions, both kept;
4. concentration revision: fits become stale, integrations untouched;
5. a local source change (one injection time fixed): only the dependent
   injections are recomputed, the affected set is recorded;
6. branch ``overlap-check``: alternate integration parameters exercise the
   overlap / gap / endpoint rules without disturbing mainline conclusions.
"""

import math

from . import binding, service
from .integrate import IntegParams

BRANCH_MAIN = "main"
BRANCH_OVERLAP = "overlap-check"


def synth_power():
    """Synthetic but domain-shaped power trace with an overlapping injection
    pair and a data gap inside one integration window."""
    volumes = [0.4] + [2.5] * 11
    times = [120.0 + i * 110.0 for i in range(12)]
    times[6] = times[5] + 40.0  # overlapping pair: peaks 5 and 6 overlap
    q_J = binding.per_injection_J("one_site", (3e5, -35000.0, 0.9),
                                  volumes, 0.10, 1.2, 1.4)
    heats_uJ = [q * 1e6 + 180.0 for q in q_J]  # + constant dilution heat
    gap = (times[8] + 20.0, times[8] + 45.0)  # data gap inside window of inj 8
    t = []
    x = 0.0
    while x <= 1500.0:
        if not (gap[0] < x < gap[1]):
            t.append(x)
        x += 1.0
    tau = 22.0
    p = []
    for x in t:
        val = 12.0 + 0.05 * math.sin(x / 7.0)
        for ti, q in zip(times, heats_uJ):
            if x >= ti:
                val += q / tau * math.exp(-(x - ti) / tau)
        p.append(round(val, 6))
    power = {"time": t, "power": p, "time_unit": "s", "power_unit": "uJ/s",
             "gaps": [list(gap)]}
    schedule = {"time": times, "volume": volumes,
                "time_unit": "s", "volume_unit": "uL"}
    return power, schedule


def build(store):
    power, schedule = synth_power()

    # 1. initial import: seven separately versioned materials
    service.import_material(store, "power_trace", power)
    service.import_material(store, "injection_schedule", schedule)
    service.import_material(store, "concentrations",
                            {"syringe": 1.2, "cell": 0.10, "unit": "mM"})
    service.import_material(store, "temperature", {"value": 25.0, "unit": "C"})
    service.import_material(store, "cell_volume", {"value": 1.4, "unit": "mL"})
    service.import_material(store, "baseline",
                            {"mode": "constant", "value": 12.0, "power_unit": "uJ/s"})
    # blank curve deliberately shorter than the injection series
    service.import_material(store, "blank_curve",
                            {"heats": [180.0] * 10, "unit": "uJ"})

    # 2. integrate -> discard first injection -> dilution -> fit
    r1 = service.run_integration(store)
    service.decide(store, "discard_injection", "injection:0",
                   "首注体积小（0.4 µL），按 ITC 惯例舍弃", "陈研",
                   payload={"injection": 0})
    service.run_dilution(store, rule="by_index")
    f1 = service.run_fit(store)

    # 3. conflicting human decisions, both retained
    service.decide(store, "accept_model", f1["result_id"],
                   "一位点模型残差无结构，优先简约模型", "陈研",
                   payload={"model": "one_site"})
    service.decide(store, "accept_model", f1["result_id"],
                   "等优前提下不能排除多位点，保留两位点", "李测",
                   payload={"model": "two_site"})

    # 4. concentration revision: fits stale, integrations unchanged
    service.revise_material(store, "concentrations",
                            {"syringe": 1.2, "cell": 0.12, "unit": "mM"})

    # 5. local source change: fix injection 6 time; only local deps recomputed
    sched2 = dict(schedule)
    sched2["time"] = list(schedule["time"])
    sched2["time"][6] = schedule["time"][5] + 55.0
    service.revise_material(store, "injection_schedule", sched2)
    r2 = service.run_integration(store)

    # 6. branch exercising overlap / gap / endpoint rules with other params
    service.run_integration(
        store, branch=BRANCH_OVERLAP,
        params=IntegParams(max_window_s=60.0, endpoint="nearest", version="int-1.1"))

    return {"integration_v1": r1["result_id"], "integration_v2": r2["result_id"],
            "fit": f1["result_id"]}
