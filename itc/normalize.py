"""Derive molar-normalized per-injection heats.

This layer sits below model fitting and above raw integration.  It depends
on concentrations and volumes (schedule + conditions facets) and on the
blank-corrected heats, but NOT on the raw power samples: therefore a
concentration revision changes normalized heats and invalidates fits while
leaving raw integrals byte-identical.
"""

from __future__ import annotations

import numpy as np

from .diagnostics import SEMANTIC_INVALID, diagnostic
from .hashing import digest
from .integrate import FIRST_INJECTION_DISCARDED
from .units import is_finite_q


def normalized_heats(integral_result, blank_result, schedule_material,
                     conditions_material):
    cond = conditions_material.payload
    sched = {i["index"]: i for i in schedule_material.payload["injections"]}
    diags = []
    needed = ["cell_concentration", "syringe_concentration", "cell_volume"]
    missing = [k for k in needed if not cond.get(k) or cond[k].get("value") is None]
    if missing:
        return {"rows": [], "diagnostics": [
            diagnostic(SEMANTIC_INVALID,
                       f"conditions missing/non-convertible: {', '.join(missing)}")]}

    c0 = cond["cell_concentration"]["value"]          # mol/L
    x0 = cond["syringe_concentration"]["value"]       # mol/L
    v0 = cond["cell_volume"]["value"]                 # L
    c_by_idx = {r["index"]: r for r in integral_result["records"]}

    rows = []
    total_vol = 0.0
    for corr in sorted(blank_result["corrected"], key=lambda r: r["index"]):
        idx = corr["index"]
        inj = sched.get(idx)
        rec = c_by_idx.get(idx)
        if not inj or not is_finite_q(inj["volume"]):
            diags.append(diagnostic(SEMANTIC_INVALID,
                                    f"injection {idx}: missing schedule volume"))
            continue
        dv = inj["volume"]["value"]
        total_vol += dv
        v_i = v0 + total_vol
        c_i = c0 * v0 / v_i
        x_i = x0 * total_vol / v_i
        moles_inj = x0 * dv
        usable = bool(corr["usable_in_fit"])
        q_j = corr["corrected_heat_j"]
        q_per_mol = (q_j / moles_inj) if (q_j is not None and moles_inj > 0) else None
        rows.append({
            "index": idx,
            "volume_l": dv,
            "volume_raw": inj["volume"]["raw"],
            "cell_c": c_i,
            "ligand_c": x_i,
            "syringe_c_ref": x0,
            "v_i": v_i,
            "moles_injected_mol": moles_inj,
            "q_j": q_j,
            "q_per_mol_J_per_mol": q_per_mol,
            "usable": usable,
            "flags": corr["flags"],
            "coverage": rec["coverage"] if rec else None,
            "integral_result_id": integral_result["result_id"],
            "blank_result_id": blank_result["result_id"],
        })
    result = {
        "rows": rows,
        "diagnostics": diags,
        "parents": {
            "schedule": schedule_material.version_id,
            "conditions": conditions_material.version_id,
            "integral": integral_result["result_id"],
            "blank": blank_result["result_id"],
        },
    }
    result["result_id"] = digest("normalized", {
        "parents": result["parents"],
        "rows": [{"index": r["index"], "q_j": round(r["q_j"], 12) if r["q_j"] is not None else None,
                  "cell_c": r["cell_c"], "ligand_c": r["ligand_c"],
                  "volume_l": r["volume_l"], "v_i": r["v_i"],
                  "usable": r["usable"]} for r in rows],
    })
    return result
