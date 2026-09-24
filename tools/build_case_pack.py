"""Generate the dedicated offline ITC case pack under ./case.

Run: .venv/bin/python tools/build_case_pack.py

The output is real domain material with units and time references on every
quantity, original lexical representations preserved (string numbers), and
separate versioned files per material kind.
"""
from __future__ import annotations

import json
from pathlib import Path

from itc import casebuild

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "case"


def write(rel: str, obj: dict) -> None:
    path = CASE / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def main() -> None:
    # --- titration run, initial versions --------------------------------- #
    write("materials/titration/power.v1.json",
          {"schema": "itc-material/power/1.0",
           **casebuild._samples_payload(
               casebuild.make_sample_trace_v1(),
               time_ref="t=0 at acquisition start")})
    write("materials/titration/injections.v1.json",
          {"schema": "itc-material/injections/1.0",
           **casebuild.injections_payload(casebuild.INJ_TIMES_S,
                                          casebuild.INJ_VOLUMES_UL)})
    write("materials/titration/concentration.v1.json",
          {"schema": "itc-material/concentration/1.0",
           **casebuild.concentration_payload(casebuild.SYRINGE_MM,
                                             casebuild.CELL_UM)})
    write("materials/titration/concentration.v2.json",
          {"schema": "itc-material/concentration/1.0",
           **casebuild.concentration_payload(0.32, casebuild.CELL_UM,
                                             syringe_raw="0.32")})
    write("materials/titration/temperature.v1.json",
          {"schema": "itc-material/temperature/1.0",
           **casebuild.temperature_payload(casebuild.TEMP_C)})
    write("materials/titration/cell_volume.v1.json",
          {"schema": "itc-material/cell_volume/1.0",
           **casebuild.cell_volume_payload(casebuild.V0_ML)})
    write("materials/titration/baseline.v1.json",
          {"schema": "itc-material/baseline/1.0",
           **casebuild.baseline_payload(1.0e-6)})
    # v1 blank link: ordinal correspondence but the blank's first shot is
    # discarded -> sample ordinal 6 ends up without a counterpart
    write("materials/titration/blank_link.v1.json",
          {"schema": "itc-material/blank_link/1.0",
           **casebuild.blank_link_payload(
               casebuild_repr_blank(), rule="ordinal",
               discard_first=True,
               note="空白首注为预注射，按规则丢弃；对应缺失须显式记录")})

    # --- dilution blank run ---------------------------------------------- #
    write("materials/blank/power.v1.json",
          {"schema": "itc-material/power/1.0",
           **casebuild._samples_payload(casebuild.make_blank_trace(),
                                        time_ref="t=0 at acquisition start")})
    write("materials/blank/injections.v1.json",
          {"schema": "itc-material/injections/1.0",
           **casebuild.injections_payload(casebuild.BLANK_INJ_TIMES_S,
                                          casebuild.BLANK_VOLUMES_UL,
                                          discard_ordinal=1)})
    write("materials/blank/baseline.v1.json",
          {"schema": "itc-material/baseline/1.0",
           **casebuild.baseline_payload(0.8e-6)})

    # --- local-only power fragment patch (490-560 s) --------------------- #
    write("materials/power.patch.490-560.json",
          {"schema": "itc-material/power-patch/1.0",
           **casebuild.patched_power_payload()})

    # --- an intentionally unconvertible value kept in place -------------- #
    bad = {"schema": "itc-material/temperature/1.0",
           "temperature": {"value": "warm-ish", "unit": "°C"},
           "note": "故意无法转换：导入时留在原位置并附诊断"}
    write("materials/rejected/temperature.unconvertible.json", bad)

    # --- replay narrative -------------------------------------------------- #
    replay = {
        "schema": "itc-replay/1.0",
        "title": "滴定热谱：ITC 注射积分、稀释热与绑定模型审议",
        "runs": {"titration": "titration-2026-09-24",
                 "blank": "blank-buffer-2026-09-24",
                 "flatrun": "flatrun-2026-09-24"},
        "situations": [
            {"code": "overlapping_peaks", "ordinals": [3, 4],
             "rule": "injection windows end at the next injection time"},
            {"code": "first_injection_discard", "ordinal": 1},
            {"code": "blank_counterpart_missing", "ordinal": 6,
             "rule": "ordinal + discard_first_in_blank"},
            {"code": "concentration_revision",
             "from_mM": 0.40, "to_mM": 0.32,
             "effect": "models stale; original integrals unchanged"},
            {"code": "multiple_equivalent_models", "branch": "boundary-flat",
             "rule": "keep all candidates within SSR relative tolerance"},
            {"code": "data_gap_and_endpoint", "ordinal": 5,
             "gap_s": [355.0, 372.0], "endpoint": "before_gap"},
            {"code": "local_power_patch", "interval_s": [490.0, 560.0],
             "effect": "only ordinal 7/8 derived records change"},
            {"code": "conflicting_decisions", "model": "one_site",
             "decisions": ["accept", "reject"]},
        ],
        "units_and_references": {
            "time": "s from acquisition start",
            "power": "cal/s", "energy": "cal (display µcal)",
            "volume": "L (source µL/mL retained)",
            "concentration": "M (source mM/µM retained)",
            "temperature": "K (source °C retained)",
        },
    }
    write("replay/replay.json", replay)
    print("case pack written to", CASE)


def casebuild_repr_blank() -> str:
    return "blank-buffer-2026-09-24"


if __name__ == "__main__":
    main()
