"""Generate the offline ITC case pack (*.itc domain files).

Run:  python3 scripts/make_case_data.py
The data are deterministic (fixed RNG seed) and carry units on every header.
"""
from __future__ import annotations

import math
import os
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE_DIR = os.path.join(ROOT, "cases", "titration-heat-review")
CE_DIR = os.path.join(ROOT, "cases", "counterexample-itc")

TAU = 9.0          # power peak relaxation time, seconds
DT = 1.0           # sampling interval, seconds


def _power_series(times, injections, *, amp_func, baseline=0.0,
                  noise=0.02, seed=1, gaps=(), na_points=()):
    """Sum exponential peaks per injection: amp * exp(-(t-t0)/tau)."""
    import random
    rng = random.Random(seed)
    rows = []
    gap_ranges = list(gaps)
    for t in times:
        if any(a <= t <= b for (a, b) in gap_ranges):
            continue  # absent coordinate => structural data gap
        p = baseline
        for inj in injections:
            t0 = inj["time_s"]
            if t >= t0:
                p += amp_func(inj) * math.exp(-(t - t0) / TAU)
        p += rng.gauss(0.0, noise)
        if t in na_points:
            p = None
        rows.append((t, p))
    return rows


def _format_bundle(*, kind, title, temperature="25 C", cell_volume="1.37 mL",
                   macromolecule="20 uM", ligand="200 uM", injections,
                   power_rows, baseline_segments=((0.0, 60.0, 0.0),),
                   extra_headers=""):
    out = []
    out.append("ITC-BUNDLE 1.0")
    out.append(f"kind: {kind}")
    out.append(f"title: {title}")
    out.append(f"temperature: {temperature}")
    out.append(f"cell_volume: {cell_volume}")
    if kind == "sample":
        out.append(f"macromolecule_concentration: {macromolecule}")
        out.append(f"ligand_concentration: {ligand}")
    else:
        out.append("macromolecule_concentration: 0 uM")
        out.append(f"ligand_concentration: {ligand}")
    if extra_headers:
        out.append(extra_headers.rstrip("\n"))
    out.append("@@injections")
    out.append("index,time_s,volume_uL")
    for inj in injections:
        out.append(f"{inj['index']},{inj['time_s']:.1f},{inj['volume_uL']:.2f}")
    out.append("@@baseline")
    out.append("start_s,end_s,offset_uW")
    for seg in baseline_segments:
        out.append(f"{seg[0]:.1f},{seg[1]:.1f},{seg[2]:.3f}")
    out.append("@@power")
    out.append("t_s,p_uW")
    for t, p in power_rows:
        out.append(f"{t:.1f}," + ("NA" if p is None else f"{p:.4f}"))
    return "\n".join(out) + "\n"


# --- Main review run: a clean one-site-style titration ---------------------
def main_review():
    injections = [
        {"index": i, "time_s": 60 + (i - 1) * 180, "volume_uL": 6.0}
        for i in range(1, 13)
    ]
    # heats decrease from saturation toward dilution (exothermic sign).
    amplitudes = [5.6, 5.2, 4.7, 4.1, 3.4, 2.7, 2.0, 1.45, 1.0, 0.72,
                  0.55, 0.46]
    amp_by_index = {inj["index"]: amp for inj, amp in
                    zip(injections, amplitudes)}
    times = [float(t) for t in range(0, 60 + 12 * 180 + 120, int(DT))]
    rows = _power_series(
        times, injections,
        amp_func=lambda inj: amp_by_index[inj["index"]],
        noise=0.015, seed=11,
    )
    return _format_bundle(
        kind="sample", title="CA9-ITC-014 lysozyme titration (298.15 K)",
        injections=injections, power_rows=rows,
        baseline_segments=((0.0, 60.0, 0.0),),
    )


def main_blank():
    injections = [
        {"index": i, "time_s": 60 + (i - 1) * 180, "volume_uL": 6.0}
        for i in range(1, 13)
    ]
    # Small constant dilution heat per injection.
    amp = {inj["index"]: 0.35 for inj in injections}
    times = [float(t) for t in range(0, 60 + 12 * 180 + 120, int(DT))]
    rows = _power_series(times, injections, amp_func=lambda i: amp[i["index"]],
                         noise=0.01, seed=12)
    return _format_bundle(
        kind="blank",
        title="CA9-ITC-014 blank: ligand into buffer (298.15 K)",
        ligand="200 uM", injections=injections, power_rows=rows,
    )


# --- Counterexample run: every edge case in one power stream ---------------
def counterexample_run():
    # 8 injections. Times deliberately collapse for overlap and one later
    # injection straddles a structural gap.
    times_inj = [60, 180, 190, 360, 540, 720, 900, 1000]
    injections = [
        {"index": i + 1, "time_s": t, "volume_uL": 6.0}
        for i, t in enumerate(times_inj)
    ]
    amplitudes = [6.0, 5.0, 4.2, 3.5, 2.9, 2.3, 1.7, 1.2]
    amp = {inj["index"]: a for inj, a in zip(injections, amplitudes)}
    times = [float(t) for t in range(0, 1140, int(DT))]
    # Gap sits inside the integration window of injection 6 (t=720): the
    # window crosses absent coordinates.
    gaps = [(742.0, 786.0)]
    # A single missing sample inside injection 3 window (overlapping with 2):
    # coordinate is kept, sample is NA.
    na_points = (215.0,)
    rows = _power_series(
        times, injections, amp_func=lambda i: amp[i["index"]],
        noise=0.02, seed=21, gaps=gaps, na_points=na_points,
    )
    return _format_bundle(
        kind="sample",
        title="Counterexample CE-01: overlaps, discard, gaps (298.15 K)",
        injections=injections, power_rows=rows,
        baseline_segments=((0.0, 60.0, 0.0),),
    )


def counterexample_blank():
    # Blank carries only 7 injections: correspondence for injection 8 is
    # deliberately missing.
    times_inj = [60, 180, 190, 360, 540, 720, 900]
    injections = [
        {"index": i + 1, "time_s": t, "volume_uL": 6.0}
        for i, t in enumerate(times_inj)
    ]
    amp = {inj["index"]: 0.3 for inj in injections}
    times = [float(t) for t in range(0, 1140, int(DT))]
    rows = _power_series(times, injections, amp_func=lambda i: amp[i["index"]],
                         noise=0.012, seed=22)
    return _format_bundle(
        kind="blank",
        title="Counterexample CE-01 blank: 7 of 8 injections matched",
        ligand="200 uM", injections=injections, power_rows=rows,
    )


def flat_run():
    """Nearly constant per-injection heat: dilution/one-site become equi-good."""
    injections = [
        {"index": i, "time_s": 60 + (i - 1) * 180, "volume_uL": 6.0}
        for i in range(1, 9)
    ]
    amp = {inj["index"]: 0.55 for inj in injections}
    times = [float(t) for t in range(0, 60 + 8 * 180 + 120, int(DT))]
    rows = _power_series(times, injections, amp_func=lambda i: amp[i["index"]],
                         noise=0.004, seed=31)
    return _format_bundle(
        kind="sample",
        title="Probe CE-02: constant injection heats, weak affinity",
        macromolecule="5 uM", ligand="80 uM",
        injections=injections, power_rows=rows,
    )


def concentration_revision():
    """Same stream as the main review but syringe concentration corrected."""
    text = main_review()
    return text.replace(
        "ligand_concentration: 200 uM",
        "ligand_concentration: 240 uM",
    ).replace(
        "title: CA9-ITC-014 lysozyme titration (298.15 K)",
        "title: CA9-ITC-014 ligand concentration revised 200->240 uM",
    )


def blank_revision():
    """Local-dependency-only change: blank curve re-measured, sample intact."""
    injections = [
        {"index": i, "time_s": 60 + (i - 1) * 180, "volume_uL": 6.0}
        for i in range(1, 13)
    ]
    amp = {inj["index"]: 0.22 for inj in injections}
    times = [float(t) for t in range(0, 60 + 12 * 180 + 120, int(DT))]
    rows = _power_series(times, injections, amp_func=lambda i: amp[i["index"]],
                         noise=0.008, seed=13)
    return _format_bundle(
        kind="blank",
        title="CA9-ITC-014 blank revision: lower dilution heat (298.15 K)",
        ligand="200 uM", injections=injections, power_rows=rows,
    )


RULES_INTEGRATION = textwrap.dedent("""\
    # ITC rule set: integration windows
    rule_set: integration
    rule_set_version: 1.0
    pre_injection_seconds: 10
    # Window ends when the tail returns within return_tolerance_uW of the
    # baseline for at least settle_seconds; a later injection truncates it.
    return_tolerance_uW: 0.15
    settle_seconds: 6
    max_window_seconds: 150
    # A single NA sample is interpolated; absent coordinates (a gap) make the
    # window unintegrable. Windows never span a data gap.
    discard_first_injection: false
""")

RULES_INTEGRATION_V2 = textwrap.dedent("""\
    # ITC rule set: integration windows (first injection now discarded)
    rule_set: integration
    rule_set_version: 1.1
    pre_injection_seconds: 10
    return_tolerance_uW: 0.15
    settle_seconds: 6
    max_window_seconds: 150
    discard_first_injection: true
    change_note: first shot of the CE-01 syringe is known to be short; discard it
""")

RULES_BLANK = textwrap.dedent("""\
    # ITC rule set: dilution heat correspondence
    rule_set: blank_correction
    rule_set_version: 1.0
    # Match blank to sample by ordinal injection index and volume; if there is
    # no corresponding blank injection the sample heat is flagged and left
    # uncorrected (strict mode rejects the run).
    correspondence: injection_index
    require_volume_match: true
    on_missing: flag_uncorrected
""")

RULES_BLANK_V2 = textwrap.dedent("""\
    # ITC rule set: dilution heat correspondence (time-based fallback)
    rule_set: blank_correction
    rule_set_version: 1.1
    correspondence: injection_index_time_within_20s
    require_volume_match: true
    on_missing: flag_uncorrected
    change_note: allow a blank injection within 20 s to stand in
""")

RULES_MODEL = textwrap.dedent("""\
    # ITC rule set: binding model review
    rule_set: model
    rule_set_version: 1.0
    candidates: one_site,two_site,dilution
    # Equi-optimal when relative SSE is within tie_rel_tol or AIC within
    # tie_aic; all legal candidates are retained in stable order.
    tie_rel_tol: 0.05
    tie_aic: 2.0
    max_evaluations: 6000
""")


def write_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def main() -> None:
    review = {
        "review_run.itc": main_review(),
        "review_blank.itc": main_blank(),
        "review_run_syringe_revised.itc": concentration_revision(),
        "review_blank_revision.itc": blank_revision(),
        "flat_probe.itc": flat_run(),
        "rules_integration_v1.txt": RULES_INTEGRATION,
        "rules_integration_v2.txt": RULES_INTEGRATION_V2,
        "rules_blank_v1.txt": RULES_BLANK,
        "rules_blank_v2.txt": RULES_BLANK_V2,
        "rules_model_v1.txt": RULES_MODEL,
    }
    for name, content in review.items():
        write_file(os.path.join(CASE_DIR, name), content)

    ce = {
        "ce01_run.itc": counterexample_run(),
        "ce01_blank.itc": counterexample_blank(),
        "rules_integration_v1.txt": RULES_INTEGRATION,
        "rules_integration_v2.txt": RULES_INTEGRATION_V2,
        "rules_blank_v1.txt": RULES_BLANK,
        "rules_blank_v2.txt": RULES_BLANK_V2,
        "rules_model_v1.txt": RULES_MODEL,
    }
    for name, content in ce.items():
        write_file(os.path.join(CE_DIR, name), content)
    print("wrote", len(review), "review files and", len(ce),
          "counterexample files")


if __name__ == "__main__":
    main()
