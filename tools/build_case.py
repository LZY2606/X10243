"""Build the bundled offline case pack.

Domain narrative: titration thermogram review of a 1:1 binding experiment.
The pack is deliberately not a generic spreadsheet: every file carries domain
units and the scenario walks every tricky rule.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

CASE_DIR = pathlib.Path(__file__).resolve().parent.parent / "cases" / "itc-titration-heat-review"

DT = 0.1                 # sampling period, s
BASELINE_MW = 0.012      # instrumental power offset
SHOTS = 12
INTERVAL = 24.0          # post window 40 s > interval -> overlapping peaks
TAU_S = 0.9
RISE_S = 0.8
DECAY_S = 1.4


def peak(t: np.ndarray, t0: float, amp_mw: float, tau: float = TAU_S,
         rise: float = 0.25) -> np.ndarray:
    dt = t - t0
    y = np.zeros_like(t)
    m = dt >= 0
    # rise + exponential relaxation gives an ITC-like peak
    y[m] = amp_mw * np.exp(-dt[m] / tau) * (1.0 - np.exp(-dt[m] / rise))
    return y


def exp_peak(t: np.ndarray, t0: float, amp_mw: float) -> np.ndarray:
    """Fast rise/decay pulse that vanishes within ~5 s, while peaks from
    adjacent injections still visually overlap (interval 24 s < post window
    40 s). Analytic area per unit amplitude:

        integral (1 - e^-t/r) e^-t/d dt = d - 1/(1/d + 1/r)
    """
    dt = t - t0
    y = np.zeros_like(t)
    m = dt >= 0
    y[m] = amp_mw * (1.0 - np.exp(-dt[m] / RISE_S)) * np.exp(-dt[m] / DECAY_S)
    return y


PULSE_AREA_S = DECAY_S - 1.0 / (1.0 / DECAY_S + 1.0 / RISE_S)


def build_power(amplitudes: list[float], *, seed: int = 7, gap: bool = True):
    t_end = SHOTS * INTERVAL + 150.0
    t = np.arange(0.0, t_end, DT)
    rng = np.random.default_rng(seed)
    y = np.full_like(t, BASELINE_MW)
    y += 0.0003 * np.linspace(0, 1, t.size)         # slow drift
    for k, amp in enumerate(amplitudes, start=1):
        y += exp_peak(t, (k - 1) * INTERVAL, amp)
    y += rng.normal(0, 0.00015, t.size)
    if gap:
        # acquisition gap inside the 7th post-injection window
        g0 = int(((6) * INTERVAL + 30) / DT)
        g1 = g0 + 12
        t = np.concatenate([t[:g0], t[g1:]])
        y = np.concatenate([y[:g0], y[g1:]])
    return t, y


def _model_heats(*, n: float, k: float, dh: float) -> np.ndarray:
    cell = 270e-6
    dv = 6e-6
    m_cell = 0.5e-3
    syringe = 10e-3
    q_prev = 0.0
    accum = 0.0
    heats = []
    for shot in range(1, SHOTS + 1):
        v_after = cell + accum + dv
        accum += dv
        x_total = (m_cell * cell + accum * syringe) / v_after
        a = x_total + n * m_cell + 1.0 / k
        bound = 0.5 * (a - math.sqrt(max(a * a - 4 * n * m_cell * x_total, 0.0)))
        q_now = v_after * dh * bound
        dq = q_now + dv / v_after * q_prev - cell / v_after * q_prev
        q_prev = q_now
        heats.append(dq)
    return np.array(heats)


def _heat_to_amp(heat_j: float) -> float:
    """Convert a per-shot heat (J) to peak amplitude (mW).

    The pulse kernel has analytic area PULSE_AREA_S seconds; heat =
    amplitude (W) * area (s). The function returns mW.
    """
    amp_w = heat_j / PULSE_AREA_S
    return amp_w * 1000.0


def write_series(path: pathlib.Path, t, y_mw, time_col="time_s(s)",
                 val_col="power_uW(uW)", scale: float = 1000.0):
    lines = [f"{time_col},{val_col}"]
    for tt, vv in zip(t, y_mw):
        if not np.isfinite(vv):
            lines.append(f"{tt:.3f},BAD")
        else:
            lines.append(f"{tt:.3f},{vv * scale:.6f}")
    path.write_text("\n".join(lines) + "\n")


def main():
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    (CASE_DIR / "revision").mkdir(exist_ok=True)
    (CASE_DIR / "patch").mkdir(exist_ok=True)

    # generate thermodynamics-consistent differential heats (J) for a 1:1
    # site, then convert each heat to a peak amplitude in mW using the
    # analytic area of the rise/decay kernel.
    heats_j = _model_heats(n=1.0, k=2.0e5, dh=-3.0e4)
    amps = [_heat_to_amp(h) for h in heats_j]
    # first injection is a deliberately bad partial injection (40% volume)
    amps[0] *= 0.4
    t, y = build_power(amps)
    write_series(CASE_DIR / "power.csv", t, y)

    # baseline curve, separately versioned (slightly different drift)
    tb = np.arange(0, t[-1] + DT, DT)
    yb = np.full_like(tb, BASELINE_MW) + 0.0003 * np.linspace(0, 1, tb.size)
    write_series(CASE_DIR / "baseline.csv", tb, yb)

    # blank dilution curve, constant small dilution heat (~ -8 uJ/shot);
    # the 12th blank run segment is deleted further below.
    blank_amp = _heat_to_amp(-8e-6)
    blank_amps = [blank_amp] * SHOTS
    tblank = np.arange(0.0, SHOTS * INTERVAL + 150.0, DT)
    yblank = np.full_like(tblank, BASELINE_MW) + 0.0003 * np.linspace(0, 1, tblank.size)
    for k, amp in enumerate(blank_amps, start=1):
        if amp != 0.0:
            yblank += exp_peak(tblank, (k - 1) * INTERVAL, amp)
    yblank += np.random.default_rng(11).normal(0, 0.00012, tblank.size)
    # the 12th blank run is genuinely missing: delete its acquisition segment
    missing_t0 = (SHOTS - 1) * INTERVAL - 5.0
    missing_t1 = (SHOTS - 1) * INTERVAL + 140.0
    keep = ~((tblank >= missing_t0) & (tblank <= missing_t1))
    tblank, yblank = tblank[keep], yblank[keep]
    write_series(CASE_DIR / "blank.csv", tblank, yblank)

    # schedules
    sched = ["shot,time_s(s),volume_uL(uL),duration_s(s)"]
    for k in range(1, SHOTS + 1):
        sched.append(f"{k},{(k - 1) * INTERVAL:.1f},6.00,8.0")
    (CASE_DIR / "schedule.csv").write_text("\n".join(sched) + "\n")
    bsched = ["shot,time_s(s),volume_uL(uL),duration_s(s)"]
    for k in range(1, SHOTS + 1):
        bsched.append(f"{k},{(k - 1) * INTERVAL:.1f},6.00,8.0")
    (CASE_DIR / "blank_schedule.csv").write_text("\n".join(bsched) + "\n")

    (CASE_DIR / "conditions.csv").write_text(
        "name,value\n"
        "temperature_C(C),25.0\n"
        "cell_volume_uL(uL),270.0\n")
    (CASE_DIR / "concentration.csv").write_text(
        "role,syringe_M(mM),cell_M(mM)\n"
        "initial,10.00,0.500\n")

    # revision: syringe concentration corrected upward (stales fits only)
    (CASE_DIR / "revision" / "concentration_v2.csv").write_text(
        "role,syringe_M(mM),cell_M(mM)\n"
        "revised,11.00,0.500\n")

    # local power patch: a handful of samples around shot 4 only
    t_patch = (3 * INTERVAL) + 6.0
    idx = int(t_patch / DT)
    patch_rows = ["time_s(s),power_w(W)"]
    for j in range(3):
        patch_rows.append(f"{t[idx + j]:.3f},{(y[idx + j] / 1000.0) + 0.000002:.9f}")
    (CASE_DIR / "patch" / "power_local.csv").write_text("\n".join(patch_rows) + "\n")

    manifest = {
        "domain": "titration thermogram: ITC injection integration, dilution heat and binding-model review",
        "time_basis": "experiment_seconds; t=0 is acquisition start, shared by sample runs",
        "coordinate_basis": "all series time columns are seconds relative to acquisition start",
        "materials": {
            "power": "power.csv", "baseline": "baseline.csv", "blank": "blank.csv",
            "schedule": "schedule.csv", "blank_schedule": "blank_schedule.csv",
            "conditions": "conditions.csv", "concentration": "concentration.csv"},
        "revision": {"concentration": "revision/concentration_v2.csv"},
        "local_patch": {"power": "patch/power_local.csv"},
        "features": [
            "overlapping_post_injection_peaks", "first_injection_discard",
            "blank_correspondence_missing_shot_12",
            "concentration_revision_invalidates_fit_not_integral",
            "multiple_equivalent_models", "data_gap_shot_7", "integration_endpoints"],
        "injection_interval_s": INTERVAL,
        "integration_post_window_s": 120,
        "units": {"power": "uW (canonical W)", "time": "s", "volume": "uL (canonical L)",
                  "temperature": "C (canonical K)", "concentration": "mM (canonical M)"},
    }
    (CASE_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("case written to", CASE_DIR)


if __name__ == "__main__":
    main()
