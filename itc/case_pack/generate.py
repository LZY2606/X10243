"""Generate the offline domain case pack.

This is ITC material, not a generic spreadsheet: every run is a sampled power
curve with injection times/volumes, concentrations, temperature, pool volume
and a separately versioned baseline and blank control.

Three runs:
  titration/  - full 12-injection binding titration (sample + baseline)
  blank/      - buffer-into-buffer dilution control (power + schedule)
  boundary/   - overlap peaks, a data gap, gap-straddling injection and a
                short final window (endpoint test)

Revisions live alongside originals: concentration_revised/v2 and the
boundary schedule v2 (a local source edit). Deterministic: noise is fixed.
"""
from __future__ import annotations

import math
import os

import numpy as np

SEED = 20260925
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _peak(t, t0, amp, rise=1.6, decay=8.0):
    z = t - t0
    out = np.empty_like(t, dtype=float)
    neg = z < 0
    out[neg] = np.exp(np.maximum(z[neg] / rise, -50.0))
    out[~neg] = np.exp(np.minimum(-z[~neg] / decay, 50.0))
    return amp * out


def _write_power(path, t, p, unit="ucal/s"):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"time[s],power[{unit}]\n")
        for tt, pp in zip(t, p):
            fh.write(f"{tt:.3f},{pp:.5f}\n")


def _write_schedule(path, times, volumes=None, index0=1, tunit="s",
                    vunit="uL"):
    if volumes is None or np.isscalar(volumes):
        vol = volumes
        pairs = [(tt, vol) for tt in times] if np.isscalar(vol) else list(times)
    else:
        pairs = list(zip(times, volumes))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"index,time[{tunit}],volume[{vunit}]\n")
        for i, (tt, vv) in enumerate(pairs, start=index0):
            fh.write(f"{i},{tt},{vv}\n")


def _write_keyvalue(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("name,value,unit\n")
        for name, val, unit in rows:
            fh.write(f"{name},{val},{unit}\n")


def _baseline_curve(t, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    drift = 0.05 * np.sin(t / 90.0) + 0.02 * t / t[-1]
    return scale * drift + rng.normal(0, 0.02 * scale, size=t.shape)


def _ligand_axis(volumes, v0, l0):
    x = np.zeros(len(volumes))
    total, disp = 0.0, 1.0
    for i, dv in enumerate(volumes):
        total += dv * disp
        disp *= (1.0 - dv / v0)
        x[i] = l0 * total / v0
    return x


def _one_site_heats(n, lgk, dh, x, m0, v0):
    k = 10.0 ** lgk
    mt = n * m0
    s_sum = mt + x + 1.0e6 / k
    nb = (s_sum - np.sqrt(np.maximum(s_sum ** 2 - 4.0 * mt * x, 0.0))) \
        / (2.0 * mt)
    qtot = (m0 * v0 / 1_000_000.0) * n * dh * nb
    q = np.empty_like(qtot)
    q[0] = qtot[0]
    q[1:] = qtot[1:] - qtot[:-1]
    return q


def _peak_area(rise=1.6, decay=8.0):
    tt = np.arange(-30.0, 40.0, 0.05)
    pk = _peak(tt, 0.0, 1.0, rise, decay)
    return float(np.trapezoid(pk, tt)) if hasattr(np, "trapezoid") \
        else float(np.trapz(pk, tt))  # noqa: NPY201


def generate(force: bool = False):
    os.makedirs(DATA, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ----- titration (sample): heats derived from a one-site model -----
    n_inj = 12
    duration = 120 + n_inj * 110 + 120
    t = np.arange(0.0, duration, 1.0)
    inj_times = list(range(120, 120 + n_inj * 110, 110))
    volumes = [5.0] * n_inj
    v0, m0, l0 = 250.0, 50.0, 400.0
    n_sites, k_per_m, dh = 1.0, 1.0e5, -8000.0  # uCal per umol
    x = _ligand_axis(volumes, v0, l0)
    q = _one_site_heats(n_sites, np.log10(k_per_m), dh, x, m0, v0)
    q = q - 3.0  # small constant dilution contribution (uCal)
    amps = [hi / _peak_area() for hi in q]
    base = _baseline_curve(t, SEED + 1)
    p = base.copy()
    for t0, a in zip(inj_times, amps):
        p += _peak(t, t0, a)
    d = os.path.join(DATA, "titration"); os.makedirs(d, exist_ok=True)
    _write_power(os.path.join(d, "power.csv"), t, p)
    _write_schedule(os.path.join(d, "injections.csv"), inj_times, volumes)
    _write_keyvalue(os.path.join(d, "concentration.csv"), [
        ("macromolecule", 50.0, "uM"),
        ("ligand", 1000.0, "uM"),
    ])
    _write_keyvalue(os.path.join(d, "conditions.csv"), [
        ("temperature", 298.15, "K"),
        ("cell_volume", 250.0, "uL"),
    ])
    # separately versioned baseline curve for the titration
    bt = np.arange(0.0, 1500.0, 1.0)
    bp = _baseline_curve(bt, SEED + 2)
    bd = os.path.join(d, "baseline"); os.makedirs(bd, exist_ok=True)
    _write_power(os.path.join(bd, "baseline.csv"), bt, bp)

    # ----- blank control (dilution) -----
    n_blank = 12
    bdur = 120 + n_blank * 110 + 60
    bt2 = np.arange(0.0, bdur, 1.0)
    blank_times = list(range(120, 120 + n_blank * 110, 110))
    blank_base = _baseline_curve(bt2, SEED + 3, scale=0.8)
    bp2 = blank_base.copy()
    for t0 in blank_times:
        bp2 += _peak(bt2, t0, -0.4, rise=1.4, decay=7.0)
    bdir = os.path.join(DATA, "blank"); os.makedirs(bdir, exist_ok=True)
    _write_power(os.path.join(bdir, "power.csv"), bt2, bp2)
    _write_schedule(os.path.join(bdir, "injections.csv"), blank_times,
                    [10.0] * len(blank_times))
    bbase = _baseline_curve(bt2, SEED + 4, scale=0.8)
    bbd = os.path.join(bdir, "baseline"); os.makedirs(bbd, exist_ok=True)
    _write_power(os.path.join(bbd, "baseline.csv"), bt2, bbase)

    # ----- boundary run -----
    # overlapping peaks, a 120-170 gap, a gap-straddling injection and a
    # short final window ending at the series edge.
    t3 = np.arange(0.0, 240.0, 1.0)
    gap = (t3 >= 120.0) & (t3 < 170.0)
    t3m = np.ma.array(t3, mask=gap)
    base3 = _baseline_curve(np.ma.filled(t3m, np.nan), SEED + 5)
    # produce power with NaN over the gap so the parser detects it
    p3 = base3.copy()
    for t0, a in ((20.0, -20.0), (55.0, -20.0), (80.0, -18.0),
                  (160.0, -15.0), (214.0, -12.0)):
        p3 += np.where(np.isnan(p3), np.nan, _peak(t3, t0, a))
    p3[gap] = np.nan
    dir3 = os.path.join(DATA, "boundary"); os.makedirs(dir3, exist_ok=True)
    _write_power_with_gap(os.path.join(dir3, "power.csv"), t3, p3)
    btimes = [20.0, 55.0, 80.0, 160.0, 214.0]
    _write_schedule(os.path.join(dir3, "injections.csv"), btimes,
                    [8.0] * len(btimes))
    _write_keyvalue(os.path.join(dir3, "concentration.csv"), [
        ("macromolecule", 50.0, "uM"),
        ("ligand", 600.0, "uM"),
    ])
    _write_keyvalue(os.path.join(dir3, "conditions.csv"), [
        ("temperature", 25.0, "C"),
        ("cell_volume", 250.0, "uL"),
    ])
    b3 = _baseline_curve(t3, SEED + 6); b3[gap] = np.nan
    b3d = os.path.join(dir3, "baseline"); os.makedirs(b3d, exist_ok=True)
    _write_power_with_gap(os.path.join(b3d, "baseline.csv"), t3, b3)

    # ----- equi-optimal weak-binding run -----
    # weak binding plus a real dilution contribution makes the two-site and
    # dilution hypotheses statistically indistinguishable (both legal,
    # retained), while one-site cannot fit.
    n_eq = 12
    edur = 120 + n_eq * 110 + 120
    te = np.arange(0.0, edur, 1.0)
    etimes = list(range(120, 120 + n_eq * 110, 110))
    evol = [5.0] * n_eq
    ex = _ligand_axis(evol, 250.0, 120.0)
    eq = _one_site_heats(0.8, 3.2, -3000.0, ex, 50.0, 250.0)
    eq = eq - (0.2 + 4.0 * (np.array(evol) * 120.0 / 1000.0))
    rng_eq = np.random.default_rng(3)
    eq = eq + rng_eq.normal(0, 0.4, size=eq.shape)
    ebase = _baseline_curve(te, SEED + 7)
    ep = ebase.copy()
    for t0, heat in zip(etimes, eq):
        ep += _peak(te, t0, heat / _peak_area())
    edir = os.path.join(DATA, "equiopt"); os.makedirs(edir, exist_ok=True)
    _write_power(os.path.join(edir, "power.csv"), te, ep)
    _write_schedule(os.path.join(edir, "injections.csv"), etimes, evol)
    _write_keyvalue(os.path.join(edir, "concentration.csv"), [
        ("macromolecule", 50.0, "uM"),
        ("ligand", 120.0, "uM"),
    ])
    _write_keyvalue(os.path.join(edir, "conditions.csv"), [
        ("temperature", 298.15, "K"),
        ("cell_volume", 250.0, "uL"),
    ])
    ebd = os.path.join(edir, "baseline"); os.makedirs(ebd, exist_ok=True)
    _write_power(os.path.join(ebd, "baseline.csv"), te,
                 _baseline_curve(te, SEED + 8))

    # ----- revisions -----
    rev = os.path.join(d, "concentration_revised")
    os.makedirs(rev, exist_ok=True)
    _write_keyvalue(os.path.join(rev, "concentration.csv"), [
        ("macromolecule", 52.0, "uM"),
        ("ligand", 1000.0, "uM"),
    ])
    # local source edit that only changes the boundary schedule window:
    # moves the first two injections closer (still overlapping) but does not
    # touch concentrations / conditions
    rev2 = os.path.join(dir3, "schedule_revised")
    os.makedirs(rev2, exist_ok=True)
    _write_schedule(os.path.join(rev2, "injections.csv"),
                    [18.0, 46.0, 80.0, 160.0, 214.0], [8.0] * 5)
    return DATA


def _write_power_with_gap(path, t, p):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("time[s],power[ucal/s]\n")
        for tt, pp in zip(t, p):
            if pp is None or (isinstance(pp, float) and math.isnan(pp)):
                continue  # omit rows in the gap -> parser detects the interval
            fh.write(f"{tt:.3f},{pp:.5f}\n")


FILES = {
    "sample_power": "titration/power.csv",
    "sample_schedule": "titration/injections.csv",
    "sample_concentration": "titration/concentration.csv",
    "sample_conditions": "titration/conditions.csv",
    "sample_baseline": "titration/baseline/baseline.csv",
    "blank_power": "blank/power.csv",
    "blank_schedule": "blank/injections.csv",
    "blank_baseline": "blank/baseline/baseline.csv",
    "boundary_power": "boundary/power.csv",
    "boundary_schedule": "boundary/injections.csv",
    "boundary_concentration": "boundary/concentration.csv",
    "boundary_conditions": "boundary/conditions.csv",
    "boundary_baseline": "boundary/baseline/baseline.csv",
    "equiopt_power": "equiopt/power.csv",
    "equiopt_schedule": "equiopt/injections.csv",
    "equiopt_concentration": "equiopt/concentration.csv",
    "equiopt_conditions": "equiopt/conditions.csv",
    "equiopt_baseline": "equiopt/baseline/baseline.csv",
    "sample_concentration_v2":
        "titration/concentration_revised/concentration.csv",
    "boundary_schedule_v2":
        "boundary/schedule_revised/injections.csv",
}

MANIFEST = {
    "title": "滴定热谱：ITC 注射积分、稀释热与绑定模型审议",
    "narrative": [
        "1. 导入原始功率曲线、注射时序、浓度、温度、池体积与分开版本化的"
        "基线和空白对照。",
        "2. 触发重叠注射峰、首注舍弃、空白对应缺失、浓度修订、多个等优模型、"
        "数据缺口与积分端点边界。",
        "3. 复核一位点、两位点与稀释假设，接受/否定/调整均可回溯到材料版本、"
        "规则版本与人工决定。",
    ],
}


def read_case_file(key: str) -> str:
    rel = FILES[key]
    with open(os.path.join(DATA, rel), encoding="utf-8") as fh:
        return fh.read()


if __name__ == "__main__":
    out = generate(force=True)
    print("generated case pack at", out)

