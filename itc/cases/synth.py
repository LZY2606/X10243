"""合成但具备真实结构的 ITC 原始材料。

功率峰由一位点/两位点热力学曲线正向生成，再叠加仪器基线；
空白运行只保留稀释热背景。所有量以仪器原始单位字符串给出（保留原始表示）。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.models import PARAM_SPECS, cumulative_heat


def _peak(t: np.ndarray, t0: float, amp: float, tau: float = 9.0,
          rise: float = 2.5) -> np.ndarray:
    x = t - t0
    out = np.zeros_like(t)
    m = x >= 0
    # 快速上升 + 指数衰减的仪器响应形状
    out[m] = amp * (1.0 - np.exp(-x[m] / rise)) * np.exp(-x[m] / tau)
    return out


def injection_table(n: int = 20, first_vol: float = 1.0,
                    vol: float = 2.0, spacing: float = 120.0
                    ) -> list[dict[str, Any]]:
    return [{"ordinal": i + 1,
             "volume": (first_vol if i == 0 else vol),
             "time": i * spacing,
             "label": f"I{i+1}"} for i in range(n)]


def binding_heats(injections: list[dict[str, Any]], c_syringe_m: float,
                  c_cell_m: float, v0_l: float,
                  model: str = "one_site",
                  theta: dict[str, float] | None = None,
                  dilution_kcal_mol: float = 0.15
                  ) -> list[float]:
    """返回每针观测 q（µcal），含稀释背景。"""
    theta = theta or {"n": 1.0, "logK": 6.0, "dH": -8.0}
    out = []
    cum_v = 0.0
    q_prev = 0.0
    for inj in injections:
        dv = inj["volume"] * 1e-6
        if model == "dilution":
            dq = 0.0
        else:
            c_now = c_cell_m * v0_l / (v0_l + cum_v)
            ligand_m = c_syringe_m * cum_v / (v0_l + cum_v)
            q = cumulative_heat(model, theta, ligand_m, c_now, v0_l)
            dq = q - q_prev * (1.0 - dv / v0_l)
            q_prev = q
        moles = c_syringe_m * dv
        dq += dilution_kcal_mol * 1e6 * moles
        out.append(float(dq))
        cum_v += dv
    return out


def make_curve(injections: list[dict[str, Any]], heats_ucal: list[float],
               dt: float = 1.0, end_pad: float = 180.0,
               baseline: str | float = "curve",
               overlap: bool = False, gap_rows: tuple[int, ...] = (),
               noise_seed: int = 7
               ) -> str:
    """生成原始 CSV 文本（time,power[,gap]）。

    overlap=True 时把部分注射间隔缩短到峰宽以内，制造重叠后峰。
    gap_rows 指定的采样行 power 留空并标 gap=1。
    """
    rng = np.random.default_rng(noise_seed)
    times = [inj["time"] for inj in injections]
    if overlap:
        # 第 11/12/13 针压缩间隔到 14 秒，峰宽 ~30 秒 -> 后峰重叠
        times = list(times)
        times[10] = times[9] + 60
        times[11] = times[10] + 14
        times[12] = times[11] + 14
    t_end = max(times) + 120.0 + end_pad
    t = np.arange(0.0, t_end + dt, dt)
    p = np.zeros_like(t)
    for t0, q in zip(times, heats_ucal):
        amp = q / 9.0  # 峰面积近似 amp*tau
        p += _peak(t, t0, amp, tau=9.0)
    if baseline == "curve":
        base = 0.02 + 0.0015 * np.sin(t / 400.0) + 0.00004 * t
    else:
        base = np.full_like(t, float(baseline))
    p = p + base + rng.normal(0, 0.002, size=t.size)
    lines = ["time,power" + (",gap" if gap_rows else "")]
    gap_set = set(gap_rows)
    for i, (ti, pi) in enumerate(zip(t, p)):
        if i in gap_set:
            lines.append(f"{ti:.2f},,1")
        else:
            lines.append(f"{ti:.2f},{pi:.5f}"
                         + (",0" if gap_rows else ""))
    return "\n".join(lines) + "\n"


def blank_material(injections: list[dict[str, Any]],
                   c_syringe_m: float, v0_l: float,
                   dilution_kcal_mol: float = 0.15,
                   truncate_at: int | None = None) -> dict[str, Any]:
    blanks = [{**inj, "volume": inj["volume"]}
              for inj in injections[:truncate_at or len(injections)]]
    heats = binding_heats(blanks, c_syringe_m, 0.0, v0_l,
                          model="dilution",
                          dilution_kcal_mol=dilution_kcal_mol)
    csv_text = make_curve(blanks, heats, baseline=0.01,
                          noise_seed=13)
    return {
        "format": "blank_run",
        "source_kind": "blank_run",
        "volume_unit": "uL",
        "time_unit": "s",
        "curve": {
            "format": "csv",
            "raw": csv_text,
            "columns": ["time", "power"],
            "time_unit": "s",
            "power_unit": "ucal/s",
            "samples": _samples_from_csv(csv_text),
            "gap_rows": [],
            "diagnostics": [],
            "time_base": "seconds_from_acquisition_start",
        },
        "injections": [{"ordinal": b["ordinal"], "volume_raw": b["volume"],
                        "time_raw": b["time"], "label": b["label"]}
                       for b in blanks],
    }


def _samples_from_csv(text: str) -> list[dict[str, Any]]:
    out = []
    for i, line in enumerate(text.strip().splitlines()[1:]):
        parts = line.split(",")
        gap = len(parts) > 2 and parts[2].strip() == "1"
        out.append({"row": i, "time_raw": parts[0],
                    "power_raw": parts[1], "gap": gap})
    return out


def power_payload(csv_text: str) -> dict[str, Any]:
    return {
        "format": "csv", "raw": csv_text,
        "columns": ["time", "power"]
        + (["gap"] if ",gap" in csv_text.splitlines()[0] else []),
        "time_unit": "s", "power_unit": "ucal/s",
        "samples": _samples_from_csv(csv_text),
        "gap_rows": [s["row"] for s in _samples_from_csv(csv_text)
                     if s["gap"]],
        "diagnostics": [],
        "time_base": "seconds_from_acquisition_start",
    }


def injections_payload(injections: list[dict[str, Any]],
                       times: list[float] | None = None) -> dict[str, Any]:
    rows = []
    for i, inj in enumerate(injections):
        rows.append({"ordinal": inj["ordinal"],
                     "volume_raw": inj["volume"],
                     "time_raw": (times[i] if times is not None
                                  else inj["time"]),
                     "label": inj["label"]})
    return {"format": "structured", "volume_unit": "uL", "time_unit": "s",
            "time_base": "seconds_from_acquisition_start",
            "injections": rows}


def concentration_payload(syringe_m: float, cell_m: float,
                          syringe_unit: str = "mM",
                          cell_unit: str = "uM") -> dict[str, Any]:
    def q(v_m: float, unit: str) -> dict[str, Any]:
        factor = {"mM": 1e3, "uM": 1e6, "M": 1.0}[unit]
        return {"raw": round(v_m * factor, 4), "unit": unit,
                "analyte": None}
    return {"format": "structured",
            "syringe": q(syringe_m, syringe_unit),
            "cell": q(cell_m, cell_unit)}


def baseline_payload(kind: str = "curve", text: str | None = None
                     ) -> dict[str, Any]:
    if kind == "constant":
        return {"format": "constant", "value_raw": 0.02,
                "power_unit": "ucal/s"}
    return {
        "format": "csv", "raw": text or "",
        "columns": ["time", "power"],
        "time_unit": "s", "power_unit": "ucal/s",
        "samples": _samples_from_csv(text) if text else [],
        "gap_rows": [], "diagnostics": [],
        "time_base": "seconds_from_acquisition_start",
    }


def temperature_payload(celsius: float = 25.0) -> dict[str, Any]:
    return {"format": "structured", "value_raw": celsius, "unit": "C"}


def cell_volume_payload(ml: float = 0.202) -> dict[str, Any]:
    return {"format": "structured", "value_raw": ml, "unit": "ml"}
