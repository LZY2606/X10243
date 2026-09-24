"""生成离线案例包 cases/titration-thermogram-v1/（领域材料，非通用表格）。

叙事：12 针蛋白-配体滴定，25 °C，池 202.4 µL。
- 注射间隔 180 s；第 4/5 针故意靠近（间隔 110 s）=> 重叠注射峰
- 第 6 针窗口内删除 20 s 采样 => 数据缺口 + 积分端点裁剪情形
- 空白只有 11 针 => 第 12 针空白对应缺失
- 首针 2 µL，惯例舍弃
- concentrations_v2.csv 是一次浓度修订（不改原积分，只使模型过期）
- blank_power_v2.csv 是只影响局部依赖的源数据改动（空白基线）
"""
from __future__ import annotations

import csv
import math
import os
import pathlib

DT = 2.0
T_END = 12 * 180.0 + 180.0
V0 = 202.4
INJ_TIMES = [120.0 + i * 180.0 for i in range(12)]
INJ_TIMES[4] = INJ_TIMES[3] + 110.0          # 第 5 针与第 4 针重叠
INJ_TIMES[5:] = [INJ_TIMES[4] + 180.0 + i * 180.0
                 for i in range(12 - 5)]
VOLUMES = [2.0] + [10.0] * 11
N_TRUE = 1.0
K_TRUE = 1.0e6          # M^-1
DH_TRUE_UJ_PMOL = -41.84   # -10 kcal/mol ≈ -41.84 µJ/pmol
QD_TRUE = -1.2            # 稀释热 µJ（体积加权）


def _peak(t, t0, area, tau_r=3.0, tau_d=18.0):
    if t <= t0:
        return 0.0
    u = t - t0
    norm = tau_d / (tau_r + tau_d)
    return area / (60.0 * norm) * (math.exp(-u / tau_d) - math.exp(-u / tau_r))


def _heat_i(i):
    """Wiseman 近似（µJ），含真实稀释项。"""
    mt = 20.0
    ratio = 200.0 / 20.0
    x = sum(VOLUMES[: i + 1]) / V0
    L = ratio * x / (1 + x)
    M = mt / (1 + x)
    th_prev = _theta(i - 1)
    th = _theta(i)
    v_i = VOLUMES[i]
    return N_TRUE * M * V0 * DH_TRUE_UJ_PMOL * (th - th_prev) + QD_TRUE * (
        v_i / 10.0)


def _theta(i):
    if i < 0:
        return 0.0
    mt = 20.0
    ratio = 200.0 / 20.0
    x = sum(VOLUMES[: i + 1]) / V0
    L_uM = ratio * x / (1 + x)
    K_uM = K_TRUE * 1e-6
    return K_uM * L_uM / (1 + K_uM * L_uM)


def _noise(i_row: int) -> float:
    # 确定性低幅噪声（线性同余），保证案例可重放
    z = (1103515245 * (i_row + 7) + 12345) & 0x7FFFFFFF
    return ((z / 0x7FFFFFFF) - 0.5) * 0.08


def build(outdir: str) -> str:
    root = pathlib.Path(outdir)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    rows = []
    irow = 0
    t = 0.0
    while t <= T_END:
        in_gap = INJ_TIMES[5] + 78.0 <= t <= INJ_TIMES[5] + 98.0  # 第 6 针窗内缺口
        if not in_gap:
            p = 0.05 + _noise(irow)
            for k, t0 in enumerate(INJ_TIMES):
                p += _peak(t, t0, _heat_i(k))
            rows.append((round(t, 3), round(p, 5)))
        t += DT
        irow += 1
    _write_csv(raw / "power.csv", ["time", "power"],
               [(f"{tt} s", f"{pp} µW") for tt, pp in rows])

    base = [(round(tt, 3), round(0.05 + 0.01 * math.sin(tt / 400.0), 5))
            for tt in [i * DT for i in range(int(T_END / DT) + 1)]]
    _write_csv(raw / "baseline.csv", ["time", "power"],
               [(f"{tt} s", f"{pp} µW") for tt, pp in base])

    _write_csv(raw / "injections.csv", ["injection", "time", "volume"],
               [(str(i + 1), f"{INJ_TIMES[i]} s", f"{VOLUMES[i]} µL")
                for i in range(12)])

    _write_csv(raw / "concentrations_v1.csv",
               ["injection", "ligand_concentration", "macromolecule_concentration"],
               [(str(i + 1), "200 µM", "20 µM") for i in range(12)])
    _write_csv(raw / "concentrations_v2.csv",
               ["injection", "ligand_concentration", "macromolecule_concentration"],
               [(str(i + 1), "210 µM", "20 µM") for i in range(12)])

    _write_csv(raw / "conditions.csv", ["key", "value", "kind", "unit"], [
        ("temperature", "25.0", "temperature", "C"),
        ("cell_volume", "202.4", "volume", "µL"),
        ("stir_rate", "750", "dimensionless", ""),
    ])

    # 空白：11 针（第 12 针对应缺失），恒定小稀释峰
    blank_times = [120.0 + i * 180.0 for i in range(11)]
    brows = []
    irow = 0
    for i in range(int(T_END / DT) + 1):
        tt = i * DT
        p = 0.03 + _noise(irow + 999)
        for t0 in blank_times:
            p += _peak(tt, t0, QD_TRUE * (10.0 / 10.0), tau_r=3.0, tau_d=16.0)
        brows.append((round(tt, 3), round(p, 5)))
        irow += 1
    _write_csv(raw / "blank_power_v1.csv", ["time", "power"],
               [(f"{tt} s", f"{pp} µW") for tt, pp in brows])
    _write_csv(raw / "blank_injections.csv", ["injection", "time", "volume"],
               [(str(i + 1), f"{blank_times[i]} s", "10 µL")
                for i in range(11)])
    # 空白 v2：基线整体 +0.02 µW（只影响空白派生层）
    _write_csv(raw / "blank_power_v2.csv", ["time", "power"],
               [(f"{tt} s", f"{round(pp + 0.02, 5)} µW") for tt, pp in brows])

    (root / "case.json").write_text(
        '{\n'
        '  "case_id": "itc-titration-thermogram-v1",\n'
        '  "title": "滴定热谱：ITC 注射积分、稀释热与绑定模型审议",\n'
        '  "narrative": [\n'
        '    {"step": 1, "act": "import_initial", "materials": ["power", "baseline", "injections", "concentrations_v1", "conditions", "blank_power_v1+blank_injections"]},\n'
        '    {"step": 2, "act": "integrate", "params": {"lag_s": 60, "width_s": 120}},\n'
        '    {"step": 3, "act": "decide", "decision": "discard_injection", "index": 1},\n'
        '    {"step": 4, "act": "decide", "decision": "window_endpoint", "index": 6, "clip_to_gap": true},\n'
        '    {"step": 5, "act": "fit_models", "expect_tie": false},\n'
        '    {"step": 6, "act": "branch", "name": "conflict-dilution-only", "decision": "reject_all_binding", "accept": "dilution"},\n'
        '    {"step": 7, "act": "revise", "stream": "concentration", "file": "concentrations_v2.csv", "expect_models_stale": true, "expect_integrations_unchanged": true},\n'
        '    {"step": 8, "act": "revise", "stream": "blank", "power": "blank_power_v2.csv", "expect_local_only": true}\n'
        '  ],\n'
        '  "boundary_cases": ["overlapping_peaks:4-5", "first_injection_discard:1", "blank_correspondence_missing:12", "concentration_revision:v1->v2", "multiple_equivalent_models", "data_gap:6", "integration_endpoint:6", "conflicting_human_decisions:main-vs-branch"]\n'
        '}\n', encoding="utf-8")
    return str(root)


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    out = build(os.path.join(os.path.dirname(__file__), "..",
                             "cases", "titration-thermogram-v1"))
    print("case built:", out)
