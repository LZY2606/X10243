"""离线案例包：滴定热谱——ITC 注射积分、稀释热与绑定模型审议。

这是领域材料，不是通用表格：
* power.csv            样本功率采样（重叠峰、数据缺口、一个不可解析坏格）
* injections.csv       注射体积与时间（首注预冲量；第 9/10 注重叠）
* concentration.txt    初始浓度（v1；revisions/ 里是修订 v2）
* temperature.txt      25 °C（原文保留，派生层转 K）
* cell_volume.txt      1.4 mL
* baseline.txt         基线标量（独立版本化）
* blank.txt            空白对照（独立曲线 + 自带注射时序，故意缺第 13 注）
* narrative.md         连续叙事
热谱按一位点 Wiseman 模型（含体积置换项）正向合成。
"""
from __future__ import annotations

import json
import os

import numpy as np

# ---------- 合成参数（真值） ----------
TRUE = {"Ka": 1.0e5, "n": 1.0, "dH_kcal": -12.0}
SYRINGE_V1_M = 0.60e-3
CELL_M = 0.030e-3
V0_L = 1.4e-3
TAU = 8.0
DT = 2.0
BASELINE_POWER = 5.0
NOISE = 0.03
SEED = 20260925

# (index, time_s, volume_uL)；首注预冲；9/10 重叠
INJECTION_PLAN = [
    (1, 120.0, 2.0),
    (2, 300.0, 10.0), (3, 480.0, 10.0), (4, 660.0, 10.0), (5, 840.0, 10.0),
    (6, 1020.0, 10.0), (7, 1200.0, 10.0), (8, 1380.0, 10.0),
    (9, 1530.0, 8.0), (10, 1576.0, 8.0),
    (11, 1780.0, 10.0), (12, 1960.0, 10.0), (13, 2140.0, 10.0),
    (14, 2320.0, 10.0), (15, 2520.0, 10.0), (16, 2700.0, 10.0),
    (17, 2880.0, 10.0), (18, 3060.0, 10.0), (19, 3240.0, 10.0), (20, 3420.0, 10.0),
]
GAP = (2360.0, 2440.0)          # 卡住第 14 注的回归尾巴
BAD_CELL_INDEX = 317            # ~634 s
T_END = 3620.0

BLANK_DIL_PER_10UL_CAL = 1.2e-7  # 0.12 µcal / 10 µL
SYRINGE_V1 = "0.60 mM"
SYRINGE_V2 = "0.54 mM"
CELL_CONC = "0.030 mM"


def true_injection_heats(syringe_m: float = SYRINGE_V1_M) -> list[float]:
    dv = np.array([v for (_, _, v) in INJECTION_PLAN], dtype=float) * 1e-6
    V = V0_L + np.cumsum(dv)
    X = np.cumsum(dv * syringe_m) / V
    M = CELL_M
    K = TRUE["Ka"]
    b = 1.0 + K * (TRUE["n"] * M + X)
    disc = np.maximum(b * b - 4.0 * K * K * TRUE["n"] * M * X, 0.0)
    f = np.clip((X - (b - np.sqrt(disc)) / (2.0 * K)) / (TRUE["n"] * M), 0.0, 1.0)
    Q = 1000.0 * TRUE["n"] * TRUE["dH_kcal"] * V * M * f   # cal
    Qp = np.concatenate(([0.0], Q[:-1]))
    dq = Q - Qp + (dv / V) * 0.5 * (Q + Qp)
    return [float(x) for x in dq]


HEATS_V1 = true_injection_heats(SYRINGE_V1_M)


def _pulse(t: np.ndarray, t0: float, heat_cal: float, tau: float = TAU) -> np.ndarray:
    dt = t - t0
    return np.where(dt >= 0, heat_cal * (1.0e6 / tau) * np.exp(-np.maximum(dt, 0.0) / tau), 0.0)


def build_power_text(heats: list[float] | None = None,
                     gap: tuple[float, float] | None = GAP,
                     t_end: float = T_END, seed: int = SEED,
                     bad_cell: int | None = BAD_CELL_INDEX) -> str:
    heats = heats or HEATS_V1
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, t_end + 1e-9, DT)
    if gap is not None:
        t = t[(t < gap[0]) | (t > gap[1])]
    p = np.full_like(t, BASELINE_POWER) + rng.normal(0.0, NOISE, size=t.shape)
    for (i, t0, v), h in zip(INJECTION_PLAN, heats):
        p += _pulse(t, t0, h)
    lines = ["# time_unit: s", "# power_unit: µcal/s", "time,power"]
    for n, (tt, pp) in enumerate(zip(t, p)):
        if bad_cell is not None and n == bad_cell:
            lines.append(f"{tt:.1f},N/A (传感器饱和)")
        else:
            lines.append(f"{tt:.1f},{pp:.4f}")
    return "\n".join(lines) + "\n"


def build_injections_text() -> str:
    lines = ["# time_unit: s", "# volume_unit: µL", "index,time,volume"]
    for (i, t0, v) in INJECTION_PLAN:
        lines.append(f"{i},{t0:.1f},{v:.2f}")
    return "\n".join(lines) + "\n"


def build_blank_text() -> str:
    rng = np.random.default_rng(997731)
    t = np.arange(0.0, T_END + 1e-9, DT)
    p = np.full_like(t, BASELINE_POWER) + rng.normal(0.0, NOISE, size=t.shape)
    kept = [(i, t0, v) for (i, t0, v) in INJECTION_PLAN if i != 13]
    for (i, t0, v) in kept:
        h = BLANK_DIL_PER_10UL_CAL * (v / 10.0)
        p += _pulse(t, t0, h, tau=7.0)
    lines = ["# time_unit: s", "# power_unit: µcal/s", "# volume_unit: µL",
             "@@ power", "time,power"]
    for tt, pp in zip(t, p):
        lines.append(f"{tt:.1f},{pp:.4f}")
    lines += ["@@ injections", "index,time,volume"]
    for (i, t0, v) in kept:
        lines.append(f"{i},{t0:.1f},{v:.2f}")
    return "\n".join(lines) + "\n"


def concentration_text(syringe: str = SYRINGE_V1, cell: str = CELL_CONC) -> str:
    return f"syringe = {syringe}\ncell = {cell}\n"


NARRATIVE = """# 滴定热谱：ITC 注射积分、稀释热与绑定模型审议（离线案例包）

一条连续叙事，不是彼此无关的按钮：

1. 初始导入后核对：功率采样、注射体积与时间、浓度、温度、池体积、基线、空白曲线，
   七类材料分别版本化，原始表示全部保留（含坏格 `N/A (传感器饱和)`）。
2. 边界组合（重放实际经过每一种）：
   * 重叠注射峰 —— 第 9/10 注间隔 46 s；峰重叠，积分窗各自独立且不强制切分。
   * 首注舍弃 —— 第 1 注 2 µL 预冲，按策略舍弃并留痕。
   * 数据缺口与积分端点 —— 2360–2440 s 无采样；第 14 注窗被缺口夹断，绝不跨越。
   * 空白对应缺失 —— 空白序列缺第 13 注；稀释扣除说明 by_index 规则且不强行扣除。
   * 浓度修订 —— 注射器 0.60 mM → 0.54 mM：旧模型过期，原积分不改变。
   * 多个等优模型 —— 一位点/两位点 ΔAIC ≤ 2 时全部保留为合法候选。
3. 第二分支 `review-b` 记录互相冲突的人工决定（主分支接受一位点，分支接受两位点）。
4. 最终复核：联动页面同时展示功率曲线、注射区间、积分点与拟合残差，
   比较一位点、两位点与稀释假设；每个显示值都有稳定身份。
5. 导出 → 删除本地数据库 → 导入：身份、顺序、分支与来源关系保持。
"""


def generate(outdir: str) -> dict:
    os.makedirs(os.path.join(outdir, "revisions"), exist_ok=True)
    files = {
        "power.csv": build_power_text(),
        "injections.csv": build_injections_text(),
        "concentration.txt": concentration_text(),
        "temperature.txt": "temperature = 25 °C\n",
        "cell_volume.txt": "cell_volume = 1.4 mL\n",
        "baseline.txt": "baseline = 5.0 µcal/s\n",
        "blank.txt": build_blank_text(),
        "narrative.md": NARRATIVE,
        "revisions/concentration_v2.txt": concentration_text(SYRINGE_V2, CELL_CONC),
    }
    manifest = {
        "format": "itc-casepack/1",
        "title": "滴定热谱：ITC 注射积分、稀释热与绑定模型审议",
        "seed": SEED,
        "units_basis": {
            "time": "s（相对采集起点）", "power": "µcal/s", "energy": "cal",
            "volume": "µL 原文 → L 规范化",
            "concentration": "mM 原文 → M 规范化",
            "temperature": "°C 原文 → K 规范化",
        },
        "scenarios": [
            {"code": "overlapping_peaks", "injections": [9, 10]},
            {"code": "first_injection_discarded", "injections": [1]},
            {"code": "gap_and_endpoint", "gap_s": list(GAP), "injections": [14]},
            {"code": "blank_missing", "injections": [13]},
            {"code": "concentration_revision", "v1": SYRINGE_V1, "v2": SYRINGE_V2},
            {"code": "multiple_equivalent_models"},
            {"code": "conflicting_manual_decisions"},
        ],
        "files": sorted(files),
    }
    for name, text in files.items():
        path = os.path.join(outdir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    with open(os.path.join(outdir, "casepack.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "casepacks/itc-titration"
    print(json.dumps(generate(out), ensure_ascii=False, indent=2))
