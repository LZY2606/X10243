"""生成离线领域案例包 casepack/：滴定热谱——ITC 注射积分、稀释热与绑定模型审议。

叙事（v1）：
  13 次注射，首注 0.4 µL 舍弃；inj:4/inj:5 仅隔 50 s（峰重叠）；
  1148-1212 s 存在数据缺口（落在 inj:6 窗内）；空白对照 v1 只覆盖到 1700 s，
  inj:8 之后空白对应缺失。真值：一位点 K=1e6 /M、ΔH=-40 kJ/mol。
单位刻意多样：时间 s、功率 uJ/s、体积 mL、温度 °C；并放一个无法转换的字段。
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = os.path.join(HERE, "casepack")

DT = 2.0
T_END = 2660.0
TAU = 25.0
V0_UL = 1400.0
CELL_MM = 0.05
SYR_MM_1, SYR_MM_2 = 1.80, 1.76
K_PER_MM, DH, N = 1.0e3, -4.0e4, 1.0

INJECTIONS = [
    (0, 100.0, 0.4),
    (1, 300.0, 5.0), (2, 480.0, 5.0), (3, 660.0, 5.0),
    (4, 840.0, 5.0), (5, 890.0, 5.0),          # 重叠峰（间隔 50 s）
    (6, 1070.0, 5.0),                           # 窗内含缺口
    (7, 1250.0, 5.0), (8, 1430.0, 5.0), (9, 1610.0, 5.0),
    (10, 1790.0, 5.0), (11, 1970.0, 5.0), (12, 2150.0, 5.0),
]
GAP = (1148.0, 1212.0)   # 删除 (1148,1212) 之间样本：缺口区间
BLANK_V1_END = 1700.0


def _dilution_heats(n: int) -> np.ndarray:
    i = np.arange(n)
    return -60.0 * np.exp(-i / 6.0)   # µJ，仅空白/稀释贡献


def _binding_heats(volumes: list[float], syr: float) -> np.ndarray:
    import sys
    sys.path.insert(0, HERE)
    from itc.binding import Conditions, model_heats
    cond = Conditions(CELL_MM, syr, V0_UL, 298.15, volumes)
    return model_heats(
        "one_site", np.array([np.log(K_PER_MM), DH, N]), cond)


def _step_curve(times: np.ndarray, heats: np.ndarray,
                noise_seed: int, with_baseline: bool) -> np.ndarray:
    """阶跃响应合成曲线：每次注射把累积热水平抬升 q_i。

    抬升在区间内以指数形式出现（功率 = Σ Δq/τ·exp(-(t-t_i)/τ)），
    到达下一注射时刻前已完全展开，因此 [t_i,t_{i+1}] 积分恒等于 q_i。
    50 s 间隔的重叠峰通过把第 4 次注射的抬升时间压缩到 50 s 实现，
    窗内份额随之变小并被标记 overlap。
    """
    rng = np.random.RandomState(noise_seed)
    if with_baseline:
        p = -12.0 + 0.0004 * times + rng.normal(0, 0.10, size=times.shape)
    else:
        p = rng.normal(0, 0.04, size=times.shape)
    p = p.copy()
    for k, ((idx, ti, _v), q) in enumerate(zip(INJECTIONS, heats)):
        if k + 1 < len(INJECTIONS):
            span = INJECTIONS[k + 1][1] - ti
            # 抬升在 99.5% 完成于下一注射前（τ 按区间宽度收缩）
            tau_k = max(span / 6.0, 4.0)
        else:
            tau_k = TAU
        m = times >= ti
        p[m] += (q / tau_k) * np.exp(-(times[m] - ti) / tau_k)
    return p


_curve = _step_curve


def _write_csv(path: str, headers: dict[str, str], times: np.ndarray,
               p: np.ndarray) -> None:
    lines = [f"# {k}: {v}" for k, v in headers.items()]
    lines += ["time,power"]
    for tt, pp in zip(times, p):
        lines.append(f"{tt:.1f},{pp:.6f}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    os.makedirs(CASE, exist_ok=True)
    volumes = [v for _, _, v in INJECTIONS]
    q_bind = _binding_heats(volumes, SYR_MM_1)
    q_dil = _dilution_heats(len(INJECTIONS))
    q_raw = q_bind + q_dil

    times = np.arange(0.0, T_END + 1e-9, DT)
    keep = ~((times > GAP[0]) & (times < GAP[1]))
    t_gap = times[keep]
    p_main = _curve(t_gap, q_raw, noise_seed=7, with_baseline=True)
    _write_csv(os.path.join(CASE, "power_curve_v1.csv"),
               {"kind": "power", "time_unit": "s", "power_unit": "uJ/s"},
               t_gap, p_main)

    # 基线：稀疏点（每 20 s），覆盖全程
    tb = np.arange(0.0, T_END + 1e-9, 20.0)
    pb = -12.0 + 0.0004 * tb
    _write_csv(os.path.join(CASE, "baseline_v1.csv"),
               {"kind": "baseline", "time_unit": "s",
                "power_unit": "uJ/s"}, tb, pb)

    # 空白 v1：仅到 1700 s
    tb1 = np.arange(0.0, BLANK_V1_END + 1e-9, DT)
    pb1 = _curve(tb1, q_dil, noise_seed=13, with_baseline=False)
    _write_csv(os.path.join(CASE, "blank_curve_v1.csv"),
               {"kind": "blank", "time_unit": "s", "power_unit": "uJ/s"},
               tb1, pb1)

    # 空白 v2：覆盖全程（局部源数据改动，只影响原本缺失对应的注射）
    tb2 = np.arange(0.0, 2400.0 + 1e-9, DT)
    pb2 = _curve(tb2, q_dil, noise_seed=13, with_baseline=False)
    _write_csv(os.path.join(CASE, "blank_curve_v2.csv"),
               {"kind": "blank", "time_unit": "s", "power_unit": "uJ/s"},
               tb2, pb2)

    # 注射时序
    with open(os.path.join(CASE, "injections_v1.csv"), "w",
              encoding="utf-8") as fh:
        fh.write("# kind: schedule\n# time_unit: s\n# volume_unit: uL\n")
        fh.write("index,time,volume\n")
        for idx, ti, v in INJECTIONS:
            fh.write(f"{idx},{ti:.1f},{v}\n")

    # 浓度 v1（含一个无法转换的备用字段，原值保留 + 诊断）
    conc1 = {
        "cell": {"value": 0.10, "unit": "mM"},
        "syringe": {"value": 1.80, "unit": "mM"},
        "cell_concentration_alt": {"value": "0.1?", "unit": "mM"},
        "note": "操作者手抄备份值，字迹不清，保留不用于计算",
    }
    with open(os.path.join(CASE, "concentrations_v1.json"), "w",
              encoding="utf-8") as fh:
        json.dump(conc1, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(CASE, "concentrations_v2.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "cell": {"value": 0.10, "unit": "mM"},
            "syringe": {"value": 1.76, "unit": "mM"},
            "revision_reason": "复称后滴定剂浓度修订为 1.76 mM",
        }, fh, ensure_ascii=False, indent=2)

    with open(os.path.join(CASE, "temperature_v1.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"temperature": {"value": 25.0, "unit": "C"}}, fh,
                  ensure_ascii=False, indent=2)
    with open(os.path.join(CASE, "cell_volume_v1.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"cell_volume": {"value": 1.4, "unit": "mL"}}, fh,
                  ensure_ascii=False, indent=2)

    with open(os.path.join(CASE, "narrative.md"), "w",
              encoding="utf-8") as fh:
        fh.write(NARRATIVE)
    print("casepack written:", CASE)


NARRATIVE = """# 滴定热谱：ITC 注射积分、稀释热与绑定模型审议（离线案例）

## 叙事主线
1. 生物物理实验室导入原始功率曲线、注射时序、空白对照，以及浓度、温度、池体积、基线。
2. 审阅逐注射积分：首注 0.4 µL 舍弃；inj:4 与 inj:5 间隔 50 s，注射后峰重叠（允许并标记）；
   inj:6 窗口包含 1148–1212 s 数据缺口，积分窗在缺口处裁剪，不跨越缺口；
   空白曲线 v1 只到 1700 s，inj:8 之后稀释热对应缺失，heat_net 留空并附诊断。
3. 比较一位点 / 两位点 / 稀释假设。
4. 参数修订：注射器浓度 1.80 → 1.76 mM（concentrations v2），既有模型拟合立即过期，
   但原积分不依赖浓度，数值与身份不变。
5. 只影响局部依赖的源数据改动：空白曲线 v2 覆盖全程，重新积分后仅 inj:8..inj:12 的
   净热发生变化；其余注射逐点一致。
6. 互相冲突的人工决定：main 接受一位点；另一分支否定一位点并接受两位点，两条决定都保留。

## 坐标基准与单位
时间 s；功率 µJ/s；热 µJ；体积 µL（池体积原文 1.4 mL）；浓度 mM；温度原文 25 °C。
所有量在派生层规范化，导入层保留原始文本与原始单位表示。

## 真值（供回收测试）
K = 1.0e6 /M，ΔH = -40000 J/mol，n = 1；稀释热 q_i = -60*exp(-i/6) µJ。
"""


if __name__ == "__main__":
    main()
