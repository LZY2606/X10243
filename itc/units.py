"""单位与坐标基准。

材料层保留原始表示（原样字符串、原始单位、原始时间基准）；
只有派生层（积分 / 拟合）才调用这里的规范化。无法转换的值
留在原位置并附诊断，绝不为了画图而丢掉。
"""

from .outcomes import diag

# 规范化目标：时间 s，功率 uJ/s，体积 uL，浓度 M，温度 K。
_FACTORS = {
    "time": {"s": 1.0, "sec": 1.0, "min": 60.0},
    "power": {"uJ/s": 1.0, "µJ/s": 1.0, "mcal/s": 4184.0, "uJ.sec-1": 1.0},
    "volume": {"uL": 1.0, "µL": 1.0, "mL": 1000.0},
    "conc": {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6},
    "temp": {"K": 1.0},
}


def factor(kind, unit):
    """返回换算系数；无法转换返回 None。"""
    table = _FACTORS.get(kind, {})
    if unit in table:
        return table[unit]
    if kind == "temp" and unit in ("C", "degC"):
        return None  # 需要偏移量，走专门函数
    return None


def convert_temp(value, unit):
    if unit == "K":
        return float(value)
    if unit in ("C", "degC"):
        return float(value) + 273.15
    return None


def normalize_samples(payload, stream):
    """把 {time_unit, unit, samples:[[t, y], ...]} 规范化。

    返回 (t_list, y_list, diagnostics)。单位无法转换时返回
    (None, None, diagnostics)，原始 payload 仍留在材料层。
    单个坏行跳过并给出带下标的诊断，不丢原始行。
    """
    diagnostics = []
    t_unit = payload.get("time_unit", "s")
    y_unit = payload.get("unit")
    tf = factor("time", t_unit)
    if tf is None:
        diagnostics.append(diag(
            "unit_unconvertible", f"{stream}.time_unit",
            f"时间单位无法转换: {t_unit!r}", value=t_unit))
        return None, None, diagnostics
    yf = factor("power", y_unit)
    if yf is None:
        diagnostics.append(diag(
            "unit_unconvertible", f"{stream}.unit",
            f"功率单位无法转换: {y_unit!r}", value=y_unit))
        return None, None, diagnostics
    ts, ys = [], []
    for i, row in enumerate(payload.get("samples", [])):
        try:
            t, y = float(row[0]), float(row[1])
        except (TypeError, ValueError, IndexError):
            diagnostics.append(diag(
                "sample_not_numeric", f"{stream}.samples[{i}]",
                f"采样行不是数值，保留在原始材料中: {row!r}", row=row))
            continue
        ts.append(t * tf)
        ys.append(y * yf)
    return ts, ys, diagnostics


def normalize_injections(payload):
    """{time_unit, volume_unit, injections:[[t, vol], ...]} -> 规范化列表。"""
    diagnostics = []
    tf = factor("time", payload.get("time_unit", "s"))
    vf = factor("volume", payload.get("volume_unit", "uL"))
    if tf is None or vf is None:
        bad = payload.get("time_unit") if tf is None else payload.get("volume_unit")
        diagnostics.append(diag(
            "unit_unconvertible", "injections",
            f"注射时序单位无法转换: {bad!r}", value=bad))
        return None, diagnostics
    out = []
    for i, row in enumerate(payload.get("injections", [])):
        try:
            t, v = float(row[0]), float(row[1])
        except (TypeError, ValueError, IndexError):
            diagnostics.append(diag(
                "sample_not_numeric", f"injections.injections[{i}]",
                f"注射行不是数值: {row!r}", row=row))
            continue
        out.append({"t": t * tf, "volume_uL": v * vf, "source_index": i})
    return out, diagnostics
