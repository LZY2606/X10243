"""单位与坐标基准：导入时保留原始表示，派生层才规范化。

无法转换的值留在原位置并附诊断，不为画图而丢弃。
"""

CANONICAL = {
    "time": "s",
    "power": "uJ/s",
    "heat": "uJ",
    "volume": "uL",
    "concentration": "uM",
    "temperature": "K",
}

# 线性换算因子：canonical = value * factor
_LINEAR = {
    ("s", "s"): 1.0, ("ms", "s"): 1e-3, ("min", "s"): 60.0,
    ("uJ/s", "uJ/s"): 1.0, ("ucal/s", "uJ/s"): 4.184, ("mcal/s", "uJ/s"): 4184.0,
    ("uJ", "uJ"): 1.0, ("ucal", "uJ"): 4.184, ("mcal", "uJ"): 4184.0, ("J", "uJ"): 1e6,
    ("nL", "uL"): 1e-3, ("uL", "uL"): 1.0, ("mL", "uL"): 1e3,
    ("nM", "uM"): 1e-3, ("uM", "uM"): 1.0, ("mM", "uM"): 1e3,
    ("K", "K"): 1.0,
}

# 带偏移换算：canonical = value * factor + offset
_OFFSET = {
    ("degC", "K"): (1.0, 273.15),
}


def normalize_value(value, unit, kind, position):
    """返回 (canonical_value, canonical_unit, diagnostic_or_None)。

    无法转换时返回 (None, None, diagnostic)，原始值由调用方留在原位置。
    """
    canon = CANONICAL.get(kind)
    if canon is None:
        return None, None, {
            "position": position, "code": "unknown_quantity_kind",
            "message": f"未知量纲类别 {kind!r}，保留原值 {value!r} {unit!r}",
        }
    if unit is None:
        unit = canon  # 缺省即声明为规范单位（显式声明，非伪装）
    key = (unit, canon)
    if key in _LINEAR:
        return value * _LINEAR[key], canon, None
    if key in _OFFSET:
        factor, offset = _OFFSET[key]
        return value * factor + offset, canon, None
    return None, None, {
        "position": position, "code": "unit_unconvertible",
        "message": f"单位 {unit!r} 无法转换为 {canon}，原值 {value!r} 保留在原位置",
    }


def normalize_series(values, unit, kind, position):
    """规范化数组；返回 (values_or_None, unit, diagnostic_or_None)。"""
    canon = CANONICAL[kind]
    if unit is None:
        unit = canon
    key = (unit, canon)
    if key in _LINEAR:
        factor = _LINEAR[key]
        return [v * factor for v in values], canon, None
    if key in _OFFSET:
        factor, offset = _OFFSET[key]
        return [v * factor + offset for v in values], canon, None
    return None, None, {
        "position": position, "code": "unit_unconvertible",
        "message": f"序列单位 {unit!r} 无法转换为 {canon}",
    }
