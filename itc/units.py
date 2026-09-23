import math
from .errors import SemanticNoSolutionError, diagnostic

_TIME = {"s": 1.0, "sec": 1.0, "second": 1.0, "min": 60.0, "minute": 60.0}
_POWER_PREFIX = {"µ": 1e-6, "u": 1e-6, "m": 1e-3, "": 1.0}
_ENERGY_PREFIX = {"µ": 1e-6, "u": 1e-6, "m": 1e-3, "": 1.0}
_VOLUME = {"l": 1.0, "ml": 1e-3, "µl": 1e-6, "ul": 1e-6, "mL": 1e-3}
_CONCENTRATION = {"m": 1e-3, "mm": 1e-3, "µm": 1e-6, "um": 1e-6, "nm": 1e-9, "": 1.0}


def _number(raw, path):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise SemanticNoSolutionError("无法把原始表示转换为数值", [diagnostic(path, "必须是数字或数字字符串", raw)])
    if not math.isfinite(value):
        raise SemanticNoSolutionError("数值必须有限", [diagnostic(path, "不接受 NaN 或无穷值", raw)])
    return value


def normalize_time(raw, unit, path="time"):
    value = _number(raw, path)
    factor = _TIME.get(str(unit).strip())
    if factor is None:
        raise SemanticNoSolutionError("未知时间单位", [diagnostic(f"{path}.unit", "支持 s 与 min", unit)])
    return {"raw": raw, "unit": unit, "normalized_value": value * factor, "normalized_unit": "s"}


def normalize_power(raw, unit, path="power"):
    text = str(unit).replace("μ", "µ").strip()
    if not text.endswith("W"):
        raise SemanticNoSolutionError("未知功率单位", [diagnostic(f"{path}.unit", "支持 µW、mW、W", unit)])
    prefix = text[:-1]
    factor = _POWER_PREFIX.get(prefix)
    if factor is None:
        raise SemanticNoSolutionError("未知功率词头", [diagnostic(f"{path}.unit", "支持 µ、m 或无词头", unit)])
    value = _number(raw, path)
    return {"raw": raw, "unit": unit, "normalized_value": value * factor * 1e6, "normalized_unit": "µW"}


def normalize_energy(raw, unit, path="energy"):
    text = str(unit).replace("μ", "µ").strip()
    if text in ("cal",):
        factor, name = 4.184, "J"
    elif text.endswith("cal"):
        prefix = text[:-3]
        mult = _ENERGY_PREFIX.get(prefix)
        if mult is None:
            raise SemanticNoSolutionError("未知热量词头", [diagnostic(f"{path}.unit", "支持 µcal、mcal、cal", unit)])
        factor, name = mult * 4.184, "J"
    elif text.endswith("J"):
        prefix = text[:-1]
        mult = _ENERGY_PREFIX.get(prefix)
        if mult is None:
            raise SemanticNoSolutionError("未知能量词头", [diagnostic(f"{path}.unit", "支持 µJ、mJ、J", unit)])
        factor, name = mult, "J"
    else:
        raise SemanticNoSolutionError("未知能量单位", [diagnostic(f"{path}.unit", "支持 µJ、mJ、J、µcal、mcal、cal", unit)])
    value = _number(raw, path)
    return {"raw": raw, "unit": unit, "normalized_value": value * factor * 1e6, "normalized_unit": "µJ"}


def normalize_volume(raw, unit, path="volume"):
    text = str(unit).strip()
    factor = _VOLUME.get(text)
    if factor is None:
        raise SemanticNoSolutionError("未知体积单位", [diagnostic(f"{path}.unit", "支持 L、mL、µL", unit)])
    value = _number(raw, path)
    if value <= 0:
        raise SemanticNoSolutionError("体积必须为正", [diagnostic(path, "必须大于 0", raw, "(0, inf)")])
    return {"raw": raw, "unit": unit, "normalized_value": value * factor * 1e6, "normalized_unit": "µL"}


def normalize_concentration(raw, unit, path="concentration"):
    text = str(unit).replace("μ", "µ").strip().lower()
    factor = _CONCENTRATION.get(text)
    if factor is None:
        raise SemanticNoSolutionError("未知浓度单位", [diagnostic(f"{path}.unit", "支持 M、mM、µM、nM", unit)])
    value = _number(raw, path)
    if value < 0:
        raise SemanticNoSolutionError("浓度不能为负", [diagnostic(path, "必须不小于 0", raw, "[0, inf)")])
    return {"raw": raw, "unit": unit, "normalized_value": value * factor * 1e3, "normalized_unit": "mM"}


def normalize_temperature(raw, unit, path="temperature"):
    value = _number(raw, path)
    text = str(unit).strip()
    if text in ("K", "kelvin"):
        kelvin = value
    elif text in ("C", "°C", "degC", "celsius"):
        kelvin = value + 273.15
    else:
        raise SemanticNoSolutionError("未知温度单位", [diagnostic(f"{path}.unit", "支持 K 与 °C", unit)])
    if kelvin <= 0:
        raise SemanticNoSolutionError("温度必须大于绝对零度", [diagnostic(path, "必须 > 0 K", raw, "(0, inf) K")])
    return {"raw": raw, "unit": unit, "normalized_value": kelvin, "normalized_unit": "K"}


def normalized_scalar(payload, kind, path):
    if not isinstance(payload, dict):
        raise SemanticNoSolutionError("标量材料必须携带值和单位", [diagnostic(path, "需要 {value, unit}", payload)])
    func = {
        "time": normalize_time,
        "power": normalize_power,
        "energy": normalize_energy,
        "volume": normalize_volume,
        "concentration": normalize_concentration,
        "temperature": normalize_temperature,
    }[kind]
    result = func(payload.get("value"), payload.get("unit"), path)
    result["coordinate_basis"] = payload.get("coordinate_basis", "material_local_clock")
    return result
