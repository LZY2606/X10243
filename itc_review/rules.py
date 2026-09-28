"""可修改决定之一：分析规则版本。

规则是被版本化的“决定”，与不可变材料分离。内容相同的规则得到相同身份。
"""
from __future__ import annotations

from .ids import digest

DEFAULTS = {
    "pre_window_s": 30.0,        # 注射前基线窗（秒）
    "post_window_s": 150.0,      # 注射后积分窗（秒）
    "gap_factor": 3.0,           # 相邻采样间隔 > gap_factor * 中位间隔 => 数据缺口
    "blank_correspondence": "index",  # index（按注射序号对应）| nearest_time
    "discard_first": True,       # 首注舍弃
    "tie_epsilon_aicc": 2.0,     # |ΔAICc| <= epsilon 视为等优
    "fit_evaluations": 4000,     # 拟合求值预算
}

REQUIRED = ("pre_window_s", "post_window_s", "gap_factor", "blank_correspondence",
            "discard_first", "tie_epsilon_aicc", "fit_evaluations")

ALLOWED_CORRESPONDENCE = ("index", "nearest_time")


def build_rule_version(values: dict | None = None, source: str = "rules") -> dict:
    merged = dict(DEFAULTS)
    if values:
        merged.update({k: v for k, v in values.items() if k in DEFAULTS})
    from .errors import SemanticUnresolvable, Diagnostic

    diags = []
    if merged["blank_correspondence"] not in ALLOWED_CORRESPONDENCE:
        diags.append(Diagnostic(
            "INVALID_CORRESPONDENCE",
            f"{source}.blank_correspondence",
            f"空白对应规则必须是 {ALLOWED_CORRESPONDENCE} 之一",
            merged["blank_correspondence"],
        ))
    if merged["post_window_s"] <= 0 or merged["pre_window_s"] < 0:
        diags.append(Diagnostic("INVALID_WINDOW", source, "积分窗必须为正"))
    if merged["gap_factor"] <= 1:
        diags.append(Diagnostic("INVALID_GAP_FACTOR", source, "gap_factor 必须 > 1"))
    if diags:
        raise SemanticUnresolvable("规则参数语义无解", diags)
    payload = {
        "kind": "rules",
        "source": source,
        "values": merged,
    }
    payload["id"] = digest("rules", [payload["source"], merged])
    return payload
