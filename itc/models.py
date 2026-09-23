"""共享常量与错误类型。

三类必须可区分的失败结果（资源预算耗尽 / 格式损坏 / 语义无解），
外加带位置的诊断结构。任何路径都不得用空值或默认值伪装成功。
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---- 事件类型（机器可读取的事件；不只是日志）----
EV_POWER_IMPORTED = "power_imported"
EV_INJECTIONS_IMPORTED = "injections_imported"
EV_CONCENTRATION_IMPORTED = "concentration_imported"
EV_CONDITIONS_IMPORTED = "conditions_imported"
EV_BASELINE_IMPORTED = "baseline_imported"
EV_BLANK_IMPORTED = "blank_imported"

EV_POWER_PATCHED = "power_patched"            # 只影响局部依赖的源数据改动
EV_CONCENTRATION_REVISED = "concentration_revised"  # 浓度修订：模型过期、原积分不动

EV_FIRST_DISCARD_SET = "first_discard_set"    # 首注舍弃（规则决定）
EV_INTEGRATION_RUN = "integration_run"
EV_BLANK_MATCH_SET = "blank_match_set"        # 稀释热对应规则（必须显式说明）
EV_FIT_RUN = "fit_run"
EV_DECISION = "decision"                      # accept / reject / adjust
EV_INVALIDATED = "invalidated"                # 依赖失效（如浓度修订使模型过期）

MATERIAL_KINDS = (
    "power", "injections", "concentration",
    "conditions", "baseline", "blank",
)
KIND_FILE = {
    "power": "power.csv",
    "injections": "injections.csv",
    "concentration": "concentration.json",
    "conditions": "conditions.json",
    "baseline": "baseline.csv",
    "blank": "blank.csv",
}

# ---- 候选模型（排序键稳定；等价输入得到稳定顺序）----
MODEL_ONE_SITE = "one_site"
MODEL_TWO_SITE = "two_site"
MODEL_DILUTION = "dilution_only"
MODEL_ORDER = (MODEL_DILUTION, MODEL_ONE_SITE, MODEL_TWO_SITE)
MODEL_LABELS = {
    MODEL_ONE_SITE: "一位点 1:1 绑定",
    MODEL_TWO_SITE: "两位点独立绑定",
    MODEL_DILUTION: "稀释热假设（无结合）",
}

# 判定两个候选“等优”的相对阈值：|sse_a-sse_b| <= tol*(1+|sse_b|)
TIE_REL_TOL = 1e-6
# 积分重叠判定容差（秒）
EPS_TIME = 1e-9
# 内部计算的规范单位（派生层），原始材料始终保留
CANON_ENERGY = "uJ"
CANON_POWER = "uW"
CANON_TIME = "s"
CANON_CONC = "mM"
CANON_VOLUME = "uL"
CANON_TEMP = "degC"


@dataclass
class Diagnostic:
    """带位置的诊断。pos 指向材料内具体位置（如 row、interval）。"""
    code: str
    message: str
    pos: str = ""
    severity: str = "error"          # error | warning | info
    value: str | None = None         # 无法转换时，原值以字符串留在原位

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "pos": self.pos,
            "severity": self.severity,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Diagnostic":
        return cls(
            code=d["code"], message=d["message"], pos=d.get("pos", ""),
            severity=d.get("severity", "error"), value=d.get("value"),
        )


class ITCError(Exception):
    """语义无解：输入能解析，但在领域规则下没有合法结果。"""

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


class FormatCorrupted(ITCError):
    """格式损坏：原始表示无法按声明格式解析。"""


class BudgetExhausted(ITCError):
    """资源预算耗尽：输入规模超过允许的样本/对象上限。"""


def ensure(cond: bool, code: str, message: str, pos: str = "",
           diagnostics: list[Diagnostic] | None = None) -> None:
    if not cond:
        diags = list(diagnostics or [])
        diags.append(Diagnostic(code=code, message=message, pos=pos))
        raise ITCError(message, diags)
