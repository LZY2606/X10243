"""三类必须区分的结果：资源预算耗尽 / 格式损坏 / 语义无解。

所有领域操作返回 dict，至少含 ``status`` 与 ``diagnostics``。
诊断永远带位置（path / stream / injection），不允许用空值或默认值伪装成功。
"""

OK = "ok"
BUDGET_EXHAUSTED = "budget_exhausted"
CORRUPT_FORMAT = "corrupt_format"
SEMANTIC_UNSOLVABLE = "semantic_unsolvable"


def ok(**extra):
    out = {"status": OK, "diagnostics": []}
    out.update(extra)
    return out


def failure(status, diagnostics, **extra):
    assert status in (BUDGET_EXHAUSTED, CORRUPT_FORMAT, SEMANTIC_UNSOLVABLE), status
    out = {"status": status, "diagnostics": list(diagnostics)}
    out.update(extra)
    return out


def diag(code, path, message, **extra):
    d = {"code": code, "path": path, "message": message}
    d.update(extra)
    return d
