"""ITC 注射积分 / 稀释热 / 绑定模型审议领域包。

所有业务规则（积分窗、空白对应、绑定拟合、版本失效）都在本包内实现，
FastAPI 与 SQLite 只是外壳，函数可被测试直接调用。
"""

__version__ = "1.0.0"
RULE_VERSIONS = {
    "integrate": "integrate@3",
    "blank_rule": "blank-rule@2",
    "binding": "binding@4",
}
