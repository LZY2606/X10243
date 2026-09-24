"""贯穿材料、事件与派生层的稳定标识与规则版本。

规则版本是计算身份的一部分：规则改动会产生新的内容寻址节点，
旧事件重放时仍解析到旧节点，历史结论不被覆盖。
"""

SCHEMA_VERSION = "1.0.0"

# 不可变材料槽（分开版本化）
SLOT_POWER = "power"
SLOT_INJECTIONS = "injections"
SLOT_CONCENTRATIONS = "concentrations"
SLOT_TEMPERATURE = "temperature"
SLOT_CELL_VOLUME = "cell_volume"
SLOT_BASELINE = "baseline"
SLOT_BLANK = "blank"
MATERIAL_SLOTS = (
    SLOT_POWER,
    SLOT_INJECTIONS,
    SLOT_CONCENTRATIONS,
    SLOT_TEMPERATURE,
    SLOT_CELL_VOLUME,
    SLOT_BASELINE,
    SLOT_BLANK,
)
SLOT_LABELS_ZH = {
    SLOT_POWER: "功率采样",
    SLOT_INJECTIONS: "注射体积与时间",
    SLOT_CONCENTRATIONS: "浓度",
    SLOT_TEMPERATURE: "温度",
    SLOT_CELL_VOLUME: "池体积",
    SLOT_BASELINE: "基线",
    SLOT_BLANK: "空白对照曲线",
}

# 决定类型
DEC_DISCARD_FIRST = "discard_first"
DEC_DISCARD_INJECTIONS = "discard_injections"
DEC_ENDPOINT_OVERRIDE = "endpoint_override"
DEC_BLANK_RULE = "blank_rule"
DEC_POLICY = "policy"
DEC_MODEL_VERDICT = "model_verdict"

# 空白对应规则
BLANK_RULE_INDEX = "match_by_index"
BLANK_RULE_VOLUME = "match_by_volume"
BLANK_RULE_NONE = "no_blank"
BLANK_RULE_VERSION = "blank-match-v1"

# 积分 / 归一规则版本
WINDOW_RULE_VERSION = "window-v2"
INTEGRATION_RULE_VERSION = "integration-trapezoid-v1"
NORMALIZATION_RULE_VERSION = "normalization-wiseman-v1"

# 派生节点类型
NODE_WINDOW = "window"
NODE_RAW_HEAT = "raw_heat"
NODE_BLANK_MAP = "blank_map"
NODE_NORMALIZED = "normalized_heat"
NODE_ISOTHERM = "isotherm"
NODE_FIT = "fit"

# 节点 / 窗口状态
STATUS_OK = "ok"
STATUS_BLOCKED_GAP = "blocked:gap"
STATUS_BLOCKED_BASELINE = "blocked:baseline_missing"
STATUS_BLOCKED_INJECTION = "blocked:injection_missing"
STATUS_DISCARDED = "discarded"
STATUS_BLANK_MISSING = "blank_missing"
STATUS_EXPIRED = "expired"

# 事件类型
EVT_PROJECT_CREATED = "project_created"
EVT_MATERIAL_UPLOADED = "material_uploaded"
EVT_DECISION = "decision_applied"
EVT_REVIEW = "review_verdict"
EVT_BRANCH_OPENED = "branch_opened"
EVT_NOTE = "note"

# 三种互不等价的失败结果
ERR_BUDGET = "resource_budget_exhausted"
ERR_FORMAT = "format_corrupt"
ERR_SEMANTIC = "semantic_unsolvable"

# 时间基准
TIME_BASE = "seconds_from_acquisition_start"
