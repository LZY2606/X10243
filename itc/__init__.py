"""ITC 注射积分 / 稀释热 / 绑定模型审议应用。

分层约定（模块边界即业务边界，均可直接被测试调用）：

units      单位与坐标基准：导入时保留原始表示，派生层才规范化
ident      稳定身份 / 规则版本 / 内容指纹
diagnostics 带位置的诊断（缺失输入绝不伪装成成功）
ingest     原始材料解析（功率曲线、注射时序、浓度、基线、空白）
materials  不可变材料版本（每类材料独立版本化）
integration 注射区间与峰积分（允许峰重叠，积分窗不跨数据缺口）
blank      空白/稀释热对应与扣除
models     一位点、多位点、稀释假设候选拟合
ledger     仅追加事件账本（分支、决定、结构化判定）
storage    SQLite 存储、多步导入操作与半写恢复
project    导出 / 导入往返
replay     重放案例脚本
case_pack  离线案例包（滴定热谱领域材料）
service    领域服务编排（FastAPI 之外的纯 Python 入口）
web        SVG 渲染与页面视图
"""

RULES_VERSION = "itc.rules.1.0.0"
