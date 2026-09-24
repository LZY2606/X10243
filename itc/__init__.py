"""ITC 滴定热谱审议：注射积分、稀释热扣除与绑定模型候选。

分层：
  units       量纲/单位换算与带单位原始值
  parsing     原始 CSV 导入（保留原始表示，不可转换留在原位置）
  materials   不可变、内容寻址、版本化材料
  integration 注射积分（峰可重叠，积分窗不跨数据缺口）
  binding     一位点 / 多位点 / 稀释假设候选
  derivation  派生层（输入摘要 + 参数版本 + 父级依赖，过期判断）
  events      机器可读事件与分支重放
  store       SQLite 存储与半写恢复
  archive     导出 / 导入往返（zip）
  svg         服务端联动图（热谱、注射区间、积分点、残差）
  casepack    离线领域案例包加载器
  engine      用例编排（业务规则在服务端）
"""

VERSION = "1.0.0"
