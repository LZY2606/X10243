# 滴定热谱：ITC 注射积分、稀释热与绑定模型审议

本案例包面向生物物理实验室的完整叙事：导入原始功率曲线、注射时序和空白对照，
审阅每次注射积分与绑定模型候选。所有材料离线可用，不依赖任何云服务。

## 初始版本（分开版本化的不可变材料）

| 材料 | 身份 | 单位 / 坐标基准 |
| --- | --- | --- |
| 功率采样 | `mat-power` v1 | µcal/s，t 为 run 起点秒；1000–1050 s 为检测器复位缺口 |
| 注射体积和时间 | `mat-injections` v1 | µL / s；18 注，首注 2 µL，其余 10 µL，间隔 90 s |
| 浓度 | `mat-concentrations` v1 | 池 0.06 mM，针筒 0.80 mM；杂质 0.3 mg/mL（不可换算，原地保留+诊断） |
| 温度 | `mat-temperature` v1 | 25.0 °C（派生层换算为 K） |
| 池体积 | `mat-cell_volume` v1 | 1400 µL |
| 基线 | `mat-baseline` v1 | 0.8 − 2e-4·t µcal/s |
| 空白曲线 | `mat-blank` v1 | µcal；缺第 4 针（瓶签误读） |

## 回放步骤（`scenario.json` 的 `steps`）

1. `import:materials` — 初始导入，八种材料各自独立版本化。
2. `compute:integration` — 窗宽 90 s、舍弃首注。注射峰相互重叠（τ=35 s，间隔 90 s）
   仍逐窗积分；第 9、10 针的积分窗跨越 1000–1050 s 数据缺口，产生
   `integration_window_crosses_gap` 定位诊断，不伪造数值。
3. `compute:blank_correction` — 规则 `by_index` 显式记录；第 4 针空白对应缺失，
   原始热保留在原位置并附 `blank_correspondence_missing`。
4. `compute:model_fit` — 一位点 / 两位点 / 稀释三族候选；两位点存在交换简并，
   多个等优候选全部保留并稳定排序。
5. `revise:concentrations` — 针筒浓度 0.80 → 0.84 mM（重新称量）。
   既有模型拟合随即过期（stale），但原积分不变。
6. `revise:blank` — 找回第 4 针空白；只影响空白校正及其下游，积分结果不动。
7. `compute:blank_correction` — 基于空白 v2 的新结果，旧结果保留可回查。
8. `compute:model_fit` — 基于浓度 v2 的重新拟合。
9. `compute:model_fit`（后段 14–17 针）— 稀释假设检验：纯稀释模型被数据否定，
   判断依据以机器可读事件保存。
10. `decision:accept`（main 分支）— 接受一位点候选。
11. `decision:reject` + `decision:accept`（review-b 分支）— 审核者 B 否定一位点、
    接受稀释假设；互相冲突的人工决定在两个分支上同时保留。

## 反查路径

终态任意显示值都带稳定身份（`inj-07`、`res-model_fit-…`、`cand-two_site-…`、
`evt-0009`、`dec-0003`）。从任一候选可反查：所属拟合事件 → 输入摘要与参数版本 →
材料版本（浓度 v1/v2、空白 v1/v2）→ 造成差异的具体修订事件与人工决定。
导出 → 删除本地数据库 → 再导入后，身份、顺序与来源关系保持不变。
