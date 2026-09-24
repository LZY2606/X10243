# 滴定热谱 ITC 审议应用

本地 FastAPI + NumPy + SQLite + SVG 应用，面向 ITC 功率曲线、注射时序、空白对照、每次注射积分和绑定模型候选审议。

安装：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

测试与启动：

```bash
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 5354
```

打开 `http://127.0.0.1:5354` 即进入自带离线案例审阅页。

领域案例包位于 `cases/itc_review/`，包含原始功率、基线、空白、注射时序、浓度、重放事件和说明；不是通用表格。
