"""原始材料导入：保留原始表示，逐格附诊断；格式损坏单独成类。"""

import csv
import io

from .errors import CorruptFormat, Diagnostic
from .units import parse_quantity


def parse_csv(text: str, *, path="<input>"):
    """严格 CSV：未闭合引号等结构问题 => CORRUPT_FORMAT。"""
    if not isinstance(text, str):
        raise CorruptFormat("输入不是文本", code="NOT_TEXT", path=path)
    reader = csv.reader(io.StringIO(text), strict=True, skipinitialspace=False)
    try:
        rows = list(reader)
    except csv.Error as exc:
        raise CorruptFormat(f"CSV 结构损坏：{exc}", code="BAD_CSV", path=path)
    if not rows:
        raise CorruptFormat("空文件：缺少表头", code="EMPTY_FILE", path=path)
    header = [h.strip() for h in rows[0]]
    if any(h == "" for h in header):
        raise CorruptFormat("表头存在空列名", code="BLANK_HEADER", path=path)
    if len(set(header)) != len(header):
        raise CorruptFormat(f"表头列名重复：{header}", code="DUP_HEADER", path=path)
    data = []
    for i, row in enumerate(rows[1:], start=2):
        if not any(c.strip() for c in row):
            continue  # 纯空行忽略（不携带任何信息）
        if len(row) != len(header):
            raise CorruptFormat(
                f"第 {i} 行列数 {len(row)} 与表头 {len(header)} 不一致",
                code="RAGGED_ROW", path=f"{path}:row{i}")
        data.append({header[j]: row[j] for j in range(len(header))})
    return {"header": header, "rows": data, "path": path}


def _col(dim):
    return {"time": "time", "power": "power"}.get(dim, dim)


def _qty_from_header(row, col, dim, line, diags):
    raw = row.get(col)
    q = parse_quantity(raw, dim, path=f"row{line}:{col}")
    if q.diagnostic:
        diags.append(q.diagnostic)
    return q.to_dict()


def parse_trace(text, *, kind, path="<trace>"):
    """功率/基线/空白曲线：表头形如 time_s,power_µcal/s；status 可标 gap。"""
    doc = parse_csv(text, path=path)
    header = doc["header"]
    time_col = next((h for h in header if h.startswith("time")), None)
    power_col = next((h for h in header if h.startswith("power")), None)
    if not time_col or not power_col:
        raise CorruptFormat(
            f"{kind} 曲线需要 time_* 与 power_* 两列，得到 {header}",
            code="BAD_TRACE_HEADER", path=path)
    time_dim = "time"
    samples = []
    diags = []
    for n, row in enumerate(doc["rows"], start=2):
        tq = _qty_from_header(row, time_col, time_dim, n, diags)
        pq = _qty_from_header(row, power_col, "power", n, diags)
        status = (row.get("status") or "").strip().lower()
        is_gap = status == "gap"
        if is_gap:
            diags.append(Diagnostic(
                "MARKED_GAP", f"第 {n} 行被标记为数据缺口",
                path=f"row{n}:status", severity="warning",
                related=[f"sample@{n - 2}"]).to_dict())
        valid = tq["converted"] and pq["converted"] and not is_gap
        if not valid and not is_gap:
            diags.append(Diagnostic(
                "INVALID_SAMPLE",
                f"第 {n} 行样本不可用（时间或功率无法规范化），按缺口保留原位置",
                path=f"row{n}", severity="error",
                related=[f"sample@{n - 2}"]).to_dict())
        samples.append({
            "ordinal": n - 2,
            "sourceRow": n,
            "time": tq,
            "power": pq,
            "status": status or "ok",
            "valid": valid,
        })
    return {
        "kind": kind,
        "path": path,
        "timeHeader": time_col,
        "powerHeader": power_col,
        "samples": samples,
        "diagnostics": diags,
    }


def parse_injections(text, *, path="<injections>"):
    """注射时序：injection_no, t_inj_<time>, volume_<vol>, win_pre_<t>, win_post_<t>。"""
    doc = parse_csv(text, path=path)
    header = doc["header"]
    required_prefix = ("injection_no", "t_inj", "volume", "win_pre", "win_post")
    for pref in required_prefix:
        if not any(h.startswith(pref) for h in header):
            raise CorruptFormat(
                f"注射时序缺少 {pref}* 列，得到 {header}",
                code="BAD_INJECTION_HEADER", path=path)
    cols = {p: next(h for h in header if h.startswith(p)) for p in required_prefix}
    entries = []
    diags = []
    seen = set()
    for n, row in enumerate(doc["rows"], start=2):
        rec = {"sourceRow": n}
        raw_no = (row.get(cols["injection_no"]) or "").strip()
        rec["injectionNoRaw"] = raw_no
        try:
            rec["injectionNo"] = int(raw_no)
        except ValueError:
            diags.append(Diagnostic(
                "BAD_INJECTION_NO", f"第 {n} 行注射编号 {raw_no!r} 不是整数",
                severity="error", path=f"row{n}:{cols['injection_no']}").to_dict())
            rec["injectionNo"] = None
        if rec["injectionNo"] in seen:
            diags.append(Diagnostic(
                "DUP_INJECTION_NO", f"注射编号 {rec['injectionNo']} 重复",
                severity="error", path=f"row{n}").to_dict())
        seen.add(rec["injectionNo"])
        for key, dim in (("t_inj", "time"), ("volume", "volume"),
                         ("win_pre", "time"), ("win_post", "time")):
            q = _qty_from_header(row, cols[key], dim, n, diags)
            rec[{"t_inj": "tInj"}.get(key,
                  {"volume": "volume", "win_pre": "winPre",
                   "win_post": "winPost"}[key])] = q
        entries.append(rec)
    entries.sort(key=lambda e: (e["injectionNo"]
                                if e["injectionNo"] is not None else 1 << 30))
    return {"kind": "injection_schedule", "path": path,
            "columns": cols, "entries": entries, "diagnostics": diags}


def parse_concentrations(text, *, path="<concentrations>"):
    """逐注浓度：injection_no, ligand_<conc>；注射器浓度单独版本化。"""
    doc = parse_csv(text, path=path)
    header = doc["header"]
    lig = next((h for h in header if h.startswith("ligand")), None)
    if "injection_no" not in header or not lig:
        raise CorruptFormat(
            f"浓度表需要 injection_no 与 ligand_* 列，得到 {header}",
            code="BAD_CONC_HEADER", path=path)
    entries, diags = [], []
    for n, row in enumerate(doc["rows"], start=2):
        raw_no = (row.get("injection_no") or "").strip()
        try:
            no = int(raw_no)
        except ValueError:
            diags.append(Diagnostic(
                "BAD_INJECTION_NO", f"第 {n} 行注射编号 {raw_no!r} 不是整数",
                severity="error", path=f"row{n}:injection_no").to_dict())
            no = None
        q = _qty_from_header(row, lig, "concentration", n, diags)
        entries.append({"sourceRow": n, "injectionNo": no,
                        "injectionNoRaw": raw_no, "ligand": q})
    entries.sort(key=lambda e: e["injectionNo"] if e["injectionNo"] is not None
                 else 1 << 30)
    return {"kind": "concentrations", "path": path,
            "ligandHeader": lig, "entries": entries, "diagnostics": diags}


_COND_DIM = {
    "temperature": "temperature",
    "cell_volume": "volume",
    "cell_macromolecule": "concentration",
    "syringe_ligand": "concentration",
    "reference_power": "power",
}


def parse_conditions(text, *, path="<conditions>"):
    """实验条件：name,value（value 自带单位，如 '25.0 °C'）。"""
    doc = parse_csv(text, path=path)
    if doc["header"][:2] != ["name", "value"]:
        raise CorruptFormat(
            f"条件表表头必须是 name,value，得到 {doc['header']}",
            code="BAD_COND_HEADER", path=path)
    values, diags = {}, []
    for n, row in enumerate(doc["rows"], start=2):
        name = (row.get("name") or "").strip()
        dim = _COND_DIM.get(name)
        if dim is None:
            diags.append(Diagnostic(
                "UNKNOWN_CONDITION", f"第 {n} 行未知条件名 {name!r}，原样保留",
                severity="warning", path=f"row{n}:name",
                related=[name]).to_dict())
            values[name] = {"raw": row.get("value"), "dimension": None,
                            "value": None, "converted": False,
                            "diagnostic": diags[-1]}
            continue
        q = parse_quantity(row.get("value"), dim, path=f"row{n}:value")
        if q.diagnostic:
            diags.append(q.diagnostic)
        values[name] = q.to_dict()
    for required in ("temperature", "cell_volume"):
        if required not in values:
            diags.append(Diagnostic(
                "MISSING_CONDITION", f"缺少必需条件 {required}",
                severity="error", path=path).to_dict())
    return {"kind": "conditions", "path": path,
            "values": values, "diagnostics": diags}
