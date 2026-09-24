"""不可变材料层：功率采样、基线、空白、注射时序、浓度、条件分开版本化。

材料是内容寻址的：同内容 => 同 id；修订产生新材料并记录 supersedes 谱系。
材料永不就地修改（“浓度修订使模型过期但不改原积分”的根基）。
"""

from .canonical import canonical_json, content_hash
from .errors import SemanticNoSolution
from .units import require
from .units import Quantity  # noqa: F401  (类型说明)


KINDS = (
    "power_trace", "baseline_trace", "blank_trace",
    "injection_schedule", "concentrations", "conditions",
)


def build_material(kind, label, body, *, supersedes=None, created_event=None):
    if kind not in KINDS:
        raise ValueError(f"未知材料类型 {kind}")
    mat = {
        "kind": kind,
        "label": label,
        "body": body,
        "supersedes": supersedes,
        "createdEvent": created_event,
    }
    mat["contentHash"] = content_hash(
        {"kind": kind, "label": label, "body": body})
    mat["id"] = "mat_" + mat["contentHash"].split(":", 1)[1][:16]
    return mat


def verify_material(mat):
    expect = content_hash(
        {"kind": mat["kind"], "label": mat["label"], "body": mat["body"]})
    if expect != mat["contentHash"]:
        raise SemanticNoSolution(
            "材料内容与 contentHash 不一致（存储损坏或被篡改）",
            code="MATERIAL_HASH_MISMATCH",
            path=mat.get("id"), details={"expected": expect,
                                         "actual": mat["contentHash"]})
    return True


def lineage(newest_id, get_material):
    """按版本从新到旧返回材料 id 链。"""
    chain = []
    mid = newest_id
    seen = set()
    while mid and mid not in seen:
        seen.add(mid)
        chain.append(mid)
        mid = get_material(mid).get("supersedes")
    return chain


# ---------------------------------------------------------------- 投影

def trace_arrays(mat):
    """返回 (times, powers, valid)（规范单位 s / µcal/s）与缺口区间列表。"""
    import numpy as np
    body = mat["body"]
    times, powers, valid = [], [], []
    for s in body["samples"]:
        t = s["time"]["value"] if s["time"]["converted"] else None
        p = s["power"]["value"] if s["power"]["converted"] else None
        times.append(t)
        powers.append(p)
        valid.append(bool(s["valid"]))
    gaps = []
    run = None
    for i, ok in enumerate(valid):
        if not ok and run is None:
            run = [i, i]
        elif not ok:
            run[1] = i
        elif ok and run is not None:
            gaps.append(tuple(run))
            run = None
    if run is not None:
        gaps.append(tuple(run))
    return (np.array([t if t is not None else np.nan for t in times]),
            np.array([p if p is not None else np.nan for p in powers]),
            np.array(valid, dtype=bool), gaps)


def gap_intervals_seconds(mat):
    """缺口在时间轴上的 [gap_start, gap_end]（秒），端点取缺口两侧最近有效样本。"""
    t, _, valid, runs = trace_arrays(mat)
    out = []
    n = len(t)
    for a, b in runs:
        left = t[a - 1] if a > 0 and valid[a - 1] else None
        right = t[b + 1] if b + 1 < n and valid[b + 1] else None
        out.append({"sampleStart": a, "sampleEnd": b,
                    "tStart": float(left) if left is not None else None,
                    "tEnd": float(right) if right is not None else None})
    return out


def injections(mat):
    """规范注射参数（秒/升）；任一缺失给出带位置诊断记录。"""
    out = []
    for e in mat["body"]["entries"]:
        rec = {"injectionNo": e["injectionNo"], "sourceRow": e["sourceRow"],
               "problems": []}
        for field, key in (("tInj", "tInj"), ("volume", "volume"),
                           ("winPre", "winPre"), ("winPost", "winPost")):
            q = e[key]
            rec[key] = q["value"] if q["converted"] else None
            if not q["converted"]:
                rec["problems"].append(
                    {"field": field, "path": q.get("diagnostic", {})
                     .get("path"), "message": q.get("diagnostic", {})
                     .get("message"), "raw": q.get("raw")})
        out.append(rec)
    return out


def concentration_map(mat):
    out = {}
    for e in mat["body"]["entries"]:
        q = e["ligand"]
        out[e["injectionNo"]] = {
            "value": q["value"] if q["converted"] else None,
            "raw": q["raw"], "sourceRow": e["sourceRow"],
            "problem": None if q["converted"] else q.get("diagnostic")}
    return out


def condition(mat, name):
    q = mat["body"]["values"][name]
    return q["value"] if q.get("converted") else None
