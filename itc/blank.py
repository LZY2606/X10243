"""稀释热扣除：空白曲线与逐注空白必须声明对应规则。

对应规则（结构化，不是自由文本日志）：
  matchRule: {strategy: "by_injection_no", blankScheduleId, note}
逐注空白缺失 => 不猜测、不借用相邻注射；该点保留未扣除形态，fitEligibility=false。
"""

from .canonical import short_id
from .integration import integrate, ALG_VERSION as INTEGRATION_ALG


def blank_energies(blank_power_mat, blank_injection_mat,
                   blank_baseline_mat=None, *, endpoint_overrides=None,
                   discard_first=True):
    res = integrate(blank_power_mat, blank_injection_mat, blank_baseline_mat,
                    endpoint_overrides=endpoint_overrides,
                    discard_first=discard_first)
    table = {}
    for w in res["windows"]:
        table[w["injectionNo"]] = w
    return {"integration": res, "byNo": table,
            "inputs": res["inputs"]}


def subtract_dilution(sample_integration, blank_pack,
                      *, match_strategy="by_injection_no",
                      discard_first=True):
    """生成逐注 corrected point；对应关系逐条可追溯。"""
    points = []
    diagnostics = []
    correspondence = {
        "strategy": match_strategy,
        "blankInputs": blank_pack["inputs"],
        "version": "dilution:1",
    }
    for w in sample_integration["windows"]:
        no = w["injectionNo"]
        point = {
            "injectionNo": no,
            "sampleWindowUid": w["uid"],
            "rawEnergy": w.get("energy"),
            "rawEnergyUnit": "µcal",
            "dilutionEnergy": None,
            "correctedEnergy": None,
            "status": w["status"],
            "discarded": bool(w.get("discarded")),
            "match": None,
            "fitEligibility": False,
        }
        if w["status"] != "integrated":
            point["reason"] = "SAMPLE_INTEGRAL_REJECTED"
            points.append(point)
            continue
        if point["discarded"]:
            point["reason"] = "FIRST_INJECTION_DISCARDED"
            points.append(point)
            continue
        bw = blank_pack["byNo"].get(no)
        if bw is None:
            point["reason"] = "BLANK_CORRESPONDENCE_MISSING"
            point["correctedEnergy"] = w.get("energy")  # 保留未扣除形态
            point["dilutionEnergy"] = None
            point["fitEligibility"] = False
            diagnostics.append({
                "code": "BLANK_CORRESPONDENCE_MISSING", "severity": "warning",
                "message": (f"注射 {no} 没有按规则 {match_strategy} 找到空白对应；"
                            "不借用相邻注射，该点保留未扣除形态且不进入拟合"),
                "related": [f"injection:{no}", w["uid"]],
                "matchRule": correspondence})
            points.append(point)
            continue
        if bw["status"] != "integrated":
            point["reason"] = "BLANK_INTEGRAL_REJECTED"
            point["correctedEnergy"] = w.get("energy")
            diagnostics.append({
                "code": "BLANK_INTEGRAL_REJECTED", "severity": "warning",
                "message": f"注射 {no} 的空白积分被拒绝（{bw['diagnostics']}）",
                "related": [f"injection:{no}"]})
            points.append(point)
            continue
        # 体积比例缩放：空白注射体积可能与样品不同（同浓度稀释物）
        scale = w.get("volume") and bw.get("volume") and \
            bw["volume"] and w["volume"] / bw["volume"] or 1.0
        dilution = bw["energy"] * scale
        point["dilutionEnergy"] = dilution
        point["correctedEnergy"] = w["energy"] - dilution
        point["match"] = {"strategy": match_strategy,
                          "blankWindowUid": bw["uid"],
                          "blankInjectionNo": no,
                          "volumeScale": scale}
        point["fitEligibility"] = True
        point["reason"] = "CORRECTED"
        points.append(point)

    basis = {"sample": sample_integration["inputs"],
             "blank": blank_pack["inputs"], "rule": correspondence}
    for p in points:
        p["uid"] = short_id(
            {"no": p["injectionNo"], "raw": p["rawEnergy"],
             "dil": p["dilutionEnergy"], "match": p["match"],
             "v": "dilution:1"}, "pt")
    return {"version": "dilution:1", "matchRule": correspondence,
            "points": points, "diagnostics": diagnostics,
            "identityBasis": basis}
