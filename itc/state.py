"""从事件链重放构造视图。

回到任意旧事件时，恢复的是当时依赖版本与可见结论，而不是用最新计算覆盖历史：
- 材料版本由该事件祖先路径上“最近的一次导入/修订”决定；
- 产物由祖先路径上创建事件给出；invalidated 事件把后代模型标记为 stale；
- 冲突来自互不相容的人工决定（同一拟合产物在分叉链上被 accept/reject）。
"""
from __future__ import annotations

from .models import (
    EV_POWER_IMPORTED, EV_INJECTIONS_IMPORTED, EV_CONCENTRATION_IMPORTED,
    EV_CONDITIONS_IMPORTED, EV_BASELINE_IMPORTED, EV_BLANK_IMPORTED,
    EV_POWER_PATCHED, EV_CONCENTRATION_REVISED,
    EV_INTEGRATION_RUN, EV_FIT_RUN, EV_DECISION, EV_INVALIDATED,
    EV_FIRST_DISCARD_SET, EV_BLANK_MATCH_SET,
)

IMPORT_KIND = {
    EV_POWER_IMPORTED: "power",
    EV_INJECTIONS_IMPORTED: "injections",
    EV_CONCENTRATION_IMPORTED: "concentration",
    EV_CONDITIONS_IMPORTED: "conditions",
    EV_BASELINE_IMPORTED: "baseline",
    EV_BLANK_IMPORTED: "blank",
}
LATEST_KINDS = {"power", "injections", "concentration", "conditions"}
# baseline/blank 不做“静默最新”：积分/稀释事件显式记录所用材料 id
REVISED_KIND = {EV_POWER_PATCHED: "power",
                EV_CONCENTRATION_REVISED: "concentration"}


def reconstruct(store, head_event_id: str) -> dict:
    """沿 parent（可跨链 fork）重放到指定事件，返回当时完整视图。"""
    chain = store.ancestors(head_event_id)
    materials: dict[str, dict] = {}
    integration = None
    dilution = None
    fit = None
    first_discard = None
    blank_match = None
    stale_artifacts: set[str] = set()
    decisions = []
    timelines = []

    for ev in chain:
        p = ev["payload"]
        if ev["type"] in IMPORT_KIND:
            kind = IMPORT_KIND[ev["type"]]
            mid = p.get("material_id")
            mat = store.get_material(mid)
            if kind in LATEST_KINDS:
                materials[kind] = mat
            else:
                materials.setdefault(kind + "_versions", []).append(mat)
                materials[kind] = mat
        elif ev["type"] in REVISED_KIND:
            kind = REVISED_KIND[ev["type"]]
            mat = store.get_material(p["material_id"])
            materials[kind] = mat
        elif ev["type"] == EV_FIRST_DISCARD_SET:
            first_discard = p
        elif ev["type"] == EV_INTEGRATION_RUN:
            aid = p.get("artifact_id")
            integration = store.get_artifact(aid)
        elif ev["type"] == EV_BLANK_MATCH_SET:
            aid = p.get("artifact_id")
            dilution = store.get_artifact(aid)
            blank_match = p
        elif ev["type"] == EV_FIT_RUN:
            aid = p.get("artifact_id")
            fit = store.get_artifact(aid)
        elif ev["type"] == EV_INVALIDATED:
            for aid in p.get("artifact_ids", []):
                stale_artifacts.add(aid)
        elif ev["type"] == EV_DECISION:
            decisions.append(p)
        timelines.append({
            "event_id": ev["id"], "seq": ev["seq"], "chain_id": ev["chain_id"],
            "type": ev["type"], "parent_id": ev["parent_id"],
            "payload": ev["payload"], "inputs": ev["inputs"],
            "params": ev["params"], "parents": ev["parents"],
        })

    visible = {
        "head_event_id": head_event_id,
        "chain_id": chain[-1]["chain_id"] if chain else None,
        "materials": {k: v for k, v in materials.items()
                      if not k.endswith("_versions")},
        "baseline_versions": materials.get("baseline_versions", []),
        "blank_versions": materials.get("blank_versions", []),
        "first_discard": first_discard,
        "integration": integration,
        "dilution": dilution,
        "blank_match": blank_match,
        "fit": fit,
        "stale_artifacts": sorted(stale_artifacts),
        "decisions": decisions,
        "timeline": timelines,
    }
    _annotate_staleness(visible)
    return visible


def _annotate_staleness(v: dict) -> None:
    stale = v["stale_artifacts"]
    for key in ("integration", "dilution", "fit"):
        art = v.get(key)
        if art and art["id"] in stale:
            art["stale"] = True
    # 稀释依赖积分；积分过期时稀释结果也标记过期
    if v.get("integration") and v["integration"].get("stale") and v.get("dilution"):
        if v["dilution"].get("inputs", {}).get("integration_artifact") == \
                v["integration"]["id"]:
            v["dilution"]["stale"] = True
    # 拟合依赖稀释；稀释过期或浓度修订→拟合过期
    fit = v.get("fit")
    if fit:
        conc_id = (v.get("materials", {}).get("concentration") or {}).get("id")
        fit_conc = (fit.get("inputs") or {}).get("concentration_material")
        if fit.get("stale"):
            pass
        elif v.get("dilution") and v["dilution"].get("stale"):
            fit["stale"] = True
        elif conc_id and fit_conc and conc_id != fit_conc:
            fit["stale"] = True


def compute_conflicts(store) -> list[dict]:
    """互不相容的人工决定：同一拟合产物在分叉链上被给出不同 verdict。"""
    events = store.all_events()
    decisions = [e for e in events if e["type"] == EV_DECISION]
    by_target: dict[str, list[dict]] = {}
    for ev in decisions:
        target = ev["payload"].get("target_artifact")
        if target:
            by_target.setdefault(target, []).append(ev)
    conflicts = []
    for target, evs in by_target.items():
        verdicts = {e["payload"].get("verdict") for e in evs}
        if len(verdicts) > 1:
            # 仅当这些决定互不构成祖先关系（真正分叉）才算冲突
            idset = {e["id"] for e in evs}
            divergent = []
            for e in evs:
                anc = {a["id"] for a in store.ancestors(e["id"])}
                others = idset - {e["id"]}
                if not (others & anc):
                    divergent.append(e)
            div_verdicts = {e["payload"].get("verdict") for e in divergent}
            if len(div_verdicts) > 1:
                conflicts.append({
                    "target_artifact": target,
                    "events": [{"event_id": e["id"], "chain_id": e["chain_id"],
                                "verdict": e["payload"].get("verdict"),
                                "note": e["payload"].get("note", "")}
                               for e in divergent],
                })
    conflicts.sort(key=lambda c: c["target_artifact"])
    return conflicts


def chains_overview(store) -> list[dict]:
    heads = store.head_events()
    out = []
    for chain_id, head in sorted(heads.items()):
        chain = store.ancestors(head["id"])
        decisions = [e for e in chain if e["type"] == EV_DECISION]
        fork_parent = None
        fork_event = None
        if chain and chain[0]["parent_id"]:
            fork_parent = chain[0]["parent_id"]
            fork_event = chain[0]["id"]
        out.append({
            "chain_id": chain_id,
            "head_event_id": head["id"],
            "n_events": len(chain),
            "fork_event_id": fork_event,
            "fork_parent_event_id": fork_parent,
            "decision_count": len(decisions),
            "last_verdict": decisions[-1]["payload"].get("verdict")
            if decisions else None,
        })
    return out
