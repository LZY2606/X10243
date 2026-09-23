"""Application facade: imports, decisions, branching, derived state, provenance."""

from __future__ import annotations

import copy
from typing import Optional

from .budget import Budget
from .engine import compute_series
from .project import PARSERS, material_identity, replay
from .store import Store
from .util import digest


class ProjectService:
    def __init__(self, store: Store):
        self.store = store

    # ------------------------------------------------------------- reading
    def _state(self, project_id, branch, *, at_seq: Optional[int] = None):
        events = self.store.list_events(project_id, branch)
        # fork lineage events from other branches are needed to materialise clone
        if at_seq is not None:
            events = [e for e in events if e["seq"] <= at_seq]
        all_events = self.store.list_events(project_id)
        if at_seq is not None:
            all_events = [e for e in all_events if not (e["branch"] == branch and e["seq"] > at_seq)]
        states, heads = replay(all_events, lambda bid: self._blob_text(bid))
        return states.get(branch), heads

    def _blob_text(self, bid):
        row = self.store.get_blob(bid)
        return None if row is None else row["content"]

    def state(self, project_id, branch="main", *, at_seq=None):
        state, _ = self._state(project_id, branch, at_seq=at_seq)
        return state

    def review(self, project_id, branch="main", *, at_seq=None, budget=None):
        state = self.state(project_id, branch, at_seq=at_seq)
        if state is None:
            return None
        series_views = {}
        for sid in state["series_order"]:
            role = state["series"][sid]["role"]
            view = compute_series(state, sid, budget=budget)
            series_views[sid] = view
        return {"project_id": project_id, "branch": branch, "at_seq": at_seq,
                "title": state["title"], "series": series_views,
                "series_order": state["series_order"],
                "blank_links": state["blank_links"],
                "decisions": state["decisions"],
                "diagnostics": state["diagnostics"],
                "forks": state["forks"]}

    # ------------------------------------------------------------- writing
    def create_project(self, project_id, title, branch="main"):
        return self.store.append(project_id, branch, "project_created",
                                 {"project_id": project_id, "title": title})

    def add_series(self, project_id, series_id, role, label="", branch="main"):
        return self.store.append(project_id, branch, "series_added",
                                 {"series_id": series_id, "role": role, "label": label})

    def add_material(self, project_id, series_id, kind, content, *, path=None,
                     branch="main", budget=None):
        parser = PARSERS[kind]
        parsed = parser(content, path=path or f"{series_id}/{kind}.csv",
                        budget=budget)
        bid = self.store.put_blob(content, kind=kind)
        mid = material_identity(parsed)
        ev = self.store.append(project_id, branch, "material_added",
                               {"series_id": series_id, "kind": kind,
                                "blob_id": bid, "material_id": mid,
                                "path": path,
                                "diagnostics": parsed.get("diagnostics", [])},
                               blobs={kind: bid})
        return {"event": ev, "material_id": mid, "parsed": parsed,
                "diagnostics": parsed.get("diagnostics", [])}

    def link_blank(self, project_id, sample_series_id, blank_series_id,
                   rule=None, branch="main"):
        return self.store.append(project_id, branch, "blank_linked",
                                 {"sample_series_id": sample_series_id,
                                  "blank_series_id": blank_series_id,
                                  "rule": rule or {"mode": "ordinal"}})

    def add_decision(self, project_id, series_id, kind, params, branch="main",
                     *, note=""):
        decisions = self.state(project_id, branch)["decisions"].get(series_id, [])
        decision = {"kind": kind, "params": params, "note": note,
                    "decision_id": digest(["decision", kind, params, len(decisions)],
                                          length=12, prefix="dn_")}
        return self.store.append(project_id, branch, "decision_added",
                                 {"series_id": series_id, "decision": decision})

    def fork(self, project_id, new_branch, from_branch="main", *, at_seq=None,
             reason=""):
        state = self.state(project_id, from_branch)
        if state is None:
            raise KeyError(f"branch {from_branch} not found")
        events = self.store.list_events(project_id, from_branch)
        head_seq = at_seq if at_seq is not None else max(e["seq"] for e in events)
        return self.store.append(project_id, new_branch, "branch_forked",
                                 {"new_branch": new_branch,
                                  "from_branch": from_branch,
                                  "at_seq": head_seq, "reason": reason})

    def branches(self, project_id):
        return self.store.branches(project_id)

    def event_log(self, project_id, branch=None):
        return self.store.list_events(project_id, branch)

    # -------------------------------------------------------- half-write API
    def stage_decision(self, project_id, series_id, kind, params, branch="main"):
        decisions = self.state(project_id, branch)["decisions"].get(series_id, [])
        decision = {"kind": kind, "params": params,
                    "decision_id": digest(["staged", kind, params, len(decisions)],
                                          length=12, prefix="dn_")}
        return self.store.append(project_id, branch, "decision_added",
                                 {"series_id": series_id, "decision": decision},
                                 stage_only=True)

    def promote(self, token):
        return self.store.promote_staged(token)
