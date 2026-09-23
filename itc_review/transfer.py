"""Self-contained project export / import round-trip and half-write recovery."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Any

from . import __version__
from .storage import Repository, connect, transaction

BUNDLE_NAME = "itc-project.json"


def export_project(repo: Repository, *, case_name: str = "case") -> dict[str, Any]:
    """Freeze the entire event-sourced project into one immutable bundle."""
    branches = repo.branches()
    events = repo.all_events()
    bundle = {
        "bundle": "itc-review-project",
        "bundle_version": 1,
        "app_version": __version__,
        "case_name": case_name,
        "materials": repo.materials(),
        "results": repo.results(),
        "result_refs": [
            {"scope": s, **{"ref_key": k, "result_id": v}}
            for s in {r["branch"] for r in branches} | {"project"}
            for k, v in repo.refs(s).items()
        ],
        "branches": branches,
        "events": events,
    }
    return bundle


def write_bundle(bundle: dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / BUNDLE_NAME
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def stage_import(bundle: dict[str, Any], stage_root: str | Path) -> tuple[str, Path]:
    """Materialize a bundle into a staging directory before the commit."""
    stage_id = bundle.get("case_name", "case") + "-" + uuid.uuid4().hex[:12]
    stage_dir = Path(stage_root) / stage_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload_path = stage_dir / BUNDLE_NAME
    tmp = payload_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, payload_path)
    return stage_id, stage_dir


def recover_orphan_stages(repo: Repository, stage_root: str | Path) -> list[dict[str, Any]]:
    """Find staged-but-never-committed imports after a crash (half-write)."""
    found: list[dict[str, Any]] = []
    root = Path(stage_root)
    if not root.exists():
        return found
    known = {s["id"]: s for s in repo.stages()}
    for child in root.iterdir():
        marker = child / BUNDLE_NAME
        tmp = child / (BUNDLE_NAME + ".tmp")
        if marker.exists() and child.name.split("-")[-1] not in known:
            with open(marker, encoding="utf-8") as fh:
                manifest = json.load(fh)
            repo.register_stage(child.name, str(child),
                                {"case_name": manifest.get("case_name")})
            repo.commit_stage(child.name)  # fully materialized -> safe commit
            found.append({"id": child.name, "state": "recovered",
                          "path": str(child)})
        elif tmp.exists() and not marker.exists():
            found.append({"id": child.name, "state": "aborted-half-write",
                          "path": str(child)})
            tmp.unlink()
    return found


def import_bundle(repo: Repository, bundle: dict[str, Any], *,
                  stage_root: str | Path) -> dict[str, Any]:
    """Replay a bundle into storage, preserving every identity and ordering."""
    if bundle.get("bundle") != "itc-review-project":
        raise ValueError("not an itc-review-project bundle")
    stage_id, stage_dir = stage_import(bundle, stage_root)
    repo.register_stage(stage_id, str(stage_dir),
                        {"case_name": bundle.get("case_name"),
                         "events": len(bundle["events"])})
    report = {"materials_inserted": 0, "materials_seen": 0,
              "results_inserted": 0, "results_seen": 0,
              "events_inserted": 0, "events_seen": 0, "stage_id": stage_id}

    with transaction(repo.conn):
        for mat in bundle["materials"]:
            inserted = repo.add_material(
                id=mat["id"], kind=mat["kind"], role=mat["role"],
                version=mat["version"], lineage=mat["lineage"],
                supersedes=mat["supersedes"], payload=mat["payload"],
                digest=mat["digest"])
            report["materials_inserted" if inserted else "materials_seen"] += 1
        for res in bundle["results"]:
            inserted = repo.add_result(
                id=res["id"], kind=res["kind"], status=res["status"],
                inputs=res["inputs"], params_version=res["params_version"],
                parents=res["parents"], payload=res["payload"],
                digest=res["digest"], stale=res.get("stale", False))
            report["results_inserted" if inserted else "results_seen"] += 1
        for ref in bundle.get("result_refs", []):
            repo.set_ref(ref["scope"], ref["ref_key"], ref["result_id"])
        for br in bundle["branches"]:
            repo.ensure_branch(br["name"], br.get("parent_branch"),
                               br.get("forked_at"))
        ordered = sorted(bundle["events"], key=lambda e: (e["branch"], e["seq"]))
        for ev in ordered:
            exists = repo.conn.execute(
                "SELECT 1 FROM events WHERE id=?", (ev["id"],)).fetchone()
            if exists:
                report["events_seen"] += 1
                continue
            repo.append_event(
                id=ev["id"], branch=ev["branch"], seq=ev["seq"],
                action=ev["action"], payload=ev["payload"],
                parent_event=ev["parent_event"])
            report["events_inserted"] += 1
        for br in bundle["branches"]:
            if br.get("head_event"):
                repo.set_head(br["name"], br["head_event"])
        repo.commit_stage(stage_id)
    return report
