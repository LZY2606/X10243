"""SQLite storage layer.

Layering:
  material_versions  - immutable source materials (power curves, injection
                       schedules, concentration sets, temperature/pool/blank,
                       baselines), each separately versioned by content hash.
  material_points    - raw samples belonging to a material version; the raw
                       textual representation plus parsed value/unit is kept.
  ingest_attempts    - every ingestion fragment incl. failures (corrupt
                       format / semantic impasse / budget exhausted).
  rule_versions      - named rule/parameter versions used by derivations.
  events             - append-only decision/event tree (branches).
  results            - immutable derived results, bound to (event, parents,
                       rule_version, input_summary).
  result_items       - per-injection / per-parameter rows with stable ids.
  result_diagnostics - located diagnostics attached to failed/incomplete work.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator, Optional

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS material_versions (
  material_id   TEXT NOT NULL,
  kind          TEXT NOT NULL,           -- power_curve|injection_schedule|concentration_set|meta|blank_curve|baseline
  version       INTEGER NOT NULL,
  content_hash  TEXT NOT NULL,
  source_label TEXT NOT NULL,
  origin_uri   TEXT,
  payload_json  TEXT NOT NULL,           -- structured content incl. raw representation
  supersedes    TEXT,                    -- material_id/version of previous revision (same logical material)
  created_event TEXT,
  PRIMARY KEY (material_id, version)
);

CREATE TABLE IF NOT EXISTS material_points (
  material_id TEXT NOT NULL,
  version     INTEGER NOT NULL,
  seq         INTEGER NOT NULL,
  coord_raw   TEXT NOT NULL,             -- e.g. time "12.5 s"
  value_raw   TEXT NOT NULL,             -- e.g. power "-3.1 ucal/s"
  coord_value REAL,                      -- canonical, NULL when unconverted
  value_value REAL,                      -- canonical, NULL when unconverted
  coord_unit  TEXT,
  value_unit  TEXT,
  coord_diag  TEXT,
  value_diag  TEXT,
  gap_after   INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (material_id, version, seq)
);

CREATE TABLE IF NOT EXISTS ingest_attempts (
  attempt_id    TEXT PRIMARY KEY,
  kind          TEXT NOT NULL,
  status        TEXT NOT NULL,           -- accepted|corrupt_format|semantic_impasse|budget_exhausted
  source_label  TEXT NOT NULL,
  fragment      TEXT NOT NULL,           -- the input fragment exactly as presented
  diagnostics   TEXT NOT NULL,           -- JSON list of {path,message,...}
  material_id   TEXT,
  version       INTEGER,
  event_id      TEXT
);

CREATE TABLE IF NOT EXISTS rule_versions (
  rule_id     TEXT PRIMARY KEY,          -- e.g. rule_integration_v1
  family      TEXT NOT NULL,             -- integration|blank_match|binding|stale
  version     INTEGER NOT NULL,
  params_json TEXT NOT NULL,
  UNIQUE(family, version)
);

CREATE TABLE IF NOT EXISTS events (
  event_id    TEXT PRIMARY KEY,
  seq         INTEGER NOT NULL,          -- global replay order (stable, assigned per branch path)
  branch      TEXT NOT NULL,
  parent_id   TEXT,
  kind        TEXT NOT NULL,             -- import|set_baseline|integrate|blank_match|concentration_revision|fit|decision|note
  action      TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  accepted    INTEGER NOT NULL DEFAULT 1,
  created_at_order INTEGER NOT NULL,     -- insertion counter only, not part of identity
  UNIQUE(branch, seq)
);

CREATE TABLE IF NOT EXISTS results (
  result_id   TEXT PRIMARY KEY,
  event_id    TEXT NOT NULL,
  family      TEXT NOT NULL,             -- integration|dilution|binding
  status      TEXT NOT NULL,             -- ok|invalid|stale|ambiguous
  rule_id     TEXT NOT NULL,
  input_summary_json TEXT NOT NULL,
  parents_json TEXT NOT NULL,            -- [result_id ...]
  payload_json TEXT NOT NULL,
  rank_index  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_results_event ON results(event_id);

CREATE TABLE IF NOT EXISTS result_items (
  item_id     TEXT PRIMARY KEY,
  result_id   TEXT NOT NULL,
  item_key    TEXT NOT NULL,             -- e.g. inj:3 / param:delta_h
  seq         INTEGER NOT NULL,
  payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS result_diagnostics (
  diag_id     TEXT PRIMARY KEY,
  result_id   TEXT,
  event_id    TEXT,
  path        TEXT NOT NULL,
  message     TEXT NOT NULL,
  payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS branches (
  name        TEXT PRIMARY KEY,
  head_event_id TEXT,
  fork_event_id TEXT,
  label       TEXT NOT NULL
);
"""


def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = os.environ.get("ITC_DB_PATH", "itc_lab.db")
    fresh = not os.path.exists(db_path) or os.path.getsize(db_path) == 0
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(SCHEMA)
    if fresh:
        conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('schema','1')")
        conn.commit()
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def jdumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def jloads(text: str) -> Any:
    return json.loads(text)
