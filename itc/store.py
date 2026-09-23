"""SQLite-backed project store.

Layers:
- materials: immutable, separately versioned per kind (power_trace,
  injections, concentrations, temperature, cell_volume, baseline,
  blank_trace, blank_injections); raw text always preserved;
- computations: derived results with input digest, parameter version and
  parent dependencies; failures are recorded with their distinct status;
- events: append-only machine-readable judgments;
- decisions: the mutable human layer, branch-aware; conflicting decisions on
  different branches are all retained.

All ids are sequence-derived and deterministic, so equivalent inputs replay
to identical archives and export/import preserves identity, order and provenance.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager

from . import materials as mat
from .results import Result, OK, DIAGNOSTICS, FORMAT_CORRUPT
from .units import Diagnostic

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS branches(
  branch_id TEXT PRIMARY KEY,
  parent_branch TEXT,
  fork_seq INTEGER,
  label TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS materials(
  material_id TEXT NOT NULL,
  version_no INTEGER NOT NULL,
  kind TEXT NOT NULL,
  raw_text TEXT NOT NULL,
  raw_format TEXT NOT NULL,
  normalized_json TEXT,
  diagnostics_json TEXT NOT NULL DEFAULT '[]',
  supersedes TEXT,
  seq INTEGER NOT NULL,
  PRIMARY KEY(material_id, version_no));
CREATE TABLE IF NOT EXISTS computations(
  computation_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  param_version TEXT NOT NULL,
  input_digest TEXT NOT NULL,
  parents_json TEXT NOT NULL,
  status TEXT NOT NULL,
  result_json TEXT,
  diagnostics_json TEXT NOT NULL DEFAULT '[]',
  seq INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS events(
  event_id TEXT PRIMARY KEY,
  seq INTEGER NOT NULL,
  branch_id TEXT NOT NULL DEFAULT 'main',
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  parents_json TEXT NOT NULL DEFAULT '[]');
CREATE TABLE IF NOT EXISTS decisions(
  decision_id TEXT PRIMARY KEY,
  seq INTEGER NOT NULL,
  branch_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  target_id TEXT NOT NULL,
  verdict TEXT NOT NULL,
  rationale TEXT NOT NULL DEFAULT '',
  payload_json TEXT NOT NULL DEFAULT '{}',
  supersedes TEXT);
"""
