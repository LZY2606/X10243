"""ITC injection-integration / binding-model review application.

Layers:
  domain      - raw quantities, units, normalization (raw kept verbatim)
  store       - SQLite append-only storage: materials, events, computations, decisions
  integrate   - per-injection integration engine (gaps, overlap, blank rules)
  binding     - one-site / two-site / dilution model fitting
  archive     - counterexample archive builder (case-pack narrative)
  export_import / recovery - round-trip and half-write recovery
"""
