"""Domain core for the ITC injection-integration / binding-model review app.

The package is deliberately split into layers that the test-suite can call
directly, without going through the browser:

units      physical quantities: original representation + normalized value
ids        stable, content-derived identities
diagnostics structured, positioned diagnostics (3 distinct failure kinds)
parser     ITC bundle import (keeps raw strings; never fakes success)
materials  versioned immutable materials (the evidence layer)
integrate  per-injection peak integration, gaps, overlap, blank subtraction
models     one-site / two-site / dilution binding-model candidates
engine     derived state, validity, provenance and replay
storage    SQLite-backed event store, derived cache, export/import/recovery
casepack   offline domain case package (synthetic titration narrative)
svg        linked server-side charts
"""

RULES_VERSION = "rules-v1.0.0"
