"""ITC injection-integration and binding-model review domain package.

Layering:
  units.py    - raw quantities carry units; normalization happens only in derived layer
  domain.py   - integration, blank/dilution subtraction, binding model candidates
  store.py    - SQLite event-sourced storage, export/import, recovery
  service.py  - orchestration callable directly from tests or the web layer
  casepack.py - offline domain case pack + counterexample archive replay
  svgplot.py  - server-side linked SVG views
"""
