"""Local ITC (isothermal titration calorimetry) review application.

Domain layers:
  units       -- quantities that keep their original representation
  diagnostics -- the three distinct failure outcomes
  hashing     -- stable identities / deterministic ordering
  materials   -- versioned immutable source materials
  rules       -- versioned computation rules
  integrate   -- per-injection peak integration (overlap / gap aware)
  blank       -- dilution-heat correspondence and subtraction
  models      -- one-site, multi-site and dilution binding candidates
  replay      -- pure event projection ("structural state, replayable evidence")
  storage     -- SQLite event store with crash recovery and export/import
  bootstrap   -- the bundled offline titration case package
"""

__version__ = "1.0.0"
