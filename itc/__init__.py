"""Local ITC (isothermal titration calorimetry) review application.

Domain layers:

* :mod:`itc.units`      -- original representation + unit conversion diagnostics
* :mod:`itc.canonical`  -- canonical JSON + stable identities
* :mod:`itc.power`      -- power series normalisation and gap detection
* :mod:`itc.injection`  -- injection schedule normalisation
* :od:`itc.integrate`   -- per-injection integration windows
* :mod:`itc.blank`      -- blank/dilution matching and subtraction
* :mod:`itc.models`     -- one-site, multi-site and dilution candidate fits
* :mod:`itc.derive`     -- deterministic derivation of the branch state
* :mod:`itc.events`     -- append-only event log / branches / material versions
* :mod:`itc.store`      -- SQLite persistence and half-write recovery
* :mod:`itc.archive`    -- project export / import round trip
"""

SCHEMA_VERSION = 1
APP_VERSION = "1.0.0"
