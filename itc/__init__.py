"""Local ITC injection integration and binding-model review application.

The package is split into layers that can each be exercised directly by tests:

``itc.units``       domain quantities with units and coordinate bases
``itc.ingest``      raw material parsing (original representation preserved)
``itc.integrate``   per-injection integration with gaps / overlap / blanks
``itc.binding``     one-site, two-site and dilution binding hypotheses
``itc.events``      append-only event log, materials, branches and replay
``itc.project``     workflow facade (import / integrate / fit / decide)
``itc.transport``   export bundle and import with half-write recovery
``itc.casepack``    the offline domain case (titration thermogram narrative)
``itc.svg``         linked charts (power trace, intervals, points, residuals)
``itc.app``         FastAPI wiring
"""

__all__ = ["__version__"]
__version__ = "1.0.0"
