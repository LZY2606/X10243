"""Local ITC injection-integration and binding-model review toolkit.

The package is deliberately split into layers that can be tested without a
browser or running server:

* :mod:`itc.units`      - quantities carry units/coordinate bases
* :mod:`itc.materials`  - immutable raw material ingestion/parsing
* :mod:`itc.integration` - per-injection heat integration (pure functions)
* :mod:`itc.binding`    - one-site / multi-site / dilution model fits
* :mod:`itc.storage`    - SQLite persistence + half-write recovery
* :mod:`itc.casepack`   - offline case pack export/import
* :mod:`itc.seed`       - bundled titration thermogram narrative
* :mod:`itc.service`    - business rules / event projection
* :mod:`itc.svg`        - server rendered review graphics
"""

__all__ = [
    "canonical",
    "errors",
    "units",
    "materials",
    "integration",
    "binding",
    "storage",
    "casepack",
    "service",
    "svg",
    "seed",
]
