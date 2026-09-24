"""Domain layer for the local ITC injection-integration / binding-model review app.

The package is deliberately split so that the business rules live here and can
be driven directly from tests or from the FastAPI service layer in ``itc``:

* :mod:`itcdomain.units`      - raw representation preservation + SI normalisation
* :mod:`itcdomain.canonical`  - stable hashing / identity helpers
* :mod:`itcdomain.materials`  - versioned immutable material value objects
* :mod:`itcdomain.parser`     - raw ``.itc`` bundle ingestion
* :mod:`itcdomain.integration`- injection window integration over gapped power
* :mod:`itcdomain.dilution`   - blank / dilution-heat correspondence rules
* :mod:`itcdomain.models`     - one-site, two-site and dilution binding models
* :mod:`itcdomain.state`      - event-replay derived state / branching
* :mod:`itcdomain.store`      - SQLite event store with half-write recovery
* :mod:`itcdomain.archive`    - project export / import round trip
* :mod:`itcdomain.casepack`   - built-in offline titration-thermogram case pack
"""

__all__ = [
    "units",
    "canonical",
    "materials",
    "parser",
    "integration",
    "dilution",
    "models",
    "state",
    "store",
    "archive",
    "casepack",
]
