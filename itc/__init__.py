"""Local ITC injection-integration and binding-model review application.

The package is split into immutable materials, an event log, a deterministic
calculation engine and storage/export layers. Nothing business-critical lives
only in the browser: every rule in :mod:`itc.engine` can be invoked directly
from Python and is covered by the test-suite.
"""

__all__ = ["__version__"]
__version__ = "1.0.0"
