"""Local ITC review application: injection integration + binding-model review.

Layers:
- results:  three distinct failure kinds (resource / format / semantic) plus Ok
- units:    raw quantities preserved on import; normalization only in the derived layer
- integrate: per-injection heat integration (overlap allowed, gaps never crossed)
- binding:  one-site / two-site / dilution model candidates (NumPy-only LM)
- store:    SQLite event store with batch half-write recovery
- service:  application service; every computation records inputs, params, parents
"""
