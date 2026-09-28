"""Titration calorimetry (ITC) domain package.

Immutable materials, replayable decisions, injection integration and
binding-model review. All business rules live in this package so they can be
called directly from tests, the storage layer and the FastAPI layer.
"""

from .errors import ItcError

__all__ = ["ItcError"]
