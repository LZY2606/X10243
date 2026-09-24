"""Stable exception categories required by the ITC workflow."""
from __future__ import annotations


class BudgetExhausted(Exception):
    code = "RESOURCE_BUDGET_EXHAUSTED"


class CorruptFormat(Exception):
    code = "FORMAT_CORRUPT"


class SemanticNoSolution(Exception):
    code = "SEMANTIC_NO_SOLUTION"
