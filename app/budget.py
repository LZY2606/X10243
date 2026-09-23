"""Explicit resource budgets.  Exhaustion is its own outcome (BUDGET_EXHAUSTED)."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import BudgetExhausted


@dataclass
class Budget:
    name: str
    limit: int
    used: int = 0
    notes: str = ""

    def tick(self, n: int = 1):
        self.used += n
        if self.used > self.limit:
            raise BudgetExhausted(
                f"{self.name} budget exhausted: used {self.used} > limit {self.limit}",
                budget=self.name,
                used=self.used,
                partial=self.notes,
            )

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit
