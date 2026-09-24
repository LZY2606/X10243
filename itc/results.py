"""计算结果的三态容器与带位置诊断。

资源预算耗尽（RESOURCE_EXHAUSTED）、格式损坏（FORMAT_ERROR）、语义无解
（UNRESOLVABLE）必须是三种不同结果；输入不完整返回带位置的诊断，
不得用空值或默认值伪装成功。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

OK = "ok"
RESOURCE_EXHAUSTED = "resource_exhausted"
FORMAT_ERROR = "format_error"
UNRESOLVABLE = "unresolvable"

T = TypeVar("T")


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    location: str = ""          # 如 "power.csv:row 142 / field t"
    value: str = ""             # 原始表示（保留）
    severity: str = "error"     # error | warning

    def render(self) -> str:
        loc = f" @ {self.location}" if self.location else ""
        val = f" (value={self.value!r})" if self.value else ""
        return f"[{self.severity}:{self.code}]{loc} {self.message}{val}"


@dataclass
class Outcome(Generic[T]):
    kind: str = OK
    value: T | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.kind == OK

    @classmethod
    def success(cls, value: T, diagnostics: list[Diagnostic] | None = None) -> "Outcome[T]":
        return cls(OK, value, diagnostics or [])

    @classmethod
    def fail(cls, kind: str, diags: list[Diagnostic] | Diagnostic) -> "Outcome[T]":
        if isinstance(diags, Diagnostic):
            diags = [diags]
        return cls(kind, None, diags)

    @classmethod
    def resource(cls, code: str, message: str, location: str = "") -> "Outcome[T]":
        return cls.fail(RESOURCE_EXHAUSTED, Diagnostic(code, message, location))

    @classmethod
    def format_error(cls, code: str, message: str, location: str = "", value: str = "") -> "Outcome[T]":
        return cls.fail(FORMAT_ERROR, Diagnostic(code, message, location, value))

    @classmethod
    def unresolvable(cls, code: str, message: str, location: str = "") -> "Outcome[T]":
        return cls.fail(UNRESOLVABLE, Diagnostic(code, message, location))

    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "warning"]
