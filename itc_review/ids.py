"""内容寻址与稳定身份。

材料版本、规则版本、派生结果都由其输入内容决定；
等价输入必然得到等价身份与稳定顺序。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def digest(prefix: str, payload: Any, length: int = 16) -> str:
    h = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{h}"


def sha256(payload: Any) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def stable_deps(deps: dict[str, Any]) -> list:
    """依赖摘要的稳定排序表示。"""
    out = []
    for key in sorted(deps):
        out.append([key, deps[key]])
    return out
