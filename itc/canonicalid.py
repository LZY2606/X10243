"""确定性、内容寻址的稳定身份。

同一内容无论经过多少次导出/导入、在什么机器上，得到的 id 都相同；
等价输入按 (关键排序键, id) 得到稳定顺序。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def cid(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def key_hash(payload: Any) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()
