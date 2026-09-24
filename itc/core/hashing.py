"""规范化序列化与内容寻址哈希。

等价输入（键顺序不同、浮点同值）得到相同摘要；内容寻址节点身份由此确定，
因此导出/导入后等价计算仍指向同一稳定节点。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    return json.dumps(_stable(obj), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _stable(obj: Any):
    if isinstance(obj, float):
        if obj != obj:
            return {"__float__": "nan"}
        if obj == float("inf"):
            return {"__float__": "inf"}
        if obj == float("-inf"):
            return {"__float__": "neg_inf"}
        return {"__float__": repr(obj)}
    if isinstance(obj, dict):
        return {str(k): _stable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stable(v) for v in obj]
    return obj


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()[:16]


def stable_key(node: dict[str, Any]) -> tuple:
    """等价输入得到稳定顺序：先类型、再注射键、再身份哈希。"""
    return (
        str(node.get("node_type", "")),
        node.get("injection_key") if node.get("injection_key") is not None
        else "",
        str(node.get("id", "")),
    )
