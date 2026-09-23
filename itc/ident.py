"""稳定身份与内容指纹。

材料、派生结果、事件都使用结构化内容的指纹，而不是自增整数：
这样导出后删除本地数据库再导入，身份、顺序与来源关系仍保持。

规范化 JSON：键排序、无空白、浮点数往返格式（repr，保证 numpy 舍入一致）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from . import RULES_VERSION


def _default(obj: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"不可规范化的对象: {type(obj)!r}")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=_default, allow_nan=False)


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def short_hash(obj: Any, length: int = 12) -> str:
    return content_hash(obj)[:length]


def material_id(kind: str, payload: dict) -> str:
    """不可变材料身份：种类 + 完整内容（含 raw 表示）。"""
    return f"mat_{kind}_{short_hash({'v': 1, 'kind': kind, 'payload': payload})}"


def derived_id(kind: str, inputs: dict, params: dict) -> str:
    """派生结果身份：类型 + 输入摘要 + 参数版本 + 父级依赖。

    等价输入（指纹相同）得到相同身份与稳定顺序，绝不依赖生成时刻。
    """
    return f"der_{kind}_{short_hash({'v': 1, 'kind': kind, 'inputs': inputs, 'params': params})}"


def event_id(seq: int, payload: dict) -> str:
    return f"evt_{seq:05d}_{short_hash(payload, 8)}"


def make_event(seq: int, etype: str, payload: dict, branch: str,
               timestamp: str, parent_hash: str, rules_version: str = RULES_VERSION) -> dict:
    """构造仅追加事件。链上每环包含前一环哈希，形成可重放证据。"""
    body = {
        "seq": seq,
        "type": etype,
        "branch": branch,
        "ts": timestamp,
        "rules": rules_version,
        "payload": payload,
        "prev": parent_hash,
    }
    body["hash"] = "sha256:" + content_hash({k: v for k, v in body.items() if k != "hash"})
    body["id"] = event_id(seq, {"type": etype, "payload": payload, "branch": branch, "ts": timestamp})
    return body


def stable_order_key(ev: dict) -> tuple:
    """等价输入的稳定顺序：(全局序号, 分支, 身份)。"""
    return (ev["seq"], ev["branch"], ev["id"])
