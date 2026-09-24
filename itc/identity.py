"""稳定身份：等价输入得到稳定顺序与稳定 ID。

ID 全部由内容哈希导出（content-addressed），不依赖自增主键，
因此导出->删除库->导入后身份与顺序保持。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    """确定性 JSON 序列化（键排序、无空白差异）。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(prefix: str, *parts: Any) -> str:
    payload = canonical(parts if len(parts) > 1 else parts[0])
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


def material_id(kind: str, source_ref: str, revision: int, canonical_content: Any) -> str:
    return digest("mat", [kind, source_ref, revision, canonical_content])


def rule_id(rule_set: str, version: str, content: Any) -> str:
    return digest("rule", [rule_set, version, content])


def result_id(result_kind: str, input_summary: Any, params_version: Any, parents: list[str]) -> str:
    return digest("res", [result_kind, input_summary, params_version, sorted(parents)])


def event_id(seq: int, branch: str, action: str, payload: Any, parent_event: str | None) -> str:
    # seq 保证分支内顺序稳定；内容保证跨库等价事件同 ID
    return digest("evt", [seq, branch, action, payload, parent_event])


def point_id(parent: str, label: str) -> str:
    return digest("pt", [parent, label])
