"""确定性身份。

身份是内容寻址的：相同的规范内容永远得到相同 id；
材料按 (kind, branch_id, lineage, content_hash) 版本化，
派生结果按 (kind, inputs hash) 身份化 —— 等价输入因此得到稳定顺序。
"""
from __future__ import annotations

import hashlib
import json
import uuid

NAMESPACE = uuid.UUID("7f4a9c21-5e3b-4a6d-9b8c-1d2e3f405161")


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def content_hash(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def stable_id(kind: str, payload: object) -> str:
    """uuid5 → 确定性 id，前缀带领域类型。"""
    h = content_hash(payload)
    return f"{kind}_{uuid.uuid5(NAMESPACE, h).hex[:16]}"


def short_hash(payload: object) -> str:
    return content_hash(payload)[:12]
