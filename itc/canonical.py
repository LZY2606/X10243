"""确定性编码：等价输入得到稳定顺序与稳定身份的基础。"""

import hashlib
import json


def canonical_json(obj) -> bytes:
    """sort_keys + 固定分隔符 + 不转义非 ASCII。"""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def content_hash(obj) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj)).hexdigest()


def short_id(obj, prefix: str) -> str:
    return f"{prefix}_{hashlib.sha256(canonical_json(obj)).hexdigest()[:16]}"


def stable_key(rec) -> str:
    """记录的稳定排序键：优先显式 ordinal，再退化为规范编码。"""
    if isinstance(rec, dict):
        for k in ("ordinal", "index", "injection_no", "seq", "t_start"):
            if k in rec:
                return (0, rec[k], canonical_json(rec))
    return (1, b"", canonical_json(rec))
