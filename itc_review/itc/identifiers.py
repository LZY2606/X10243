"""稳定身份：同一输入摘要永远得到同一身份，保证等价输入顺序稳定。"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    """确定性 JSON（键排序、无空白差异）。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_id(kind: str, payload: Any) -> str:
    """内容寻址身份：kind + 规范化 JSON 的 sha256 前 16 位。"""
    h = hashlib.sha256(f"{kind}|{canonical(payload)}".encode("utf-8")).hexdigest()
    return f"{kind}_{h[:16]}"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


MATERIAL_PREFIX = {
    "power": "mat_pow",
    "schedule": "mat_inj",
    "concentration": "mat_con",
    "temperature": "mat_tmp",
    "cellvolume": "mat_cel",
    "baseline": "mat_bas",
    "blank": "mat_blk",
}


def material_id(kind: str, version: int, raw_text: str) -> str:
    """材料身份由 类型 + 版本号 + 原始字节 决定（原始表示逐字保留）。"""
    h = hash_text(raw_text)[:16]
    return f"{MATERIAL_PREFIX[kind]}_v{version}_{h}"
