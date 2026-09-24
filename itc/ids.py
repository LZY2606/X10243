"""稳定身份（stable identity）。

身份完全由内容决定，而不是由数据库自增主键决定：
- 材料/结果用内容哈希得到稳定 token；
- 事件身份绑定材料 token 与来源（source），使“同一直接来源改动”可被精确识别；
- 导出删除后再导入，同一内容得到同一身份与同一等价顺序。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """确定性 JSON：键排序、紧凑分隔符、保证 unicode 原样保留。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(obj: Any, *, prefix: str = "", length: int = 16) -> str:
    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:length]}" if prefix else digest[:length]


def material_id(kind: str, key: str, raw: str, source: str) -> str:
    return content_hash(
        {"kind": kind, "key": key, "raw": raw, "source": source}, prefix="mat_"
    )


def result_id(kind: str, input_summary: dict[str, Any]) -> str:
    return content_hash({"kind": kind, "input": input_summary}, prefix="res_")


def event_id(seq: int, kind: str, token: str, branch: str) -> str:
    """事件稳定身份：序号 + 类型 + 主体 token + 分支。

    序号在同一项目、同一线性化序列上确定；重放产生完全一致的历史。
    """
    return content_hash(
        {"seq": seq, "kind": kind, "token": token, "branch": branch}, prefix="evt_"
    )


def stable_ordinal(token: str) -> int:
    """等价输入的稳定顺序键：取哈希前 8 个十六进制位。"""
    return int(token[-8:], 16)
