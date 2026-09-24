"""规范化 JSON、稳定摘要与稳定身份生成。"""
import hashlib
import json


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj):
    """内容摘要：等价输入得到等价摘要，从而得到稳定顺序与稳定身份。"""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:16]


def short_digest(obj, n=8):
    return digest(obj)[:n]
