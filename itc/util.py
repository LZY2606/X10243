import hashlib
import json


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj):
    return hashlib.sha256(canon(obj).encode("utf-8")).hexdigest()
