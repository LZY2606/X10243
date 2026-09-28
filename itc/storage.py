def stable_event_id(project_id: str, branch: str, seq: int, kind: str, payload: Dict[str, Any]) -> str:
    # 身份哈希 *不含* 分支名：建分支时复制的祖先事件保持同一身份，
    # 仅分叉后不同载荷产生不同事件 ID。
    h = hashlib.sha256()
    h.update(f"{project_id}|{seq}|{kind}|".encode())
    h.update(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())
    return "evt_" + h.hexdigest()[:16]