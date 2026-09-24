"""不可变材料与项目结构。

分层原则：
- MaterialVersion 是不可变材料（功率采样、注射时序、浓度、温度、池体积、
  基线、空白曲线各自独立 material_id，各自版本化）。
- Event 是只追加的决定/状态（接受、否定、端点调整、规则选择、分支）。
- Result 是派生层产物（积分、绑定模型拟合），记录输入摘要、参数版本
  与父级依赖，永远不改写历史材料。
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

MATERIAL_KINDS = (
    "power_sample",   # 池功率采样 P(t)，run_role=sample|blank
    "injections",     # 注射时序：每注的体积与时间
    "syringe_conc",   # 注射器浓度（可修订 -> 新版本）
    "cell_conc",      # 池中大分子浓度
    "temperature",    # 温度
    "cell_volume",    # 池体积
    "baseline",       # 基线说明（前段/后段窗口或空白链接）
)
SAMPLE_KIND = "sample"
BLANK_KIND = "blank"


def canon_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_hash(content: Any) -> str:
    return hashlib.sha256(canon_json(content).encode("utf-8")).hexdigest()


def stable_id(namespace: uuid.UUID, *parts: str) -> str:
    return str(uuid.uuid5(namespace, "|".join(parts)))


PROJECT_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def material_identity(project_id: str, kind: str, role: str) -> str:
    """同 kind+role 的多个版本共享同一稳定身份。"""
    return stable_id(PROJECT_NS, project_id, "mat", kind, role)


def version_identity(material_id: str, digest: str) -> str:
    return stable_id(PROJECT_NS, material_id, digest)


@dataclass
class Point:
    t_raw: str
    p_raw: str | None = None
    missing: bool = False
    note: str | None = None


@dataclass
class MaterialVersion:
    project_id: str
    kind: str
    role: str               # "sample" / "blank" / "main"
    version: int
    content: dict[str, Any]
    origin: str             # 直接来源：文件名/人工/修订
    created_event: str
    supersedes: str | None = None
    material_id: str = ""
    version_id: str = ""
    content_digest: str = ""

    def __post_init__(self):
        if not self.material_id:
            self.material_id = material_identity(self.project_id, self.kind, self.role)
        self.content_digest = self.content_digest or content_hash(self.content)
        if not self.version_id:
            self.version_id = version_identity(self.material_id, self.content_digest)

    def to_row(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict) -> "MaterialVersion":
        return cls(**row)


def current_versions(versions: list[MaterialVersion]) -> dict[tuple[str, str], MaterialVersion]:
    """每个 (kind, role) 身份的最新版本；版本号相同则取确定性较小 version_id。"""
    out: dict[tuple[str, str], MaterialVersion] = {}
    for mv in sorted(versions, key=lambda v: (v.kind, v.role, v.version, v.version_id)):
        key = (v.kind, v.role)
        cur = out.get(key)
        if cur is None or (v.version, v.version_id) >= (cur.version, cur.version_id):
            out[key] = v
    return out


def parse_points(series: list[dict]) -> list[Point]:
    out = []
    for i, item in enumerate(series):
        out.append(Point(
            t_raw=str(item["t"]),
            p_raw=None if item.get("p") is None else str(item["p"]),
            missing=bool(item.get("missing", item.get("p") is None)),
            note=item.get("note"),
        ))
    return out


def gaps_from_points(points: list[Point], *, location: str) -> list[tuple[int, int]]:
    """返回缺口索引区间 [(start, end_exclusive), ...]，缺口处不插值。"""
    gaps = []
    i = 0
    n = len(points)
    while i < n:
        if points[i].missing:
            j = i
            while j < n and points[j].missing:
                j += 1
            gaps.append((i, j))
            i = j
        else:
            i += 1
    return gaps
