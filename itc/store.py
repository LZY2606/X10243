"""SQLite 存储层: 不可变材料(分种类版本化) + 事件 + 派生结果 + 操作日志。

- 每次写操作先登记 op_log(state='applying') 并提交, 再在单事务内写业务表,
  最后标记 'done'。崩溃后重开时 recover() 把残留的 'applying' 标记为 'aborted'
  (其事务已随连接回滚, 不会留下半写状态)。
- 身份稳定: material/result 的 id 由种类 + 内容摘要派生; event 的 id 由序号 + 摘要派生。
- 等价输入得到稳定顺序: 同一 (kind, digest) 幂等, 保留首次序号。
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS materials (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  version INTEGER NOT NULL,
  payload TEXT NOT NULL,
  digest TEXT NOT NULL,
  seq INTEGER NOT NULL,
  supersedes TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  seq INTEGER NOT NULL,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  parents TEXT NOT NULL,
  input_digest TEXT NOT NULL,
  param_version TEXT NOT NULL,
  status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  event_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  payload TEXT NOT NULL,
  digest TEXT NOT NULL,
  status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS op_log (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  state TEXT NOT NULL,
  detail TEXT
);
"""

MATERIAL_KINDS = (
    "power_curve", "injection_schedule", "concentrations", "temperature",
    "cell_volume", "baseline", "blank_curve",
)


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_of(obj) -> str:
    return "sha256:" + hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


class Store:
    def __init__(self, path=":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self.recover()

    def close(self):
        self.conn.close()

    # ---------- 崩溃恢复 ----------
    def recover(self):
        """把上次崩溃残留的 'applying' 操作标记为 'aborted'。

        业务写都在单事务内, 崩溃即整体回滚, 因此这里只需修正操作日志,
        不存在半写的材料/事件/结果。
        """
        cur = self.conn.execute("SELECT id FROM op_log WHERE state='applying'")
        stale = [r["id"] for r in cur.fetchall()]
        for op_id in stale:
            self.conn.execute("UPDATE op_log SET state='aborted' WHERE id=?", (op_id,))
        self.conn.commit()
        return stale

    # ---------- 序号 ----------
    def _next_seq(self):
        row = self.conn.execute("SELECT value FROM meta WHERE key='seq'").fetchone()
        seq = 1 if row is None else int(row["value"]) + 1
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES('seq',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(seq),),
        )
        return seq

    def head_seq(self):
        row = self.conn.execute("SELECT value FROM meta WHERE key='seq'").fetchone()
        return 0 if row is None else int(row["value"])

    # ---------- 事务化操作 ----------
    @contextlib.contextmanager
    def operation(self, kind, detail=None):
        """一个原子操作: 登记 op_log -> 事务写 -> 标记 done。"""
        op_id = f"op-{int(time.time()*1e6)}-{kind}"
        self.conn.execute(
            "INSERT INTO op_log(id,kind,state,detail) VALUES(?,?,?,?)",
            (op_id, kind, "applying", json.dumps(detail, ensure_ascii=False) if detail else None),
        )
        self.conn.commit()
        try:
            with self.conn:  # 单事务; 异常即整体回滚
                yield self
            self.conn.execute("UPDATE op_log SET state='done' WHERE id=?", (op_id,))
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            self.conn.execute("UPDATE op_log SET state='aborted' WHERE id=?", (op_id,))
            self.conn.commit()
            raise

    # ---------- 材料 ----------
    def put_material(self, kind, payload, supersedes=None):
        assert kind in MATERIAL_KINDS, f"未知材料种类 {kind}"
        dg = digest_of({"kind": kind, "payload": payload})
        mat_id = f"mat/{kind}/{dg.split(':')[1][:12]}"
        row = self.conn.execute("SELECT * FROM materials WHERE id=?", (mat_id,)).fetchone()
        if row:
            return self._mat(row)
        seq = self._next_seq()
        vrow = self.conn.execute(
            "SELECT COALESCE(MAX(version),0) AS v FROM materials WHERE kind=?", (kind,)
        ).fetchone()
        version = int(vrow["v"]) + 1
        self.conn.execute(
            "INSERT INTO materials(id,kind,version,payload,digest,seq,supersedes) "
            "VALUES(?,?,?,?,?,?,?)",
            (mat_id, kind, version, canonical(payload), dg, seq, supersedes),
        )
        return self.material(mat_id)

    def _mat(self, row):
        return {
            "id": row["id"], "kind": row["kind"], "version": row["version"],
            "payload": json.loads(row["payload"]), "digest": row["digest"],
            "seq": row["seq"], "supersedes": row["supersedes"],
        }

    def material(self, mat_id):
        row = self.conn.execute("SELECT * FROM materials WHERE id=?", (mat_id,)).fetchone()
        return self._mat(row) if row else None

    def materials(self, kind=None, as_of=None):
        sql, args = "SELECT * FROM materials", []
        conds = []
        if kind:
            conds.append("kind=?"); args.append(kind)
        if as_of is not None:
            conds.append("seq<=?"); args.append(as_of)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY seq"
        return [self._mat(r) for r in self.conn.execute(sql, args).fetchall()]

    def latest_material(self, kind, as_of=None):
        ms = self.materials(kind=kind, as_of=as_of)
        return ms[-1] if ms else None

    # ---------- 事件 ----------
    def put_event(self, kind, payload, parents=(), input_digest="", param_version="", status="ok"):
        seq = self._next_seq()
        dg = digest_of({"kind": kind, "payload": payload, "parents": list(parents), "seq": seq})
        evt_id = f"evt/{seq:06d}/{dg.split(':')[1][:8]}"
        self.conn.execute(
            "INSERT INTO events(id,seq,kind,payload,parents,input_digest,param_version,status) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (evt_id, seq, kind, canonical(payload), canonical(list(parents)),
             input_digest, param_version, status),
        )
        return self.event(evt_id)

    def _evt(self, row):
        return {
            "id": row["id"], "seq": row["seq"], "kind": row["kind"],
            "payload": json.loads(row["payload"]), "parents": json.loads(row["parents"]),
            "input_digest": row["input_digest"], "param_version": row["param_version"],
            "status": row["status"],
        }

    def event(self, evt_id):
        row = self.conn.execute("SELECT * FROM events WHERE id=?", (evt_id,)).fetchone()
        return self._evt(row) if row else None

    def events(self, as_of=None, kind=None):
        sql, args, conds = "SELECT * FROM events", [], []
        if as_of is not None:
            conds.append("seq<=?"); args.append(as_of)
        if kind:
            conds.append("kind=?"); args.append(kind)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY seq"
        return [self._evt(r) for r in self.conn.execute(sql, args).fetchall()]

    # ---------- 结果 ----------
    def put_result(self, kind, event_id, payload, status="ok"):
        dg = digest_of({"kind": kind, "payload": payload})
        res_id = f"res/{kind}/{dg.split(':')[1][:12]}"
        row = self.conn.execute("SELECT * FROM results WHERE id=?", (res_id,)).fetchone()
        if row:
            return self._res(row)  # 幂等: 等价输入 -> 同一身份, 保留首次序号
        seq = self._next_seq()
        self.conn.execute(
            "INSERT INTO results(id,kind,event_id,seq,payload,digest,status) VALUES(?,?,?,?,?,?,?)",
            (res_id, kind, event_id, seq, canonical(payload), dg, status),
        )
        return self.result(res_id)

    def _res(self, row):
        return {
            "id": row["id"], "kind": row["kind"], "event_id": row["event_id"],
            "seq": row["seq"], "payload": json.loads(row["payload"]),
            "digest": row["digest"], "status": row["status"],
        }

    def result(self, res_id):
        row = self.conn.execute("SELECT * FROM results WHERE id=?", (res_id,)).fetchone()
        return self._res(row) if row else None

    def results(self, kind=None, as_of=None):
        sql, args, conds = "SELECT * FROM results", [], []
        if kind:
            conds.append("kind=?"); args.append(kind)
        if as_of is not None:
            conds.append("seq<=?"); args.append(as_of)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY seq"
        return [self._res(r) for r in self.conn.execute(sql, args).fetchall()]

    # ---------- 导出辅助 ----------
    def dump(self):
        """全库快照(稳定顺序), 用于导出与往返一致性校验。"""
        ops = [dict(r) for r in self.conn.execute(
            "SELECT * FROM op_log ORDER BY rowid").fetchall()]
        return {
            "meta": {r["key"]: r["value"] for r in self.conn.execute("SELECT * FROM meta")},
            "materials": [self._mat(r) for r in self.conn.execute(
                "SELECT * FROM materials ORDER BY seq")],
            "events": [self._evt(r) for r in self.conn.execute(
                "SELECT * FROM events ORDER BY seq")],
            "results": [self._res(r) for r in self.conn.execute(
                "SELECT * FROM results ORDER BY seq")],
            "op_log": ops,
        }
