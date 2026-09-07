# Memory Repository - 记忆持久化(规格 Rule 7: Service → Repository → Database)
# 运行指南:
#   默认 InMemoryMemoryRepository,不依赖外部数据库(Rule 5)
#   Phase 6+: 可切换为 PostgreSQLMemoryRepository (pgvector 语义检索)
#   通过配置 memory.repository_backend 或 build_repository() 构造

import logging
from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy import and_, or_, select

from app.memory.models import Memory, MemoryType

logger = logging.getLogger(__name__)


class MemoryRepository(Protocol):
    """记忆仓储接口(委托模式,支持多后端)。"""

    backend: str

    def save(self, memory: Memory) -> Memory:
        """保存一条记忆。"""
        ...

    def find_by_user(
        self, user_id: str, mem_type: MemoryType | None = None
    ) -> list[Memory]:
        """查询某用户全部记忆(可按类型过滤)。"""
        ...

    def find_active(
        self, user_id: str, predicate: str | None = None
    ) -> list[Memory]:
        """查询某用户当前有效记忆。"""
        ...

    def find_similar(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> list[tuple[float, Memory]]:
        """向量相似度检索 (pgvector 后端可用,否则空列表)。

        返回 [(cosine_similarity, Memory), ...] 已按相似度降序。
        """
        ...

    def deactivate(self, memory_id: str) -> None:
        """将指定记忆标记为失效(valid_to 设为当前时间)。"""
        ...


class InMemoryMemoryRepository:
    """基于内存字典的 Memory 仓储实现,无需外部依赖。"""

    backend: str = "in_memory"

    def __init__(self) -> None:
        """初始化按 user_id 索引的存储。"""
        self._store: dict[str, list[Memory]] = {}

    def save(self, memory: Memory) -> Memory:
        """保存记忆(追加,冲突由 temporal 层处理)。"""
        self._store.setdefault(memory.user_id, []).append(memory)
        logger.info(
            "Memory saved: user=%s predicate=%s value=%s",
            memory.user_id, memory.predicate, memory.value,
        )
        return memory

    def find_by_user(
        self, user_id: str, mem_type: MemoryType | None = None
    ) -> list[Memory]:
        """返回某用户全部记忆,可按类型过滤。"""
        items = list(self._store.get(user_id, []))
        if mem_type is not None:
            items = [m for m in items if m.type == mem_type]
        return items

    def find_active(
        self, user_id: str, predicate: str | None = None
    ) -> list[Memory]:
        """返回某用户当前有效记忆,可按谓词过滤。"""
        items = self.find_by_user(user_id)
        active = [m for m in items if m.is_active()]
        if predicate is not None:
            active = [m for m in active if m.predicate == predicate]
        return active

    def find_similar(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> list[tuple[float, Memory]]:
        """内存后端降级实现:基于 embedding 字段做精确余弦相似度。

        若保存时未填充 embedding,则返回空列表。
        """
        active = self.find_active(user_id)
        scored: list[tuple[float, Memory]] = []
        for mem in active:
            if mem.embedding is None:
                continue
            sim = _cosine_similarity(query_embedding, mem.embedding)
            if sim >= min_similarity:
                scored.append((sim, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def deactivate(self, memory_id: str) -> None:
        """将指定记忆置为失效(就地更新 valid_to)。"""
        for items in self._store.values():
            for idx, mem in enumerate(items):
                if mem.id == memory_id:
                    now = datetime.now(timezone.utc)
                    items[idx] = mem.model_copy(update={"valid_to": now})
                    logger.info("Memory %s deactivated", memory_id)
                    return


class PostgreSQLMemoryRepository:
    """基于 PostgreSQL + pgvector 的 Memory 仓储实现。"""

    backend: str = "postgres"

    def __init__(self) -> None:
        """延迟加载 ORM 模块,避免未安装 pgvector 时 import 失败。"""
        from app.memory.orm import new_session  # noqa: F401
        self._session_cls = new_session
        logger.info("PostgreSQLMemoryRepository initialized (pgvector backend)")

    def save(self, memory: Memory) -> Memory:
        """Upsert 一条记忆:若 id 存在则更新,否则插入。"""
        from app.memory.orm import memory_to_orm, orm_to_memory

        with self._session_cls() as session:
            from app.memory.orm import MemoryORM

            existing = session.get(MemoryORM, memory.id)
            orm_obj = memory_to_orm(memory)
            if existing is None:
                session.add(orm_obj)
            else:
                for col in MemoryORM.__table__.columns.keys():
                    if col == "id":
                        continue
                    setattr(existing, col, getattr(orm_obj, col))
            session.commit()
            logger.info(
                "Memory saved(pg): user=%s predicate=%s value=%s",
                memory.user_id, memory.predicate, memory.value,
            )
            return memory

    def find_by_user(
        self, user_id: str, mem_type: MemoryType | None = None
    ) -> list[Memory]:
        from app.memory.orm import MemoryORM, orm_to_memory

        with self._session_cls() as session:
            stmt = select(MemoryORM).where(MemoryORM.user_id == user_id)
            if mem_type is not None:
                stmt = stmt.where(MemoryORM.type == mem_type.value)
            stmt = stmt.order_by(MemoryORM.valid_from.desc())
            rows = session.execute(stmt).scalars().all()
            return [orm_to_memory(r) for r in rows]

    def find_active(
        self, user_id: str, predicate: str | None = None
    ) -> list[Memory]:
        from app.memory.orm import MemoryORM, orm_to_memory

        now = datetime.now(timezone.utc)
        with self._session_cls() as session:
            stmt = select(MemoryORM).where(
                MemoryORM.user_id == user_id,
                MemoryORM.valid_from <= now,
                or_(MemoryORM.valid_to.is_(None), MemoryORM.valid_to > now),
            )
            if predicate is not None:
                stmt = stmt.where(MemoryORM.predicate == predicate)
            stmt = stmt.order_by(MemoryORM.valid_from.desc())
            rows = session.execute(stmt).scalars().all()
            return [orm_to_memory(r) for r in rows]

    def find_similar(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> list[tuple[float, Memory]]:
        """pgvector ANN 检索:使用余弦距离 (<=> 操作符),返回相似度。"""
        from app.memory.orm import MemoryORM, orm_to_memory
        from sqlalchemy import func

        now = datetime.now(timezone.utc)
        with self._session_cls() as session:
            distance = MemoryORM.embedding.cosine_distance(query_embedding)
            stmt = (
                select(MemoryORM, (1 - distance).label("sim"))
                .where(
                    MemoryORM.user_id == user_id,
                    MemoryORM.embedding.is_not(None),
                    MemoryORM.valid_from <= now,
                    or_(MemoryORM.valid_to.is_(None), MemoryORM.valid_to > now),
                )
                .order_by(distance.asc())
                .limit(top_k)
            )
            rows = session.execute(stmt).all()
            return [
                (float(r.sim), orm_to_memory(r.MemoryORM))
                for r in rows
                if float(r.sim) >= min_similarity
            ]

    def deactivate(self, memory_id: str) -> None:
        from app.memory.orm import MemoryORM

        now = datetime.now(timezone.utc)
        with self._session_cls() as session:
            row = session.get(MemoryORM, memory_id)
            if row is not None:
                row.valid_to = now
                session.commit()
                logger.info("Memory %s deactivated(pg)", memory_id)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


_default_repo: MemoryRepository | None = None


def build_repository(backend: str = "in_memory") -> MemoryRepository:
    """按 backend 名称构造仓储实例。in_memory | postgres"""
    if backend == "postgres":
        return PostgreSQLMemoryRepository()
    return InMemoryMemoryRepository()


def get_memory_repository() -> MemoryRepository:
    """获取默认 Memory 仓储单例。读取配置 memory.repository_backend。"""
    global _default_repo
    if _default_repo is None:
        try:
            from app.config import get_app_config
            backend = get_app_config().memory.repository_backend
        except Exception:
            backend = "in_memory"
        _default_repo = build_repository(backend)
    return _default_repo


def reset_memory_repository() -> None:
    """重置单例(测试用)。"""
    global _default_repo
    _default_repo = None
