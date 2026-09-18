# Memory Repository - 记忆持久化(规格 Rule 7: Service → Repository → Database)
# 运行指南:
#   企业级统一使用 PostgreSQLMemoryRepository (pgvector 语义检索)
#   通过 build_repository() 或 get_memory_repository() 构造实例

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


class PostgreSQLMemoryRepository:
    """基于 PostgreSQL + pgvector 的 Memory 仓储实现(企业级生产后端)。"""

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


_default_repo: MemoryRepository | None = None


def build_repository() -> MemoryRepository:
    """构造 MemoryRepository 企业级实例(PostgreSQL + pgvector)。"""
    return PostgreSQLMemoryRepository()


def get_memory_repository() -> MemoryRepository:
    """获取默认 Memory 仓储单例(PostgreSQL 后端)。"""
    global _default_repo
    if _default_repo is None:
        _default_repo = build_repository()
    return _default_repo


def reset_memory_repository() -> None:
    """重置单例(测试用)。"""
    global _default_repo
    _default_repo = None
