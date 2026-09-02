# Memory Repository - 记忆持久化(规格 Rule 7: Service → Repository → Database)
# 运行指南:
#   默认 InMemoryMemoryRepository,不依赖外部数据库(Rule 5)
#   后续可替换为 SQLAlchemyMemoryRepository(PostgreSQL + pgvector)

import logging
from datetime import datetime, timezone
from typing import Protocol

from app.memory.models import Memory, MemoryType

logger = logging.getLogger(__name__)


class MemoryRepository(Protocol):
    """记忆仓储接口(委托模式,支持多后端)。"""

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

    def deactivate(self, memory_id: str) -> None:
        """将指定记忆标记为失效(valid_to 设为当前时间)。"""
        ...


class InMemoryMemoryRepository:
    """基于内存字典的 Memory 仓储实现,无需外部依赖。"""

    def __init__(self) -> None:
        """初始化按 user_id 索引的存储。"""
        self._store: dict[str, list[Memory]] = {}

    def save(self, memory: Memory) -> Memory:
        """保存记忆(追加,冲突由 temporal 层处理)。"""
        self._store.setdefault(memory.user_id, []).append(memory)
        logger.info("Memory saved: user=%s predicate=%s value=%s", memory.user_id, memory.predicate, memory.value)
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

    def deactivate(self, memory_id: str) -> None:
        """将指定记忆置为失效(就地更新 valid_to)。"""
        for items in self._store.values():
            for idx, mem in enumerate(items):
                if mem.id == memory_id:
                    now = datetime.now(timezone.utc)
                    items[idx] = mem.model_copy(update={"valid_to": now})
                    logger.info("Memory %s deactivated", memory_id)
                    return


_default_repo: MemoryRepository | None = None


def get_memory_repository() -> MemoryRepository:
    """获取默认 Memory 仓储单例。"""
    global _default_repo
    if _default_repo is None:
        _default_repo = InMemoryMemoryRepository()
    return _default_repo
