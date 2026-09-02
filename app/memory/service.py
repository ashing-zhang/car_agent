# Memory Service - Agent 调用层(规格 Rule 7: Agent → Service → Repository)
# 运行指南:
#   service = get_memory_service()
#   saved = service.remember("我喜欢车内24度", user_id, session_id)
#   pref = service.recall_preference(user_id, "preferred_temperature")

import logging

from app.config import get_app_config
from app.memory.extractor import MemoryExtractor
from app.memory.models import Memory
from app.memory.repository import MemoryRepository, get_memory_repository
from app.memory.retriever import MemoryRetriever
from app.memory.temporal import resolve_conflict

logger = logging.getLogger(__name__)


class MemoryService:
    """记忆服务,编排提取、冲突解决、检索。"""

    def __init__(
        self,
        extractor: MemoryExtractor,
        retriever: MemoryRetriever,
        repository: MemoryRepository,
    ) -> None:
        """注入提取器、检索器、仓储。"""
        self._extractor = extractor
        self._retriever = retriever
        self._repo = repository

    def remember(
        self, user_message: str, user_id: str, session_id: str
    ) -> list[Memory]:
        """提取候选记忆,执行冲突解决后持久化。"""
        candidates = self._extractor.extract(user_message, user_id, session_id)
        saved: list[Memory] = []
        for candidate in candidates:
            self._resolve_and_persist(candidate)
            saved.append(candidate)
        if saved:
            logger.info("Persisted %d memories for user=%s", len(saved), user_id)
        return saved

    def _resolve_and_persist(self, candidate: Memory) -> None:
        """对新记忆执行冲突解决,失效同谓词旧记忆后保存。"""
        active_old = self._repo.find_active(candidate.user_id, candidate.predicate)
        for old in active_old:
            if old.predicate == candidate.predicate and old.type == candidate.type:
                self._repo.deactivate(old.id)
        resolve_conflict(candidate, active_old)
        self._repo.save(candidate)

    def recall_preference(self, user_id: str, predicate: str) -> Memory | None:
        """检索指定谓词的最新有效偏好。"""
        return self._retriever.retrieve_preference(user_id, predicate)

    def recall_relevant(self, user_id: str, query: str) -> list[Memory]:
        """检索与查询相关的记忆。"""
        return self._retriever.retrieve_relevant(user_id, query)

    def has_preference(self, user_id: str, predicate: str) -> bool:
        """判断用户是否存在有效偏好。"""
        return self.recall_preference(user_id, predicate) is not None


_default_service: MemoryService | None = None


def get_memory_service() -> MemoryService:
    """获取默认 MemoryService 单例。"""
    global _default_service
    if _default_service is not None:
        return _default_service
    config = get_app_config().memory
    repo = get_memory_repository()
    retriever = MemoryRetriever(repo, top_k=config.retrieval_top_k)
    extractor = MemoryExtractor(min_confidence=config.min_confidence)
    _default_service = MemoryService(extractor, retriever, repo)
    return _default_service
