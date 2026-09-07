# Memory Service - Agent 调用层(规格 Rule 7: Agent → Service → Repository)
# 运行指南:
#   service = get_memory_service()
#   saved = service.remember("我喜欢车内24度", user_id, session_id)
#   pref = service.recall_preference(user_id, "preferred_temperature")
#   Phase 6+: 保存时自动计算 embedding 供 pgvector 语义检索

import logging

from app.config import get_app_config
from app.memory.embedding import EmbeddingProvider, get_embedding_provider
from app.memory.extractor import MemoryExtractor
from app.memory.models import Memory
from app.memory.repository import MemoryRepository, get_memory_repository
from app.memory.retriever import MemoryRetriever
from app.memory.temporal import resolve_conflict

logger = logging.getLogger(__name__)


class MemoryService:
    """记忆服务,编排提取、Embedding、冲突解决、检索。"""

    def __init__(
        self,
        extractor: MemoryExtractor,
        retriever: MemoryRetriever,
        repository: MemoryRepository,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._extractor = extractor
        self._retriever = retriever
        self._repo = repository
        self._embed = embedding_provider or get_embedding_provider()

    def remember(
        self, user_message: str, user_id: str, session_id: str
    ) -> list[Memory]:
        """提取候选记忆,计算 embedding,冲突解决后持久化。"""
        candidates = self._extractor.extract(user_message, user_id, session_id)
        saved: list[Memory] = []
        for candidate in candidates:
            self._ensure_embedding(candidate)
            self._resolve_and_persist(candidate)
            saved.append(candidate)
        if saved:
            logger.info(
                "Persisted %d memories for user=%s (backend=%s)",
                len(saved), user_id, self._repo.backend,
            )
        return saved

    def _ensure_embedding(self, memory: Memory) -> None:
        """若记忆未填充 embedding,则使用 provider 计算。任何失败均记录日志并跳过。"""
        if memory.embedding is not None:
            return
        try:
            memory.embedding = self._embed.embed(memory.to_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding compute failed for memory %s: %s", memory.id, exc)

    def _resolve_and_persist(self, candidate: Memory) -> None:
        """对新记忆执行冲突解决,失效同谓词旧记忆后保存。"""
        active_old = self._repo.find_active(candidate.user_id, candidate.predicate)
        for old in active_old:
            if old.predicate == candidate.predicate and old.type == candidate.type:
                self._repo.deactivate(old.id)
        resolve_conflict(candidate, active_old)
        self._repo.save(candidate)

    def recall_preference(self, user_id: str, predicate: str) -> Memory | None:
        """检索指定谓词的最新有效偏好(规则优先,不依赖向量)。"""
        return self._retriever.retrieve_preference(user_id, predicate)

    def recall_relevant(self, user_id: str, query: str) -> list[Memory]:
        """检索与查询相关的记忆:向量语义检索优先,降级关键词匹配。"""
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
    embed = get_embedding_provider()
    retriever = MemoryRetriever(repo, top_k=config.retrieval_top_k, embedding_provider=embed)
    extractor = MemoryExtractor(min_confidence=config.min_confidence)
    _default_service = MemoryService(extractor, retriever, repo, embedding_provider=embed)
    return _default_service


def reset_memory_service() -> None:
    """重置单例(测试用)。"""
    global _default_service
    _default_service = None
