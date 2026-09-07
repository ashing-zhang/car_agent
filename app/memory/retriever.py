# Memory Retriever - 记忆检索(规格第14节 Context Aggregation)
# 运行指南:
#   Phase 6+: 优先 pgvector 语义向量检索;若记忆未填充 embedding 则降级关键词匹配
#   Retrieval 时 latest valid preference 优先于 historical(规格第12节)

import logging
from typing import TYPE_CHECKING

from app.memory.embedding import EmbeddingProvider, get_embedding_provider
from app.memory.models import Memory, MemoryType, PREFERENCE_PREDICATES
from app.memory.temporal import get_latest_active

if TYPE_CHECKING:
    from app.memory.repository import MemoryRepository

logger = logging.getLogger(__name__)

SEMANTIC_MIN_SIMILARITY = 0.35
SEMANTIC_SCORE_WEIGHT = 0.6
KEYWORD_SCORE_WEIGHT = 0.4


class MemoryRetriever:
    """记忆检索器:向量语义检索优先,关键词匹配作为降级/辅助。"""

    def __init__(
        self,
        repository: "MemoryRepository",
        top_k: int = 5,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._repo = repository
        self._top_k = top_k
        self._embed = embedding_provider or get_embedding_provider()

    def retrieve_preference(
        self, user_id: str, predicate: str
    ) -> Memory | None:
        """返回指定谓词的最新有效偏好(规则检索,不依赖向量)。"""
        active = self._repo.find_active(user_id, predicate)
        latest = get_latest_active(active, predicate)
        if latest is not None:
            logger.info("Retrieved preference %s=%s", predicate, latest.value)
        return latest

    def retrieve_relevant(
        self, user_id: str, query: str
    ) -> list[Memory]:
        """根据查询文本检索相关记忆:向量优先 -> 关键词降级。"""
        results: list[Memory] = []
        try:
            query_vec = self._embed.embed(query)
            similar = self._repo.find_similar(
                user_id, query_vec,
                top_k=self._top_k * 2,
                min_similarity=SEMANTIC_MIN_SIMILARITY,
            )
            if similar:
                scored = [
                    (
                        sim * SEMANTIC_SCORE_WEIGHT
                        + self._keyword_score(mem, query) * KEYWORD_SCORE_WEIGHT,
                        mem,
                    )
                    for sim, mem in similar
                ]
                scored.sort(key=lambda x: x[0], reverse=True)
                results = [m for _, m in scored[: self._top_k]]
                logger.info(
                    "Retrieved %d relevant memories (vector-based, backend=%s)",
                    len(results), self._repo.backend,
                )
                return results
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector retrieval failed, fallback to keyword: %s", exc)

        active = self._repo.find_active(user_id)
        scored: list[tuple[float, Memory]] = []
        for mem in active:
            score = self._keyword_score(mem, query)
            if score > 0:
                scored.append((score, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [m for _, m in scored[: self._top_k]]
        logger.info(
            "Retrieved %d relevant memories (keyword fallback, backend=%s)",
            len(results), self._repo.backend,
        )
        return results

    def _keyword_score(self, memory: Memory, query: str) -> float:
        """关键词/谓词匹配分数(降级通道与向量融合通道共用)。"""
        score = 0.0
        q_low = query.lower()
        pred_low = memory.predicate.lower()
        val_low = str(memory.value).lower()
        if pred_low in q_low or q_low in pred_low:
            score += 0.5
        if val_low and (val_low in q_low or q_low in val_low):
            score += 0.3
        if memory.type == MemoryType.PREFERENCE:
            keywords = PREFERENCE_PREDICATES.get(memory.predicate, ())
            if any(kw in query for kw in keywords):
                score += 0.2
        return score
