# Memory Retriever - 记忆检索(规格第14节 Context Aggregation)
# 运行指南:
#   Phase 3 用谓词/关键词检索;Phase 6+ 可扩展 pgvector 语义检索
#   Retrieval 时 latest valid preference 优先于 historical(规格第12节)

import logging

from app.memory.models import Memory, MemoryType
from app.memory.repository import MemoryRepository
from app.memory.temporal import get_latest_active

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """记忆检索器,封装 Repository 查询与时序规则。"""

    def __init__(self, repository: MemoryRepository, top_k: int = 5) -> None:
        """注入仓储与 top_k 参数。"""
        self._repo = repository
        self._top_k = top_k

    def retrieve_preference(
        self, user_id: str, predicate: str
    ) -> Memory | None:
        """返回指定谓词的最新有效偏好。"""
        active = self._repo.find_active(user_id, predicate)
        latest = get_latest_active(active, predicate)
        if latest is not None:
            logger.info("Retrieved preference %s=%s", predicate, latest.value)
        return latest

    def retrieve_relevant(
        self, user_id: str, query: str
    ) -> list[Memory]:
        """根据查询文本检索相关记忆(关键词匹配,后续可换向量检索)。"""
        active = self._repo.find_active(user_id)
        scored: list[tuple[float, Memory]] = []
        for mem in active:
            score = self._relevance_score(mem, query)
            if score > 0:
                scored.append((score, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [m for _, m in scored[: self._top_k]]
        logger.info("Retrieved %d relevant memories for query", len(results))
        return results

    def _relevance_score(self, memory: Memory, query: str) -> float:
        """计算记忆与查询的简单相关性分数。"""
        score = 0.0
        if memory.predicate in query:
            score += 0.5
        if str(memory.value) in query:
            score += 0.3
        if memory.type == MemoryType.PREFERENCE:
            for kw in ("温度", "冷", "热", "暖", "空调"):
                if kw in query:
                    score += 0.2
                    break
        return score
