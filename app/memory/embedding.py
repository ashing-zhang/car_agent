# Memory Embedding Provider - 文本向量化抽象层
# 运行指南:
#   默认: get_embedding_provider() 依据 .env 选择 DashScope兼容 或 Mock
#   无 API key 时自动降级为 MockEmbeddingProvider,使用伪随机向量
#   用于 pgvector 语义检索: memory_text -> vector(1536)

import hashlib
import logging
from typing import Protocol

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIM = 1536
DEFAULT_EMBEDDING_MODEL = "text-embedding-v2"


class EmbeddingProvider(Protocol):
    """文本嵌入提供者接口(委托模式,支持多后端替换)。"""

    dimension: int

    def embed(self, text: str) -> list[float]:
        """将单条文本编码为向量。"""
        ...

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """批量编码文本为向量列表。"""
        ...


class DashScopeEmbeddingProvider:
    """基于 DashScope(阿里云百炼) OpenAI 兼容端点的 Embedding 实现。

    使用 text-embedding-v2 模型,输出 1536 维向量。
    """

    dimension: int = DEFAULT_EMBEDDING_DIM

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        logger.info(
            "DashScopeEmbeddingProvider initialized: model=%s, dim=%d",
            model,
            self.dimension,
        )

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._model, input=[text])
        return resp.data[0].embedding

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]


class MockEmbeddingProvider:
    """基于文本哈希的伪随机向量实现,用于测试与无 API key 场景。

    对相同文本返回确定性向量,保证测试可重复;向量维度与生产一致(1536)。
    """

    dimension: int = DEFAULT_EMBEDDING_DIM

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        logger.info("MockEmbeddingProvider initialized: dim=%d", self.dimension)

    def embed(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def _hash_to_vector(self, text: str) -> list[float]:
        vec: list[float] = []
        h = hashlib.sha256((f"{self._seed}:" + text).encode("utf-8")).digest()
        for i in range(self.dimension):
            byte_idx = i % len(h)
            shift = (i // len(h)) % 8
            val = ((h[byte_idx] >> shift) & 0xFF) / 255.0
            vec.append((val - 0.5) * 2.0)
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


_default_provider: EmbeddingProvider | None = None


def get_embedding_provider(force_mock: bool = False) -> EmbeddingProvider:
    """获取默认 Embedding Provider 单例,无 API key 时降级为 Mock。"""
    global _default_provider
    if _default_provider is not None:
        return _default_provider
    settings = get_settings()
    api_key = settings.resolved_llm_api_key()
    if force_mock or not api_key:
        logger.info("Using MockEmbeddingProvider (no API key or forced mock)")
        _default_provider = MockEmbeddingProvider()
        return _default_provider
    _default_provider = DashScopeEmbeddingProvider(
        base_url=settings.resolved_llm_base_url(),
        api_key=api_key,
    )
    return _default_provider


def reset_embedding_provider() -> None:
    """重置单例(测试用)。"""
    global _default_provider
    _default_provider = None
