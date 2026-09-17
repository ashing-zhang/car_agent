# Memory Embedding Provider - 文本向量化抽象层
# 运行指南:
#   get_embedding_provider() 依据 .env 加载 DashScope/OpenAI 兼容端点 Embedding
#   未配置 API Key 时抛出 EmbeddingConfigurationError,禁止静默降级
#   用于 pgvector 语义检索: memory_text -> vector(1536)

import logging
from typing import Protocol

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIM = 1536
DEFAULT_EMBEDDING_MODEL = "text-embedding-v2"

_MISSING_KEY_HINT = (
    "未检测到 Embedding API Key。请在项目根目录的 .env 文件中配置以下任一环境变量:\n"
    "  - DASHSCOPE_API_KEY=your_aliyun_dashscope_api_key (阿里云百炼, 自动使用 dashscope_base_url)\n"
    "  - LLM_API_KEY=your_custom_api_key (配合 LLM_BASE_URL 使用)"
)


class EmbeddingConfigurationError(RuntimeError):
    """Embedding 配置错误(如缺少 API Key),用于调用方捕获并展示用户提示。"""


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
        """初始化 Embedding 客户端,空 API Key 立即报错。"""
        if not api_key:
            raise EmbeddingConfigurationError(_MISSING_KEY_HINT)
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        logger.info(
            "DashScopeEmbeddingProvider initialized: model=%s, dim=%d",
            model,
            self.dimension,
        )

    def embed(self, text: str) -> list[float]:
        """对单条文本执行 Embedding。"""
        resp = self._client.embeddings.create(model=self._model, input=[text])
        return resp.data[0].embedding

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """对批量文本执行 Embedding,空列表快速返回。"""
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]


_default_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """获取默认 Embedding Provider 单例;缺少 API Key 时抛出 EmbeddingConfigurationError。"""
    global _default_provider
    if _default_provider is not None:
        return _default_provider
    settings = get_settings()
    api_key = settings.resolved_llm_api_key()
    if not api_key:
        raise EmbeddingConfigurationError(_MISSING_KEY_HINT)
    _default_provider = DashScopeEmbeddingProvider(
        base_url=settings.resolved_llm_base_url(),
        api_key=api_key,
    )
    return _default_provider


def reset_embedding_provider() -> None:
    """重置单例(测试用)。"""
    global _default_provider
    _default_provider = None
