# LLM Provider - 可替换的 LLM 抽象层(规格 Rule 6)
# 运行指南:
#   get_llm_provider() 依据 .env 自动加载 OpenAI 兼容(如阿里云百炼 Qwen)端点
#   未配置 LLM API Key 时抛出明确错误,禁止静默降级
#   LLM 调用必须可替换: 通过 OpenAICompatibleProvider 统一封装兼容端点

import logging
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from app.config import LLMConfig, get_app_config, get_settings

logger = logging.getLogger(__name__)

_MISSING_KEY_HINT = (
    "未检测到 LLM API Key。请在项目根目录的 .env 文件中配置以下任一环境变量:\n"
    "  - DASHSCOPE_API_KEY=your_aliyun_dashscope_api_key (阿里云百炼, 自动使用 dashscope_base_url)\n"
    "  - LLM_API_KEY=your_custom_api_key (配合 LLM_BASE_URL / LLM_MODEL 使用)"
)


class LLMConfigurationError(RuntimeError):
    """LLM 配置错误(如缺少 API Key),用于调用方捕获并展示用户提示。"""


class LLMProvider(Protocol):
    """LLM 提供者接口(委托模式,支持多后端替换)。"""

    def bind_tools(self, tools: list[Any]) -> "LLMProvider":
        """绑定可用工具列表,返回支持 tool calling 的实例。"""
        ...

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """根据消息列表生成回复(可能含 tool_calls)。"""
        ...


class OpenAICompatibleProvider:
    """基于 OpenAI 兼容端点(阿里云百炼 Qwen 等)的 LLM 实现。"""

    def __init__(self, base_url: str, api_key: str, model: str, config: LLMConfig) -> None:
        """初始化 ChatOpenAI 客户端,并立即对空 key 报错。"""
        if not api_key:
            raise LLMConfigurationError(_MISSING_KEY_HINT)
        self._llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.request_timeout,
        )
        self._bound: Any = self._llm
        logger.info("OpenAICompatibleProvider initialized: model=%s, base_url=%s", model, base_url)

    def bind_tools(self, tools: list[Any]) -> "OpenAICompatibleProvider":
        """绑定工具并返回自身(链式调用)。"""
        self._bound = self._llm.bind_tools(tools)
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """调用绑定了工具的 LLM 生成回复。"""
        return self._bound.invoke(messages)


_default_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    """获取默认 LLM Provider 单例;未配置 API Key 时抛出 LLMConfigurationError。"""
    global _default_provider
    if _default_provider is not None:
        return _default_provider
    settings = get_settings()
    config = get_app_config().llm
    api_key = settings.resolved_llm_api_key()
    if not api_key:
        raise LLMConfigurationError(_MISSING_KEY_HINT)
    _default_provider = OpenAICompatibleProvider(
        base_url=settings.resolved_llm_base_url(),
        api_key=api_key,
        model=settings.resolved_llm_model(),
        config=config,
    )
    return _default_provider


def reset_llm_provider_singleton() -> None:
    """重置单例缓存,仅用于测试/配置变更场景。"""
    global _default_provider
    _default_provider = None
