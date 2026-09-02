# LLM Provider - 可替换的 LLM 抽象层(规格 Rule 6)
# 运行指南:
#   默认: get_llm_provider() 依据 .env 自动选择 OpenAI 兼容(Qwen) 或 Mock
#   无 API key 时自动降级为 MockLLMProvider,不依赖外部服务
#   LLM 调用必须可替换: OpenAICompatibleProvider / MockLLMProvider

import logging
import re
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from app.config import LLMConfig, get_app_config, get_settings

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """LLM 提供者接口(委托模式,支持多后端替换)。"""

    def bind_tools(self, tools: list[Any]) -> "LLMProvider":
        """绑定可用工具列表,返回支持 tool calling 的实例。"""
        ...

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """根据消息列表生成回复(可能含 tool_calls)。"""
        ...


class OpenAICompatibleProvider:
    """基于 OpenAI 兼容端点(阿里云百炼 Qwen)的 LLM 实现。"""

    def __init__(self, base_url: str, api_key: str, model: str, config: LLMConfig) -> None:
        """初始化 ChatOpenAI 客户端。"""
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


class MockLLMProvider:
    """基于规则的 Mock LLM,无 API key 时使用,模拟 tool calling。"""

    def __init__(self) -> None:
        """初始化工具缓存与调用计数。"""
        self._tools: list[Any] = []
        self._call_count: int = 0

    def bind_tools(self, tools: list[Any]) -> "MockLLMProvider":
        """缓存可用工具列表。"""
        self._tools = tools
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """根据消息内容模拟工具调用或文本回复;收到工具结果后生成总结。"""
        self._call_count += 1
        last = messages[-1] if messages else None
        last_type = getattr(last, "type", "") if last else ""
        if last_type == "tool":
            return AIMessage(content=self._summarize_tool_result(last))
        user_text = ""
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "human":
                user_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
        full_text = " ".join(
            m.content if isinstance(m.content, str) else str(m.content) for m in messages
        )
        tool_call = self._match_intent(user_text, full_text)
        if tool_call is not None:
            logger.info("MockLLM triggered tool call: %s", tool_call["name"])
            return AIMessage(content="", tool_calls=[tool_call])
        response = self._text_response(user_text)
        return AIMessage(content=response)

    def _summarize_tool_result(self, tool_msg: BaseMessage) -> str:
        """根据工具执行结果生成面向用户的中文总结。"""
        content = tool_msg.content if isinstance(tool_msg.content, str) else str(tool_msg.content)
        if "超出安全范围" in content or "policy" in content:
            return "抱歉,该温度超出安全范围,我无法设置。"
        if "车速" in content:
            return content
        if "车内温度" in content and "已将" in content:
            return "已为您调整车内温度。"
        return content

    def _match_intent(self, text: str, full_text: str = "") -> dict | None:
        """根据关键词匹配决定是否触发工具调用;冷热场景优先使用偏好温度。"""
        if any(k in text for k in ("天气", "下雨", "下雪", "外面下")) and "路况" not in text and "交通" not in text:
            return {"name": "get_weather", "args": {}, "id": f"mock_{self._call_count}", "type": "tool_call"}
        if any(k in text for k in ("路况", "交通状况", "前方交通", "交通密度", "交通情况")):
            return {"name": "get_traffic", "args": {}, "id": f"mock_{self._call_count}", "type": "tool_call"}
        if any(k in text for k in ("周围", "场景", "环境", "还适合", "路线", "前方")):
            return {"name": "get_camera_scene", "args": {}, "id": f"mock_{self._call_count}", "type": "tool_call"}
        if any(k in text for k in ("暂停音乐", "停止音乐", "暂停播放", "停止播放", "暂停")):
            return {"name": "pause_media", "args": {}, "id": f"mock_{self._call_count}", "type": "tool_call"}
        if any(k in text for k in ("播放", "放点", "放首", "音乐", "歌单", "听歌")):
            return {"name": "play_media", "args": {"playlist": "默认歌单"}, "id": f"mock_{self._call_count}", "type": "tool_call"}
        if "音量" in text:
            vm = re.search(r"(\d+)", text)
            level = int(vm.group(1)) if vm else 15
            return {"name": "set_volume", "args": {"level": level}, "id": f"mock_{self._call_count}", "type": "tool_call"}
        if any(k in text for k in ("车速", "速度", "电量", "续航", "车辆状态", "speed", "battery")):
            return {"name": "get_vehicle_status", "args": {}, "id": f"mock_{self._call_count}", "type": "tool_call"}
        if any(k in text for k in ("车内温度", "多少度", "cabin")) and "调" not in text and "设" not in text:
            return {"name": "get_cabin_temperature", "args": {}, "id": f"mock_{self._call_count}", "type": "tool_call"}
        m = re.search(r"(\d+(?:\.\d+)?)\s*度?", text)
        if any(k in text for k in ("冷", "热", "调", "温度")) and m:
            return {
                "name": "set_temperature",
                "args": {"temperature_c": float(m.group(1))},
                "id": f"mock_{self._call_count}",
                "type": "tool_call",
            }
        if any(k in text for k in ("冷", "热")) and not m:
            pref_temp = self._extract_preference_temp(full_text)
            temp = pref_temp if pref_temp is not None else 24.0
            return {
                "name": "set_temperature",
                "args": {"temperature_c": temp},
                "id": f"mock_{self._call_count}",
                "type": "tool_call",
            }
        if any(k in text for k in ("开空调", "开ac", "启动空调")):
            return {"name": "set_ac", "args": {"enabled": True}, "id": f"mock_{self._call_count}", "type": "tool_call"}
        if any(k in text for k in ("关空调", "关ac", "关闭空调")):
            return {"name": "set_ac", "args": {"enabled": False}, "id": f"mock_{self._call_count}", "type": "tool_call"}
        return None

    def _extract_preference_temp(self, full_text: str) -> float | None:
        """从上下文文本中提取用户偏好温度。"""
        match = re.search(r"偏好[^0-9]*(\d+(?:\.\d+)?)", full_text)
        if match:
            return float(match.group(1))
        return None

    def _text_response(self, text: str) -> str:
        """生成默认文本回复。"""
        return "我已了解您的需求,正在为您处理。"


_default_provider: LLMProvider | None = None


def get_llm_provider(force_mock: bool = False) -> LLMProvider:
    """获取默认 LLM Provider 单例,无 API key 时降级为 Mock。"""
    global _default_provider
    if _default_provider is not None:
        return _default_provider
    settings = get_settings()
    config = get_app_config().llm
    api_key = settings.resolved_llm_api_key()
    if force_mock or not api_key:
        logger.info("Using MockLLMProvider (no API key or forced mock)")
        _default_provider = MockLLMProvider()
        return _default_provider
    _default_provider = OpenAICompatibleProvider(
        base_url=settings.resolved_llm_base_url(),
        api_key=api_key,
        model=settings.resolved_llm_model(),
        config=config,
    )
    return _default_provider
